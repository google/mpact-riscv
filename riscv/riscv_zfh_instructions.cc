// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "riscv/riscv_zfh_instructions.h"

#include <sys/types.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>

#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/register.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_fp_host.h"
#include "riscv/riscv_fp_info.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_instruction_helpers.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

using HalfFP = ::mpact::sim::generic::HalfFP;
using ::mpact::sim::generic::IsMpactFp;
using ::mpact::sim::generic::operator*;  // NOLINT: is used below (clang error).
using ::mpact::sim::generic::IsMpactFp;

namespace {

template <typename T>
struct DataTypeRegValue {};

template <>
struct DataTypeRegValue<float> {
  using type = RVFpRegister::ValueType;
};

template <>
struct DataTypeRegValue<double> {
  using type = RVFpRegister::ValueType;
};

template <>
struct DataTypeRegValue<HalfFP> {
  using type = RVFpRegister::ValueType;
};

template <>
struct DataTypeRegValue<int32_t> {
  using type = RV32Register::ValueType;
};

template <>
struct DataTypeRegValue<uint32_t> {
  using type = RV32Register::ValueType;
};

template <>
struct DataTypeRegValue<int64_t> {
  using type = RV64Register::ValueType;
};

template <>
struct DataTypeRegValue<uint64_t> {
  using type = RV64Register::ValueType;
};

// This is a soft conversion from a float or double to a half precision value.
// It is not a direct conversion from the floating point format to the half
// format. Instead, it uses the floating point hardware to do the conversion.
// This is done to get the correct rounding behavior for free from the FPU.
template <typename T>
HalfFP ConvertToHalfFP(T input_value, FPRoundingMode rm, uint32_t &fflags) {
  using UIntType = typename FPTypeInfo<T>::UIntType;
  using IntType = typename FPTypeInfo<T>::IntType;
  UIntType in_int = absl::bit_cast<UIntType>(input_value);
  HalfFP half_fp = {.value = 0x0000};

  // Extract the mantissa, exponent and sign.
  UIntType mantissa = in_int & FPTypeInfo<T>::kSigMask;
  UIntType exponent =
      (in_int & FPTypeInfo<T>::kExpMask) >> FPTypeInfo<T>::kSigSize;
  UIntType sign = in_int >> (FPTypeInfo<T>::kBitSize - 1);

  if (std::isnan(input_value)) {
    half_fp.value = FPTypeInfo<HalfFP>::kCanonicalNaN;
    if (FPTypeInfo<T>::IsSNaN(input_value)) {
      fflags |= static_cast<UIntType>(FPExceptions::kInvalidOp);
    }
    return half_fp;
  }

  if (std::isinf(input_value)) {
    half_fp.value = FPTypeInfo<HalfFP>::kPosInf;
    half_fp.value |= (sign & 1) << (FPTypeInfo<HalfFP>::kBitSize - 1);
    return half_fp;
  }

  if (in_int == 0 || in_int == 1ULL << (FPTypeInfo<T>::kBitSize - 1)) {
    half_fp.value =
        in_int >> (FPTypeInfo<T>::kBitSize - FPTypeInfo<HalfFP>::kBitSize);
    return half_fp;
  }

  IntType bias_diff = FPTypeInfo<T>::kExpBias - FPTypeInfo<HalfFP>::kExpBias;
  IntType unbounded_half_exponent = static_cast<IntType>(exponent) - bias_diff;
  IntType sig_size_diff =
      FPTypeInfo<T>::kSigSize - FPTypeInfo<HalfFP>::kSigSize;
  UIntType half_inf_exponent = ((1 << FPTypeInfo<HalfFP>::kExpSize) - 1);
  UIntType source_type_inf_exponent = ((1 << FPTypeInfo<T>::kExpSize) - 1);

  // Create a temp float with the smallest normal exponent and input mantissa.
  T ftmp = absl::bit_cast<T>(
      (sign << (FPTypeInfo<T>::kBitSize - 1)) |
      (static_cast<UIntType>(1ULL) << FPTypeInfo<T>::kSigSize) | mantissa);

  // Create a divisor float that will be used for shifting the mantissa in a
  // rounding aware way. The amount of shifting depends on if the result is
  // subnormal or normal.
  T fdiv = 0;
  UIntType default_fdiv_exp = FPTypeInfo<T>::kExpBias + sig_size_diff;
  UIntType fdiv_exp = default_fdiv_exp;
  if (unbounded_half_exponent > 0) {
    fdiv_exp = default_fdiv_exp;
  } else if (unbounded_half_exponent < 0) {
    // shift_count: emin - unbiased exponent
    IntType shift_count = 1 - static_cast<int>(exponent) + bias_diff;
    fdiv_exp = default_fdiv_exp + shift_count;
    fdiv_exp = std::min(fdiv_exp, source_type_inf_exponent - 1);
  } else {
    fdiv_exp = default_fdiv_exp + 1;
  }
  fdiv = absl::bit_cast<T>(fdiv_exp << FPTypeInfo<T>::kSigSize);

  // Shift right by doing division.
  T fres = ftmp / fdiv;
  UIntType res = absl::bit_cast<UIntType>(fres);

  // Shift left by doing multiplication.
  T fmultiply = absl::bit_cast<T>(default_fdiv_exp << FPTypeInfo<T>::kSigSize);
  T fres2 = fres * fmultiply;
  UIntType res2 = absl::bit_cast<UIntType>(fres2);

  // Update the exponent if rounding caused an increase.
  IntType exp_diff = static_cast<IntType>((res2 >> FPTypeInfo<T>::kSigSize) &
                                          source_type_inf_exponent) -
                     1;
  UIntType new_exponent = (exponent + exp_diff) & source_type_inf_exponent;

  UIntType half_exponent = 0;
  if (unbounded_half_exponent > 0) {
    half_exponent = new_exponent - bias_diff;
  } else if (unbounded_half_exponent < 0) {
    // Guaranteed subnormal. Nothing to do.
  } else {
    // This case could be normal or subnormal depending on the rounding result.
    half_exponent = (res2 >> FPTypeInfo<T>::kSigSize) & half_inf_exponent;
  }

  UIntType half_mantissa =
      (res2 >> sig_size_diff) & FPTypeInfo<HalfFP>::kSigMask;
  if (unbounded_half_exponent < 0) {  // Guaranteed Subnormal
    half_mantissa = (res & (1 << FPTypeInfo<HalfFP>::kSigSize))
                        ? ((res >> 1) & FPTypeInfo<HalfFP>::kSigMask)
                        : res & FPTypeInfo<HalfFP>::kSigMask;
  }

  // Handle the rules for overflowing to infinity depending on the rounding
  // mode.
  if (half_exponent >= half_inf_exponent) {
    fflags |= static_cast<uint32_t>(FPExceptions::kOverflow);
    fflags |= static_cast<uint32_t>(FPExceptions::kInexact);
    switch (rm) {
      case FPRoundingMode::kRoundToNearest:
        half_exponent = half_inf_exponent;
        half_mantissa = 0;
        break;
      case FPRoundingMode::kRoundTowardsZero:
        half_exponent = half_inf_exponent - 1;
        half_mantissa = FPTypeInfo<HalfFP>::kSigMask;
        break;
      case FPRoundingMode::kRoundDown:
        half_exponent = sign ? half_inf_exponent : half_inf_exponent - 1;
        half_mantissa = sign ? 0 : FPTypeInfo<HalfFP>::kSigMask;
        break;
      case FPRoundingMode::kRoundUp:
        half_exponent = sign ? half_inf_exponent - 1 : half_inf_exponent;
        half_mantissa = sign ? FPTypeInfo<HalfFP>::kSigMask : 0;
        break;
      default:
        half_exponent = half_inf_exponent;
        half_mantissa = 0;
        break;
    }
  }

  // Construct the half float.
  half_fp.value = half_mantissa |
                  (half_exponent << FPTypeInfo<HalfFP>::kSigSize) |
                  (sign << (FPTypeInfo<HalfFP>::kBitSize - 1));

  // Do an arithmetic reconstruction of the float to check for exactness.
  T trailing_significand_float = static_cast<T>(half_mantissa);
  T precision_factor = std::pow(2.0, -1.0 * FPTypeInfo<HalfFP>::kSigSize);
  IntType unbiased_exponent =
      (half_exponent == 0 ? 1 : half_exponent) - FPTypeInfo<HalfFP>::kExpBias;
  T exponent_factor = std::pow(2.0, unbiased_exponent);
  T sign_factor = sign == 1 ? -1.0 : 1.0;
  T implicit_bit_adjustment = half_exponent == 0 ? 0.0 : 1.0;
  T reconstructed_value = ((trailing_significand_float * precision_factor) +
                           implicit_bit_adjustment) *
                          exponent_factor * sign_factor;
  bool exact_conversion = reconstructed_value == input_value;

  // Handle flags for the specific underflow case.
  if (!exact_conversion && (unbounded_half_exponent < 0 ||
                            (unbounded_half_exponent == 0 && fres2 != ftmp))) {
    fflags |= static_cast<uint32_t>(FPExceptions::kUnderflow);
  }

  // Handle flags for the specific inexact case.
  if (!exact_conversion && (fres2 != ftmp)) {
    fflags |= static_cast<uint32_t>(FPExceptions::kInexact);
  }
  return half_fp;
}

// Convert from half precision to single or double precision.
template <typename T>
inline T ConvertFromHalfFP(HalfFP half_fp, uint32_t &fflags) {
  using UIntType = typename FPTypeInfo<T>::UIntType;
  using HalfFPUIntType = typename FPTypeInfo<HalfFP>::UIntType;
  HalfFPUIntType in_int = half_fp.value;

  if (FPTypeInfo<HalfFP>::IsNaN(half_fp)) {
    if (FPTypeInfo<HalfFP>::IsSNaN(half_fp)) {
      fflags |= static_cast<uint32_t>(FPExceptions::kInvalidOp);
    }
    UIntType uint_value = FPTypeInfo<T>::kCanonicalNaN;
    return absl::bit_cast<T>(uint_value);
  }

  if (FPTypeInfo<HalfFP>::IsInf(half_fp)) {
    UIntType uint_value = FPTypeInfo<T>::kPosInf;
    UIntType sign = in_int >> (FPTypeInfo<HalfFP>::kBitSize - 1);
    uint_value |= sign << (FPTypeInfo<T>::kBitSize - 1);
    return absl::bit_cast<T>(uint_value);
  }

  if (in_int == 0 || in_int == 1 << (FPTypeInfo<HalfFP>::kBitSize - 1)) {
    UIntType uint_value =
        static_cast<UIntType>(in_int)
        << (FPTypeInfo<T>::kBitSize - FPTypeInfo<HalfFP>::kBitSize);
    return absl::bit_cast<T>(uint_value);
  }

  UIntType in_sign = FPTypeInfo<HalfFP>::SignBit(half_fp);
  UIntType in_exp =
      (in_int & FPTypeInfo<HalfFP>::kExpMask) >> FPTypeInfo<HalfFP>::kSigSize;
  UIntType in_sig = in_int & FPTypeInfo<HalfFP>::kSigMask;
  UIntType out_int = 0;
  UIntType out_sig = in_sig;
  if (in_exp == 0 && in_sig != 0) {
    // Handle subnormal half precision inputs. They always result in a normal
    // float or double. Calculate how much shifting is needed move the MSB to
    // the location of the implicit bit. Then it can be handled as a normal
    // value from here on.
    int32_t shift_count =
        (1 + FPTypeInfo<HalfFP>::kSigSize) -
        (std::numeric_limits<UIntType>::digits - absl::countl_zero(out_sig));
    out_sig = (out_sig << shift_count) & FPTypeInfo<HalfFP>::kSigMask;
    in_exp = 1 - shift_count;
  }
  out_int |= in_sign << (FPTypeInfo<T>::kBitSize - 1);
  out_int |= (in_exp + FPTypeInfo<T>::kExpBias - FPTypeInfo<HalfFP>::kExpBias)
             << FPTypeInfo<T>::kSigSize;
  out_int |=
      out_sig << (FPTypeInfo<T>::kSigSize - FPTypeInfo<HalfFP>::kSigSize);
  return absl::bit_cast<T>(out_int);
}

template <typename Result, typename Argument>
void RiscVZfhCvtHelper(
    const Instruction *instruction,
    std::function<Result(Argument, FPRoundingMode, uint32_t &)> operation) {
  RiscVCsrDestinationOperand *fflags_dest =
      static_cast<RiscVCsrDestinationOperand *>(instruction->Destination(1));
  using DstRegValue = typename DataTypeRegValue<Result>::type;
  uint32_t fflags = 0;

  Argument lhs;
  if constexpr (IsMpactFp<Argument>::value) {
    lhs = GetNaNBoxedSource<RVFpRegister::ValueType, Argument>(instruction, 0);
    if (FPTypeInfo<Argument>::IsSNaN(lhs)) {
      fflags |= *FPExceptions::kInvalidOp;
    }
  } else {
    lhs = generic::GetInstructionSource<Argument>(instruction, 0);
  }
  // Get the rounding mode.
  int rm_value = generic::GetInstructionSource<int>(instruction, 1);

  auto *rv_fp = static_cast<RiscVState *>(instruction->state())->rv_fp();
  // If the rounding mode is dynamic, read it from the current state.
  if (rm_value == *FPRoundingMode::kDynamic) {
    if (!rv_fp->rounding_mode_valid()) {
      LOG(ERROR) << "Invalid rounding mode";
      return;
    }
    rm_value = *rv_fp->GetRoundingMode();
  }

  Result dest_value;
  {
    ScopedFPRoundingMode scoped_rm(rv_fp->host_fp_interface(), rm_value);
    dest_value = operation(lhs, static_cast<FPRoundingMode>(rm_value), fflags);
  }
  fflags_dest->GetRiscVCsr()->SetBits(fflags);
  auto *reg = static_cast<generic::RegisterDestinationOperand<DstRegValue> *>(
                  instruction->Destination(0))
                  ->GetRegister();

  if (sizeof(DstRegValue) > sizeof(Result) && IsMpactFp<Result>::value) {
    // If the floating point value is narrower than the register, the upper
    // bits have to be set to all ones.
    using UReg = typename std::make_unsigned<DstRegValue>::type;
    using UInt = typename FPTypeInfo<Result>::UIntType;
    auto dest_u_value = *reinterpret_cast<UInt *>(&dest_value);
    UReg reg_value = std::numeric_limits<UReg>::max();
    int shift = 8 * sizeof(Result);
    reg_value = (reg_value << shift) | dest_u_value;
    reg->data_buffer()->template Set<DstRegValue>(0, reg_value);
    return;
  }
  reg->data_buffer()->template Set<Result>(0, dest_value);
}

// Generic helper function enabling HalfFP operations in native datatypes.
template <typename Argument, typename IntermediateType>
void RiscVZfhUnaryHelper(
    const Instruction *instruction,
    std::function<IntermediateType(IntermediateType)> operation) {
  RiscVCsrDestinationOperand *fflags_dest =
      static_cast<RiscVCsrDestinationOperand *>(instruction->Destination(1));
  uint32_t fflags = 0;
  RiscVUnaryFloatNaNBoxOp<RVFpRegister::ValueType, RVFpRegister::ValueType,
                          HalfFP, Argument>(
      instruction, [instruction, &operation, &fflags](Argument a) -> HalfFP {
        RiscVFPState *rv_fp =
            static_cast<RiscVState *>(instruction->state())->rv_fp();
        int rm_value = generic::GetInstructionSource<int>(instruction, 1);

        // If the rounding mode is dynamic, read it from the current state.
        if (rm_value == *FPRoundingMode::kDynamic) {
          if (!rv_fp->rounding_mode_valid()) {
            LOG(ERROR) << "Invalid rounding mode";
          }
          rm_value = *(rv_fp->GetRoundingMode());
        }
        FPRoundingMode rm = static_cast<FPRoundingMode>(rm_value);
        IntermediateType argument1 =
            ConvertFromHalfFP<IntermediateType>(a, fflags);
        IntermediateType result;
        {
          ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface(), rm);
          result = operation(argument1);
        }
        // To get the correct fflags we need a combination of host flags from
        // the operation and the conversion flags. Copy the host flags and merge
        // them with the conversion flags.
        fflags |= rv_fp->fflags()->GetUint32();
        {
          // ConvertToHalfFP pollutes the host flags so we need to create a
          // ScopedFPRoundingMode to restore the host flags.
          ScopedFPRoundingMode scoped_rm(rv_fp->host_fp_interface(), rm_value);
          return ConvertToHalfFP(result, rm, fflags);
        }
      });
  fflags_dest->GetRiscVCsr()->SetBits(fflags);
}

// Generic helper function enabling HalfFP operations in native datatypes.
template <typename Argument, typename IntermediateType>
void RiscVZfhBinaryHelper(
    const Instruction *instruction,
    std::function<IntermediateType(IntermediateType, IntermediateType)>
        operation) {
  RiscVCsrDestinationOperand *fflags_dest =
      static_cast<RiscVCsrDestinationOperand *>(instruction->Destination(1));
  uint32_t fflags = 0;
  RiscVBinaryFloatNaNBoxOp<RVFpRegister::ValueType, HalfFP, Argument>(
      instruction,
      [instruction, &operation, &fflags](Argument a, Argument b) -> HalfFP {
        RiscVFPState *rv_fp =
            static_cast<RiscVState *>(instruction->state())->rv_fp();
        int rm_value = generic::GetInstructionSource<int>(instruction, 2);
        // If the rounding mode is dynamic, read it from the current state.
        if (rm_value == *FPRoundingMode::kDynamic) {
          if (!rv_fp->rounding_mode_valid()) {
            LOG(ERROR) << "Invalid rounding mode";
          }
          rm_value = *(rv_fp->GetRoundingMode());
        }
        FPRoundingMode rm = static_cast<FPRoundingMode>(rm_value);
        IntermediateType argument1 =
            ConvertFromHalfFP<IntermediateType>(a, fflags);
        IntermediateType argument2 =
            ConvertFromHalfFP<IntermediateType>(b, fflags);
        IntermediateType result;
        {
          ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface(), rm);
          result = operation(argument1, argument2);
        }
        // To get the correct fflags we need a combination of host flags from
        // the operation and the conversion flags. Copy the host flags and merge
        // them with the conversion flags.
        fflags |= rv_fp->fflags()->GetUint32();
        {
          // ConvertToHalfFP pollutes the host flags so we need to create a
          // ScopedFPRoundingMode to restore the host flags.
          ScopedFPRoundingMode scoped_rm(rv_fp->host_fp_interface(), rm_value);
          return ConvertToHalfFP(result, rm, fflags);
        }
      });
  fflags_dest->GetRiscVCsr()->SetBits(fflags);
}

}  // namespace

namespace RV32 {

// Move a half precision value from a float register to a 32 bit integer
// register.
void RiscVZfhFMvxh(const Instruction *instruction) {
  RiscVUnaryFloatOp<uint32_t, HalfFP>(instruction, [](HalfFP a) -> uint32_t {
    if (FPTypeInfo<HalfFP>::SignBit(a)) {
      // Repeat the sign bit for negative values.
      return 0xFFFF'0000 | a.value;
    }
    return static_cast<uint32_t>(a.value);
  });
}

// Convert from half precision to integer.
void RiscVZfhCvtWh(const Instruction *instruction) {
  RiscVConvertFloatWithFflagsOp<typename RV32Register::ValueType, HalfFP,
                                int32_t>(instruction);
}

// Convert from integer to half precision.
void RiscVZfhCvtHw(const Instruction *instruction) {
  RiscVZfhCvtHelper<HalfFP, int32_t>(
      instruction,
      [](int32_t a, FPRoundingMode rm, uint32_t &fflags) -> HalfFP {
        float input_float = static_cast<float>(a);
        return ConvertToHalfFP(input_float, rm, fflags);
      });
}

// Convert from unsigned integer to half precision.
void RiscVZfhCvtHwu(const Instruction *instruction) {
  RiscVZfhCvtHelper<HalfFP, uint32_t>(
      instruction,
      [](uint32_t a, FPRoundingMode rm, uint32_t &fflags) -> HalfFP {
        float input_float = static_cast<float>(a);
        return ConvertToHalfFP(input_float, rm, fflags);
      });
}

// Convert from half precision to unsigned integer.
void RiscVZfhCvtWuh(const Instruction *instruction) {
  RiscVConvertFloatWithFflagsOp<typename RV32Register::ValueType, HalfFP,
                                uint32_t>(instruction);
}

// Compare two half precision values for equality.
void RiscVZfhFcmpeq(const Instruction *instruction) {
  RiscVBinaryFloatNaNBoxOp<RVFpRegister::ValueType, uint64_t, HalfFP>(
      instruction, [](HalfFP a, HalfFP b) -> uint64_t {
        float a_f;
        float b_f;
        uint32_t unused_fflags = 0;
        if (FPTypeInfo<HalfFP>::IsSNaN(a)) {
          a_f = absl::bit_cast<float>(FPTypeInfo<float>::kPosInf | 1);
        } else {
          a_f = ConvertFromHalfFP<float>(a, unused_fflags);
        }
        if (FPTypeInfo<HalfFP>::IsSNaN(b)) {
          b_f = absl::bit_cast<float>(FPTypeInfo<float>::kPosInf | 1);
        } else {
          b_f = ConvertFromHalfFP<float>(b, unused_fflags);
        }
        return a_f == b_f ? 1 : 0;
      });
}

// Compare two half precision values for less than.
void RiscVZfhFcmplt(const Instruction *instruction) {
  RiscVBinaryFloatNaNBoxOp<RVFpRegister::ValueType, uint64_t, HalfFP>(
      instruction, [](HalfFP a, HalfFP b) -> uint64_t {
        float a_f;
        float b_f;
        uint32_t unused_fflags = 0;
        if (FPTypeInfo<HalfFP>::IsNaN(a)) {
          a_f = absl::bit_cast<float>(FPTypeInfo<float>::kPosInf | 1);
        } else {
          a_f = ConvertFromHalfFP<float>(a, unused_fflags);
        }
        if (FPTypeInfo<HalfFP>::IsNaN(b)) {
          b_f = absl::bit_cast<float>(FPTypeInfo<float>::kPosInf | 1);
        } else {
          b_f = ConvertFromHalfFP<float>(b, unused_fflags);
        }
        return a_f < b_f ? 1 : 0;
      });
}

// Compare two half precision values for less than or equal to.
void RiscVZfhFcmple(const Instruction *instruction) {
  RiscVBinaryFloatNaNBoxOp<RVFpRegister::ValueType, uint64_t, HalfFP>(
      instruction, [](HalfFP a, HalfFP b) -> uint64_t {
        float a_f;
        float b_f;
        uint32_t unused_fflags = 0;
        if (FPTypeInfo<HalfFP>::IsNaN(a)) {
          a_f = absl::bit_cast<float>(FPTypeInfo<float>::kPosInf | 1);
        } else {
          a_f = ConvertFromHalfFP<float>(a, unused_fflags);
        }
        if (FPTypeInfo<HalfFP>::IsNaN(b)) {
          b_f = absl::bit_cast<float>(FPTypeInfo<float>::kPosInf | 1);
        } else {
          b_f = ConvertFromHalfFP<float>(b, unused_fflags);
        }
        return a_f <= b_f ? 1 : 0;
      });
}

// Classify a half precision value.
void RiscVZfhFclass(const Instruction *instruction) {
  RiscVUnaryOp<RV32Register, uint32_t, HalfFP>(
      instruction, [](HalfFP a) -> uint32_t {
        return static_cast<uint32_t>(ClassifyFP(a));
      });
}

}  // namespace RV32

namespace RV64 {
// Move a half precision value from a float register to a 32 bit integer
// register.
void RiscVZfhFMvxh(const Instruction *instruction) {
  RiscVUnaryFloatOp<uint64_t, HalfFP>(instruction, [](HalfFP a) -> uint64_t {
    if (FPTypeInfo<HalfFP>::SignBit(a)) {
      // Repeat the sign bit for negative values.
      return 0xFFFF'FFFF'FFFF'0000 | a.value;
    }
    return static_cast<uint64_t>(a.value);
  });
}

}  // namespace RV64

void RiscVZfhFlhChild(const Instruction *instruction) {
  using FPUInt = typename FPTypeInfo<HalfFP>::UIntType;
  LoadContext *context = static_cast<LoadContext *>(instruction->context());
  auto value = context->value_db->Get<FPUInt>(0);
  auto *reg =
      static_cast<
          generic::RegisterDestinationOperand<RVFpRegister::ValueType> *>(
          instruction->Destination(0))
          ->GetRegister();
  if (sizeof(RVFpRegister::ValueType) > sizeof(FPUInt)) {
    // NaN box the loaded value.
    auto reg_value = std::numeric_limits<RVFpRegister::ValueType>::max();
    reg_value <<= sizeof(FPUInt) * 8;
    reg_value |= value;
    reg->data_buffer()->Set<RVFpRegister::ValueType>(0, reg_value);
    return;
  }
  reg->data_buffer()->Set<RVFpRegister::ValueType>(0, value);
}

// Move a half precision value from an integer register to a float register.
void RiscVZfhFMvhx(const Instruction *instruction) {
  RiscVUnaryFloatOp<HalfFP, uint64_t>(instruction, [](uint64_t a) -> HalfFP {
    return HalfFP{.value = static_cast<uint16_t>(a)};
  });
}

// Convert from half precision to single precision.
void RiscVZfhCvtSh(const Instruction *instruction) {
  RiscVZfhCvtHelper<float, HalfFP>(
      instruction, [](HalfFP a, FPRoundingMode rm, uint32_t &fflags) -> float {
        return ConvertFromHalfFP<float>(a, fflags);
      });
}

// Convert from single precision to half precision.
void RiscVZfhCvtHs(const Instruction *instruction) {
  RiscVZfhCvtHelper<HalfFP, float>(
      instruction, [](float a, FPRoundingMode rm, uint32_t &fflags) -> HalfFP {
        return ConvertToHalfFP(a, rm, fflags);
      });
}

// Convert from half precision to double precision.
void RiscVZfhCvtDh(const Instruction *instruction) {
  RiscVZfhCvtHelper<double, HalfFP>(
      instruction, [](HalfFP a, FPRoundingMode rm, uint32_t &fflags) -> double {
        return ConvertFromHalfFP<double>(a, fflags);
      });
}

// Convert from double precision to half precision.
void RiscVZfhCvtHd(const Instruction *instruction) {
  RiscVZfhCvtHelper<HalfFP, double>(
      instruction, [](double a, FPRoundingMode rm, uint32_t &fflags) -> HalfFP {
        return ConvertToHalfFP(a, rm, fflags);
      });
}

// Add two half precision values. Do the calculation in single precision.
void RiscVZfhFadd(const Instruction *instruction) {
  RiscVZfhBinaryHelper<HalfFP, float>(
      instruction, [](float a, float b) -> float { return a + b; });
}

// Subtract two half precision values. Do the calculation in single precision.
void RiscVZfhFsub(const Instruction *instruction) {
  RiscVZfhBinaryHelper<HalfFP, float>(
      instruction, [](float a, float b) -> float { return a - b; });
}

// Multiply two half precision values. Do the calculation in single precision.
void RiscVZfhFmul(const Instruction *instruction) {
  RiscVZfhBinaryHelper<HalfFP, float>(
      instruction, [](float a, float b) -> float { return a * b; });
}

// Divide two half precision values. Do the calculation in single precision.
void RiscVZfhFdiv(const Instruction *instruction) {
  RiscVZfhBinaryHelper<HalfFP, float>(
      instruction, [](float a, float b) -> float { return a / b; });
}

// Take the minimum of two half precision values. Do the operation in single
// precision.
void RiscVZfhFmin(const Instruction *instruction) {
  RiscVZfhBinaryHelper<HalfFP, float>(instruction,
                                      [](float a, float b) -> float {
                                        // On ARM std::fminf returns NaN if
                                        // either input is NaN. Add extra checks
                                        // to make X86 and ARM behavior the
                                        // same.
                                        if (std::isnan(a)) {
                                          return b;
                                        } else if (std::isnan(b)) {
                                          return a;
                                        }
                                        return std::fminf(a, b);
                                      });
}

// Take the maximum of two half precision values. Do the operation in single
// precision.
void RiscVZfhFmax(const Instruction *instruction) {
  RiscVZfhBinaryHelper<HalfFP, float>(instruction,
                                      [](float a, float b) -> float {
                                        // On ARM std::fmaxf returns NaN if
                                        // either input is NaN. Add extra checks
                                        // to make X86 and ARM behavior the
                                        // same.
                                        if (std::isnan(a)) {
                                          return b;
                                        } else if (std::isnan(b)) {
                                          return a;
                                        }
                                        return std::fmaxf(a, b);
                                      });
}

// Calculate the square root of a half precision value. Do the operation in
// single precision and then convert back to half precision.
void RiscVZfhFsqrt(const Instruction *instruction) {
  RiscVZfhUnaryHelper<HalfFP, float>(
      instruction, [](float a) -> float { return std::sqrt(a); });
}

// The result is the exponent and significand of the first source with the
// sign bit of the second source.
void RiscVZfhFsgnj(const Instruction *instruction) {
  RiscVBinaryFloatNaNBoxOp<RVFpRegister::ValueType, HalfFP, HalfFP>(
      instruction, [](HalfFP a, HalfFP b) -> HalfFP {
        uint16_t mask =
            FPTypeInfo<HalfFP>::kExpMask | FPTypeInfo<HalfFP>::kSigMask;
        return HalfFP{.value = static_cast<uint16_t>((a.value & mask) |
                                                     (b.value & ~mask))};
      });
}

// The result is the exponent and significand of the first source with the
// opposite sign bit of the second source.
void RiscVZfhFsgnjn(const Instruction *instruction) {
  RiscVBinaryFloatNaNBoxOp<RVFpRegister::ValueType, HalfFP, HalfFP>(
      instruction, [](HalfFP a, HalfFP b) -> HalfFP {
        uint16_t mask =
            FPTypeInfo<HalfFP>::kExpMask | FPTypeInfo<HalfFP>::kSigMask;
        return HalfFP{.value = static_cast<uint16_t>((a.value & mask) |
                                                     (~b.value & ~mask))};
      });
}

// The result is the exponent and significand of the first source with the
// sign bit that is the exclusive or of the two source sign bits.
void RiscVZfhFsgnjx(const Instruction *instruction) {
  RiscVBinaryFloatNaNBoxOp<RVFpRegister::ValueType, HalfFP, HalfFP>(
      instruction, [](HalfFP a, HalfFP b) -> HalfFP {
        uint16_t mask =
            FPTypeInfo<HalfFP>::kExpMask | FPTypeInfo<HalfFP>::kSigMask;
        return HalfFP{.value = static_cast<uint16_t>(
                          (a.value & mask) | ((a.value ^ b.value) & ~mask))};
      });
}

// TODO(b/409778536): Factor out generic unimplemented instruction semantic
//                    function.
void RV32VUnimplementedInstruction(const Instruction *instruction) {
  auto *state = static_cast<RiscVState *>(instruction->state());
  state->Trap(/*is_interrupt*/ false, /*trap_value*/ 0,
              *ExceptionCode::kIllegalInstruction,
              /*epc*/ instruction->address(), instruction);
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
