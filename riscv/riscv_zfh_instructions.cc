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
      fflags_dest->GetRiscVCsr()->SetBits(*FPExceptions::kInvalidOp);
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
  if (zfh_internal::UseHostFlagsForConversion()) {
    ScopedFPStatus set_fp_status(rv_fp->host_fp_interface(), rm_value);
    dest_value = operation(lhs, static_cast<FPRoundingMode>(rm_value), fflags);
  } else {
    ScopedFPRoundingMode scoped_rm(rv_fp->host_fp_interface(), rm_value);
    dest_value = operation(lhs, static_cast<FPRoundingMode>(rm_value), fflags);
  }
  if (!zfh_internal::UseHostFlagsForConversion()) {
    fflags_dest->GetRiscVCsr()->SetBits(fflags);
  }
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

template <typename T>
inline HalfFP ConvertToHalfFP(T input_value, FPRoundingMode rm,
                              uint32_t &fflags);

template <>
inline HalfFP ConvertToHalfFP(float input_value, FPRoundingMode rm,
                              uint32_t &fflags) {
  return ConvertSingleToHalfFP(input_value, rm, fflags);
}

template <>
inline HalfFP ConvertToHalfFP(double input_value, FPRoundingMode rm,
                              uint32_t &fflags) {
  return ConvertDoubleToHalfFP(input_value, rm, fflags);
}

// Generic helper function enabling HalfFP operations in native datatypes.
template <typename Result, typename Argument>
void RiscVZfhUnaryHelper(
    const Instruction *instruction,
    std::function<Result(Argument, FPRoundingMode, uint32_t &)> operation) {
  uint32_t fflags = 0;
  RiscVFPState *rv_fp =
      static_cast<RiscVState *>(instruction->state())->rv_fp();
  int rm_value = generic::GetInstructionSource<int>(instruction, 1);

  // If the rounding mode is dynamic, read it from the current state.
  if (rm_value == *FPRoundingMode::kDynamic) {
    if (!rv_fp->rounding_mode_valid()) {
      LOG(ERROR) << "Invalid rounding mode";
      return;
    }
    rm_value = *(rv_fp->GetRoundingMode());
  }
  FPRoundingMode rm = static_cast<FPRoundingMode>(rm_value);
  RiscVCsrDestinationOperand *fflags_dest =
      static_cast<RiscVCsrDestinationOperand *>(instruction->Destination(1));
  bool arguments_contain_snan = false;
  RiscVUnaryFloatNaNBoxOp<RVFpRegister::ValueType, RVFpRegister::ValueType,
                          Result, Argument>(
      instruction,
      [rv_fp, rm, &fflags, &operation,
       &arguments_contain_snan](Argument a) -> Result {
        Result result;
        if (FPTypeInfo<Argument>::IsSNaN(a)) {
          arguments_contain_snan = true;
        }
        if (zfh_internal::UseHostFlagsForConversion()) {
          result = operation(a, rm, fflags);
        } else {
          ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface(), rm);
          result = operation(a, rm, fflags);
        }
        return result;
      });
  if (!zfh_internal::UseHostFlagsForConversion()) {
    fflags_dest->GetRiscVCsr()->SetBits(fflags);
  }
  if (arguments_contain_snan) {
    fflags_dest->GetRiscVCsr()->SetBits(*FPExceptions::kInvalidOp);
  }
}

// Generic helper function enabling HalfFP operations in native datatypes.
template <typename Argument, typename IntermediateType>
void RiscVZfhBinaryHelper(
    const Instruction *instruction,
    std::function<IntermediateType(IntermediateType, IntermediateType)>
        operation) {
  RiscVFPState *rv_fp =
      static_cast<RiscVState *>(instruction->state())->rv_fp();
  int rm_value = generic::GetInstructionSource<int>(instruction, 2);

  // If the rounding mode is dynamic, read it from the current state.
  if (rm_value == *FPRoundingMode::kDynamic) {
    if (!rv_fp->rounding_mode_valid()) {
      LOG(ERROR) << "Invalid rounding mode";
      return;
    }
    rm_value = *(rv_fp->GetRoundingMode());
  }
  FPRoundingMode rm = static_cast<FPRoundingMode>(rm_value);
  RiscVCsrDestinationOperand *fflags_dest =
      static_cast<RiscVCsrDestinationOperand *>(instruction->Destination(1));
  uint32_t fflags = fflags_dest->GetRiscVCsr()->GetUint32();
  bool arguments_contain_snan = false;
  IntermediateType b_emin =
      std::pow(2.0, 1 - FPTypeInfo<IntermediateType>::kExpBias);
  IntermediateType result;
  RiscVBinaryFloatNaNBoxOp<RVFpRegister::ValueType, HalfFP, Argument>(
      instruction,
      [&operation, &arguments_contain_snan, &fflags, rv_fp, &rm, &result](
          Argument a, Argument b) -> HalfFP {
        IntermediateType a_f;
        IntermediateType b_f;
        if (FPTypeInfo<Argument>::IsSNaN(a)) {
          a_f = absl::bit_cast<IntermediateType>(
              FPTypeInfo<IntermediateType>::kPosInf | 1);
          arguments_contain_snan = true;
        } else {
          a_f = ConvertFromHalfFP<IntermediateType>(a, fflags);
        }
        if (FPTypeInfo<Argument>::IsSNaN(b)) {
          b_f = absl::bit_cast<IntermediateType>(
              FPTypeInfo<IntermediateType>::kPosInf | 1);
          arguments_contain_snan = true;
        } else {
          b_f = ConvertFromHalfFP<IntermediateType>(b, fflags);
        }
        if (zfh_internal::UseHostFlagsForConversion()) {
          result = operation(a_f, b_f);
        } else {
          ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface(), rm);
          result = operation(a_f, b_f);
        }
        if (!zfh_internal::UseHostFlagsForConversion()) {
          fflags |= rv_fp->fflags()->GetUint32();
        }
        return ConvertToHalfFP(result, rm, fflags);
      });
  if (arguments_contain_snan) {
    fflags_dest->GetRiscVCsr()->SetBits(*FPExceptions::kInvalidOp);
  }
  if (!zfh_internal::UseHostFlagsForConversion()) {
    fflags_dest->GetRiscVCsr()->Write(fflags);
  }
  // When the result is less than b_emin before rounding we need to set the
  // underflow flag.
  if ((fflags_dest->GetRiscVCsr()->GetUint32() & *FPExceptions::kInexact) &&
      result != 0 && std::abs(result) < b_emin) {
    fflags_dest->GetRiscVCsr()->SetBits(*FPExceptions::kUnderflow);
  }
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
  RiscVZfhUnaryHelper<HalfFP, HalfFP>(
      instruction, [](HalfFP a, FPRoundingMode rm, uint32_t &fflags) -> HalfFP {
        float input_f = ConvertFromHalfFP<float>(a, fflags);
        if (!std::isnan(input_f) && input_f < 0) {
          fflags |= static_cast<uint32_t>(FPExceptions::kInvalidOp);
        }
        return ConvertToHalfFP(std::sqrt(input_f), rm, fflags);
      });
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
