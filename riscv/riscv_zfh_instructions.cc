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

#include <cstdint>
#include <functional>
#include <limits>

#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/register.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_fp_host.h"
#include "riscv/riscv_fp_info.h"
#include "riscv/riscv_instruction_helpers.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

using HalfFP = ::mpact::sim::generic::HalfFP;

namespace {

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
  uint32_t fflags = 0;
  RiscVFPState *fp_state =
      static_cast<RiscVState *>(instruction->state())->rv_fp();
  int rm_value = generic::GetInstructionSource<int>(instruction, 1);

  // If the rounding mode is dynamic, read it from the current state.
  if (rm_value == *FPRoundingMode::kDynamic) {
    if (!fp_state->rounding_mode_valid()) {
      LOG(ERROR) << "Invalid rounding mode";
      return;
    }
    rm_value = *(fp_state->GetRoundingMode());
  }
  FPRoundingMode rm = static_cast<FPRoundingMode>(rm_value);
  RiscVCsrDestinationOperand *fflags_dest =
      static_cast<RiscVCsrDestinationOperand *>(instruction->Destination(1));
  RiscVUnaryFloatNaNBoxOp<RVFpRegister::ValueType, RVFpRegister::ValueType,
                          Result, Argument>(
      instruction, [fp_state, rm, &fflags, &operation](Argument a) -> Result {
        Result result;
        if (zfh_internal::UseHostFlagsForConversion()) {
          result = operation(a, rm, fflags);
        } else {
          ScopedFPStatus set_fpstatus(fp_state->host_fp_interface(), rm);
          result = operation(a, rm, fflags);
        }
        return result;
      });
  if (!zfh_internal::UseHostFlagsForConversion()) {
    fflags_dest->GetRiscVCsr()->SetBits(fflags);
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

}  // namespace RV32

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
  uint32_t fflags = 0;
  RiscVCsrDestinationOperand *fflags_dest =
      static_cast<RiscVCsrDestinationOperand *>(instruction->Destination(1));
  RiscVUnaryFloatNaNBoxOp<RVFpRegister::ValueType, RVFpRegister::ValueType,
                          float, HalfFP>(
      instruction, [&fflags](HalfFP a) -> float {
        return ConvertFromHalfFP<float>(a, fflags);
      });
  fflags_dest->GetRiscVCsr()->SetBits(fflags);
}

// Convert from single precision to half precision.
void RiscVZfhCvtHs(const Instruction *instruction) {
  RiscVZfhCvtHelper<HalfFP, float>(
      instruction, [](float a, FPRoundingMode rm, uint32_t &fflags) -> HalfFP {
        return ConvertSingleToHalfFP(a, rm, fflags);
      });
}

// Convert from half precision to double precision.
void RiscVZfhCvtDh(const Instruction *instruction) {
  uint32_t fflags = 0;
  RiscVCsrDestinationOperand *fflags_dest =
      static_cast<RiscVCsrDestinationOperand *>(instruction->Destination(1));
  RiscVUnaryFloatNaNBoxOp<RVFpRegister::ValueType, RVFpRegister::ValueType,
                          double, HalfFP>(
      instruction, [&fflags](HalfFP a) -> double {
        return ConvertFromHalfFP<double>(a, fflags);
      });
  fflags_dest->GetRiscVCsr()->SetBits(fflags);
}

// Convert from double precision to half precision.
void RiscVZfhCvtHd(const Instruction *instruction) {
  RiscVZfhCvtHelper<HalfFP, double>(
      instruction, [](double a, FPRoundingMode rm, uint32_t &fflags) -> HalfFP {
        return ConvertDoubleToHalfFP(a, rm, fflags);
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
