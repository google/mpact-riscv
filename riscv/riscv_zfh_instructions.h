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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_ZFH_INSTRUCTIONS_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_ZFH_INSTRUCTIONS_H_

#include <cstdint>

#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_fp_info.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::Instruction;
using HalfFP = ::mpact::sim::generic::HalfFP;

namespace RV32 {
// Source Operands:
//   frs1: Float Register
// Destination Operands:
//   rd: Integer Register
void RiscVZfhFMvxh(const Instruction *instruction);
}  // namespace RV32

namespace RV64 {
// Source Operands:
//   frs1: Float Register
// Destination Operands:
//   rd: Integer Register
void RiscVZfhFMvxh(const Instruction *instruction);
}  // namespace RV64

// Source Operands: *none*
// Destination Operands:
//   frd: Float Register
void RiscVZfhFlhChild(const Instruction *instruction);

// Source Operands:
//   rs1: Integer Register
// Destination Operands:
//   frd: Float Register
void RiscVZfhFMvhx(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   rm: Literal Operand (rounding mode)
// Destination Operands:
//   frd: Float Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhCvtSh(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   rm: Literal Operand (rounding mode)
// Destination Operands:
//   frd: Float Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhCvtHs(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   rm: Literal Operand (rounding mode)
// Destination Operands:
//   frd: Float Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhCvtDh(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   rm: Literal Operand (rounding mode)
// Destination Operands:
//   frd: Float Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhCvtHd(const Instruction *instruction);

// TODO(b/409778536): Factor out generic unimplemented instruction semantic
//                    function.
void RV32VUnimplementedInstruction(const Instruction *instruction);
HalfFP ConvertSingleToHalfFP(float, FPRoundingMode, uint32_t &);
HalfFP ConvertDoubleToHalfFP(double, FPRoundingMode, uint32_t &);

namespace zfh_internal {
bool UseHostFlagsForConversion();
}  // namespace zfh_internal

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_ZFH_INSTRUCTIONS_H_
