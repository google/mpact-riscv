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

#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"

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

// Source Operands:
//   rs1: Integer Register
// Destination Operands:
//   frd: Float Register
void RiscVZfhFMvhx(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   rm: Literal Operand (rounding mode)
// Destination Operands:
//   rd: Integer Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhCvtWh(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   rm: Literal Operand (rounding mode)
// Destination Operands:
//   rd: Integer Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhCvtWuh(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   frs2: Float Register
// Destination Operands:
//   rd: Integer Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhFcmpeq(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   frs2: Float Register
// Destination Operands:
//   rd: Integer Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhFcmplt(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   frs2: Float Register
// Destination Operands:
//   rd: Integer Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhFcmple(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
// Destination Operands:
//   rd: Integer Register
void RiscVZfhFclass(const Instruction *instruction);

}  // namespace RV32

namespace RV64 {
// Source Operands:
//   frs1: Float Register
// Destination Operands:
//   rd: Integer Register
void RiscVZfhFMvxh(const Instruction *instruction);

// Source Operands:
//   rs1: Integer Register
// Destination Operands:
//   frd: Float Register
void RiscVZfhFMvhx(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   rm: Literal Operand (rounding mode)
// Destination Operands:
//   rd: Integer Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhCvtWh(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   rm: Literal Operand (rounding mode)
// Destination Operands:
//   rd: Integer Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhCvtWuh(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   frs2: Float Register
// Destination Operands:
//   rd: Integer Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhFcmpeq(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   frs2: Float Register
// Destination Operands:
//   rd: Integer Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhFcmplt(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   frs2: Float Register
// Destination Operands:
//   rd: Integer Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhFcmple(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
// Destination Operands:
//   rd: Integer Register
void RiscVZfhFclass(const Instruction *instruction);

}  // namespace RV64

// Source Operands: *none*
// Destination Operands:
//   frd: Float Register
void RiscVZfhFlhChild(const Instruction *instruction);

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

// Source Operands:
//   frs1: Float Register
//   frs2: Float Register
//   rm: Literal Operand (rounding mode)
// Destination Operands:
//   frd: Float Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhFadd(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   frs2: Float Register
//   rm: Literal Operand (rounding mode)
// Destination Operands:
//   frd: Float Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhFsub(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   frs2: Float Register
//   rm: Literal Operand (rounding mode)
// Destination Operands:
//   frd: Float Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhFmul(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   frs2: Float Register
//   rm: Literal Operand (rounding mode)
// Destination Operands:
//   frd: Float Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhFdiv(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   frs2: Float Register
//   rm: Literal Operand (rounding mode)
// Destination Operands:
//   frd: Float Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhFmin(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   frs2: Float Register
//   rm: Literal Operand (rounding mode)
// Destination Operands:
//   frd: Float Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhFmax(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   frs2: Float Register
// Destination Operands:
//   frd: Float Register
void RiscVZfhFsgnj(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   frs2: Float Register
// Destination Operands:
//   frd: Float Register
void RiscVZfhFsgnjn(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   frs2: Float Register
// Destination Operands:
//   frd: Float Register
void RiscVZfhFsgnjx(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   rm: Literal Operand (rounding mode)
// Destination Operands:
//   frd: Float Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhFsqrt(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   frs2: Float Register
//   frs3: Float Register
//   rm: Literal Operand (rounding mode)
// Destination Operands:
//   frd: Float Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhFmadd(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   frs2: Float Register
//   frs3: Float Register
//   rm: Literal Operand (rounding mode)
// Destination Operands:
//   frd: Float Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhFmsub(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   frs2: Float Register
//   frs3: Float Register
//   rm: Literal Operand (rounding mode)
// Destination Operands:
//   frd: Float Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhFnmadd(const Instruction *instruction);

// Source Operands:
//   frs1: Float Register
//   frs2: Float Register
//   frs3: Float Register
//   rm: Literal Operand (rounding mode)
// Destination Operands:
//   frd: Float Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhFnmsub(const Instruction *instruction);

// Source Operands:
//   rs1: Integer Register
//   rm: Literal Operand (rounding mode)
// Destination Operands:
//   frd: Float Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhCvtHw(const Instruction *instruction);

// Source Operands:
//   rs1: Integer Register
//   rm: Literal Operand (rounding mode)
// Destination Operands:
//   frd: Float Register
//   fflags: Accrued Exception Flags field in FCSR
void RiscVZfhCvtHwu(const Instruction *instruction);

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_ZFH_INSTRUCTIONS_H_
