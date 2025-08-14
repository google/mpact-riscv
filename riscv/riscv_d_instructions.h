// Copyright 2023 Google LLC
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

#ifndef MPACT_RISCV_RISCV_RISCV_D_INSTRUCTIONS_H_
#define MPACT_RISCV_RISCV_RISCV_D_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

// This file declares the semantic functions that implement the scalar
// double precision floating point instructions in the RiscV architecture.

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::Instruction;

// Double precision arithmetic instructions each take 3 source operands and
// 1 destination register operand. Source 0 and 1 are the data sources to the
// operation, source 2 is the rounding mode specifier in the instruction.
void RiscVDAdd(const Instruction* instruction);
void RiscVDSub(const Instruction* instruction);
void RiscVDMul(const Instruction* instruction);
void RiscVDDiv(const Instruction* instruction);
// Double precision square root takes 2 source operands, the source register
// and the rounding mode, and 1 destination register operand.
void RiscVDSqrt(const Instruction* instruction);
// The double precision Min and Max instructions each take 2 source register
// operands and 1 destination register operand.
void RiscVDMin(const Instruction* instruction);
void RiscVDMax(const Instruction* instruction);
// The following four multiply-accumulate instruction each take 3 source
// register operands (0, 1, 2) and the rounding mode (3), and one destination
// register operand.
void RiscVDMadd(const Instruction* instruction);
void RiscVDMsub(const Instruction* instruction);
void RiscVDNmadd(const Instruction* instruction);
void RiscVDNmsub(const Instruction* instruction);
// The double precision conversion instructions each take 1 source register
// operand, the rounding mode, and 1 destination register operand.
void RiscVDCvtDw(const Instruction* instruction);
void RiscVDCvtDwu(const Instruction* instruction);
void RiscVDCvtSd(const Instruction* instruction);
void RiscVDCvtDs(const Instruction* instruction);
// The double precision sign manipulation instructions take 2 source register
// operands and a single destination register operand.
void RiscVDSgnj(const Instruction* instruction);
void RiscVDSgnjn(const Instruction* instruction);
void RiscVDSgnjx(const Instruction* instruction);

namespace RV32 {

// Store float instruction semantic function, source operand 0 is the base
// register, source operand 1 is the offset, while source operand 2 is the value
// to be stored referred to by rs2.
void RiscVDSd(const Instruction* instruction);
// The double precision conversion instructions each take 1 source register
// operand, the rounding mode, and 1 destination register operand.
void RiscVDCvtWd(const Instruction* instruction);
void RiscVDCvtWud(const Instruction* instruction);
// The double precision compare instructions take 2 source register operands
// and a single destination register operand.
void RiscVDCmpeq(const Instruction* instruction);
void RiscVDCmplt(const Instruction* instruction);
void RiscVDCmple(const Instruction* instruction);
// The double precision class instruction takes a single source register operand
// and a single destination register operand.
void RiscVDClass(const Instruction* instruction);

}  // namespace RV32

// The following instruction semantic functions are only valid in RiscV 64.
namespace RV64 {

// Store float instruction semantic function, source operand 0 is the base
// register, source operand 1 is the offset, while source operand 2 is the value
// to be stored referred to by rs2.
void RiscVDSd(const Instruction* instruction);
// The double precision conversion instructions each take 1 source register
// operand, the rounding mode, and 1 destination register operand.
void RiscVDCvtWd(const Instruction* instruction);
void RiscVDCvtWud(const Instruction* instruction);
// The double precision compare instructions take 2 source register operands
// and a single destination register operand.
void RiscVDCmpeq(const Instruction* instruction);
void RiscVDCmplt(const Instruction* instruction);
void RiscVDCmple(const Instruction* instruction);
// The double precision class instruction takes a single source register operand
// and a single destination register operand.
void RiscVDClass(const Instruction* instruction);

// The double precision conversion instructions each take 1 source register
// operand, the rounding mode, and 1 destination register operand.
void RiscVDCvtLd(const Instruction* instruction);
void RiscVDCvtLud(const Instruction* instruction);
void RiscVDCvtDl(const Instruction* instruction);
void RiscVDCvtDlu(const Instruction* instruction);
// The move instruction takes a single register source operand and a single
// destination register operand.
void RiscVDMvxd(const Instruction* instruction);
void RiscVDMvdx(const Instruction* instruction);

}  // namespace RV64

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_D_INSTRUCTIONS_H_
