// Copyright 2024 Google LLC
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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_BITMANIP_INSTRUCTIONS_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_BITMANIP_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

// This file contains the declarations of the semantic functions for the
// bit manipulation instructions in RiscV.

namespace mpact::sim::riscv {

using ::mpact::sim::generic::Instruction;

namespace RV32 {

// This function takes 3 source operands, rs1, rs2, and a shift amount. It takes
// one destination operand rd.
void RiscVShAdd(const Instruction *instruction);
// This function takes 2 source operands, rs1, rs2, and one destination operand
// rd.
void RiscVAndn(const Instruction *instruction);
// This function takes 2 source operands, rs1, rs2, and one destination operand
// rd.
void RiscVOrn(const Instruction *instruction);
// This function takes 2 source operands, rs1, rs2, and one destination operand
// rd.
void RiscVXnor(const Instruction *instruction);
// This function takes 1 source operands, rs1, and one destination operand rd.
void RiscVClz(const Instruction *instruction);
// This function takes 1 source operands, rs1, and one destination operand rd.
void RiscVCtz(const Instruction *instruction);
// This function takes 1 source operands, rs1, and one destination operand rd.
void RiscVCpop(const Instruction *instruction);
// This function takes 2 source operands, rs1, rs2, and one destination operand
// rd.
void RiscVMax(const Instruction *instruction);
// This function takes 2 source operands, rs1, rs2, and one destination operand
// rd.
void RiscVMaxu(const Instruction *instruction);
// This function takes 2 source operands, rs1, rs2, and one destination operand
// rd.
void RiscVMin(const Instruction *instruction);
// This function takes 2 source operands, rs1, rs2, and one destination operand
// rd.
void RiscVMinu(const Instruction *instruction);
// This function takes 1 source operands, rs1, and one destination operand rd.
void RiscVSextB(const Instruction *instruction);
// This function takes 1 source operands, rs1, and one destination operand rd.
void RiscVSextH(const Instruction *instruction);
// This function takes 1 source operands, rs1, and one destination operand rd.
void RiscVZextH(const Instruction *instruction);
// This function takes 2 source operands, rs1, rs2, and one destination operand
// rd.
void RiscVrol(const Instruction *instruction);
// This function takes 2 source operands, rs1, rs2, and one destination operand
// rd.
void RiscVror(const Instruction *instruction);
// This function takes 1 source operands, rs1, and one destination operand rd.
void RiscVOrcb(const Instruction *instruction);
// This function takes 1 source operands, rs1, and one destination operand rd.
void RiscVRev8(const Instruction *instruction);
// This function takes 2 source operands, rs1, rs2, and one destination operand
// rd.
void RiscVClmul(const Instruction *instruction);
// This function takes 2 source operands, rs1, rs2, and one destination operand
// rd.
void RiscVClmulh(const Instruction *instruction);
// This function takes 2 source operands, rs1, rs2, and one destination operand
// rd.
void RiscVClmulr(const Instruction *instruction);
// The following functions take 2 source operands, rs1, rs2/imm, and one
// destination operand rd.
void RiscVBclr(const Instruction *instruction);
void RiscVBext(const Instruction *instruction);
void RiscVBinv(const Instruction *instruction);
void RiscVBset(const Instruction *instruction);

}  // namespace RV32

}  // namespace mpact::sim::riscv

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_BITMANIP_INSTRUCTIONS_H_
