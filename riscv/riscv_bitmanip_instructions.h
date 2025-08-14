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

// This function takes 3 source operands, rs1, rs2, a shift amount, and one
// destination operand rd.
void RiscVShAdd(const Instruction* instruction);
// These functions take 2 source operands, rs1, rs2, and one destination operand
// rd.
void RiscVAndn(const Instruction* instruction);
void RiscVOrn(const Instruction* instruction);
void RiscVXnor(const Instruction* instruction);
// Performs the bitwise negation of rs1 (this instruction is strictly not part
// of Zbb, but part of the 16 bit Zcb extension). It is implemented here because
// it fits with the other bitwise instructions.
void RiscVNot(const generic::Instruction* instruction);
// These functions take 1 source operands, rs1, and one destination operand rd.
void RiscVClz(const Instruction* instruction);
void RiscVCtz(const Instruction* instruction);
void RiscVCpop(const Instruction* instruction);
// These functions take 2 source operands, rs1, rs2, and one destination operand
// rd.
void RiscVMax(const Instruction* instruction);
void RiscVMaxu(const Instruction* instruction);
void RiscVMin(const Instruction* instruction);
void RiscVMinu(const Instruction* instruction);
// These functions take 1 source operands, rs1, and one destination operand rd.
void RiscVSextB(const Instruction* instruction);
void RiscVSextH(const Instruction* instruction);
void RiscVZextB(const Instruction* instruction);
void RiscVZextH(const Instruction* instruction);
// These functions take 2 source operands, rs1, rs2, and one destination operand
// rd.
void RiscVRol(const Instruction* instruction);
void RiscVRor(const Instruction* instruction);
// These functions take 1 source operands, rs1, and one destination operand rd.
void RiscVOrcb(const Instruction* instruction);
void RiscVRev8(const Instruction* instruction);
// These functions take 2 source operands, rs1, rs2, and one destination operand
// rd.
void RiscVClmul(const Instruction* instruction);
void RiscVClmulh(const Instruction* instruction);
void RiscVClmulr(const Instruction* instruction);
// The following functions take 2 source operands, rs1, rs2/imm, and one
// destination operand rd.
void RiscVBclr(const Instruction* instruction);
void RiscVBext(const Instruction* instruction);
void RiscVBinv(const Instruction* instruction);
void RiscVBset(const Instruction* instruction);

}  // namespace RV32

namespace RV64 {
// These functions take 3 source operands, rs1, rs2, a shift amount, and one
// one destination operand rd.
void RiscVAddUw(const Instruction* instruction);
void RiscVShAdd(const Instruction* instruction);
void RiscVShAddUw(const Instruction* instruction);
void RiscVSlliUw(const Instruction* instruction);
// These functions take 2 source operands, rs1, rs2, and one destination
// operand rd.
void RiscVAndn(const Instruction* instruction);
void RiscVOrn(const Instruction* instruction);
void RiscVXnor(const Instruction* instruction);
// These functions take 1 source operands, rs1, and one destination operand rd.
void RiscVClz(const Instruction* instruction);
void RiscVClzw(const Instruction* instruction);
void RiscVCtz(const Instruction* instruction);
void RiscVCtzw(const Instruction* instruction);
void RiscVCpop(const Instruction* instruction);
void RiscVCpopw(const Instruction* instruction);
// These functions take 2 source operands, rs1, rs2, and one destination operand
// rd.
void RiscVMax(const Instruction* instruction);
void RiscVMaxu(const Instruction* instruction);
void RiscVMin(const Instruction* instruction);
void RiscVMinu(const Instruction* instruction);
// These functions take 1 source operands, rs1, and one destination operand rd.
void RiscVSextB(const Instruction* instruction);
void RiscVSextH(const Instruction* instruction);
void RiscVZextH(const Instruction* instruction);
// These functions take 2 source operands, rs1, rs2, and one destination operand
// rd.
void RiscVRol(const Instruction* instruction);
void RiscVRor(const Instruction* instruction);
void RiscVRolw(const Instruction* instruction);
void RiscVRorw(const Instruction* instruction);
// These functions take 1 source operands, rs1, and one destination operand rd.
void RiscVOrcb(const Instruction* instruction);
void RiscVRev8(const Instruction* instruction);
// These function takes 2 source operands, rs1, rs2, and one destination operand
// rd.
void RiscVClmul(const Instruction* instruction);
void RiscVClmulh(const Instruction* instruction);
void RiscVClmulr(const Instruction* instruction);
// These functions take 2 source operands, rs1, rs2/imm, and one destination
// operand rd.
void RiscVBclr(const Instruction* instruction);
void RiscVBext(const Instruction* instruction);
void RiscVBinv(const Instruction* instruction);
void RiscVBset(const Instruction* instruction);

}  // namespace RV64

}  // namespace mpact::sim::riscv

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_BITMANIP_INSTRUCTIONS_H_
