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

#ifndef RISCV_RISCV_B_INSTRUCTIONS_H_
#define RISCV_RISCV_B_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

// This file contains the declarations of the instruction semantic functions
// for the RiscV bit manipulation instructions.

namespace mpact {
namespace sim {
namespace riscv {
namespace RV64 {

// The following instructions are part of the Zba extension.

// Performs an XLEN-wide addition between rs2 and the zero-extended least-
// significant word of rs1.
void RiscVAddUw(const generic::Instruction* instruction);

// Shifts rs1 to the left by 1, 2, or 3 bit(s) and adds it to rs2.
void RiscVSh1add(const generic::Instruction* instruction);
void RiscVSh2add(const generic::Instruction* instruction);
void RiscVSh3add(const generic::Instruction* instruction);

// Performs an XLEN-wide addition of two addends. The first addend is rs2. The
// second addend is the unsigned value formed by extracting the least-
// significant word of rs1 and shifting it left by 1, 2, or 3 bit(s).
void RiscVSh1adduw(const generic::Instruction* instruction);
void RiscVSh2adduw(const generic::Instruction* instruction);
void RiscVSh3adduw(const generic::Instruction* instruction);

// Takes the least-significant word of rs1, zero-extends it, and shifts it left
// by the immediate.
void RiscVSlliuw(const generic::Instruction* instruction);

// The following instructions are part of the Zbb extension.

// Performs the bitwise operations and, or, and xor between rs1 and the bitwise
// inversion of rs2.
void RiscVAndn(const generic::Instruction* instruction);
void RiscVOrn(const generic::Instruction* instruction);
void RiscVXnor(const generic::Instruction* instruction);

// Counts the leading (clz) or trailing (ctz) zeros in either the full XLEN
// register or in the least significant word (4-bytes) of the register.
void RiscVClz(const generic::Instruction* instruction);
void RiscVClzw(const generic::Instruction* instruction);
void RiscVCtz(const generic::Instruction* instruction);
void RiscVCtzw(const generic::Instruction* instruction);

// Counts the number of set bits in the full register or in the least
// significant word of the register.
void RiscVCpop(const generic::Instruction* instruction);
void RiscVCpopw(const generic::Instruction* instruction);

// Performs a signed or unsigned (u) maximum or minimum operation between the
// source registers, and stores the result in the destination register.
void RiscVMax(const generic::Instruction* instruction);
void RiscVMaxu(const generic::Instruction* instruction);
void RiscVMin(const generic::Instruction* instruction);
void RiscVMinu(const generic::Instruction* instruction);

// Sign extends (sext) or zero extends (zext) the word (w), half-word (h) or
// byte (b) in the source register, and stores the result in the destination
// register.
//
// Note that other size encodings are pseudo-instructions:
//  sext.w => addiw rd, rs1, zero
//  zext.b => andi rd, rs1, 0xFF
void RiscVSexth(const generic::Instruction* instruction);
void RiscVSextb(const generic::Instruction* instruction);
void RiscVZextw(const generic::Instruction* instruction);
void RiscVZexth(const generic::Instruction* instruction);

// Performs a rotate left (rol) or right (ror) of the full XLEN bits or least
// significant word (w) in register rs1 by the log2(XLEN) least significant bits
// in source register rs2 or the specified immediate value (i).
void RiscVRol(const generic::Instruction* instruction);
void RiscVRolw(const generic::Instruction* instruction);
void RiscVRor(const generic::Instruction* instruction);
void RiscVRorw(const generic::Instruction* instruction);
void RiscVRori(const generic::Instruction* instruction);
void RiscVRoriw(const generic::Instruction* instruction);

// Combines the bits within each byte using bitwise logical OR. This sets the
// bits of each byte in the result rd to all zeros if no bit within the
// respective byte of rs is set, or to all ones if any bit within the respective
// byte of rs is set.
void RiscVOrcb(const generic::Instruction* instruction);

// Reverses the order of the bytes in rs.
void RiscVRev8(const generic::Instruction* instruction);

}  // namespace RV64
}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // RISCV_RISCV_B_INSTRUCTIONS_H_
