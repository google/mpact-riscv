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

#ifndef MPACT_RISCV_RISCV_RISCV_VECTOR_OPI_INSTRUCTIONS_H_
#define MPACT_RISCV_RISCV_RISCV_VECTOR_OPI_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

namespace mpact {
namespace sim {
namespace riscv {

// This file declares the vector instruction semantic functions for most of the
// vector instructions in the OPIVV, OPIVX, and OPIVI encoding spaces. The
// exceptions are vector permute instructions and some vector reduction
// instructions.

using Instruction = ::mpact::sim::generic::Instruction;

// Integer vector operations.

// Element wide vector add. This instruction takes three source operands and
// a vector destination operand. Source 0 is the vs2 vector source. Source 1
// is either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a vector
// mask operand.
void Vadd(Instruction* inst);
// Element wide vector subtract. This instruction takes three source operands
// and a vector destination operand. Source 0 is the vs2 vector source. Source 1
// is either vs1 (vector) or rs1 (scalar). Source 2 is a vector
// mask operand.
void Vsub(Instruction* inst);
// Element wide vector reverse subtract. This instruction takes three source
// operands and a vector destination operand. Source 0 is the vs2 vector source.
// Source 1 is either rs1 (scalar), or an immediate. Source 2 is a vector mask
// operand.
void Vrsub(Instruction* inst);
// Element wide bitwise and. This instruction takes three source operands and
// a vector destination operand. Source 0 is the vs2 vector source. Source 1
// is either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a vector
// mask operand.
void Vand(Instruction* inst);
// Element wide bitwise or. This instruction takes three source operands and
// a vector destination operand. Source 0 is the vs2 vector source. Source 1
// is either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a vector
// mask operand.
void Vor(Instruction* inst);
// Element wide bitwise xor. This instruction takes three source operands and
// a vector destination operand. Source 0 is the vs2 vector source. Source 1
// is either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a vector
// mask operand.
void Vxor(Instruction* inst);
// Element wide logical left shift. This instruction takes three source operands
// and a vector destination operand. Source 0 is the vs2 vector source. Source 1
// is either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a vector
// mask operand.
void Vsll(Instruction* inst);
// Element wide logical right shift. This instruction takes three source
// operands and a vector destination operand. Source 0 is the vs2 vector source.
// Source 1 is either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a
// vector mask operand.
void Vsrl(Instruction* inst);
// Element wide arithmetic right shift. This instruction takes three source
// operands and a vector destination operand. Source 0 is the vs2 vector source.
// Source 1 is either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a
// vector mask operand.
void Vsra(Instruction* inst);
// Element wide narrowing logical right shift. This instruction takes three
// source operands and a vector destination operand. Source 0 is the vs2 vector
// source. Source 1 is either vs1 (vector), rs1 (scalar), or an immediate.
// Source 2 is a vector mask operand.
void Vnsrl(Instruction* inst);
// Element wide narrowing arithmetic right shift. This instruction takes three
// source operands and a vector destination operand. Source 0 is the vs2 vector
// source. Source 1 is either vs1 (vector), rs1 (scalar), or an immediate.
// Source 2 is a vector mask operand.
void Vnsra(Instruction* inst);
// Vector signed min (pairwise). This instruction takes three source operands
// and a vector destination operand. Source 0 is the vs2 vector source. Source 1
// is either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a vector
// mask operand.
void Vmin(Instruction* inst);
// Vector unsigned min (pairwise). This instruction takes three source operands
// and a vector destination operand. Source 0 is the vs2 vector source. Source 1
// is either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a vector
// mask operand.
void Vminu(Instruction* inst);
// Vector signed max (pairwise). This instruction takes three source operands
// and a vector destination operand. Source 0 is the vs2 vector source. Source 1
// is either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a vector
// mask operand.
void Vmax(Instruction* inst);
// Vector unsigned max (pairwise). This instruction takes three source operands
// and a vector destination operand. Source 0 is the vs2 vector source. Source 1
// is either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a vector
// mask operand.
void Vmaxu(Instruction* inst);
// Vector mask set equal. This instruction takes three source operands and a
// vector destination operand. Source 0 is the vs2 vector source. Source 1 is
// either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a vector mask
// operand.
void Vmseq(Instruction* inst);
// Vector mask set not equal.  This instruction takes three source operands and
// a vector destination operand. Source 0 is the vs2 vector source. Source 1 is
// either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a vector mask
// operand.
void Vmsne(Instruction* inst);
// Vector mask set less than unsigned. This instruction takes three source
// operands and a vector destination operand. Source 0 is the vs2 vector source.
// Source 1 is either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a
// vector mask operand.
void Vmsltu(Instruction* inst);
// Vector mask set less than signed. This instruction takes three source
// operands and a vector destination operand. Source 0 is the vs2 vector source.
// Source 1 is either vs1 (vector) or rs1 (scalar). Source 2 is a vector mask
// operand.
void Vmslt(Instruction* inst);
// Vector mask set less or equal unsigned. This instruction takes three source
// operands and a vector destination operand. Source 0 is the vs2 vector source.
// Source 1 is either vs1 (vector) or rs1 (scalar). Source 2 is a vector mask
// operand.
void Vmsleu(Instruction* inst);
// Vector mask set less or equal signed. This instruction takes three source
// operands and a vector destination operand. Source 0 is the vs2 vector source.
// Source 1 is either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a
// vector mask operand.
void Vmsle(Instruction* inst);
// Vector mask set greater than unsigned. This instruction takes three source
// operands and a vector destination operand. Source 0 is the vs2 vector source.
// Source 1 is either rs1 (scalar), or an immediate. Source 2 is a vector mask
// operand.
void Vmsgtu(Instruction* inst);
// Vector mask set greater than signed. This instruction takes three source
// operands and a vector destination operand. Source 0 is the vs2 vector source.
// Source 1 is either rs1 (scalar), or an immediate. Source 2 is a vector mask
// operand.
void Vmsgt(Instruction* inst);
// Vector saturating unsigned add. This instruction takes three source operands
// and a vector destination operand. Source 0 is the vs2 vector source. Source 1
// is either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a vector
// mask operand.
void Vsaddu(Instruction* inst);
// Vector saturating signed add. This instruction takes three source operands
// and a vector destination operand. Source 0 is the vs2 vector source. Source 1
// is either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a vector
// mask operand.
void Vsadd(Instruction* inst);
// Vector saturating unsigned subtrract. This instruction takes three source
// operands and a vector destination operand. Source 0 is the vs2 vector source.
// Source 1 is either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a
// vector mask operand.
void Vssubu(Instruction* inst);
// Vector saturating subtract. This instruction takes three source operands and
// a vector destination operand. Source 0 is the vs2 vector source. Source 1 is
// either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a vector mask
// operand.
void Vssub(Instruction* inst);
// Vector add with carry. This instruction takes three source operands and a
// vector destination operand. Source 0 is the vs2 vector source. Source 1 is
// either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a vector mask
// operand that contains the carry in values.
void Vadc(Instruction* inst);
// Vector add with carry - carry generate. This instruction takes three source
// operands and a vector destination operand. Source 0 is the vs2 vector source.
// Source 1 is either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a
// vector mask operand that contains the carry in values. The output of this
// instruction is the carry outs of each element wise addition. It is stored in
// the format of the vector flags.
void Vmadc(Instruction* inst);
// Vector subtract with borrow. This instruction takes three source operands and
// a vector destination operand. Source 0 is the vs2 vector source. Source 1 is
// either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a vector mask
// operand that contains the borrow values.
void Vsbc(Instruction* inst);
// Vector subtract with borrow - borrow generate. This instruction takes three
// source operands and a vector destination operand. Source 0 is the vs2 vector
// source. Source 1 is either vs1 (vector), rs1 (scalar), or an immediate.
// Source 2 is a vector mask operand that contains the borrow values. The output
// of this instruction is the borrow outs of each element wise subtraction. It
// is stored in the format of the vector flags.
void Vmsbc(Instruction* inst);
// Vector pairwise merge. This instruction takes three source operands and a
// vector destination operand. Source 0 is the vs2 vector source. Source 1 is
// either vs1 (vector), rs1 (scalar), or an immediate. Source 2 is a vector mask
// operand. This semantic function also captures the functionality of vmv.vv,
// vmv.vx, and vmv.vi, in which case vs2 is register group v0, and the mask
// is all ones.
void Vmerge(Instruction* inst);
// Vector register move. This instruction takes one source operands and a
// vector destination operand. Source 0 is the vs2 vector source. The num_regs
// value is part of the opcode and should be bound to the semantic function at
// decode.
void Vmvr(int num_regs, Instruction* inst);
// Vector logical right shift with rounding. This instruction takes three
// source operands and a vector destination operand. Source 0 is the vs2 vector
// source. Source 1 is either vs1 (vector), rs1 (scalar), or an immediate.
// Source 2 is a vector mask operand.
void Vssrl(Instruction* inst);
// Vector arithmetic right shift with rounding. This instruction takes three
// source operands and a vector destination operand. Source 0 is the vs2 vector
// source. Source 1 is either vs1 (vector), rs1 (scalar), or an immediate.
// Source 2 is a vector mask operand.
void Vssra(Instruction* inst);
// Vector logical right shift with rounding and (unsigned) saturation from SEW *
// 2 to SEW wide elements. This instruction takes three
// source operands and a vector destination operand. Source 0 is the vs2 vector
// source. Source 1 is either vs1 (vector), rs1 (scalar), or an immediate.
// Source 2 is a vector mask operand.
void Vnclipu(Instruction* inst);
// Vector arithmetic right shift with rounding and (signed) saturation from SEW
// * 2 to SEW wide elements. This instruction takes three
// source operands and a vector destination operand. Source 0 is the vs2 vector
// source. Source 1 is either vs1 (vector), rs1 (scalar), or an immediate.
// Source 2 is a vector mask operand.
void Vnclip(Instruction* inst);
// Vector fractional multiply. This instruction takes three
// source operands and a vector destination operand. Source 0 is the vs2 vector
// source. Source 1 is either vs1 (vector) or rs1 (scalar). Source 2 is a vector
// mask operand.
void Vsmul(Instruction* inst);

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_VECTOR_OPI_INSTRUCTIONS_H_
