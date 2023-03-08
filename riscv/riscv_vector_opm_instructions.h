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

#ifndef MPACT_RISCV_RISCV_RISCV_VECTOR_OPM_INSTRUCTIONS_H_
#define MPACT_RISCV_RISCV_RISCV_VECTOR_OPM_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

namespace mpact {
namespace sim {
namespace riscv {

using Instruction = ::mpact::sim::generic::Instruction;

// Integer vector operations.
// Integer average unsigned add. This instruction takes 3 source operands and
// one destination. Source 0 is vs2, source 1 is vs1 (or rs1), and source 2
// is vector mask. The destination operand is a vector register group.
void Vaaddu(const Instruction *inst);
// Integer average signed add. This instruction takes 3 source operands and
// one destination. Source 0 is vs2, source 1 is vs1 (or rs1), and source 2
// is vector mask. The destination operand is a vector register group.
void Vaadd(const Instruction *inst);
// Integer average unsigned subtract. This instruction takes 3 source operands
// and one destination. Source 0 is vs2, source 1 is vs1 (or rs1), and source 2
// is vector mask. The destination operand is a vector register group.
void Vasubu(const Instruction *inst);
// Integer average signed subtract. This instruction takes 3 source operands
// and one destination. Source 0 is vs2, source 1 is vs1 (or rs1), and source 2
// is vector mask. The destination operand is a vector register group.
void Vasub(const Instruction *inst);
// The following instructions are vector mask logical operations. Each takes
// two source operands, vs2 and vs1 vector registers, and one destination
// operand, vd vector destination register.
void Vmandnot(const Instruction *inst);
void Vmand(const Instruction *inst);
void Vmor(const Instruction *inst);
void Vmxor(const Instruction *inst);
void Vmornot(const Instruction *inst);
void Vmnand(const Instruction *inst);
void Vmnor(const Instruction *inst);
void Vmxnor(const Instruction *inst);
// Integer unsigned division. This instruction takes 3 source operands and
// one destination. Source 0 is vs2, source 1 is vs1 (or rs1), and source 2
// is vector mask. The destination operand is a vector register group.
void Vdivu(const Instruction *inst);
// Integer signed division. This instruction takes 3 source operands and
// one destination. Source 0 is vs2, source 1 is vs1 (or rs1), and source 2
// is vector mask. The destination operand is a vector register group.
void Vdiv(const Instruction *inst);
// Integer unsigned remainder. This instruction takes 3 source operands and
// one destination. Source 0 is vs2, source 1 is vs1 (or rs1), and source 2
// is vector mask. The destination operand is a vector register group.
void Vremu(const Instruction *inst);
// Integer signed remainder. This instruction takes 3 source operands and
// one destination. Source 0 is vs2, source 1 is vs1 (or rs1), and source 2
// is vector mask. The destination operand is a vector register group.
void Vrem(const Instruction *inst);
// Integer unsigned multiply high. This instruction takes 3 source operands and
// one destination. Source 0 is vs2, source 1 is vs1 (or rs1), and source 2
// is vector mask. The destination operand is a vector register group.
void Vmulhu(const Instruction *inst);
// Integer signed multiply. This instruction takes 3 source operands and
// one destination. Source 0 is vs2, source 1 is vs1 (or rs1), and source 2
// is vector mask. The destination operand is a vector register group.
void Vmul(const Instruction *inst);
// Integer signed-unsigned multiply high. This instruction takes 3 source
// operands and one destination. Source 0 is vs2, source 1 is vs1 (or rs1), and
// source 2 is vector mask. The destination operand is a vector register group.
void Vmulhsu(const Instruction *inst);
// Integer signed multiply high. This instruction takes 3 source operands and
// one destination. Source 0 is vs2, source 1 is vs1 (or rs1), and source 2
// is vector mask. The destination operand is a vector register group.
void Vmulh(const Instruction *inst);
// Integer multiply add (vs1 * vd) + vs2. This instruction takes 4 source
// operands and one destination operand. Source 0 is vs2, source 1 is vs1 (or
// rs1), source 2 is Vd as a source operand, and source 4 is vector mask. The
// destination operand is the Vd register group.
void Vmadd(const Instruction *inst);
// Integer multiply subtract -(vs1 * vd) + vs2. This instruction takes 4
// source operands and one destination operand. Source 0 is vs2, source 1 is vs1
// (or rs1), source 2 is Vd as a source operand, and source 4 is vector mask.
// The destination operand is the Vd register group.
void Vnmsub(const Instruction *inst);
// Integer multiply add (vs1 * vs2) + vd. This instruction takes 4 source
// operands and one destination operand. Source 0 is vs2, source 1 is vs1 (or
// rs1), source 2 is Vd as a source operand, and source 4 is vector mask. The
// destination operand is the Vd register group.
void Vmacc(const Instruction *inst);
// Integer multiply subtract -(vs1 * vs2) + vd. This instruction takes 4 source
// operands and one destination operand. Source 0 is vs2, source 1 is vs1 (or
// rs1), source 2 is Vd as a source operand, and source 4 is vector mask. The
// destination operand is the Vd register group.
void Vnmsac(const Instruction *inst);
// Integer widening unsigned addition. This instruction takes 3 source operands
// and one destination. Source 0 is vs2, source 1 is vs1 (or rs1), and source 2
// is vector mask. The destination operand is a vector register group.
void Vwaddu(const Instruction *inst);
// Integer widening signed addition. This instruction takes 3 source operands
// and one destination. Source 0 is vs2, source 1 is vs1 (or rs1), and source 2
// is vector mask. The destination operand is a vector register group.
void Vwadd(const Instruction *inst);
// Integer widening unsigned subtraction. This instruction takes 3 source
// operands and one destination. Source 0 is vs2, source 1 is vs1 (or rs1), and
// source 2 is vector mask. The destination operand is a vector register group.
void Vwsubu(const Instruction *inst);
// Integer widening signed subtraction. This instruction takes 3 source operands
// and one destination. Source 0 is vs2, source 1 is vs1 (or rs1), and source 2
// is vector mask. The destination operand is a vector register group.
void Vwsub(const Instruction *inst);
// Integer widening unsigned addition with one wide source operand. This
// instruction takes 3 source operands and one destination. Source 0 is vs2
// (wide), source 1 is vs1 (or rs1), and source 2 is vector mask. The
// destination operand is a vector register group.
void Vwadduw(const Instruction *inst);
// Integer widening signed addition with one wide source operand. This
// instruction takes 3 source operands and one destination. Source 0 is vs2
// (wide), source 1 is vs1 (or rs1), and source 2 is vector mask. The
// destination operand is a vector register group.
void Vwaddw(const Instruction *inst);
// Integer widening unsigned subtraction with one wide source operand. This
// instruction takes 3 source operands and one destination. Source 0 is vs2
// (wide), source 1 is vs1 (or rs1), and source 2 is vector mask. The
// destination operand is a vector register group.
void Vwsubuw(const Instruction *inst);
// Integer widening signed subtraction with one wide source operand. This
// instruction takes 3 source operands and one destination. Source 0 is vs2
// (wide), source 1 is vs1 (or rs1), and source 2 is vector mask. The
// destination operand is a vector register group.
void Vwsubw(const Instruction *inst);
// Integer widening unsigned multiplication. This instruction takes 3 source
// operands and one destination. Source 0 is vs2, source 1 is vs1 (or rs1), and
// source 2 is vector mask. The destination operand is a vector register group.
void Vwmulu(const Instruction *inst);
// Integer widening signed by unsigned multiplication. This instruction takes 3
// source operands and one destination. Source 0 is vs2, source 1 is vs1 (or
// rs1), and source 2 is vector mask. The destination operand is a vector
// register group.
void Vwmulsu(const Instruction *inst);
// Integer widening signed multiplication. This instruction takes 3 source
// operands and one destination. Source 0 is vs2, source 1 is vs1 (or rs1), and
// source 2 is vector mask. The destination operand is a vector register group.
void Vwmul(const Instruction *inst);
// Integer widening signed multiply and add (vs2 * vs1) + vd. This instruction
// takes 4 source operands and one destination operand. Source 0 is vs2, source
// 1 is vs1 (or rs1), source 2 is Vd as a source operand, and source 4 is vector
// mask. The destination operand is the Vd register group.
void Vwmaccu(const Instruction *inst);
// Integer widening unsigned multiply and add (vs2 * vs1) + vd. This instruction
// takes 4 source operands and one destination operand. Source 0 is vs2, source
// 1 is vs1 (or rs1), source 2 is Vd as a source operand, and source 4 is vector
// mask. The destination operand is the Vd register group.
void Vwmacc(const Instruction *inst);
// Integer widening unsigned by signed multiply and add (vs2 * vs1) + vd. This
// instruction takes 4 source operands and one destination operand. Source 0 is
// vs2, source 1 is vs1 (or rs1), source 2 is Vd as a source operand, and source
// 4 is vector mask. The destination operand is the Vd register group.
void Vwmaccus(const Instruction *inst);
// Integer widening signed by unsigned multiply and add (vs2 * vs1) + vd. This
// instruction takes 4 source operands and one destination operand. Source 0 is
// vs2, source 1 is vs1 (or rs1), source 2 is Vd as a source operand, and source
// 4 is vector mask. The destination operand is the Vd register group.
void Vwmaccsu(const Instruction *inst);

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_VECTOR_OPM_INSTRUCTIONS_H_
