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

#ifndef RISCV_RISCV_VECTOR_PERMUTE_INSTRUCTIONS_H_
#define RISCV_RISCV_VECTOR_PERMUTE_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

namespace mpact {
namespace sim {
namespace riscv {

using Instruction = ::mpact::sim::generic::Instruction;

// Vector register gather instruction. This instruction takes three source
// operands and one destination operand. Source 0 is the vector from which
// elements are gathered, source 1 is the index vector, and source 2 is the
// vector mask register. The destination operand is the target vector register.
void Vrgather(Instruction *inst);
// Vector register gather instruction with 16 bit indices. This instruction
// takes three source operands and one destination operand. Source 0 is the
// vector from which elements are gathered, source 1 is the index vector, and
// source 2 is the vector mask register. The destination operand is the target
// vector register.
void Vrgatherei16(Instruction *inst);
// Vector slide up instruction. This instruction takes three source operands
// and one destination operand. Source 0 is the vector source register that
// contains the values that are 'slid' up. Source 1 is a scalar register or
// immediate that specifies the number of 'entries' by which source 0 values
// are slid up. Source 2 is the vector mask register. The destination operand
// is the target vector register.
void Vslideup(Instruction *inst);
// Vector slide down instruction. This instruction takes three source operands
// and one destination operand. Source 0 is the vector source register that
// contains the values that are 'slid' down. Source 1 is a scalar register or
// immediate that specifies the number of 'entries' by which source 0 values
// are slid down. Source 2 is the vector mask register. The destination operand
// is the target vector register.
void Vslidedown(Instruction *inst);
// Vector slide up instruction. This instruction takes three source operands
// and one destination operand. Source 0 is the vector source register that
// contains the values that are 'slid' up by 1. Source 1 is a scalar register or
// immediate that specifies the value written into the 'empty' slot. Source 2 is
// the vector mask register. The destination operand is the target vector
// register.
void Vslide1up(Instruction *inst);
// Vector slide down instruction. This instruction takes three source operands
// and one destination operand. Source 0 is the vector source register that
// contains the values that are 'slid' down. Source 1 is a scalar register or
// immediate that specifies the value written into the 'empty' slot. Source 2 is
// the vector mask register. The destination operand is the target vector
// register.
void Vslide1down(Instruction *inst);
// Vector fp slide up instruction. This instruction takes three source operands
// and one destination operand. Source 0 is the vector source register that
// contains the values that are 'slid' up by 1. Source 1 is a floating point
// register or immediate that specifies the value written into the 'empty' slot.
// Source 2 is the vector mask register. The destination operand is the target
// vector register.
void Vfslide1up(Instruction *inst);
// Vector fp slide down instruction. This instruction takes three source
// operands and one destination operand. Source 0 is the vector source register
// that contains the values that are 'slid' down. Source 1 is a floating point
// register or immediate that specifies the value written into the 'empty' slot.
// Source 2 is the vector mask register. The destination operand is the target
// vector register.
void Vfslide1down(Instruction *inst);
// Vector compress instruction. This instruction takes two source operands and
// one destination operand. Source 0 is the source value vector register. Source
// 1 is a mask register, with specifies which elements of source 0 should be
// selected and packed into the destination register.
void Vcompress(Instruction *inst);

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // RISCV_RISCV_VECTOR_PERMUTE_INSTRUCTIONS_H_
