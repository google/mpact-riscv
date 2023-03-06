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

#ifndef RISCV_RISCV_VECTOR_FP_INSTRUCTIONS_H_
#define RISCV_RISCV_VECTOR_FP_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

// This file declares the main binary and ternary floating point instruction
// semantic functions.

namespace mpact {
namespace sim {
namespace riscv {

using Instruction = ::mpact::sim::generic::Instruction;

// Vector floating point arithmetic instructions. Each of these instructions
// take three source operands and one destination operand. Source 0 is a vector
// register group, source 1 is either a vector register group or a scalar
// register, and source 2 is the mask register. Destination 0 is a vector
// register group.
void Vfadd(const Instruction *inst);
void Vfsub(const Instruction *inst);
void Vfrsub(const Instruction *inst);
void Vfwadd(const Instruction *inst);
void Vfwsub(const Instruction *inst);
void Vfwaddw(const Instruction *inst);
void Vfwsubw(const Instruction *inst);
void Vfmul(const Instruction *inst);
void Vfdiv(const Instruction *inst);
void Vfrdiv(const Instruction *inst);
void Vfwmul(const Instruction *inst);

// Vector floating point multiply and add/subtract instructions. Each of these
// instructions take four source operands and one destination operand. Source 0
// is a vector register group, source 1 is either a vector register group or a
// scalar register, source 2 is a vector register group, and source 3 is the
// mask register. Destination 0 is a vector register group.
void Vfmadd(const Instruction *inst);
void Vfnmadd(const Instruction *inst);
void Vfmsub(const Instruction *inst);
void Vfnmsub(const Instruction *inst);
void Vfmacc(const Instruction *inst);
void Vfnmacc(const Instruction *inst);
void Vfmsac(const Instruction *inst);
void Vfnmsac(const Instruction *inst);
void Vfwmacc(const Instruction *inst);
void Vfwnmacc(const Instruction *inst);
void Vfwmsac(const Instruction *inst);
void Vfwnmsac(const Instruction *inst);

// Vector floating point sign modification instructions. Each of these
// instructions take three source operands and one destination operand. Source 0
// is a vector register group, source 1 is either a vector register group or a
// scalar register, and source 2 is the mask register. Destination 0 is a vector
// register group.
void Vfsgnj(const Instruction *inst);
void Vfsgnjn(const Instruction *inst);
void Vfsgnjx(const Instruction *inst);

// Vector selection instructions. Each of these instructions take three source
// operands and one destination operand. Source 0 is a vector register group,
// source 1 is either a vector register group or a scalar register, and source 2
// is the mask register. Destination 0 is a vector register group.
void Vfmin(const Instruction *inst);
void Vfmax(const Instruction *inst);
void Vfmerge(const Instruction *inst);

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // RISCV_RISCV_VECTOR_FP_INSTRUCTIONS_H_
