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

#ifndef MPACT_RISCV_RISCV_RISCV_VECTOR_REDUCTION_INSTRUCTIONS_H_
#define MPACT_RISCV_RISCV_RISCV_VECTOR_REDUCTION_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

namespace mpact {
namespace sim {
namespace riscv {

using Instruction = ::mpact::sim::generic::Instruction;

// Each of these instruction semantic functions take 3 sources. Source 0 is
// vector register vs2, source 1 is vector register vs1, and source 2 is a
// vector mask operand. There is a single vector destination operand.
// Each reduction applies the reduction operation to the 0 element of source
// operand 1 (vs1), and all unmasked elements of source operand 0 (vs2). The
// result is written to the 0 element of destination operand vd.

// Vector sum reduction.
void Vredsum(Instruction* inst);
// Vector and reduction.
void Vredand(Instruction* inst);
// Vector or reduction.
void Vredor(Instruction* inst);
// Vector xor reduction.
void Vredxor(Instruction* inst);
// Vector unsigned min reduction.
void Vredminu(Instruction* inst);
// Vector signed min reduction.
void Vredmin(Instruction* inst);
// Vector unsigned max reduction.
void Vredmaxu(Instruction* inst);
// Vector signed max reduction.
void Vredmax(Instruction* inst);
// Vector unsigned widening sum reduction. The result is 2 * SEW.
void Vwredsumu(Instruction* inst);
// vector signed widening sum reduction. The result is 2 * SEW.
void Vwredsum(Instruction* inst);

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_VECTOR_REDUCTION_INSTRUCTIONS_H_
