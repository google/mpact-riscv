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

#ifndef MPACT_RISCV_RISCV_RISCV_VECTOR_FP_REDUCTION_INSTRUCTIONS_H_
#define MPACT_RISCV_RISCV_RISCV_VECTOR_FP_REDUCTION_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

// This file declares the semantic functions for the vector floating point
// reduction instructions.

namespace mpact {
namespace sim {
namespace riscv {

using Instruction = ::mpact::sim::generic::Instruction;

// Each of these instruction semantic functions take 3 source operands and 1
// destination operand. Source 0 is a vector register group, source 1 is a
// vector register, and source 2 is the vector mask register. Destination
// operand 0 is a vector register group.
void Vfredosum(const Instruction *inst);
void Vfwredosum(const Instruction *inst);
void Vfredmin(const Instruction *inst);
void Vfredmax(const Instruction *inst);

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_VECTOR_FP_REDUCTION_INSTRUCTIONS_H_
