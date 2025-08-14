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

#ifndef MPACT_RISCV_RISCV_RISCV_VECTOR_FP_COMPARE_INSTRUCTIONS_H_
#define MPACT_RISCV_RISCV_RISCV_VECTOR_FP_COMPARE_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

// This file declares the vector floating point compare instructions.

namespace mpact {
namespace sim {
namespace riscv {

using Instruction = ::mpact::sim::generic::Instruction;

// Each of these instructions take 3 source operands, and one destination
// operand. Source operand 0 is a vector register group, source operand 1 is
// either a vector register group or a scalar floating point register (Vmfgt and
// Vmfge only take the scalar register), source operand 2 is the vector mask
// register. Destination operand 0 is a vector register treated as the
// destination mask register.
void Vmfeq(const Instruction* inst);
void Vmfle(const Instruction* inst);
void Vmflt(const Instruction* inst);
void Vmfne(const Instruction* inst);
void Vmfgt(const Instruction* inst);
void Vmfge(const Instruction* inst);

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_VECTOR_FP_COMPARE_INSTRUCTIONS_H_
