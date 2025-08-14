// Copyright 2025 Google LLC
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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_BASIC_BIT_MANIPULATION_INSTRUCTIONS_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_BASIC_BIT_MANIPULATION_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

namespace mpact {
namespace sim {
namespace riscv {

using Instruction = ::mpact::sim::generic::Instruction;

void RV32VUnimplementedInstruction(const Instruction* inst);

// Vector bit manipulation instructions.

// Zvkb subset of instructions
void Vandn(Instruction*);
void Vbrev8(Instruction*);
void Vrev8(Instruction*);
void Vrol(Instruction*);
void Vror(Instruction*);

// Zvbb instructions
void Vbrev(Instruction*);
void Vclz(Instruction*);
void Vctz(Instruction*);
// There is a name collision with an existing Vcpop instruction that stores the
// result in a scalar register. This implementation stores the result in a
// vector register.
void VectorVcpop(Instruction*);
void Vwsll(Instruction*);

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_BASIC_BIT_MANIPULATION_INSTRUCTIONS_H_
