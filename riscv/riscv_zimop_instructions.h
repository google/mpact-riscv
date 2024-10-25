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

// This file declares the semantic functions for the Zimop instructions.

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_ZIMOP_INSTRUCTIONS_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_ZIMOP_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

namespace mpact::sim::riscv {

using ::mpact::sim::generic::Instruction;

namespace RV32 {

// This semantic function ignores the source operand(s) and takes one
// destination operand.
void RiscVMop(const Instruction *inst);

}  // namespace RV32

namespace RV64 {

// This semantic function ignores the source operand(s) and takes one
// destination operand.
void RiscVMop(const Instruction *inst);

}  // namespace RV64

}  // namespace mpact::sim::riscv

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_ZIMOP_INSTRUCTIONS_H_
