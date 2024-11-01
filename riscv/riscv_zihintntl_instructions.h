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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_ZIHINTNTL_INSTRUCTIONS_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_ZIHINTNTL_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

// This file declares the semantic function for the RiscV non-temporal locality
// hint instructions.

namespace mpact::sim::riscv {

using ::mpact::sim::generic::Instruction;

// These instructions take no source or destination operands.
void RiscVNtlP1(const Instruction *);
void RiscVNtlPall(const Instruction *);
void RiscVNtlS1(const Instruction *);
void RiscVNtlAll(const Instruction *);

}  // namespace mpact::sim::riscv

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_ZIHINTNTL_INSTRUCTIONS_H_
