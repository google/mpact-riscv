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

#include "riscv/riscv_zicbop_instructions.h"

#include "mpact/sim/generic/instruction.h"

// This file contains the definitions of the instruction semantic functions
// for the cache prefetch instructions in the Zicbop extension. For now, these
// instructions are implemented as no-ops.

namespace mpact::sim::riscv {

using ::mpact::sim::generic::Instruction;

namespace RV32 {

// These instructions are not implemented for now.
void RiscVPrefetchI(const Instruction *inst) { /* empty */ }
void RiscVPrefetchR(const Instruction *inst) { /* empty */ }
void RiscVPrefetchW(const Instruction *inst) { /* empty */ }

}  // namespace RV32

namespace RV64 {

// These instructions are not implemented for now.
void RiscVPrefetchI(const Instruction *inst) { /* empty */ }
void RiscVPrefetchR(const Instruction *inst) { /* empty */ }
void RiscVPrefetchW(const Instruction *inst) { /* empty */ }

}  // namespace RV64

}  // namespace mpact::sim::riscv
