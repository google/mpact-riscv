// Copyright 2026 Google LLC
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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_ZICBOM_INSTRUCTIONS_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_ZICBOM_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

// This file contains the declarations of the instruction semantic functions
// for the cache block management instructions in the Zicbom extension.

namespace mpact::sim::riscv {

using ::mpact::sim::generic::Instruction;

namespace RV32 {

// Each of these instructions takes one source operand: rs1, which contains the
// base address of the cache block to be cleaned, flushed, or invalidated.
void RiscVCboClean(const Instruction* inst);
void RiscVCboFlush(const Instruction* inst);
void RiscVCboInval(const Instruction* inst);

}  // namespace RV32

namespace RV64 {

// Each of these instructions takes one source operand: rs1, which contains the
// base address of the cache block to be cleaned, flushed, or invalidated.
void RiscVCboClean(const Instruction* inst);
void RiscVCboFlush(const Instruction* inst);
void RiscVCboInval(const Instruction* inst);

}  // namespace RV64

}  // namespace mpact::sim::riscv

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_ZICBOM_INSTRUCTIONS_H_
