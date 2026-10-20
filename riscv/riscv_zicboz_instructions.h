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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_ZICBOZ_INSTRUCTIONS_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_ZICBOZ_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

// This file contains the declarations of the instruction semantic functions
// for the cache block zero instruction in the Zicboz extension.

namespace mpact::sim::riscv {

using ::mpact::sim::generic::Instruction;

namespace RV32 {

// Writes zeros to the `cache_block_size`-byte aligned cache block containing
// the effective address specified in `rs1`.
void RiscVCboZero(int cache_block_size, const Instruction* inst);

}  // namespace RV32

namespace RV64 {

// Writes zeros to the `cache_block_size`-byte aligned cache block containing
// the effective address specified in `rs1`.
void RiscVCboZero(int cache_block_size, const Instruction* inst);

}  // namespace RV64

}  // namespace mpact::sim::riscv

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_ZICBOZ_INSTRUCTIONS_H_
