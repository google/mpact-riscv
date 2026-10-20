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

#include "riscv/riscv_zicboz_instructions.h"

#include <cstdint>
#include <cstring>

#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "riscv/riscv_state.h"

// This file contains the definitions of the instruction semantic functions
// for the cache block zero instruction in the Zicboz extension.

namespace mpact::sim::riscv {

using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::generic::Instruction;

namespace {

// `cbo.zero` is a cache-block management instruction that writes zeros to the
// entire cache block containing the effective address in `rs1`.
//
// Unlike `Zicbom` cache management instructions (`cbo.clean`, `cbo.flush`,
// `cbo.inval`), which only affect microarchitectural cache state and can be
// modeled as no-ops when caches are absent, `cbo.zero` modifies architectural
// memory state so that subsequent loads from the block return zero. The
// instruction is modeled by aligning `rs1` down to a `cache_block_size`-byte
// boundary and storing `cache_block_size` bytes of zeros to memory.
template <typename UIntReg>
void CboZero(int cache_block_size, const Instruction* inst) {
  auto* const state = static_cast<RiscVState*>(inst->state());
  const UIntReg address = generic::GetInstructionSource<UIntReg>(inst, 0);
  const UIntReg block_mask = ~static_cast<UIntReg>(cache_block_size - 1);
  const UIntReg block_address = address & block_mask;
  DataBuffer* const db = state->db_factory()->Allocate(cache_block_size);
  std::memset(db->raw_ptr(), 0, cache_block_size);
  state->StoreMemory(inst, block_address, db);
  db->DecRef();
}

}  // namespace

namespace RV32 {

// Executes the RV32 `cbo.zero` instruction.
void RiscVCboZero(int cache_block_size, const Instruction* inst) {
  CboZero<uint32_t>(cache_block_size, inst);
}

}  // namespace RV32

namespace RV64 {

// Executes the RV64 `cbo.zero` instruction.
void RiscVCboZero(int cache_block_size, const Instruction* inst) {
  CboZero<uint64_t>(cache_block_size, inst);
}

}  // namespace RV64

}  // namespace mpact::sim::riscv
