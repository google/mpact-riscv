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

#ifndef MPACT_RISCV_RISCV_RISCV_TEST_MEM_WATCHER_H_
#define MPACT_RISCV_RISCV_RISCV_TEST_MEM_WATCHER_H_

#include <cstdint>
#include <string>

#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/ref_count.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace mpact {
namespace sim {
namespace riscv {

// This class implements the memory interface and is used to capture memory
// loads and stores so that they can be printed in a trace similar to the
// that of Spike (reference RiscV simulator).
//
// For each load and/or store, the address (and in case of stores) are appended
// in text form to the trace_string. This trace string can be read out and
// and printed, and subsequently cleared at any time, though it is assumed that
// this always occur at the end of the simulator cpu cycle.

class RiscVTestMemWatcher : public util::MemoryInterface {
 public:
  explicit RiscVTestMemWatcher(util::MemoryInterface* memory)
      : memory_(memory) {}

  void Load(uint64_t address, generic::DataBuffer* db,
            generic::Instruction* inst,
            generic::ReferenceCount* context) override;
  void Load(generic::DataBuffer* address_db, generic::DataBuffer* mask_db,
            int el_size, generic::DataBuffer* db, generic::Instruction* inst,
            generic::ReferenceCount* context) override;
  void Store(uint64_t address, generic::DataBuffer* db) override;
  void Store(generic::DataBuffer* address_db, generic::DataBuffer* mask_db,
             int el_size, generic::DataBuffer* db) override;

  const std::string& trace_str() const { return trace_str_; }
  void clear_trace_str() { trace_str_.clear(); }

 private:
  util::MemoryInterface* memory_;
  std::string trace_str_;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_TEST_MEM_WATCHER_H_
