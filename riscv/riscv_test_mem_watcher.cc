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

#include "riscv/riscv_test_mem_watcher.h"

#include <cstdint>

#include "absl/strings/str_cat.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/ref_count.h"

namespace mpact {
namespace sim {
namespace riscv {

// For now, the loads/and stores are assumed to be scalar loads/stores. Tracing
// support for vector loads/stores will be added later as necessary, when full
// support for RiscV vector instructions 1.0 is added to the RiscV torture test.

void RiscVTestMemWatcher::Load(uint64_t address, generic::DataBuffer* db,
                               generic::Instruction* inst,
                               generic::ReferenceCount* context) {
  absl::StrAppend(&trace_str_, " mem 0x", absl::Hex(address, absl::kZeroPad16));
  memory_->Load(address, db, inst, context);
}

void RiscVTestMemWatcher::Load(generic::DataBuffer* address_db,
                               generic::DataBuffer* mask_db, int el_size,
                               generic::DataBuffer* db,
                               generic::Instruction* inst,
                               generic::ReferenceCount* context) {
  for (auto const& address : address_db->Get<uint64_t>()) {
    absl::StrAppend(&trace_str_, " mem 0x",
                    absl::Hex(address, absl::kZeroPad16));
  }
  memory_->Load(address_db, mask_db, el_size, db, inst, context);
}

void RiscVTestMemWatcher::Store(uint64_t address, generic::DataBuffer* db) {
  absl::StrAppend(&trace_str_, " mem 0x", absl::Hex(address, absl::kZeroPad16));
  switch (db->size<uint8_t>()) {
    case 1:
      absl::StrAppend(&trace_str_, " 0x",
                      absl::Hex(db->Get<uint8_t>(0), absl::kZeroPad2));
      break;
    case 2:
      absl::StrAppend(&trace_str_, " 0x",
                      absl::Hex(db->Get<uint16_t>(0), absl::kZeroPad4));
      break;
    case 4:
      absl::StrAppend(&trace_str_, " 0x",
                      absl::Hex(db->Get<uint32_t>(0), absl::kZeroPad8));
      break;
    case 8:
      absl::StrAppend(&trace_str_, " 0x",
                      absl::Hex(db->Get<uint64_t>(0), absl::kZeroPad16));
      break;
    default:
      break;
  }
  memory_->Store(address, db);
}

void RiscVTestMemWatcher::Store(generic::DataBuffer* address_db,
                                generic::DataBuffer* mask_db, int el_size,
                                generic::DataBuffer* db) {
  for (auto const& address : address_db->Get<uint64_t>()) {
    absl::StrAppend(&trace_str_, " mem 0x",
                    absl::Hex(address, absl::kZeroPad16));
    switch (el_size) {
      case 1:
        absl::StrAppend(&trace_str_, " 0x",
                        absl::Hex(db->Get<uint8_t>(0), absl::kZeroPad2));
        break;
      case 2:
        absl::StrAppend(&trace_str_, " 0x",
                        absl::Hex(db->Get<uint16_t>(0), absl::kZeroPad4));
        break;
      case 4:
        absl::StrAppend(&trace_str_, " 0x",
                        absl::Hex(db->Get<uint32_t>(0), absl::kZeroPad8));
        break;
      case 8:
        absl::StrAppend(&trace_str_, " 0x",
                        absl::Hex(db->Get<uint64_t>(0), absl::kZeroPad16));
        break;
      default:
        break;
    }
  }
  memory_->Store(address_db, mask_db, el_size, db);
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
