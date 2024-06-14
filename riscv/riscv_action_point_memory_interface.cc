// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "riscv/riscv_action_point_memory_interface.h"

#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace mpact::sim::riscv {

RiscVActionPointMemoryInterface::RiscVActionPointMemoryInterface(
    MemoryInterface *memory, InvalidateFcn invalidate_fcn)
    : memory_(memory), invalidate_fcn_(std::move(invalidate_fcn)) {
  // Allocate two data buffers (32 and 16 bit) once, so we don't have to
  // do it every time we access breakpoint instructions.
  db4_ = db_factory_.Allocate<uint32_t>(1);
  db2_ = db_factory_.Allocate<uint16_t>(1);
}

RiscVActionPointMemoryInterface::~RiscVActionPointMemoryInterface() {
  if (db4_ != nullptr) db4_->DecRef();
  if (db2_ != nullptr) db2_->DecRef();
  for (auto &[unused, inst_info_ptr] : instruction_map_) {
    delete inst_info_ptr;
  }
  instruction_map_.clear();
}

absl::Status RiscVActionPointMemoryInterface::WriteOriginalInstruction(
    uint64_t address) {
  auto it = instruction_map_.find(address);
  if (it == instruction_map_.end()) {
    return absl::NotFoundError(absl::StrCat(
        "No instruction information exist for address: ", absl::Hex(address)));
  }
  if (it->second->size == 2) {
    db2_->Set<uint16_t>(
        0, static_cast<uint16_t>(it->second->og_instruction_word & 0xffff));
    memory_->Store(address, db2_);
  } else if (it->second->size == 4) {
    db4_->Set<uint32_t>(0, it->second->og_instruction_word);
    memory_->Store(address, db4_);
  }
  // Invalidate the instruction decode for this address.
  invalidate_fcn_(address);
  return absl::OkStatus();
}

absl::Status RiscVActionPointMemoryInterface::WriteBreakpointInstruction(
    uint64_t address) {
  auto it = instruction_map_.find(address);
  if (it == instruction_map_.end()) {
    // This is a new breakpoint. First find the instruction size, then save
    // the original instruction.
    memory_->Load(address, db4_, nullptr, nullptr);
    uint32_t instruction_word = db4_->Get<uint32_t>(0);
    int size = GetInstructionSize(instruction_word);
    if (size == 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid instruction size: ", size, " at ", absl::Hex(address)));
    }
    if (size == 2) instruction_word &= 0xffff;
    auto *inst_info = new InstructionInfo;
    inst_info->og_instruction_word = instruction_word;
    inst_info->size = size;
    it = instruction_map_.insert(std::make_pair(address, inst_info)).first;
  }
  // Write the breakpoint instruction to memory.
  if (it->second->size == 2) {
    db2_->Set<uint16_t>(0, kEBreak16);
    memory_->Store(address, db2_);
  } else if (it->second->size == 4) {
    db4_->Set<uint32_t>(0, kEBreak32);
    memory_->Store(address, db4_);
  }
  invalidate_fcn_(address);
  // Invalidate the instruction decode for this address.
  return absl::OkStatus();
}

int RiscVActionPointMemoryInterface::GetInstructionSize(
    uint32_t instruction_word) const {
  if ((instruction_word & 0b11) != 0b11) return 2;
  if ((instruction_word & 0b111'11) != 0b11111) return 4;
  return 0;
}

}  // namespace mpact::sim::riscv
