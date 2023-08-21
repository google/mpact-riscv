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

#include "riscv/riscv_breakpoint.h"

#include <cstdint>
#include <list>
#include <utility>

#include "absl/functional/bind_front.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace mpact {
namespace sim {
namespace riscv {

RiscVBreakpointManager::RiscVBreakpointManager(
    MemoryInterface *memory, InvalidateFcn invalidate_fcn,
    InstructionSizeFcn instruction_size_fcn)
    : memory_(memory),
      invalidate_fcn_(std::move(invalidate_fcn)),
      instruction_size_fcn_(std::move(instruction_size_fcn)) {
  // Allocate two data buffers (32 and 16 bit) once, so we don't have to
  // do it every time we access breakpoint instructions.
  db4_ = db_factory_.Allocate<uint32_t>(1);
  db2_ = db_factory_.Allocate<uint16_t>(1);
}

RiscVBreakpointManager::RiscVBreakpointManager(MemoryInterface *memory,
                                               InvalidateFcn invalidate_fcn)
    : RiscVBreakpointManager(
          memory, std::move(invalidate_fcn),
          absl::bind_front(&RiscVBreakpointManager::GetInstructionSize, this)) {
}

RiscVBreakpointManager::~RiscVBreakpointManager() {
  ClearAllBreakpoints();
  if (db4_ != nullptr) db4_->DecRef();
  if (db2_ != nullptr) db2_->DecRef();
}

bool RiscVBreakpointManager::HasBreakpoint(uint64_t address) {
  return breakpoint_map_.contains(address);
}

absl::Status RiscVBreakpointManager::SetBreakpoint(uint64_t address) {
  if (HasBreakpoint(address))
    return absl::AlreadyExistsError(
        absl::StrCat("Error SetBreakpoint: Breakpoint at ", absl::Hex(address),
                     " already exists"));

  // Read the instruction word at the location. Determine the size, then swap it
  // with the correct ebreak instruction.
  memory_->Load(address, db4_, nullptr, nullptr);
  uint32_t iword = db4_->Get<uint32_t>(0);
  int size = instruction_size_fcn_(address, iword);
  switch (size) {
    case 2:
      db2_->Set<uint16_t>(0, kEBreak16);
      memory_->Store(address, db2_);
      break;
    case 4:
      db4_->Set<uint32_t>(0, kEBreak32);
      memory_->Store(address, db4_);
      break;
    default:
      return absl::InternalError(absl::StrCat(
          "Error SetBreakpoint: No valid instruction of size 16 or 32 bits at ",
          absl::Hex(address)));
  }

  // Invalidate the decode cache entry.
  invalidate_fcn_(address);
  // Insert into the map.
  auto *bp_info = new BreakpointInfo(/*is_active=*/true, address, size, iword);
  breakpoint_map_.insert(std::make_pair(address, bp_info));

  return absl::OkStatus();
}

absl::Status RiscVBreakpointManager::ClearBreakpoint(uint64_t address) {
  // Lookup the breakpoint.
  auto iter = breakpoint_map_.find(address);
  if (iter == breakpoint_map_.end()) {
    auto msg = absl::StrCat("Error ClearBreakpoint: No breakpoint set for ",
                            absl::Hex(address));
    LOG(WARNING) << msg;
    return absl::NotFoundError(msg);
  }
  auto *bp = iter->second;

  // If it is active, first disable the breakpoint.
  if (bp->is_active) {
    auto status = DisableBreakpoint(address);
    if (!status.ok()) return status;
  }

  // Erase it from the breakpoint info map, delete the pointer.
  breakpoint_map_.erase(iter);
  delete bp;
  return absl::OkStatus();
}

absl::Status RiscVBreakpointManager::DisableBreakpoint(uint64_t address) {
  // Lookup the breakpoint.
  auto iter = breakpoint_map_.find(address);
  if (iter == breakpoint_map_.end()) {
    auto msg = absl::StrCat("Error DisableBreakpoint: No breakpoint set for ",
                            absl::Hex(address));
    LOG(WARNING) << msg;
    return absl::NotFoundError(msg);
  }

  auto *bp = iter->second;

  // If breakpoint is already disabled, return ok.
  if (!bp->is_active) return absl::OkStatus();

  // Write the original instruction back to memory, the
  if (bp->size == 2) {
    db2_->Set<uint16_t>(0,
                        static_cast<uint16_t>(bp->instruction_word & 0xffff));
    memory_->Store(bp->address, db2_);
  } else if (bp->size == 4) {
    db4_->Set<uint32_t>(0, bp->instruction_word);
    memory_->Store(bp->address, db4_);
  } else {
    auto msg = "DisableBreakpoint: wrong breakpoint size";
    return absl::InternalError(msg);
  }

  // Invalidate the decode cache entry.
  invalidate_fcn_(bp->address);
  bp->is_active = false;
  return absl::OkStatus();
}

absl::Status RiscVBreakpointManager::EnableBreakpoint(uint64_t address) {
  auto iter = breakpoint_map_.find(address);
  if (iter == breakpoint_map_.end())
    return absl::NotFoundError(absl::StrCat(
        "Error EnableBreakpoint: No breakpoint set for ", absl::Hex(address)));

  auto *bp = iter->second;

  // If breakpoint is already enabled, return ok.
  if (bp->is_active) return absl::OkStatus();

  // Write the appropriate ebreak instruction to the memory location.
  if (bp->size == 2) {
    db2_->Set<uint16_t>(0, kEBreak16);
    memory_->Store(bp->address, db2_);
  } else if (bp->size == 4) {
    db4_->Set<uint32_t>(0, kEBreak32);
    memory_->Store(bp->address, db4_);
  } else {
    auto msg = "EnableBreakpoint: wrong breakpoint size";
    LOG(WARNING) << msg;
    return absl::InternalError(msg);
  }
  // Invalidate the decode cache entry.
  invalidate_fcn_(bp->address);

  bp->is_active = true;
  return absl::OkStatus();
}
void RiscVBreakpointManager::ClearAllBreakpoints() {
  std::list<uint64_t> address_list;
  for (auto &[unused, bp_ptr] : breakpoint_map_) {
    address_list.push_back(bp_ptr->address);
  }
  for (auto address : address_list) {
    (void)ClearBreakpoint(address);
  }
  breakpoint_map_.clear();
}

bool RiscVBreakpointManager::IsBreakpoint(uint64_t address) const {
  auto iter = breakpoint_map_.find(address);
  if (iter == breakpoint_map_.end()) return false;

  auto *bp_info = iter->second;
  return bp_info->is_active;
}

int RiscVBreakpointManager::GetInstructionSize(
    uint64_t, uint32_t instruction_word) const {
  if ((instruction_word & 0b11) != 0b11) return 2;
  if ((instruction_word & 0b111'11) != 0b11111) return 4;
  return 0;
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
