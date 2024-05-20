#include "riscv/riscv_action_point.h"

#include <cstdint>
#include <utility>

#include "absl/functional/bind_front.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace mpact::sim::riscv {

RiscVActionPointManager::RiscVActionPointManager(
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

RiscVActionPointManager::RiscVActionPointManager(MemoryInterface *memory,
                                                 InvalidateFcn invalidate_fcn)
    : RiscVActionPointManager(
          memory, std::move(invalidate_fcn),
          absl::bind_front(&RiscVActionPointManager::GetInstructionSize,
                           this)) {}

RiscVActionPointManager::~RiscVActionPointManager() {
  for (auto &[unused, ap_ptr] : action_point_map_) {
    for (auto &[unused, ai_ptr] : ap_ptr->action_map) {
      ai_ptr->action_fcn = nullptr;
      delete ai_ptr;
    }
    ap_ptr->action_map.clear();
    delete ap_ptr;
  }
  action_point_map_.clear();
  if (db4_ != nullptr) db4_->DecRef();
  if (db2_ != nullptr) db2_->DecRef();
}

bool RiscVActionPointManager::HasActionPoint(uint64_t address) const {
  return action_point_map_.contains(address);
}

absl::StatusOr<int> RiscVActionPointManager::SetAction(uint64_t address,
                                                       ActionFcn action_fcn) {
  auto it = action_point_map_.find(address);
  ActionPointInfo *ap = nullptr;
  if (it == action_point_map_.end()) {
    // If the action point doesn't exist, create a new one.
    memory_->Load(address, db4_, nullptr, nullptr);
    auto instruction_word = db4_->Get<uint32_t>(0);
    int size = instruction_size_fcn_(address, instruction_word);
    if ((size != 2) && (size != 4)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid instruction size: ", size, " at ", absl::Hex(address)));
    }
    ap = new ActionPointInfo(address, size, instruction_word);
    action_point_map_.insert(std::make_pair(address, ap));
  } else {
    ap = it->second;
  }
  // Add the function as an enabled action.
  int id = ap->next_id++;
  auto *action_info =
      new ActionInfo(std::move(action_fcn), /*is_enabled=*/true);
  ap->action_map.insert(std::make_pair(id, action_info));
  ap->num_active++;
  // If there is only one active action (this), write a breakpoint instruction.
  if (ap->num_active == 1) WriteBreakpointInstruction(ap);
  return id;
}

absl::Status RiscVActionPointManager::ClearAction(uint64_t address, int id) {
  // Find the action point.
  auto it = action_point_map_.find(address);
  if (it == action_point_map_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No action point found at: ", absl::Hex(address)));
  }
  // Find the action.
  auto *ap = it->second;
  auto action_it = ap->action_map.find(id);
  if (action_it == ap->action_map.end()) {
    return absl::NotFoundError(
        absl::StrCat("No action ", id, "found at: ", absl::Hex(address)));
  }
  // Remove the action.
  auto *action_info = action_it->second;
  ap->action_map.erase(action_it);
  ap->num_active -= action_info->is_enabled ? 1 : 0;
  delete action_info;
  // If there are no other active actions, write back the original instruction.
  if (ap->num_active == 0) WriteOriginalInstruction(ap);
  return absl::OkStatus();
}

absl::Status RiscVActionPointManager::EnableAction(uint64_t address, int id) {
  // Find the action point.
  auto it = action_point_map_.find(address);
  if (it == action_point_map_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No action point found at: ", absl::Hex(address)));
  }
  auto *ap = it->second;
  // Find the action.
  auto action_it = ap->action_map.find(id);
  if (action_it == ap->action_map.end()) {
    return absl::NotFoundError(
        absl::StrCat("No action ", id, "found at: ", absl::Hex(address)));
  }
  // If it is already enabled, return ok.
  if (action_it->second->is_enabled) return absl::OkStatus();
  action_it->second->is_enabled = true;
  ap->num_active++;
  // If there is only one active action (this), write a breakpoint instruction.
  if (ap->num_active == 1) WriteBreakpointInstruction(ap);
  return absl::OkStatus();
}

absl::Status RiscVActionPointManager::DisableAction(uint64_t address, int id) {
  // Find the action point.
  auto it = action_point_map_.find(address);
  if (it == action_point_map_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No action point found at: ", absl::Hex(address)));
  }
  // Find the action.
  auto *ap = it->second;
  auto action_it = ap->action_map.find(id);
  if (action_it == ap->action_map.end()) {
    return absl::NotFoundError(
        absl::StrCat("No action ", id, "found at: ", absl::Hex(address)));
  }
  // If it is already disabled, return ok.
  if (!action_it->second->is_enabled) return absl::OkStatus();
  // Disable the action.
  action_it->second->is_enabled = false;
  ap->num_active--;
  // If there are no active actions, write back the original instruction.
  if (ap->num_active == 0) WriteOriginalInstruction(ap);
  return absl::OkStatus();
}

bool RiscVActionPointManager::IsActionPointActive(uint64_t address) const {
  // Find the action point.
  auto it = action_point_map_.find(address);
  if (it == action_point_map_.end()) return false;

  auto *ap = it->second;
  return ap->num_active > 0;
}

bool RiscVActionPointManager::IsActionEnabled(uint64_t address, int id) const {
  // Find the action point.
  auto it = action_point_map_.find(address);
  if (it == action_point_map_.end()) return false;

  auto *ap = it->second;
  // Find the action.
  auto action_it = ap->action_map.find(id);
  if (action_it == ap->action_map.end()) return false;
  return action_it->second->is_enabled;
}

void RiscVActionPointManager::ClearAllActionPoints() {
  for (auto &[address, ap_ptr] : action_point_map_) {
    for (auto &[unused, action_ptr] : ap_ptr->action_map) {
      delete action_ptr;
    }
    ap_ptr->action_map.clear();
    WriteOriginalInstruction(ap_ptr);
    delete ap_ptr;
  }
  action_point_map_.clear();
}

void RiscVActionPointManager::PerformActions(uint64_t address) {
  auto it = action_point_map_.find(address);
  if (it == action_point_map_.end()) {
    LOG(ERROR) << absl::StrCat("No action point found at: ",
                               absl::Hex(address));
  }

  for (auto &[id, action_ptr] : it->second->action_map) {
    if (action_ptr->is_enabled) action_ptr->action_fcn(address, id);
  }
}

void RiscVActionPointManager::WriteOriginalInstruction(uint64_t address) {
  // Find the action point.
  auto it = action_point_map_.find(address);
  // If the action point doesn't exist for this address, return, since the
  // original instruction is already there.
  if (it == action_point_map_.end()) return;

  WriteOriginalInstruction(it->second);
}

void RiscVActionPointManager::WriteBreakpointInstruction(uint64_t address) {
  // Find the action point.
  auto it = action_point_map_.find(address);
  // If the action point doesn't exist for this address, just return.
  if (it == action_point_map_.end()) return;

  WriteBreakpointInstruction(it->second);
}

void RiscVActionPointManager::WriteOriginalInstruction(ActionPointInfo *ap) {
  // Write the original instruction back to memory.
  if (ap->size == 2) {
    db2_->Set<uint16_t>(0,
                        static_cast<uint16_t>(ap->instruction_word & 0xffff));
    memory_->Store(ap->address, db2_);
  } else if (ap->size == 4) {
    db4_->Set<uint32_t>(0, ap->instruction_word);
    memory_->Store(ap->address, db4_);
  } else {
    LOG(ERROR) << absl::StrCat("Invalid stop point size: ", ap->size, " at ",
                               absl::Hex(ap->address));
    return;
  }
  // Invalidate the decode cache entry.
  invalidate_fcn_(ap->address);
}

void RiscVActionPointManager::WriteBreakpointInstruction(ActionPointInfo *ap) {
  if (ap->size == 2) {
    db2_->Set<uint16_t>(0, kEBreak16);
    memory_->Store(ap->address, db2_);
  } else if (ap->size == 4) {
    db4_->Set<uint32_t>(0, kEBreak32);
    memory_->Store(ap->address, db4_);
  } else {
    LOG(ERROR) << absl::StrCat("Invalid stop point size: ", ap->size, " at ",
                               absl::Hex(ap->address));
    return;
  }
  // Invalidate the decode cache entry.
  invalidate_fcn_(ap->address);
}

int RiscVActionPointManager::GetInstructionSize(
    uint64_t, uint32_t instruction_word) const {
  if ((instruction_word & 0b11) != 0b11) return 2;
  if ((instruction_word & 0b111'11) != 0b11111) return 4;
  return 0;
}

}  // namespace mpact::sim::riscv
