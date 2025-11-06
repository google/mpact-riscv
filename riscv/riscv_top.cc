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

#include "riscv/riscv_top.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>  // NOLINT: third_party code.
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "mpact/sim/generic/action_point_manager_base.h"
#include "mpact/sim/generic/breakpoint_manager.h"
#include "mpact/sim/generic/component.h"
#include "mpact/sim/generic/core_debug_interface.h"
#include "mpact/sim/generic/counters.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/decode_cache.h"
#include "mpact/sim/generic/decoder_interface.h"
#include "riscv/riscv_action_point_memory_interface.h"
#include "riscv/riscv_counter_csr.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_debug_interface.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
// Uncomment if using resource checks below.
// #include "mpact/sim/generic/resource_operand_interface.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/cache.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "mpact/sim/util/memory/memory_watcher.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::ActionPointManagerBase;
using ::mpact::sim::generic::BreakpointManager;

// Local helper function used to execute instructions.
static inline bool ExecuteInstruction(Instruction* inst) {
  // The following code can be used to model stalls due to latency of operand
  // writes that are used by subsequent instructions. Instruction latencies
  // are defined in the .isa file.
  /*
  for (auto *resource : inst->ResourceHold()) {
    if (!resource->IsFree()) {
      return false;
    }
  }
  for (auto *resource : inst->ResourceAcquire()) {
    resource->Acquire();
  }
  */
  inst->Execute(nullptr);
  // Comment out instruction logging during execution.
  // LOG(INFO) << "[" << std::hex << inst->address() << "] " <<
  // inst->AsString();
  return true;
}

RiscVTop::RiscVTop(std::string name, RiscVState* state,
                   generic::DecoderInterface* decoder)
    : Component(name),
      state_(state),
      rv_decoder_(decoder),
      counter_num_instructions_("num_instructions", 0),
      counter_num_cycles_("num_cycles", 0),
      icache_config_("icache", ""),
      dcache_config_("dcache", "") {
  CHECK_OK(AddConfig(&icache_config_));
  icache_config_.AddValueWrittenCallback(
      [this]() { ConfigureCache(icache_, icache_config_); });
  CHECK_OK(AddConfig(&dcache_config_));
  dcache_config_.AddValueWrittenCallback(
      [this]() { ConfigureCache(dcache_, dcache_config_); });
  Initialize();
}

RiscVTop::~RiscVTop() {
  // If the simulator is still running, request a halt (set halted_ to true),
  // and wait until the simulator finishes before continuing the destructor.
  if (run_status_ == RunStatus::kRunning) {
    run_halted_->WaitForNotification();
    delete run_halted_;
    run_halted_ = nullptr;
  }

  if (branch_trace_db_ != nullptr) branch_trace_db_->DecRef();

  delete icache_;
  delete dcache_;
  if (inst_db_) inst_db_->DecRef();
  delete rv_breakpoint_manager_;
  delete rv_action_point_manager_;
  delete rv_action_point_memory_interface_;
  delete rv_decode_cache_;
  delete memory_watcher_;
}

void RiscVTop::Initialize() {
  pc_ = state_->registers()->at(RiscVState::kPcName);
  rv_decode_cache_ = generic::DecodeCache::Create({16 * 1024, 2}, rv_decoder_);

  // Replace the memory with the memory watcher.
  memory_watcher_ = new util::MemoryWatcher(state_->memory());
  state_->set_memory(memory_watcher_);

  counter_hardware_perf_.resize(kNumHardwarePerfCounters);
  for (int i = 0; i < kNumHardwarePerfCounters; i++) {
    counter_hardware_perf_[i].Initialize(absl::StrCat("hardware_perf_", i), 0);
    CHECK_OK(AddCounter(&counter_hardware_perf_[i]))
        << "Failed to register hardware_perf counter";
  }

  // Register instruction and cycle counters.
  CHECK_OK(AddCounter(&counter_num_instructions_))
      << "Failed to register instruction counter";
  CHECK_OK(AddCounter(&counter_num_cycles_))
      << "Failed to register cycle counter";
  // Register opcode counters.
  int num_opcodes = rv_decoder_->GetNumOpcodes();
  counter_opcode_.resize(num_opcodes);
  for (int i = 0; i < num_opcodes; i++) {
    counter_opcode_.push_back(generic::SimpleCounter<uint64_t>());
    counter_opcode_[i].Initialize(
        absl::StrCat("num_", rv_decoder_->GetOpcodeName(i)), 0);
    CHECK_OK(AddCounter(&counter_opcode_[i]))
        << "Failed to register opcode counter";
  }

  // Connect counters to instret(h) and mcycle(h) CSRs.
  CHECK_OK(SetCsrCounter("minstret", counter_num_instructions_));
  CHECK_OK(SetCsrCounter("mcycle", counter_num_cycles_));

  // Connect Zicntr counters to cycle(h), time(h), and instret(h) CSRs.
  CHECK_OK(SetCsrCounter("instret", counter_num_instructions_));
  CHECK_OK(SetCsrCounter("cycle", counter_num_cycles_));
  CHECK_OK(SetCsrCounter("time", counter_num_cycles_));

  // Add Zihpm counters - Unprivileged.
  for (int i = 0; i < kNumHardwarePerfCounters; i++) {
    std::string name =
        absl::StrCat("hpmcounter", i + kMinimumHardwarePerfIndex);
    CHECK_OK(SetCsrCounter(name, counter_hardware_perf_[i]));
  }

  // Add Zihpm counters - Machine Privileged.
  for (int i = 0; i < kNumHardwarePerfCounters; i++) {
    std::string name =
        absl::StrCat("mhpmcounter", i + kMinimumHardwarePerfIndex);
    CHECK_OK(SetCsrCounter(name, counter_hardware_perf_[i]));
  }

  // Set up break and action points.
  rv_action_point_memory_interface_ = new RiscVActionPointMemoryInterface(
      state_->memory(),
      absl::bind_front(&generic::DecodeCache::Invalidate, rv_decode_cache_));
  rv_action_point_manager_ =
      new ActionPointManagerBase(rv_action_point_memory_interface_);
  rv_breakpoint_manager_ = new BreakpointManager(
      rv_action_point_manager_,
      [this]() { RequestHalt(HaltReason::kSoftwareBreakpoint, nullptr); });

  // Set the ebreak handler callback.
  state_->AddEbreakHandler([this](const Instruction* inst) {
    if (rv_action_point_manager_->IsActionPointActive(inst->address())) {
      // Need to request a halt so that the action point can be stepped past
      // after executing the actions. However, an action may override the
      // particular halt reason (e.g., breakpoints).
      RequestHalt(HaltReason::kActionPoint, inst);
      rv_action_point_manager_->PerformActions(inst->address());
      return true;
    }
    return false;
  });
  inst_db_ = db_factory_.Allocate<uint32_t>(1);
  // Branch trace.
  branch_trace_db_ = db_factory_.Allocate<BranchTraceEntry>(kBranchTraceSize);
  branch_trace_ =
      reinterpret_cast<BranchTraceEntry*>(branch_trace_db_->raw_ptr());
  for (int i = 0; i < kBranchTraceSize; i++) {
    branch_trace_[i] = {0, 0, 0};
  }
}

void RiscVTop::ConfigureCache(Cache*& cache, Config<std::string>& config) {
  if (cache != nullptr) {
    LOG(WARNING) << "Cache already configured - ignored";
    return;
  }
  auto cfg_str = config.GetValue();
  if (cfg_str.empty()) {
    LOG(WARNING) << "Cache configuration is empty - ignored";
  }
  cache = new util::Cache(config.name(), this);
  absl::Status status = cache->Configure(cfg_str, &counter_num_cycles_);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to configure instruction cache: " << status.message();
  }
}

absl::Status RiscVTop::Halt() {
  // If it is already halted, just return.
  if (run_status_ == RunStatus::kHalted) {
    return absl::OkStatus();
  }
  // If it is not running, then there's an error.
  if (run_status_ != RunStatus::kRunning) {
    return absl::FailedPreconditionError("RiscVTop::Halt: Core is not running");
  }
  halt_reason_ = *HaltReason::kUserRequest;
  halted_ = true;
  return absl::OkStatus();
}

absl::Status RiscVTop::Halt(HaltReason halt_reason) {
  RequestHalt(halt_reason, nullptr);
  return absl::OkStatus();
}

absl::Status RiscVTop::Halt(HaltReasonValueType halt_reason) {
  RequestHalt(halt_reason, nullptr);
  return absl::OkStatus();
}

absl::Status RiscVTop::StepPastBreakpoint() {
  uint64_t pc = state_->pc_operand()->AsUint64(0);
  uint64_t bpt_pc = pc;
  // Disable the breakpoint.
  (void)rv_action_point_manager_->ap_memory_interface()
      ->WriteOriginalInstruction(pc);
  // Execute the real instruction.
  auto real_inst = rv_decode_cache_->GetDecodedInstruction(pc);
  real_inst->IncRef();
  uint64_t next_pc = pc + real_inst->size();
  bool executed = false;
  if (icache_) ICacheFetch(pc);
  do {
    executed = ExecuteInstruction(real_inst);
    counter_num_cycles_.Increment(1);
    state_->AdvanceDelayLines();
  } while (!executed);
  // Increment counters.
  counter_opcode_[real_inst->opcode()].Increment(1);
  counter_num_instructions_.Increment(1);
  real_inst->DecRef();
  // Re-enable the breakpoint.
  (void)rv_action_point_manager_->ap_memory_interface()
      ->WriteBreakpointInstruction(bpt_pc);
  // Check for interrupt.
  if (state_->is_interrupt_available()) {
    uint64_t epc = pc;
    if (executed) {
      epc = state_->branch() ? state_->pc_operand()->AsUint64(0) : next_pc;
    }
    state_->TakeAvailableInterrupt(epc);  // Will set state_->branch().
  }
  if (state_->branch()) {
    state_->set_branch(false);
    auto new_pc = state_->pc_operand()->AsUint64(0);
    AddToBranchTrace(pc, bpt_pc);
    next_pc = new_pc;
  }
  SetPc(next_pc);
  return absl::OkStatus();
}

absl::StatusOr<int> RiscVTop::Step(int num) {
  if (num <= 0) {
    return absl::InvalidArgumentError("Step count must be > 0");
  }
  // If the simulator is running, return with an error.
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError("RiscVTop::Step: Core must be halted");
  }
  run_status_ = RunStatus::kSingleStep;
  int count = 0;
  halted_ = false;
  paused_ = false;
  halt_reason_ = *HaltReason::kNone;
  // First check to see if the previous halt was due to a breakpoint. If so,
  // need to step over the breakpoint.
  if (need_to_step_over_) {
    need_to_step_over_ = false;
    auto status = StepPastBreakpoint();
    if (!status.ok()) {
      run_status_ = RunStatus::kHalted;
      return status;
    }
    paused_ |= halted_;
    count++;
  }

  // Step the simulator forward until the number of steps have been achieved, or
  // there is a halt request.
  auto pc_operand = state_->pc_operand();
  // This holds the value of the current pc, and post-loop, the address of
  // the most recently executed instruction.
  uint64_t pc;
  // At the top of the loop this holds the address of the instruction to be
  // executed next. Post-loop it holds the address of the next instruction to
  // be executed.
  uint64_t next_pc = pc_operand->AsUint64(0);
  pc = next_pc;
  while (!paused_ && (count < num)) {
    SetPc(pc);
    auto* inst = rv_decode_cache_->GetDecodedInstruction(pc);
    // Set the next_pc to the next sequential instruction.
    next_pc = pc + inst->size();
    bool executed = false;
    if (icache_) ICacheFetch(pc);
    do {
      executed = ExecuteInstruction(inst);
      counter_num_cycles_.Increment(1);
      state_->AdvanceDelayLines();
      // Check for interrupt.
      if (state_->is_interrupt_available()) {
        uint64_t epc = pc;
        if (executed) {
          epc = state_->branch() ? state_->pc_operand()->AsUint64(0) : next_pc;
        }
        state_->TakeAvailableInterrupt(epc);  // Will set state_->branch().
      }
    } while (!executed);
    paused_ |= halted_;
    count++;
    // Update counters.
    counter_opcode_[inst->opcode()].Increment(1);
    counter_num_instructions_.Increment(1);
    // Get the next pc value.
    auto pc_val = state_->pc_operand()->AsUint64(0);
    if (state_->branch()) {
      state_->set_branch(false);
      AddToBranchTrace(pc, pc_val);
      next_pc = pc_val;
    }
    if (!paused_) {
      pc = next_pc;
      continue;
    }
    // If it's an action point, just step over and continue.
    if (halt_reason_ == *HaltReason::kActionPoint) {
      auto status = StepPastBreakpoint();
      if (!status.ok()) {
        run_status_ = RunStatus::kHalted;
        halted_ = true;
        return status;
      }
      // Reset the halt reason and continue;
      paused_ = halted_;
      halt_reason_ = *HaltReason::kNone;
      need_to_step_over_ = false;
      continue;
    }
    break;
  }
  // Update the pc register, now that it can be read.
  if (halt_reason_ == *HaltReason::kSoftwareBreakpoint) {
    // If at a breakpoint, keep the pc at the current value.
    SetPc(pc);
  } else {
    // Otherwise set it to point to the next instruction.
    SetPc(next_pc);
  }
  // If there is no halt request, there is no specific halt reason.
  if (!halted_) {
    halt_reason_ = *HaltReason::kNone;
  }
  run_status_ = RunStatus::kHalted;
  return count;
}

absl::Status RiscVTop::Run() {
  // Verify that the core isn't running already.
  if (run_status_ == RunStatus::kRunning) {
    return absl::FailedPreconditionError(
        "RiscVTop::Run: core is already running");
  }
  // First check to see if the previous halt was due to a breakpoint. If so,
  // need to step over the breakpoint.
  if (need_to_step_over_) {
    need_to_step_over_ = false;
    auto status = StepPastBreakpoint();
    if (!status.ok()) return status;
  }
  // The simulator is now run in a separate thread so as to allow a user
  // interface to continue operating. Allocate a new run_halted_ Notification
  // object, as they are single use only.
  run_halted_ = new absl::Notification();
  run_started_ = new absl::Notification();
  // The thread is detached so it executes without having to be joined.
  std::thread([this]() {
    run_status_ = RunStatus::kRunning;
    halted_ = false;
    paused_ = false;
    halt_reason_ = *HaltReason::kNone;
    run_started_->Notify();
    auto pc_operand = state_->pc_operand();
    // At the top of the loop this holds the address of the instruction to be
    // executed next. Post-loop it holds the address of the next instruction to
    // be executed.
    uint64_t next_pc = pc_operand->AsUint64(0);
    // This holds the value of the current pc, and post-loop, the address of
    // the most recently executed instruction.
    uint64_t pc = next_pc;
    while (!paused_) {
      auto* inst = rv_decode_cache_->GetDecodedInstruction(pc);
      SetPc(pc);
      next_pc = pc + inst->size();
      bool executed = false;
      if (icache_) ICacheFetch(pc);
      do {
        // Try executing the instruction. If it fails, advance a cycle
        // and try again.
        executed = ExecuteInstruction(inst);
        counter_num_cycles_.Increment(1);
        state_->AdvanceDelayLines();
        // Check for interrupt.
        if (state_->is_interrupt_available()) {
          uint64_t epc = pc;
          if (executed) {
            epc =
                state_->branch() ? state_->pc_operand()->AsUint64(0) : next_pc;
          }
          state_->TakeAvailableInterrupt(epc);  // Will set state_->branch().
        }
      } while (!executed);
      paused_ |= halted_;
      // Update counters.
      counter_opcode_[inst->opcode()].Increment(1);
      counter_num_instructions_.Increment(1);
      // Get the next pc value.
      uint64_t pc_val = pc_operand->AsUint64(0);
      if (state_->branch()) {
        state_->set_branch(false);
        AddToBranchTrace(pc, pc_val);
        next_pc = pc_val;
      }
      if (!paused_) {
        pc = next_pc;
        continue;
      }
      // If it's an action point, just step over and continue executing, as
      // this is not a full breakpoint.
      if (halt_reason_ == *HaltReason::kActionPoint) {
        auto status = StepPastBreakpoint();
        if (!status.ok()) {
          // If there is an error, signal a simulator error.
          halt_reason_ = *HaltReason::kSimulatorError;
          break;
        };
        // Reset the halt reason and continue;
        paused_ = halted_;
        halt_reason_ = *HaltReason::kNone;
        need_to_step_over_ = false;
        continue;
      }
      break;
    }
    // Update the pc register, now that it can be read.
    if (halt_reason_ == *HaltReason::kSoftwareBreakpoint) {
      // If at a breakpoint, keep the pc at the current value.
      SetPc(pc);
    } else {
      // Otherwise set it to point to the next instruction.
      SetPc(next_pc);
    }
    run_status_ = RunStatus::kHalted;
    // Notify that the run has completed.
    run_halted_->Notify();
  }).detach();
  run_started_->WaitForNotification();
  delete run_started_;
  run_started_ = nullptr;
  return absl::OkStatus();
}

absl::Status RiscVTop::Wait() {
  // If the simulator isn't running, then just return after deleting
  // the notification object.
  if (run_status_ != RunStatus::kRunning) {
    delete run_halted_;
    run_halted_ = nullptr;
    return absl::OkStatus();
  }

  // Wait for the simulator to finish - i.e., a notification on run_halted_.
  run_halted_->WaitForNotification();
  // Now delete the notification object - it is single use only.
  delete run_halted_;
  run_halted_ = nullptr;
  return absl::OkStatus();
}

absl::StatusOr<RiscVTop::RunStatus> RiscVTop::GetRunStatus() {
  return run_status_;
}

absl::StatusOr<RiscVTop::HaltReasonValueType> RiscVTop::GetLastHaltReason() {
  return halt_reason_;
}

absl::StatusOr<uint64_t> RiscVTop::ReadRegister(const std::string& name) {
  auto iter = state_->registers()->find(name);

  // Was the register found? If not try CSRs.
  if (iter == state_->registers()->end()) {
    auto result = state_->csr_set()->GetCsr(name);
    if (!result.ok()) {
      // See if it is $branch_trace_head.
      if (name == "$branch_trace_head") return branch_trace_head_;
      if (name == "$branch_trace_size") return branch_trace_size_;
      return absl::NotFoundError(
          absl::StrCat("Register '", name, "' not found"));
    }
    auto* csr = *result;
    auto xlen = state_->xlen();
    switch (xlen) {
      case RiscVXlen::RV32:
        return csr->GetUint32();
      case RiscVXlen::RV64:
        return csr->GetUint64();
      default:
        return absl::InternalError("Unknown Xlen value");
    }
  }

  auto* db = (iter->second)->data_buffer();
  uint64_t value;
  switch (db->size<uint8_t>()) {
    case 1:
      value = static_cast<uint64_t>(db->Get<uint8_t>(0));
      break;
    case 2:
      value = static_cast<uint64_t>(db->Get<uint16_t>(0));
      break;
    case 4:
      value = static_cast<uint64_t>(db->Get<uint32_t>(0));
      break;
    case 8:
      value = static_cast<uint64_t>(db->Get<uint64_t>(0));
      break;
    default:
      return absl::InternalError("Register size is not 1, 2, 4, or 8 bytes");
  }
  return value;
}

absl::Status RiscVTop::WriteRegister(const std::string& name, uint64_t value) {
  // The registers aren't protected by a mutex, so let's not write them while
  // the simulator is running.
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError("WriteRegister: Core must be halted");
  }
  auto iter = state_->registers()->find(name);
  // Was the register found? If not try CSRs.
  if (iter == state_->registers()->end()) {
    auto result = state_->csr_set()->GetCsr(name);
    if (!result.ok()) {
      if (name == "$branch_trace_size") {
        return ResizeBranchTrace(value);
      }
      return absl::NotFoundError(
          absl::StrCat("Register '", name, "' not found"));
    }
    auto* csr = *result;
    auto xlen = state_->xlen();
    switch (xlen) {
      case RiscVXlen::RV32:
        csr->Set(static_cast<uint32_t>(value));
        return absl::OkStatus();
      case RiscVXlen::RV64:
        csr->Set(value);
        return absl::OkStatus();
      default:
        return absl::InternalError("Unknown Xlen value");
    }
  }

  // If stopped at a software breakpoint and the pc is changed, change the
  // halt reason, since the next instruction won't be were we stopped.
  if ((name == "pc") && (halt_reason_ == *HaltReason::kSoftwareBreakpoint)) {
    halt_reason_ = *HaltReason::kNone;
  }

  auto* db = (iter->second)->data_buffer();
  switch (db->size<uint8_t>()) {
    case 1:
      db->Set<uint8_t>(0, static_cast<uint8_t>(value));
      break;
    case 2:
      db->Set<uint16_t>(0, static_cast<uint16_t>(value));
      break;
    case 4:
      db->Set<uint32_t>(0, static_cast<uint32_t>(value));
      break;
    case 8:
      db->Set<uint64_t>(0, static_cast<uint64_t>(value));
      break;
    default:
      return absl::InternalError("Register size is not 1, 2, 4, or 8 bytes");
  }
  return absl::OkStatus();
}

absl::StatusOr<DataBuffer*> RiscVTop::GetRegisterDataBuffer(
    const std::string& name) {
  // The registers aren't protected by a mutex, so let's not access them while
  // the simulator is running.
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError(
        "GetRegisterDataBuffer: Core must be halted");
  }
  if (name == "$branch_trace") return branch_trace_db_;
  auto iter = state_->registers()->find(name);
  if (iter == state_->registers()->end()) {
    return absl::NotFoundError(absl::StrCat("Register '", name, "' not found"));
  }
  return iter->second->data_buffer();
}

absl::StatusOr<size_t> RiscVTop::ReadMemory(uint64_t address, void* buffer,
                                            size_t length) {
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError("ReadMemory: Core must be halted");
  }
  if (address > state_->max_physical_address()) {
    return absl::InvalidArgumentError("Invalid memory address");
  }
  uint64_t length64 = static_cast<uint64_t>(length);
  length = std::min(length64, state_->max_physical_address() - address + 1);
  auto* db = db_factory_.Allocate(length);
  // Load bypassing any watch points/semihosting.
  state_->memory()->Load(address, db, nullptr, nullptr);
  std::memcpy(buffer, db->raw_ptr(), length);
  db->DecRef();
  return length;
}

absl::StatusOr<size_t> RiscVTop::WriteMemory(uint64_t address,
                                             const void* buffer,
                                             size_t length) {
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError("WriteMemory: Core must be halted");
  }
  if (address > state_->max_physical_address()) {
    return absl::InvalidArgumentError("Invalid memory address");
  }
  uint64_t length64 = static_cast<uint64_t>(length);
  length = std::min(length64, state_->max_physical_address() - address + 1);
  auto* db = db_factory_.Allocate(length);
  std::memcpy(db->raw_ptr(), buffer, length);
  // Store bypassing any watch points/semihosting.
  state_->memory()->Store(address, db);
  db->DecRef();
  return length;
}

bool RiscVTop::HasBreakpoint(uint64_t address) {
  return rv_breakpoint_manager_->HasBreakpoint(address);
}

absl::Status RiscVTop::SetSwBreakpoint(uint64_t address) {
  // Don't try if the simulator is running.
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError(
        "SetSwBreakpoint: Core must be halted");
  }
  // If there is no breakpoint manager, return an error.
  if (rv_breakpoint_manager_ == nullptr) {
    return absl::InternalError("Breakpoints are not enabled");
  }
  // Try setting the breakpoint.
  return rv_breakpoint_manager_->SetBreakpoint(address);
}

absl::Status RiscVTop::ClearSwBreakpoint(uint64_t address) {
  // Don't try if the simulator is running.
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError("ClearSwt: Core must be halted");
  }
  if (rv_breakpoint_manager_ == nullptr) {
    return absl::InternalError("Breakpoints are not enabled");
  }
  return rv_breakpoint_manager_->ClearBreakpoint(address);
}

absl::Status RiscVTop::ClearAllSwBreakpoints() {
  // Don't try if the simulator is running.
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError(
        "ClearAllSwBreakpoints: Core must be halted");
  }
  if (rv_breakpoint_manager_ == nullptr) {
    return absl::InternalError("Breakpoints are not enabled");
  }
  rv_breakpoint_manager_->ClearAllBreakpoints();
  return absl::OkStatus();
}

absl::StatusOr<int> RiscVTop::SetActionPoint(
    uint64_t address, absl::AnyInvocable<void(uint64_t, int)> action) {
  if (rv_action_point_manager_ == nullptr) {
    return absl::InternalError("Action points are not enabled");
  }
  auto res = rv_action_point_manager_->SetAction(address, std::move(action));
  if (!res.ok()) return res;
  return res.value();
}

absl::Status RiscVTop::ClearActionPoint(uint64_t address, int id) {
  if (rv_action_point_manager_ == nullptr) {
    return absl::InternalError("Action points are not enabled");
  }
  return rv_action_point_manager_->ClearAction(address, id);
}

absl::Status RiscVTop::EnableAction(uint64_t address, int id) {
  if (rv_action_point_manager_ == nullptr) {
    return absl::InternalError("Action points are not enabled");
  }
  return rv_action_point_manager_->EnableAction(address, id);
}

absl::Status RiscVTop::DisableAction(uint64_t address, int id) {
  if (rv_action_point_manager_ == nullptr) {
    return absl::InternalError("Action points are not enabled");
  }
  return rv_action_point_manager_->DisableAction(address, id);
}

// Watch points.
absl::Status RiscVTop::SetDataWatchpoint(uint64_t address, size_t length,
                                         AccessType access_type) {
  if ((access_type == AccessType::kLoad) ||
      (access_type == AccessType::kLoadStore)) {
    auto rd_memory_status = memory_watcher_->SetLoadWatchCallback(
        util::MemoryWatcher::AddressRange(address, address + length - 1),
        [this](uint64_t address, int size) {
          set_halt_string(absl::StrFormat(
              "Watchpoint triggered due to load from %08x", address));
          RequestHalt(*HaltReason::kDataWatchPoint, nullptr);
        });
    if (!rd_memory_status.ok()) return rd_memory_status;
  }
  if ((access_type == AccessType::kStore) ||
      (access_type == AccessType::kLoadStore)) {
    auto wr_memory_status = memory_watcher_->SetStoreWatchCallback(
        util::MemoryWatcher::AddressRange(address, address + length - 1),
        [this](uint64_t address, int size) {
          set_halt_string(absl::StrFormat(
              "Watchpoint triggered due to store to %08x", address));
          RequestHalt(*HaltReason::kDataWatchPoint, nullptr);
        });
    if (!wr_memory_status.ok()) {
      if (access_type == AccessType::kLoadStore) {
        // Error recovery - ignore return value.
        (void)memory_watcher_->ClearLoadWatchCallback(address);
      }
      return wr_memory_status;
    }
  }
  return absl::OkStatus();
}

absl::Status RiscVTop::ClearDataWatchpoint(uint64_t address,
                                           AccessType access_type) {
  if ((access_type == AccessType::kLoad) ||
      (access_type == AccessType::kLoadStore)) {
    auto rd_memory_status = memory_watcher_->ClearLoadWatchCallback(address);
    if (!rd_memory_status.ok()) return rd_memory_status;
  }
  if ((access_type == AccessType::kStore) ||
      (access_type == AccessType::kLoadStore)) {
    auto wr_memory_status = memory_watcher_->ClearStoreWatchCallback(address);
    if (!wr_memory_status.ok()) return wr_memory_status;
  }
  return absl::OkStatus();
}

absl::StatusOr<Instruction*> RiscVTop::GetInstruction(uint64_t address) {
  // If requesting the instruction at an action point, we need to write the
  // original instruction back to memory before getting the disassembly.
  bool inst_swap = rv_action_point_manager_->IsActionPointActive(address);
  if (inst_swap) {
    (void)rv_action_point_manager_->ap_memory_interface()
        ->WriteOriginalInstruction(address);
    // Invalidate the cache entry so that the original instruction is used.
    rv_decode_cache_->Invalidate(address);
  }
  // Get the decoded instruction.
  Instruction* inst = rv_decode_cache_->GetDecodedInstruction(address);
  inst->IncRef();
  // Swap back if required.
  if (inst_swap) {
    (void)rv_action_point_manager_->ap_memory_interface()
        ->WriteBreakpointInstruction(address);
  }
  return inst;
}

absl::StatusOr<std::string> RiscVTop::GetDisassembly(uint64_t address) {
  // Don't try if the simulator is running.
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError("GetDissasembly: Core must be halted");
  }

  auto res = GetInstruction(address);
  if (!res.ok()) return res.status();
  Instruction* inst = res.value();
  auto disasm = inst != nullptr ? inst->AsString() : "Invalid instruction";
  inst->DecRef();
  return disasm;
}

void RiscVTop::RequestHalt(HaltReasonValueType halt_reason,
                           const Instruction* inst) {
  // First set the halt_reason_, then the halt flag.
  halt_reason_ = halt_reason;
  // Action point halts are always called from the thread that is executing the
  // instructions. In this case we set paused_ to true, and not halted_, since
  // we want to keep running after the action point by resetting paused_. If we
  // use halted_ there would be a race condition between clearing halted_ and
  // an asynchronous halt request from a different thread.
  if (halt_reason == *HaltReason::kActionPoint) {
    paused_ = true;
  } else {
    halted_ = true;
  }
  // If the halt reason is either sw breakpoint or action point, set
  // need_to_step_over to true.
  if ((halt_reason_ == *HaltReason::kSoftwareBreakpoint) ||
      (halt_reason_ == *HaltReason::kActionPoint)) {
    need_to_step_over_ = true;
  }
}

void RiscVTop::RequestHalt(HaltReason halt_reason, const Instruction* inst) {
  RequestHalt(*halt_reason, inst);
}

void RiscVTop::SetPc(uint64_t value) {
  if (pc_->data_buffer()->size<uint8_t>() == 4) {
    pc_->data_buffer()->Set<uint32_t>(0, static_cast<uint32_t>(value));
  } else {
    pc_->data_buffer()->Set<uint64_t>(0, value);
  }
}

absl::Status RiscVTop::ResizeBranchTrace(size_t size) {
  if (absl::popcount(size) != 1) {
    return absl::InvalidArgumentError("Invalid size - must be a power of 2");
  }
  auto* new_db = db_factory_.Allocate<BranchTraceEntry>(size);
  auto* new_trace = reinterpret_cast<BranchTraceEntry*>(new_db->raw_ptr());
  if (new_db == nullptr) {
    return absl::InternalError("Failed to allocate new branch trace buffer");
  }
  // Copy entries from the old buffer to the new buffer, but do it so that
  // the most recent entry of the old buffer is at the end of the newly
  // allocated buffer. That way, if the new buffer is smaller, we don't have to
  // do too much special handling.
  int new_index = size - 1;
  int old_index = branch_trace_head_;
  while ((new_index >= 0) && (branch_trace_[old_index].count > 0)) {
    new_trace[new_index] = branch_trace_[old_index];
    new_index--;
    old_index--;
    if (old_index < 0) {
      old_index = branch_trace_size_ - 1;
    }
    // Stop if we get to the beginning of the old trace.
    if (old_index == branch_trace_head_) break;
  }
  while (new_index >= 0) {
    new_trace[new_index] = {0, 0, 0};
    new_index--;
  }
  branch_trace_db_->DecRef();
  branch_trace_db_ = new_db;
  branch_trace_ = new_trace;
  branch_trace_size_ = size;
  branch_trace_mask_ = branch_trace_size_ - 1;
  branch_trace_head_ = branch_trace_mask_;
  return absl::OkStatus();
}

void RiscVTop::AddToBranchTrace(uint64_t from, uint64_t to) {
  // Get the most recent entry.
  auto& entry = branch_trace_[branch_trace_head_];
  // If the branch is the same as the previous, just increment its count.
  if ((from == entry.from) && (to == entry.to)) {
    entry.count++;
    return;
  }
  branch_trace_head_ = (branch_trace_head_ + 1) & branch_trace_mask_;
  branch_trace_[branch_trace_head_] = {static_cast<uint32_t>(from),
                                       static_cast<uint32_t>(to), 1};
}

absl::Status RiscVTop::SetCsrCounter(
    absl::string_view name, generic::SimpleCounter<uint64_t>& counter) {
  absl::StatusOr<RiscVCsrInterface*> csr_result =
      state_->csr_set()->GetCsr(name);
  if (!csr_result.ok()) {
    return csr_result.status();
  }
  switch (state_->xlen()) {
    case RiscVXlen::RV32:
      dynamic_cast<RiscVCounterCsr<uint32_t, RiscVState>*>(*csr_result)
          ->set_counter(&counter);
      csr_result = state_->csr_set()->GetCsr(absl::StrCat(name, "h"));
      if (!csr_result.ok()) {
        return csr_result.status();
      }
      dynamic_cast<RiscVCounterCsrHigh<RiscVState>*>(*csr_result)
          ->set_counter(&counter);
      break;
    case RiscVXlen::RV64:
      dynamic_cast<RiscVCounterCsr<uint64_t, RiscVState>*>(*csr_result)
          ->set_counter(&counter);
      break;
    default:
      return absl::InternalError("Unknown Xlen value");
  }
  return absl::OkStatus();
}

void RiscVTop::EnableStatistics() {
  for (auto& [unused, counter_ptr] : counter_map()) {
    if (counter_ptr->GetName() == "pc") continue;
    counter_ptr->SetIsEnabled(true);
  }
}

void RiscVTop::DisableStatistics() {
  for (auto& [unused, counter_ptr] : counter_map()) {
    if (counter_ptr->GetName() == "pc") continue;
    counter_ptr->SetIsEnabled(false);
  }
}

void RiscVTop::ICacheFetch(uint64_t address) {
  icache_->Load(address, inst_db_, nullptr, nullptr);
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
