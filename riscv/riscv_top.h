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

#ifndef MPACT_RISCV_RISCV_RISCV_TOP_H_
#define MPACT_RISCV_RISCV_RISCV_TOP_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "mpact/sim/generic/component.h"
#include "mpact/sim/generic/core_debug_interface.h"
#include "mpact/sim/generic/counters.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/decode_cache.h"
#include "mpact/sim/generic/decoder_interface.h"
#include "mpact/sim/generic/register.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "mpact/sim/util/memory/memory_watcher.h"
#include "riscv/riscv32g_enums.h"
#include "riscv/riscv64g_enums.h"
#include "riscv/riscv_action_point.h"
#include "riscv/riscv_breakpoint.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::operator*;  // NOLINT: is used below (clang error).

// Top level class for the RiscV32G simulator. This is the main interface for
// interacting and controlling execution of programs running on the simulator.
// This class brings together the decoder, the architecture state, and control.
class RiscVTop : public generic::Component, public generic::CoreDebugInterface {
 public:
  using RunStatus = generic::CoreDebugInterface::RunStatus;
  using HaltReason = generic::CoreDebugInterface::HaltReason;

  // Simple constructor, the memories are created and owned by the RiscVTop
  // instance.
  RiscVTop(std::string name, RiscVXlen xlen);
  // Constructors without the atomic memory interface. Either one (unified),
  // or two (inst and data) interfaces are passed in.
  RiscVTop(std::string name, util::MemoryInterface *memory, RiscVXlen xlen);
  RiscVTop(std::string name, util::MemoryInterface *inst_memory,
           util::MemoryInterface *data_memory, RiscVXlen xlen);
  // Constructors with the atomic memory interface. Either one (unified),
  // or two (inst and data) interfaces are passed in.
  RiscVTop(std::string name, util::MemoryInterface *memory, RiscVXlen xlen,
           util::AtomicMemoryOpInterface *atomic_memory);
  RiscVTop(std::string name, util::MemoryInterface *inst_memory,
           util::MemoryInterface *data_memory, RiscVXlen xlen,
           util::AtomicMemoryOpInterface *atomic_memory);
  ~RiscVTop() override;

  // Methods inherited from CoreDebugInterface.
  absl::Status Halt() override;
  absl::StatusOr<int> Step(int num) override;
  absl::Status Run() override;
  absl::Status Wait() override;

  absl::StatusOr<RunStatus> GetRunStatus() override;
  absl::StatusOr<HaltReasonValueType> GetLastHaltReason() override;

  // Register access by register name.
  absl::StatusOr<uint64_t> ReadRegister(const std::string &name) override;
  absl::Status WriteRegister(const std::string &name, uint64_t value) override;
  absl::StatusOr<generic::DataBuffer *> GetRegisterDataBuffer(
      const std::string &name) override;
  // Read and Write memory methods bypass any semihosting.
  absl::StatusOr<size_t> ReadMemory(uint64_t address, void *buf,
                                    size_t length) override;
  absl::StatusOr<size_t> WriteMemory(uint64_t address, const void *buf,
                                     size_t length) override;

  bool HasBreakpoint(uint64_t address) override;
  absl::Status SetSwBreakpoint(uint64_t address) override;
  absl::Status ClearSwBreakpoint(uint64_t address) override;
  absl::Status ClearAllSwBreakpoints() override;

  absl::StatusOr<Instruction *> GetInstruction(uint64_t address) override;
  absl::StatusOr<std::string> GetDisassembly(uint64_t address) override;

  // Called when a halt is requested.
  void RequestHalt(HaltReason halt_reason, const Instruction *inst);
  void RequestHalt(HaltReasonValueType halt_reason, const Instruction *inst);

  // Accessors.
  RiscVState *state() const { return state_; }
  util::MemoryInterface *data_memory() const { return data_memory_; }
  util::MemoryInterface *inst_memory() const { return inst_memory_; }

 private:
  // Initialize the top.
  void Initialize();
  // Helper method to step past a breakpoint.
  absl::Status StepPastBreakpoint();
  // Set the pc value.
  void SetPc(uint64_t value);

  // The DB factory is used to manage data buffers for memory read/writes.
  generic::DataBufferFactory db_factory_;
  // Current status and last halt reasons.
  RunStatus run_status_ = RunStatus::kHalted;
  HaltReasonValueType halt_reason_ = *HaltReason::kNone;
  // Halting flag. This is set to true when execution must halt.
  bool halted_ = false;
  // Set to true if the next instruction requires a step-over.
  bool need_to_step_over_ = false;
  absl::Notification *run_halted_ = nullptr;
  // The local RiscV32 state.
  RiscVState *state_;
  RiscVFPState *fp_state_;
  // Action point manager.
  RiscVActionPointManager *rv_action_point_manager_ = nullptr;
  // Breakpoint manager.
  RiscVBreakpointManager *rv_breakpoint_manager_ = nullptr;
  // The pc register instance.
  generic::RegisterBase *pc_;
  // RiscV32 decoder instance.
  generic::DecoderInterface *rv_decoder_ = nullptr;
  // Decode cache, memory and memory watcher.
  generic::DecodeCache *rv_decode_cache_ = nullptr;
  util::MemoryInterface *inst_memory_ = nullptr;
  util::MemoryInterface *data_memory_ = nullptr;
  util::AtomicMemoryOpInterface *atomic_memory_ = nullptr;
  bool owns_memory_ = false;
  util::MemoryWatcher *watcher_ = nullptr;
  // Counter for the number of instructions simulated.
  generic::SimpleCounter<uint64_t> counter_opcode_[std::max(
      static_cast<int>(isa32::OpcodeEnum::kPastMaxValue),
      static_cast<int>(isa64::OpcodeEnum::kPastMaxValue))];
  generic::SimpleCounter<uint64_t> counter_num_instructions_;
  generic::SimpleCounter<uint64_t> counter_num_cycles_;
  absl::flat_hash_map<uint32_t, std::string> register_id_map_;
  RiscVXlen xlen_ = RiscVXlen::RV32;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_TOP_H_
