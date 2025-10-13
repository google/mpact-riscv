/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_RENODE_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_RENODE_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mpact/sim/generic/core_debug_interface.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/decoder_interface.h"
#include "mpact/sim/util/memory/atomic_memory.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "mpact/sim/util/memory/memory_use_profiler.h"
#include "mpact/sim/util/memory/single_initiator_router.h"
#include "mpact/sim/util/other/instruction_profiler.h"
#include "mpact/sim/util/program_loader/elf_program_loader.h"
#include "mpact/sim/util/renode/renode_debug_interface.h"
#include "mpact/sim/util/renode/socket_cli.h"
#include "riscv/debug_command_shell.h"
#include "riscv/riscv_arm_semihost.h"
#include "riscv/riscv_cli_forwarder.h"
#include "riscv/riscv_clint.h"
#include "riscv/riscv_instrumentation_control.h"
#include "riscv/riscv_renode_cli_top.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_top.h"

// This file defines a wrapper class for the RiscVTop that adds Arm
// semihosting. In addition, the .cc class defines a global namespace
// function that is used by the renode wrapper to create a top simulator
// instance.
//
// In addition, when the configuration specifies a command line interface port,
// an object of the SocketCLI class is instantiated to provide a command line
// interface accessible over a socket. In this case the wrapper no longer
// directly calls the top simulator control class, but routes the calls through
// a combined ReNode/CLI interface that manages the priorities and access of
// ReNode and command line commands to the simulator control class.

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::DataBufferFactory;
using ::mpact::sim::riscv::RiscVArmSemihost;
using ::mpact::sim::riscv::RiscVClint;
using ::mpact::sim::util::AtomicMemory;
using ::mpact::sim::util::ElfProgramLoader;
using ::mpact::sim::util::FlatDemandMemory;
using ::mpact::sim::util::InstructionProfiler;
using ::mpact::sim::util::MemoryInterface;
using ::mpact::sim::util::MemoryUseProfiler;
using ::mpact::sim::util::SingleInitiatorRouter;
using ::mpact::sim::util::renode::SocketCLI;

class RiscVRenode : public util::renode::RenodeDebugInterface {
 public:
  // Supported IRQ request types.
  enum class IrqType {
    kMachineSoftwareInterrupt = 0x3,
    kMachineExternalInterrupt = 0xb,
  };

  enum RenodeState {
    kIdle = 0,
    kStepping = 1,
    kRunning = 2,
  };

  enum CLIState {
    kDisconnected = 0,
    kConnected = 1,
  };

  using ::mpact::sim::generic::CoreDebugInterface::HaltReason;
  using ::mpact::sim::generic::CoreDebugInterface::RunStatus;
  using RenodeCpuRegister = ::mpact::sim::util::renode::RenodeCpuRegister;

  // Constructor takes a name and a memory interface that is used for memory
  // transactions routed to the system bus.
  RiscVRenode(std::string name, MemoryInterface* renode_sysbus, RiscVXlen xlen);
  ~RiscVRenode() override;

  absl::StatusOr<uint64_t> LoadExecutable(const char* elf_file_name,
                                          bool for_symbols_only) override;
  // Step the core by num instructions.
  absl::StatusOr<int> Step(int num) override;
  // Returns the reason for the most recent halt.
  absl::StatusOr<HaltReasonValueType> GetLastHaltReason() override;
  // Read/write the numeric id registers.
  absl::StatusOr<uint64_t> ReadRegister(uint32_t reg_id) override;
  absl::Status WriteRegister(uint32_t reg_id, uint64_t value) override;
  // Get register data buffer call. Not implemented, stubbed out to return null.
  // Read/write the buffers to memory.
  absl::StatusOr<size_t> ReadMemory(uint64_t address, void* buf,
                                    size_t length) override;
  absl::StatusOr<size_t> WriteMemory(uint64_t address, const void* buf,
                                     size_t length) override;
  // Return register information.
  int32_t GetRenodeRegisterInfoSize() const override;
  absl::Status GetRenodeRegisterInfo(int32_t index, int32_t max_len, char* name,
                                     RenodeCpuRegister& info) override;

  // Set configuration value.
  absl::Status SetConfig(const char* config_names[],
                         const char* config_values[], int size) override;

  // Set IRQ value for supported IRQs. Supported irq_nums are:
  //          MachineSoftwareInterrupt = 0x3
  //          handled by the clint for now: MachineTimerInterrupt = 0x7
  //          MachineExternalInterrupt = 0xb
  // These correspond to the msip and meip bits of the mip register.
  absl::Status SetIrqValue(int32_t irq_num, bool irq_value) override;

 private:
  std::string name_;
  [[maybe_unused]] MemoryInterface* renode_sysbus_ = nullptr;
  RiscVState* rv_state_ = nullptr;
  RiscVFPState* rv_fp_state_ = nullptr;
  generic::DecoderInterface* rv_decoder_ = nullptr;
  RiscVTop* riscv_top_ = nullptr;
  RiscVArmSemihost* semihost_ = nullptr;
  SingleInitiatorRouter* router_ = nullptr;
  SingleInitiatorRouter* renode_router_ = nullptr;
  DataBufferFactory db_factory_;
  AtomicMemory* atomic_memory_ = nullptr;
  FlatDemandMemory* memory_ = nullptr;
  RiscVClint* clint_ = nullptr;
  SocketCLI* socket_cli_ = nullptr;
  RiscVRenodeCLITop* riscv_renode_cli_top_ = nullptr;
  RiscVCLIForwarder* riscv_cli_forwarder_ = nullptr;
  ElfProgramLoader* program_loader_ = nullptr;
  DebugCommandShell* cmd_shell_ = nullptr;
  InstructionProfiler* inst_profiler_ = nullptr;
  MemoryUseProfiler* mem_profiler_ = nullptr;
  RiscVInstrumentationControl* instrumentation_control_ = nullptr;
  [[maybe_unused]] uint64_t stack_size_ = 32 * 1024;
  [[maybe_unused]] uint64_t stack_end_ = 0;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_RENODE_H_
