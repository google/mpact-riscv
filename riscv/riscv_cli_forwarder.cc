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

#include "riscv/riscv_cli_forwarder.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mpact/sim/generic/core_debug_interface.h"
#include "riscv/riscv_renode_cli_top.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::AccessType;
using RunStatus = ::mpact::sim::generic::CoreDebugInterface::RunStatus;
using HaltReasonValueType =
    ::mpact::sim::generic::CoreDebugInterface::HaltReasonValueType;

RiscVCLIForwarder::RiscVCLIForwarder(RiscVRenodeCLITop *riscv_cli_top)
    : riscv_cli_top_(riscv_cli_top) {}

// Forward the calls to the RiscVRenodeCLITop class - CLI methods.

absl::Status RiscVCLIForwarder::SetDataWatchpoint(uint64_t address,
                                                  size_t length,
                                                  AccessType access_type) {
  return riscv_cli_top_->CLISetDataWatchpoint(address, length, access_type);
}

absl::Status RiscVCLIForwarder::ClearDataWatchpoint(uint64_t address,
                                                    AccessType access_type) {
  return riscv_cli_top_->CLIClearDataWatchpoint(address, access_type);
}

absl::StatusOr<int> RiscVCLIForwarder::SetActionPoint(
    uint64_t address, absl::AnyInvocable<void(uint64_t, int)> action) {
  return riscv_cli_top_->CLISetActionPoint(address, std::move(action));
}

absl::Status RiscVCLIForwarder::ClearActionPoint(uint64_t address, int id) {
  return riscv_cli_top_->CLIClearActionPoint(address, id);
}

absl::Status RiscVCLIForwarder::EnableAction(uint64_t address, int id) {
  return riscv_cli_top_->CLIEnableAction(address, id);
}

absl::Status RiscVCLIForwarder::DisableAction(uint64_t address, int id) {
  return riscv_cli_top_->CLIDisableAction(address, id);
}

absl::Status RiscVCLIForwarder::Halt() { return riscv_cli_top_->CLIHalt(); }

absl::Status RiscVCLIForwarder::Halt(HaltReason halt_reason) {
  riscv_cli_top_->CLIRequestHalt(halt_reason, nullptr);
  return absl::OkStatus();
}

absl::Status RiscVCLIForwarder::Halt(HaltReasonValueType halt_reason) {
  riscv_cli_top_->CLIRequestHalt(halt_reason, nullptr);
  return absl::OkStatus();
}

absl::StatusOr<int> RiscVCLIForwarder::Step(int num) {
  return riscv_cli_top_->CLIStep(num);
}

absl::Status RiscVCLIForwarder::Run() { return riscv_cli_top_->CLIRun(); }

absl::Status RiscVCLIForwarder::Wait() { return riscv_cli_top_->CLIWait(); }

// Returns the current run status.
absl::StatusOr<RunStatus> RiscVCLIForwarder::GetRunStatus() {
  return riscv_cli_top_->CLIGetRunStatus();
}

// Returns the reason for the most recent halt.
absl::StatusOr<HaltReasonValueType> RiscVCLIForwarder::GetLastHaltReason() {
  return riscv_cli_top_->CLIGetLastHaltReason();
}

// Read/write the named registers.
absl::StatusOr<uint64_t> RiscVCLIForwarder::ReadRegister(
    const std::string &name) {
  return riscv_cli_top_->CLIReadRegister(name);
}

absl::Status RiscVCLIForwarder::WriteRegister(const std::string &name,
                                              uint64_t value) {
  return riscv_cli_top_->CLIWriteRegister(name, value);
}

absl::StatusOr<DataBuffer *> RiscVCLIForwarder::GetRegisterDataBuffer(
    const std::string &name) {
  return riscv_cli_top_->CLIGetRegisterDataBuffer(name);
}

// Read/write the buffers to memory.
absl::StatusOr<size_t> RiscVCLIForwarder::ReadMemory(uint64_t address,
                                                     void *buf, size_t length) {
  return riscv_cli_top_->CLIReadMemory(address, buf, length);
}

absl::StatusOr<size_t> RiscVCLIForwarder::WriteMemory(uint64_t address,
                                                      const void *buf,
                                                      size_t length) {
  return riscv_cli_top_->CLIWriteMemory(address, buf, length);
}

bool RiscVCLIForwarder::HasBreakpoint(uint64_t address) {
  return riscv_cli_top_->CLIHasBreakpoint(address);
}

absl::Status RiscVCLIForwarder::SetSwBreakpoint(uint64_t address) {
  return riscv_cli_top_->CLISetSwBreakpoint(address);
}

absl::Status RiscVCLIForwarder::ClearSwBreakpoint(uint64_t address) {
  return riscv_cli_top_->CLIClearSwBreakpoint(address);
}

absl::Status RiscVCLIForwarder::ClearAllSwBreakpoints() {
  return riscv_cli_top_->CLIClearAllSwBreakpoints();
}

absl::StatusOr<Instruction *> RiscVCLIForwarder::GetInstruction(
    uint64_t address) {
  return riscv_cli_top_->CLIGetInstruction(address);
}

absl::StatusOr<std::string> RiscVCLIForwarder::GetDisassembly(
    uint64_t address) {
  return riscv_cli_top_->CLIGetDisassembly(address);
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
