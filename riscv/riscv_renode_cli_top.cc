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

#include "riscv/riscv_renode_cli_top.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mpact/sim/util/renode/renode_cli_top.h"
#include "riscv/riscv_debug_interface.h"
#include "riscv/riscv_top.h"

namespace mpact {
namespace sim {
namespace riscv {

RiscVRenodeCLITop::RiscVRenodeCLITop(RiscVTop *riscv_top, bool wait_for_cli)
    : util::renode::RenodeCLITop(riscv_top, wait_for_cli),
      riscv_top_(riscv_top) {}

absl::Status RiscVRenodeCLITop::CLISetDataWatchpoint(uint64_t address,
                                                     size_t length,
                                                     AccessType access_type) {
  return DoWhenInControl<absl::Status>([this, address, length, access_type]() {
    return riscv_top_->SetDataWatchpoint(address, length, access_type);
  });
}

absl::Status RiscVRenodeCLITop::CLIClearDataWatchpoint(uint64_t address,
                                                       AccessType access_type) {
  return DoWhenInControl<absl::Status>([this, address, access_type]() {
    return riscv_top_->ClearDataWatchpoint(address, access_type);
  });
}

absl::StatusOr<int> RiscVRenodeCLITop::CLISetActionPoint(
    uint64_t address, absl::AnyInvocable<void(uint64_t, int)> action) {
  return DoWhenInControl<absl::StatusOr<int>>([this, address, &action]() {
    return riscv_top_->SetActionPoint(address, std::move(action));
  });
}

absl::Status RiscVRenodeCLITop::CLIClearActionPoint(uint64_t address, int id) {
  return DoWhenInControl<absl::Status>([this, address, id]() {
    return riscv_top_->ClearActionPoint(address, id);
  });
}

absl::Status RiscVRenodeCLITop::CLIEnableAction(uint64_t address, int id) {
  return DoWhenInControl<absl::Status>(
      [this, address, id]() { return riscv_top_->EnableAction(address, id); });
}

absl::Status RiscVRenodeCLITop::CLIDisableAction(uint64_t address, int id) {
  return DoWhenInControl<absl::Status>(
      [this, address, id]() { return riscv_top_->DisableAction(address, id); });
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
