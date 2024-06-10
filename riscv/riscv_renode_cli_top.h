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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_RENODE_CLI_TOP_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_RENODE_CLI_TOP_H_

#include <cstddef>
#include <cstdint>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mpact/sim/util/renode/renode_cli_top.h"
#include "riscv/riscv_debug_interface.h"
#include "riscv/riscv_top.h"

namespace mpact {
namespace sim {
namespace riscv {

// This class extends the RenodeCLITop with a few features specific to the
// CherIoT CLI.
class RiscVRenodeCLITop : public util::renode::RenodeCLITop {
 public:
  RiscVRenodeCLITop(RiscVTop *riscv_top, bool wait_for_cli);

  absl::Status CLISetDataWatchpoint(uint64_t address, size_t length,
                                    AccessType access_type);
  absl::Status CLIClearDataWatchpoint(uint64_t address, AccessType access_type);

  absl::StatusOr<int> CLISetActionPoint(
      uint64_t address, absl::AnyInvocable<void(uint64_t, int)> action);
  absl::Status CLIClearActionPoint(uint64_t address, int id);
  absl::Status CLIEnableAction(uint64_t address, int id);
  absl::Status CLIDisableAction(uint64_t address, int id);

 private:
  RiscVTop *riscv_top_ = nullptr;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_RENODE_CLI_TOP_H_
