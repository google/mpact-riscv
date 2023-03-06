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

#include "riscv/riscv_sim_csrs.h"

#include "riscv/riscv_state.h"
namespace mpact {
namespace sim {
namespace riscv {

uint32_t RiscVSimModeCsr::GetUint32() {
  return static_cast<uint32_t>(state_->privilege_mode());
}

uint64_t RiscVSimModeCsr::GetUint64() {
  return static_cast<uint64_t>(state_->privilege_mode());
}

void RiscVSimModeCsr::Set(uint32_t value) {
  state_->set_privilege_mode(static_cast<PrivilegeMode>(value));
}

void RiscVSimModeCsr::Set(uint64_t value) {
  state_->set_privilege_mode(static_cast<PrivilegeMode>(value));
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
