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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_RENODE_REGISTER_INFO_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_RENODE_REGISTER_INFO_H_

#include <vector>

#include "mpact/sim/util/renode/renode_debug_interface.h"

namespace mpact {
namespace sim {
namespace riscv {

// This file defines a class that is used to store the register information
// that needs to be provided to renode for the RiscV registers.

class RiscVRenodeRegisterInfo {
 public:
  using RenodeRegisterInfo = std::vector<util::renode::RenodeCpuRegister>;

  static const RenodeRegisterInfo& GetRenodeRegisterInfo();

 private:
  RiscVRenodeRegisterInfo();
  static RiscVRenodeRegisterInfo* Instance();
  void InitializeRenodeRegisterInfo();
  const RenodeRegisterInfo& GetRenodeRegisterInfoPrivate();

  static RiscVRenodeRegisterInfo* instance_;
  RenodeRegisterInfo renode_register_info_;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_RENODE_REGISTER_INFO_H_
