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

#include "riscv/riscv_renode_register_info.h"

#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_debug_info.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::operator*;  // NOLINT - used below.

RiscVRenodeRegisterInfo *RiscVRenodeRegisterInfo::instance_ = nullptr;

void RiscVRenodeRegisterInfo::InitializeRenodeRegisterInfo() {
  using DbgReg = DebugRegisterEnum;

  renode_register_info_ = {
      {*DbgReg::kPc, 32, true, false},   {*DbgReg::kX0, 32, true, true},
      {*DbgReg::kX1, 32, true, false},   {*DbgReg::kX2, 32, true, false},
      {*DbgReg::kX3, 32, true, false},   {*DbgReg::kX4, 32, true, false},
      {*DbgReg::kX5, 32, true, false},   {*DbgReg::kX6, 32, true, false},
      {*DbgReg::kX7, 32, true, false},   {*DbgReg::kX8, 32, true, false},
      {*DbgReg::kX9, 32, true, false},   {*DbgReg::kX10, 32, true, false},
      {*DbgReg::kX11, 32, true, false},  {*DbgReg::kX12, 32, true, false},
      {*DbgReg::kX13, 32, true, false},  {*DbgReg::kX14, 32, true, false},
      {*DbgReg::kX15, 32, true, false},  {*DbgReg::kX16, 32, true, false},
      {*DbgReg::kX17, 32, true, false},  {*DbgReg::kX18, 32, true, false},
      {*DbgReg::kX19, 32, true, false},  {*DbgReg::kX20, 32, true, false},
      {*DbgReg::kX21, 32, true, false},  {*DbgReg::kX22, 32, true, false},
      {*DbgReg::kX23, 32, true, false},  {*DbgReg::kX24, 32, true, false},
      {*DbgReg::kX25, 32, true, false},  {*DbgReg::kX26, 32, true, false},
      {*DbgReg::kX27, 32, true, false},  {*DbgReg::kX28, 32, true, false},
      {*DbgReg::kX29, 32, true, false},  {*DbgReg::kX30, 32, true, false},
      {*DbgReg::kX31, 32, true, false},  {*DbgReg::kF0, 32, false, false},
      {*DbgReg::kF1, 32, false, false},  {*DbgReg::kF2, 32, false, false},
      {*DbgReg::kF3, 32, false, false},  {*DbgReg::kF4, 32, false, false},
      {*DbgReg::kF5, 32, false, false},  {*DbgReg::kF6, 32, false, false},
      {*DbgReg::kF7, 32, false, false},  {*DbgReg::kF8, 32, false, false},
      {*DbgReg::kF9, 32, false, false},  {*DbgReg::kF10, 32, false, false},
      {*DbgReg::kF11, 32, false, false}, {*DbgReg::kF12, 32, false, false},
      {*DbgReg::kF13, 32, false, false}, {*DbgReg::kF14, 32, false, false},
      {*DbgReg::kF15, 32, false, false}, {*DbgReg::kF16, 32, false, false},
      {*DbgReg::kF17, 32, false, false}, {*DbgReg::kF18, 32, false, false},
      {*DbgReg::kF19, 32, false, false}, {*DbgReg::kF20, 32, false, false},
      {*DbgReg::kF21, 32, false, false}, {*DbgReg::kF22, 32, false, false},
      {*DbgReg::kF23, 32, false, false}, {*DbgReg::kF24, 32, false, false},
      {*DbgReg::kF25, 32, false, false}, {*DbgReg::kF26, 32, false, false},
      {*DbgReg::kF27, 32, false, false}, {*DbgReg::kF28, 32, false, false},
      {*DbgReg::kF29, 32, false, false}, {*DbgReg::kF30, 32, false, false},
      {*DbgReg::kF31, 32, false, false},
  };
}

RiscVRenodeRegisterInfo::RiscVRenodeRegisterInfo() {
  InitializeRenodeRegisterInfo();
}

const RiscVRenodeRegisterInfo::RenodeRegisterInfo &
RiscVRenodeRegisterInfo::GetRenodeRegisterInfo() {
  return Instance()->GetRenodeRegisterInfoPrivate();
}

RiscVRenodeRegisterInfo *RiscVRenodeRegisterInfo::Instance() {
  if (instance_ == nullptr) {
    instance_ = new RiscVRenodeRegisterInfo();
  }
  return instance_;
}

const RiscVRenodeRegisterInfo::RenodeRegisterInfo &
RiscVRenodeRegisterInfo::GetRenodeRegisterInfoPrivate() {
  return renode_register_info_;
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
