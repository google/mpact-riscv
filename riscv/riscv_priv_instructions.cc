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

#include "riscv/riscv_priv_instructions.h"

#include <type_traits>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_misa.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_xstatus.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::operator*;  // NOLINT: is used below (clang error).

namespace RV32 {

using RegisterType = RV32Register;
using UIntReg =
    typename std::make_unsigned<typename RegisterType::ValueType>::type;

void RiscVPrivURet(const Instruction *inst) {
  // TODO Fill in semantics.
}

void RiscVPrivSRet(const Instruction *inst) {
  RiscVState *state = static_cast<RiscVState *>(inst->state());
  if (state->privilege_mode() != PrivilegeMode::kSupervisor) {
    state->Trap(/*is_interrupt*/ false, /*trap_value*/ 0,
                *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  if (state->mstatus()->tsr()) {
    state->Trap(/*is_interrupt*/ false, /*trap_value*/ 0,
                *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  auto res = state->csr_set()->GetCsr(*RiscVCsrEnum::kSEpc);
  if (!res.ok()) {
    LOG(ERROR) << absl::StrCat("At PC=", absl::Hex(inst->address()),
                               " sret: cannot access sepc");
    return;
  }
  auto *sepc = *res;
  // Get db for PC.
  auto *db = inst->Destination(0)->AllocateDataBuffer();
  // Write the contents of mepc to the pc.
  db->SetSubmit<UIntReg>(0, sepc->AsUint32());
  // Need to update new privilege level.
  res = state->csr_set()->GetCsr(*RiscVCsrEnum::kMStatus);
  if (!res.ok()) {
    LOG(ERROR) << absl::StrCat("At PC=", absl::Hex(inst->address()),
                               " sret: cannot access mstatus");
    return;
  }
  auto *mstatus = static_cast<RiscVMStatus *>(*res);
  // Get misa too.
  res = state->csr_set()->GetCsr(*RiscVCsrEnum::kMIsa);
  if (!res.ok()) {
    LOG(ERROR) << absl::StrCat("At PC=", absl::Hex(inst->address()),
                               " sret: cannot access isa");
    return;
  }
  // Set mstatus:mpp to new privilege mode as per RiscV Privileged Architectures
  // Arch V20190608-Priv-MSU-Ratified page 21:
  // When executing an xRet instruction, supposing xPP holds the value y, xIE
  // is set to xPIE; the privilege mode is changed to y; xPIE is set to 1; and
  // xPP is set to U (or M if user-mode is not supported).
  auto target_mode = mstatus->spp();
  // Set mstatus:mie to the value of mstatus:mpie.
  mstatus->set_sie(mstatus->spie());
  // Set mstatus:mpie to 1.
  mstatus->set_spie(1);
  mstatus->set_spp(*PrivilegeMode::kUser & 0b1);
  state->set_privilege_mode(static_cast<PrivilegeMode>(target_mode));
  state->SignalReturnFromInterrupt();
  mstatus->Submit();
}

void RiscVPrivMRet(const Instruction *inst) {
  RiscVState *state = static_cast<RiscVState *>(inst->state());
  if (state->privilege_mode() != PrivilegeMode::kMachine) {
    state->Trap(/*is_interrupt*/ false, /*trap_value*/ 0,
                *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  auto res = state->csr_set()->GetCsr(*RiscVCsrEnum::kMEpc);
  if (!res.ok()) {
    LOG(ERROR) << absl::StrCat("At PC=", absl::Hex(inst->address()),
                               " mret: cannot access mepc");
    return;
  }
  auto *mepc = *res;
  // Get db for PC.
  auto *db = inst->Destination(0)->AllocateDataBuffer();
  // Write the contents of mepc to the pc.
  db->SetSubmit<UIntReg>(0, mepc->AsUint32());
  // Need to update new privilege level.
  res = state->csr_set()->GetCsr(*RiscVCsrEnum::kMStatus);
  if (!res.ok()) {
    LOG(ERROR) << absl::StrCat("At PC=", absl::Hex(inst->address()),
                               " mret: cannot access mstatus");
    return;
  }
  auto *mstatus = static_cast<RiscVMStatus *>(*res);
  // Get misa too.
  res = state->csr_set()->GetCsr(*RiscVCsrEnum::kMIsa);
  if (!res.ok()) {
    LOG(ERROR) << absl::StrCat("At PC=", absl::Hex(inst->address()),
                               " mret: cannot access isa");
    return;
  }
  auto *misa = static_cast<RiscVMIsa *>(*res);
  // Set mstatus:mpp to new privilege mode as per RiscV Privileged Architectures
  // Arch V20190608-Priv-MSU-Ratified page 21:
  // When executing an xRet instruction, supposing xPP holds the value y, xIE
  // is set to xPIE; the privilege mode is changed to y; xPIE is set to 1; and
  // xPP is set to U (or M if user-mode is not supported).
  auto target_mode = mstatus->mpp();
  // Set mstatus:mie to the value of mstatus:mpie.
  mstatus->set_mie(mstatus->mpie());
  // Set mstatus:mpie to 1.
  mstatus->set_mpie(1);
  if (misa->HasUserMode()) {
    mstatus->set_mpp(*PrivilegeMode::kUser);
  } else {
    mstatus->set_mpp(*PrivilegeMode::kMachine);
  }
  state->set_privilege_mode(static_cast<PrivilegeMode>(target_mode));
  state->SignalReturnFromInterrupt();
  mstatus->Submit();
}

}  // namespace RV32

namespace RV64 {

using RegisterType = RV64Register;
using UIntReg =
    typename std::make_unsigned<typename RegisterType::ValueType>::type;

void RiscVPrivURet(const Instruction *inst) {
  // TODO Fill in semantics.
}

void RiscVPrivSRet(const Instruction *inst) {
  RiscVState *state = static_cast<RiscVState *>(inst->state());
  if (state->privilege_mode() != PrivilegeMode::kSupervisor) {
    LOG(ERROR) << "sret executed when not in Supervisor mode";
    state->Trap(/*is_interrupt*/ false, /*trap_value*/ 0,
                *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  if (state->mstatus()->tsr()) {
    state->Trap(/*is_interrupt*/ false, /*trap_value*/ 0,
                *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  auto res = state->csr_set()->GetCsr(*RiscVCsrEnum::kSEpc);
  if (!res.ok()) {
    LOG(ERROR) << absl::StrCat("At PC=", absl::Hex(inst->address()),
                               " sret: cannot access sepc");
    return;
  }
  auto *sepc = *res;
  // Get db for PC.
  auto *db = inst->Destination(0)->AllocateDataBuffer();
  // Write the contents of mepc to the pc.
  db->SetSubmit<UIntReg>(0, sepc->AsUint64());
  // Need to update new privilege level.
  res = state->csr_set()->GetCsr(*RiscVCsrEnum::kMStatus);
  if (!res.ok()) {
    LOG(ERROR) << absl::StrCat("At PC=", absl::Hex(inst->address()),
                               " sret: cannot access mstatus");
    return;
  }
  auto *mstatus = static_cast<RiscVMStatus *>(*res);
  // Get misa too.
  res = state->csr_set()->GetCsr(*RiscVCsrEnum::kMIsa);
  if (!res.ok()) {
    LOG(ERROR) << absl::StrCat("At PC=", absl::Hex(inst->address()),
                               " sret: cannot access isa");
    return;
  }
  // Set mstatus:mpp to new privilege mode as per RiscV Privileged Architectures
  // Arch V20190608-Priv-MSU-Ratified page 21:
  // When executing an xRet instruction, supposing xPP holds the value y, xIE
  // is set to xPIE; the privilege mode is changed to y; xPIE is set to 1; and
  // xPP is set to U (or M if user-mode is not supported).
  auto target_mode = mstatus->spp();
  // Set mstatus:mie to the value of mstatus:mpie.
  mstatus->set_sie(mstatus->spie());
  // Set mstatus:mpie to 1.
  mstatus->set_spie(1);
  mstatus->set_spp(*PrivilegeMode::kUser & 0b1);
  state->set_privilege_mode(static_cast<PrivilegeMode>(target_mode));
  state->SignalReturnFromInterrupt();
  mstatus->Submit();
}

void RiscVPrivMRet(const Instruction *inst) {
  RiscVState *state = static_cast<RiscVState *>(inst->state());
  if (state->privilege_mode() != PrivilegeMode::kMachine) {
    state->Trap(/*is_interrupt*/ false, /*trap_value*/ 0,
                *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  auto res = state->csr_set()->GetCsr(*RiscVCsrEnum::kMEpc);
  if (!res.ok()) {
    LOG(ERROR) << absl::StrCat("At PC=", absl::Hex(inst->address()),
                               " mret: cannot access mepc");
    return;
  }
  auto *mepc = *res;
  // Get db for PC.
  auto *db = inst->Destination(0)->AllocateDataBuffer();
  // Write the contents of mepc to the pc.
  db->SetSubmit<UIntReg>(0, mepc->AsUint64());
  // Need to update new privilege level.
  res = state->csr_set()->GetCsr(*RiscVCsrEnum::kMStatus);
  if (!res.ok()) {
    LOG(ERROR) << absl::StrCat("At PC=", absl::Hex(inst->address()),
                               " mret: cannot access mstatus");
    return;
  }
  auto *mstatus = static_cast<RiscVMStatus *>(*res);
  // Get misa too.
  res = state->csr_set()->GetCsr(*RiscVCsrEnum::kMIsa);
  if (!res.ok()) {
    LOG(ERROR) << absl::StrCat("At PC=", absl::Hex(inst->address()),
                               " mret: cannot access isa");
    return;
  }
  auto *misa = static_cast<RiscVMIsa *>(*res);
  // Set mstatus:mpp to new privilege mode as per RiscV Privileged Architectures
  // Arch V20190608-Priv-MSU-Ratified page 21:
  // When executing an xRet instruction, supposing xPP holds the value y, xIE
  // is set to xPIE; the privilege mode is changed to y; xPIE is set to 1; and
  // xPP is set to U (or M if user-mode is not supported).
  auto target_mode = mstatus->mpp();
  // Set mstatus:mie to the value of mstatus:mpie.
  mstatus->set_mie(mstatus->mpie());
  // Set mstatus:mpie to 1.
  mstatus->set_mpie(1);
  if (misa->HasUserMode()) {
    mstatus->set_mpp(*PrivilegeMode::kUser);
  } else {
    mstatus->set_mpp(*PrivilegeMode::kMachine);
  }
  state->set_privilege_mode(static_cast<PrivilegeMode>(target_mode));
  state->SignalReturnFromInterrupt();
  mstatus->Submit();
}

}  // namespace RV64

void RiscVPrivWfi(const Instruction *inst) {
  // WFI is treated as a no-op, unless the user sets a callback.
  RiscVState *state = static_cast<RiscVState *>(inst->state());
  state->WFI(inst);
}

void RiscVPrivSFenceVmaZZ(const Instruction *inst) {
  auto *state = static_cast<RiscVState *>(inst->state());
  PrivilegeMode mode = state->privilege_mode();
  auto *mstatus = state->mstatus();
  if ((mode == PrivilegeMode::kUser) ||
      ((mode == PrivilegeMode::kSupervisor) && mstatus->tvm())) {
    state->Trap(/*is_interrupt*/ false, /*trap_value*/ 0,
                *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  // TODO Fill in semantics.
}

void RiscVPrivSFenceVmaZN(const Instruction *inst) {
  auto *state = static_cast<RiscVState *>(inst->state());
  PrivilegeMode mode = state->privilege_mode();
  auto *mstatus = state->mstatus();
  if ((mode == PrivilegeMode::kUser) ||
      ((mode == PrivilegeMode::kSupervisor) && mstatus->tvm())) {
    state->Trap(/*is_interrupt*/ false, /*trap_value*/ 0,
                *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  // TODO Fill in semantics.
}

void RiscVPrivSFenceVmaNZ(const Instruction *inst) {
  auto *state = static_cast<RiscVState *>(inst->state());
  PrivilegeMode mode = state->privilege_mode();
  auto *mstatus = state->mstatus();
  if ((mode == PrivilegeMode::kUser) ||
      ((mode == PrivilegeMode::kSupervisor) && mstatus->tvm())) {
    state->Trap(/*is_interrupt*/ false, /*trap_value*/ 0,
                *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  // TODO Fill in semantics.
}

void RiscVPrivSFenceVmaNN(const Instruction *inst) {
  auto *state = static_cast<RiscVState *>(inst->state());
  PrivilegeMode mode = state->privilege_mode();
  auto *mstatus = state->mstatus();
  if ((mode == PrivilegeMode::kUser) ||
      ((mode == PrivilegeMode::kSupervisor) && mstatus->tvm())) {
    state->Trap(/*is_interrupt*/ false, /*trap_value*/ 0,
                *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  // TODO Fill in semantics.
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
