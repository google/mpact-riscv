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

#include "riscv/riscv_xip_xie.h"

#include <cstdint>

#include "riscv/riscv_csr.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

// Machine Interrupt Pending methods.
RiscVMIp::RiscVMIp(uint32_t initial_value, ArchState *state)
    : RiscVSimpleCsr<uint32_t>("mip", RiscVCsrEnum::kMIp, initial_value,
                               kReadMask, kWriteMask, state) {}
void RiscVMIp::Set(uint32_t value) {
  this->RiscVSimpleCsr<uint32_t>::Set(value);
  if (value != 0) {
    state()->CheckForInterrupt();
  }
}

void RiscVMIp::Set(uint64_t value) { Set(static_cast<uint32_t>(value)); }

uint32_t RiscVMIp::GetUint32() {
  return this->RiscVSimpleCsr<uint32_t>::GetUint32() | ext_seip_;
}

uint64_t RiscVMIp::GetUint64() { return GetUint32(); }

// Supervisor Interrupt Pending methods.
RiscVSIp::RiscVSIp(RiscVMIp *mip, RiscVCsrInterface *mideleg, ArchState *state)
    : RiscVSimpleCsr<uint32_t>("sip", RiscVCsrEnum::kSIp, kReadMask, kWriteMask,
                               state),
      mip_(mip),
      mideleg_(mideleg) {}

void RiscVSIp::Set(uint32_t value) {
  auto wmask = write_mask() | mideleg_->GetUint32();
  mip_->Set((mip_->GetUint32() & wmask) | (value & wmask));
}

void RiscVSIp::Set(uint64_t value) { Set(static_cast<uint32_t>(value)); }

// Machine Interrupt Enable methods.
RiscVMIe::RiscVMIe(uint32_t initial_value, ArchState *state)
    : RiscVSimpleCsr<uint32_t>("mie", RiscVCsrEnum::kMIe, initial_value,
                               kReadMask, kWriteMask, state) {}
void RiscVMIe::Set(uint32_t value) {
  this->RiscVSimpleCsr<uint32_t>::Set(value);
  if (value != 0) {
    state()->CheckForInterrupt();
  }
}

uint32_t RiscVSIp::GetUint32() {
  return mip_->GetUint32() & (mideleg_->GetUint32() | read_mask());
}

uint64_t RiscVSIp::GetUint64() { return GetUint32(); }

void RiscVMIe::Set(uint64_t value) {
  this->RiscVSimpleCsr<uint32_t>::Set(value);
}

// Supervisor Interrupt Enable methods.
RiscVSIe::RiscVSIe(RiscVMIe *mie, RiscVCsrInterface *mideleg, ArchState *state)
    : RiscVSimpleCsr<uint32_t>("sie", RiscVCsrEnum::kSIe, kReadMask, kWriteMask,
                               state),
      mie_(mie),
      mideleg_(mideleg) {}

void RiscVSIe::Set(uint32_t value) {
  auto wmask = write_mask() | mideleg_->GetUint32();
  mie_->Set((mie_->GetUint32() & wmask) | (value & wmask));
}

void RiscVSIe::Set(uint64_t value) { Set(static_cast<uint32_t>(value)); }

uint32_t RiscVSIe::GetUint32() {
  return mie_->GetUint32() & (mideleg_->GetUint32() | read_mask());
}

uint64_t RiscVSIe::GetUint64() { return GetUint32(); }

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
