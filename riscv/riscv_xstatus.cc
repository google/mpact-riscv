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

#include "riscv/riscv_xstatus.h"

#include <cstdint>

#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_misa.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::operator*;  // NOLINT: is used below (clang error).

// Helper functions to convert between 32 and 64 bit views of mstatus.
static inline uint64_t StretchMStatus32(uint32_t value) {
  uint64_t value64 = static_cast<uint64_t>(value);
  value64 = ((value64 & 0xc000'0000) << 32) | (value64 & 0x03ff'ffff);
  return value64;
}

static inline uint32_t CompressMStatus64(uint64_t value) {
  uint32_t value32 = ((value >> 32) & 0xc000'0000) | (value & 0x3fff'ffff);
  return value32;
}

// Constructors.

RiscVMStatus::RiscVMStatus(uint32_t initial_value, ArchState* state,
                           RiscVMIsa* misa)
    : RiscVMStatus(StretchMStatus32(initial_value), state, RiscVXlen::RV32,
                   misa) {}

RiscVMStatus::RiscVMStatus(uint64_t initial_value, ArchState* state,
                           RiscVMIsa* misa)
    : RiscVMStatus(initial_value, state, RiscVXlen::RV64, misa) {}

RiscVMStatus::RiscVMStatus(uint64_t initial_value, ArchState* state,
                           RiscVXlen xlen, RiscVMIsa* misa)
    : RiscVSimpleCsr<uint64_t>("mstatus", RiscVCsrEnum::kMStatus, initial_value,
                               kReadMask, kWriteMask, state),
      misa_(misa) {
  // All bits except for uxl and sxl, since that doesn't appear in the 32 bit
  // view.
  set_mask_from_32_ = 0xffff'fff0'ffff'ffffULL;
  // Set UXL and SXL be the current xlen.
  uint64_t xlen_val = xlen == RiscVXlen::RV64 ? 0b10 : 0b01;
  // If misa doesn't show support for user or supervisor, clear corresponding
  // bits from values and read/write masks.
  if (misa_->HasSupervisorMode()) {
    xlen_val &= 0b0011;
  }
  if (misa_->HasUserMode()) {
    xlen_val &= 0b1100;
  }
  xlen_val <<= 32;
  Set(GetUint64() | xlen_val);
  read_mask_32_ = CompressMStatus64(read_mask());
  write_mask_32_ = CompressMStatus64(write_mask());
}

// Override implementations.
uint32_t RiscVMStatus::AsUint32() {
  // The value is stored as if xlen is 64, so adjust the layout.
  return GetUint32() & read_mask_32_;
}

void RiscVMStatus::Write(uint32_t value) { Set(value & write_mask_32_); }

void RiscVMStatus::SetBits(uint32_t bits) {
  uint32_t new_value = GetUint32() | (bits & write_mask_32_);
  Set(new_value);
}

void RiscVMStatus::ClearBits(uint32_t bits) {
  uint32_t new_value = GetUint32() & ~(bits & write_mask_32_);
  Set(new_value);
}

uint32_t RiscVMStatus::GetUint32() {
  // The value is stored as if xlen is 64, so adjust the layout.
  return CompressMStatus64(GetUint64());
}

void RiscVMStatus::Set(uint32_t value) {
  // Keep the sxl and uxl bits in the 64 bit value.
  uint64_t new_value = (StretchMStatus32(value) & set_mask_from_32_) |
                       (GetUint64() & ~set_mask_from_32_);
  Set(new_value);
}

void RiscVMStatus::Set(uint64_t value) {
  this->RiscVSimpleCsr<uint64_t>::Set(value);
  // If an interrupt enable might be set, check for interrupts.
  if (value & 0xf) {
    state()->CheckForInterrupt();
  }
}

void RiscVMStatus::set_mpp(uint32_t value) {
  // Don't allow values for unsupported privilege modes.
  if ((value == *PrivilegeMode::kUser) && !misa_->HasUserMode()) return;
  if ((value == *PrivilegeMode::kSupervisor) && !misa_->HasSupervisorMode())
    return;
  SetterHelper<11, 0b11>(value);
}

// RiscVSStatus methods.

RiscVSStatus::RiscVSStatus(RiscVMStatus* mstatus, RiscVState* state)
    : RiscVSimpleCsr<uint64_t>("sstatus", RiscVCsrEnum::kSStatus, kReadMask,
                               kWriteMask, state),
      mstatus_(mstatus) {
  set_mask_from_32_ = 0xffff'fff0'ffff'ffffULL;
  read_mask_32_ = CompressMStatus64(read_mask());
  write_mask_32_ = CompressMStatus64(write_mask());
}

uint32_t RiscVSStatus::AsUint32() { return CompressMStatus64(AsUint64()); }

uint64_t RiscVSStatus::AsUint64() { return mstatus_->AsUint64(); }

void RiscVSStatus::Set(uint32_t value) {
  uint64_t new_value = StretchMStatus32(value);
  Set(new_value);
}

// Only update those bytes of mstatus that can be written into sstatus.
void RiscVSStatus::Set(uint64_t value) {
  mstatus_->Set((mstatus_->GetUint64() & ~write_mask()) |
                (value & write_mask()));
}

void RiscVSStatus::Write(uint32_t value) { Set(value & write_mask_32_); }

void RiscVSStatus::SetBits(uint32_t bits) {
  uint32_t new_value = GetUint32() | (bits & write_mask_32_);
  Set(new_value);
}

void RiscVSStatus::ClearBits(uint32_t bits) {
  uint32_t new_value = GetUint32() & ~(bits & write_mask_32_);
  Set(new_value);
}

uint32_t RiscVSStatus::GetUint32() { return CompressMStatus64(GetUint64()); }

uint64_t RiscVSStatus::GetUint64() { return mstatus_->GetUint64(); }

// RiscVUStatus methods.

RiscVUStatus::RiscVUStatus(RiscVMStatus* mstatus, RiscVState* state)
    : RiscVSimpleCsr<uint64_t>("ustatus", RiscVCsrEnum::kUStatus, kReadMask,
                               kWriteMask, state),
      mstatus_(mstatus) {
  set_mask_from_32_ = 0xffff'fff0'ffff'ffffULL;
  read_mask_32_ = CompressMStatus64(read_mask());
  write_mask_32_ = CompressMStatus64(write_mask());
}

uint32_t RiscVUStatus::AsUint32() { return CompressMStatus64(AsUint64()); }

uint64_t RiscVUStatus::AsUint64() { return mstatus_->AsUint64(); }

void RiscVUStatus::Write(uint32_t value) { Set(value & write_mask_32_); }

void RiscVUStatus::SetBits(uint32_t bits) {
  uint32_t new_value = GetUint32() | (bits & write_mask_32_);
  Set(new_value);
}

void RiscVUStatus::ClearBits(uint32_t bits) {
  uint32_t new_value = GetUint32() & ~(bits & write_mask_32_);
  Set(new_value);
}

void RiscVUStatus::Set(uint32_t value) {
  uint64_t new_value = StretchMStatus32(value);
  Set(new_value);
}

// Only update those bytes of mstatus that can be written into ustatus.
void RiscVUStatus::Set(uint64_t value) {
  mstatus_->Set((mstatus_->GetUint64() & ~write_mask()) |
                (value & write_mask()));
}

uint32_t RiscVUStatus::GetUint32() { return CompressMStatus64(GetUint64()); }

uint64_t RiscVUStatus::GetUint64() { return mstatus_->GetUint64(); }

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
