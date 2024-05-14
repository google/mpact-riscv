// Copyright 2024 Google LLC
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

#include "riscv/riscv_clint.h"

#include <cstdint>
#include <cstring>

#include "absl/log/log.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/ref_count.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_xip_xie.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::generic::ReferenceCount;
using EC = ::mpact::sim::riscv::ExceptionCode;

RiscVClint::RiscVClint(int period, MipExternalWriteInterface *mip_interface)
    : mip_interface_(mip_interface), period_(period) {
  // Set the initial values.
  SetValue(0);
  mip_interface_->set_mtip(mtip_);
  mip_interface->set_msip(msip_ & 0b1);
  int bit = mtime_ >= mtimecmp_;
  if (bit == mtip_) return;
  mip_interface_->set_mtip(bit);
  mtip_ = bit;
}

// Reset of the clint block.
void RiscVClint::Reset() {
  // Reset clears mtime and msip.
  WriteMTimeLow(0);
  WriteMTimeHigh(0);
  WriteMSip(0);
}

// Called by the counter whenever its value is changed.
void RiscVClint::SetValue(const uint64_t &val) {
  update_counter_++;
  if (update_counter_ >= period_) {
    update_counter_ = 0;
    mtime_++;
    int bit = mtime_ >= mtimecmp_;
    if (bit == mtip_) return;
    mip_interface_->set_mtip(bit);
    mtip_ = bit;
  }
}

// Implementation of the memory load interface for reading memory mapped
// registers.
void RiscVClint::Load(uint64_t address, DataBuffer *db, Instruction *inst,
                      ReferenceCount *context) {
  uint32_t offset = address & 0xffff;
  switch (db->size<uint8_t>()) {
    case 1:
      db->Set<uint8_t>(0, static_cast<uint8_t>(Read(offset)));
      break;
    case 2:
      db->Set<uint16_t>(0, static_cast<uint16_t>(Read(offset)));
      break;
    case 4:
      db->Set<uint32_t>(0, static_cast<uint32_t>(Read(offset)));
      break;
    case 8:
      db->Set<uint32_t>(0, static_cast<uint32_t>(Read(offset)));
      db->Set<uint32_t>(1, static_cast<uint32_t>(Read(offset + 4)));
      break;
    default:
      ::memset(db->raw_ptr(), 0, sizeof(db->size<uint8_t>()));
      break;
  }
  // Execute the instruction to process and write back the load data.
  if (nullptr != inst) {
    if (db->latency() > 0) {
      inst->IncRef();
      if (context != nullptr) context->IncRef();
      inst->state()->function_delay_line()->Add(db->latency(),
                                                [inst, context]() {
                                                  inst->Execute(context);
                                                  if (context != nullptr)
                                                    context->DecRef();
                                                  inst->DecRef();
                                                });
    } else {
      inst->Execute(context);
    }
  }
}

// No support for vector loads.
void RiscVClint::Load(DataBuffer *address_db, DataBuffer *mask_db, int el_size,
                      DataBuffer *db, Instruction *inst,
                      ReferenceCount *context) {
  LOG(FATAL) << "RiscVClint does not support vector loads";
}

// Implementation of memory store interface to support writes to memory mapped
// registers.
void RiscVClint::Store(uint64_t address, DataBuffer *db) {
  uint32_t offset = address & 0xffff;
  switch (db->size<uint8_t>()) {
    case 1:
      return Write(offset, static_cast<uint32_t>(db->Get<uint8_t>(0)));
    case 2:
      return Write(offset, static_cast<uint32_t>(db->Get<uint16_t>(0)));
    case 4:
      return Write(offset, static_cast<uint32_t>(db->Get<uint32_t>(0)));
    case 8:
      return Write(offset, static_cast<uint32_t>(db->Get<uint32_t>(0)));
      return Write(offset + 4, static_cast<uint32_t>(db->Get<uint32_t>(1)));
    default:
      return;
  }
}

// No support for vector stores.
void RiscVClint::Store(DataBuffer *address, DataBuffer *mask, int el_size,
                       DataBuffer *db) {
  LOG(FATAL) << "RiscVClint does not support vector stores";
}

uint32_t RiscVClint::Read(uint32_t offset) {
  switch (offset) {
    case 0x0000:  // msip
      return msip_;
    case 0x4000:  // mtimecmp low
      return static_cast<uint32_t>(mtimecmp_ & 0xffff'ffffULL);
    case 0x4004:  // mtimecmp high
      return static_cast<uint32_t>(mtimecmp_ >> 32);
    case 0xbff8:  // mtime low
      return ReadMTimeLow();
    case 0xbffc:  // mtime high
      return ReadMTimeHigh();
    default:
      return 0;
  }
}

void RiscVClint::Write(uint32_t offset, uint32_t value) {
  switch (offset) {
    case 0x0000:  // msip
      WriteMSip(value);
      return;
    case 0x4000:  // mtimecmp low
      WriteMTimeCmpLow(value);
      return;
    case 0x4004:  // mtimecmp high
      WriteMTimeCmpHigh(value);
      return;
    case 0xbff8:  // mtime low
      WriteMTimeLow(value);
      update_counter_ = 0;
      return;
    case 0xbffc:  // mtime high
      WriteMTimeHigh(value);
      return;
    default:
      // Ignore write.
      return;
  }
}

// Reading mtime is equivalent to reading the cycle counter, extracting the
// right half, and adding any offset produced by a previous write to the
// register.
uint32_t RiscVClint::ReadMTimeLow() {
  return static_cast<uint32_t>(mtime_ & 0xffff'ffffULL);
}

uint32_t RiscVClint::ReadMTimeHigh() {
  return static_cast<uint32_t>(mtime_ >> 32);
}

// Write to msip updates the msip bit in the mip register.
void RiscVClint::WriteMSip(uint32_t value) {
  msip_ = value & 0b1;
  mip_interface_->set_msip(msip_ & 0b1);
}

// Update the time compare registers.
void RiscVClint::WriteMTimeCmpLow(uint32_t value) {
  mtimecmp_ =
      (mtimecmp_ & 0xffff'ffff'0000'0000ULL) | static_cast<uint64_t>(value);

  int bit = mtime_ >= mtimecmp_;
  if (bit == mtip_) return;
  mip_interface_->set_mtip(bit);
  mtip_ = bit;
}

void RiscVClint::WriteMTimeCmpHigh(uint32_t value) {
  mtimecmp_ = (mtimecmp_ & 0x0000'0000'ffff'ffffULL) |
              (static_cast<uint64_t>(value) << 32);
  int bit = mtime_ >= mtimecmp_;
  if (bit == mtip_) return;
  mip_interface_->set_mtip(bit);
  mtip_ = bit;
}

// Update the mtime register.
void RiscVClint::WriteMTimeLow(uint32_t value) {
  mtime_ = (mtime_ & 0xffff'ffff'0000'0000ULL) | static_cast<uint64_t>(value);
}

void RiscVClint::WriteMTimeHigh(uint32_t value) {
  mtime_ = (mtime_ & 0xffff'ffffULL) | (static_cast<uint64_t>(value) << 32);
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
