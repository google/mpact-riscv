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

#include "riscv/riscv_csr.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mpact/sim/generic/data_buffer.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

void RiscVCsrWriteDb::SetDataBuffer(generic::DataBuffer *db) {
  auto db_size = db->size<uint8_t>();
  if (db_size == 4) {
    csr_->Write(db->Get<uint32_t>(0));
    return;
  }
  if (db_size == 8) {
    csr_->Write(db->Get<uint64_t>(0));
    return;
  }
  LOG(ERROR) << "Attempted to write CSR with width != 32 or 64";
}

void RiscVCsrClearBitsDb::SetDataBuffer(generic::DataBuffer *db) {
  auto db_size = db->size<uint8_t>();
  if (db_size == 4) {
    csr_->ClearBits(db->Get<uint32_t>(0));
    return;
  }
  if (db_size == 8) {
    csr_->ClearBits(db->Get<uint64_t>(0));
    return;
  }
  LOG(ERROR) << "Attempted to clear CSR with width != 32 or 64";
}

void RiscVCsrSetBitsDb::SetDataBuffer(generic::DataBuffer *db) {
  auto db_size = db->size<uint8_t>();
  if (db_size == 4) {
    csr_->SetBits(db->Get<uint32_t>(0));
    return;
  }
  if (db_size == 8) {
    csr_->SetBits(db->Get<uint64_t>(0));
    return;
  }
  LOG(ERROR) << "Attempted to set CSR with width != 32 or 64";
}

absl::Status RiscVCsrSet::AddCsr(RiscVCsrInterface *csr) {
  if (csr == nullptr) {
    return absl::InvalidArgumentError("csr is nullptr");
  }
  auto index_ptr = csr_index_map_.find(csr->index());
  auto name_ptr = csr_name_map_.find(csr->name());
  if (index_ptr != csr_index_map_.end()) {
    return absl::AlreadyExistsError(absl::StrCat(
        "CSR 0x", absl::Hex(csr->index()), " already added to CSR set"));
  }
  if (name_ptr != csr_name_map_.end()) {
    return absl::AlreadyExistsError(
        absl::StrCat("'", csr->name(), "'", " already added to CSR set"));
  }

  csr_index_map_.insert(std::make_pair(csr->index(), csr));
  csr_name_map_.insert(std::make_pair(csr->name(), csr));
  return absl::OkStatus();
}

absl::StatusOr<RiscVCsrInterface *> RiscVCsrSet::GetCsr(
    absl::string_view name) {
  auto name_ptr = csr_name_map_.find(name);
  if (name_ptr == csr_name_map_.end()) {
    return absl::NotFoundError(absl::StrCat("No such CSR: '", name, "'"));
  }
  return name_ptr->second;
}

absl::StatusOr<RiscVCsrInterface *> RiscVCsrSet::GetCsr(uint64_t index) {
  auto index_ptr = csr_index_map_.find(index);
  if (index_ptr == csr_index_map_.end()) {
    return absl::NotFoundError(absl::StrCat("No such CSR index: ", index));
  }
  return index_ptr->second;
}

absl::Status RiscVCsrSet::RemoveCsr(uint64_t csr_index) {
  auto index_ptr = csr_index_map_.find(csr_index);
  if (index_ptr == csr_index_map_.end()) {
    return absl::NotFoundError(
        absl::StrCat("CSR 0x", absl::Hex(csr_index), " not found"));
  }
  auto name_ptr = csr_name_map_.find(index_ptr->second->name());
  csr_index_map_.erase(index_ptr);
  csr_name_map_.erase(name_ptr);
  return absl::OkStatus();
}

RiscVCsrSourceOperand::RiscVCsrSourceOperand(RiscVCsrInterface *csr,
                                             std::string op_name)
    : csr_(csr), op_name_(op_name) {}

RiscVCsrSourceOperand::RiscVCsrSourceOperand(RiscVCsrInterface *csr)
    : RiscVCsrSourceOperand(csr, csr->name()) {}

bool RiscVCsrSourceOperand::AsBool(int i) {
  return static_cast<bool>(csr_->AsUint32());
}
int8_t RiscVCsrSourceOperand::AsInt8(int i) {
  return static_cast<int8_t>(csr_->AsUint32());
}
uint8_t RiscVCsrSourceOperand::AsUint8(int i) {
  return static_cast<uint8_t>(csr_->AsUint32());
}
int16_t RiscVCsrSourceOperand::AsInt16(int i) {
  return static_cast<int16_t>(csr_->AsUint32());
}
uint16_t RiscVCsrSourceOperand::AsUint16(int i) {
  return static_cast<uint16_t>(csr_->AsUint32());
}
int32_t RiscVCsrSourceOperand::AsInt32(int i) {
  return static_cast<int32_t>(csr_->AsUint32());
}
uint32_t RiscVCsrSourceOperand::AsUint32(int i) {
  return static_cast<uint32_t>(csr_->AsUint32());
}
int64_t RiscVCsrSourceOperand::AsInt64(int i) {
  return static_cast<int64_t>(csr_->AsUint64());
}
uint64_t RiscVCsrSourceOperand::AsUint64(int i) {
  return static_cast<uint64_t>(csr_->AsUint64());
}

// Implementation of the destination op templated class methods.
RiscVCsrDestinationOperand::RiscVCsrDestinationOperand(
    RiscVCsrInterface *csr, generic::DataBufferDestination *db_dest,
    int latency, std::string op_name)
    : csr_(csr),
      db_dest_(db_dest),
      db_factory_(csr_->state()->db_factory()),
      latency_(latency),
      delay_line_(csr_->state()->data_buffer_delay_line()),
      op_name_(op_name) {}

RiscVCsrDestinationOperand::RiscVCsrDestinationOperand(
    RiscVCsrInterface *csr, generic::DataBufferDestination *db_dest,
    int latency)
    : RiscVCsrDestinationOperand(csr, db_dest, latency, csr->name()) {}

void RiscVCsrDestinationOperand::InitializeDataBuffer(generic::DataBuffer *db) {
  db->set_destination(db_dest_);
  db->set_latency(latency_);
  db->set_delay_line(delay_line_);
}

generic::DataBuffer *RiscVCsrDestinationOperand::CopyDataBuffer() {
  generic::DataBuffer *db = db_factory_->Allocate(csr_->size());
  if (csr_->size() == 4) {
    db->Set<uint32_t>(0, csr_->AsUint32());
  } else if (csr_->size() == 8) {
    db->Set<uint64_t>(0, csr_->AsUint64());
  }
  InitializeDataBuffer(db);
  return db;
}

generic::DataBuffer *RiscVCsrDestinationOperand::AllocateDataBuffer() {
  generic::DataBuffer *db = db_factory_->Allocate(csr_->size());
  InitializeDataBuffer(db);
  return db;
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
