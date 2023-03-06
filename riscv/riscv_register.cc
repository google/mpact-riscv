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

#include "riscv/riscv_register.h"

#include <algorithm>
#include <any>
#include <limits>
#include <string>
#include <vector>

#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

using DataBuffer = generic::DataBuffer;

RV32VectorSourceOperand::RV32VectorSourceOperand(
    absl::Span<generic::RegisterBase *> reg_span, std::string op_name)
    : op_name_(op_name) {
  if (reg_span.empty()) return;
  if (reg_span[0] == nullptr) return;

  vector_size_ = reg_span[0]->shape()[0];
  vector_byte_size_ = vector_size_ * reg_span[0]->element_size();
  group_size_ = reg_span.size();
  for (int i = 0; i < group_size_; i++) {
    // If there is a nullptr, break off the loop and truncate the potential
    // vector group.
    if (reg_span[i] == nullptr) {
      group_size_ = i;
      break;
    }
    // Verify that the vector registers in the group all have the same length.
    if (reg_span[i]->shape()[0] != vector_size_) {
      registers_[i] = nullptr;
      group_size_ = i;
      break;
    }
    registers_.push_back(reg_span[i]);
  }
}

RV32VectorSourceOperand::RV32VectorSourceOperand(
    absl::Span<generic::RegisterBase *> reg_span)
    : RV32VectorSourceOperand(reg_span, reg_span[0]->name()) {}

RV32VectorSourceOperand::RV32VectorSourceOperand(generic::RegisterBase *reg,
                                                 std::string op_name)
    : op_name_(op_name) {
  if (reg == nullptr) return;

  vector_size_ = 1;
  vector_byte_size_ = vector_size_ * reg->element_size();
  group_size_ = 1;
  registers_.push_back(reg);
}

RV32VectorSourceOperand::RV32VectorSourceOperand(generic::RegisterBase *reg)
    : RV32VectorSourceOperand(reg, reg->name()) {}

bool RV32VectorSourceOperand::AsBool(int i) {
  int offset = i % vector_byte_size_;
  int group = i / vector_byte_size_;
  return static_cast<bool>(
      registers_[group]->data_buffer()->Get<uint8_t>(offset));
}

int8_t RV32VectorSourceOperand::AsInt8(int i) {
  int offset = i % vector_byte_size_;
  int group = i / vector_byte_size_;
  return registers_[group]->data_buffer()->Get<int8_t>(offset);
}

uint8_t RV32VectorSourceOperand::AsUint8(int i) {
  int offset = i % vector_byte_size_;
  int group = i / vector_byte_size_;
  return registers_[group]->data_buffer()->Get<uint8_t>(offset);
}

int16_t RV32VectorSourceOperand::AsInt16(int i) {
  int offset = i % (vector_byte_size_ >> 1);
  int group = i / (vector_byte_size_ >> 1);
  return registers_[group]->data_buffer()->Get<int16_t>(offset);
}

uint16_t RV32VectorSourceOperand::AsUint16(int i) {
  int offset = i % (vector_byte_size_ >> 1);
  int group = i / (vector_byte_size_ >> 1);
  return registers_[group]->data_buffer()->Get<uint16_t>(offset);
}

int32_t RV32VectorSourceOperand::AsInt32(int i) {
  int offset = i % (vector_byte_size_ >> 2);
  int group = i / (vector_byte_size_ >> 2);
  return registers_[group]->data_buffer()->Get<int32_t>(offset);
}

uint32_t RV32VectorSourceOperand::AsUint32(int i) {
  int offset = i % (vector_byte_size_ >> 2);
  int group = i / (vector_byte_size_ >> 2);
  return registers_[group]->data_buffer()->Get<uint32_t>(offset);
}

int64_t RV32VectorSourceOperand::AsInt64(int i) {
  int offset = i % (vector_byte_size_ >> 3);
  int group = i / (vector_byte_size_ >> 3);
  return registers_[group]->data_buffer()->Get<int64_t>(offset);
}

uint64_t RV32VectorSourceOperand::AsUint64(int i) {
  int offset = i % (vector_byte_size_ >> 3);
  int group = i / (vector_byte_size_ >> 3);
  return registers_[group]->data_buffer()->Get<uint64_t>(offset);
}

RV32VectorTrueOperand::RV32VectorTrueOperand(RiscVState *state)
    : RV32VectorSourceOperand(
          state->GetRegister<RVVectorRegister>("TrueReg").first) {
  // Ensure the value is all ones.
  auto [reg, created] = state->GetRegister<RVVectorRegister>("TrueReg");
  if (!created) return;
  auto data = reg->data_buffer()->Get<uint64_t>();
  for (int i = 0; i < data.size(); i++) {
    data[i] = std::numeric_limits<uint64_t>::max();
  }
}

RV32VectorDestinationOperand::RV32VectorDestinationOperand(
    absl::Span<generic::RegisterBase *> reg_span, int latency,
    std::string op_name)
    : db_factory_(reg_span[0]->arch_state()->db_factory()),
      delay_line_(reg_span[0]->arch_state()->data_buffer_delay_line()),
      latency_(latency),
      op_name_(op_name) {
  if (reg_span.empty()) return;
  if (reg_span[0] == nullptr) return;

  vector_size_ = reg_span[0]->shape()[0];
  vector_byte_size_ = vector_size_ * reg_span[0]->element_size();
  group_size_ = reg_span.size();
  for (int i = 0; i < group_size_; i++) {
    // If there is a nullptr, break off the loop and truncate the potential
    // vector group.
    if (reg_span[i] == nullptr) {
      group_size_ = i;
      break;
    }
    // Verify that the vector registers in the group all have the same length.
    if (reg_span[i]->shape()[0] != vector_size_) {
      registers_[i] = nullptr;
      group_size_ = i;
      break;
    }
    registers_.push_back(reg_span[i]);
  }
}

RV32VectorDestinationOperand::RV32VectorDestinationOperand(
    absl::Span<generic::RegisterBase *> reg_span, int latency)
    : RV32VectorDestinationOperand(reg_span, latency, reg_span[0]->name()) {}

RV32VectorDestinationOperand::RV32VectorDestinationOperand(
    generic::RegisterBase *reg, int latency, std::string op_name)
    : op_name_(op_name) {
  if (reg == nullptr) return;

  vector_size_ = 1;
  vector_byte_size_ = vector_size_ * reg->element_size();
  group_size_ = 1;
  registers_.push_back(reg);
}

RV32VectorDestinationOperand::RV32VectorDestinationOperand(
    generic::RegisterBase *reg, int latency)
    : RV32VectorDestinationOperand(reg, latency, reg->name()) {}

DataBuffer *RV32VectorDestinationOperand::AllocateDataBuffer() {
  return AllocateDataBuffer(0);
}

void RV32VectorDestinationOperand::InitializeDataBuffer(DataBuffer *db) {
  InitializeDataBuffer(0, db);
}

DataBuffer *RV32VectorDestinationOperand::CopyDataBuffer() {
  return CopyDataBuffer(0);
}

int RV32VectorDestinationOperand::latency() const { return latency_; }

std::any RV32VectorDestinationOperand::GetObject() const {
  return GetObject(0);
}

std::vector<int> RV32VectorDestinationOperand::shape() const {
  return registers_[0]->shape();
}

std::string RV32VectorDestinationOperand::AsString() const { return op_name_; }

DataBuffer *RV32VectorDestinationOperand::AllocateDataBuffer(int i) {
  DataBuffer *db = db_factory_->Allocate(registers_[i]->size());
  InitializeDataBuffer(i, db);
  return db;
}

void RV32VectorDestinationOperand::InitializeDataBuffer(int i, DataBuffer *db) {
  db->set_destination(registers_[i]);
  db->set_latency(latency_);
  db->set_delay_line(delay_line_);
}

DataBuffer *RV32VectorDestinationOperand::CopyDataBuffer(int i) {
  DataBuffer *db = db_factory_->MakeCopyOf(registers_[i]->data_buffer());
  InitializeDataBuffer(i, db);
  return db;
}

std::any RV32VectorDestinationOperand::GetObject(int i) const {
  return std::any(registers_[i]);
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
