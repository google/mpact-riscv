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

#include "riscv/riscv32gzb_encoding.h"

#include <cstdint>
#include <new>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "mpact/sim/generic/simple_resource.h"
#include "mpact/sim/generic/simple_resource_operand.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv32gzb_bin_decoder.h"
#include "riscv/riscv32gzb_decoder.h"
#include "riscv/riscv32gzb_enums.h"
#include "riscv/riscv_encoding_common.h"
#include "riscv/riscv_getters.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"

namespace mpact::sim::riscv::isa32gzb {

using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.

RiscV32GZBEncoding::RiscV32GZBEncoding(RiscVState *state)
    : state_(state),
      inst_word_(0),
      opcode_(OpcodeEnum::kNone),
      format_(FormatEnum::kNone) {
  resource_pool_ = new generic::SimpleResourcePool("RiscV32GZB", 128);
  resource_delay_line_ =
      state_->CreateAndAddDelayLine<generic::SimpleResourceDelayLine>(8);
  // Initialize getters.
  source_op_getters_.emplace(*SourceOpEnum::kNone, []() { return nullptr; });
  dest_op_getters_.emplace(*DestOpEnum::kNone,
                           [](int latency) { return nullptr; });
  simple_resource_getters_.emplace(*SimpleResourceEnum::kNone,
                                   []() { return nullptr; });
  complex_resource_getters_.emplace(
      *ComplexResourceEnum::kNone,
      [](int latency, int end) { return nullptr; });
  AddRiscVSourceGetters<SourceOpEnum, Extractors, RV32Register, RVFpRegister>(
      source_op_getters_, this);
  AddRiscVDestGetters<DestOpEnum, Extractors, RV32Register, RVFpRegister>(
      dest_op_getters_, this);
  AddRiscVSimpleResourceGetters<SimpleResourceEnum, Extractors>(
      simple_resource_getters_, this);
  // Verify that there are getters for each enum value.
  for (int i = *SourceOpEnum::kNone; i < *SourceOpEnum::kPastMaxValue; ++i) {
    if (source_op_getters_.find(i) == source_op_getters_.end()) {
      LOG(ERROR) << "No getter for source op enum value " << i;
    }
  }
  for (int i = *DestOpEnum::kNone; i < *DestOpEnum::kPastMaxValue; ++i) {
    if (dest_op_getters_.find(i) == dest_op_getters_.end()) {
      LOG(ERROR) << "No getter for destination op enum value " << i;
    }
  }
  for (int i = *SimpleResourceEnum::kNone;
       i < *SimpleResourceEnum::kPastMaxValue; ++i) {
    if (simple_resource_getters_.find(i) == simple_resource_getters_.end()) {
      LOG(ERROR) << "No getter for simple resource enum value " << i;
    }
  }
}

RiscV32GZBEncoding::~RiscV32GZBEncoding() { delete resource_pool_; }

void RiscV32GZBEncoding::ParseInstruction(uint32_t inst_word) {
  inst_word_ = inst_word;
  if ((inst_word_ & 0x3) == 3) {
    auto [opcode, format] = DecodeRiscV32GZBInst32WithFormat(inst_word_);
    opcode_ = opcode;
    format_ = format;
    return;
  }
  auto [opcode, format] = DecodeRiscV32GZBInst16WithFormat(
      static_cast<uint16_t>(inst_word & 0xffff));
  opcode_ = opcode;
  format_ = format;
}

ResourceOperandInterface *RiscV32GZBEncoding::GetComplexResourceOperand(
    SlotEnum, int, OpcodeEnum, ComplexResourceEnum resource, int begin,
    int end) {
  int index = static_cast<int>(resource);
  auto iter = complex_resource_getters_.find(index);
  if (iter == complex_resource_getters_.end()) {
    LOG(WARNING) << "No complex resource getter found for resource: " << index;
    return nullptr;
  }
  return (iter->second)(begin, end);
}

ResourceOperandInterface *RiscV32GZBEncoding::GetSimpleResourceOperand(
    SlotEnum, int, OpcodeEnum, SimpleResourceVector &resource_vec, int end) {
  if (resource_vec.empty()) return nullptr;
  auto *resource_set = resource_pool_->CreateResourceSet();
  for (auto resource_enum : resource_vec) {
    int index = static_cast<int>(resource_enum);
    auto iter = simple_resource_getters_.find(index);
    if (iter == simple_resource_getters_.end()) {
      LOG(WARNING) << "No getter for simple resource " << index;
      continue;
    }
    auto *resource = (iter->second)();
    auto status = resource_set->AddResource(resource);
    if (!status.ok()) {
      LOG(ERROR) << "Unable to add resource to resource set ("
                 << static_cast<int>(resource_enum) << ")";
    }
  }
  auto *op = new generic::SimpleResourceOperand(resource_set, end,
                                                resource_delay_line_);
  return op;
}

DestinationOperandInterface *RiscV32GZBEncoding::GetDestination(
    SlotEnum, int, OpcodeEnum opcode, DestOpEnum dest_op, int dest_no,
    int latency) {
  int index = static_cast<int>(dest_op);
  auto iter = dest_op_getters_.find(index);
  if (iter == dest_op_getters_.end()) {
    LOG(ERROR) << absl::StrCat("No getter for destination op enum value ",
                               index, "for instruction ",
                               kOpcodeNames[static_cast<int>(opcode)]);
    return nullptr;
  }
  return (iter->second)(latency);
}

SourceOperandInterface *RiscV32GZBEncoding::GetSource(SlotEnum, int,
                                                      OpcodeEnum opcode,
                                                      SourceOpEnum source_op,
                                                      int source_no) {
  int index = static_cast<int>(source_op);
  auto iter = source_op_getters_.find(index);
  if (iter == source_op_getters_.end()) {
    LOG(ERROR) << absl::StrCat("No getter for source op enum value ", index,
                               " for instruction ",
                               kOpcodeNames[static_cast<int>(opcode)]);
    return nullptr;
  }
  return (iter->second)();
}

}  // namespace mpact::sim::riscv::isa32gzb
