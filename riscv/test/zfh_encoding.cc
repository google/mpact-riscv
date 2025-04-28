// Copyright 2025 Google LLC
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

#include "riscv/test/zfh_encoding.h"

#include <cstdint>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "mpact/sim/generic/simple_resource_operand.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_getter_helpers.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/test/riscv_getters_zfh.h"
#include "riscv/zfh_bin_decoder.h"
#include "riscv/zfh_decoder.h"
#include "riscv/zfh_enums.h"

namespace mpact::sim::riscv::zfh {

using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.

ZFHEncoding::ZFHEncoding(RiscVState *state)
    : state_(state),
      inst_word_(0),
      opcode_(OpcodeEnum::kNone),
      format_(FormatEnum::kNone) {
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

  AddRiscVZfhSourceScalarGetters<SourceOpEnum, Extractors, RVXRegister>(
      source_op_getters_, this);
  AddRiscVZfhSourceFloatGetters<SourceOpEnum, Extractors, RVFpRegister>(
      source_op_getters_, this);
  AddRiscVZfhDestScalarGetters<DestOpEnum, Extractors, RVXRegister>(
      dest_op_getters_, this);
  AddRiscVZfhDestFloatGetters<DestOpEnum, Extractors, RVFpRegister>(
      dest_op_getters_, this);
  AddRiscVZfhSimpleResourceGetters<SimpleResourceEnum, Extractors>(
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

ZFHEncoding::~ZFHEncoding() { delete resource_pool_; }

void ZFHEncoding::ParseInstruction(uint32_t inst_word) {
  inst_word_ = inst_word;
  auto [opcode, format] = DecodeRiscVZfhInst32WithFormat(inst_word_);
  opcode_ = opcode;
  format_ = format;
}

ResourceOperandInterface *ZFHEncoding::GetComplexResourceOperand(
    SlotEnum, int, OpcodeEnum, ComplexResourceEnum resource, int begin,
    int end) {
  return nullptr;
}

ResourceOperandInterface *ZFHEncoding::GetSimpleResourceOperand(
    SlotEnum, int, OpcodeEnum, SimpleResourceVector &resource_vec, int end) {
  return nullptr;
}

DestinationOperandInterface *ZFHEncoding::GetDestination(SlotEnum, int,
                                                         OpcodeEnum opcode,
                                                         DestOpEnum dest_op,
                                                         int dest_no,
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

SourceOperandInterface *ZFHEncoding::GetSource(SlotEnum, int, OpcodeEnum opcode,
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

}  // namespace mpact::sim::riscv::zfh
