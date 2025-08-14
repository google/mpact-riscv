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

#include "riscv/zvbb_encoding.h"

#include <cstdint>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "mpact/sim/generic/simple_resource_operand.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_getter_helpers.h"
#include "riscv/riscv_getters_zvbb.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/zvbb_bin_decoder.h"
#include "riscv/zvbb_decoder.h"
#include "riscv/zvbb_enums.h"

namespace mpact::sim::riscv::zvbb {

using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.

ZVBBEncoding::ZVBBEncoding(RiscVState* state)
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

  AddRiscVZvbbSourceVectorGetters<SourceOpEnum, Extractors, RVVectorRegister>(
      source_op_getters_, this);
  AddRiscvZvbbSourceScalarGetters<SourceOpEnum, Extractors, RV32Register>(
      source_op_getters_, this);

  AddRiscVZvbbDestGetters<DestOpEnum, Extractors, RVVectorRegister>(
      dest_op_getters_, this);

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

ZVBBEncoding::~ZVBBEncoding() { delete resource_pool_; }

void ZVBBEncoding::ParseInstruction(uint32_t inst_word) {
  inst_word_ = inst_word;
  auto [opcode, format] = DecodeRiscVZvbbInst32WithFormat(inst_word_);
  opcode_ = opcode;
  format_ = format;
}

ResourceOperandInterface* ZVBBEncoding::GetComplexResourceOperand(
    SlotEnum, int, OpcodeEnum, ComplexResourceEnum resource, int begin,
    int end) {
  return nullptr;
}

ResourceOperandInterface* ZVBBEncoding::GetSimpleResourceOperand(
    SlotEnum, int, OpcodeEnum, SimpleResourceVector& resource_vec, int end) {
  return nullptr;
}

DestinationOperandInterface* ZVBBEncoding::GetDestination(SlotEnum, int,
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

SourceOperandInterface* ZVBBEncoding::GetSource(SlotEnum, int,
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

}  // namespace mpact::sim::riscv::zvbb
