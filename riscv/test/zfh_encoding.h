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

#ifndef THIRD_PARTY_MPACT_RISCV_TEST_ZFH_ENCODING_H_
#define THIRD_PARTY_MPACT_RISCV_TEST_ZFH_ENCODING_H_

#include <cstdint>
#include <string>

#include "absl/log/log.h"
#include "mpact/sim/generic/operand_interface.h"
#include "mpact/sim/generic/simple_resource.h"
#include "mpact/sim/generic/simple_resource_operand.h"
#include "riscv/riscv_encoding_common.h"
#include "riscv/riscv_getter_helpers.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/test/riscv_getters_zfh.h"
#include "riscv/zfh32_bin_decoder.h"
#include "riscv/zfh32_decoder.h"
#include "riscv/zfh32_enums.h"
#include "riscv/zfh64_bin_decoder.h"
#include "riscv/zfh64_decoder.h"
#include "riscv/zfh64_enums.h"

namespace mpact::sim::riscv::zfh {

template <int XLen>
class ZfhEncoding;

template <int XLen>
struct ZfhTraits;

template <>
struct ZfhTraits<32> {
  using EncodingBase = ::mpact::sim::riscv::zfh32::ZFH32EncodingBase;
  using SlotEnum = ::mpact::sim::riscv::zfh32::SlotEnum;
  using OpcodeEnum = ::mpact::sim::riscv::zfh32::OpcodeEnum;
  using FormatEnum = ::mpact::sim::riscv::zfh32::FormatEnum;
  using DestOpEnum = ::mpact::sim::riscv::zfh32::DestOpEnum;
  using SourceOpEnum = ::mpact::sim::riscv::zfh32::SourceOpEnum;
  using ComplexResourceEnum = ::mpact::sim::riscv::zfh32::ComplexResourceEnum;
  using SimpleResourceEnum = ::mpact::sim::riscv::zfh32::SimpleResourceEnum;
  using PredOpEnum = ::mpact::sim::riscv::zfh32::PredOpEnum;
  using SimpleResourceVector = ::mpact::sim::riscv::zfh32::SimpleResourceVector;
  using Extractors = ::mpact::sim::riscv::zfh32::Extractors;
  using XRegister = ::mpact::sim::riscv::RV32Register;
  using SelfEncoding = ZfhEncoding<32>;
  static constexpr int kXLen = 32;
  static constexpr const char* const* kOpcodeNames =
      ::mpact::sim::riscv::zfh32::kOpcodeNames;
};

template <>
struct ZfhTraits<64> {
  using EncodingBase = ::mpact::sim::riscv::zfh64::ZFH64EncodingBase;
  using SlotEnum = ::mpact::sim::riscv::zfh64::SlotEnum;
  using OpcodeEnum = ::mpact::sim::riscv::zfh64::OpcodeEnum;
  using FormatEnum = ::mpact::sim::riscv::zfh64::FormatEnum;
  using DestOpEnum = ::mpact::sim::riscv::zfh64::DestOpEnum;
  using SourceOpEnum = ::mpact::sim::riscv::zfh64::SourceOpEnum;
  using ComplexResourceEnum = ::mpact::sim::riscv::zfh64::ComplexResourceEnum;
  using SimpleResourceEnum = ::mpact::sim::riscv::zfh64::SimpleResourceEnum;
  using PredOpEnum = ::mpact::sim::riscv::zfh64::PredOpEnum;
  using SimpleResourceVector = ::mpact::sim::riscv::zfh64::SimpleResourceVector;
  using Extractors = ::mpact::sim::riscv::zfh64::Extractors;
  using XRegister = ::mpact::sim::riscv::RV64Register;
  using SelfEncoding = ZfhEncoding<64>;
  static constexpr int kXLen = 64;
  static constexpr const char* const* kOpcodeNames =
      ::mpact::sim::riscv::zfh64::kOpcodeNames;
};

template <int XLen>
class ZfhEncoding : public ZfhTraits<XLen>::EncodingBase,
                    public RiscVEncodingCommon {
 public:
  using OpcodeEnum = typename ZfhTraits<XLen>::OpcodeEnum;
  using FormatEnum = typename ZfhTraits<XLen>::FormatEnum;
  using DestOpEnum = typename ZfhTraits<XLen>::DestOpEnum;
  using SourceOpEnum = typename ZfhTraits<XLen>::SourceOpEnum;
  using ComplexResourceEnum = typename ZfhTraits<XLen>::ComplexResourceEnum;
  using SimpleResourceEnum = typename ZfhTraits<XLen>::SimpleResourceEnum;
  using PredOpEnum = typename ZfhTraits<XLen>::PredOpEnum;
  using SlotEnum = typename ZfhTraits<XLen>::SlotEnum;
  using SimpleResourceVector = typename ZfhTraits<XLen>::SimpleResourceVector;
  using Extractors = typename ZfhTraits<XLen>::Extractors;
  using XRegister = typename ZfhTraits<XLen>::XRegister;

  explicit ZfhEncoding(RiscVState* state)
      : state_(state),
        inst_word_(0),
        opcode_(OpcodeEnum::kNone),
        format_(FormatEnum::kNone) {
    resource_delay_line_ =
        state_->CreateAndAddDelayLine<generic::SimpleResourceDelayLine>(8);
    // Initialize getters.
    source_op_getters_.insert({*SourceOpEnum::kNone, []() { return nullptr; }});
    dest_op_getters_.insert(
        {*DestOpEnum::kNone, [](int latency) { return nullptr; }});
    simple_resource_getters_.insert(
        {*SimpleResourceEnum::kNone, []() { return nullptr; }});
    complex_resource_getters_.insert(
        {*ComplexResourceEnum::kNone,
         [](int latency, int end) { return nullptr; }});

    AddRiscVZfhSourceScalarGetters<SourceOpEnum, Extractors, XRegister>(
        source_op_getters_, this);
    AddRiscVZfhSourceFloatGetters<SourceOpEnum, Extractors, RVFpRegister>(
        source_op_getters_, this);
    AddRiscVZfhDestScalarGetters<DestOpEnum, Extractors, XRegister>(
        dest_op_getters_, this);
    AddRiscVZfhDestFloatGetters<DestOpEnum, Extractors, RVFpRegister>(
        dest_op_getters_, this);
    AddRiscVZfhSimpleResourceGetters<SimpleResourceEnum, Extractors>(
        simple_resource_getters_, this);

    // Verify that there are getters for each enum value.
    for (int i = *SourceOpEnum::kNone; i < *SourceOpEnum::kPastMaxValue; ++i) {
      if (!source_op_getters_.contains(i)) {
        LOG(ERROR) << "No getter for source op enum value " << i;
      }
    }
    for (int i = *DestOpEnum::kNone; i < *DestOpEnum::kPastMaxValue; ++i) {
      if (!dest_op_getters_.contains(i)) {
        LOG(ERROR) << "No getter for destination op enum value " << i;
      }
    }
    for (int i = *SimpleResourceEnum::kNone;
         i < *SimpleResourceEnum::kPastMaxValue; ++i) {
      if (!simple_resource_getters_.contains(i)) {
        LOG(ERROR) << "No getter for simple resource enum value " << i;
      }
    }
  }

  ~ZfhEncoding() { delete resource_pool_; }

  void ParseInstruction(uint32_t inst_word) {
    inst_word_ = inst_word;
    if constexpr (XLen == 32) {
      auto [opcode, format] =
          ::mpact::sim::riscv::zfh32::DecodeRiscVZfhInst32WithFormat(
              inst_word_);
      opcode_ = opcode;
      format_ = format;
    } else {
      auto [opcode, format] =
          ::mpact::sim::riscv::zfh64::DecodeRiscVZfhInst32WithFormat(
              inst_word_);
      opcode_ = opcode;
      format_ = format;
    }
  }

  OpcodeEnum GetOpcode(SlotEnum, int) override { return opcode_; }
  FormatEnum GetFormat(SlotEnum, int) { return format_; }

  ::mpact::sim::generic::PredicateOperandInterface* GetPredicate(
      SlotEnum, int, OpcodeEnum, PredOpEnum) override {
    return nullptr;
  }

  ResourceOperandInterface* GetSimpleResourceOperand(
      SlotEnum, int, OpcodeEnum, SimpleResourceVector& resource_vec,
      int end) override {
    return nullptr;
  }

  ResourceOperandInterface* GetComplexResourceOperand(
      SlotEnum, int, OpcodeEnum, ComplexResourceEnum resource, int begin,
      int end) override {
    return nullptr;
  }

  DestinationOperandInterface* GetDestination(SlotEnum, int, OpcodeEnum opcode,
                                              DestOpEnum dest_op, int dest_no,
                                              int latency) override {
    int index = static_cast<int>(dest_op);
    auto iter = dest_op_getters_.find(index);
    if (iter == dest_op_getters_.end()) {
      LOG(ERROR) << absl::StrCat(
          "No getter for destination op enum value ", index, "for instruction ",
          ZfhTraits<XLen>::kOpcodeNames[static_cast<int>(opcode)]);
      return nullptr;
    }
    return (iter->second)(latency);
  }

  SourceOperandInterface* GetSource(SlotEnum, int, OpcodeEnum opcode,
                                    SourceOpEnum source_op,
                                    int source_no) override {
    int index = static_cast<int>(source_op);
    auto iter = source_op_getters_.find(index);
    if (iter == source_op_getters_.end()) {
      LOG(ERROR) << absl::StrCat(
          "No getter for source op enum value ", index, " for instruction ",
          ZfhTraits<XLen>::kOpcodeNames[static_cast<int>(opcode)]);
      return nullptr;
    }
    return (iter->second)();
  }

  int GetLatency(SlotEnum, int, OpcodeEnum, DestOpEnum, int) override {
    return 0;
  }

  // Methods inherited from RiscVEncodingCommon.
  RiscVState* state() const override { return state_; }
  generic::SimpleResourcePool* resource_pool() override {
    return resource_pool_;
  }
  uint32_t inst_word() const override { return inst_word_; }

  const SourceOpGetterMap& source_op_getters() { return source_op_getters_; }
  const DestOpGetterMap& dest_op_getters() { return dest_op_getters_; }
  const SimpleResourceGetterMap& simple_resource_getters() {
    return simple_resource_getters_;
  }
  const ComplexResourceGetterMap& complex_resource_getters() {
    return complex_resource_getters_;
  }

 private:
  std::string GetSimpleResourceName(SimpleResourceEnum resource_enum);

  RiscVState* state_;
  uint32_t inst_word_;
  OpcodeEnum opcode_;
  FormatEnum format_;
  SourceOpGetterMap source_op_getters_;
  DestOpGetterMap dest_op_getters_;
  SimpleResourceGetterMap simple_resource_getters_;
  ComplexResourceGetterMap complex_resource_getters_;
  generic::SimpleResourceDelayLine* resource_delay_line_ = nullptr;
  generic::SimpleResourcePool* resource_pool_ = nullptr;
};
}  // namespace mpact::sim::riscv::zfh

#endif  // THIRD_PARTY_MPACT_RISCV_TEST_ZFH_ENCODING_H_
