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

#include "riscv/riscv64g_vec_encoding.h"

#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "riscv/riscv64gv_enums.h"
#include "riscv/riscv_state.h"

// This file contains tests for the RiscV64GZBEncoding class to ensure that
// the instruction decoding is correct.

namespace {

using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.

using mpact::sim::riscv::RiscVState;
using mpact::sim::riscv::RiscVXlen;
using mpact::sim::riscv::isa64v::ComplexResourceEnum;
using mpact::sim::riscv::isa64v::DestOpEnum;
using mpact::sim::riscv::isa64v::kComplexResourceNames;
using mpact::sim::riscv::isa64v::kDestOpNames;
using mpact::sim::riscv::isa64v::kSimpleResourceNames;
using mpact::sim::riscv::isa64v::kSourceOpNames;
using mpact::sim::riscv::isa64v::RiscV64GVecEncoding;
using mpact::sim::riscv::isa64v::SimpleResourceEnum;
using mpact::sim::riscv::isa64v::SourceOpEnum;
using mpact::sim::util::FlatDemandMemory;
using SlotEnum = mpact::sim::riscv::isa64v::SlotEnum;
using OpcodeEnum = mpact::sim::riscv::isa64v::OpcodeEnum;

class RiscV64GVecEncodingTest : public testing::Test {
 protected:
  RiscV64GVecEncodingTest() {
    state_ = new RiscVState("test", RiscVXlen::RV64, &memory_);
    enc_ = new RiscV64GVecEncoding(state_);
  }
  ~RiscV64GVecEncodingTest() override {
    delete enc_;
    delete state_;
  }

  FlatDemandMemory memory_;
  RiscVState* state_;
  RiscV64GVecEncoding* enc_;
};

TEST_F(RiscV64GVecEncodingTest, SourceOperands) {
  auto& getters = enc_->source_op_getters();
  for (int i = *SourceOpEnum::kNone; i < *SourceOpEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No source operand for enum value " << i
                                     << " (" << kSourceOpNames[i] << ")";
  }
}

TEST_F(RiscV64GVecEncodingTest, DestOperands) {
  auto& getters = enc_->dest_op_getters();
  for (int i = *DestOpEnum::kNone; i < *DestOpEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No dest operand for enum value " << i
                                     << " (" << kDestOpNames[i] << ")";
  }
}

TEST_F(RiscV64GVecEncodingTest, SimpleResources) {
  auto& getters = enc_->simple_resource_getters();
  for (int i = *SimpleResourceEnum::kNone;
       i < *SimpleResourceEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No source operand for enum value " << i
                                     << " (" << kSimpleResourceNames[i] << ")";
  }
}

TEST_F(RiscV64GVecEncodingTest, ComplexResources) {
  auto& getters = enc_->source_op_getters();
  for (int i = *ComplexResourceEnum::kNone;
       i < *ComplexResourceEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No source operand for enum value " << i
                                     << " (" << kComplexResourceNames[i] << ")";
  }
}

}  // namespace
