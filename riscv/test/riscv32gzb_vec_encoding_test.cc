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

#include "riscv/riscv32gzb_vec_encoding.h"

#include <cstdint>

#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "riscv/riscv32gvzb_enums.h"
#include "riscv/riscv_state.h"

// This file contains tests for the RiscV32GZBVecEncoding class to ensure that
// the instruction decoding is correct.

namespace {

using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.

using mpact::sim::riscv::RiscVState;
using mpact::sim::riscv::RiscVXlen;
using mpact::sim::riscv::isa32gvzb::ComplexResourceEnum;
using mpact::sim::riscv::isa32gvzb::DestOpEnum;
using mpact::sim::riscv::isa32gvzb::kComplexResourceNames;
using mpact::sim::riscv::isa32gvzb::kDestOpNames;
using mpact::sim::riscv::isa32gvzb::kSimpleResourceNames;
using mpact::sim::riscv::isa32gvzb::kSourceOpNames;
using mpact::sim::riscv::isa32gvzb::RiscV32GZBVecEncoding;
using mpact::sim::riscv::isa32gvzb::SimpleResourceEnum;
using mpact::sim::riscv::isa32gvzb::SourceOpEnum;
using mpact::sim::util::FlatDemandMemory;
using SlotEnum = mpact::sim::riscv::isa32gvzb::SlotEnum;
using OpcodeEnum = mpact::sim::riscv::isa32gvzb::OpcodeEnum;

class RiscV32GZBVecEncodingTest : public testing::Test {
 protected:
  RiscV32GZBVecEncodingTest() {
    state_ = new RiscVState("test", RiscVXlen::RV32, &memory_);
    enc_ = new RiscV32GZBVecEncoding(state_);
  }
  ~RiscV32GZBVecEncodingTest() override {
    delete enc_;
    delete state_;
  }

  FlatDemandMemory memory_;
  RiscVState* state_;
  RiscV32GZBVecEncoding* enc_;
};

TEST_F(RiscV32GZBVecEncodingTest, SourceOperands) {
  auto& getters = enc_->source_op_getters();
  for (int i = *SourceOpEnum::kNone; i < *SourceOpEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No source operand for enum value " << i
                                     << " (" << kSourceOpNames[i] << ")";
  }
}

TEST_F(RiscV32GZBVecEncodingTest, DestOperands) {
  auto& getters = enc_->dest_op_getters();
  for (int i = *DestOpEnum::kNone; i < *DestOpEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No dest operand for enum value " << i
                                     << " (" << kDestOpNames[i] << ")";
  }
}

TEST_F(RiscV32GZBVecEncodingTest, SimpleResources) {
  auto& getters = enc_->simple_resource_getters();
  for (int i = *SimpleResourceEnum::kNone;
       i < *SimpleResourceEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No source operand for enum value " << i
                                     << " (" << kSimpleResourceNames[i] << ")";
  }
}

TEST_F(RiscV32GZBVecEncodingTest, ComplexResources) {
  auto& getters = enc_->source_op_getters();
  for (int i = *ComplexResourceEnum::kNone;
       i < *ComplexResourceEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No source operand for enum value " << i
                                     << " (" << kComplexResourceNames[i] << ")";
  }
}

TEST_F(RiscV32GZBVecEncodingTest, Rev8AndZextHOpcodes) {
  constexpr uint32_t kRv32Rev8 = 0b011'0100'11000'00000'101'00000'0010011;
  constexpr uint32_t kRv64Rev8 = 0b011'0101'11000'00000'101'00000'0010011;
  constexpr uint32_t kRv32Zexth = 0b000'0100'00000'00000'100'00000'0110011;
  enc_->ParseInstruction(kRv32Rev8);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gvzb, 0), OpcodeEnum::kRev8);
  enc_->ParseInstruction(kRv64Rev8);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gvzb, 0), OpcodeEnum::kNone);
  enc_->ParseInstruction(kRv32Zexth);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gvzb, 0), OpcodeEnum::kZextH);
}

}  // namespace
