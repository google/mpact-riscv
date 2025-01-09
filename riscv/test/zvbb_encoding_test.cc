// Copyright 2025 Google LLC
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

#include "riscv/zvbb_encoding.h"

#include <cstdint>

#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "riscv/riscv_state.h"
#include "riscv/zvbb_enums.h"

// This file contains tests for the RiscV32GZBEncoding class to ensure that
// the instruction decoding is correct.

namespace {

using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.

using mpact::sim::riscv::RiscVState;
using mpact::sim::riscv::RiscVXlen;
using mpact::sim::util::FlatDemandMemory;

using mpact::sim::riscv::zvbb::kComplexResourceNames;
using mpact::sim::riscv::zvbb::kDestOpNames;
using mpact::sim::riscv::zvbb::kSimpleResourceNames;
using mpact::sim::riscv::zvbb::kSourceOpNames;

using SlotEnum = mpact::sim::riscv::zvbb::SlotEnum;
using OpcodeEnum = mpact::sim::riscv::zvbb::OpcodeEnum;
using SourceOpEnum = mpact::sim::riscv::zvbb::SourceOpEnum;
using DestOpEnum = mpact::sim::riscv::zvbb::DestOpEnum;
using SimpleResourceEnum = mpact::sim::riscv::zvbb::SimpleResourceEnum;
using ComplexResourceEnum = mpact::sim::riscv::zvbb::ComplexResourceEnum;

using mpact::sim::riscv::zvbb::ZVBBEncoding;

// Constexpr for opcodes for vector basic bit manipulation instructions.
constexpr uint32_t kVandnVv = 0b000001'0'00000'00000'000'00000'1010111;
constexpr uint32_t kVandnVx = 0b000001'0'00000'00000'100'00000'1010111;
constexpr uint32_t kVbrev8V = 0b010010'0'00000'01000'010'00000'1010111;
constexpr uint32_t kVrev8V = 0b010010'0'00000'01001'010'00000'1010111;
constexpr uint32_t kVrolVv = 0b010101'0'00000'00000'000'00000'1010111;
constexpr uint32_t kVrolVx = 0b010101'0'00000'00000'100'00000'1010111;
constexpr uint32_t kVrorVv = 0b010100'0'00000'00000'000'00000'1010111;
constexpr uint32_t kVrorVx = 0b010100'0'00000'00000'100'00000'1010111;
constexpr uint32_t kVrorVi_i5_0 = 0b01010'0'0'00000'00000'011'00000'1010111;
constexpr uint32_t kVrorVi_i5_1 = 0b01010'1'0'00000'00000'011'00000'1010111;

constexpr uint32_t kVbrevV = 0b010010'0'00000'01010'010'00000'1010111;
constexpr uint32_t kVclzV = 0b010010'0'00000'01100'010'00000'1010111;
constexpr uint32_t kVctzV = 0b010010'0'00000'01101'010'00000'1010111;
constexpr uint32_t kVcpopV = 0b010010'0'00000'01110'010'00000'1010111;
constexpr uint32_t kVwsllVv = 0b110101'0'00000'00000'000'00000'1010111;
constexpr uint32_t kVwsllVx = 0b110101'0'00000'00000'100'00000'1010111;
constexpr uint32_t kVwsllVi = 0b110101'0'00000'00000'011'00000'1010111;

class ZVBBEncodingTest : public testing::Test {
 protected:
  ZVBBEncodingTest() {
    state_ = new RiscVState("test", RiscVXlen::RV32, &memory_);
    enc_ = new ZVBBEncoding(state_);
  }

  ~ZVBBEncodingTest() override {
    delete enc_;
    delete state_;
  }

  FlatDemandMemory memory_;
  RiscVState *state_;
  ZVBBEncoding *enc_;
};

TEST_F(ZVBBEncodingTest, SourceOperands) {
  auto &getters = enc_->source_op_getters();
  for (int i = *SourceOpEnum::kNone; i < *SourceOpEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No source operand for enum value " << i
                                     << " (" << kSourceOpNames[i] << ")";
  }
}

TEST_F(ZVBBEncodingTest, DestOperands) {
  auto &getters = enc_->dest_op_getters();
  for (int i = *DestOpEnum::kNone; i < *DestOpEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No dest operand for enum value " << i
                                     << " (" << kDestOpNames[i] << ")";
  }
}

TEST_F(ZVBBEncodingTest, SimpleResources) {
  auto &getters = enc_->simple_resource_getters();
  for (int i = *SimpleResourceEnum::kNone;
       i < *SimpleResourceEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No source operand for enum value " << i
                                     << " (" << kSimpleResourceNames[i] << ")";
  }
}

TEST_F(ZVBBEncodingTest, ComplexResources) {
  auto &getters = enc_->source_op_getters();
  for (int i = *ComplexResourceEnum::kNone;
       i < *ComplexResourceEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No source operand for enum value " << i
                                     << " (" << kComplexResourceNames[i] << ")";
  }
}

TEST_F(ZVBBEncodingTest, VandnVv) {
  enc_->ParseInstruction(kVandnVv);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscvZvbb, 0), OpcodeEnum::kVandnVv);
}

TEST_F(ZVBBEncodingTest, VandnVx) {
  enc_->ParseInstruction(kVandnVx);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscvZvbb, 0), OpcodeEnum::kVandnVx);
}

TEST_F(ZVBBEncodingTest, Vbrev8V) {
  enc_->ParseInstruction(kVbrev8V);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscvZvbb, 0), OpcodeEnum::kVbrev8V);
}

TEST_F(ZVBBEncodingTest, Vrev8V) {
  enc_->ParseInstruction(kVrev8V);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscvZvbb, 0), OpcodeEnum::kVrev8V);
}

TEST_F(ZVBBEncodingTest, VrolVv) {
  enc_->ParseInstruction(kVrolVv);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscvZvbb, 0), OpcodeEnum::kVrolVv);
}

TEST_F(ZVBBEncodingTest, VrolVx) {
  enc_->ParseInstruction(kVrolVx);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscvZvbb, 0), OpcodeEnum::kVrolVx);
}

TEST_F(ZVBBEncodingTest, VrorVv) {
  enc_->ParseInstruction(kVrorVv);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscvZvbb, 0), OpcodeEnum::kVrorVv);
}

TEST_F(ZVBBEncodingTest, VrorVx) {
  enc_->ParseInstruction(kVrorVx);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscvZvbb, 0), OpcodeEnum::kVrorVx);
}

TEST_F(ZVBBEncodingTest, VrorVi) {
  enc_->ParseInstruction(kVrorVi_i5_0);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscvZvbb, 0), OpcodeEnum::kVrorVi);
  enc_->ParseInstruction(kVrorVi_i5_1);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscvZvbb, 0), OpcodeEnum::kVrorVi);
}

TEST_F(ZVBBEncodingTest, VbrevV) {
  enc_->ParseInstruction(kVbrevV);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscvZvbb, 0), OpcodeEnum::kVbrevV);
}

TEST_F(ZVBBEncodingTest, VclzV) {
  enc_->ParseInstruction(kVclzV);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscvZvbb, 0), OpcodeEnum::kVclzV);
}

TEST_F(ZVBBEncodingTest, VctzV) {
  enc_->ParseInstruction(kVctzV);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscvZvbb, 0), OpcodeEnum::kVctzV);
}

TEST_F(ZVBBEncodingTest, VcpopV) {
  enc_->ParseInstruction(kVcpopV);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscvZvbb, 0), OpcodeEnum::kVcpopV);
}

TEST_F(ZVBBEncodingTest, VwsllVv) {
  enc_->ParseInstruction(kVwsllVv);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscvZvbb, 0), OpcodeEnum::kVwsllVv);
}

TEST_F(ZVBBEncodingTest, VwsllVx) {
  enc_->ParseInstruction(kVwsllVx);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscvZvbb, 0), OpcodeEnum::kVwsllVx);
}

TEST_F(ZVBBEncodingTest, VwsllVi) {
  enc_->ParseInstruction(kVwsllVi);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscvZvbb, 0), OpcodeEnum::kVwsllVi);
}

}  // namespace
