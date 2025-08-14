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

#include "riscv/riscv32gzb_encoding.h"

#include <cstdint>

#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "riscv/riscv32gzb_enums.h"
#include "riscv/riscv_state.h"

// This file contains tests for the RiscV32GZBEncoding class to ensure that
// the instruction decoding is correct.

namespace {

using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.

using mpact::sim::riscv::RiscVState;
using mpact::sim::riscv::RiscVXlen;
using mpact::sim::riscv::isa32gzb::ComplexResourceEnum;
using mpact::sim::riscv::isa32gzb::DestOpEnum;
using mpact::sim::riscv::isa32gzb::kComplexResourceNames;
using mpact::sim::riscv::isa32gzb::kDestOpNames;
using mpact::sim::riscv::isa32gzb::kSimpleResourceNames;
using mpact::sim::riscv::isa32gzb::kSourceOpNames;
using mpact::sim::riscv::isa32gzb::RiscV32GZBEncoding;
using mpact::sim::riscv::isa32gzb::SimpleResourceEnum;
using mpact::sim::riscv::isa32gzb::SourceOpEnum;
using mpact::sim::util::FlatDemandMemory;
using SlotEnum = mpact::sim::riscv::isa32gzb::SlotEnum;
using OpcodeEnum = mpact::sim::riscv::isa32gzb::OpcodeEnum;

// Constexpr for opcodes for bit manipulation instructions.

// RV32Zba
constexpr uint32_t kSh1Add = 0b001'0000'00000'00000'010'00000'0110011;
constexpr uint32_t kSh2Add = 0b001'0000'00000'00000'100'00000'0110011;
constexpr uint32_t kSh3Add = 0b001'0000'00000'00000'110'10000'0110011;

// RV32Zbb
constexpr uint32_t kAndn = 0b010'0000'00000'00000'111'00000'0110011;
constexpr uint32_t kOrn = 0b010'0000'00000'00000'110'00000'0110011;
constexpr uint32_t kXnor = 0b010'0000'00000'00000'100'00000'0110011;
constexpr uint32_t kClz = 0b011'0000'00000'00000'001'00000'0010011;
constexpr uint32_t kCtz = 0b011'0000'00001'00000'001'00000'0010011;
constexpr uint32_t kCpop = 0b011'0000'00010'00000'001'00000'0010011;
constexpr uint32_t kMax = 0b000'0101'00000'00000'110'00000'0110011;
constexpr uint32_t kMin = 0b000'0101'00000'00000'100'00000'0110011;
constexpr uint32_t kMaxu = 0b000'0101'00000'00000'111'00000'0110011;
constexpr uint32_t kMinu = 0b000'0101'00000'00000'101'00000'0110011;
constexpr uint32_t kSextb = 0b011'0000'00100'00000'001'00000'0010011;
constexpr uint32_t kSexth = 0b011'0000'00101'00000'001'00000'0010011;
constexpr uint32_t kZexth = 0b000'0100'00000'00000'100'00000'0110011;
constexpr uint32_t kRol = 0b011'0000'00000'00000'001'00000'0110011;
constexpr uint32_t kRor = 0b011'0000'00000'00000'101'00000'0110011;
constexpr uint32_t kRori = 0b011'0000'00000'00000'101'00000'0010011;
constexpr uint32_t kOrcb = 0b001'0100'00111'00000'101'00000'0010011;
constexpr uint32_t kRev8 = 0b011'0100'11000'00000'101'00000'0010011;

// RV32Zbc
constexpr uint32_t kClmul = 0b000'0101'00000'00000'001'00000'0110011;
constexpr uint32_t kClmulh = 0b000'0101'00000'00000'011'00000'0110011;
constexpr uint32_t kClmulr = 0b000'0101'00000'00000'010'00000'0110011;

// RV32Zbs
constexpr uint32_t kBclr = 0b010'0100'00000'00000'001'00000'0110011;
constexpr uint32_t kBclri = 0b010'0100'00000'00000'001'00000'0010011;
constexpr uint32_t kBext = 0b010'0100'00000'00000'101'00000'0110011;
constexpr uint32_t kBexti = 0b010'0100'00000'00000'101'00000'0010011;
constexpr uint32_t kBinv = 0b011'0100'00000'00000'001'00000'0110011;
constexpr uint32_t kBinvi = 0b011'0100'00000'00000'001'00000'0010011;
constexpr uint32_t kBset = 0b001'0100'00000'00000'001'00000'0110011;
constexpr uint32_t kBseti = 0b001'0100'00000'00000'001'00000'0010011;

class RiscV32GZBEncodingTest : public testing::Test {
 protected:
  RiscV32GZBEncodingTest() {
    state_ = new RiscVState("test", RiscVXlen::RV32, &memory_);
    enc_ = new RiscV32GZBEncoding(state_);
  }
  ~RiscV32GZBEncodingTest() override {
    delete enc_;
    delete state_;
  }

  FlatDemandMemory memory_;
  RiscVState* state_;
  RiscV32GZBEncoding* enc_;
};

TEST_F(RiscV32GZBEncodingTest, SourceOperands) {
  auto& getters = enc_->source_op_getters();
  for (int i = *SourceOpEnum::kNone; i < *SourceOpEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No source operand for enum value " << i
                                     << " (" << kSourceOpNames[i] << ")";
  }
}

TEST_F(RiscV32GZBEncodingTest, DestOperands) {
  auto& getters = enc_->dest_op_getters();
  for (int i = *DestOpEnum::kNone; i < *DestOpEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No dest operand for enum value " << i
                                     << " (" << kDestOpNames[i] << ")";
  }
}

TEST_F(RiscV32GZBEncodingTest, SimpleResources) {
  auto& getters = enc_->simple_resource_getters();
  for (int i = *SimpleResourceEnum::kNone;
       i < *SimpleResourceEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No source operand for enum value " << i
                                     << " (" << kSimpleResourceNames[i] << ")";
  }
}

TEST_F(RiscV32GZBEncodingTest, ComplexResources) {
  auto& getters = enc_->source_op_getters();
  for (int i = *ComplexResourceEnum::kNone;
       i < *ComplexResourceEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No source operand for enum value " << i
                                     << " (" << kComplexResourceNames[i] << ")";
  }
}

TEST_F(RiscV32GZBEncodingTest, Zba) {
  enc_->ParseInstruction(kSh1Add);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kSh1add);
  enc_->ParseInstruction(kSh2Add);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kSh2add);
  enc_->ParseInstruction(kSh3Add);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kSh3add);
}

TEST_F(RiscV32GZBEncodingTest, Zbb) {
  enc_->ParseInstruction(kAndn);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kAndn);
  enc_->ParseInstruction(kOrn);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kOrn);
  enc_->ParseInstruction(kXnor);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kXnor);
  enc_->ParseInstruction(kClz);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kClz);
  enc_->ParseInstruction(kCtz);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kCtz);
  enc_->ParseInstruction(kCpop);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kCpop);
  enc_->ParseInstruction(kMax);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kMax);
  enc_->ParseInstruction(kMin);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kMin);
  enc_->ParseInstruction(kMaxu);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kMaxu);
  enc_->ParseInstruction(kMinu);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kMinu);
  enc_->ParseInstruction(kSextb);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kSextB);
  enc_->ParseInstruction(kSexth);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kSextH);
  enc_->ParseInstruction(kZexth);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kZextH);
  enc_->ParseInstruction(kRol);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kRol);
  enc_->ParseInstruction(kRor);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kRor);
  enc_->ParseInstruction(kRori);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kRori);
  enc_->ParseInstruction(kOrcb);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kOrcb);
  enc_->ParseInstruction(kRev8);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kRev8);
}

TEST_F(RiscV32GZBEncodingTest, Zbc) {
  enc_->ParseInstruction(kClmul);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kClmul);
  enc_->ParseInstruction(kClmulh);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kClmulh);
  enc_->ParseInstruction(kClmulr);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kClmulr);
}

TEST_F(RiscV32GZBEncodingTest, Zbs) {
  enc_->ParseInstruction(kBclr);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kBclr);
  enc_->ParseInstruction(kBclri);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kBclri);
  enc_->ParseInstruction(kBext);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kBext);
  enc_->ParseInstruction(kBexti);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kBexti);
  enc_->ParseInstruction(kBinv);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kBinv);
  enc_->ParseInstruction(kBinvi);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kBinvi);
  enc_->ParseInstruction(kBset);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kBset);
  enc_->ParseInstruction(kBseti);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32gzb, 0), OpcodeEnum::kBseti);
}

}  // namespace
