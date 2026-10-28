// Copyright 2026 Google LLC
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

// Functional encoding and binary decoding tests for the RISC-V Vector Matrix
// Extension (Zvt v0.4 and legacy v0.3 compatibility instructions).

#include <cstdint>

#include "googletest/include/gtest/gtest.h"
#include "riscv/riscv_zvt_bin_decoder.h"
#include "riscv/riscv_zvt_enums.h"

namespace {

using ::mpact::sim::riscv::zvt::DecodeZvtInst32;
using ::mpact::sim::riscv::zvt::OpcodeEnum;

// Base RV32G instruction.
constexpr uint32_t kAdd = 0b0000000'00011'00010'000'00001'0110011;

// Zvt v0.4 instructions.
constexpr uint32_t kMsetmtype = 0b1000010'00000'00000'111'00000'1010111;
constexpr uint32_t kMsetmtypei = 0b1000100'00100'00000'111'00000'1010111;
constexpr uint32_t kMsettn = 0b1000100'00000'00000'111'00000'1010111;
constexpr uint32_t kMsettm = 0b1000100'00001'00000'111'00000'1010111;
constexpr uint32_t kMsettk = 0b1000100'00010'00000'111'00000'1010111;
constexpr uint32_t kVtzero = 0b010000'1'11110'00000'110'00000'1010111;
constexpr uint32_t kVtdiscard = 0b010000'1'11100'00000'110'00000'1010111;
constexpr uint32_t kVtmvVt = 0b010000'1'11111'00000'110'00000'1010111;
constexpr uint32_t kVtmvTv = 0b010111'1'00000'00000'110'00000'1010111;
constexpr uint32_t kVtle8 = 0b0001001'00000'00000'111'00000'0000111;
constexpr uint32_t kVtle16 = 0b0011001'00000'00000'111'00000'0000111;
constexpr uint32_t kVtle32 = 0b0101001'00000'00000'111'00000'0000111;
constexpr uint32_t kVtle64 = 0b0111001'00000'00000'111'00000'0000111;
constexpr uint32_t kVtse8 = 0b0001001'00000'00000'111'00000'0100111;
constexpr uint32_t kVtse16 = 0b0011001'00000'00000'111'00000'0100111;
constexpr uint32_t kVtse32 = 0b0101001'00000'00000'111'00000'0100111;
constexpr uint32_t kVtse64 = 0b0111001'00000'00000'111'00000'0100111;
constexpr uint32_t kVtmmuTvv = 0b1111000'00000'00000'000'00000'1110111;
constexpr uint32_t kVtmmsTvv = 0b1111000'00000'00000'000'00001'1110111;
constexpr uint32_t kVtfmmTvv = 0b1111000'00000'00000'001'00000'1110111;
constexpr uint32_t kVtfmmAltTvv = 0b1111000'00000'00000'001'00001'1110111;

// Legacy Zvt v0.3 instructions.
constexpr uint32_t kV0_3_Msetmtype = 0b1000001'00010'00001'111'00011'1010111;
constexpr uint32_t kV0_3_Msettn = 0b1000010'00000'00001'111'00011'1010111;
constexpr uint32_t kV0_3_Msettm = 0b1000010'00001'00001'111'00011'1010111;
constexpr uint32_t kV0_3_Msettk = 0b1000010'00010'00001'111'00011'1010111;

TEST(RiscVZvtEncodingTest, DecodeZvtV0_4ConfigurationInstructions) {
  EXPECT_EQ(DecodeZvtInst32(kMsetmtype), OpcodeEnum::kMsetmtype);
  EXPECT_EQ(DecodeZvtInst32(kMsetmtypei), OpcodeEnum::kMsetmtypei);
  EXPECT_EQ(DecodeZvtInst32(kMsettn), OpcodeEnum::kMsettn);
  EXPECT_EQ(DecodeZvtInst32(kMsettm), OpcodeEnum::kMsettm);
  EXPECT_EQ(DecodeZvtInst32(kMsettk), OpcodeEnum::kMsettk);
}

TEST(RiscVZvtEncodingTest, DecodeZvtV0_4TileManipulationInstructions) {
  EXPECT_EQ(DecodeZvtInst32(kVtzero), OpcodeEnum::kVtzero);
  EXPECT_EQ(DecodeZvtInst32(kVtdiscard), OpcodeEnum::kVtdiscard);
  EXPECT_EQ(DecodeZvtInst32(kVtmvVt), OpcodeEnum::kVtmvVT);
  EXPECT_EQ(DecodeZvtInst32(kVtmvTv), OpcodeEnum::kVtmvTV);
}

TEST(RiscVZvtEncodingTest, DecodeZvtV0_4TileLoadStoreInstructions) {
  EXPECT_EQ(DecodeZvtInst32(kVtle8), OpcodeEnum::kVtle8);
  EXPECT_EQ(DecodeZvtInst32(kVtle16), OpcodeEnum::kVtle16);
  EXPECT_EQ(DecodeZvtInst32(kVtle32), OpcodeEnum::kVtle32);
  EXPECT_EQ(DecodeZvtInst32(kVtle64), OpcodeEnum::kVtle64);

  EXPECT_EQ(DecodeZvtInst32(kVtse8), OpcodeEnum::kVtse8);
  EXPECT_EQ(DecodeZvtInst32(kVtse16), OpcodeEnum::kVtse16);
  EXPECT_EQ(DecodeZvtInst32(kVtse32), OpcodeEnum::kVtse32);
  EXPECT_EQ(DecodeZvtInst32(kVtse64), OpcodeEnum::kVtse64);
}

TEST(RiscVZvtEncodingTest, DecodeZvtV0_4MatrixMultiplyInstructions) {
  EXPECT_EQ(DecodeZvtInst32(kVtmmuTvv), OpcodeEnum::kVtmmuTvv);
  EXPECT_EQ(DecodeZvtInst32(kVtmmsTvv), OpcodeEnum::kVtmmsTvv);
  EXPECT_EQ(DecodeZvtInst32(kVtfmmTvv), OpcodeEnum::kVtfmmTvv);
  EXPECT_EQ(DecodeZvtInst32(kVtfmmAltTvv), OpcodeEnum::kVtfmmAltTvv);
}

TEST(RiscVZvtEncodingTest, RejectNonZvtInstructions) {
  EXPECT_EQ(DecodeZvtInst32(kAdd), OpcodeEnum::kNone);
  EXPECT_EQ(DecodeZvtInst32(0x00000000), OpcodeEnum::kNone);
}

TEST(RiscVZvtEncodingTest, DecodeLegacyZvtV0_3CompatibilityInstructions) {
  EXPECT_EQ(DecodeZvtInst32(kV0_3_Msetmtype), OpcodeEnum::kMsetmtype);
  EXPECT_EQ(DecodeZvtInst32(kV0_3_Msettn), OpcodeEnum::kMsettn);
  EXPECT_EQ(DecodeZvtInst32(kV0_3_Msettm), OpcodeEnum::kMsettm);
  EXPECT_EQ(DecodeZvtInst32(kV0_3_Msettk), OpcodeEnum::kMsettk);
}

}  // namespace
