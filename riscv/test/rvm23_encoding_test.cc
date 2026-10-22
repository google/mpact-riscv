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

#include <cstdint>

#include "googletest/include/gtest/gtest.h"
#include "riscv/rvm23_bin_decoder.h"
#include "riscv/rvm23_enums.h"

namespace {

using ::mpact::sim::riscv::rvm23::DecodeRVM23Inst16;
using ::mpact::sim::riscv::rvm23::DecodeRVM23Inst32;
using ::mpact::sim::riscv::rvm23::OpcodeEnum;

// Base RV32G instructions.
constexpr uint32_t kAdd = 0b0000000'00011'00010'000'00001'0110011;
constexpr uint32_t kLui = 0b00000000000000000001'00001'0110111;

// Zbb instructions.
constexpr uint32_t kAndn = 0b0100000'00011'00010'111'00001'0110011;

// Zicond instructions.
constexpr uint32_t kCzeroEqz = 0b0000111'00011'00010'101'00001'0110011;

// 16-bit compressed instructions.
constexpr uint16_t kCnop = 0x0001;

TEST(RVM23EncodingTest, Decode32BitBaseAndExtensions) {
  EXPECT_EQ(DecodeRVM23Inst32(kAdd), OpcodeEnum::kAdd);
  EXPECT_EQ(DecodeRVM23Inst32(kLui), OpcodeEnum::kLui);
  EXPECT_EQ(DecodeRVM23Inst32(kAndn), OpcodeEnum::kAndn);
  EXPECT_EQ(DecodeRVM23Inst32(kCzeroEqz), OpcodeEnum::kCzeroEqz);
}

TEST(RVM23EncodingTest, Decode16BitCompressed) {
  EXPECT_EQ(DecodeRVM23Inst16(kCnop), OpcodeEnum::kCnop);
  EXPECT_EQ(DecodeRVM23Inst16(0x0000), OpcodeEnum::kNone);
}

}  // namespace
