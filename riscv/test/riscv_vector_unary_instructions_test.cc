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

#include "riscv/riscv_vector_unary_instructions.h"

#include <algorithm>
#include <cstring>
#include <ios>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/random/random.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_vector_state.h"
#include "riscv/test/riscv_vector_instructions_test_base.h"

namespace {

using ::absl::Span;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RVVectorRegister;
using ::mpact::sim::riscv::SameSignedType;
using ::mpact::sim::riscv::test::RiscVVectorInstructionsTestBase;

using ::mpact::sim::riscv::Vcpop;
using ::mpact::sim::riscv::Vfirst;
using ::mpact::sim::riscv::Vid;
using ::mpact::sim::riscv::Viota;
using ::mpact::sim::riscv::Vmsbf;
using ::mpact::sim::riscv::Vmsif;
using ::mpact::sim::riscv::Vmsof;
using ::mpact::sim::riscv::VmvFromScalar;
using ::mpact::sim::riscv::VmvToScalar;
using ::mpact::sim::riscv::Vsext2;
using ::mpact::sim::riscv::Vsext4;
using ::mpact::sim::riscv::Vsext8;
using ::mpact::sim::riscv::Vzext2;
using ::mpact::sim::riscv::Vzext4;
using ::mpact::sim::riscv::Vzext8;

using ::mpact::sim::riscv::test::kA5Mask;
using ::mpact::sim::riscv::test::kLmulSettingByLogSize;
using ::mpact::sim::riscv::test::kRd;
using ::mpact::sim::riscv::test::kRdName;
using ::mpact::sim::riscv::test::kRs1;
using ::mpact::sim::riscv::test::kRs1Name;
using ::mpact::sim::riscv::test::kSewSettingsByByteSize;
using ::mpact::sim::riscv::test::kVd;
using ::mpact::sim::riscv::test::kVectorLengthInBytes;
using ::mpact::sim::riscv::test::kVmask;
using ::mpact::sim::riscv::test::kVmaskName;
using ::mpact::sim::riscv::test::kVs2;
using ::mpact::sim::riscv::test::kVs2Name;

using SignedXregType = SameSignedType<RV32Register::ValueType, int64_t>::type;

constexpr uint8_t k5AMask[kVectorLengthInBytes] = {
    0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a,
    0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a,
    0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a,
    0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a,
    0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a,
    0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a,
};

constexpr uint8_t kE7Mask[kVectorLengthInBytes] = {
    0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7,
    0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7,
    0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7,
    0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7,
    0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7,
    0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7, 0xe7,
};

constexpr uint8_t kAllOnesMask[kVectorLengthInBytes] = {
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
};

class RiscVVectorUnaryInstructionsTest
    : public RiscVVectorInstructionsTestBase {};

// Test move vector element 0 to scalar register.
TEST_F(RiscVVectorUnaryInstructionsTest, VmvToScalar) {
  SetSemanticFunction(&VmvToScalar);
  AppendRegisterOperands({}, {kRs1Name});
  AppendVectorRegisterOperands({kVs2}, {});
  for (int byte_sew : {1, 2, 4, 8}) {
    int vlen = kVectorLengthInBytes / byte_sew;
    uint32_t vtype =
        (kSewSettingsByByteSize[byte_sew] << 3) | kLmulSettingByLogSize[4];
    ConfigureVectorUnit(vtype, vlen);
    // Test 10 different values.
    for (int i = 0; i < 10; i++) {
      int64_t value;
      switch (byte_sew) {
        case 1: {
          auto val8 = RandomValue<int8_t>();
          value = static_cast<int64_t>(val8);
          SetVectorRegisterValues<int8_t>(
              {{kVs2Name, absl::Span<const int8_t>(&val8, 1)}});
          break;
        }
        case 2: {
          auto val16 = RandomValue<int16_t>();
          value = static_cast<int64_t>(val16);
          SetVectorRegisterValues<int16_t>(
              {{kVs2Name, absl::Span<const int16_t>(&val16, 1)}});
          break;
        }
        case 4: {
          auto val32 = RandomValue<int32_t>();
          value = static_cast<int64_t>(val32);
          SetVectorRegisterValues<int32_t>(
              {{kVs2Name, absl::Span<const int32_t>(&val32, 1)}});
          break;
        }
        case 8: {
          auto val64 = RandomValue<int64_t>();
          value = val64;
          SetVectorRegisterValues<int64_t>(
              {{kVs2Name, absl::Span<const int64_t>(&val64, 1)}});
          break;
        }
      }
      instruction_->Execute();
      EXPECT_EQ(xreg_[kRs1]->data_buffer()->Get<SignedXregType>(0),
                static_cast<SignedXregType>(value));
    }
  }
}

// Test move scalar to vector element 0.
TEST_F(RiscVVectorUnaryInstructionsTest, VmvFromScalar) {
  SetSemanticFunction(&VmvFromScalar);
  AppendRegisterOperands({kRs1Name}, {});
  AppendVectorRegisterOperands({}, {kVs2});
  for (int byte_sew : {1, 2, 4, 8}) {
    int vlen = kVectorLengthInBytes / byte_sew;
    uint32_t vtype =
        (kSewSettingsByByteSize[byte_sew] << 3) | kLmulSettingByLogSize[4];
    ConfigureVectorUnit(vtype, vlen);
    // Test 10 different values.
    for (int i = 0; i < 10; i++) {
      auto value = RandomValue<SignedXregType>();
      SetRegisterValues<SignedXregType>({{kRs1Name, value}});
      instruction_->Execute();
      switch (byte_sew) {
        case 1:
          EXPECT_EQ(vreg_[kVs2]->data_buffer()->Get<int8_t>(0),
                    static_cast<int8_t>(value));
          break;
        case 2:
          EXPECT_EQ(vreg_[kVs2]->data_buffer()->Get<int16_t>(0),
                    static_cast<int16_t>(value));
          break;
        case 4:
          EXPECT_EQ(vreg_[kVs2]->data_buffer()->Get<int32_t>(0),
                    static_cast<int32_t>(value));
          break;
        case 8:
          EXPECT_EQ(vreg_[kVs2]->data_buffer()->Get<int64_t>(0),
                    static_cast<int64_t>(value));
          break;
      }
    }
  }
}

// Test vector mask population count.
TEST_F(RiscVVectorUnaryInstructionsTest, Vcpop) {
  uint32_t vtype = (kSewSettingsByByteSize[1] << 3) | kLmulSettingByLogSize[7];
  SetSemanticFunction(&Vcpop);
  AppendVectorRegisterOperands({kVs2, kVmask}, {});
  AppendRegisterOperands({}, {kRdName});
  for (int vlen : {1, 8, 32, 48, 127, 200}) {
    ConfigureVectorUnit(vtype, vlen);
    // All 1s for mask and vector.
    SetVectorRegisterValues<uint8_t>(
        {{kVs2Name, kAllOnesMask}, {kVmaskName, kAllOnesMask}});
    instruction_->Execute();
    EXPECT_EQ(xreg_[kRd]->data_buffer()->Get<RV32Register::ValueType>(0), vlen);

    // Mask is inverse of vector. Will result in 0.
    SetVectorRegisterValues<uint8_t>(
        {{kVs2Name, kA5Mask}, {kVmaskName, k5AMask}});
    instruction_->Execute();
    EXPECT_EQ(xreg_[kRd]->data_buffer()->Get<RV32Register::ValueType>(0), 0);
  }
}

// Test vector mask find first set.
TEST_F(RiscVVectorUnaryInstructionsTest, Vfirst) {
  SetSemanticFunction(&Vfirst);
  AppendVectorRegisterOperands({kVs2, kVmask}, {});
  AppendRegisterOperands({}, {kRdName});
  uint8_t reg_value[kVectorLengthInBytes];
  // Set vtype to byte vector, and vector lmul to 8.
  uint32_t vtype = (kSewSettingsByByteSize[1] << 3) | kLmulSettingByLogSize[7];
  ConfigureVectorUnit(vtype, kVectorLengthInBytes * 8);

  // Clear the reg_value array.
  std::memset(reg_value, 0, kVectorLengthInBytes);
  // Set the register values.
  SetVectorRegisterValues<uint8_t>(
      {{kVs2Name, reg_value}, {kVmaskName, kAllOnesMask}});
  // Execute the instruction. The result should be minus 1.
  instruction_->Execute();
  EXPECT_EQ(xreg_[kRd]->data_buffer()->Get<SignedXregType>(0), -1);

  // Pick a random location 20 times and set that bit to 1. Test first that
  // the correct value is returned, then clear the mask bit that corresponds to
  // that value, and ensure that now the result is -1.
  for (int i = 0; i < 20; i++) {
    // Clear the reg_value array.
    std::memset(reg_value, 0, kVectorLengthInBytes);
    // Get a random value for index to set.
    uint32_t index = absl::Uniform(absl::IntervalClosed, bitgen_, 0,
                                   kVectorLengthInBytes * 8 - 1);
    // Compute the byte and bit index to be set.
    auto byte_index = index >> 3;
    auto bit_index = index & 0b111;
    // Set the bit in the source register.
    reg_value[byte_index] |= 1 << bit_index;
    // Set the register values.
    SetVectorRegisterValues<uint8_t>(
        {{kVs2Name, reg_value}, {kVmaskName, kAllOnesMask}});
    // Execute the instruction. The result should be the index value.
    instruction_->Execute();
    EXPECT_EQ(xreg_[kRd]->data_buffer()->Get<RV32Register::ValueType>(0),
              index);

    // Clear the mask bit for the bit that was set.
    auto mask_db = vreg_[kVmask]->data_buffer()->Get<uint8_t>();
    mask_db[byte_index] &= ~(1 << bit_index);
    // Execute the instruction. The result should be minus 1.
    instruction_->Execute();
    EXPECT_EQ(xreg_[kRd]->data_buffer()->Get<SignedXregType>(0), -1);
  }
}

// Zero extension from sew/2 to sew. Test for sew of 16, 32 and 64 bits.
TEST_F(RiscVVectorUnaryInstructionsTest, Vzext2_16) {
  SetSemanticFunction(&Vzext2);
  UnaryOpTestHelperV<uint16_t, uint8_t>(
      "Vzext2_16", /*sew*/ 16, instruction_,
      [](uint8_t vs2) -> uint16_t { return vs2; });
}

TEST_F(RiscVVectorUnaryInstructionsTest, Vzext2_32) {
  SetSemanticFunction(&Vzext2);
  UnaryOpTestHelperV<uint32_t, uint16_t>(
      "Vzext2_32", /*sew*/ 32, instruction_,
      [](uint16_t vs2) -> uint32_t { return vs2; });
}

TEST_F(RiscVVectorUnaryInstructionsTest, Vzext2_64) {
  SetSemanticFunction(&Vzext2);
  UnaryOpTestHelperV<uint64_t, uint32_t>(
      "Vzext2_64", /*sew*/ 64, instruction_,
      [](uint32_t vs2) -> uint64_t { return vs2; });
}

// Sign extension from sew/2 to sew. Testing for sew of 16, 32 and 64.
TEST_F(RiscVVectorUnaryInstructionsTest, Vsext2_16) {
  SetSemanticFunction(&Vsext2);
  UnaryOpTestHelperV<int16_t, int8_t>("Vsext2_16", /*sew*/ 16, instruction_,
                                      [](int8_t vs2) -> int16_t {
                                        int16_t res = vs2;
                                        res <<= 8;
                                        res >>= 8;
                                        return res;
                                      });
}

TEST_F(RiscVVectorUnaryInstructionsTest, Vsext2_32) {
  SetSemanticFunction(&Vsext2);
  UnaryOpTestHelperV<int32_t, int16_t>("Vsext2_32", /*sew*/ 32, instruction_,
                                       [](int16_t vs2) -> int32_t {
                                         int32_t res = vs2;
                                         return (res << 16) >> 16;
                                       });
}

TEST_F(RiscVVectorUnaryInstructionsTest, Vsext2_64) {
  SetSemanticFunction(&Vsext2);
  UnaryOpTestHelperV<int64_t, int32_t>("Vsext2_64", /*sew*/ 64, instruction_,
                                       [](int32_t vs2) -> int64_t {
                                         int64_t res = vs2;
                                         return (res << 32) >> 32;
                                       });
}

// Zero extension from sew/4 to sew. Testing for sew of 32 and 64.
TEST_F(RiscVVectorUnaryInstructionsTest, Vzext4_32) {
  SetSemanticFunction(&Vzext4);
  UnaryOpTestHelperV<uint32_t, uint8_t>(
      "Vzext4_32", /*sew*/ 32, instruction_,
      [](uint8_t vs2) -> uint32_t { return vs2; });
}

TEST_F(RiscVVectorUnaryInstructionsTest, Vzext4_64) {
  SetSemanticFunction(&Vzext4);
  UnaryOpTestHelperV<uint64_t, uint16_t>(
      "Vzext4_32", /*sew*/ 64, instruction_,
      [](uint16_t vs2) -> uint64_t { return vs2; });
}

// Sign extension from sew/4 to sew. Testing for sew of 32 and 64.
TEST_F(RiscVVectorUnaryInstructionsTest, Vsext4_32) {
  SetSemanticFunction(&Vsext4);
  UnaryOpTestHelperV<int32_t, int8_t>("Vzext4_32", /*sew*/ 32, instruction_,
                                      [](int8_t vs2) -> int32_t {
                                        int32_t res = vs2;
                                        return (res << 24) >> 24;
                                      });
}

TEST_F(RiscVVectorUnaryInstructionsTest, Vsext4_64) {
  SetSemanticFunction(&Vsext4);
  UnaryOpTestHelperV<int64_t, int16_t>("Vzext4_64", /*sew*/ 64, instruction_,
                                       [](int16_t vs2) -> int64_t {
                                         int64_t res = vs2;
                                         return (res << 48) >> 48;
                                       });
}

// Zero extension from sew/8 to sew. Testing for sew of 64.
TEST_F(RiscVVectorUnaryInstructionsTest, Vzext8_64) {
  SetSemanticFunction(&Vzext8);
  UnaryOpTestHelperV<uint64_t, uint8_t>("Vsext8_64", /*sew*/ 64, instruction_,
                                        [](uint8_t vs2) -> uint64_t {
                                          uint64_t res = vs2;
                                          return (res << 56) >> 56;
                                        });
}

// Sign extension from sew/8 to sew. Testing for sew of 64.
TEST_F(RiscVVectorUnaryInstructionsTest, Vsext8_64) {
  SetSemanticFunction(&Vsext8);
  UnaryOpTestHelperV<int64_t, int8_t>("Vzext8_64", /*sew*/ 64, instruction_,
                                      [](int8_t vs2) -> int64_t {
                                        int64_t res = vs2;
                                        return (res << 56) >> 56;
                                      });
}

// Test "set before first mask bit".
TEST_F(RiscVVectorUnaryInstructionsTest, Vmsbf) {
  SetSemanticFunction(&Vmsbf);
  AppendVectorRegisterOperands({kVs2, kVmask}, {kVd});
  uint8_t reg_value[kVectorLengthInBytes];
  // Set vtype to byte vector, and vector lmul to 8.
  uint32_t vtype = (kSewSettingsByByteSize[1] << 3) | kLmulSettingByLogSize[7];
  ConfigureVectorUnit(vtype, kVectorLengthInBytes * 8);

  // Clear the reg_value array.
  std::memset(reg_value, 0, kVectorLengthInBytes);
  // Set the register values.
  SetVectorRegisterValues<uint8_t>(
      {{kVs2Name, reg_value}, {kVmaskName, kAllOnesMask}});
  // Execute the instruction. The result should be all 1s.
  instruction_->Execute();
  auto dest_span = vreg_[kVd]->data_buffer()->Get<uint8_t>();
  for (int i = 0; i < kVectorLengthInBytes; i++) {
    EXPECT_EQ(dest_span[i], 0b1111'1111) << "Index: " << i;
  }

  // Pick a random location 20 times and set that bit to 1. Test first that
  // the vector mask is produced.
  for (int i = 0; i < 20; i++) {
    // Clear the reg_value array.
    // Set the register values.
    SetVectorRegisterValues<uint8_t>(
        {{kVs2Name, kA5Mask}, {kVmaskName, k5AMask}});
    // Get a random value for which bit index to set.
    uint32_t index = absl::Uniform(absl::IntervalClosedOpen, bitgen_, 0,
                                   kVectorLengthInBytes * 8);
    // Compute the byte and bit index to be set.
    auto byte_index = index >> 3;
    auto bit_index = index & 0b111;
    // Set the bit in the source register and mask registers.
    auto mask_span = vreg_[kVmask]->data_buffer()->Get<uint8_t>();
    auto src_span = vreg_[kVs2]->data_buffer()->Get<uint8_t>();
    mask_span[byte_index] |= 1 << bit_index;
    src_span[byte_index] |= 1 << bit_index;
    // Execute the instruction. The result should be the index value.
    instruction_->Execute();
    // Check results.
    dest_span = vreg_[kVd]->data_buffer()->Get<uint8_t>();
    // First check all the flag values before the byte where the index is.
    for (int j = 0; j < byte_index; j++) {
      EXPECT_EQ(dest_span[j],
                (0b1111'1111 & mask_span[j]) | (dest_span[j] & ~mask_span[j]));
    }
    // Check the flag values of the byte where the index is.
    EXPECT_EQ(dest_span[byte_index],
              (((1 << bit_index) - 1) & mask_span[byte_index]) |
                  (dest_span[byte_index] & ~mask_span[byte_index]));
    // Check the flag values after the byte where the index is.
    for (int j = byte_index + 1; j < kVectorLengthInBytes; j++) {
      EXPECT_EQ(dest_span[j], dest_span[j] & ~mask_span[j]);
    }
  }
}

TEST_F(RiscVVectorUnaryInstructionsTest, Vmsof) {
  SetSemanticFunction(&Vmsof);
  AppendVectorRegisterOperands({kVs2, kVmask}, {kVd});
  uint8_t reg_value[kVectorLengthInBytes];
  // Set vtype to byte vector, and vector lmul to 8.
  uint32_t vtype = (kSewSettingsByByteSize[1] << 3) | kLmulSettingByLogSize[7];
  ConfigureVectorUnit(vtype, kVectorLengthInBytes * 8);

  // Clear the reg_value array.
  std::memset(reg_value, 0, kVectorLengthInBytes);
  // Set the register values.
  SetVectorRegisterValues<uint8_t>(
      {{kVs2Name, reg_value}, {kVmaskName, kAllOnesMask}});
  // Execute the instruction. The result should be all 1s.
  instruction_->Execute();
  auto dest_span = vreg_[kVd]->data_buffer()->Get<uint8_t>();
  for (int i = 0; i < kVectorLengthInBytes; i++) {
    EXPECT_EQ(dest_span[i], 0) << "Index: " << i;
  }

  // Pick a random location 20 times and set that bit to 1. Test first that
  // the vector mask is produced.
  for (int i = 0; i < 20; i++) {
    // Clear the reg_value array.
    // Set the register values.
    SetVectorRegisterValues<uint8_t>(
        {{kVs2Name, kA5Mask}, {kVmaskName, k5AMask}});
    // Get a random value for which bit index to set.
    uint32_t index = absl::Uniform(absl::IntervalClosedOpen, bitgen_, 0,
                                   kVectorLengthInBytes * 8);
    // Compute the byte and bit index to be set.
    auto byte_index = index >> 3;
    auto bit_index = index & 0b111;
    // Set the bit in the source register and mask registers.
    auto mask_span = vreg_[kVmask]->data_buffer()->Get<uint8_t>();
    auto src_span = vreg_[kVs2]->data_buffer()->Get<uint8_t>();
    mask_span[byte_index] |= 1 << bit_index;
    src_span[byte_index] |= 1 << bit_index;
    // Execute the instruction. The result should be the index value.
    instruction_->Execute();
    // Check results.
    dest_span = vreg_[kVd]->data_buffer()->Get<uint8_t>();
    // First check all the flag values before the byte where the index is.
    for (int j = 0; j < byte_index; j++) {
      EXPECT_EQ(dest_span[j], dest_span[j] & ~mask_span[j]);
    }
    // Check the flag values of the byte where the index is.
    EXPECT_EQ(dest_span[byte_index],
              ((1 << bit_index) & mask_span[byte_index]) |
                  (dest_span[byte_index] & ~mask_span[byte_index]))
        << " dest: " << std::hex << (int)dest_span[byte_index]
        << "  mask: " << (int)mask_span[byte_index];
    // Check the flag values after the byte where the index is.
    for (int j = byte_index + 1; j < kVectorLengthInBytes; j++) {
      EXPECT_EQ(dest_span[j], dest_span[j] & ~mask_span[j]);
    }
  }
}

TEST_F(RiscVVectorUnaryInstructionsTest, Vmsif) {
  SetSemanticFunction(&Vmsif);
  AppendVectorRegisterOperands({kVs2, kVmask}, {kVd});
  uint8_t reg_value[kVectorLengthInBytes];
  // Set vtype to byte vector, and vector lmul to 8.
  uint32_t vtype = (kSewSettingsByByteSize[1] << 3) | kLmulSettingByLogSize[7];
  ConfigureVectorUnit(vtype, kVectorLengthInBytes * 8);

  // Clear the reg_value array.
  std::memset(reg_value, 0, kVectorLengthInBytes);
  // Set the register values.
  SetVectorRegisterValues<uint8_t>(
      {{kVs2Name, reg_value}, {kVmaskName, kAllOnesMask}});
  // Execute the instruction. The result should be all 1s.
  instruction_->Execute();
  auto dest_span = vreg_[kVd]->data_buffer()->Get<uint8_t>();
  for (int i = 0; i < kVectorLengthInBytes; i++) {
    EXPECT_EQ(dest_span[i], 0b1111'1111) << "Index: " << i;
  }

  // Pick a random location 20 times and set that bit to 1. Test first that
  // the vector mask is produced.
  for (int i = 0; i < 20; i++) {
    // Clear the reg_value array.
    // Set the register values.
    SetVectorRegisterValues<uint8_t>(
        {{kVs2Name, kA5Mask}, {kVmaskName, k5AMask}});
    // Get a random value for which bit index to set.
    uint32_t index = absl::Uniform(absl::IntervalClosedOpen, bitgen_, 0,
                                   kVectorLengthInBytes * 8);
    // Compute the byte and bit index to be set.
    auto byte_index = index >> 3;
    auto bit_index = index & 0b111;
    // Set the bit in the source register and mask registers.
    auto mask_span = vreg_[kVmask]->data_buffer()->Get<uint8_t>();
    auto src_span = vreg_[kVs2]->data_buffer()->Get<uint8_t>();
    mask_span[byte_index] |= 1 << bit_index;
    src_span[byte_index] |= 1 << bit_index;
    // Execute the instruction. The result should be the index value.
    instruction_->Execute();
    // Check results.
    dest_span = vreg_[kVd]->data_buffer()->Get<uint8_t>();
    // First check all the flag values before the byte where the index is.
    for (int j = 0; j < byte_index; j++) {
      EXPECT_EQ(dest_span[j],
                (0b1111'1111 & mask_span[j]) | (dest_span[j] & ~mask_span[j]));
    }
    // Check the flag values of the byte where the index is.
    EXPECT_EQ(dest_span[byte_index],
              (((1 << (bit_index + 1)) - 1) & mask_span[byte_index]) |
                  (dest_span[byte_index] & ~mask_span[byte_index]));
    // Check the flag values after the byte where the index is.
    for (int j = byte_index + 1; j < kVectorLengthInBytes; j++) {
      EXPECT_EQ(dest_span[j], dest_span[j] & ~mask_span[j]);
    }
  }
}

// Helper function for testing Viota instructions.
template <typename T>
void TestViota(RiscVVectorUnaryInstructionsTest *tester, Instruction *inst) {
  // Set up vector unit.
  int byte_sew = sizeof(T);
  uint32_t vtype =
      (kSewSettingsByByteSize[byte_sew] << 3) | kLmulSettingByLogSize[7];
  tester->ConfigureVectorUnit(vtype, 1024);
  int vlen = tester->rv_vector()->vector_length();
  int num_reg =
      (vlen * byte_sew + kVectorLengthInBytes - 1) / kVectorLengthInBytes;
  int num_per_reg = kVectorLengthInBytes / byte_sew;

  // Set up instruction.
  tester->SetSemanticFunction(&Viota);
  tester->AppendVectorRegisterOperands({kVmask}, {kVd});
  tester->SetVectorRegisterValues<uint8_t>({{kVmaskName, kE7Mask}});
  int count = vlen;
  for (int reg = kVd; reg < kVd + num_reg; reg++) {
    auto reg_span = tester->vreg()[reg]->data_buffer()->Get<T>();
    for (int i = 0; i < num_per_reg; i++) {
      reg_span[i] = static_cast<T>(count--);
    }
  }

  // Execute instruction.
  inst->Execute();

  // Check results.
  auto mask_span = tester->vreg()[kVmask]->data_buffer()->Get<T>();
  count = 0;
  for (int i = 0; i < vlen; i++) {
    int reg = kVd + i / num_per_reg;
    int reg_index = i % num_per_reg;
    auto value = tester->vreg()[reg]->data_buffer()->Get<T>(reg_index);
    int mask_index = i >> 3;
    int mask_offset = i & 0b111;
    int mask_value = (mask_span[mask_index] >> mask_offset) & 0b1;
    if (mask_value) {
      EXPECT_EQ(value, static_cast<T>(count)) << "active index: " << i;
      count++;
    } else {
      EXPECT_EQ(value, static_cast<T>(vlen - i)) << "inactive index: " << i;
    }
  }
}

// Test the viota instruction.
TEST_F(RiscVVectorUnaryInstructionsTest, Viota8) {
  TestViota<uint8_t>(this, instruction_);
}

TEST_F(RiscVVectorUnaryInstructionsTest, Viota16) {
  TestViota<uint16_t>(this, instruction_);
}

TEST_F(RiscVVectorUnaryInstructionsTest, Viota32) {
  TestViota<uint32_t>(this, instruction_);
}

TEST_F(RiscVVectorUnaryInstructionsTest, Viota64) {
  TestViota<uint64_t>(this, instruction_);
}

// Helper function for testing Vid instructions.
template <typename T>
void TestVid(RiscVVectorUnaryInstructionsTest *tester, Instruction *inst) {
  // Initialize the vector unit.
  int byte_sew = sizeof(T);
  uint32_t vtype =
      (kSewSettingsByByteSize[byte_sew] << 3) | kLmulSettingByLogSize[7];
  int vlen = tester->rv_vector()->vector_length();
  int num_reg =
      (vlen * byte_sew + kVectorLengthInBytes - 1) / kVectorLengthInBytes;
  int num_per_reg = kVectorLengthInBytes / byte_sew;
  tester->ConfigureVectorUnit(vtype, 1024);

  // Configure the instruction.
  tester->SetSemanticFunction(&Vid);
  tester->AppendVectorRegisterOperands({kVmask}, {kVd});
  tester->SetVectorRegisterValues<uint8_t>({{kVmaskName, kE7Mask}});
  int count = vlen;
  for (int reg = kVd; reg < kVd + num_reg; reg++) {
    auto reg_span = tester->vreg()[reg]->data_buffer()->Get<T>();
    for (int i = 0; i < num_per_reg; i++) {
      reg_span[i] = static_cast<T>(count--);
    }
  }

  // Execute the instruction.
  inst->Execute();

  // Check the results.
  auto mask_span = tester->vreg()[kVmask]->data_buffer()->Get<T>();
  count = 0;
  for (int i = 0; i < vlen; i++) {
    int reg = kVd + i / num_per_reg;
    int reg_index = i % num_per_reg;
    auto value = tester->vreg()[reg]->data_buffer()->Get<T>(reg_index);
    int mask_index = i >> 3;
    int mask_offset = i & 0b111;
    int mask_value = (mask_span[mask_index] >> mask_offset) & 0b1;
    if (mask_value) {
      EXPECT_EQ(value, static_cast<T>(i)) << "active index: " << i;
      count++;
    } else {
      EXPECT_EQ(value, static_cast<T>(vlen - i)) << "inactive index: " << i;
    }
  }
}

// Vid instructions.
TEST_F(RiscVVectorUnaryInstructionsTest, Vid8) {
  TestVid<uint8_t>(this, instruction_);
}

TEST_F(RiscVVectorUnaryInstructionsTest, Vid16) {
  TestVid<uint16_t>(this, instruction_);
}

TEST_F(RiscVVectorUnaryInstructionsTest, Vid32) {
  TestVid<uint32_t>(this, instruction_);
}

TEST_F(RiscVVectorUnaryInstructionsTest, Vid64) {
  TestVid<uint64_t>(this, instruction_);
}
}  // namespace
