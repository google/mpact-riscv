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

#include "riscv/test/zfh_encoding.h"

#include <sys/types.h>

#include <cstdint>
#include <ios>
#include <memory>
#include <tuple>
#include <vector>

#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/operand_interface.h"
#include "mpact/sim/generic/register.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/zfh32_enums.h"
#include "riscv/zfh64_enums.h"

// Test that hand crafted zfh instructions are decoded and parsed correctly.

namespace {

using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.
using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::generic::ImmediateOperand;
using ::mpact::sim::generic::RegisterDestinationOperand;
using ::mpact::sim::generic::SourceOperandInterface;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::RVFpRegister;
using ::mpact::sim::riscv::zfh::ZfhEncoding;
using ::mpact::sim::util::FlatDemandMemory;

//                           imm12      | rs1 |   | rd  | opcode
constexpr uint32_t kFlh = 0b000000000000'00000'001'00000'0000111;
//                          imm7   | rs2 | rs1 |   |imm5 | opcode
constexpr uint32_t kFsh = 0b0000000'00000'00000'001'00000'0100111;
//                            func7  |     | rs1 |fn3| rd  | opcode
constexpr uint32_t kFmvXh = 0b1110010'00000'00000'000'00000'1010011;
//                            func7  |     | rs1 |fn3| rd  | opcode
constexpr uint32_t kFmvHx = 0b1111010'00000'00000'000'00000'1010011;
//                             func7  |     | rs1 |rm | rd  | opcode
constexpr uint32_t kFcvtSh = 0b0100000'00010'00000'000'00000'1010011;
//                             func7  |     | rs1 |rm | rd  | opcode
constexpr uint32_t kFcvtHs = 0b0100010'00000'00000'000'00000'1010011;
//                             func7  |     | rs1 |rm | rd  | opcode
constexpr uint32_t kFcvtDh = 0b0100001'00010'00000'000'00000'1010011;
//                             func7  |     | rs1 |rm | rd  | opcode
constexpr uint32_t kFcvtHd = 0b0100010'00001'00000'000'00000'1010011;
//                            func7  | rs2 | rs1 |rm | rd  | opcode
constexpr uint32_t kFaddH = 0b0000010'00000'00000'000'00000'1010011;
//                            func7  | rs2 | rs1 |rm | rd  | opcode
constexpr uint32_t kFsubH = 0b0000110'00000'00000'000'00000'1010011;
//                            func7  | rs2 | rs1 |rm | rd  | opcode
constexpr uint32_t kFmulH = 0b0001010'00000'00000'000'00000'1010011;
//                            func7  | rs2 | rs1 |rm | rd  | opcode
constexpr uint32_t kFdivH = 0b0001110'00000'00000'000'00000'1010011;
//                            func7  | rs2 | rs1 |fn3| rd  | opcode
constexpr uint32_t kFminH = 0b0010110'00000'00000'000'00000'1010011;
//                            func7  | rs2 | rs1 |fn3| rd  | opcode
constexpr uint32_t kFmaxH = 0b0010110'00000'00000'001'00000'1010011;
//                             func7  | rs2 | rs1 |fn3| rd  | opcode
constexpr uint32_t kFsgnjH = 0b0010010'00000'00000'000'00000'1010011;
//                              func7  | rs2 | rs1 |fn3| rd  | opcode
constexpr uint32_t kFsgnjnH = 0b0010010'00000'00000'001'00000'1010011;
//                              func7  | rs2 | rs1 |fn3| rd  | opcode
constexpr uint32_t kFsgnjxH = 0b0010010'00000'00000'010'00000'1010011;
//                             func7  |     | rs1 |rm | rd  | opcode
constexpr uint32_t kFsqrtH = 0b0101110'00000'00000'000'00000'1010011;
//                             func7  |     | rs1 |rm | rd  | opcode
constexpr uint32_t kFcvtHw = 0b1101010'00000'00000'000'00000'1010011;
//                             func7  |     | rs1 |rm | rd  | opcode
constexpr uint32_t kFcvtWh = 0b1100010'00000'00000'000'00000'1010011;
//                              func7  |     | rs1 |rm | rd  | opcode
constexpr uint32_t kFcvtHwu = 0b1101010'00001'00000'000'00000'1010011;
//                              func7  |     | rs1 |rm | rd  | opcode
constexpr uint32_t kFcvtWuh = 0b1100010'00001'00000'000'00000'1010011;
//                              func7  | rs2 | rs1 |fn3| rd  | opcode
constexpr uint32_t kFcmpeqH = 0b1010010'00000'00000'010'00000'1010011;
//                              func7  | rs2 | rs1 |fn3| rd  | opcode
constexpr uint32_t kFcmpltH = 0b1010010'00000'00000'001'00000'1010011;
//                              func7  | rs2 | rs1 |fn3| rd  | opcode
constexpr uint32_t kFcmpleH = 0b1010010'00000'00000'000'00000'1010011;
//                              func7  |     | rs1 |fn3| rd  | opcode
constexpr uint32_t kFclassH = 0b1110010'00000'00000'001'00000'1010011;
//                             rs3  |f2| rs2 | rs1 |rm | rd  | opcode
constexpr uint32_t kFmaddH = 0b00000'10'00000'00000'000'00000'1000011;
//                             rs3  |f2| rs2 | rs1 |rm | rd  | opcode
constexpr uint32_t kFmsubH = 0b00001'10'00000'00000'000'00000'1000111;
//                              rs3  |f2| rs2 | rs1 |rm | rd  | opcode
constexpr uint32_t kFnmaddH = 0b00000'10'00000'00000'000'00000'1001111;
//                              rs3  |f2| rs2 | rs1 |rm | rd  | opcode
constexpr uint32_t kFnmsubH = 0b00000'10'00000'00000'000'00000'1001011;
//   64 bit only                func7 |     | rs1 |rm | rd  | opcode
constexpr uint32_t kFcvtLh = 0b1100010'00010'00000'000'00000'1010011;
//   64 bit only                 func7 |     | rs1 |rm | rd  | opcode
constexpr uint32_t kFcvtLuh = 0b1100010'00011'00000'000'00000'1010011;
//   64 bit only                func7 |     | rs1 |rm | rd  | opcode
constexpr uint32_t kFcvtHl = 0b1101010'00010'00000'000'00000'1010011;
//   64 bit only                 func7 |     | rs1 |rm | rd  | opcode
constexpr uint32_t kFcvtHlu = 0b1101010'00011'00000'000'00000'1010011;

struct Zfh32Config {
  using XRegister = ::mpact::sim::riscv::RV32Register;
  using XValue = typename XRegister::ValueType;
  using SlotEnum = ::mpact::sim::riscv::zfh32::SlotEnum;
  using DestOpEnum = ::mpact::sim::riscv::zfh32::DestOpEnum;
  using SourceOpEnum = ::mpact::sim::riscv::zfh32::SourceOpEnum;
  using OpcodeEnum = ::mpact::sim::riscv::zfh32::OpcodeEnum;
  using ComplexResourceEnum = ::mpact::sim::riscv::zfh32::ComplexResourceEnum;
  using SimpleResourceEnum = ::mpact::sim::riscv::zfh32::SimpleResourceEnum;
  static constexpr RiscVXlen rvXLen = RiscVXlen::RV32;
  static constexpr int kXLen = 32;
  static constexpr int slot = static_cast<int>(SlotEnum::kRiscv32Zfh);
  static constexpr const char* const* kSourceOpNames =
      ::mpact::sim::riscv::zfh32::kSourceOpNames;
  static constexpr const char* const* kDestOpNames =
      ::mpact::sim::riscv::zfh32::kDestOpNames;
  static constexpr const char* const* kComplexResourceNames =
      ::mpact::sim::riscv::zfh32::kComplexResourceNames;
};

struct Zfh64Config {
  using XRegister = ::mpact::sim::riscv::RV64Register;
  using XValue = typename XRegister::ValueType;
  using SlotEnum = ::mpact::sim::riscv::zfh64::SlotEnum;
  using DestOpEnum = ::mpact::sim::riscv::zfh64::DestOpEnum;
  using SourceOpEnum = ::mpact::sim::riscv::zfh64::SourceOpEnum;
  using OpcodeEnum = ::mpact::sim::riscv::zfh64::OpcodeEnum;
  using ComplexResourceEnum = ::mpact::sim::riscv::zfh64::ComplexResourceEnum;
  using SimpleResourceEnum = ::mpact::sim::riscv::zfh64::SimpleResourceEnum;
  static constexpr RiscVXlen rvXLen = RiscVXlen::RV64;
  static constexpr int kXLen = 64;
  static constexpr int slot = static_cast<int>(SlotEnum::kRiscv64Zfh);
  static constexpr const char* const* kSourceOpNames =
      ::mpact::sim::riscv::zfh64::kSourceOpNames;
  static constexpr const char* const* kDestOpNames =
      ::mpact::sim::riscv::zfh64::kDestOpNames;
  static constexpr const char* const* kComplexResourceNames =
      ::mpact::sim::riscv::zfh64::kComplexResourceNames;
};

template <typename ConfigT>
struct ZfhEncodingTest : public testing::Test {
 protected:
  ZfhEncodingTest() {
    state_ = new RiscVState("test", ConfigT::rvXLen, &memory_);
    enc_ = new ZfhEncoding<ConfigT::kXLen>(state_);
    expected_slot_ = static_cast<typename ConfigT::SlotEnum>(ConfigT::slot);
  }

  ~ZfhEncodingTest() override {
    delete enc_;
    delete state_;
  }

  typename ConfigT::XValue RandomizeScalarRegister(int);
  void ParseInstructionWithRs1(uint32_t, int);
  void ParseInstructionWithRd(uint32_t, int);
  typename ConfigT::XValue GetRs1SourceValue(typename ConfigT::OpcodeEnum);
  typename ConfigT::XValue GetRdDestinationValue(typename ConfigT::OpcodeEnum);
  void FloatFrdHelper(uint32_t, typename ConfigT::OpcodeEnum);
  void FloatSourceHelper(uint32_t, typename ConfigT::OpcodeEnum,
                         typename ConfigT::SourceOpEnum, int);
  void FloatFrs1Helper(uint32_t, typename ConfigT::OpcodeEnum);
  void FloatFrs2Helper(uint32_t, typename ConfigT::OpcodeEnum);
  void FloatFrs3Helper(uint32_t, typename ConfigT::OpcodeEnum);
  void FloatRmHelper(uint32_t, typename ConfigT::OpcodeEnum);
  typename ConfigT::OpcodeEnum GetOpcode();

  FlatDemandMemory memory_;
  RiscVState* state_;
  ZfhEncoding<ConfigT::kXLen>* enc_;
  absl::BitGen gen_;
  typename ConfigT::SlotEnum expected_slot_;
};

template <typename ConfigT>
typename ConfigT::OpcodeEnum ZfhEncodingTest<ConfigT>::GetOpcode() {
  return static_cast<typename ConfigT::OpcodeEnum>(
      enc_->GetOpcode(expected_slot_, 0));
}

// Puts a random value in the expected register and checks that the source
// operand returns that value.
template <typename ConfigT>
typename ConfigT::XValue ZfhEncodingTest<ConfigT>::RandomizeScalarRegister(
    int register_index) {
  using XRegister = typename ConfigT::XRegister;
  using XValue = typename ConfigT::XValue;
  XValue register_value = register_index ? absl::Uniform<XValue>(gen_) : 0;
  XRegister* rs1_reg;
  std::tie(rs1_reg, std::ignore) = state_->GetRegister<XRegister>(
      absl::StrCat("x", static_cast<uint32_t>(register_index)));
  rs1_reg->data_buffer()->template Set<XValue>(0, register_value);
  return register_value;
}

template <typename ConfigT>
void ZfhEncodingTest<ConfigT>::ParseInstructionWithRs1(
    uint32_t binary_instruction, int rs1_index) {
  int rs1_offset = 15;
  uint32_t rs1_adjustment = static_cast<uint32_t>(rs1_index) << rs1_offset;
  enc_->ParseInstruction(binary_instruction | rs1_adjustment);
}

template <typename ConfigT>
void ZfhEncodingTest<ConfigT>::ParseInstructionWithRd(
    uint32_t binary_instruction, int rd_index) {
  int rd_offset = 7;
  uint32_t rd_adjustment = static_cast<uint32_t>(rd_index) << rd_offset;
  enc_->ParseInstruction(binary_instruction | rd_adjustment);
}

template <typename ConfigT>
typename ConfigT::XValue ZfhEncodingTest<ConfigT>::GetRs1SourceValue(
    typename ConfigT::OpcodeEnum opcode_enum) {
  using XValue = typename ConfigT::XValue;
  std::unique_ptr<SourceOperandInterface> src(enc_->GetSource(
      expected_slot_, 0, opcode_enum, ConfigT::SourceOpEnum::kRs1, 0));

  XValue observed_value;
  if constexpr (ConfigT::kXLen == 32) {
    observed_value = src->AsUint32(0);
  } else {
    observed_value = src->AsUint64(0);
  }
  return observed_value;
}

template <typename ConfigT>
typename ConfigT::XValue ZfhEncodingTest<ConfigT>::GetRdDestinationValue(
    typename ConfigT::OpcodeEnum opcode_enum) {
  using XValue = typename ConfigT::XValue;
  using XRegister = typename ConfigT::XRegister;
  std::unique_ptr<RegisterDestinationOperand<XRegister>> dst(
      static_cast<RegisterDestinationOperand<XRegister>*>(enc_->GetDestination(
          expected_slot_, 0, opcode_enum, ConfigT::DestOpEnum::kRd, 0, 0)));
  std::unique_ptr<DataBuffer> db(dst->CopyDataBuffer());
  return db->template Get<XValue>(0);
}

// TODO: b/421177084 - Move EXPECT_* out of helpers and back to test body.
template <typename ConfigT>
void ZfhEncodingTest<ConfigT>::FloatFrdHelper(
    uint32_t binary_instruction, typename ConfigT::OpcodeEnum opcode_enum) {
  int frd_offset = 7;
  uint32_t base_instruction = binary_instruction & ~(0x0000'001F << frd_offset);
  for (int frd_index = 0; frd_index < 32; ++frd_index) {
    uint64_t expected_value = absl::Uniform<uint64_t>(gen_);
    uint32_t frd_adjustment = static_cast<uint32_t>(frd_index) << frd_offset;

    // Set the register value with a test value.
    RVFpRegister* frd_reg;
    std::tie(frd_reg, std::ignore) =
        state_->GetRegister<RVFpRegister>(absl::StrCat("f", frd_index));
    frd_reg->data_buffer()->Set<uint64_t>(0, expected_value);

    // Parse the instruction and get the destination operand.
    enc_->ParseInstruction(base_instruction | frd_adjustment);
    std::unique_ptr<RegisterDestinationOperand<RVFpRegister>> dst(
        static_cast<RegisterDestinationOperand<RVFpRegister>*>(
            enc_->GetDestination(expected_slot_, 0, opcode_enum,
                                 ConfigT::DestOpEnum::kFrd, 0, 0)));

    // Pull the value from the destination operand and compare it to the
    // expected value.
    uint64_t observed_value =
        dst->GetRegister()->data_buffer()->Get<uint64_t>(0);
    EXPECT_EQ(observed_value, expected_value)
        << "frd_index: " << frd_index << ", expected_value: " << std::hex
        << expected_value << ", observed value: " << std::hex << observed_value;
  }
}

// TODO: b/421177084 - Move EXPECT_* out of helpers and back to test body.
// Puts a random value in the expected register and checks that the source
// operand returns that value.
template <typename ConfigT>
void ZfhEncodingTest<ConfigT>::FloatSourceHelper(
    uint32_t binary_instruction, typename ConfigT::OpcodeEnum opcode_enum,
    typename ConfigT::SourceOpEnum source_op_enum, int offset) {
  using RegisterValue = typename RVFpRegister::ValueType;
  uint32_t base_instruction = binary_instruction & ~(0x0000'001F << offset);
  for (uint32_t frs1_index = 0; frs1_index < 32; ++frs1_index) {
    uint32_t src_adjustment = frs1_index << offset;
    RegisterValue expected_value = absl::Uniform<RegisterValue>(gen_);

    // Set the register value with a test value.
    RVFpRegister* frs1_reg;
    std::tie(frs1_reg, std::ignore) =
        state_->GetRegister<RVFpRegister>(absl::StrCat("f", frs1_index));
    frs1_reg->data_buffer()->Set<RegisterValue>(0, expected_value);

    // Parse the instruction and get the source operand.
    enc_->ParseInstruction(base_instruction | src_adjustment);
    std::unique_ptr<SourceOperandInterface> src(
        enc_->GetSource(expected_slot_, 0, opcode_enum, source_op_enum, 0));

    // Pull the value from the source operand and compare it to the expected
    // value.
    EXPECT_EQ(src->AsUint64(0), expected_value)
        << "frs1_index: " << frs1_index << ", expected_value: " << std::hex
        << expected_value << ", observed value: " << std::hex
        << src->AsUint64(0);
  }
}

template <typename ConfigT>
void ZfhEncodingTest<ConfigT>::FloatFrs1Helper(
    uint32_t binary_instruction, typename ConfigT::OpcodeEnum opcode_enum) {
  FloatSourceHelper(binary_instruction, opcode_enum,
                    ConfigT::SourceOpEnum::kFrs1, 15);
}

template <typename ConfigT>
void ZfhEncodingTest<ConfigT>::FloatFrs2Helper(
    uint32_t binary_instruction, typename ConfigT::OpcodeEnum opcode_enum) {
  FloatSourceHelper(binary_instruction, opcode_enum,
                    ConfigT::SourceOpEnum::kFrs2, 20);
}

template <typename ConfigT>
void ZfhEncodingTest<ConfigT>::FloatFrs3Helper(
    uint32_t binary_instruction, typename ConfigT::OpcodeEnum opcode_enum) {
  FloatSourceHelper(binary_instruction, opcode_enum,
                    ConfigT::SourceOpEnum::kFrs3, 27);
}

// TODO: b/421177084 - Move EXPECT_* out of helpers and back to test body.
template <typename ConfigT>
void ZfhEncodingTest<ConfigT>::FloatRmHelper(
    uint32_t binary_instruction, typename ConfigT::OpcodeEnum opcode_enum) {
  for (int rm = 0; rm <= 6; ++rm) {
    uint32_t rm_adjustment = rm << 12;
    enc_->ParseInstruction(kFcvtSh | rm_adjustment);
    std::unique_ptr<SourceOperandInterface> src(
        enc_->GetSource(expected_slot_, 0, ConfigT::OpcodeEnum::kFcvtSh,
                        ConfigT::SourceOpEnum::kRm, 0));
    EXPECT_EQ(src->AsUint32(0), rm);
  }
}

using MyTypes = ::testing::Types<Zfh32Config, Zfh64Config>;
TYPED_TEST_SUITE(ZfhEncodingTest, MyTypes);

TYPED_TEST(ZfhEncodingTest, SourceOperands) {
  auto& getters = this->enc_->source_op_getters();
  for (int i = *TypeParam::SourceOpEnum::kNone;
       i < *TypeParam::SourceOpEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i))
        << "No source operand for enum value " << i << " ("
        << TypeParam::kSourceOpNames[i] << ")";
  }
}

TYPED_TEST(ZfhEncodingTest, DestOperands) {
  auto& getters = this->enc_->dest_op_getters();
  for (int i = *TypeParam::DestOpEnum::kNone;
       i < *TypeParam::DestOpEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i))
        << "No dest operand for enum value " << i << " ("
        << TypeParam::kDestOpNames[i] << ")";
  }
}

// TODO: b/420935002 - Add simple resource getters and tests when we start using
// them.

TYPED_TEST(ZfhEncodingTest, ComplexResources) {
  auto& getters = this->enc_->source_op_getters();
  for (int i = *TypeParam::ComplexResourceEnum::kNone;
       i < *TypeParam::ComplexResourceEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i))
        << "No source operand for enum value " << i << " ("
        << TypeParam::kComplexResourceNames[i] << ")";
  }
}

TYPED_TEST(ZfhEncodingTest, Flh) {
  this->enc_->ParseInstruction(kFlh);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFlh);
}

TYPED_TEST(ZfhEncodingTest, Flh_imm12) {
  for (int iter = 0; iter < 1000; ++iter) {
    int32_t expected_imm =
        absl::Uniform<int32_t>(this->gen_, -1 * (1 << 11), 1 << 11);
    bool sign = expected_imm < 0;
    uint32_t imm_adjustment =
        (sign ? 0x8000'0000 : 0) | ((expected_imm & 0x0000'07FF) << 20);
    this->enc_->ParseInstruction(kFlh | imm_adjustment);
    std::unique_ptr<ImmediateOperand<int32_t>> src(
        static_cast<ImmediateOperand<int32_t>*>(this->enc_->GetSource(
            this->expected_slot_, 0, TypeParam::OpcodeEnum::kFlh,
            TypeParam::SourceOpEnum::kIImm12, 0)));
    EXPECT_EQ(src->AsInt32(0), expected_imm);
  }
}

TYPED_TEST(ZfhEncodingTest, Flh_rs1) {
  using XValue = typename TypeParam::XValue;
  std::vector<int> failed_scalar_sources;
  for (int register_index = 0; register_index < 32; register_index++) {
    XValue test_register_value = this->RandomizeScalarRegister(register_index);
    this->ParseInstructionWithRs1(kFlh, register_index);
    XValue source_value = this->GetRs1SourceValue(TypeParam::OpcodeEnum::kFlh);
    if (source_value != test_register_value) {
      failed_scalar_sources.push_back(register_index);
    }
  }
  EXPECT_THAT(failed_scalar_sources, ::testing::IsEmpty());
}

TYPED_TEST(ZfhEncodingTest, Flh_frd) {
  this->FloatFrdHelper(kFlh, TypeParam::OpcodeEnum::kFlh);
}

TYPED_TEST(ZfhEncodingTest, Fsh) {
  this->enc_->ParseInstruction(kFsh);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFsh);
}

TYPED_TEST(ZfhEncodingTest, Fsh_imm12) {
  for (int iter = 0; iter < 1000; ++iter) {
    int32_t expected_imm =
        absl::Uniform<int32_t>(this->gen_, -1 * (1 << 11), 1 << 11);
    bool sign = expected_imm < 0;
    uint32_t imm_adjustment = (sign ? 0x8000'0000 : 0) |
                              ((expected_imm & 0x0000'001F) << 7) |
                              ((expected_imm & 0x0000'07E0) << 20);
    this->enc_->ParseInstruction(kFsh | imm_adjustment);
    std::unique_ptr<ImmediateOperand<int32_t>> src(
        static_cast<ImmediateOperand<int32_t>*>(this->enc_->GetSource(
            this->expected_slot_, 0, TypeParam::OpcodeEnum::kFsh,
            TypeParam::SourceOpEnum::kSImm12, 0)));
    EXPECT_EQ(src->AsInt32(0), expected_imm);
  }
}

TYPED_TEST(ZfhEncodingTest, Fsh_rs1) {
  using XValue = typename TypeParam::XValue;
  std::vector<int> failed_scalar_sources;
  for (int register_index = 0; register_index < 32; register_index++) {
    XValue test_register_value = this->RandomizeScalarRegister(register_index);
    this->ParseInstructionWithRs1(kFsh, register_index);
    XValue source_value = this->GetRs1SourceValue(TypeParam::OpcodeEnum::kFsh);
    if (source_value != test_register_value) {
      failed_scalar_sources.push_back(register_index);
    }
  }
  EXPECT_THAT(failed_scalar_sources, ::testing::IsEmpty());
}

TYPED_TEST(ZfhEncodingTest, Fsh_frs2) {
  this->FloatFrs2Helper(kFsh, TypeParam::OpcodeEnum::kFsh);
}

TYPED_TEST(ZfhEncodingTest, FmvXh) {
  this->enc_->ParseInstruction(kFmvXh);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFmvXh);
}

TYPED_TEST(ZfhEncodingTest, FmvXh_frs1) {
  this->FloatFrs1Helper(kFmvXh, TypeParam::OpcodeEnum::kFmvXh);
}

TYPED_TEST(ZfhEncodingTest, FmvXh_rd) {
  using XRegister = typename TypeParam::XRegister;
  using XValue = typename TypeParam::XRegister::ValueType;
  for (uint32_t rd_index = 1; rd_index < 32; ++rd_index) {
    uint32_t rd_adjustment = rd_index << 7;
    XValue expected_value = absl::Uniform<XValue>(this->gen_);
    XRegister* rd_reg;
    std::tie(rd_reg, std::ignore) =
        this->state_->template GetRegister<XRegister>(
            absl::StrCat("x", rd_index));
    rd_reg->data_buffer()->template Set<XValue>(0, expected_value);
    this->enc_->ParseInstruction(kFmvXh | rd_adjustment);
    std::unique_ptr<RegisterDestinationOperand<XRegister>> dst(
        static_cast<RegisterDestinationOperand<XRegister>*>(
            this->enc_->GetDestination(this->expected_slot_, 0,
                                       TypeParam::OpcodeEnum::kFmvXh,
                                       TypeParam::DestOpEnum::kRd, 0, 0)));
    XValue observed_value =
        dst->GetRegister()->data_buffer()->template Get<XValue>(0);
    EXPECT_EQ(observed_value, expected_value);
  }
}

TYPED_TEST(ZfhEncodingTest, FmvHx) {
  this->enc_->ParseInstruction(kFmvHx);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFmvHx);
}

TYPED_TEST(ZfhEncodingTest, FmvHx_rs1) {
  using XValue = typename TypeParam::XValue;
  std::vector<int> failed_scalar_sources;
  for (int register_index = 0; register_index < 32; register_index++) {
    XValue test_register_value = this->RandomizeScalarRegister(register_index);
    this->ParseInstructionWithRs1(kFmvHx, register_index);
    XValue source_value =
        this->GetRs1SourceValue(TypeParam::OpcodeEnum::kFmvHx);
    if (source_value != test_register_value) {
      failed_scalar_sources.push_back(register_index);
    }
  }
  EXPECT_THAT(failed_scalar_sources, ::testing::IsEmpty());
}

TYPED_TEST(ZfhEncodingTest, FmvHx_frd) {
  this->FloatFrdHelper(kFmvHx, TypeParam::OpcodeEnum::kFmvHx);
}

TYPED_TEST(ZfhEncodingTest, FcvtSh) {
  this->enc_->ParseInstruction(kFcvtSh);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFcvtSh);
}

TYPED_TEST(ZfhEncodingTest, FcvtSh_frs1) {
  this->FloatFrs1Helper(kFcvtSh, TypeParam::OpcodeEnum::kFcvtSh);
}

TYPED_TEST(ZfhEncodingTest, FcvtSh_frd) {
  this->FloatFrdHelper(kFcvtSh, TypeParam::OpcodeEnum::kFcvtSh);
}

TYPED_TEST(ZfhEncodingTest, FcvtSh_rm) {
  this->FloatRmHelper(kFcvtSh, TypeParam::OpcodeEnum::kFcvtSh);
}

TYPED_TEST(ZfhEncodingTest, FcvtHs) {
  this->enc_->ParseInstruction(kFcvtHs);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFcvtHs);
}

TYPED_TEST(ZfhEncodingTest, FcvtHs_frs1) {
  this->FloatFrs1Helper(kFcvtHs, TypeParam::OpcodeEnum::kFcvtHs);
}

TYPED_TEST(ZfhEncodingTest, FcvtHs_frd) {
  this->FloatFrdHelper(kFcvtHs, TypeParam::OpcodeEnum::kFcvtHs);
}

TYPED_TEST(ZfhEncodingTest, FcvtHs_rm) {
  this->FloatRmHelper(kFcvtHs, TypeParam::OpcodeEnum::kFcvtHs);
}

TYPED_TEST(ZfhEncodingTest, FcvtDh) {
  this->enc_->ParseInstruction(kFcvtDh);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFcvtDh);
}

TYPED_TEST(ZfhEncodingTest, FcvtDh_frs1) {
  this->FloatFrs1Helper(kFcvtDh, TypeParam::OpcodeEnum::kFcvtDh);
}

TYPED_TEST(ZfhEncodingTest, FcvtDh_frd) {
  this->FloatFrdHelper(kFcvtDh, TypeParam::OpcodeEnum::kFcvtDh);
}

TYPED_TEST(ZfhEncodingTest, FcvtDh_rm) {
  this->FloatRmHelper(kFcvtDh, TypeParam::OpcodeEnum::kFcvtDh);
}

TYPED_TEST(ZfhEncodingTest, FcvtHd) {
  this->enc_->ParseInstruction(kFcvtHd);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFcvtHd);
}

TYPED_TEST(ZfhEncodingTest, FcvtHd_frs1) {
  this->FloatFrs1Helper(kFcvtHd, TypeParam::OpcodeEnum::kFcvtHd);
}

TYPED_TEST(ZfhEncodingTest, FcvtHd_frd) {
  this->FloatFrdHelper(kFcvtHd, TypeParam::OpcodeEnum::kFcvtHd);
}

TYPED_TEST(ZfhEncodingTest, FcvtHd_rm) {
  this->FloatRmHelper(kFcvtHd, TypeParam::OpcodeEnum::kFcvtHd);
}

TYPED_TEST(ZfhEncodingTest, FaddH) {
  this->enc_->ParseInstruction(kFaddH);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFaddH);
}

TYPED_TEST(ZfhEncodingTest, FaddH_frs1) {
  this->FloatFrs1Helper(kFaddH, TypeParam::OpcodeEnum::kFaddH);
}

TYPED_TEST(ZfhEncodingTest, FaddH_frs2) {
  this->FloatFrs2Helper(kFaddH, TypeParam::OpcodeEnum::kFaddH);
}

TYPED_TEST(ZfhEncodingTest, FaddH_frd) {
  this->FloatFrdHelper(kFaddH, TypeParam::OpcodeEnum::kFaddH);
}

TYPED_TEST(ZfhEncodingTest, FaddH_rm) {
  this->FloatRmHelper(kFaddH, TypeParam::OpcodeEnum::kFaddH);
}

TYPED_TEST(ZfhEncodingTest, FsubH) {
  this->enc_->ParseInstruction(kFsubH);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFsubH);
}

TYPED_TEST(ZfhEncodingTest, FsubH_frs1) {
  this->FloatFrs1Helper(kFsubH, TypeParam::OpcodeEnum::kFsubH);
}

TYPED_TEST(ZfhEncodingTest, FsubH_frs2) {
  this->FloatFrs2Helper(kFsubH, TypeParam::OpcodeEnum::kFsubH);
}

TYPED_TEST(ZfhEncodingTest, FsubH_frd) {
  this->FloatFrdHelper(kFsubH, TypeParam::OpcodeEnum::kFsubH);
}

TYPED_TEST(ZfhEncodingTest, FsubH_rm) {
  this->FloatRmHelper(kFsubH, TypeParam::OpcodeEnum::kFsubH);
}

TYPED_TEST(ZfhEncodingTest, FmulH) {
  this->enc_->ParseInstruction(kFmulH);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFmulH);
}

TYPED_TEST(ZfhEncodingTest, FmulH_frs1) {
  this->FloatFrs1Helper(kFmulH, TypeParam::OpcodeEnum::kFmulH);
}

TYPED_TEST(ZfhEncodingTest, FmulH_frs2) {
  this->FloatFrs2Helper(kFmulH, TypeParam::OpcodeEnum::kFmulH);
}

TYPED_TEST(ZfhEncodingTest, FmulH_frd) {
  this->FloatFrdHelper(kFmulH, TypeParam::OpcodeEnum::kFmulH);
}

TYPED_TEST(ZfhEncodingTest, FmulH_rm) {
  this->FloatRmHelper(kFmulH, TypeParam::OpcodeEnum::kFmulH);
}

TYPED_TEST(ZfhEncodingTest, FdivH) {
  this->enc_->ParseInstruction(kFdivH);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFdivH);
}

TYPED_TEST(ZfhEncodingTest, FdivH_frs1) {
  this->FloatFrs1Helper(kFdivH, TypeParam::OpcodeEnum::kFdivH);
}

TYPED_TEST(ZfhEncodingTest, FdivH_frs2) {
  this->FloatFrs2Helper(kFdivH, TypeParam::OpcodeEnum::kFdivH);
}

TYPED_TEST(ZfhEncodingTest, FdivH_frd) {
  this->FloatFrdHelper(kFdivH, TypeParam::OpcodeEnum::kFdivH);
}

TYPED_TEST(ZfhEncodingTest, FdivH_rm) {
  this->FloatRmHelper(kFdivH, TypeParam::OpcodeEnum::kFdivH);
}

TYPED_TEST(ZfhEncodingTest, FminH) {
  this->enc_->ParseInstruction(kFminH);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFminH);
}

TYPED_TEST(ZfhEncodingTest, FminH_frs1) {
  this->FloatFrs1Helper(kFminH, TypeParam::OpcodeEnum::kFminH);
}

TYPED_TEST(ZfhEncodingTest, FminH_frs2) {
  this->FloatFrs2Helper(kFminH, TypeParam::OpcodeEnum::kFminH);
}

TYPED_TEST(ZfhEncodingTest, FminH_frd) {
  this->FloatFrdHelper(kFminH, TypeParam::OpcodeEnum::kFminH);
}

TYPED_TEST(ZfhEncodingTest, FmaxH) {
  this->enc_->ParseInstruction(kFmaxH);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFmaxH);
}

TYPED_TEST(ZfhEncodingTest, FmaxH_frs1) {
  this->FloatFrs1Helper(kFmaxH, TypeParam::OpcodeEnum::kFmaxH);
}

TYPED_TEST(ZfhEncodingTest, FmaxH_frs2) {
  this->FloatFrs2Helper(kFmaxH, TypeParam::OpcodeEnum::kFmaxH);
}

TYPED_TEST(ZfhEncodingTest, FmaxH_frd) {
  this->FloatFrdHelper(kFmaxH, TypeParam::OpcodeEnum::kFmaxH);
}

TYPED_TEST(ZfhEncodingTest, FsgnjH) {
  this->enc_->ParseInstruction(kFsgnjH);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFsgnjH);
}

TYPED_TEST(ZfhEncodingTest, FsgnjH_frs1) {
  this->FloatFrs1Helper(kFsgnjH, TypeParam::OpcodeEnum::kFsgnjH);
}

TYPED_TEST(ZfhEncodingTest, FsgnjH_frs2) {
  this->FloatFrs2Helper(kFsgnjH, TypeParam::OpcodeEnum::kFsgnjH);
}

TYPED_TEST(ZfhEncodingTest, FsgnjH_frd) {
  this->FloatFrdHelper(kFsgnjH, TypeParam::OpcodeEnum::kFsgnjH);
}

TYPED_TEST(ZfhEncodingTest, FsgnjnH) {
  this->enc_->ParseInstruction(kFsgnjnH);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFsgnjnH);
}

TYPED_TEST(ZfhEncodingTest, FsgnjnH_frs1) {
  this->FloatFrs1Helper(kFsgnjnH, TypeParam::OpcodeEnum::kFsgnjnH);
}

TYPED_TEST(ZfhEncodingTest, FsgnjnH_frs2) {
  this->FloatFrs2Helper(kFsgnjnH, TypeParam::OpcodeEnum::kFsgnjnH);
}

TYPED_TEST(ZfhEncodingTest, FsgnjnH_frd) {
  this->FloatFrdHelper(kFsgnjnH, TypeParam::OpcodeEnum::kFsgnjnH);
}

TYPED_TEST(ZfhEncodingTest, FsgnjxH) {
  this->enc_->ParseInstruction(kFsgnjxH);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFsgnjxH);
}

TYPED_TEST(ZfhEncodingTest, FsgnjxH_frs1) {
  this->FloatFrs1Helper(kFsgnjxH, TypeParam::OpcodeEnum::kFsgnjxH);
}

TYPED_TEST(ZfhEncodingTest, FsgnjxH_frs2) {
  this->FloatFrs2Helper(kFsgnjxH, TypeParam::OpcodeEnum::kFsgnjxH);
}

TYPED_TEST(ZfhEncodingTest, FsgnjxH_frd) {
  this->FloatFrdHelper(kFsgnjxH, TypeParam::OpcodeEnum::kFsgnjxH);
}

TYPED_TEST(ZfhEncodingTest, FsqrtH) {
  this->enc_->ParseInstruction(kFsqrtH);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFsqrtH);
}

TYPED_TEST(ZfhEncodingTest, FsqrtH_frs1) {
  this->FloatFrs1Helper(kFsqrtH, TypeParam::OpcodeEnum::kFsqrtH);
}

TYPED_TEST(ZfhEncodingTest, FsqrtH_rm) {
  this->FloatRmHelper(kFsqrtH, TypeParam::OpcodeEnum::kFsqrtH);
}

TYPED_TEST(ZfhEncodingTest, FsqrtH_frd) {
  this->FloatFrdHelper(kFsqrtH, TypeParam::OpcodeEnum::kFsqrtH);
}

TYPED_TEST(ZfhEncodingTest, FcvtHw) {
  this->enc_->ParseInstruction(kFcvtHw);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFcvtHw);
}

TYPED_TEST(ZfhEncodingTest, FcvtHw_rs1) {
  using XValue = typename TypeParam::XValue;
  std::vector<int> failed_scalar_sources;
  for (int register_index = 0; register_index < 32; register_index++) {
    XValue test_register_value = this->RandomizeScalarRegister(register_index);
    this->ParseInstructionWithRs1(kFcvtHw, register_index);
    XValue source_value =
        this->GetRs1SourceValue(TypeParam::OpcodeEnum::kFcvtHw);
    if (source_value != test_register_value) {
      failed_scalar_sources.push_back(register_index);
    }
  }
  EXPECT_THAT(failed_scalar_sources, ::testing::IsEmpty());
}

TYPED_TEST(ZfhEncodingTest, FcvtHw_rm) {
  this->FloatRmHelper(kFcvtHw, TypeParam::OpcodeEnum::kFcvtHw);
}

TYPED_TEST(ZfhEncodingTest, FcvtHw_frd) {
  this->FloatFrdHelper(kFcvtHw, TypeParam::OpcodeEnum::kFcvtHw);
}

TYPED_TEST(ZfhEncodingTest, FcvtWh) {
  this->enc_->ParseInstruction(kFcvtWh);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFcvtWh);
}

TYPED_TEST(ZfhEncodingTest, FcvtWh_frs1) {
  this->FloatFrs1Helper(kFcvtWh, TypeParam::OpcodeEnum::kFcvtWh);
}

TYPED_TEST(ZfhEncodingTest, FcvtWh_rm) {
  this->FloatRmHelper(kFcvtWh, TypeParam::OpcodeEnum::kFcvtWh);
}

TYPED_TEST(ZfhEncodingTest, FcvtWh_rd) {
  using XValue = typename TypeParam::XValue;
  std::vector<int> failed_scalar_destinations;
  for (int register_index = 1; register_index < 32; register_index++) {
    XValue test_register_value = this->RandomizeScalarRegister(register_index);
    this->ParseInstructionWithRd(kFcvtWh, register_index);
    XValue destination_value =
        this->GetRdDestinationValue(TypeParam::OpcodeEnum::kFcvtWh);
    if (destination_value != test_register_value) {
      failed_scalar_destinations.push_back(register_index);
    }
  }
  EXPECT_THAT(failed_scalar_destinations, ::testing::IsEmpty());
}

TYPED_TEST(ZfhEncodingTest, FcvtHwu) {
  this->enc_->ParseInstruction(kFcvtHwu);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFcvtHwu);
}

TYPED_TEST(ZfhEncodingTest, FcvtHwu_rs1) {
  using XValue = typename TypeParam::XValue;
  std::vector<int> failed_scalar_sources;
  for (int register_index = 0; register_index < 32; register_index++) {
    XValue test_register_value = this->RandomizeScalarRegister(register_index);
    this->ParseInstructionWithRs1(kFcvtHwu, register_index);
    XValue source_value =
        this->GetRs1SourceValue(TypeParam::OpcodeEnum::kFcvtHwu);
    if (source_value != test_register_value) {
      failed_scalar_sources.push_back(register_index);
    }
  }
  EXPECT_THAT(failed_scalar_sources, ::testing::IsEmpty());
}

TYPED_TEST(ZfhEncodingTest, FcvtHwu_rm) {
  this->FloatRmHelper(kFcvtHwu, TypeParam::OpcodeEnum::kFcvtHwu);
}

TYPED_TEST(ZfhEncodingTest, FcvtHwu_frd) {
  this->FloatFrdHelper(kFcvtHwu, TypeParam::OpcodeEnum::kFcvtHwu);
}

TYPED_TEST(ZfhEncodingTest, FcvtWuh) {
  this->enc_->ParseInstruction(kFcvtWuh);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFcvtWuh);
}

TYPED_TEST(ZfhEncodingTest, FcvtWuh_frs1) {
  this->FloatFrs1Helper(kFcvtWuh, TypeParam::OpcodeEnum::kFcvtWuh);
}

TYPED_TEST(ZfhEncodingTest, FcvtWuh_rm) {
  this->FloatRmHelper(kFcvtWuh, TypeParam::OpcodeEnum::kFcvtWuh);
}

TYPED_TEST(ZfhEncodingTest, FcvtWuh_rd) {
  using XValue = typename TypeParam::XValue;
  std::vector<int> failed_scalar_destinations;
  for (int register_index = 1; register_index < 32; register_index++) {
    XValue test_register_value = this->RandomizeScalarRegister(register_index);
    this->ParseInstructionWithRd(kFcvtWuh, register_index);
    XValue destination_value =
        this->GetRdDestinationValue(TypeParam::OpcodeEnum::kFcvtWuh);
    if (destination_value != test_register_value) {
      failed_scalar_destinations.push_back(register_index);
    }
  }
  EXPECT_THAT(failed_scalar_destinations, ::testing::IsEmpty());
}

TYPED_TEST(ZfhEncodingTest, FcmpeqH) {
  this->enc_->ParseInstruction(kFcmpeqH);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFcmpeqH);
}

TYPED_TEST(ZfhEncodingTest, FcmpeqH_frs1) {
  this->FloatFrs1Helper(kFcmpeqH, TypeParam::OpcodeEnum::kFcmpeqH);
}

TYPED_TEST(ZfhEncodingTest, FcmpeqH_frs2) {
  this->FloatFrs2Helper(kFcmpeqH, TypeParam::OpcodeEnum::kFcmpeqH);
}

TYPED_TEST(ZfhEncodingTest, FcmpeqH_frd) {
  this->FloatFrdHelper(kFcmpeqH, TypeParam::OpcodeEnum::kFcmpeqH);
}

TYPED_TEST(ZfhEncodingTest, FcmpltH) {
  this->enc_->ParseInstruction(kFcmpltH);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFcmpltH);
}

TYPED_TEST(ZfhEncodingTest, FcmpltH_frs1) {
  this->FloatFrs1Helper(kFcmpltH, TypeParam::OpcodeEnum::kFcmpltH);
}

TYPED_TEST(ZfhEncodingTest, FcmpltH_frs2) {
  this->FloatFrs2Helper(kFcmpltH, TypeParam::OpcodeEnum::kFcmpltH);
}

TYPED_TEST(ZfhEncodingTest, FcmpltH_frd) {
  this->FloatFrdHelper(kFcmpltH, TypeParam::OpcodeEnum::kFcmpltH);
}

TYPED_TEST(ZfhEncodingTest, FcmpleH) {
  this->enc_->ParseInstruction(kFcmpleH);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFcmpleH);
}

TYPED_TEST(ZfhEncodingTest, FcmpleH_frs1) {
  this->FloatFrs1Helper(kFcmpleH, TypeParam::OpcodeEnum::kFcmpleH);
}

TYPED_TEST(ZfhEncodingTest, FcmpleH_frs2) {
  this->FloatFrs2Helper(kFcmpleH, TypeParam::OpcodeEnum::kFcmpleH);
}

TYPED_TEST(ZfhEncodingTest, FcmpleH_frd) {
  this->FloatFrdHelper(kFcmpleH, TypeParam::OpcodeEnum::kFcmpleH);
}

TYPED_TEST(ZfhEncodingTest, FclassH) {
  this->enc_->ParseInstruction(kFclassH);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFclassH);
}

TYPED_TEST(ZfhEncodingTest, FclassH_frs1) {
  this->FloatFrs1Helper(kFclassH, TypeParam::OpcodeEnum::kFclassH);
}

TYPED_TEST(ZfhEncodingTest, FclassH_frd) {
  this->FloatFrdHelper(kFclassH, TypeParam::OpcodeEnum::kFclassH);
}

TYPED_TEST(ZfhEncodingTest, FmaddH) {
  this->enc_->ParseInstruction(kFmaddH);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFmaddH);
}

TYPED_TEST(ZfhEncodingTest, FmaddH_frs1) {
  this->FloatFrs1Helper(kFmaddH, TypeParam::OpcodeEnum::kFmaddH);
}

TYPED_TEST(ZfhEncodingTest, FmaddH_frs2) {
  this->FloatFrs2Helper(kFmaddH, TypeParam::OpcodeEnum::kFmaddH);
}

TYPED_TEST(ZfhEncodingTest, FmaddH_frs3) {
  this->FloatFrs3Helper(kFmaddH, TypeParam::OpcodeEnum::kFmaddH);
}

TYPED_TEST(ZfhEncodingTest, FmaddH_rm) {
  this->FloatRmHelper(kFmaddH, TypeParam::OpcodeEnum::kFmaddH);
}

TYPED_TEST(ZfhEncodingTest, FmaddH_frd) {
  this->FloatFrdHelper(kFmaddH, TypeParam::OpcodeEnum::kFmaddH);
}

TYPED_TEST(ZfhEncodingTest, FmsubH) {
  this->enc_->ParseInstruction(kFmsubH);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFmsubH);
}

TYPED_TEST(ZfhEncodingTest, FmsubH_frs1) {
  this->FloatFrs1Helper(kFmsubH, TypeParam::OpcodeEnum::kFmsubH);
}

TYPED_TEST(ZfhEncodingTest, FmsubH_frs2) {
  this->FloatFrs2Helper(kFmsubH, TypeParam::OpcodeEnum::kFmsubH);
}

TYPED_TEST(ZfhEncodingTest, FmsubH_frs3) {
  this->FloatFrs3Helper(kFmsubH, TypeParam::OpcodeEnum::kFmsubH);
}

TYPED_TEST(ZfhEncodingTest, FmsubH_rm) {
  this->FloatRmHelper(kFmsubH, TypeParam::OpcodeEnum::kFmsubH);
}

TYPED_TEST(ZfhEncodingTest, FmsubH_frd) {
  this->FloatFrdHelper(kFmsubH, TypeParam::OpcodeEnum::kFmsubH);
}

TYPED_TEST(ZfhEncodingTest, FnmaddH) {
  this->enc_->ParseInstruction(kFnmaddH);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFnmaddH);
}

TYPED_TEST(ZfhEncodingTest, FnmaddH_frs1) {
  this->FloatFrs1Helper(kFnmaddH, TypeParam::OpcodeEnum::kFnmaddH);
}

TYPED_TEST(ZfhEncodingTest, FnmaddH_frs2) {
  this->FloatFrs2Helper(kFnmaddH, TypeParam::OpcodeEnum::kFnmaddH);
}

TYPED_TEST(ZfhEncodingTest, FnmaddH_frs3) {
  this->FloatFrs3Helper(kFnmaddH, TypeParam::OpcodeEnum::kFnmaddH);
}

TYPED_TEST(ZfhEncodingTest, FnmaddH_rm) {
  this->FloatRmHelper(kFnmaddH, TypeParam::OpcodeEnum::kFnmaddH);
}

TYPED_TEST(ZfhEncodingTest, FnmaddH_frd) {
  this->FloatFrdHelper(kFnmaddH, TypeParam::OpcodeEnum::kFnmaddH);
}

TYPED_TEST(ZfhEncodingTest, FnmsubH) {
  this->enc_->ParseInstruction(kFnmsubH);
  EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFnmsubH);
}

TYPED_TEST(ZfhEncodingTest, FnmsubH_frs1) {
  this->FloatFrs1Helper(kFnmsubH, TypeParam::OpcodeEnum::kFnmsubH);
}

TYPED_TEST(ZfhEncodingTest, FnmsubH_frs2) {
  this->FloatFrs2Helper(kFnmsubH, TypeParam::OpcodeEnum::kFnmsubH);
}

TYPED_TEST(ZfhEncodingTest, FnmsubH_frs3) {
  this->FloatFrs3Helper(kFnmsubH, TypeParam::OpcodeEnum::kFnmsubH);
}

TYPED_TEST(ZfhEncodingTest, FnmsubH_rm) {
  this->FloatRmHelper(kFnmsubH, TypeParam::OpcodeEnum::kFnmsubH);
}

TYPED_TEST(ZfhEncodingTest, FnmsubH_frd) {
  this->FloatFrdHelper(kFnmsubH, TypeParam::OpcodeEnum::kFnmsubH);
}

TYPED_TEST(ZfhEncodingTest, FcvtLh) {
  if constexpr (TypeParam::kXLen == 64) {
    this->enc_->ParseInstruction(kFcvtLh);
    EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFcvtLh);
  }
}

TYPED_TEST(ZfhEncodingTest, FcvtLh_frs1) {
  if constexpr (TypeParam::kXLen == 64) {
    this->FloatFrs1Helper(kFcvtLh, TypeParam::OpcodeEnum::kFcvtLh);
  }
}

TYPED_TEST(ZfhEncodingTest, FcvtLh_rm) {
  if constexpr (TypeParam::kXLen == 64) {
    this->FloatRmHelper(kFcvtLh, TypeParam::OpcodeEnum::kFcvtLh);
  }
}

TYPED_TEST(ZfhEncodingTest, FcvtLh_rd) {
  using XValue = typename TypeParam::XValue;
  if constexpr (TypeParam::kXLen == 64) {
    std::vector<int> failed_scalar_destinations;
    for (int register_index = 1; register_index < 32; register_index++) {
      XValue test_register_value =
          this->RandomizeScalarRegister(register_index);
      this->ParseInstructionWithRd(kFcvtLh, register_index);
      XValue destination_value =
          this->GetRdDestinationValue(TypeParam::OpcodeEnum::kFcvtLh);
      if (destination_value != test_register_value) {
        failed_scalar_destinations.push_back(register_index);
      }
    }
    EXPECT_THAT(failed_scalar_destinations, ::testing::IsEmpty());
  }
}

TYPED_TEST(ZfhEncodingTest, FcvtLuh) {
  if constexpr (TypeParam::kXLen == 64) {
    this->enc_->ParseInstruction(kFcvtLuh);
    EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFcvtLuh);
  }
}

TYPED_TEST(ZfhEncodingTest, FcvtLuh_frs1) {
  if constexpr (TypeParam::kXLen == 64) {
    this->FloatFrs1Helper(kFcvtLuh, TypeParam::OpcodeEnum::kFcvtLuh);
  }
}

TYPED_TEST(ZfhEncodingTest, FcvtLuh_rm) {
  if constexpr (TypeParam::kXLen == 64) {
    this->FloatRmHelper(kFcvtLuh, TypeParam::OpcodeEnum::kFcvtLuh);
  }
}

TYPED_TEST(ZfhEncodingTest, FcvtLuh_rd) {
  using XValue = typename TypeParam::XValue;
  if constexpr (TypeParam::kXLen == 64) {
    std::vector<int> failed_scalar_destinations;
    for (int register_index = 1; register_index < 32; register_index++) {
      XValue test_register_value =
          this->RandomizeScalarRegister(register_index);
      this->ParseInstructionWithRd(kFcvtLuh, register_index);
      XValue destination_value =
          this->GetRdDestinationValue(TypeParam::OpcodeEnum::kFcvtLuh);
      if (destination_value != test_register_value) {
        failed_scalar_destinations.push_back(register_index);
      }
    }
    EXPECT_THAT(failed_scalar_destinations, ::testing::IsEmpty());
  }
}

TYPED_TEST(ZfhEncodingTest, FcvtHl) {
  if constexpr (TypeParam::kXLen == 64) {
    this->enc_->ParseInstruction(kFcvtHl);
    EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFcvtHl);
  }
}

TYPED_TEST(ZfhEncodingTest, FcvtHl_rs1) {
  using XValue = typename TypeParam::XValue;
  if constexpr (TypeParam::kXLen == 64) {
    std::vector<int> failed_scalar_sources;
    for (int register_index = 0; register_index < 32; register_index++) {
      XValue test_register_value =
          this->RandomizeScalarRegister(register_index);
      this->ParseInstructionWithRs1(kFcvtHl, register_index);
      XValue source_value =
          this->GetRs1SourceValue(TypeParam::OpcodeEnum::kFcvtHl);
      if (source_value != test_register_value) {
        failed_scalar_sources.push_back(register_index);
      }
    }
    EXPECT_THAT(failed_scalar_sources, ::testing::IsEmpty());
  }
}

TYPED_TEST(ZfhEncodingTest, FcvtHl_rm) {
  if constexpr (TypeParam::kXLen == 64) {
    this->FloatRmHelper(kFcvtHl, TypeParam::OpcodeEnum::kFcvtHl);
  }
}

TYPED_TEST(ZfhEncodingTest, FcvtHl_frd) {
  if constexpr (TypeParam::kXLen == 64) {
    this->FloatFrdHelper(kFcvtHl, TypeParam::OpcodeEnum::kFcvtHl);
  }
}

TYPED_TEST(ZfhEncodingTest, FcvtHlu) {
  if constexpr (TypeParam::kXLen == 64) {
    this->enc_->ParseInstruction(kFcvtHlu);
    EXPECT_EQ(this->GetOpcode(), TypeParam::OpcodeEnum::kFcvtHlu);
  }
}

TYPED_TEST(ZfhEncodingTest, FcvtHlu_rs1) {
  using XValue = typename TypeParam::XValue;
  if constexpr (TypeParam::kXLen == 64) {
    std::vector<int> failed_scalar_sources;
    for (int register_index = 0; register_index < 32; register_index++) {
      XValue test_register_value =
          this->RandomizeScalarRegister(register_index);
      this->ParseInstructionWithRs1(kFcvtHlu, register_index);
      XValue source_value =
          this->GetRs1SourceValue(TypeParam::OpcodeEnum::kFcvtHlu);
      if (source_value != test_register_value) {
        failed_scalar_sources.push_back(register_index);
      }
    }
    EXPECT_THAT(failed_scalar_sources, ::testing::IsEmpty());
  }
}

TYPED_TEST(ZfhEncodingTest, FcvtHlu_rm) {
  if constexpr (TypeParam::kXLen == 64) {
    this->FloatRmHelper(kFcvtHlu, TypeParam::OpcodeEnum::kFcvtHlu);
  }
}

TYPED_TEST(ZfhEncodingTest, FcvtHlu_frd) {
  if constexpr (TypeParam::kXLen == 64) {
    this->FloatFrdHelper(kFcvtHlu, TypeParam::OpcodeEnum::kFcvtHlu);
  }
}

}  // namespace
