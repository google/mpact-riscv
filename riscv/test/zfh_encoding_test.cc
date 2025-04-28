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

#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/operand_interface.h"
#include "mpact/sim/generic/register.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/zfh_enums.h"
#include "testing/base/public/gunit.h"

// Test that hand crafted zfh instructions are decoded and parsed correctly.

namespace {

using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.
using ::mpact::sim::generic::ImmediateOperand;
using ::mpact::sim::generic::RegisterDestinationOperand;
using ::mpact::sim::generic::SourceOperandInterface;
using mpact::sim::riscv::RiscVState;
using mpact::sim::riscv::RiscVXlen;
using mpact::sim::riscv::RVFpRegister;
using mpact::sim::riscv::RVXRegister;
using mpact::sim::util::FlatDemandMemory;

using mpact::sim::riscv::zfh::kComplexResourceNames;
using mpact::sim::riscv::zfh::kDestOpNames;
using mpact::sim::riscv::zfh::kSourceOpNames;

using SlotEnum = mpact::sim::riscv::zfh::SlotEnum;
using OpcodeEnum = mpact::sim::riscv::zfh::OpcodeEnum;
using SourceOpEnum = mpact::sim::riscv::zfh::SourceOpEnum;
using DestOpEnum = mpact::sim::riscv::zfh::DestOpEnum;
using ComplexResourceEnum = mpact::sim::riscv::zfh::ComplexResourceEnum;

using mpact::sim::riscv::zfh::ZFHEncoding;

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

class ZfhEncodingTest : public testing::Test {
 protected:
  ZfhEncodingTest() {
    state_ = new RiscVState("test", RiscVXlen::RV32, &memory_);
    enc_ = new ZFHEncoding(state_);
  }

  ~ZfhEncodingTest() override {
    delete enc_;
    delete state_;
  }

  void ScalarRs1Helper(uint32_t, OpcodeEnum);
  void FloatFrdHelper(uint32_t, OpcodeEnum);
  void FloatSourceHelper(uint32_t, OpcodeEnum, int);
  void FloatSourceHelper(uint32_t, OpcodeEnum, SourceOpEnum, int);
  void FloatFrs1Helper(uint32_t, OpcodeEnum);
  void FloatFrs2Helper(uint32_t, OpcodeEnum);
  void FloatRmHelper(uint32_t, OpcodeEnum);

  FlatDemandMemory memory_;
  RiscVState *state_;
  ZFHEncoding *enc_;
  absl::BitGen gen_;
};

void ZfhEncodingTest::ScalarRs1Helper(uint32_t binary_instruction,
                                      OpcodeEnum opcode_enum) {
  int rs1_offset = 15;
  uint32_t base_instruction = binary_instruction & ~(0x0000'001F << rs1_offset);
  for (int rs1_index = 0; rs1_index < 32; ++rs1_index) {
    uint32_t expected_value = rs1_index ? absl::Uniform<uint32_t>(gen_) : 0;

    // Set the register value with a test value.
    RVXRegister *rs1_reg;
    std::tie(rs1_reg, std::ignore) = state_->GetRegister<RVXRegister>(
        absl::StrCat("x", static_cast<uint32_t>(rs1_index)));
    rs1_reg->data_buffer()->Set<uint32_t>(0, expected_value);

    // Parse the instruction and get the source operand.
    uint32_t rs1_adjustment = static_cast<uint32_t>(rs1_index) << rs1_offset;
    enc_->ParseInstruction(base_instruction | rs1_adjustment);
    std::unique_ptr<SourceOperandInterface> src(enc_->GetSource(
        SlotEnum::kRiscv32ZfhMin, 0, opcode_enum, SourceOpEnum::kRs1, 0));

    // Pull the value from the source operand and compare it to the expected
    // value.
    EXPECT_EQ(src->AsUint32(0), expected_value)
        << "rs1_index: " << rs1_index << ", expected_value: " << std::hex
        << expected_value << ", observed value: " << std::hex
        << src->AsUint32(0);
  }
}

void ZfhEncodingTest::FloatFrdHelper(uint32_t binary_instruction,
                                     OpcodeEnum opcode_enum) {
  int frd_offset = 7;
  uint32_t base_instruction = binary_instruction & ~(0x0000'001F << frd_offset);
  for (int frd_index = 0; frd_index < 32; ++frd_index) {
    uint64_t expected_value = absl::Uniform<uint64_t>(gen_);
    uint32_t frd_adjustment = static_cast<uint32_t>(frd_index) << frd_offset;

    // Set the register value with a test value.
    RVFpRegister *frd_reg;
    std::tie(frd_reg, std::ignore) =
        state_->GetRegister<RVFpRegister>(absl::StrCat("f", frd_index));
    frd_reg->data_buffer()->Set<uint64_t>(0, expected_value);

    // Parse the instruction and get the destination operand.
    enc_->ParseInstruction(base_instruction | frd_adjustment);
    std::unique_ptr<RegisterDestinationOperand<RVFpRegister>> dst(
        static_cast<RegisterDestinationOperand<RVFpRegister> *>(
            enc_->GetDestination(SlotEnum::kRiscv32ZfhMin, 0, opcode_enum,
                                 DestOpEnum::kFrd, 0, 0)));

    // Pull the value from the destination operand and compare it to the
    // expected value.
    uint64_t observed_value =
        dst->GetRegister()->data_buffer()->Get<uint64_t>(0);
    EXPECT_EQ(observed_value, expected_value)
        << "frd_index: " << frd_index << ", expected_value: " << std::hex
        << expected_value << ", observed value: " << std::hex << observed_value;
  }
}

void ZfhEncodingTest::FloatSourceHelper(uint32_t binary_instruction,
                                        OpcodeEnum opcode_enum,
                                        SourceOpEnum source_op_enum,
                                        int offset) {
  uint32_t base_instruction = binary_instruction & ~(0x0000'001F << offset);
  for (uint32_t frs1_index = 0; frs1_index < 32; ++frs1_index) {
    uint32_t src_adjustment = frs1_index << offset;
    uint64_t expected_value = absl::Uniform<uint64_t>(gen_);

    // Set the register value with a test value.
    RVFpRegister *frs1_reg;
    std::tie(frs1_reg, std::ignore) =
        state_->GetRegister<RVFpRegister>(absl::StrCat("f", frs1_index));
    frs1_reg->data_buffer()->Set<uint64_t>(0, expected_value);

    // Parse the instruction and get the source operand.
    enc_->ParseInstruction(base_instruction | src_adjustment);
    std::unique_ptr<SourceOperandInterface> src(enc_->GetSource(
        SlotEnum::kRiscv32ZfhMin, 0, opcode_enum, source_op_enum, 0));

    // Pull the value from the source operand and compare it to the expected
    // value.
    EXPECT_EQ(src->AsUint64(0), expected_value)
        << "frs1_index: " << frs1_index << ", expected_value: " << std::hex
        << expected_value << ", observed value: " << std::hex
        << src->AsUint64(0);
  }
}

void ZfhEncodingTest::FloatFrs1Helper(uint32_t binary_instruction,
                                      OpcodeEnum opcode_enum) {
  FloatSourceHelper(binary_instruction, opcode_enum, SourceOpEnum::kFrs1, 15);
}

void ZfhEncodingTest::FloatFrs2Helper(uint32_t binary_instruction,
                                      OpcodeEnum opcode_enum) {
  FloatSourceHelper(binary_instruction, opcode_enum, SourceOpEnum::kFrs2, 20);
}

void ZfhEncodingTest::FloatRmHelper(uint32_t binary_instruction,
                                    OpcodeEnum opcode_enum) {
  for (int rm = 0; rm <= 6; ++rm) {
    uint32_t rm_adjustment = rm << 12;
    enc_->ParseInstruction(kFcvtSh | rm_adjustment);
    std::unique_ptr<SourceOperandInterface> src(
        enc_->GetSource(SlotEnum::kRiscv32ZfhMin, 0, OpcodeEnum::kFcvtSh,
                        SourceOpEnum::kRm, 0));
    EXPECT_EQ(src->AsUint32(0), rm);
  }
}

TEST_F(ZfhEncodingTest, SourceOperands) {
  auto &getters = enc_->source_op_getters();
  for (int i = *SourceOpEnum::kNone; i < *SourceOpEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No source operand for enum value " << i
                                     << " (" << kSourceOpNames[i] << ")";
  }
}

TEST_F(ZfhEncodingTest, DestOperands) {
  auto &getters = enc_->dest_op_getters();
  for (int i = *DestOpEnum::kNone; i < *DestOpEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No dest operand for enum value " << i
                                     << " (" << kDestOpNames[i] << ")";
  }
}

// TODO(julianmb): Add simple resource getters when appropriate.
// TEST_F(ZfhEncodingTest, SimpleResources) {
//   auto &getters = enc_->simple_resource_getters();
//   for (int i = *SimpleResourceEnum::kNone;
//        i < *SimpleResourceEnum::kPastMaxValue; ++i) {
//     EXPECT_TRUE(getters.contains(i)) << "No source operand for enum value "
//     << i
//                                      << " (" << kSimpleResourceNames[i] <<
//                                      ")";
//   }
// }

TEST_F(ZfhEncodingTest, ComplexResources) {
  auto &getters = enc_->source_op_getters();
  for (int i = *ComplexResourceEnum::kNone;
       i < *ComplexResourceEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No source operand for enum value " << i
                                     << " (" << kComplexResourceNames[i] << ")";
  }
}

TEST_F(ZfhEncodingTest, Flh) {
  enc_->ParseInstruction(kFlh);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32ZfhMin, 0), OpcodeEnum::kFlh);
}

TEST_F(ZfhEncodingTest, Flh_imm12) {
  for (int iter = 0; iter < 1000; ++iter) {
    int32_t expected_imm =
        absl::Uniform<int32_t>(gen_, -1 * (1 << 11), 1 << 11);
    bool sign = expected_imm < 0;
    uint32_t imm_adjustment =
        (sign ? 0x8000'0000 : 0) | ((expected_imm & 0x0000'07FF) << 20);
    enc_->ParseInstruction(kFlh | imm_adjustment);
    std::unique_ptr<ImmediateOperand<int32_t>> src(
        static_cast<ImmediateOperand<int32_t> *>(
            enc_->GetSource(SlotEnum::kRiscv32ZfhMin, 0, OpcodeEnum::kFlh,
                            SourceOpEnum::kIImm12, 0)));
    EXPECT_EQ(src->AsInt32(0), expected_imm);
  }
}

TEST_F(ZfhEncodingTest, Flh_rs1) { ScalarRs1Helper(kFlh, OpcodeEnum::kFlh); }

TEST_F(ZfhEncodingTest, Flh_frd) { FloatFrdHelper(kFlh, OpcodeEnum::kFlh); }

TEST_F(ZfhEncodingTest, Fsh) {
  enc_->ParseInstruction(kFsh);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32ZfhMin, 0), OpcodeEnum::kFsh);
}

TEST_F(ZfhEncodingTest, Fsh_imm12) {
  for (int iter = 0; iter < 1000; ++iter) {
    int32_t expected_imm =
        absl::Uniform<int32_t>(gen_, -1 * (1 << 11), 1 << 11);
    bool sign = expected_imm < 0;
    uint32_t imm_adjustment = (sign ? 0x8000'0000 : 0) |
                              ((expected_imm & 0x0000'001F) << 7) |
                              ((expected_imm & 0x0000'07E0) << 20);
    enc_->ParseInstruction(kFsh | imm_adjustment);
    std::unique_ptr<ImmediateOperand<int32_t>> src(
        static_cast<ImmediateOperand<int32_t> *>(
            enc_->GetSource(SlotEnum::kRiscv32ZfhMin, 0, OpcodeEnum::kFsh,
                            SourceOpEnum::kSImm12, 0)));
    EXPECT_EQ(src->AsInt32(0), expected_imm);
  }
}

TEST_F(ZfhEncodingTest, Fsh_rs1) { ScalarRs1Helper(kFsh, OpcodeEnum::kFsh); }

TEST_F(ZfhEncodingTest, Fsh_frs2) { FloatFrs2Helper(kFsh, OpcodeEnum::kFsh); }

TEST_F(ZfhEncodingTest, FmvXh) {
  enc_->ParseInstruction(kFmvXh);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32ZfhMin, 0), OpcodeEnum::kFmvXh);
}

TEST_F(ZfhEncodingTest, FmvXh_frs1) {
  FloatFrs1Helper(kFmvXh, OpcodeEnum::kFmvXh);
}

TEST_F(ZfhEncodingTest, FmvXh_rd) {
  for (uint32_t rd_index = 1; rd_index < 32; ++rd_index) {
    uint32_t rd_adjustment = rd_index << 7;
    uint32_t expected_value = absl::Uniform<uint32_t>(gen_);
    RVXRegister *rd_reg;
    std::tie(rd_reg, std::ignore) =
        state_->GetRegister<RVXRegister>(absl::StrCat("x", rd_index));
    rd_reg->data_buffer()->Set<uint32_t>(0, expected_value);
    enc_->ParseInstruction(kFmvXh | rd_adjustment);
    std::unique_ptr<RegisterDestinationOperand<RVXRegister>> dst(
        static_cast<RegisterDestinationOperand<RVXRegister> *>(
            enc_->GetDestination(SlotEnum::kRiscv32ZfhMin, 0,
                                 OpcodeEnum::kFmvXh, DestOpEnum::kRd, 0, 0)));
    uint32_t observed_value =
        dst->GetRegister()->data_buffer()->Get<uint32_t>(0);
    EXPECT_EQ(observed_value, expected_value);
  }
}

TEST_F(ZfhEncodingTest, FmvHx) {
  enc_->ParseInstruction(kFmvHx);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32ZfhMin, 0), OpcodeEnum::kFmvHx);
}

TEST_F(ZfhEncodingTest, FmvHx_rs1) {
  ScalarRs1Helper(kFmvHx, OpcodeEnum::kFmvHx);
}

TEST_F(ZfhEncodingTest, FmvHx_frd) {
  FloatFrdHelper(kFmvHx, OpcodeEnum::kFmvHx);
}

TEST_F(ZfhEncodingTest, FcvtSh) {
  enc_->ParseInstruction(kFcvtSh);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32ZfhMin, 0), OpcodeEnum::kFcvtSh);
}

TEST_F(ZfhEncodingTest, FcvtSh_frs1) {
  FloatFrs1Helper(kFcvtSh, OpcodeEnum::kFcvtSh);
}

TEST_F(ZfhEncodingTest, FcvtSh_frd) {
  FloatFrdHelper(kFcvtSh, OpcodeEnum::kFcvtSh);
}

TEST_F(ZfhEncodingTest, FcvtSh_rm) {
  FloatRmHelper(kFcvtSh, OpcodeEnum::kFcvtSh);
}

TEST_F(ZfhEncodingTest, FcvtHs) {
  enc_->ParseInstruction(kFcvtHs);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32ZfhMin, 0), OpcodeEnum::kFcvtHs);
}

TEST_F(ZfhEncodingTest, FcvtHs_frs1) {
  FloatFrs1Helper(kFcvtHs, OpcodeEnum::kFcvtHs);
}

TEST_F(ZfhEncodingTest, FcvtHs_frd) {
  FloatFrdHelper(kFcvtHs, OpcodeEnum::kFcvtHs);
}

TEST_F(ZfhEncodingTest, FcvtHs_rm) {
  FloatRmHelper(kFcvtHs, OpcodeEnum::kFcvtHs);
}

TEST_F(ZfhEncodingTest, FcvtDh) {
  enc_->ParseInstruction(kFcvtDh);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32ZfhMin, 0), OpcodeEnum::kFcvtDh);
}

TEST_F(ZfhEncodingTest, FcvtDh_frs1) {
  FloatFrs1Helper(kFcvtDh, OpcodeEnum::kFcvtDh);
}

TEST_F(ZfhEncodingTest, FcvtDh_frd) {
  FloatFrdHelper(kFcvtDh, OpcodeEnum::kFcvtDh);
}

TEST_F(ZfhEncodingTest, FcvtDh_rm) {
  FloatRmHelper(kFcvtDh, OpcodeEnum::kFcvtDh);
}

TEST_F(ZfhEncodingTest, FcvtHd) {
  enc_->ParseInstruction(kFcvtHd);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32ZfhMin, 0), OpcodeEnum::kFcvtHd);
}

TEST_F(ZfhEncodingTest, FcvtHd_frs1) {
  FloatFrs1Helper(kFcvtHd, OpcodeEnum::kFcvtHd);
}

TEST_F(ZfhEncodingTest, FcvtHd_frd) {
  FloatFrdHelper(kFcvtHd, OpcodeEnum::kFcvtHd);
}

TEST_F(ZfhEncodingTest, FcvtHd_rm) {
  FloatRmHelper(kFcvtHd, OpcodeEnum::kFcvtHd);
}

}  // namespace
