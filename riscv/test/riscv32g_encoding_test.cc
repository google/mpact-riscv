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

#include "riscv/riscv32g_encoding.h"

#include <cstdint>

#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "riscv/riscv32g_enums.h"
#include "riscv/riscv_state.h"

namespace {

using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.

using mpact::sim::riscv::RiscVState;
using mpact::sim::riscv::RiscVXlen;
using mpact::sim::riscv::isa32::ComplexResourceEnum;
using mpact::sim::riscv::isa32::DestOpEnum;
using mpact::sim::riscv::isa32::kComplexResourceNames;
using mpact::sim::riscv::isa32::kDestOpNames;
using mpact::sim::riscv::isa32::kSimpleResourceNames;
using mpact::sim::riscv::isa32::kSourceOpNames;
using mpact::sim::riscv::isa32::RiscV32GEncoding;
using mpact::sim::riscv::isa32::SimpleResourceEnum;
using mpact::sim::riscv::isa32::SourceOpEnum;
using mpact::sim::util::FlatDemandMemory;
using SlotEnum = mpact::sim::riscv::isa32::SlotEnum;
using OpcodeEnum = mpact::sim::riscv::isa32::OpcodeEnum;

// Constexpr for opcodes for RV32G instructions grouped by isa group.

// RV32I
constexpr uint32_t kLui = 0b0000000000000000000000000'0110111;
constexpr uint32_t kAuipc = 0b0000000000000000000000000'0010111;
constexpr uint32_t kJal = 0b00000000000000000000'00000'1101111;
constexpr uint32_t kJalr = 0b00000000000'00000'000'00000'1100111;
constexpr uint32_t kBeq = 0b0000000'00000'00000'000'00000'1100011;
constexpr uint32_t kBne = 0b0000000'00000'00000'001'00000'1100011;
constexpr uint32_t kBlt = 0b0000000'00000'00000'100'00000'1100011;
constexpr uint32_t kBge = 0b0000000'00000'00000'101'00000'1100011;
constexpr uint32_t kBltu = 0b0000000'00000'00000'110'00000'1100011;
constexpr uint32_t kBgeu = 0b0000000'00000'00000'111'00000'1100011;
constexpr uint32_t kLb = 0b000000000000'00000'000'00000'0000011;
constexpr uint32_t kLh = 0b000000000000'00000'001'00000'0000011;
constexpr uint32_t kLw = 0b000000000000'00000'010'00000'0000011;
constexpr uint32_t kLbu = 0b000000000000'00000'100'00000'0000011;
constexpr uint32_t kLhu = 0b000000000000'00000'101'00000'0000011;
constexpr uint32_t kSb = 0b0000000'00000'00000'000'00000'0100011;
constexpr uint32_t kSh = 0b0000000'00000'00000'001'00000'0100011;
constexpr uint32_t kSw = 0b0000000'00000'00000'010'00000'0100011;
constexpr uint32_t kAddi = 0b000000000000'00000'000'00000'0010011;
constexpr uint32_t kSlti = 0b000000000000'00000'010'00000'0010011;
constexpr uint32_t kSltiu = 0b000000000000'00000'011'00000'0010011;
constexpr uint32_t kXori = 0b000000000000'00000'100'00000'0010011;
constexpr uint32_t kOri = 0b000000000000'00000'110'00000'0010011;
constexpr uint32_t kAndi = 0b000000000000'00000'111'00000'0010011;
constexpr uint32_t kSlli = 0b0000000'00000'00000'001'00000'0010011;
constexpr uint32_t kSrli = 0b0000000'00000'00000'101'00000'0010011;
constexpr uint32_t kSrai = 0b0100000'00000'00000'101'00000'0010011;
constexpr uint32_t kAdd = 0b0000000'00000'00000'000'00000'0110011;
constexpr uint32_t kSub = 0b0100000'00000'00000'000'00000'0110011;
constexpr uint32_t kSll = 0b0000000'00000'00000'001'00000'0110011;
constexpr uint32_t kSlt = 0b0000000'00000'00000'010'00000'0110011;
constexpr uint32_t kSltu = 0b0000000'00000'00000'011'00000'0110011;
constexpr uint32_t kXor = 0b0000000'00000'00000'100'00000'0110011;
constexpr uint32_t kSrl = 0b0000000'00000'00000'101'00000'0110011;
constexpr uint32_t kSra = 0b0100000'00000'00000'101'00000'0110011;
constexpr uint32_t kOr = 0b0000000'00000'00000'110'00000'0110011;
constexpr uint32_t kAnd = 0b0000000'00000'00000'111'00000'0110011;
constexpr uint32_t kFence = 0b000000000000'00000'000'00000'0001111;
constexpr uint32_t kEcall = 0b000000000000'00000'000'00000'1110011;
constexpr uint32_t kEbreak = 0b000000000001'00000'000'00000'1110011;
// RV32 Zifencei
constexpr uint32_t kFencei = 0b000000000000'00000'001'00000'0001111;
// RV32 Zicsr
constexpr uint32_t kCsrw = 0b000000000000'00000'001'00000'1110011;
constexpr uint32_t kCsrs = 0b000000000000'00000'010'00000'1110011;
constexpr uint32_t kCsrc = 0b000000000000'00000'011'00000'1110011;
constexpr uint32_t kCsrwi = 0b000000000000'00000'101'00000'1110011;
constexpr uint32_t kCsrsi = 0b000000000000'00000'110'00000'1110011;
constexpr uint32_t kCsrci = 0b000000000000'00000'111'00000'1110011;
// RV32M
constexpr uint32_t kMul = 0b0000001'00000'00000'000'00000'0110011;
constexpr uint32_t kMulh = 0b0000001'00000'00000'001'00000'0110011;
constexpr uint32_t kMulhsu = 0b0000001'00000'00000'010'00000'0110011;
constexpr uint32_t kMulhu = 0b0000001'00000'00000'011'00000'0110011;
constexpr uint32_t kDiv = 0b0000001'00000'00000'100'00000'0110011;
constexpr uint32_t kDivu = 0b0000001'00000'00000'101'00000'0110011;
constexpr uint32_t kRem = 0b0000001'00000'00000'110'00000'0110011;
constexpr uint32_t kRemu = 0b0000001'00000'00000'111'00000'0110011;
// RV32A
constexpr uint32_t kLrw = 0b00010'0'0'00000'00000'010'00000'0101111;
constexpr uint32_t kScw = 0b00011'0'0'00000'00000'010'00000'0101111;
constexpr uint32_t kAmoswapw = 0b00001'0'0'00000'00000'010'00000'0101111;
constexpr uint32_t kAmoaddw = 0b00000'0'0'00000'00000'010'00000'0101111;
constexpr uint32_t kAmoxorw = 0b00100'0'0'00000'00000'010'00000'0101111;
constexpr uint32_t kAmoandw = 0b01100'0'0'00000'00000'010'00000'0101111;
constexpr uint32_t kAmoorw = 0b01000'0'0'00000'00000'010'00000'0101111;
constexpr uint32_t kAmominw = 0b10000'0'0'00000'00000'010'00000'0101111;
constexpr uint32_t kAmomaxw = 0b10100'0'0'00000'00000'010'00000'0101111;
constexpr uint32_t kAmominuw = 0b11000'0'0'00000'00000'010'00000'0101111;
constexpr uint32_t kAmomaxuw = 0b11100'0'0'00000'00000'010'00000'0101111;
// RV32F
constexpr uint32_t kFlw = 0b000000000000'00000'010'00000'0000111;
constexpr uint32_t kFsw = 0b0000000'00000'00000'010'00000'0100111;
constexpr uint32_t kFmadds = 0b00000'00'00000'00000'000'00000'1000011;
constexpr uint32_t kFmsubs = 0b00000'00'00000'00000'000'00000'1000111;
constexpr uint32_t kFnmsubs = 0b00000'00'00000'00000'000'00000'1001011;
constexpr uint32_t kFnmadds = 0b00000'00'00000'00000'000'00000'1001111;
constexpr uint32_t kFadds = 0b0000000'00000'00000'000'00000'1010011;
constexpr uint32_t kFsubs = 0b0000100'00000'00000'000'00000'1010011;
constexpr uint32_t kFmuls = 0b0001000'00000'00000'000'00000'1010011;
constexpr uint32_t kFdivs = 0b0001100'00000'00000'000'00000'1010011;
constexpr uint32_t kFsqrts = 0b0101100'00000'00000'000'00000'1010011;
constexpr uint32_t kFsgnjs = 0b0010000'00000'00000'000'00000'1010011;
constexpr uint32_t kFsgnjns = 0b0010000'00000'00000'001'00000'1010011;
constexpr uint32_t kFsgnjxs = 0b0010000'00000'00000'010'00000'1010011;
constexpr uint32_t kFmins = 0b0010100'00000'00000'000'00000'1010011;
constexpr uint32_t kFmaxs = 0b0010100'00000'00000'001'00000'1010011;
constexpr uint32_t kFcvtws = 0b1100000'00000'00000'000'00000'1010011;
constexpr uint32_t kFcvtwus = 0b1100000'00001'00000'000'00000'1010011;
constexpr uint32_t kFmvxw = 0b1110000'00000'00000'000'00000'1010011;
constexpr uint32_t kFeqs = 0b1010000'00000'00000'010'00000'1010011;
constexpr uint32_t kFlts = 0b1010000'00000'00000'001'00000'1010011;
constexpr uint32_t kFles = 0b1010000'00000'00000'000'00000'1010011;
constexpr uint32_t kFclasss = 0b1110000'00000'00000'001'00000'1010011;
constexpr uint32_t kFcvtsw = 0b1101000'00000'00000'000'00000'1010011;
constexpr uint32_t kFcvtswu = 0b1101000'00001'00000'000'00000'1010011;
constexpr uint32_t kFmvwx = 0b1111000'00000'00000'000'00000'1010011;
// RV32D
constexpr uint32_t kFld = 0b000000000000'00000'011'00000'0000111;
constexpr uint32_t kFsd = 0b0000000'00000'00000'011'00000'0100111;
constexpr uint32_t kFmaddd = 0b00000'01'00000'00000'000'00000'1000011;
constexpr uint32_t kFmsubd = 0b00000'01'00000'00000'000'00000'1000111;
constexpr uint32_t kFnmsubd = 0b00000'01'00000'00000'000'00000'1001011;
constexpr uint32_t kFnmaddd = 0b00000'01'00000'00000'000'00000'1001111;
constexpr uint32_t kFaddd = 0b0000001'00000'00000'000'00000'1010011;
constexpr uint32_t kFsubd = 0b0000101'00000'00000'000'00000'1010011;
constexpr uint32_t kFmuld = 0b0001001'00000'00000'000'00000'1010011;
constexpr uint32_t kFdivd = 0b0001101'00000'00000'000'00000'1010011;
constexpr uint32_t kFsqrtd = 0b0101101'00000'00000'000'00000'1010011;
constexpr uint32_t kFsgnjd = 0b0010001'00000'00000'000'00000'1010011;
constexpr uint32_t kFsgnjnd = 0b0010001'00000'00000'001'00000'1010011;
constexpr uint32_t kFsgnjxd = 0b0010001'00000'00000'010'00000'1010011;
constexpr uint32_t kFmind = 0b0010101'00000'00000'000'00000'1010011;
constexpr uint32_t kFmaxd = 0b0010101'00000'00000'001'00000'1010011;
constexpr uint32_t kFcvtsd = 0b0100000'00001'00000'000'00000'1010011;
constexpr uint32_t kFcvtds = 0b0100001'00000'00000'000'00000'1010011;
constexpr uint32_t kFeqd = 0b1010001'00000'00000'010'00000'1010011;
constexpr uint32_t kFltd = 0b1010001'00000'00000'001'00000'1010011;
constexpr uint32_t kFled = 0b1010001'00000'00000'000'00000'1010011;
constexpr uint32_t kFclassd = 0b1110001'00000'00000'001'00000'1010011;
constexpr uint32_t kFcvtwd = 0b1100001'00000'00000'000'00000'1010011;
constexpr uint32_t kFcvtwud = 0b1100001'00001'00000'000'00000'1010011;
constexpr uint32_t kFcvtdw = 0b1101001'00000'00000'000'00000'1010011;
constexpr uint32_t kFcvtdwu = 0b1101001'00001'00000'000'00000'1010011;
// RV32C
constexpr uint32_t kClwsp = 0b010'0'00000'00000'10;
constexpr uint32_t kCflwsp = 0b011'0'00000'00000'10;
constexpr uint32_t kCdldsp = 0b001'0'00000'00000'10;
constexpr uint32_t kCswsp = 0b110'000000'00000'10;
constexpr uint32_t kCfswsp = 0b111'000000'00000'10;
constexpr uint32_t kCdsdsp = 0b101'000000'00000'10;
constexpr uint32_t kClw = 0b010'000'000'00'000'00;
constexpr uint32_t kCflw = 0b011'000'000'00'000'00;
constexpr uint32_t kCdld = 0b001'000'000'00'000'00;
constexpr uint32_t kCsw = 0b110'000'000'00'000'00;
constexpr uint32_t kCfsw = 0b111'000'000'00'000'00;
constexpr uint32_t kCdsd = 0b101'000'000'00'000'00;
constexpr uint32_t kCj = 0b101'00000000000'01;
constexpr uint32_t kCjal = 0b001'00000000000'01;
constexpr uint32_t kCjr = 0b100'0'00000'00000'10;
constexpr uint32_t kCjalr = 0b100'1'00000'00000'10;
constexpr uint32_t kCbeqz = 0b110'000'000'00000'01;
constexpr uint32_t kCbnez = 0b111'000'000'00000'01;
constexpr uint32_t kCli = 0b010'0'00000'00000'01;
constexpr uint32_t kClui = 0b011'0'00000'00000'01;
constexpr uint32_t kCaddi = 0b000'0'00000'00000'01;
constexpr uint32_t kCaddi16sp = 0b011'0'00010'00000'01;
constexpr uint32_t kCaddi4spn = 0b000'00000000'000'00;
constexpr uint32_t kCslli = 0b000'0'00000'00000'10;
constexpr uint32_t kCsrli = 0b100'0'00'000'00000'01;
constexpr uint32_t kCsrai = 0b100'0'01'000'00000'01;
constexpr uint32_t kCandi = 0b100'0'10'000'00000'01;
constexpr uint32_t kCmv = 0b100'0'00000'00000'10;
constexpr uint32_t kCadd = 0b100'1'00000'00000'10;
constexpr uint32_t kCand = 0b100'0'11'000'11'000'01;
constexpr uint32_t kCor = 0b100'0'11'000'10'000'01;
constexpr uint32_t kCxor = 0b100'0'11'000'01'000'01;
constexpr uint32_t kCSub = 0b100'0'11'000'00'000'01;
constexpr uint32_t kCnop = 0b000'0'00000'00000'01;
constexpr uint32_t kCebreak = 0b100'1'00000'00000'10;
constexpr uint32_t kUret = 0b000'0000'00010'00000'000'00000'111'0011;
constexpr uint32_t kSret = 0b000'1000'00010'00000'000'00000'111'0011;
constexpr uint32_t kMret = 0b001'1000'00010'00000'000'00000'111'0011;
constexpr uint32_t kWfi = 0b000'1000'00101'00000'000'00000'111'0011;
constexpr uint32_t kSfenceVmaZz = 0b000'1001'00000'00000'000'00000'111'0011;
constexpr uint32_t kSfenceVmaZn = 0b000'1001'00001'00000'000'00000'111'0011;
constexpr uint32_t kSfenceVmaNz = 0b000'1001'00000'00001'000'00000'111'0011;
constexpr uint32_t kSfenceVmaNn = 0b000'1001'00001'00001'000'00000'111'0011;

class RiscV32GEncodingTest : public testing::Test {
 protected:
  RiscV32GEncodingTest() {
    state_ = new RiscVState("test", RiscVXlen::RV32, &memory_);
    enc_ = new RiscV32GEncoding(state_);
  }
  ~RiscV32GEncodingTest() override {
    delete enc_;
    delete state_;
  }

  FlatDemandMemory memory_;
  RiscVState *state_;
  RiscV32GEncoding *enc_;
};

constexpr int kRdValue = 1;
constexpr int kSuccValue = 0xf;
constexpr int kPredValue = 0xf;

static uint32_t SetRd(uint32_t iword, uint32_t rdval) {
  return (iword | ((rdval & 0x1f) << 7));
}

static uint32_t SetRs1(uint32_t iword, uint32_t rsval) {
  return (iword | ((rsval & 0x1f) << 15));
}

static uint32_t SetPred(uint32_t iword, uint32_t pred) {
  return (iword | ((pred & 0xf) << 24));
}

static uint32_t SetSucc(uint32_t iword, uint32_t succ) {
  return (iword | ((succ & 0xf) << 20));
}

static uint32_t Set16Rd(uint32_t iword, uint32_t val) {
  return (iword | ((val & 0x1f) << 7));
}

static uint32_t Set16Rs2(uint32_t iword, uint32_t val) {
  return (iword | ((val & 0x1f) << 2));
}

TEST_F(RiscV32GEncodingTest, SourceOperands) {
  auto &getters = enc_->source_op_getters();
  for (int i = *SourceOpEnum::kNone; i < *SourceOpEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No source operand for enum value " << i
                                     << " (" << kSourceOpNames[i] << ")";
  }
}

TEST_F(RiscV32GEncodingTest, DestOperands) {
  auto &getters = enc_->dest_op_getters();
  for (int i = *DestOpEnum::kNone; i < *DestOpEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No dest operand for enum value " << i
                                     << " (" << kDestOpNames[i] << ")";
  }
}

TEST_F(RiscV32GEncodingTest, SimpleResources) {
  auto &getters = enc_->simple_resource_getters();
  for (int i = *SimpleResourceEnum::kNone;
       i < *SimpleResourceEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No source operand for enum value " << i
                                     << " (" << kSimpleResourceNames[i] << ")";
  }
}

TEST_F(RiscV32GEncodingTest, ComplexResources) {
  auto &getters = enc_->source_op_getters();
  for (int i = *ComplexResourceEnum::kNone;
       i < *ComplexResourceEnum::kPastMaxValue; ++i) {
    EXPECT_TRUE(getters.contains(i)) << "No source operand for enum value " << i
                                     << " (" << kComplexResourceNames[i] << ")";
  }
}

TEST_F(RiscV32GEncodingTest, RV32IOpcodes) {
  enc_->ParseInstruction(SetRd(kLui, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kLui);
  enc_->ParseInstruction(SetRd(kAuipc, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kAuipc);
  enc_->ParseInstruction(SetRd(kJal, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kJal);
  enc_->ParseInstruction(SetRd(kJalr, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kJalr);
  enc_->ParseInstruction(kBeq);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kBeq);
  enc_->ParseInstruction(kBne);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kBne);
  enc_->ParseInstruction(kBlt);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kBlt);
  enc_->ParseInstruction(kBge);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kBge);
  enc_->ParseInstruction(kBltu);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kBltu);
  enc_->ParseInstruction(kBgeu);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kBgeu);
  enc_->ParseInstruction(SetRd(kLb, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kLb);
  enc_->ParseInstruction(SetRd(kLh, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kLh);
  enc_->ParseInstruction(SetRd(kLw, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kLw);
  enc_->ParseInstruction(SetRd(kLbu, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kLbu);
  enc_->ParseInstruction(SetRd(kLhu, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kLhu);
  enc_->ParseInstruction(SetRd(kSb, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kSb);
  enc_->ParseInstruction(SetRd(kSh, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kSh);
  enc_->ParseInstruction(SetRd(kSw, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kSw);
  enc_->ParseInstruction(SetRd(kAddi, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kAddi);
  enc_->ParseInstruction(SetRd(kSlti, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kSlti);
  enc_->ParseInstruction(SetRd(kSltiu, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kSltiu);
  enc_->ParseInstruction(SetRd(kXori, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kXori);
  enc_->ParseInstruction(SetRd(kOri, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kOri);
  enc_->ParseInstruction(SetRd(kAndi, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kAndi);
  enc_->ParseInstruction(SetRd(kSlli, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kSlli);
  enc_->ParseInstruction(SetRd(kSrli, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kSrli);
  enc_->ParseInstruction(SetRd(kSrai, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kSrai);
  enc_->ParseInstruction(SetRd(kAdd, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kAdd);
  enc_->ParseInstruction(SetRd(kSub, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kSub);
  enc_->ParseInstruction(SetRd(kSll, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kSll);
  enc_->ParseInstruction(SetRd(kSlt, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kSlt);
  enc_->ParseInstruction(SetRd(kSltu, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kSltu);
  enc_->ParseInstruction(SetRd(kXor, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kXor);
  enc_->ParseInstruction(SetRd(kSrl, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kSrl);
  enc_->ParseInstruction(SetRd(kSra, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kSra);
  enc_->ParseInstruction(SetRd(kOr, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kOr);
  enc_->ParseInstruction(SetRd(kAnd, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kAnd);
  enc_->ParseInstruction(SetSucc(SetPred(kFence, kPredValue), kSuccValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFence);
  enc_->ParseInstruction(kEcall);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kEcall);
  enc_->ParseInstruction(kEbreak);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kEbreak);
}

TEST_F(RiscV32GEncodingTest, ZifenceiOpcodes) {
  // RV32 Zifencei
  enc_->ParseInstruction(kFencei);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFencei);
}

TEST_F(RiscV32GEncodingTest, ZicsrOpcodes) {
  // RV32 Zicsr
  enc_->ParseInstruction(SetRd(kCsrw, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCsrrw);
  enc_->ParseInstruction(SetRd(SetRs1(kCsrs, kRdValue), kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCsrrs);
  enc_->ParseInstruction(SetRd(SetRs1(kCsrc, kRdValue), kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCsrrc);
  enc_->ParseInstruction(SetRs1(kCsrw, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCsrrwNr);
  enc_->ParseInstruction(kCsrs);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCsrrsNw);
  enc_->ParseInstruction(kCsrc);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCsrrcNw);
  enc_->ParseInstruction(SetRd(kCsrwi, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCsrrwi);
  enc_->ParseInstruction(SetRd(SetRs1(kCsrsi, kRdValue), kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCsrrsi);
  enc_->ParseInstruction(SetRd(SetRs1(kCsrci, kRdValue), kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCsrrci);
  enc_->ParseInstruction(kCsrwi);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCsrrwiNr);
  enc_->ParseInstruction(kCsrsi);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCsrrsiNw);
  enc_->ParseInstruction(kCsrci);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCsrrciNw);
  enc_->ParseInstruction(kCsrw);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kUnimp);
}

TEST_F(RiscV32GEncodingTest, RV32MOpcodes) {
  // RV32M
  enc_->ParseInstruction(kMul);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kMul);
  enc_->ParseInstruction(kMulh);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kMulh);
  enc_->ParseInstruction(kMulhsu);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kMulhsu);
  enc_->ParseInstruction(kMulhu);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kMulhu);
  enc_->ParseInstruction(kDiv);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kDiv);
  enc_->ParseInstruction(kDivu);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kDivu);
  enc_->ParseInstruction(kRem);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kRem);
  enc_->ParseInstruction(kRemu);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kRemu);
}

TEST_F(RiscV32GEncodingTest, RV32AOpcodes) {
  // RV32A
  enc_->ParseInstruction(kLrw);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kLrw);
  enc_->ParseInstruction(kScw);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kScw);
  enc_->ParseInstruction(kAmoswapw);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kAmoswapw);
  enc_->ParseInstruction(kAmoaddw);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kAmoaddw);
  enc_->ParseInstruction(kAmoxorw);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kAmoxorw);
  enc_->ParseInstruction(kAmoandw);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kAmoandw);
  enc_->ParseInstruction(kAmoorw);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kAmoorw);
  enc_->ParseInstruction(kAmominw);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kAmominw);
  enc_->ParseInstruction(kAmomaxw);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kAmomaxw);
  enc_->ParseInstruction(kAmominuw);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kAmominuw);
  enc_->ParseInstruction(kAmomaxuw);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kAmomaxuw);
}

TEST_F(RiscV32GEncodingTest, RV32FOpcodes) {
  // RV32F
  enc_->ParseInstruction(kFlw);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFlw);
  enc_->ParseInstruction(kFsw);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFsw);
  enc_->ParseInstruction(kFmadds);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFmaddS);
  enc_->ParseInstruction(kFmsubs);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFmsubS);
  enc_->ParseInstruction(kFnmsubs);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFnmsubS);
  enc_->ParseInstruction(kFnmadds);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFnmaddS);
  enc_->ParseInstruction(kFadds);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFaddS);
  enc_->ParseInstruction(kFsubs);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFsubS);
  enc_->ParseInstruction(kFmuls);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFmulS);
  enc_->ParseInstruction(kFdivs);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFdivS);
  enc_->ParseInstruction(kFsqrts);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFsqrtS);
  enc_->ParseInstruction(kFsgnjs);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFsgnjS);
  enc_->ParseInstruction(kFsgnjns);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFsgnjnS);
  enc_->ParseInstruction(kFsgnjxs);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFsgnjxS);
  enc_->ParseInstruction(kFmins);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFminS);
  enc_->ParseInstruction(kFmaxs);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFmaxS);
  enc_->ParseInstruction(kFcvtws);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFcvtWs);
  enc_->ParseInstruction(kFcvtwus);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFcvtWus);
  enc_->ParseInstruction(kFmvxw);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFmvXw);
  enc_->ParseInstruction(kFeqs);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFcmpeqS);
  enc_->ParseInstruction(kFlts);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFcmpltS);
  enc_->ParseInstruction(kFles);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFcmpleS);
  enc_->ParseInstruction(kFclasss);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFclassS);
  enc_->ParseInstruction(kFcvtsw);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFcvtSw);
  enc_->ParseInstruction(kFcvtswu);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFcvtSwu);
  enc_->ParseInstruction(kFmvwx);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFmvWx);
}

TEST_F(RiscV32GEncodingTest, RV32DOpcodes) {
  // RF32D
  enc_->ParseInstruction(kFld);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFld);
  enc_->ParseInstruction(kFsd);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFsd);
  enc_->ParseInstruction(kFmaddd);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFmaddD);
  enc_->ParseInstruction(kFmsubd);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFmsubD);
  enc_->ParseInstruction(kFnmsubd);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFnmsubD);
  enc_->ParseInstruction(kFnmaddd);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFnmaddD);
  enc_->ParseInstruction(kFaddd);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFaddD);
  enc_->ParseInstruction(kFsubd);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFsubD);
  enc_->ParseInstruction(kFmuld);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFmulD);
  enc_->ParseInstruction(kFdivd);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFdivD);
  enc_->ParseInstruction(kFsqrtd);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFsqrtD);
  enc_->ParseInstruction(kFsgnjd);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFsgnjD);
  enc_->ParseInstruction(kFsgnjnd);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFsgnjnD);
  enc_->ParseInstruction(kFsgnjxd);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFsgnjxD);
  enc_->ParseInstruction(kFmind);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFminD);
  enc_->ParseInstruction(kFmaxd);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFmaxD);
  enc_->ParseInstruction(kFcvtsd);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFcvtSd);
  enc_->ParseInstruction(kFcvtds);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFcvtDs);
  enc_->ParseInstruction(kFeqd);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFcmpeqD);
  enc_->ParseInstruction(kFltd);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFcmpltD);
  enc_->ParseInstruction(kFled);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFcmpleD);
  enc_->ParseInstruction(kFclassd);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFclassD);
  enc_->ParseInstruction(kFcvtwd);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFcvtWd);
  enc_->ParseInstruction(kFcvtwud);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFcvtWud);
  enc_->ParseInstruction(kFcvtdw);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFcvtDw);
  enc_->ParseInstruction(kFcvtdwu);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kFcvtDwu);
}

// Test for decoding compact opcodes.
TEST_F(RiscV32GEncodingTest, RV32COpcodes) {
  enc_->ParseInstruction(Set16Rd(kClwsp, 1));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kClwsp);
  enc_->ParseInstruction(kCflwsp);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCflwsp);
  enc_->ParseInstruction(kCdldsp);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCfldsp);
  enc_->ParseInstruction(kCswsp);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCswsp);
  enc_->ParseInstruction(kCfswsp);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCfswsp);
  enc_->ParseInstruction(kCdsdsp);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCfsdsp);
  enc_->ParseInstruction(kClw);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kClw);
  enc_->ParseInstruction(kCflw);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCflw);
  enc_->ParseInstruction(kCdld);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCfld);
  enc_->ParseInstruction(kCsw);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCsw);
  enc_->ParseInstruction(kCfsw);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCfsw);
  enc_->ParseInstruction(kCdsd);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCfsd);
  enc_->ParseInstruction(kCj);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCj);
  enc_->ParseInstruction(kCjal);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCjal);
  enc_->ParseInstruction(Set16Rd(kCjr, 1));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCjr);
  enc_->ParseInstruction(Set16Rd(kCjalr, 1));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCjalr);
  enc_->ParseInstruction(kCbeqz);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCbeqz);
  enc_->ParseInstruction(kCbnez);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCbnez);
  enc_->ParseInstruction(Set16Rd(kCli, 1));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCli);
  enc_->ParseInstruction(Set16Rs2(Set16Rd(kClui, 1), 5));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kClui);
  enc_->ParseInstruction(Set16Rs2(Set16Rd(kCaddi, 1), 5));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCaddi);
  enc_->ParseInstruction(Set16Rs2(kCaddi16sp, 5));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCaddi16sp);
  enc_->ParseInstruction(kCaddi4spn | 0b000'01010101'000'00);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCaddi4spn);
  enc_->ParseInstruction(Set16Rs2(Set16Rd(kCslli, 1), 5));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCslli);
  enc_->ParseInstruction(Set16Rs2(kCsrli, 5));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCsrli);
  enc_->ParseInstruction(Set16Rs2(kCsrai, 5));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCsrai);
  enc_->ParseInstruction(kCandi);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCandi);
  enc_->ParseInstruction(Set16Rs2(Set16Rd(kCmv, 1), 2));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCmv);
  enc_->ParseInstruction(Set16Rs2(Set16Rd(kCadd, 1), 2));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCadd);
  enc_->ParseInstruction(kCand);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCand);
  enc_->ParseInstruction(kCor);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCor);
  enc_->ParseInstruction(kCxor);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCxor);
  enc_->ParseInstruction(kCSub);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCsub);
  enc_->ParseInstruction(kCnop);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCnop);
  enc_->ParseInstruction(kCebreak);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCebreak);
  enc_->ParseInstruction(0x0000);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kCunimp);
}

TEST_F(RiscV32GEncodingTest, RV32PrivilegedOpcodes) {
  enc_->ParseInstruction(kUret);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kUret);
  enc_->ParseInstruction(kSret);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kSret);
  enc_->ParseInstruction(kMret);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kMret);
  enc_->ParseInstruction(kWfi);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kWfi);
  enc_->ParseInstruction(kSfenceVmaZz);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kSfenceVmaZz);
  enc_->ParseInstruction(kSfenceVmaZn);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kSfenceVmaZn);
  enc_->ParseInstruction(kSfenceVmaNz);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kSfenceVmaNz);
  enc_->ParseInstruction(kSfenceVmaNn);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kRiscv32g, 0), OpcodeEnum::kSfenceVmaNn);
}

}  // namespace
