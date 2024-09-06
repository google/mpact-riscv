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

#include <cstdint>
#include <limits>
#include <memory>
#include <string_view>
#include <tuple>
#include <vector>

#include "absl/numeric/int128.h"
#include "absl/random/random.h"
#include "absl/strings/string_view.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/instruction.h"
#include "riscv/riscv_bitmanip_instructions.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"

namespace {
using ::mpact::sim::riscv::RV64::RiscVAddUw;
using ::mpact::sim::riscv::RV64::RiscVAndn;
using ::mpact::sim::riscv::RV64::RiscVBclr;
using ::mpact::sim::riscv::RV64::RiscVBext;
using ::mpact::sim::riscv::RV64::RiscVBinv;
using ::mpact::sim::riscv::RV64::RiscVBset;
using ::mpact::sim::riscv::RV64::RiscVClmul;
using ::mpact::sim::riscv::RV64::RiscVClmulh;
using ::mpact::sim::riscv::RV64::RiscVClmulr;
using ::mpact::sim::riscv::RV64::RiscVClz;
using ::mpact::sim::riscv::RV64::RiscVClzw;
using ::mpact::sim::riscv::RV64::RiscVCpop;
using ::mpact::sim::riscv::RV64::RiscVCpopw;
using ::mpact::sim::riscv::RV64::RiscVCtz;
using ::mpact::sim::riscv::RV64::RiscVCtzw;
using ::mpact::sim::riscv::RV64::RiscVMax;
using ::mpact::sim::riscv::RV64::RiscVMaxu;
using ::mpact::sim::riscv::RV64::RiscVMin;
using ::mpact::sim::riscv::RV64::RiscVMinu;
using ::mpact::sim::riscv::RV64::RiscVOrcb;
using ::mpact::sim::riscv::RV64::RiscVOrn;
using ::mpact::sim::riscv::RV64::RiscVRev8;
using ::mpact::sim::riscv::RV64::RiscVRol;
using ::mpact::sim::riscv::RV64::RiscVRolw;
using ::mpact::sim::riscv::RV64::RiscVRor;
using ::mpact::sim::riscv::RV64::RiscVRorw;
using ::mpact::sim::riscv::RV64::RiscVSextB;
using ::mpact::sim::riscv::RV64::RiscVSextH;
using ::mpact::sim::riscv::RV64::RiscVShAdd;
using ::mpact::sim::riscv::RV64::RiscVShAddUw;
using ::mpact::sim::riscv::RV64::RiscVSlliUw;
using ::mpact::sim::riscv::RV64::RiscVXnor;
using ::mpact::sim::riscv::RV64::RiscVZextH;

using ::mpact::sim::generic::ImmediateOperand;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::RV64Register;

constexpr uint64_t kInstAddress = 0x2468;

constexpr std::string_view kX1 = "x1";
constexpr std::string_view kX2 = "x2";
constexpr std::string_view kX3 = "x3";
constexpr std::string_view kX4 = "x4";

constexpr uint64_t kUVal1 = 0x1111222233334444ULL;
constexpr uint64_t kUVal2 = 0x5555666677778888ULL;

// Test class for testing RISCV bit manipulation instructions.
class RV64BitmanipInstructionTest : public testing::Test {
 public:
  RV64BitmanipInstructionTest() {
    state_ = std::make_unique<RiscVState>("test", RiscVXlen::RV64, nullptr);
    instruction_ = std::make_unique<Instruction>(kInstAddress, state_.get());
    instruction_->set_size(4);
  }

  // Appends the source and destination operands for the register names
  // given in the two vectors.
  void AppendRegisterOperands(
      Instruction *inst, const std::vector<std::string_view> &sources,
      const std::vector<std::string_view> &destinations) {
    for (auto &reg_name : sources) {
      auto *reg = state_->GetRegister<RV64Register>(reg_name).first;
      inst->AppendSource(reg->CreateSourceOperand());
    }
    for (auto &reg_name : destinations) {
      auto *reg = state_->GetRegister<RV64Register>(reg_name).first;
      inst->AppendDestination(reg->CreateDestinationOperand(0));
    }
  }

  void AppendRegisterOperands(
      const std::vector<std::string_view> &sources,
      const std::vector<std::string_view> &destinations) {
    AppendRegisterOperands(instruction_.get(), sources, destinations);
  }

  // Appends immediate source operands with the given values.
  template <typename T>
  void AppendImmediateOperands(const std::vector<T> &values) {
    for (auto value : values) {
      auto *src = new ImmediateOperand<T>(value);
      instruction_->AppendSource(src);
    }
  }

  // Takes a vector of tuples of register names and values. Fetches each
  // named register and sets it to the corresponding value.
  template <typename T>
  void SetRegisterValues(
      const std::vector<std::tuple<std::string_view, T>> values) {
    for (auto &[reg_name, value] : values) {
      auto *reg = state_->GetRegister<RV64Register>(reg_name).first;
      auto *db = state_->db_factory()->Allocate<RV64Register::ValueType>(1);
      db->Set<T>(0, value);
      reg->SetDataBuffer(db);
      db->DecRef();
    }
  }

  // Initializes the semantic function of the instruction object.
  void SetSemanticFunction(Instruction::SemanticFunction fcn) {
    instruction_->set_semantic_function(fcn);
  }

  // Returns the value of the named register.
  template <typename T>
  T GetRegisterValue(absl::string_view reg_name) {
    auto *reg = state_->GetRegister<RV64Register>(reg_name).first;
    return reg->data_buffer()->Get<T>(0);
  }

  std::unique_ptr<RiscVState> state_;
  std::unique_ptr<Instruction> instruction_;
  absl::BitGen bitgen_;
};

// The following tests target the Zba instructions. Most have two source
// register operands and one destination register operand. One replaces a
// source register operand with an immediate operand.

TEST_F(RV64BitmanipInstructionTest, RV64VAddUw) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVAddUw);

  SetRegisterValues<uint64_t>({{kX1, kUVal1}, {kX2, kUVal2}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), (kUVal1 & 0xffff'ffffLL) + kUVal2);
}

void VshNaddTestHelper(RV64BitmanipInstructionTest &test, int shift) {
  test.AppendRegisterOperands({kX1, kX2, kX4}, {kX3});

  test.SetRegisterValues<uint64_t>(
      {{kX1, kUVal1}, {kX2, kUVal2}, {kX4, shift}});
  test.instruction_->Execute(nullptr);
  EXPECT_EQ(test.GetRegisterValue<int64_t>(kX3), (kUVal1 << shift) + kUVal2);
}

TEST_F(RV64BitmanipInstructionTest, RV64VSh1add) {
  SetSemanticFunction(&RiscVShAdd);
  VshNaddTestHelper(*this, 1);
}

TEST_F(RV64BitmanipInstructionTest, RV64VSh2add) {
  SetSemanticFunction(&RiscVShAdd);
  VshNaddTestHelper(*this, 2);
}

TEST_F(RV64BitmanipInstructionTest, RV64VSh3add) {
  SetSemanticFunction(&RiscVShAdd);
  VshNaddTestHelper(*this, 3);
}

void VshNadduwTestHelper(RV64BitmanipInstructionTest &test, int shift) {
  test.AppendRegisterOperands({kX1, kX2, kX4}, {kX3});

  test.SetRegisterValues<uint64_t>(
      {{kX1, kUVal1}, {kX2, kUVal2}, {kX4, shift}});
  test.instruction_->Execute(nullptr);
  EXPECT_EQ(test.GetRegisterValue<int64_t>(kX3),
            ((kUVal1 & 0xffff'ffffULL) << shift) + kUVal2);
}

TEST_F(RV64BitmanipInstructionTest, RV64VSh1adduw) {
  SetSemanticFunction(&RiscVShAddUw);
  VshNadduwTestHelper(*this, 1);
}

TEST_F(RV64BitmanipInstructionTest, RV64VSh2adduw) {
  SetSemanticFunction(&RiscVShAddUw);
  VshNadduwTestHelper(*this, 2);
}

TEST_F(RV64BitmanipInstructionTest, RV64VSh3adduw) {
  SetSemanticFunction(&RiscVShAddUw);
  VshNadduwTestHelper(*this, 3);
}

TEST_F(RV64BitmanipInstructionTest, RV64Slliuw) {
  constexpr int32_t kShiftValue = 3;
  AppendRegisterOperands({kX1}, {kX3});
  AppendImmediateOperands<int32_t>({kShiftValue});
  SetSemanticFunction(&RiscVSlliUw);

  SetRegisterValues<int64_t>({{kX1, kUVal1}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), (kUVal1 & 0xffff'ffffULL)
                                                << kShiftValue);
}

TEST_F(RV64BitmanipInstructionTest, RV64Andn) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVAndn);

  SetRegisterValues<uint64_t>({{kX1, kUVal1}, {kX2, kUVal2}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), kUVal1 & ~kUVal2);
}

TEST_F(RV64BitmanipInstructionTest, RV64Orn) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVOrn);

  SetRegisterValues<uint64_t>({{kX1, kUVal1}, {kX2, kUVal2}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), kUVal1 | ~kUVal2);
}

TEST_F(RV64BitmanipInstructionTest, RV64Xnor) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVXnor);

  SetRegisterValues<uint64_t>({{kX1, kUVal1}, {kX2, kUVal2}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), kUVal1 ^ ~kUVal2);
}

TEST_F(RV64BitmanipInstructionTest, RV64Clz) {
  AppendRegisterOperands({kX1}, {kX3});
  SetSemanticFunction(&RiscVClz);

  SetRegisterValues<uint64_t>({{kX1, 0x0010'0000'0000'0000ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 11);

  SetRegisterValues<uint64_t>({{kX1, 0x8000'0000'0000'0000ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0);

  SetRegisterValues<uint64_t>({{kX1, 0ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 64);
}

TEST_F(RV64BitmanipInstructionTest, RV64Clzw) {
  AppendRegisterOperands({kX1}, {kX3});
  SetSemanticFunction(&RiscVClzw);

  SetRegisterValues<uint64_t>({{kX1, 0x8000'0000'0000'0800ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 20);

  SetRegisterValues<uint64_t>({{kX1, 0x0000'0000'8000'0000ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0);

  SetRegisterValues<uint64_t>({{kX1, 0ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 32);
}

TEST_F(RV64BitmanipInstructionTest, RV64Ctz) {
  AppendRegisterOperands({kX1}, {kX3});
  SetSemanticFunction(&RiscVCtz);

  SetRegisterValues<uint64_t>({{kX1, 0x0000'0000'0000'0800ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 11);

  SetRegisterValues<uint64_t>({{kX1, 0x0000'0000'0000'0001ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0);

  SetRegisterValues<uint64_t>({{kX1, 0ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 64);
}

TEST_F(RV64BitmanipInstructionTest, RV64Ctzw) {
  AppendRegisterOperands({kX1}, {kX3});
  SetSemanticFunction(&RiscVCtzw);

  SetRegisterValues<uint64_t>({{kX1, 0x8000'0001'0010'0000ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 20);

  SetRegisterValues<uint64_t>({{kX1, 0x0000'0100'0000'0001ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0);

  SetRegisterValues<uint64_t>({{kX1, 0ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 32);
}

TEST_F(RV64BitmanipInstructionTest, RV64Cpop) {
  AppendRegisterOperands({kX1}, {kX3});
  SetSemanticFunction(&RiscVCpop);

  SetRegisterValues<uint64_t>({{kX1, 0x8003'0401'0f00'4002ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 11);

  SetRegisterValues<uint64_t>({{kX1, 0x0000'0000'0000'0000ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0);

  SetRegisterValues<uint64_t>({{kX1, 0xffff'ffff'ffff'ffffULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 64);
}

TEST_F(RV64BitmanipInstructionTest, RV64Cpopw) {
  AppendRegisterOperands({kX1}, {kX3});
  SetSemanticFunction(&RiscVCpopw);

  SetRegisterValues<uint64_t>({{kX1, 0xffff'ffff'0ff0'f0ffULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 20);

  SetRegisterValues<uint64_t>({{kX1, 0xffff'ffff'0000'0000ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0);

  SetRegisterValues<uint64_t>({{kX1, 0xffff'ffff'ffff'ffffULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 32);
}

TEST_F(RV64BitmanipInstructionTest, RV64Max) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVMax);

  SetRegisterValues<int64_t>({{kX1, 0}, {kX2, 1}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 1);

  SetRegisterValues<int64_t>({{kX1, 0x1234}, {kX2, 0x1235}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0x1235);

  SetRegisterValues<int64_t>({{kX1, -1}, {kX2, -2}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), -1);

  SetRegisterValues<int64_t>({{kX1, -1}, {kX2, 0}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0);
}

TEST_F(RV64BitmanipInstructionTest, RV64Maxu) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVMaxu);

  SetRegisterValues<uint64_t>({{kX1, 0}, {kX2, 1}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 1);

  SetRegisterValues<uint64_t>({{kX1, 0x1234}, {kX2, 0x1235}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0x1235);

  SetRegisterValues<uint64_t>(
      {{kX1, 0xffff'ffff'ffff'ffff}, {kX2, 0x8000'0000'0000'0000}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0xffff'ffff'ffff'ffff);

  SetRegisterValues<uint64_t>({{kX1, 0xffff'ffff'ffff'ffff}, {kX2, 0}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0xffff'ffff'ffff'ffff);
}

TEST_F(RV64BitmanipInstructionTest, RV64Min) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVMin);

  SetRegisterValues<int64_t>({{kX1, 0}, {kX2, 1}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0);

  SetRegisterValues<int64_t>({{kX1, 0x1234}, {kX2, 0x1235}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0x1234);

  SetRegisterValues<int64_t>({{kX1, -1}, {kX2, -2}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), -2);

  SetRegisterValues<int64_t>({{kX1, -1}, {kX2, 0}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), -1);
}

TEST_F(RV64BitmanipInstructionTest, RV64Minu) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVMinu);

  SetRegisterValues<uint64_t>({{kX1, 0}, {kX2, 1}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), 0);

  SetRegisterValues<uint64_t>({{kX1, 0x1234}, {kX2, 0x1235}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0x1234);

  SetRegisterValues<uint64_t>(
      {{kX1, 0xffff'ffff'ffff'ffff}, {kX2, 0x8000'0000'0000'0000}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0x8000'0000'0000'0000);

  SetRegisterValues<uint64_t>({{kX1, 0xffff'ffff'ffff'ffff}, {kX2, 0}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0);
}

TEST_F(RV64BitmanipInstructionTest, RV64Sexth) {
  AppendRegisterOperands({kX1}, {kX3});
  SetSemanticFunction(&RiscVSextH);

  SetRegisterValues<uint64_t>({{kX1, 0xa5a5'a5a5'a5a5'5678ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0x5678ULL);

  SetRegisterValues<uint64_t>({{kX1, 0xa5a5'a5a5'a5a5'd678ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0xffff'ffff'ffff'd678ULL);
}

TEST_F(RV64BitmanipInstructionTest, RV64Sextb) {
  AppendRegisterOperands({kX1}, {kX3});
  SetSemanticFunction(&RiscVSextB);

  SetRegisterValues<uint64_t>({{kX1, 0xa5a5'a5a5'a5a5'a578ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0x78ULL);

  SetRegisterValues<uint64_t>({{kX1, 0xa5a5'a5a5'a5a5'a5f8ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0xffff'ffff'ffff'fff8ULL);
}

/*
TEST_F(RV64BitmanipInstructionTest, RV64Zextw) {
  AppendRegisterOperands({kX1}, {kX3});
  SetSemanticFunction(&RiscVZextw);

  SetRegisterValues<uint64_t>({{kX1, 0xa5a5'a5a5'1234'5678ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0x1234'5678ULL);

  SetRegisterValues<uint64_t>({{kX1, 0xa5a5'a5a5'9234'5678ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0x9234'5678ULL);
}
*/

TEST_F(RV64BitmanipInstructionTest, RV64Zexth) {
  AppendRegisterOperands({kX1}, {kX3});
  SetSemanticFunction(&RiscVZextH);

  SetRegisterValues<uint64_t>({{kX1, 0xa5a5'a5a5'a5a5'5678ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0x5678ULL);

  SetRegisterValues<uint64_t>({{kX1, 0xa5a5'a5a5'a5a5'd678ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0xd678ULL);
}

TEST_F(RV64BitmanipInstructionTest, RV64Rol) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVRol);

  SetRegisterValues<uint64_t>({{kX1, 0x1234'5678'9abc'def0ULL}, {kX2, 16}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), 0x5678'9abc'def0'1234ULL);

  SetRegisterValues<uint64_t>({{kX1, 0x1234'5678'9abc'def0ULL}, {kX2, 64}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0x1234'5678'9abc'def0ULL);
}

TEST_F(RV64BitmanipInstructionTest, RV64Rolw) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVRolw);

  SetRegisterValues<uint64_t>({{kX1, 0x1234'5678'9abc'def0ULL}, {kX2, 16}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), 0xffff'ffff'def0'9abcULL);

  SetRegisterValues<uint64_t>({{kX1, 0x1234'5678'9abc'7ef0ULL}, {kX2, 64}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0xffff'ffff'9abc'7ef0ULL);
}

TEST_F(RV64BitmanipInstructionTest, RV64Ror) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVRor);

  SetRegisterValues<uint64_t>({{kX1, 0x1234'5678'9abc'def0ULL}, {kX2, 16}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), 0xdef0'1234'5678'9abcULL);

  SetRegisterValues<uint64_t>({{kX1, 0x1234'5678'9abc'def0ULL}, {kX2, 64}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0x1234'5678'9abc'def0ULL);
}

TEST_F(RV64BitmanipInstructionTest, RV64Rorw) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVRorw);

  SetRegisterValues<uint64_t>({{kX1, 0x1234'5678'9abc'7ef0ULL}, {kX2, 16}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), 0x7ef0'9abcULL);

  SetRegisterValues<uint64_t>({{kX1, 0x1234'5678'9abc'7ef0ULL}, {kX2, 64}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0xffff'ffff'9abc'7ef0ULL);
}

TEST_F(RV64BitmanipInstructionTest, RV64Orcb) {
  AppendRegisterOperands({kX1}, {kX3});
  SetSemanticFunction(&RiscVOrcb);

  SetRegisterValues<uint64_t>({{kX1, 0x8003'0401'0f00'4002ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0xffff'ffff'ff00'ffffULL);

  SetRegisterValues<uint64_t>({{kX1, 0}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0);

  SetRegisterValues<uint64_t>({{kX1, 0x0000'0100'0041'0000ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0x0000'ff00'00ff'0000ULL);
}

TEST_F(RV64BitmanipInstructionTest, RV64Rev8) {
  AppendRegisterOperands({kX1}, {kX3});
  SetSemanticFunction(&RiscVRev8);

  SetRegisterValues<uint64_t>({{kX1, 0x11'22'33'44'55'66'77'88ULL}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int64_t>(kX3), 0x8877'6655'4433'2211ULL);
}

TEST_F(RV64BitmanipInstructionTest, RV64Clmul) {
  using T = uint64_t;
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVClmul);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    T val2 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    SetRegisterValues<uint64_t>({{kX1, val1}, {kX2, val2}});
    instruction_->Execute(nullptr);
    uint64_t result = 0;
    for (int i = 0; i < sizeof(T) * 8; ++i) {
      if (val2 & (1ULL << i)) result ^= static_cast<uint64_t>(val1 << i);
    }
    EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), result);
  }
}

TEST_F(RV64BitmanipInstructionTest, RV64Clmulh) {
  using T = uint64_t;
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVClmulh);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    T val2 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    SetRegisterValues<uint64_t>({{kX1, val1}, {kX2, val2}});
    instruction_->Execute(nullptr);
    absl::uint128 result = 0;
    for (int i = 0; i < sizeof(T) * 8; ++i) {
      if (val2 & (1ULL << i)) result ^= (absl::MakeUint128(0, val1) << i);
    }
    EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), absl::Uint128High64(result));
  }
}

TEST_F(RV64BitmanipInstructionTest, RV64Clmulr) {
  using T = uint64_t;
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVClmulr);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    T val2 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    SetRegisterValues<uint64_t>({{kX1, val1}, {kX2, val2}});
    instruction_->Execute(nullptr);
    uint64_t result = 0;
    for (int i = 0; i < sizeof(T) * 8 - 1; ++i) {
      if ((val2 >> i) & 1) result ^= (val1 >> (sizeof(T) * 8 - 1 - i));
    }
    EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), result);
  }
}

TEST_F(RV64BitmanipInstructionTest, RV64Bclr) {
  using T = uint64_t;
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVBclr);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    T val2 = absl::Uniform(absl::IntervalClosed, bitgen_, 0, 63);
    SetRegisterValues<uint64_t>({{kX1, val1}, {kX2, val2}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), val1 & ~(1ULL << val2));
  }
}

TEST_F(RV64BitmanipInstructionTest, RV64Bext) {
  using T = uint64_t;
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVBext);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    T val2 = absl::Uniform(absl::IntervalClosed, bitgen_, 0, 63);
    SetRegisterValues<uint64_t>({{kX1, val1}, {kX2, val2}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), (val1 >> val2) & 0x1);
  }
}

TEST_F(RV64BitmanipInstructionTest, RV64Binv) {
  using T = uint64_t;
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVBinv);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    T val2 = absl::Uniform(absl::IntervalClosed, bitgen_, 0, 63);
    SetRegisterValues<uint64_t>({{kX1, val1}, {kX2, val2}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), val1 ^ (1ULL << val2));
  }
}

TEST_F(RV64BitmanipInstructionTest, RV64Bset) {
  using T = uint64_t;
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVBset);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    T val2 = absl::Uniform(absl::IntervalClosed, bitgen_, 0, 63);
    SetRegisterValues<uint64_t>({{kX1, val1}, {kX2, val2}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), val1 | (1ULL << val2));
  }
}

}  // namespace
