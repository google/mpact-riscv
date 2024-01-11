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

#include "riscv/riscv_i_instructions.h"

#include <string>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "absl/strings/string_view.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/instruction.h"

// This file contains tests for individual RiscV32I instructions.

namespace {

using ::mpact::sim::generic::ImmediateOperand;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::RV32Register;

constexpr char kX1[] = "x1";
constexpr char kX2[] = "x2";
constexpr char kX3[] = "x3";

constexpr uint32_t kInstAddress = 0x2468;
constexpr int32_t kVal1 = 0x1234;
constexpr int32_t kVal2 = -0x5678;
constexpr uint32_t kOffset = 0x246;
constexpr uint32_t kBranchTarget = kInstAddress + kOffset;
constexpr uint32_t kMemAddress = 0x1000;
constexpr uint32_t kMemValue = 0x81'92'a3'b4;
constexpr uint32_t kShift = 6;

// The test fixture allocates a machine state object and an instruction object.
// It also contains convenience methods for interacting with the instruction
// object in a more short hand form.
class RV32IInstructionTest : public testing::Test {
 public:
  RV32IInstructionTest() {
    state_ = new RiscVState("test", RiscVXlen::RV32);
    instruction_ = new Instruction(kInstAddress, state_);
    instruction_->set_size(4);
  }
  ~RV32IInstructionTest() override {
    delete state_;
    delete instruction_;
  }

  // Appends the source and destination operands for the register names
  // given in the two vectors.
  void AppendRegisterOperands(Instruction *inst,
                              const std::vector<std::string> &sources,
                              const std::vector<std::string> &destinations) {
    for (auto &reg_name : sources) {
      auto *reg = state_->GetRegister<RV32Register>(reg_name).first;
      inst->AppendSource(reg->CreateSourceOperand());
    }
    for (auto &reg_name : destinations) {
      auto *reg = state_->GetRegister<RV32Register>(reg_name).first;
      inst->AppendDestination(reg->CreateDestinationOperand(0));
    }
  }

  void AppendRegisterOperands(const std::vector<std::string> &sources,
                              const std::vector<std::string> &destinations) {
    AppendRegisterOperands(instruction_, sources, destinations);
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
  void SetRegisterValues(const std::vector<std::tuple<std::string, T>> values) {
    for (auto &[reg_name, value] : values) {
      auto *reg = state_->GetRegister<RV32Register>(reg_name).first;
      auto *db = state_->db_factory()->Allocate<RV32Register::ValueType>(1);
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
    auto *reg = state_->GetRegister<RV32Register>(reg_name).first;
    return reg->data_buffer()->Get<T>(0);
  }

  RiscVState *state_;
  Instruction *instruction_;
};

// Almost all the tests below follow the same pattern. There are two phases.
// In the first register and or immediate operands are added to the instruction,
// and the instruction semantic function under test is bound to the instruction.
// In the second phase, the values of register operands are assigned, the
// instruction is executed, and the value(s) of the output register(s) is (are)
// compared against the expected value. The second phase may be repeated for
// different combinations of register operand values.

TEST_F(RV32IInstructionTest, RV32IAdd) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVIAdd);

  SetRegisterValues<int32_t>({{kX1, kVal1}, {kX2, kVal2}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int32_t>(kX3), kVal1 + kVal2);
}

TEST_F(RV32IInstructionTest, RV32ISub) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVISub);

  SetRegisterValues<int32_t>({{kX1, kVal1}, {kX2, kVal2}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int32_t>(kX3), kVal1 - kVal2);
}

TEST_F(RV32IInstructionTest, RV32ISlt) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVISlt);

  SetRegisterValues<int32_t>({{kX1, kVal1}, {kX2, kVal2}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int32_t>(kX3), kVal1 < kVal2);

  SetRegisterValues<int32_t>({{kX1, kVal2}, {kX2, kVal1}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int32_t>(kX3), kVal2 < kVal1);
}

TEST_F(RV32IInstructionTest, RV32ISltu) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVISltu);
  SetRegisterValues<int32_t>({{kX1, kVal1}, {kX2, kVal2}});

  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int32_t>(kX3),
            static_cast<uint32_t>(kVal1) < static_cast<uint32_t>(kVal2));

  SetRegisterValues<int32_t>({{kX1, kVal2}, {kX2, kVal1}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int32_t>(kX3),
            static_cast<uint32_t>(kVal2) < static_cast<uint32_t>(kVal1));
}

TEST_F(RV32IInstructionTest, RV32IAnd) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVIAnd);
  SetRegisterValues<int32_t>({{kX1, kVal1}, {kX2, kVal2}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int32_t>(kX3), kVal1 & kVal2);
}

TEST_F(RV32IInstructionTest, RV32IOr) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVIOr);

  SetRegisterValues<int32_t>({{kX1, kVal1}, {kX2, kVal2}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int32_t>(kX3), kVal1 | kVal2);
}

TEST_F(RV32IInstructionTest, RV32IXor) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVIXor);
  SetRegisterValues<int32_t>({{kX1, kVal1}, {kX2, kVal2}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int32_t>(kX3), kVal1 ^ kVal2);
}

TEST_F(RV32IInstructionTest, RV32ISll) {
  AppendRegisterOperands({kX1}, {kX3});
  AppendImmediateOperands<uint32_t>({kShift});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVISll);

  SetRegisterValues<uint32_t>({{kX1, kMemValue}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int32_t>(kX3), kMemValue << kShift);
}

TEST_F(RV32IInstructionTest, RV32ISrl) {
  AppendRegisterOperands({kX1}, {kX3});
  AppendImmediateOperands<uint32_t>({kShift});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVISrl);

  SetRegisterValues<uint32_t>({{kX1, kMemValue}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int32_t>(kX3), kMemValue >> kShift);
}

TEST_F(RV32IInstructionTest, RV32ISra) {
  AppendRegisterOperands({kX1}, {kX3});
  AppendImmediateOperands<uint32_t>({kShift});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVISra);
  SetRegisterValues<uint32_t>({{kX1, kMemValue}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int32_t>(kX3),
            static_cast<int32_t>(kMemValue) >> kShift);
}

TEST_F(RV32IInstructionTest, RV32ILui) {
  AppendRegisterOperands({}, {kX3});
  AppendImmediateOperands<uint32_t>({kMemValue});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVILui);

  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int32_t>(kX3), kMemValue & ~0xfff);
}

TEST_F(RV32IInstructionTest, RV32IAuipc) {
  AppendRegisterOperands({}, {kX3});
  AppendImmediateOperands<uint32_t>({kMemValue});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVIAuipc);

  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint32_t>(kX3),
            instruction_->address() + (kMemValue & ~0xfff));
}

TEST_F(RV32IInstructionTest, RV32INop) {
  SetSemanticFunction(&::mpact::sim::riscv::RiscVINop);
  // Verify that the semantic functions executes without any operands.
  instruction_->Execute(nullptr);
}

TEST_F(RV32IInstructionTest, RV32IJal) {
  AppendRegisterOperands({}, {RiscVState::kPcName, kX3});
  AppendImmediateOperands<uint32_t>({kOffset});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVIJal);

  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint32_t>(RiscVState::kPcName),
            instruction_->address() + kOffset);
  EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), instruction_->address() + 4);
}

TEST_F(RV32IInstructionTest, RV32IJalr) {
  AppendRegisterOperands({kX1}, {RiscVState::kPcName, kX3});
  AppendImmediateOperands<uint32_t>({kOffset});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVIJalr);

  SetRegisterValues<uint32_t>({{kX1, kMemAddress}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint32_t>(RiscVState::kPcName),
            kOffset + kMemAddress);
  EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), instruction_->address() + 4);
}

TEST_F(RV32IInstructionTest, RV32IBeq) {
  AppendRegisterOperands({kX1, kX2}, {RiscVState::kPcName});
  AppendImmediateOperands<int32_t>({kOffset});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVIBeq);

  SetRegisterValues<int32_t>(
      {{kX1, kVal1}, {kX2, kVal1}, {RiscVState::kPcName, kInstAddress}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint32_t>(RiscVState::kPcName), kBranchTarget);

  SetRegisterValues<int32_t>(
      {{kX1, kVal1}, {kX2, kVal2}, {RiscVState::kPcName, kInstAddress}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint32_t>(RiscVState::kPcName), kInstAddress);
}

TEST_F(RV32IInstructionTest, RV32IBne) {
  AppendRegisterOperands({kX1, kX2}, {RiscVState::kPcName});
  AppendImmediateOperands<int32_t>({kOffset});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVIBne);

  SetRegisterValues<int32_t>(
      {{kX1, kVal1}, {kX2, kVal2}, {RiscVState::kPcName, kInstAddress}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint32_t>(RiscVState::kPcName), kBranchTarget);

  SetRegisterValues<int32_t>(
      {{kX1, kVal1}, {kX2, kVal1}, {RiscVState::kPcName, kInstAddress}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint32_t>(RiscVState::kPcName), kInstAddress);
}

TEST_F(RV32IInstructionTest, RV32IBlt) {
  AppendRegisterOperands({kX1, kX2}, {RiscVState::kPcName});
  AppendImmediateOperands<int32_t>({kOffset});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVIBlt);

  SetRegisterValues<int32_t>(
      {{kX1, kVal1}, {kX2, kVal1}, {RiscVState::kPcName, kInstAddress}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint32_t>(RiscVState::kPcName), kInstAddress);

  SetRegisterValues<int32_t>(
      {{kX1, kVal1}, {kX2, kVal2}, {RiscVState::kPcName, kInstAddress}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint32_t>(RiscVState::kPcName), kInstAddress);

  SetRegisterValues<int32_t>(
      {{kX1, kVal2}, {kX2, kVal1}, {RiscVState::kPcName, kInstAddress}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint32_t>(RiscVState::kPcName), kBranchTarget);
}

TEST_F(RV32IInstructionTest, RV32IBltu) {
  AppendRegisterOperands({kX1, kX2}, {RiscVState::kPcName});
  AppendImmediateOperands<int32_t>({kOffset});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVIBltu);

  SetRegisterValues<int32_t>(
      {{kX1, kVal1}, {kX2, kVal1}, {RiscVState::kPcName, kInstAddress}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint32_t>(RiscVState::kPcName), kInstAddress);

  SetRegisterValues<int32_t>(
      {{kX1, kVal1}, {kX2, kVal2}, {RiscVState::kPcName, kInstAddress}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint32_t>(RiscVState::kPcName), kBranchTarget);

  SetRegisterValues<int32_t>(
      {{kX1, kVal2}, {kX2, kVal1}, {RiscVState::kPcName, kInstAddress}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint32_t>(RiscVState::kPcName), kInstAddress);
}

TEST_F(RV32IInstructionTest, RV32IBge) {
  AppendRegisterOperands({kX1, kX2}, {RiscVState::kPcName});
  AppendImmediateOperands<int32_t>({kOffset});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVIBge);

  SetRegisterValues<int32_t>(
      {{kX1, kVal1}, {kX2, kVal1}, {RiscVState::kPcName, kInstAddress}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint32_t>(RiscVState::kPcName), kBranchTarget);

  SetRegisterValues<int32_t>(
      {{kX1, kVal1}, {kX2, kVal2}, {RiscVState::kPcName, kInstAddress}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint32_t>(RiscVState::kPcName), kBranchTarget);

  SetRegisterValues<int32_t>(
      {{kX1, kVal2}, {kX2, kVal1}, {RiscVState::kPcName, kInstAddress}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint32_t>(RiscVState::kPcName), kInstAddress);
}

TEST_F(RV32IInstructionTest, RV32IBgeu) {
  AppendRegisterOperands({kX1, kX2}, {RiscVState::kPcName});
  AppendImmediateOperands<int32_t>({kOffset});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVIBgeu);

  SetRegisterValues<int32_t>(
      {{kX1, kVal1}, {kX2, kVal1}, {RiscVState::kPcName, kInstAddress}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint32_t>(RiscVState::kPcName), kBranchTarget);

  SetRegisterValues<int32_t>(
      {{kX1, kVal1}, {kX2, kVal2}, {RiscVState::kPcName, kInstAddress}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint32_t>(RiscVState::kPcName), kInstAddress);

  SetRegisterValues<int32_t>(
      {{kX1, kVal2}, {kX2, kVal1}, {RiscVState::kPcName, kInstAddress}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint32_t>(RiscVState::kPcName), kBranchTarget);
}

// Load instructions require additional setup. First the memory locations have
// to be initialized. Second, all load instructions use a child instruction for
// the value writeback to the destination register.

TEST_F(RV32IInstructionTest, RV32ILw) {
  // Initialize memory.
  auto *db = state_->db_factory()->Allocate<uint32_t>(1);
  db->Set<uint32_t>(0, kMemValue);
  state_->StoreMemory(instruction_, kMemAddress + kOffset, db);
  db->DecRef();

  // Initialize instruction.
  AppendRegisterOperands({kX1}, {});
  AppendImmediateOperands<uint32_t>({kOffset});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVILw);
  auto *child = new Instruction(state_);
  child->set_semantic_function(&::mpact::sim::riscv::RV32::RiscVILwChild);
  AppendRegisterOperands(child, {}, {kX3});
  instruction_->AppendChild(child);
  child->DecRef();

  SetRegisterValues<uint32_t>({{kX1, kMemAddress}, {kX3, 0}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), kMemValue);
}

TEST_F(RV32IInstructionTest, RV32ILh) {
  // Initialize memory.
  auto *db = state_->db_factory()->Allocate<uint32_t>(1);
  db->Set<uint32_t>(0, kMemValue);
  state_->StoreMemory(instruction_, kMemAddress + kOffset, db);
  db->DecRef();

  // Initialize instruction.
  AppendRegisterOperands({kX1}, {});
  AppendImmediateOperands<uint32_t>({kOffset});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVILh);
  auto *child = new Instruction(state_);
  child->set_semantic_function(&::mpact::sim::riscv::RV32::RiscVILhChild);
  AppendRegisterOperands(child, {}, {kX3});
  instruction_->AppendChild(child);
  child->DecRef();

  SetRegisterValues<uint32_t>({{kX1, kMemAddress}, {kX3, 0}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int32_t>(kX3), static_cast<int16_t>(kMemValue));
}

TEST_F(RV32IInstructionTest, RV32ILhu) {
  // Initialize memory.
  auto *db = state_->db_factory()->Allocate<uint32_t>(1);
  db->Set<uint32_t>(0, kMemValue);
  state_->StoreMemory(instruction_, kMemAddress + kOffset, db);
  db->DecRef();

  // Initialize instruction.
  AppendRegisterOperands({kX1}, {});
  AppendImmediateOperands<uint32_t>({kOffset});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVILhu);
  auto *child = new Instruction(state_);
  child->set_semantic_function(&::mpact::sim::riscv::RV32::RiscVILhuChild);
  AppendRegisterOperands(child, {}, {kX3});
  instruction_->AppendChild(child);
  child->DecRef();

  SetRegisterValues<uint32_t>({{kX1, kMemAddress}, {kX3, 0}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), static_cast<uint16_t>(kMemValue));
}

TEST_F(RV32IInstructionTest, RV32ILb) {
  // Initialize memory.
  auto *db = state_->db_factory()->Allocate<uint32_t>(1);
  db->Set<uint32_t>(0, kMemValue);
  state_->StoreMemory(instruction_, kMemAddress + kOffset, db);
  db->DecRef();

  // Initialize instruction.
  AppendRegisterOperands({kX1}, {});
  AppendImmediateOperands<uint32_t>({kOffset});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVILb);
  auto *child = new Instruction(state_);
  child->set_semantic_function(&::mpact::sim::riscv::RV32::RiscVILbChild);
  AppendRegisterOperands(child, {}, {kX3});
  instruction_->AppendChild(child);
  child->DecRef();

  SetRegisterValues<uint32_t>({{kX1, kMemAddress}, {kX3, 0}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int32_t>(kX3), static_cast<int8_t>(kMemValue));
}

TEST_F(RV32IInstructionTest, RV32ILbu) {
  // Initialize memory.
  auto *db = state_->db_factory()->Allocate<uint32_t>(1);
  db->Set<uint32_t>(0, kMemValue);
  state_->StoreMemory(instruction_, kMemAddress + kOffset, db);
  db->DecRef();

  // Initialize instruction.
  AppendRegisterOperands({kX1}, {});
  AppendImmediateOperands<uint32_t>({kOffset});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVILbu);
  auto *child = new Instruction(state_);
  child->set_semantic_function(&::mpact::sim::riscv::RV32::RiscVILbuChild);
  AppendRegisterOperands(child, {}, {kX3});
  instruction_->AppendChild(child);
  child->DecRef();

  SetRegisterValues<uint32_t>({{kX1, kMemAddress}, {kX3, 0}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), static_cast<uint8_t>(kMemValue));
}

// Store instructions are similar to the ALU instructions, except that
// additional code is added after executing the instruction to fetch the value
// stored to memory.
TEST_F(RV32IInstructionTest, RV32ISw) {
  AppendRegisterOperands({kX1}, {});
  AppendImmediateOperands<uint32_t>({kOffset});
  AppendRegisterOperands({kX3}, {});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVISw);

  SetRegisterValues<uint32_t>({{kX1, kMemAddress}, {kX3, kMemValue}});
  instruction_->Execute(nullptr);
  auto *db = state_->db_factory()->Allocate<uint32_t>(1);
  state_->LoadMemory(instruction_, kMemAddress + kOffset, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), kMemValue);
  db->DecRef();
}

TEST_F(RV32IInstructionTest, RV32ISh) {
  AppendRegisterOperands({kX1}, {});
  AppendImmediateOperands<uint32_t>({kOffset});
  AppendRegisterOperands({kX3}, {});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVISh);
  SetRegisterValues<uint32_t>({{kX1, kMemAddress}, {kX3, kMemValue}});
  instruction_->Execute(nullptr);

  auto *db = state_->db_factory()->Allocate<uint32_t>(1);
  state_->LoadMemory(instruction_, kMemAddress + kOffset, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), static_cast<uint16_t>(kMemValue));
  db->DecRef();
}

TEST_F(RV32IInstructionTest, RV32ISb) {
  AppendRegisterOperands({kX1}, {});
  AppendImmediateOperands<uint32_t>({kOffset});
  AppendRegisterOperands({kX3}, {});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVISb);
  SetRegisterValues<uint32_t>({{kX1, kMemAddress}, {kX3, kMemValue}});
  instruction_->Execute(nullptr);

  auto *db = state_->db_factory()->Allocate<uint32_t>(1);
  state_->LoadMemory(instruction_, kMemAddress + kOffset, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), static_cast<uint8_t>(kMemValue));
  db->DecRef();
}

// The following instructions aren't tested yet, as the RV32I state doesn't
// implement these instructions beyond interface stubs.

TEST_F(RV32IInstructionTest, RV32IFence) {
  // TODO: implement test once the RiscVState handles the call.
}

TEST_F(RV32IInstructionTest, RV32IEcall) {
  // TODO: implement test once the RiscVState handles the call.
}

TEST_F(RV32IInstructionTest, RV32IEbreak) {
  // TODO: implement test once the RiscVState handles the call.
}

}  // namespace
