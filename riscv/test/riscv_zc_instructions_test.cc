// Copyright 2024 Google LLC
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

#include "riscv/riscv_zc_instructions.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include "absl/strings/string_view.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"

// This file contains the unit tests for the 32 bit Zcmp* instructions that
// are not covered elsewhere.

namespace {

using ::mpact::sim::generic::ImmediateOperand;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::util::FlatDemandMemory;

constexpr uint32_t kInstAddress = 0x2468;
constexpr char kX1[] = "x1";
constexpr char kX2[] = "x2";
constexpr char kX8[] = "x8";
constexpr char kX9[] = "x9";
constexpr char kX10[] = "x10";
constexpr char kX11[] = "x11";
constexpr char kX18[] = "x18";
constexpr char kX19[] = "x19";
constexpr char kX20[] = "x20";
constexpr char kX21[] = "x21";
constexpr char kX22[] = "x22";
constexpr char kX23[] = "x23";
constexpr char kX24[] = "x24";
constexpr char kX25[] = "x25";
constexpr char kX26[] = "x26";
constexpr char kX27[] = "x27";
constexpr char kX30[] = "x30";
constexpr char kX31[] = "x31";
constexpr uint32_t kMemAddress = 0x1000;

// The test fixture allocates a machine state object and an instruction object.
// It also contains convenience methods for interacting with the instruction
// object in a more short hand form.
class RV32ZcInstructionTest : public testing::Test {
 public:
  RV32ZcInstructionTest() {
    memory_ = new FlatDemandMemory();
    state_ = new RiscVState("test", RiscVXlen::RV32, memory_);
    instruction_ = new Instruction(kInstAddress, state_);
    instruction_->set_size(4);
    // Set the jump table address to 0x4000.
    state_->jvt()->Set(static_cast<uint32_t>(0x4000));
    auto *db = state_->db_factory()->Allocate<uint32_t>(256);
    auto db_span = db->Get<uint32_t>();
    for (auto i = 0; i < 256; ++i) {
      db_span[i] = 0x8000 + i * sizeof(uint64_t);
    }
    state_->StoreMemory(nullptr, 0x4000, db);
    db->DecRef();
  }

  ~RV32ZcInstructionTest() override {
    delete memory_;
    delete state_;
    instruction_->DecRef();
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
      reg->data_buffer()->template Set<T>(0, value);
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

  std::vector<std::string> GetRlistRegisters(int rlist) {
    std::vector<std::string> rlist_regs;
    if (rlist < 4) return rlist_regs;
    rlist_regs.push_back(kX1);
    if (rlist == 4) return rlist_regs;
    rlist_regs.push_back(kX8);
    if (rlist == 5) return rlist_regs;
    rlist_regs.push_back(kX9);
    if (rlist == 6) return rlist_regs;
    rlist_regs.push_back(kX18);
    if (rlist == 7) return rlist_regs;
    rlist_regs.push_back(kX19);
    if (rlist == 8) return rlist_regs;
    rlist_regs.push_back(kX20);
    if (rlist == 9) return rlist_regs;
    rlist_regs.push_back(kX21);
    if (rlist == 10) return rlist_regs;
    rlist_regs.push_back(kX22);
    if (rlist == 11) return rlist_regs;
    rlist_regs.push_back(kX23);
    if (rlist == 12) return rlist_regs;
    rlist_regs.push_back(kX24);
    if (rlist == 13) return rlist_regs;
    rlist_regs.push_back(kX25);
    if (rlist == 14) return rlist_regs;
    rlist_regs.push_back(kX26);
    rlist_regs.push_back(kX27);
    return rlist_regs;
  }

  void ResetInstruction() {
    instruction_->DecRef();
    instruction_ = new Instruction(kInstAddress, state_);
    instruction_->set_size(4);
  }

  void ResetMemory() {
    auto *db = state_->db_factory()->Allocate<uint8_t>(0x2000);
    std::memset(db->raw_ptr(), 0, 0x2000);
    state_->StoreMemory(instruction_, 0, db);
    db->DecRef();
  }

  FlatDemandMemory *memory_;
  RiscVState *state_;
  Instruction *instruction_;
};

constexpr int kNumReg[] = {0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13};
constexpr char const *kRegMap[] = {kX1,  kX8,  kX9,  kX18, kX19, kX20, kX21,
                                   kX22, kX23, kX24, kX25, kX26, kX27};
constexpr int kStackAdjBase[] = {
    0, 0, 0, 0, 16, 16, 16, 16, 32, 32, 32, 32, 48, 48, 48, 64,
};

// The push instruction pushes a set of up to 13 registers to the stack and
// updates the stack pointer according to a combination of the rlist and spimm6
// fields.
TEST_F(RV32ZcInstructionTest, RV32ZCmpPush) {
  // Initialize the registers that will be pushed to known value.
  SetRegisterValues<uint32_t>({{kX1, 1},
                               {kX8, 2},
                               {kX9, 3},
                               {kX18, 4},
                               {kX19, 5},
                               {kX20, 6},
                               {kX21, 7},
                               {kX22, 8},
                               {kX23, 9},
                               {kX24, 10},
                               {kX25, 11},
                               {kX26, 12},
                               {kX27, 13}});
  // Test each combination of rlist and spimm6.
  for (int rlist = 4; rlist < 16; ++rlist) {
    for (int spimm6 = 0; spimm6 < 64; spimm6 += 16) {
      // Use x30 and x31 in place of spimm6 and rlist. This allows us to modify
      // the values.
      AppendRegisterOperands({kX2, kX30, kX31}, {kX2});
      // Set the registers to the values we want.
      SetRegisterValues<uint32_t>(
          {{kX2, kMemAddress}, {kX30, spimm6}, {kX31, rlist}});
      // Append the [rlist] registers.
      AppendRegisterOperands(GetRlistRegisters(rlist), {});
      SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVZCmpPush);
      instruction_->Execute(nullptr);

      // Fetch memory content.
      auto *db = state_->db_factory()->Allocate<uint32_t>(13);
      state_->LoadMemory(instruction_,
                         kMemAddress - kNumReg[rlist] * sizeof(uint32_t), db,
                         nullptr, nullptr);
      auto db_span = db->Get<uint32_t>();
      // Verify the values.
      for (int i = 0; i < 13; ++i) {
        if (i < kNumReg[rlist]) {
          EXPECT_EQ(db_span[i], i + 1)
              << "i: " << i << " rlist:" << rlist << " spimm6:" << spimm6;
        } else {
          EXPECT_EQ(db_span[i], 0)
              << "i: " << i << " rlist:" << rlist << " spimm6:" << spimm6;
        }
      }
      db->DecRef();
      // Verify the stack pointer modification.
      auto adjustment = kStackAdjBase[rlist] + spimm6;
      EXPECT_EQ(GetRegisterValue<uint32_t>(kX2), kMemAddress - adjustment)
          << "rlist: " << rlist << " spimm6: " << spimm6;

      // Clear the instruction and memory.
      ResetInstruction();
      ResetMemory();
    }
  }
}

TEST_F(RV32ZcInstructionTest, RV32ZCmpPop) {
  auto *db = state_->db_factory()->Allocate<uint32_t>(13);
  // Test each combination of rlist and spimm6.
  for (int rlist = 4; rlist < 16; ++rlist) {
    for (int spimm6 = 0; spimm6 < 64; spimm6 += 16) {
      // Append the [rlist] registers.
      AppendRegisterOperands({}, GetRlistRegisters(rlist));
      // Use x30 and x31 in place of spimm6 and rlist. This allows us to modify
      // the values.
      AppendRegisterOperands({kX2, kX30, kX31}, {kX2});
      // Set the registers to the values we want.
      SetRegisterValues<uint32_t>(
          {{kX2, kMemAddress}, {kX30, spimm6}, {kX31, rlist}});
      // Clear the registers that will be popped.
      SetRegisterValues<uint32_t>({{kX1, 0},
                                   {kX8, 0},
                                   {kX9, 0},
                                   {kX10, 0xdeadbeef},
                                   {kX18, 0},
                                   {kX19, 0},
                                   {kX20, 0},
                                   {kX21, 0},
                                   {kX22, 0},
                                   {kX23, 0},
                                   {kX24, 0},
                                   {kX25, 0},
                                   {kX26, 0},
                                   {kX27, 0}});
      // Initialize memory. Write to the memory at addresses lower than the
      // adjusted stack pointer (based on the adjustment for current rlist and
      // spimm6).
      auto adjusted_sp = kMemAddress + kStackAdjBase[rlist] + spimm6;
      auto db_span = db->Get<uint32_t>();
      for (int i = 0; i < 13; ++i) {
        db_span[i] = i + 1;
      }
      state_->StoreMemory(instruction_,
                          adjusted_sp - sizeof(uint32_t) * kNumReg[rlist], db);
      SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVZCmpPop);
      // Execute the instruction.
      instruction_->Execute(nullptr);

      // Iterate over the registers and verify the expected values.
      for (int i = 0; i < 13; ++i) {
        uint32_t value = GetRegisterValue<uint32_t>(kRegMap[i]);
        if (i < kNumReg[rlist]) {
          EXPECT_EQ(value, i + 1)
              << "i: " << i << " rlist:" << rlist << " spimm6:" << spimm6;
        } else {
          EXPECT_EQ(value, 0)
              << "i: " << i << " rlist:" << rlist << " spimm6:" << spimm6;
        }
      }
      // Verify the stack pointer modification.
      EXPECT_EQ(GetRegisterValue<uint32_t>(kX2), adjusted_sp)
          << "rlist: " << rlist << " spimm6: " << spimm6;
      // Verify that x10 is unchanged.
      EXPECT_EQ(GetRegisterValue<uint32_t>(kX10), 0xdeadbeef)
          << "rlist: " << rlist << " spimm6: " << spimm6;

      // Clear the instruction.
      ResetInstruction();
    }
  }
  db->DecRef();
}

TEST_F(RV32ZcInstructionTest, RV32ZCmpPopRet) {
  auto *db = state_->db_factory()->Allocate<uint32_t>(13);
  // Test each combination of rlist and spimm6.
  for (int rlist = 4; rlist < 16; ++rlist) {
    for (int spimm6 = 0; spimm6 < 64; spimm6 += 16) {
      // Append the [rlist] registers.
      AppendRegisterOperands({}, GetRlistRegisters(rlist));
      // Use x30 and x31 in place of spimm6 and rlist. This allows us to modify
      // the values.
      AppendRegisterOperands({kX2, kX30, kX31, kX1},
                             {kX2, RiscVState::kPcName});
      // Set the registers to the values we want.
      SetRegisterValues<uint32_t>(
          {{kX2, kMemAddress}, {kX30, spimm6}, {kX31, rlist}});
      // Clear the registers that will be popped.
      SetRegisterValues<uint32_t>({{kX1, 0},
                                   {kX8, 0},
                                   {kX9, 0},
                                   {kX10, 0xdeadbeef},
                                   {kX18, 0},
                                   {kX19, 0},
                                   {kX20, 0},
                                   {kX21, 0},
                                   {kX22, 0},
                                   {kX23, 0},
                                   {kX24, 0},
                                   {kX25, 0},
                                   {kX26, 0},
                                   {kX27, 0}});
      // Initialize memory. Write to the memory at addresses lower than the
      // adjusted stack pointer (based on the adjustment for current rlist and
      // spimm6).
      auto adjusted_sp = kMemAddress + kStackAdjBase[rlist] + spimm6;
      auto db_span = db->Get<uint32_t>();
      for (int i = 0; i < 13; ++i) {
        db_span[i] = i + 1;
      }
      state_->StoreMemory(instruction_,
                          adjusted_sp - sizeof(uint32_t) * kNumReg[rlist], db);
      SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVZCmpPopRet);
      // Execute the instruction.
      instruction_->Execute(nullptr);

      // Iterate over the registers and verify the expected values.
      for (int i = 0; i < 13; ++i) {
        uint32_t value = GetRegisterValue<uint32_t>(kRegMap[i]);
        if (i < kNumReg[rlist]) {
          EXPECT_EQ(value, i + 1)
              << "i: " << i << " rlist:" << rlist << " spimm6:" << spimm6;
        } else {
          EXPECT_EQ(value, 0)
              << "i: " << i << " rlist:" << rlist << " spimm6:" << spimm6;
        }
      }
      // Verify the stack pointer modification.
      EXPECT_EQ(GetRegisterValue<uint32_t>(kX2), adjusted_sp)
          << "rlist: " << rlist << " spimm6: " << spimm6;
      // Verify that x10 is unchanged.
      EXPECT_EQ(GetRegisterValue<uint32_t>(kX10), 0xdeadbeef)
          << "rlist: " << rlist << " spimm6: " << spimm6;
      // Verify that the PC is set to ra (X1), i.e., 1.
      EXPECT_EQ(GetRegisterValue<uint32_t>(RiscVState::kPcName), 1);
      // Clear the instruction.
      ResetInstruction();
    }
  }
  db->DecRef();
}

TEST_F(RV32ZcInstructionTest, RV32ZCmpPopRetz) {
  auto *db = state_->db_factory()->Allocate<uint32_t>(13);
  // Test each combination of rlist and spimm6.
  for (int rlist = 4; rlist < 16; ++rlist) {
    for (int spimm6 = 0; spimm6 < 64; spimm6 += 16) {
      // Append the [rlist] registers.
      AppendRegisterOperands({}, GetRlistRegisters(rlist));
      // Use x30 and x31 in place of spimm6 and rlist. This allows us to modify
      // the values.
      AppendRegisterOperands({kX2, kX30, kX31, kX1},
                             {kX2, kX10, RiscVState::kPcName});
      // Set the registers to the values we want.
      SetRegisterValues<uint32_t>(
          {{kX2, kMemAddress}, {kX30, spimm6}, {kX31, rlist}});
      // Clear the registers that will be popped.
      SetRegisterValues<uint32_t>({{kX1, 0},
                                   {kX8, 0},
                                   {kX9, 0},
                                   {kX10, 0xdeadbeef},
                                   {kX18, 0},
                                   {kX19, 0},
                                   {kX20, 0},
                                   {kX21, 0},
                                   {kX22, 0},
                                   {kX23, 0},
                                   {kX24, 0},
                                   {kX25, 0},
                                   {kX26, 0},
                                   {kX27, 0}});
      // Initialize memory. Write to the memory at addresses lower than the
      // adjusted stack pointer (based on the adjustment for current rlist and
      // spimm6).
      auto adjusted_sp = kMemAddress + kStackAdjBase[rlist] + spimm6;
      auto db_span = db->Get<uint32_t>();
      for (int i = 0; i < 13; ++i) {
        db_span[i] = i + 1;
      }
      state_->StoreMemory(instruction_,
                          adjusted_sp - sizeof(uint32_t) * kNumReg[rlist], db);
      SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVZCmpPopRetz);
      // Execute the instruction.
      instruction_->Execute(nullptr);

      // Iterate over the registers and verify the expected values.
      for (int i = 0; i < 13; ++i) {
        uint32_t value = GetRegisterValue<uint32_t>(kRegMap[i]);
        if (i < kNumReg[rlist]) {
          EXPECT_EQ(value, i + 1)
              << "i: " << i << " rlist:" << rlist << " spimm6:" << spimm6;
        } else {
          EXPECT_EQ(value, 0)
              << "i: " << i << " rlist:" << rlist << " spimm6:" << spimm6;
        }
      }
      // Verify the stack pointer modification.
      EXPECT_EQ(GetRegisterValue<uint32_t>(kX2), adjusted_sp)
          << "rlist: " << rlist << " spimm6: " << spimm6;
      // Verify that x10 is zeroed.
      EXPECT_EQ(GetRegisterValue<uint32_t>(kX10), 0x0)
          << "rlist: " << rlist << " spimm6: " << spimm6;
      // Verify that the PC is set to ra (X1), i.e., 1.
      EXPECT_EQ(GetRegisterValue<uint32_t>(RiscVState::kPcName), 1);
      // Clear the instruction.
      ResetInstruction();
    }
  }
  db->DecRef();
}

TEST_F(RV32ZcInstructionTest, RV32ZMvTwoRegs) {
  AppendRegisterOperands({kX1, kX2}, {kX10, kX11});
  SetRegisterValues<uint32_t>({{kX1, 1}, {kX2, 2}, {kX10, 10}, {kX11, 11}});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVZCmpMvTwoRegs);
  instruction_->Execute(nullptr);

  EXPECT_EQ(GetRegisterValue<uint32_t>(kX10), 1);
  EXPECT_EQ(GetRegisterValue<uint32_t>(kX11), 2);
}

TEST_F(RV32ZcInstructionTest, RV32ZCmtJt) {
  // Use a register instead of the immediate for the index.
  AppendRegisterOperands({kX10}, {RiscVState::kPcName});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVZCmtJt);
  // Indices have to be less than 32.
  for (int i = 0; i < 16; ++i) {
    SetRegisterValues<uint32_t>({{kX10, i}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<uint32_t>(RiscVState::kPcName),
              0x8000 + i * sizeof(uint64_t))
        << "i: " << i;
  }
}

TEST_F(RV32ZcInstructionTest, RV32ZCmtJalt) {
  // Use a register instead of the immediate for the index.
  AppendRegisterOperands({kX10}, {RiscVState::kPcName, kX1});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVZCmtJalt);
  // Indices have to be greater or equal to 32.
  for (int i = 32; i < 48; ++i) {
    SetRegisterValues<uint32_t>({{kX10, i}, {kX1, 0}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<uint32_t>(RiscVState::kPcName),
              0x8000 + i * sizeof(uint64_t))
        << "i: " << i;
    EXPECT_EQ(GetRegisterValue<uint32_t>(kX1),
              instruction_->address() + instruction_->size());
  }
}

}  // namespace
