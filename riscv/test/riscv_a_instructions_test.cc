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

#include "riscv/riscv_a_instructions.h"

#include <cstdint>
#include <ios>
#include <string>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "absl/log/check.h"
#include "riscv/riscv_i_instructions.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/atomic_memory.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace {

using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::RV64Register;
using ::mpact::sim::util::AtomicMemory;
using ::mpact::sim::util::FlatDemandMemory;

using Operation = ::mpact::sim::util::AtomicMemoryOpInterface::Operation;

// Instruction semantic functions to test.
using ::mpact::sim::riscv::AAmoaddd;
using ::mpact::sim::riscv::AAmoaddw;
using ::mpact::sim::riscv::AAmoandd;
using ::mpact::sim::riscv::AAmoandw;
using ::mpact::sim::riscv::AAmomaxd;
using ::mpact::sim::riscv::AAmomaxud;
using ::mpact::sim::riscv::AAmomaxuw;
using ::mpact::sim::riscv::AAmomaxw;
using ::mpact::sim::riscv::AAmomind;
using ::mpact::sim::riscv::AAmominud;
using ::mpact::sim::riscv::AAmominuw;
using ::mpact::sim::riscv::AAmominw;
using ::mpact::sim::riscv::AAmoord;
using ::mpact::sim::riscv::AAmoorw;
using ::mpact::sim::riscv::AAmoswapd;
using ::mpact::sim::riscv::AAmoswapw;
using ::mpact::sim::riscv::AAmoxord;
using ::mpact::sim::riscv::AAmoxorw;
using ::mpact::sim::riscv::ALrd;
using ::mpact::sim::riscv::ALrw;
using ::mpact::sim::riscv::AScd;
using ::mpact::sim::riscv::AScw;

// The load data write back semantic functions.
using ::mpact::sim::riscv::RV64::RiscVILdChild;
using ::mpact::sim::riscv::RV64::RiscVILwChild;

// Register names.
constexpr char kX1[] = "x1";
constexpr char kX2[] = "x2";
constexpr char kX3[] = "x3";
constexpr char kX4[] = "x4";
constexpr char kX5[] = "x5";

// Common values used in the test.
constexpr uint64_t kInstAddress = 0x2468;
constexpr uint64_t kWMemAddress = 0x1000;
constexpr uint64_t kDMemAddress = 0x2000;
constexpr uint32_t kWMemContent = 0xDEADBEEF;
constexpr uint64_t kWRegContent = 0xFFFFFFFF'DEADBEEF;
constexpr uint64_t kDMemContent = 0xDEADBEEF'FEEDC0DE;
constexpr uint32_t kWA5 = 0xA5A5'5A5A;
constexpr uint64_t kDA5 = 0xA5A5'5A5A'A5A5'5A5A;

class RiscVAInstructionsTest : public ::testing::Test {
 protected:
  RiscVAInstructionsTest() {
    // Create memory objects.
    memory_ = new FlatDemandMemory();
    atomic_memory_ = new AtomicMemory(memory_);
    // Create and initialize state and instruction objects.
    state_ = new RiscVState("test", RiscVXlen::RV64, memory_, atomic_memory_);
    child_instruction_ = new Instruction(state_);
    instruction_ = new Instruction(kInstAddress, state_);
    instruction_->set_size(4);
    instruction_->AppendChild(child_instruction_);
    child_instruction_->DecRef();
    // Set the memory locations to known values.
    db_w_ = state_->db_factory()->Allocate<uint32_t>(1);
    db_w_->Set<uint32_t>(0, kWMemContent);
    db_d_ = state_->db_factory()->Allocate<uint64_t>(1);
    db_d_->Set<uint64_t>(0, kDMemContent);
    memory_->Store(kWMemAddress, db_w_);
    memory_->Store(kDMemAddress, db_d_);
  }

  ~RiscVAInstructionsTest() override {
    db_w_->DecRef();
    db_d_->DecRef();
    instruction_->DecRef();
    delete state_;
    delete atomic_memory_;
    delete memory_;
  }

  // Appends the source and destination operands for the register names
  // given in the two vectors.
  void AppendRegisterOperands(Instruction *inst,
                              const std::vector<std::string> &sources,
                              const std::vector<std::string> &destinations) {
    for (auto &reg_name : sources) {
      auto *reg = state_->GetRegister<RV64Register>(reg_name).first;
      inst->AppendSource(reg->CreateSourceOperand());
    }
    for (auto &reg_name : destinations) {
      auto *reg = state_->GetRegister<RV64Register>(reg_name).first;
      inst->AppendDestination(reg->CreateDestinationOperand(0));
    }
  }

  void AppendRegisterOperands(const std::vector<std::string> &sources,
                              const std::vector<std::string> &destinations) {
    AppendRegisterOperands(instruction_, sources, destinations);
  }

  // Takes a vector of tuples of register names and values. Fetches each
  // named register and sets it to the corresponding value.
  template <typename T>
  void SetRegisterValues(const std::vector<std::tuple<std::string, T>> values) {
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
  // Initializes the semantic function of the instruction object.
  void SetChildSemanticFunction(Instruction::SemanticFunction fcn) {
    child_instruction_->set_semantic_function(fcn);
  }

  // Returns the value of the named register.
  template <typename T>
  T GetRegisterValue(absl::string_view reg_name) {
    auto *reg = state_->GetRegister<RV64Register>(reg_name).first;
    if (std::is_signed<T>::value) {
      return static_cast<T>(reg->data_buffer()->Get<int64_t>(0));
    } else {
      return static_cast<T>(reg->data_buffer()->Get<uint64_t>(0));
    }
  }

  DataBuffer *db_w_;
  DataBuffer *db_d_;
  FlatDemandMemory *memory_;
  AtomicMemory *atomic_memory_;
  RiscVState *state_;
  Instruction *instruction_;
  Instruction *child_instruction_;
};

TEST_F(RiscVAInstructionsTest, ALrw) {
  AppendRegisterOperands({kX1, kX4, kX5}, {});
  AppendRegisterOperands(child_instruction_, {}, {kX3});
  SetRegisterValues<uint64_t>(
      {{kX1, kWMemAddress}, {kX4, 0}, {kX5, 0}, {kX3, 0}});
  SetSemanticFunction(&ALrw);
  SetChildSemanticFunction(&RiscVILwChild);
  instruction_->Execute();
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), kWRegContent)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
}

TEST_F(RiscVAInstructionsTest, ALrd) {
  AppendRegisterOperands({kX1, kX4, kX5}, {});
  AppendRegisterOperands(child_instruction_, {}, {kX3});
  SetRegisterValues<uint64_t>(
      {{kX1, kDMemAddress}, {kX4, 0}, {kX5, 0}, {kX3, 0}});
  SetSemanticFunction(&ALrd);
  SetChildSemanticFunction(&RiscVILdChild);
  instruction_->Execute();
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), kDMemContent)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
}

TEST_F(RiscVAInstructionsTest, AScw) {
  AppendRegisterOperands({kX1, kX2, kX4, kX5}, {});
  AppendRegisterOperands(child_instruction_, {}, {kX3});
  SetRegisterValues<uint64_t>(
      {{kX1, kWMemAddress}, {kX2, 1}, {kX3, 0}, {kX4, 0}, {kX5, 0}});
  SetSemanticFunction(&AScw);
  SetChildSemanticFunction(&RiscVILwChild);
  instruction_->Execute();
  // The Store conditional fails without a prior Load linked.
  EXPECT_NE(GetRegisterValue<uint64_t>(kX3), 0)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
  // The memory location will have the old value.
  memory_->Load(kWMemAddress, db_w_, nullptr, nullptr);
  EXPECT_EQ(db_w_->Get<uint32_t>(0), kWMemContent)
      << std::hex << db_w_->Get<uint32_t>(0);

  // Perform a load linked first to the address.
  CHECK_OK(atomic_memory_->PerformMemoryOp(kWMemAddress, Operation::kLoadLinked,
                                           db_w_, nullptr, nullptr));
  // Now run the instruction again.
  instruction_->Execute();
  // The Store conditional succeeds.
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), 0)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
  // The memory location will have the old value.
  memory_->Load(kWMemAddress, db_w_, nullptr, nullptr);
  EXPECT_EQ(db_w_->Get<uint32_t>(0), 1) << std::hex << db_w_->Get<uint32_t>(0);
}

TEST_F(RiscVAInstructionsTest, AScd) {
  AppendRegisterOperands({kX1, kX2, kX4, kX5}, {});
  AppendRegisterOperands(child_instruction_, {}, {kX3});
  SetRegisterValues<uint64_t>(
      {{kX1, kDMemAddress}, {kX2, 2}, {kX3, 0}, {kX4, 0}, {kX5, 0}});
  SetSemanticFunction(&AScd);
  SetChildSemanticFunction(&RiscVILdChild);
  instruction_->Execute();
  // The Store conditional fails without a prior Load linked.
  EXPECT_NE(GetRegisterValue<uint64_t>(kX3), 0)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
  // The memory location will have the old value.
  memory_->Load(kDMemAddress, db_d_, nullptr, nullptr);
  EXPECT_EQ(db_d_->Get<uint64_t>(0), kDMemContent)
      << std::hex << db_d_->Get<uint64_t>(0);

  // Perform a load linked first to the address.
  CHECK_OK(atomic_memory_->PerformMemoryOp(kDMemAddress, Operation::kLoadLinked,
                                           db_d_, nullptr, nullptr));
  // Now run the instruction again.
  instruction_->Execute();
  // The Store conditional succeeds.
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), 0)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
  // The memory location will have the old value.
  memory_->Load(kDMemAddress, db_d_, nullptr, nullptr);
  EXPECT_EQ(db_d_->Get<uint64_t>(0), 2) << std::hex << db_w_->Get<uint64_t>(0);
}

TEST_F(RiscVAInstructionsTest, AAmoswapw) {
  AppendRegisterOperands({kX1, kX2, kX4, kX5}, {});
  AppendRegisterOperands(child_instruction_, {}, {kX3});
  SetRegisterValues<uint64_t>(
      {{kX1, kWMemAddress}, {kX2, 1}, {kX3, 0}, {kX4, 0}, {kX5, 0}});
  SetSemanticFunction(&AAmoswapw);
  SetChildSemanticFunction(&RiscVILwChild);
  instruction_->Execute();
  // The memory value should now be in the register.
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), kWRegContent)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
  // The memory location will have the swap value.
  memory_->Load(kWMemAddress, db_w_, nullptr, nullptr);
  EXPECT_EQ(db_w_->Get<uint32_t>(0), 1) << std::hex << db_w_->Get<uint32_t>(0);
}

TEST_F(RiscVAInstructionsTest, AAmoswapd) {
  AppendRegisterOperands({kX1, kX2, kX4, kX5}, {});
  AppendRegisterOperands(child_instruction_, {}, {kX3});
  SetRegisterValues<uint64_t>(
      {{kX1, kDMemAddress}, {kX2, 2}, {kX3, 0}, {kX4, 0}, {kX5, 0}});
  SetSemanticFunction(&AAmoswapd);
  SetChildSemanticFunction(&RiscVILdChild);
  instruction_->Execute();
  // The memory value should now be in the register.
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), kDMemContent)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
  // The memory location will have the swap value.
  memory_->Load(kDMemAddress, db_d_, nullptr, nullptr);
  EXPECT_EQ(db_d_->Get<uint64_t>(0), 2) << std::hex << db_d_->Get<uint64_t>(0);
}

TEST_F(RiscVAInstructionsTest, AAamoaddw) {
  AppendRegisterOperands({kX1, kX2, kX4, kX5}, {});
  AppendRegisterOperands(child_instruction_, {}, {kX3});
  SetRegisterValues<uint64_t>(
      {{kX1, kWMemAddress}, {kX2, 1}, {kX3, 0}, {kX4, 0}, {kX5, 0}});
  SetSemanticFunction(&AAmoaddw);
  SetChildSemanticFunction(&RiscVILwChild);
  instruction_->Execute();
  // The memory value should now be in the register.
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), kWRegContent)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
  // The memory location will have the new value.
  memory_->Load(kWMemAddress, db_w_, nullptr, nullptr);
  EXPECT_EQ(db_w_->Get<uint32_t>(0), kWMemContent + 1)
      << std::hex << db_w_->Get<uint32_t>(0);
}

TEST_F(RiscVAInstructionsTest, AAamoaddd) {
  AppendRegisterOperands({kX1, kX2, kX4, kX5}, {});
  AppendRegisterOperands(child_instruction_, {}, {kX3});
  SetRegisterValues<uint64_t>(
      {{kX1, kDMemAddress}, {kX2, 2}, {kX3, 0}, {kX4, 0}, {kX5, 0}});
  SetSemanticFunction(&AAmoaddd);
  SetChildSemanticFunction(&RiscVILdChild);
  instruction_->Execute();
  // The memory value should now be in the register.
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), kDMemContent)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
  // The memory location will have the new value.
  memory_->Load(kDMemAddress, db_d_, nullptr, nullptr);
  EXPECT_EQ(db_d_->Get<uint64_t>(0), kDMemContent + 2)
      << std::hex << db_d_->Get<uint64_t>(0);
}

TEST_F(RiscVAInstructionsTest, AAamoandw) {
  AppendRegisterOperands({kX1, kX2, kX4, kX5}, {});
  AppendRegisterOperands(child_instruction_, {}, {kX3});
  SetRegisterValues<uint64_t>(
      {{kX1, kWMemAddress}, {kX2, kWA5}, {kX3, 0}, {kX4, 0}, {kX5, 0}});
  SetSemanticFunction(&AAmoandw);
  SetChildSemanticFunction(&RiscVILwChild);
  instruction_->Execute();
  // The memory value should now be in the register.
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), kWRegContent)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
  // The memory location will have the new value.
  memory_->Load(kWMemAddress, db_w_, nullptr, nullptr);
  EXPECT_EQ(db_w_->Get<uint32_t>(0), kWMemContent & kWA5)
      << std::hex << db_w_->Get<uint32_t>(0);
}

TEST_F(RiscVAInstructionsTest, AAamoandd) {
  AppendRegisterOperands({kX1, kX2, kX4, kX5}, {});
  AppendRegisterOperands(child_instruction_, {}, {kX3});
  SetRegisterValues<uint64_t>(
      {{kX1, kDMemAddress}, {kX2, kDA5}, {kX3, 0}, {kX4, 0}, {kX5, 0}});
  SetSemanticFunction(&AAmoandd);
  SetChildSemanticFunction(&RiscVILdChild);
  instruction_->Execute();
  // The memory value should now be in the register.
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), kDMemContent)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
  // The memory location will have the new value.
  memory_->Load(kDMemAddress, db_d_, nullptr, nullptr);
  EXPECT_EQ(db_d_->Get<uint64_t>(0), kDMemContent & kDA5)
      << std::hex << db_d_->Get<uint64_t>(0);
}

TEST_F(RiscVAInstructionsTest, AAmodorw) {
  AppendRegisterOperands({kX1, kX2, kX4, kX5}, {});
  AppendRegisterOperands(child_instruction_, {}, {kX3});
  SetRegisterValues<uint64_t>(
      {{kX1, kWMemAddress}, {kX2, kWA5}, {kX3, 0}, {kX4, 0}, {kX5, 0}});
  SetSemanticFunction(&AAmoorw);
  SetChildSemanticFunction(&RiscVILwChild);
  instruction_->Execute();
  // The memory value should now be in the register.
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), kWRegContent)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
  // The memory location will have the new value.
  memory_->Load(kWMemAddress, db_w_, nullptr, nullptr);
  EXPECT_EQ(db_w_->Get<uint32_t>(0), kWMemContent | kWA5)
      << std::hex << db_w_->Get<uint32_t>(0);
}

TEST_F(RiscVAInstructionsTest, AAmodord) {
  AppendRegisterOperands({kX1, kX2, kX4, kX5}, {});
  AppendRegisterOperands(child_instruction_, {}, {kX3});
  SetRegisterValues<uint64_t>(
      {{kX1, kDMemAddress}, {kX2, kDA5}, {kX3, 0}, {kX4, 0}, {kX5, 0}});
  SetSemanticFunction(&AAmoord);
  SetChildSemanticFunction(&RiscVILdChild);
  instruction_->Execute();
  // The memory value should now be in the register.
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), kDMemContent)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
  // The memory location will have the new value.
  memory_->Load(kDMemAddress, db_d_, nullptr, nullptr);
  EXPECT_EQ(db_d_->Get<uint64_t>(0), kDMemContent | kDA5)
      << std::hex << db_d_->Get<uint64_t>(0);
}

TEST_F(RiscVAInstructionsTest, AAmoxorw) {
  AppendRegisterOperands({kX1, kX2, kX4, kX5}, {});
  AppendRegisterOperands(child_instruction_, {}, {kX3});
  SetRegisterValues<uint64_t>(
      {{kX1, kWMemAddress}, {kX2, kWA5}, {kX3, 0}, {kX4, 0}, {kX5, 0}});
  SetSemanticFunction(&AAmoxorw);
  SetChildSemanticFunction(&RiscVILwChild);
  instruction_->Execute();
  // The memory value should now be in the register.
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), kWRegContent)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
  // The memory location will have the new value.
  memory_->Load(kWMemAddress, db_w_, nullptr, nullptr);
  EXPECT_EQ(db_w_->Get<uint32_t>(0), kWMemContent ^ kWA5)
      << std::hex << db_w_->Get<uint64_t>(0);
}

TEST_F(RiscVAInstructionsTest, AAmoxord) {
  AppendRegisterOperands({kX1, kX2, kX4, kX5}, {});
  AppendRegisterOperands(child_instruction_, {}, {kX3});
  SetRegisterValues<uint64_t>(
      {{kX1, kDMemAddress}, {kX2, kDA5}, {kX3, 0}, {kX4, 0}, {kX5, 0}});
  SetSemanticFunction(&AAmoxord);
  SetChildSemanticFunction(&RiscVILdChild);
  instruction_->Execute();
  // The memory value should now be in the register.
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), kDMemContent)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
  // The memory location will have the new value.
  memory_->Load(kDMemAddress, db_d_, nullptr, nullptr);
  EXPECT_EQ(db_d_->Get<uint64_t>(0), kDMemContent ^ kDA5)
      << std::hex << db_d_->Get<uint64_t>(0);
}

TEST_F(RiscVAInstructionsTest, AAmomaxw) {
  AppendRegisterOperands({kX1, kX2, kX4, kX5}, {});
  AppendRegisterOperands(child_instruction_, {}, {kX3});
  SetRegisterValues<uint64_t>(
      {{kX1, kWMemAddress}, {kX2, kWA5}, {kX3, 0}, {kX4, 0}, {kX5, 0}});
  SetSemanticFunction(&AAmomaxw);
  SetChildSemanticFunction(&RiscVILwChild);
  instruction_->Execute();
  // The memory value should now be in the register.
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), kWRegContent)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
  // The memory location will have the new value.
  memory_->Load(kWMemAddress, db_w_, nullptr, nullptr);
  EXPECT_EQ(db_w_->Get<int32_t>(0), std::max(static_cast<int32_t>(kWMemContent),
                                             static_cast<int32_t>(kWA5)))
      << std::hex << db_w_->Get<uint32_t>(0);
}

TEST_F(RiscVAInstructionsTest, AAmomaxd) {
  AppendRegisterOperands({kX1, kX2, kX4, kX5}, {});
  AppendRegisterOperands(child_instruction_, {}, {kX3});
  SetRegisterValues<uint64_t>(
      {{kX1, kDMemAddress}, {kX2, kDA5}, {kX3, 0}, {kX4, 0}, {kX5, 0}});
  SetSemanticFunction(&AAmomaxd);
  SetChildSemanticFunction(&RiscVILdChild);
  instruction_->Execute();
  // The memory value should now be in the register.
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), kDMemContent)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
  // The memory location will have the new value.
  memory_->Load(kDMemAddress, db_d_, nullptr, nullptr);
  EXPECT_EQ(db_d_->Get<int64_t>(0), std::max(static_cast<int64_t>(kDMemContent),
                                             static_cast<int64_t>(kDA5)))
      << std::hex << db_d_->Get<uint64_t>(0);
}

TEST_F(RiscVAInstructionsTest, AAmomaxuw) {
  AppendRegisterOperands({kX1, kX2, kX4, kX5}, {});
  AppendRegisterOperands(child_instruction_, {}, {kX3});
  SetRegisterValues<uint64_t>(
      {{kX1, kWMemAddress}, {kX2, kWA5}, {kX3, 0}, {kX4, 0}, {kX5, 0}});
  SetSemanticFunction(&AAmomaxuw);
  SetChildSemanticFunction(&RiscVILwChild);
  instruction_->Execute();
  // The memory value should now be in the register.
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), kWRegContent)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
  // The memory location will have the new value.
  memory_->Load(kWMemAddress, db_w_, nullptr, nullptr);
  EXPECT_EQ(db_w_->Get<uint32_t>(0),
            std::max(static_cast<uint32_t>(kWMemContent),
                     static_cast<uint32_t>(kWA5)))
      << std::hex << db_w_->Get<uint32_t>(0);
}

TEST_F(RiscVAInstructionsTest, AAmomaxud) {
  AppendRegisterOperands({kX1, kX2, kX4, kX5}, {});
  AppendRegisterOperands(child_instruction_, {}, {kX3});
  SetRegisterValues<uint64_t>(
      {{kX1, kDMemAddress}, {kX2, kDA5}, {kX3, 0}, {kX4, 0}, {kX5, 0}});
  SetSemanticFunction(&AAmomaxud);
  SetChildSemanticFunction(&RiscVILdChild);
  instruction_->Execute();
  // The memory value should now be in the register.
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), kDMemContent)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
  // The memory location will have the new value.
  memory_->Load(kDMemAddress, db_d_, nullptr, nullptr);
  EXPECT_EQ(db_d_->Get<uint64_t>(0),
            std::max(static_cast<uint64_t>(kDMemContent),
                     static_cast<uint64_t>(kDA5)))
      << std::hex << db_d_->Get<uint64_t>(0);
}

TEST_F(RiscVAInstructionsTest, AAmominw) {
  AppendRegisterOperands({kX1, kX2, kX4, kX5}, {});
  AppendRegisterOperands(child_instruction_, {}, {kX3});
  SetRegisterValues<uint64_t>(
      {{kX1, kWMemAddress}, {kX2, kWA5}, {kX3, 0}, {kX4, 0}, {kX5, 0}});
  SetSemanticFunction(&AAmominw);
  SetChildSemanticFunction(&RiscVILwChild);
  instruction_->Execute();
  // The memory value should now be in the register.
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), kWRegContent)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
  // The memory location will have the new value.
  memory_->Load(kWMemAddress, db_w_, nullptr, nullptr);
  EXPECT_EQ(db_w_->Get<int32_t>(0), std::min(static_cast<int32_t>(kWMemContent),
                                             static_cast<int32_t>(kWA5)))
      << std::hex << db_w_->Get<uint32_t>(0);
}

TEST_F(RiscVAInstructionsTest, AAmomind) {
  AppendRegisterOperands({kX1, kX2, kX4, kX5}, {});
  AppendRegisterOperands(child_instruction_, {}, {kX3});
  SetRegisterValues<uint64_t>(
      {{kX1, kDMemAddress}, {kX2, kDA5}, {kX3, 0}, {kX4, 0}, {kX5, 0}});
  SetSemanticFunction(&AAmomind);
  SetChildSemanticFunction(&RiscVILdChild);
  instruction_->Execute();
  // The memory value should now be in the register.
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), kDMemContent)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
  // The memory location will have the new value.
  memory_->Load(kDMemAddress, db_d_, nullptr, nullptr);
  EXPECT_EQ(db_d_->Get<int64_t>(0), std::min(static_cast<int64_t>(kDMemContent),
                                             static_cast<int64_t>(kDA5)))
      << std::hex << db_d_->Get<uint64_t>(0);
}

TEST_F(RiscVAInstructionsTest, AAmominuw) {
  AppendRegisterOperands({kX1, kX2, kX4, kX5}, {});
  AppendRegisterOperands(child_instruction_, {}, {kX3});
  SetRegisterValues<uint64_t>(
      {{kX1, kWMemAddress}, {kX2, kWA5}, {kX3, 0}, {kX4, 0}, {kX5, 0}});
  SetSemanticFunction(&AAmominuw);
  SetChildSemanticFunction(&RiscVILwChild);
  instruction_->Execute();
  // The memory value should now be in the register.
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), kWRegContent)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
  // The memory location will have the new value.
  memory_->Load(kWMemAddress, db_w_, nullptr, nullptr);
  EXPECT_EQ(db_w_->Get<uint32_t>(0),
            std::min(static_cast<uint32_t>(kWMemContent),
                     static_cast<uint32_t>(kWA5)))
      << std::hex << db_w_->Get<uint32_t>(0);
}

TEST_F(RiscVAInstructionsTest, AAmominud) {
  AppendRegisterOperands({kX1, kX2, kX4, kX5}, {});
  AppendRegisterOperands(child_instruction_, {}, {kX3});
  SetRegisterValues<uint64_t>(
      {{kX1, kDMemAddress}, {kX2, kDA5}, {kX3, 0}, {kX4, 0}, {kX5, 0}});
  SetSemanticFunction(&AAmominud);
  SetChildSemanticFunction(&RiscVILdChild);
  instruction_->Execute();
  // The memory value should now be in the register.
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), kDMemContent)
      << std::hex << GetRegisterValue<uint64_t>(kX3);
  // The memory location will have the new value.
  memory_->Load(kDMemAddress, db_d_, nullptr, nullptr);
  EXPECT_EQ(db_d_->Get<uint64_t>(0),
            std::min(static_cast<uint64_t>(kDMemContent),
                     static_cast<uint64_t>(kDA5)))
      << std::hex << db_d_->Get<uint64_t>(0);
}

}  // namespace
