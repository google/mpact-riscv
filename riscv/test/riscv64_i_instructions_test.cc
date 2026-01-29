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
#include <string>
#include <tuple>
#include <vector>

#include "absl/strings/string_view.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_i_instructions.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"

// This file contains tests for individual RiscV64I instructions.

namespace {

using ::mpact::sim::generic::ImmediateOperand;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::RV64Register;
using ::mpact::sim::util::FlatDemandMemory;

constexpr char kX1[] = "x1";
constexpr char kX3[] = "x3";

constexpr uint32_t kInstAddress = 0x2468;
constexpr uint32_t kOffset = 0x246;
constexpr uint32_t kMemAddress = 0x1000;

class RV64IInstructionTest : public testing::Test {
 public:
  RV64IInstructionTest() {
    memory_ = new FlatDemandMemory();
    state_ = new RiscVState("test", RiscVXlen::RV64, memory_);
    instruction_ = new Instruction(kInstAddress, state_);
    instruction_->set_size(4);
    state_->set_on_trap([this](bool is_interrupt, uint64_t trap_value,
                               uint64_t exception_code, uint64_t epc,
                               const Instruction* inst) {
      trap_taken_ = true;
      trap_is_interrupt_ = is_interrupt;
      trap_value_ = trap_value;
      trap_code_ = exception_code;
      trap_epc_ = epc;
      return true;
    });
  }
  ~RV64IInstructionTest() override {
    delete memory_;
    delete state_;
    delete instruction_;
  }

  // Appends the source and destination operands for the register names
  // given in the two vectors.
  void AppendRegisterOperands(Instruction* inst,
                              const std::vector<std::string>& sources,
                              const std::vector<std::string>& destinations) {
    for (auto& reg_name : sources) {
      auto* reg = state_->GetRegister<RV64Register>(reg_name).first;
      inst->AppendSource(reg->CreateSourceOperand());
    }
    for (auto& reg_name : destinations) {
      auto* reg = state_->GetRegister<RV64Register>(reg_name).first;
      inst->AppendDestination(reg->CreateDestinationOperand(0));
    }
  }

  void AppendRegisterOperands(const std::vector<std::string>& sources,
                              const std::vector<std::string>& destinations) {
    AppendRegisterOperands(instruction_, sources, destinations);
  }

  // Appends immediate source operands with the given values.
  template <typename T>
  void AppendImmediateOperands(const std::vector<T>& values) {
    for (auto value : values) {
      auto* src = new ImmediateOperand<T>(value);
      instruction_->AppendSource(src);
    }
  }

  // Takes a vector of tuples of register names and values. Fetches each
  // named register and sets it to the corresponding value.
  template <typename T>
  void SetRegisterValues(const std::vector<std::tuple<std::string, T>> values) {
    for (auto& [reg_name, value] : values) {
      auto* reg = state_->GetRegister<RV64Register>(reg_name).first;
      auto* db = state_->db_factory()->Allocate<RV64Register::ValueType>(1);
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
    auto* reg = state_->GetRegister<RV64Register>(reg_name).first;
    return reg->data_buffer()->Get<T>(0);
  }

  FlatDemandMemory* memory_;
  RiscVState* state_;
  Instruction* instruction_;
  bool trap_taken_ = false;
  uint64_t trap_epc_ = 0;
  uint64_t trap_value_ = 0;
  uint64_t trap_code_ = 0;
  bool trap_is_interrupt_ = false;
};

TEST_F(RV64IInstructionTest, RV64IJal) {
  AppendRegisterOperands({}, {RiscVState::kPcName, kX3});
  AppendImmediateOperands<uint64_t>({kOffset});
  SetSemanticFunction(&::mpact::sim::riscv::RV64::RiscVIJal);

  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint64_t>(RiscVState::kPcName),
            instruction_->address() + kOffset);
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), instruction_->address() + 4);
}

TEST_F(RV64IInstructionTest, RV64IJalMisaligned) {
  AppendRegisterOperands({}, {RiscVState::kPcName, kX3});
  AppendImmediateOperands<uint64_t>({2});
  SetSemanticFunction(&::mpact::sim::riscv::RV64::RiscVIJal);
  // Get misa and clear C bit
  auto misa = state_->csr_set()
                  ->GetCsr(static_cast<uint64_t>(
                      ::mpact::sim::riscv::RiscVCsrEnum::kMIsa))
                  .value();
  misa->Set(
      misa->GetUint64() &
      ~static_cast<uint64_t>(::mpact::sim::riscv::IsaExtensions::kCompressed));
  trap_taken_ = false;
  instruction_->Execute(nullptr);
  EXPECT_TRUE(trap_taken_);
  EXPECT_FALSE(trap_is_interrupt_);
  EXPECT_EQ(
      trap_code_,
      static_cast<uint64_t>(
          ::mpact::sim::riscv::ExceptionCode::kInstructionAddressMisaligned));
  EXPECT_EQ(trap_epc_, instruction_->address());
}

TEST_F(RV64IInstructionTest, RV64IJalr) {
  AppendRegisterOperands({kX1}, {RiscVState::kPcName, kX3});
  AppendImmediateOperands<uint64_t>({kOffset});
  SetSemanticFunction(&::mpact::sim::riscv::RV64::RiscVIJalr);

  SetRegisterValues<uint64_t>({{kX1, kMemAddress}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<uint64_t>(RiscVState::kPcName),
            kOffset + kMemAddress);
  EXPECT_EQ(GetRegisterValue<uint64_t>(kX3), instruction_->address() + 4);
}

TEST_F(RV64IInstructionTest, RV64IJalrMisaligned) {
  AppendRegisterOperands({kX1}, {RiscVState::kPcName, kX3});
  AppendImmediateOperands<uint64_t>({2});
  SetSemanticFunction(&::mpact::sim::riscv::RV64::RiscVIJalr);
  SetRegisterValues<uint64_t>({{kX1, 0}});
  // Get misa and clear C bit
  auto misa = state_->csr_set()
                  ->GetCsr(static_cast<uint64_t>(
                      ::mpact::sim::riscv::RiscVCsrEnum::kMIsa))
                  .value();
  misa->Set(
      misa->GetUint64() &
      ~static_cast<uint64_t>(::mpact::sim::riscv::IsaExtensions::kCompressed));
  trap_taken_ = false;
  instruction_->Execute(nullptr);
  EXPECT_TRUE(trap_taken_);
  EXPECT_FALSE(trap_is_interrupt_);
  EXPECT_EQ(
      trap_code_,
      static_cast<uint64_t>(
          ::mpact::sim::riscv::ExceptionCode::kInstructionAddressMisaligned));
  EXPECT_EQ(trap_epc_, instruction_->address());
}

}  // namespace