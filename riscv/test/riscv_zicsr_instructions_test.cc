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

#include "riscv/riscv_zicsr_instructions.h"

#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_state.h"

// This file contains tests for individual Zicsr instructions.

namespace {

using ::mpact::sim::generic::ImmediateOperand;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::RiscV32SimpleCsr;
using ::mpact::sim::riscv::RiscVCsrEnum;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::RV32Register;

constexpr uint32_t kInstAddress = 0x2468;

constexpr char kX1[] = "x1";
constexpr char kX3[] = "x3";

constexpr uint32_t kUScratchValue =
    static_cast<uint32_t>(RiscVCsrEnum::kUScratch);

// The test fixture allocates a machine state object and an instruction object.
// It also contains convenience methods for interacting with the instruction
// object in a more short hand form.
class ZicsrInstructionsTest : public testing::Test {
 protected:
  ZicsrInstructionsTest() {
    state_ = new RiscVState("test", RiscVXlen::RV32);
    csr_ = new RiscV32SimpleCsr("uscratch", RiscVCsrEnum::kUScratch, 0, state_);
    CHECK_OK(state_->csr_set()->AddCsr(csr_));
    instruction_ = new Instruction(kInstAddress, state_);
    instruction_->set_size(4);
  }

  ~ZicsrInstructionsTest() override {
    delete instruction_;
    delete csr_;
    delete state_;
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

  // Appends immediate source operands with the given values.
  template <typename T>
  void AppendImmediateOperands(Instruction *inst,
                               const std::vector<T> &values) {
    for (auto value : values) {
      auto *src = new ImmediateOperand<T>(value);
      inst->AppendSource(src);
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

  RiscV32SimpleCsr *csr_;
  RiscVState *state_;
  Instruction *instruction_;
};

constexpr uint32_t kCsrValue1 = 0xaaaa5555;
constexpr uint32_t kCsrValue2 = 0xa5a5a5a5;

// The following tests all follow the same pattern. First the CSR and any
// registers that are used are initialized with known values. Then the
// instruction is initialized with the proper operands. The instruction is
// executed, before checking the values of registers and CSR for correctness.

// Tests the plain Csrrw/Csrrwi function.
TEST_F(ZicsrInstructionsTest, RiscVZiCsrrw) {
  auto result = state_->csr_set()->GetCsr(kUScratchValue);
  CHECK_OK(result);
  auto *csr = result.value();
  CHECK_NE(csr, nullptr);
  csr->Set(kCsrValue1);
  SetRegisterValues<uint32_t>({{kX1, kCsrValue2}, {kX3, 0}});
  AppendRegisterOperands(instruction_, {kX1}, {kX3});
  AppendImmediateOperands(instruction_, std::vector<uint32_t>{kUScratchValue});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVZiCsrrw);

  instruction_->Execute(nullptr);

  EXPECT_EQ(GetRegisterValue<uint32_t>(kX1), kCsrValue2);
  EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), kCsrValue1);

  EXPECT_EQ(csr->AsUint32(), kCsrValue2);
}

// Tests the plain Csrrs/Csrrsi function.
TEST_F(ZicsrInstructionsTest, RiscVZiCsrrs) {
  auto result = state_->csr_set()->GetCsr(kUScratchValue);
  CHECK_OK(result);
  auto *csr = result.value();
  CHECK_NE(csr, nullptr);
  csr->Set(kCsrValue1);
  SetRegisterValues<uint32_t>({{kX1, kCsrValue2}, {kX3, 0}});
  AppendRegisterOperands(instruction_, {kX1}, {kX3});
  AppendImmediateOperands(instruction_, std::vector<uint32_t>{kUScratchValue});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVZiCsrrs);

  instruction_->Execute(nullptr);

  EXPECT_EQ(GetRegisterValue<uint32_t>(kX1), kCsrValue2);
  EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), kCsrValue1);
  EXPECT_EQ(csr->AsUint32(), kCsrValue1 | kCsrValue2);
}

// Tests the plain Cssrrc/Csrrci function.
TEST_F(ZicsrInstructionsTest, RiscVZiCsrrc) {
  auto result = state_->csr_set()->GetCsr(kUScratchValue);
  CHECK_OK(result);
  auto *csr = result.value();
  CHECK_NE(csr, nullptr);
  csr->Set(kCsrValue1);
  SetRegisterValues<uint32_t>({{kX1, kCsrValue2}, {kX3, 0}});
  AppendRegisterOperands(instruction_, {kX1}, {kX3});
  AppendImmediateOperands(instruction_, std::vector<uint32_t>{kUScratchValue});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVZiCsrrc);

  instruction_->Execute(nullptr);

  EXPECT_EQ(GetRegisterValue<uint32_t>(kX1), kCsrValue2);
  EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), kCsrValue1);
  EXPECT_EQ(csr->AsUint32(), kCsrValue1 & ~kCsrValue2);
}

// Tests Cssrw when the CSR register isn't read (register source is x0).
TEST_F(ZicsrInstructionsTest, RiscVZiCsrrwNr) {
  auto result = state_->csr_set()->GetCsr(kUScratchValue);
  CHECK_OK(result);
  auto *csr = result.value();
  CHECK_NE(csr, nullptr);
  csr->Set(kCsrValue1);
  SetRegisterValues<uint32_t>({{kX1, kCsrValue2}, {kX3, 0}});
  AppendRegisterOperands(instruction_, {kX1}, {kX3});
  AppendImmediateOperands(instruction_, std::vector<uint32_t>{kUScratchValue});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVZiCsrrwNr);

  instruction_->Execute(nullptr);

  EXPECT_EQ(GetRegisterValue<uint32_t>(kX1), kCsrValue2);
  EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), 0);
  EXPECT_EQ(csr->AsUint32(), kCsrValue2);
}

// Tests Cssr[wcs]i when the CSR register isn't written (immediate is 0).
TEST_F(ZicsrInstructionsTest, RiscVZiCsrrNw) {
  auto result = state_->csr_set()->GetCsr(kUScratchValue);
  CHECK_OK(result);
  auto *csr = result.value();
  CHECK_NE(csr, nullptr);
  csr->Set(kCsrValue1);
  SetRegisterValues<uint32_t>({{kX1, kCsrValue2}, {kX3, 0}});
  AppendRegisterOperands(instruction_, {}, {kX3});
  AppendImmediateOperands(instruction_, std::vector<uint32_t>{kUScratchValue});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVZiCsrrNw);

  instruction_->Execute(nullptr);

  EXPECT_EQ(GetRegisterValue<uint32_t>(kX1), kCsrValue2);
  EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), kCsrValue1);
  EXPECT_EQ(csr->AsUint32(), kCsrValue1);
}

}  // namespace
