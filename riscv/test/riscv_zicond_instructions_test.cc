#include "riscv/riscv_zicond_instructions.h"

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/arch_state.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "riscv/riscv_register.h"

namespace {

using ::mpact::sim::generic::ArchState;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RV64Register;

constexpr uint32_t kInstAddress = 0x2468;
constexpr char kX1[] = "x1";
constexpr char kX2[] = "x2";
constexpr char kX3[] = "x3";
constexpr uint32_t kVal1 = 0x12345678;
constexpr uint32_t kVal2 = 0x87654321;
constexpr uint32_t kVal3 = 0xdeadbeef;

class TestState : public ArchState {
 public:
  TestState() : ArchState("test") {}
};

class RiscVZicondInstructionTest : public testing::Test {
 public:
  RiscVZicondInstructionTest() {
    instruction_ = new Instruction(kInstAddress, &state_);
    instruction_->set_size(4);
    for (auto reg_name : {kX1, kX2, kX3}) {
      rv32_regs_.insert({reg_name, new RV32Register(&state_, reg_name)});
      rv64_regs_.insert({reg_name, new RV64Register(&state_, reg_name)});
    }
  }

  ~RiscVZicondInstructionTest() override {
    delete instruction_;
    for (auto reg : rv32_regs_) delete reg.second;
    for (auto reg : rv64_regs_) delete reg.second;
  }

  // Initializes the semantic function of the instruction object.
  void SetSemanticFunction(Instruction::SemanticFunction fcn) {
    instruction_->set_semantic_function(fcn);
  }

  // Returns the value of the named register.
  template <typename RegisterType>
  typename RegisterType::ValueType GetRegisterValue(
      absl::string_view reg_name) {
    RegisterType *reg;
    if constexpr (std::is_same_v<RegisterType, RV32Register>) {
      reg = rv32_regs_[reg_name];
    } else {
      reg = rv64_regs_[reg_name];
    }
    CHECK_NE(reg, nullptr);
    return reg->data_buffer()->template Get<typename RegisterType::ValueType>(
        0);
  }

  template <typename RegisterType>
  void AppendRegisterOperands(const std::vector<std::string> &sources,
                              const std::vector<std::string> &destinations) {
    absl::flat_hash_map<std::string, RegisterType *> *regs;
    if constexpr (std::is_same_v<RegisterType, RV32Register>) {
      regs = &rv32_regs_;
    } else {
      regs = &rv64_regs_;
    }
    for (auto src : sources) {
      auto *reg = (*regs)[src];
      CHECK_NE(reg, nullptr);
      instruction_->AppendSource(reg->CreateSourceOperand());
    }
    for (auto dest : destinations) {
      auto *reg = (*regs)[dest];
      CHECK_NE(reg, nullptr);
      instruction_->AppendDestination(reg->CreateDestinationOperand(0));
    }
  }

  template <typename RegisterType>
  void SetRegisterValues(
      const std::vector<
          std::tuple<std::string, typename RegisterType::ValueType>>
          values) {
    absl::flat_hash_map<std::string, RegisterType *> *regs;
    if constexpr (std::is_same_v<RegisterType, RV32Register>) {
      regs = &rv32_regs_;
    } else {
      regs = &rv64_regs_;
    }
    for (auto &[reg_name, value] : values) {
      auto *reg = (*regs)[reg_name];
      CHECK_NE(reg, nullptr);
      auto *db =
          state_.db_factory()->Allocate<typename RegisterType::ValueType>(1);
      db->template Set<typename RegisterType::ValueType>(0, value);
      reg->SetDataBuffer(db);
      db->DecRef();
    }
  }

  Instruction *instruction() { return instruction_; }

 private:
  TestState state_;
  Instruction *instruction_;
  absl::flat_hash_map<std::string, RV32Register *> rv32_regs_;
  absl::flat_hash_map<std::string, RV64Register *> rv64_regs_;
};

TEST_F(RiscVZicondInstructionTest, RV32CzeroEqz) {
  using Reg = RV32Register;
  AppendRegisterOperands<Reg>({kX1, kX2}, {kX3});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVCzeroEqz);
  SetRegisterValues<Reg>({{kX1, kVal1}, {kX2, kVal2}, {kX3, kVal3}});
  instruction()->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<Reg>(kX3), kVal1);
  SetRegisterValues<Reg>({{kX1, kVal1}, {kX2, 0}, {kX3, kVal3}});
  instruction()->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<Reg>(kX3), 0);
}

TEST_F(RiscVZicondInstructionTest, RV32CzeroNez) {
  using Reg = RV32Register;
  AppendRegisterOperands<Reg>({kX1, kX2}, {kX3});
  SetSemanticFunction(&::mpact::sim::riscv::RV32::RiscVCzeroNez);
  SetRegisterValues<Reg>({{kX1, kVal1}, {kX2, 0}, {kX3, kVal3}});
  instruction()->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<Reg>(kX3), kVal1);
  SetRegisterValues<Reg>({{kX1, kVal1}, {kX2, kVal2}, {kX3, kVal3}});
  instruction()->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<Reg>(kX3), 0);
}

TEST_F(RiscVZicondInstructionTest, RV64CzeroEqz) {
  using Reg = RV64Register;
  AppendRegisterOperands<Reg>({kX1, kX2}, {kX3});
  SetSemanticFunction(&::mpact::sim::riscv::RV64::RiscVCzeroEqz);
  SetRegisterValues<Reg>({{kX1, kVal1}, {kX2, kVal2}, {kX3, kVal3}});
  instruction()->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<Reg>(kX3), kVal1);
  SetRegisterValues<Reg>({{kX1, kVal1}, {kX2, 0}, {kX3, kVal3}});
  instruction()->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<Reg>(kX3), 0);
}

TEST_F(RiscVZicondInstructionTest, RV64CzeroNez) {
  using Reg = RV64Register;
  AppendRegisterOperands<Reg>({kX1, kX2}, {kX3});
  SetSemanticFunction(&::mpact::sim::riscv::RV64::RiscVCzeroNez);
  SetRegisterValues<Reg>({{kX1, kVal1}, {kX2, 0}, {kX3, kVal3}});
  instruction()->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<Reg>(kX3), kVal1);
  SetRegisterValues<Reg>({{kX1, kVal1}, {kX2, kVal2}, {kX3, kVal3}});
  instruction()->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<Reg>(kX3), 0);
}

}  // namespace
