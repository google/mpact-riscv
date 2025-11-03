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

#include <algorithm>
#include <cstdint>
#include <ios>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "absl/random/random.h"
#include "absl/strings/string_view.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/instruction.h"
#include "riscv/riscv_bitmanip_instructions.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"

// This file contains tests for individual RiscV32 bit manipulation
// instructions.

namespace {

using ::mpact::sim::riscv::RV32::RiscVAndn;
using ::mpact::sim::riscv::RV32::RiscVBclr;
using ::mpact::sim::riscv::RV32::RiscVBext;
using ::mpact::sim::riscv::RV32::RiscVBinv;
using ::mpact::sim::riscv::RV32::RiscVBset;
using ::mpact::sim::riscv::RV32::RiscVClmul;
using ::mpact::sim::riscv::RV32::RiscVClmulh;
using ::mpact::sim::riscv::RV32::RiscVClmulr;
using ::mpact::sim::riscv::RV32::RiscVClz;
using ::mpact::sim::riscv::RV32::RiscVCpop;
using ::mpact::sim::riscv::RV32::RiscVCtz;
using ::mpact::sim::riscv::RV32::RiscVMax;
using ::mpact::sim::riscv::RV32::RiscVMaxu;
using ::mpact::sim::riscv::RV32::RiscVMin;
using ::mpact::sim::riscv::RV32::RiscVMinu;
using ::mpact::sim::riscv::RV32::RiscVOrcb;
using ::mpact::sim::riscv::RV32::RiscVOrn;
using ::mpact::sim::riscv::RV32::RiscVRev8;
using ::mpact::sim::riscv::RV32::RiscVRol;
using ::mpact::sim::riscv::RV32::RiscVRor;
using ::mpact::sim::riscv::RV32::RiscVSextB;
using ::mpact::sim::riscv::RV32::RiscVSextH;
using ::mpact::sim::riscv::RV32::RiscVShAdd;
using ::mpact::sim::riscv::RV32::RiscVXnor;
using ::mpact::sim::riscv::RV32::RiscVZextH;

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

// The test fixture allocates a machine state object and an instruction object.
// It also contains convenience methods for interacting with the instruction
// object in a more short hand form.
class RV32BitmanipInstructionTest : public testing::Test {
 public:
  RV32BitmanipInstructionTest() {
    state_ = new RiscVState("test", RiscVXlen::RV32, nullptr);
    instruction_ = new Instruction(kInstAddress, state_);
    instruction_->set_size(4);
  }
  ~RV32BitmanipInstructionTest() override {
    delete state_;
    delete instruction_;
  }

  // Appends the source and destination operands for the register names
  // given in the two vectors.
  void AppendRegisterOperands(Instruction* inst,
                              const std::vector<std::string>& sources,
                              const std::vector<std::string>& destinations) {
    for (auto& reg_name : sources) {
      auto* reg = state_->GetRegister<RV32Register>(reg_name).first;
      inst->AppendSource(reg->CreateSourceOperand());
    }
    for (auto& reg_name : destinations) {
      auto* reg = state_->GetRegister<RV32Register>(reg_name).first;
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
      auto* reg = state_->GetRegister<RV32Register>(reg_name).first;
      auto* db = state_->db_factory()->Allocate<RV32Register::ValueType>(1);
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
    auto* reg = state_->GetRegister<RV32Register>(reg_name).first;
    return reg->data_buffer()->Get<T>(0);
  }

  RiscVState* state_;
  Instruction* instruction_;
  absl::BitGen bitgen_;
};

TEST_F(RV32BitmanipInstructionTest, RV32Sh1Add) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  AppendImmediateOperands<uint32_t>({1});
  SetSemanticFunction(&RiscVShAdd);
  SetRegisterValues<int32_t>({{kX1, kVal1}, {kX2, kVal2}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int32_t>(kX3), kVal2 + (kVal1 << 1));
}

TEST_F(RV32BitmanipInstructionTest, RV32Sh2Add) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  AppendImmediateOperands<uint32_t>({2});
  SetSemanticFunction(&RiscVShAdd);
  SetRegisterValues<int32_t>({{kX1, kVal1}, {kX2, kVal2}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int32_t>(kX3), kVal2 + (kVal1 << 2));
}

TEST_F(RV32BitmanipInstructionTest, RV32Sh3Add) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  AppendImmediateOperands<uint32_t>({3});
  SetSemanticFunction(&RiscVShAdd);
  SetRegisterValues<int32_t>({{kX1, kVal1}, {kX2, kVal2}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int32_t>(kX3), kVal2 + (kVal1 << 3));
}

TEST_F(RV32BitmanipInstructionTest, RV32Andn) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVAndn);
  SetRegisterValues<int32_t>({{kX1, kVal1}, {kX2, kVal2}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int32_t>(kX3), kVal1 & ~kVal2);
}

TEST_F(RV32BitmanipInstructionTest, RV32Orn) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVOrn);
  SetRegisterValues<int32_t>({{kX1, kVal1}, {kX2, kVal2}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int32_t>(kX3), kVal1 | ~kVal2);
}

TEST_F(RV32BitmanipInstructionTest, RV32Xnor) {
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVXnor);
  SetRegisterValues<int32_t>({{kX1, kVal1}, {kX2, kVal2}});
  instruction_->Execute(nullptr);
  EXPECT_EQ(GetRegisterValue<int32_t>(kX3), ~(kVal1 ^ kVal2));
}

TEST_F(RV32BitmanipInstructionTest, RV32Clz) {
  AppendRegisterOperands({kX1}, {kX3});
  SetSemanticFunction(&RiscVClz);
  uint32_t value = 0;
  for (int i = 0; i < 33; ++i) {
    SetRegisterValues<int32_t>({{kX1, value}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<int32_t>(kX3), 32 - i);
    value = (value << 1) | 1;
  }
}

TEST_F(RV32BitmanipInstructionTest, RV32Ctz) {
  AppendRegisterOperands({kX1}, {kX3});
  SetSemanticFunction(&RiscVCtz);
  uint32_t value = 0;
  for (int i = 0; i < 33; ++i) {
    SetRegisterValues<int32_t>({{kX1, value}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<int32_t>(kX3), 32 - i);
    value = (value >> 1) | 0x8000'0000;
  }
}

TEST_F(RV32BitmanipInstructionTest, RV32CPop) {
  AppendRegisterOperands({kX1}, {kX3});
  SetSemanticFunction(&RiscVCpop);
  uint32_t value = 0;
  for (int i = 0; i < 33; ++i) {
    SetRegisterValues<int32_t>({{kX1, value}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<int32_t>(kX3), i);
    value = (value >> 1) | 0x8000'0000;
  }
}

TEST_F(RV32BitmanipInstructionTest, RV32Max) {
  using T = int32_t;
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVMax);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    T val2 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    SetRegisterValues<uint32_t>({{kX1, val1}, {kX2, val2}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), std::max(val1, val2));
  }
}

TEST_F(RV32BitmanipInstructionTest, RV32Maxu) {
  using T = uint32_t;
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVMaxu);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    T val2 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    SetRegisterValues<uint32_t>({{kX1, val1}, {kX2, val2}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), std::max(val1, val2));
  }
}

TEST_F(RV32BitmanipInstructionTest, RV32Min) {
  using T = int32_t;
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVMin);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    T val2 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    SetRegisterValues<uint32_t>({{kX1, val1}, {kX2, val2}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), std::min(val1, val2));
  }
}

TEST_F(RV32BitmanipInstructionTest, RV32Minu) {
  using T = uint32_t;
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVMinu);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    T val2 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    SetRegisterValues<uint32_t>({{kX1, val1}, {kX2, val2}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), std::min(val1, val2));
  }
}

TEST_F(RV32BitmanipInstructionTest, RV32SextB) {
  using T = int32_t;
  AppendRegisterOperands({kX1}, {kX3});
  SetSemanticFunction(&RiscVSextB);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    SetRegisterValues<uint32_t>({{kX1, val1}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), (val1 << 24) >> 24);
  }
}

TEST_F(RV32BitmanipInstructionTest, RV32SextH) {
  using T = int32_t;
  AppendRegisterOperands({kX1}, {kX3});
  SetSemanticFunction(&RiscVSextH);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    SetRegisterValues<uint32_t>({{kX1, val1}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), (val1 << 16) >> 16);
  }
}

TEST_F(RV32BitmanipInstructionTest, RV32ZextH) {
  using T = uint32_t;
  AppendRegisterOperands({kX1}, {kX3});
  SetSemanticFunction(&RiscVZextH);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    SetRegisterValues<uint32_t>({{kX1, val1}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), val1 & 0xffff);
  }
}

TEST_F(RV32BitmanipInstructionTest, RV32Rol) {
  using T = uint32_t;
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVRol);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    T val2 = absl::Uniform(absl::IntervalClosed, bitgen_, 0, 31);
    SetRegisterValues<uint32_t>({{kX1, val1}, {kX2, val2}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<uint32_t>(kX3),
              (val1 << val2) | ((val2 == 0) ? 0 : (val1 >> (32 - val2))));
  }
}

TEST_F(RV32BitmanipInstructionTest, RV32Ror) {
  using T = uint32_t;
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVRor);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    T val2 = absl::Uniform(absl::IntervalClosed, bitgen_, 0, 31);
    SetRegisterValues<uint32_t>({{kX1, val1}, {kX2, val2}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<uint32_t>(kX3),
              (val1 >> val2) | ((val2 == 0) ? 0 : (val1 << (32 - val2))));
  }
}

TEST_F(RV32BitmanipInstructionTest, RV32Orcb) {
  using T = uint32_t;
  AppendRegisterOperands({kX1}, {kX3});
  SetSemanticFunction(&RiscVOrcb);
  for (int i = 0; i < 100; i++) {
    T val1 = 0;
    // Randomly set bytes of val1 to random values.
    for (int i = 0; i < sizeof(T); ++i) {
      bool bit = absl::Bernoulli(bitgen_, 0.5);
      if (bit) {
        val1 |= (absl::Uniform(absl::IntervalClosed, bitgen_,
                               std::numeric_limits<uint8_t>::min(),
                               std::numeric_limits<uint8_t>::max())
                 << (8 * i));
      }
    }
    SetRegisterValues<uint32_t>({{kX1, val1}});
    instruction_->Execute(nullptr);
    T result = 0;
    for (int i = 0; i < sizeof(T); ++i) {
      result |= (val1 & (0xff << (8 * i))) ? (0xff << (8 * i)) : 0;
    }
    EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), result)
        << std::hex << GetRegisterValue<uint32_t>(kX3) << " " << result;
  }
}

TEST_F(RV32BitmanipInstructionTest, RV32Rev8) {
  using T = uint32_t;
  AppendRegisterOperands({kX1}, {kX3});
  SetSemanticFunction(&RiscVRev8);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    SetRegisterValues<uint32_t>({{kX1, val1}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<uint32_t>(kX3),
              ((val1 & 0xff000000) >> 24) | ((val1 & 0x00ff0000) >> 8) |
                  ((val1 & 0x0000ff00) << 8) | ((val1 & 0x000000ff) << 24));
  }
}

TEST_F(RV32BitmanipInstructionTest, RV32Clmul) {
  using T = uint32_t;
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVClmul);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    T val2 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    SetRegisterValues<uint32_t>({{kX1, val1}, {kX2, val2}});
    instruction_->Execute(nullptr);
    uint64_t result = 0;
    for (int i = 0; i < sizeof(T) * 8; ++i) {
      if (val2 & (1 << i)) result ^= static_cast<uint64_t>(val1 << i);
    }
    EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), result & 0xffff'ffff);
  }
}

TEST_F(RV32BitmanipInstructionTest, RV32Clmulh) {
  using T = uint32_t;
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVClmulh);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    T val2 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    SetRegisterValues<uint32_t>({{kX1, val1}, {kX2, val2}});
    instruction_->Execute(nullptr);
    uint64_t result = 0;
    for (int i = 0; i < sizeof(T) * 8; ++i) {
      if (val2 & (1 << i)) result ^= (static_cast<uint64_t>(val1) << i);
    }
    EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), result >> 32);
  }
}

TEST_F(RV32BitmanipInstructionTest, RV32Clmulr) {
  using T = uint32_t;
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVClmulr);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    T val2 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    SetRegisterValues<uint32_t>({{kX1, val1}, {kX2, val2}});
    instruction_->Execute(nullptr);
    uint32_t result = 0;
    for (int i = 0; i < sizeof(T) * 8 - 1; ++i) {
      if ((val2 >> i) & 1) result ^= (val1 >> (sizeof(T) * 8 - 1 - i));
    }
    EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), result);
  }
}

TEST_F(RV32BitmanipInstructionTest, RV32Bclr) {
  using T = uint32_t;
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVBclr);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    T val2 = absl::Uniform(absl::IntervalClosed, bitgen_, 0, 31);
    SetRegisterValues<uint32_t>({{kX1, val1}, {kX2, val2}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), val1 & ~(1 << val2));
  }
}

TEST_F(RV32BitmanipInstructionTest, RV32Bext) {
  using T = uint32_t;
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVBext);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    T val2 = absl::Uniform(absl::IntervalClosed, bitgen_, 0, 31);
    SetRegisterValues<uint32_t>({{kX1, val1}, {kX2, val2}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), (val1 >> val2) & 0x1);
  }
}

TEST_F(RV32BitmanipInstructionTest, RV32Binv) {
  using T = uint32_t;
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVBinv);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    T val2 = absl::Uniform(absl::IntervalClosed, bitgen_, 0, 31);
    SetRegisterValues<uint32_t>({{kX1, val1}, {kX2, val2}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), val1 ^ (1 << val2));
  }
}

TEST_F(RV32BitmanipInstructionTest, RV32Bset) {
  using T = uint32_t;
  AppendRegisterOperands({kX1, kX2}, {kX3});
  SetSemanticFunction(&RiscVBset);
  for (int i = 0; i < 100; i++) {
    T val1 = absl::Uniform(absl::IntervalClosed, bitgen_,
                           std::numeric_limits<T>::min(),
                           std::numeric_limits<T>::max());
    T val2 = absl::Uniform(absl::IntervalClosed, bitgen_, 0, 31);
    SetRegisterValues<uint32_t>({{kX1, val1}, {kX2, val2}});
    instruction_->Execute(nullptr);
    EXPECT_EQ(GetRegisterValue<uint32_t>(kX3), val1 | (1 << val2));
  }
}

}  // namespace
