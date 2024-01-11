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

#include "riscv/riscv_vector_reduction_instructions.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "absl/random/random.h"
#include "absl/strings/string_view.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_vector_state.h"
#include "riscv/test/riscv_vector_instructions_test_base.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"

namespace {
using ::absl::Span;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::generic::WideType;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RVVectorRegister;
using ::mpact::sim::riscv::test::RiscVVectorInstructionsTestBase;

using ::mpact::sim::riscv::Vredand;
using ::mpact::sim::riscv::Vredmax;
using ::mpact::sim::riscv::Vredmaxu;
using ::mpact::sim::riscv::Vredmin;
using ::mpact::sim::riscv::Vredminu;
using ::mpact::sim::riscv::Vredor;
using ::mpact::sim::riscv::Vredsum;
using ::mpact::sim::riscv::Vredxor;
using ::mpact::sim::riscv::Vwredsum;
using ::mpact::sim::riscv::Vwredsumu;

using ::mpact::sim::riscv::test::kA5Mask;
using ::mpact::sim::riscv::test::kLmul8Values;
using ::mpact::sim::riscv::test::kLmulSettings;
using ::mpact::sim::riscv::test::kSewSettingsByByteSize;
using ::mpact::sim::riscv::test::kVd;
using ::mpact::sim::riscv::test::kVectorLengthInBytes;
using ::mpact::sim::riscv::test::kVmask;
using ::mpact::sim::riscv::test::kVmaskName;
using ::mpact::sim::riscv::test::kVs1;
using ::mpact::sim::riscv::test::kVs1Name;
using ::mpact::sim::riscv::test::kVs2;

class RiscVVectorReductionInstructionsTest
    : public RiscVVectorInstructionsTestBase {
 public:
  template <typename Vd, typename Vs2>
  void ReductionOpTestHelper(absl::string_view name, int sew, Instruction *inst,
                             std::function<Vd(Vd, Vs2)> operation) {
    int byte_sew = sew / 8;
    if (byte_sew != sizeof(Vd) && byte_sew != sizeof(Vs2)) {
      FAIL() << name << ": selected element width != any operand types"
             << "sew: " << sew << " Vd: " << sizeof(Vd)
             << " Vs2: " << sizeof(Vs2);
      return;
    }
    // Number of elements per vector register.
    constexpr int vs2_size = kVectorLengthInBytes / sizeof(Vs2);
    // Input values for 8 registers.
    Vs2 vs2_value[vs2_size * 8];
    auto vs2_span = Span<Vs2>(vs2_value);
    Vs2 vs1_value[vs2_size];
    auto vs1_span = Span<Vs2>(vs1_value);
    AppendVectorRegisterOperands({kVs2, kVs1, kVmask}, {kVd});
    // Initialize input values.
    FillArrayWithRandomValues<Vs2>(vs2_span);
    vs1_span[0] = RandomValue<Vs2>();
    auto mask_span = Span<const uint8_t>(kA5Mask);
    SetVectorRegisterValues<uint8_t>({{kVmaskName, mask_span}});
    SetVectorRegisterValues<Vs2>({{kVs1Name, Span<const Vs2>(vs1_span)}});
    // Initialize the accumulator with the value from vs1[0].
    for (int i = 0; i < 8; i++) {
      auto vs2_name = absl::StrCat("v", kVs2 + i);
      SetVectorRegisterValues<Vs2>(
          {{vs2_name, vs2_span.subspan(vs2_size * i, vs2_size)}});
    }
    // Iterate across the different lmul values.
    for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
      for (int vlen_count = 0; vlen_count < 4; vlen_count++) {
        int lmul8 = kLmul8Values[lmul_index];
        int lmul8_vs2 = lmul8 * sizeof(Vs2) / byte_sew;
        int lmul8_vd = lmul8 * sizeof(Vd) / byte_sew;
        int num_values = lmul8 * kVectorLengthInBytes / (8 * byte_sew);
        // Set vlen, but leave vlen high at least once.
        int vlen = 1024;
        if (vlen_count > 0) {
          vlen =
              absl::Uniform(absl::IntervalOpenClosed, bitgen_, 0, num_values);
        }
        num_values = std::min(num_values, vlen);
        // Configure vector unit for different lmul settings.
        uint32_t vtype =
            (kSewSettingsByByteSize[byte_sew] << 3) | kLmulSettings[lmul_index];
        ConfigureVectorUnit(vtype, vlen);
        ClearVectorRegisterGroup(kVd, 8);

        inst->Execute();

        if ((lmul8_vs2 < 1) || (lmul8_vs2 > 64)) {
          EXPECT_TRUE(rv_vector_->vector_exception());
          rv_vector_->clear_vector_exception();
          continue;
        }

        if ((lmul8_vd < 1) || (lmul8_vd > 64)) {
          EXPECT_TRUE(rv_vector_->vector_exception());
          rv_vector_->clear_vector_exception();
          continue;
        }

        EXPECT_FALSE(rv_vector_->vector_exception());
        Vd accumulator = static_cast<Vd>(vs1_span[0]);
        for (int i = 0; i < num_values; i++) {
          int mask_index = i >> 3;
          int mask_offset = i & 0b111;
          bool mask_value = (mask_span[mask_index] >> mask_offset) & 0b1;
          if (mask_value) {
            accumulator = operation(accumulator, vs2_span[i]);
          }
        }
        EXPECT_EQ(accumulator, vreg_[kVd]->data_buffer()->Get<Vd>(0));
      }
    }
  }
};

// Test functions for vector reduction instruction semantic functions. The
// vector reduction instructions take two vector source operands and a mask
// operand, and write to the first element of a destination vector operand.

// Vector sum reduction.
TEST_F(RiscVVectorReductionInstructionsTest, Vredsum8) {
  using T = uint8_t;
  SetSemanticFunction(&Vredsum);
  ReductionOpTestHelper<T, T>("Vredsum", /*sew*/ sizeof(T) * 8, instruction_,
                              [](T val0, T val1) -> T { return val0 + val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredsum16) {
  using T = uint16_t;
  SetSemanticFunction(&Vredsum);
  ReductionOpTestHelper<T, T>("Vredsum", /*sew*/ sizeof(T) * 8, instruction_,
                              [](T val0, T val1) -> T { return val0 + val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredsum32) {
  using T = uint32_t;
  SetSemanticFunction(&Vredsum);
  ReductionOpTestHelper<T, T>("Vredsum", /*sew*/ sizeof(T) * 8, instruction_,
                              [](T val0, T val1) -> T { return val0 + val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredsum64) {
  using T = uint64_t;
  SetSemanticFunction(&Vredsum);
  ReductionOpTestHelper<T, T>("Vredsum", /*sew*/ sizeof(T) * 8, instruction_,
                              [](T val0, T val1) -> T { return val0 + val1; });
}

// Vector and reduction.
TEST_F(RiscVVectorReductionInstructionsTest, Vredand8) {
  using T = uint8_t;
  SetSemanticFunction(&Vredand);
  ReductionOpTestHelper<T, T>("Vredand", /*sew*/ sizeof(T) * 8, instruction_,
                              [](T val0, T val1) -> T { return val0 & val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredand16) {
  using T = uint16_t;
  SetSemanticFunction(&Vredand);
  ReductionOpTestHelper<T, T>("Vredand", /*sew*/ sizeof(T) * 8, instruction_,
                              [](T val0, T val1) -> T { return val0 & val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredand32) {
  using T = uint32_t;
  SetSemanticFunction(&Vredand);
  ReductionOpTestHelper<T, T>("Vredand", /*sew*/ sizeof(T) * 8, instruction_,
                              [](T val0, T val1) -> T { return val0 & val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredand64) {
  using T = uint64_t;
  SetSemanticFunction(&Vredand);
  ReductionOpTestHelper<T, T>("Vredand", /*sew*/ sizeof(T) * 8, instruction_,
                              [](T val0, T val1) -> T { return val0 & val1; });
}

// Vector or reduction.
TEST_F(RiscVVectorReductionInstructionsTest, Vredor8) {
  using T = uint8_t;
  SetSemanticFunction(&Vredor);
  ReductionOpTestHelper<T, T>("Vredor", /*sew*/ sizeof(T) * 8, instruction_,
                              [](T val0, T val1) -> T { return val0 | val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredor16) {
  using T = uint16_t;
  SetSemanticFunction(&Vredor);
  ReductionOpTestHelper<T, T>("Vredor", /*sew*/ sizeof(T) * 8, instruction_,
                              [](T val0, T val1) -> T { return val0 | val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredor32) {
  using T = uint32_t;
  SetSemanticFunction(&Vredor);
  ReductionOpTestHelper<T, T>("Vredor", /*sew*/ sizeof(T) * 8, instruction_,
                              [](T val0, T val1) -> T { return val0 | val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredor64) {
  using T = uint64_t;
  SetSemanticFunction(&Vredor);
  ReductionOpTestHelper<T, T>("Vredor", /*sew*/ sizeof(T) * 8, instruction_,
                              [](T val0, T val1) -> T { return val0 | val1; });
}

// Vector xor reduction.
TEST_F(RiscVVectorReductionInstructionsTest, Vredxor8) {
  using T = uint8_t;
  SetSemanticFunction(&Vredxor);
  ReductionOpTestHelper<T, T>("Vredxor", /*sew*/ sizeof(T) * 8, instruction_,
                              [](T val0, T val1) -> T { return val0 ^ val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredxor16) {
  using T = uint16_t;
  SetSemanticFunction(&Vredxor);
  ReductionOpTestHelper<T, T>("Vredxor", /*sew*/ sizeof(T) * 8, instruction_,
                              [](T val0, T val1) -> T { return val0 ^ val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredxor32) {
  using T = uint32_t;
  SetSemanticFunction(&Vredxor);
  ReductionOpTestHelper<T, T>("Vredxor", /*sew*/ sizeof(T) * 8, instruction_,
                              [](T val0, T val1) -> T { return val0 ^ val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredxor64) {
  using T = uint64_t;
  SetSemanticFunction(&Vredxor);
  ReductionOpTestHelper<T, T>("Vredxor", /*sew*/ sizeof(T) * 8, instruction_,
                              [](T val0, T val1) -> T { return val0 ^ val1; });
}

// Vector unsigned min reduction.
TEST_F(RiscVVectorReductionInstructionsTest, Vredminu8) {
  using T = uint8_t;
  SetSemanticFunction(&Vredminu);
  ReductionOpTestHelper<T, T>(
      "Vredminu", /*sew*/ sizeof(T) * 8, instruction_,
      [](T val0, T val1) -> T { return val0 < val1 ? val0 : val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredminu16) {
  using T = uint16_t;
  SetSemanticFunction(&Vredminu);
  ReductionOpTestHelper<T, T>(
      "Vredminu", /*sew*/ sizeof(T) * 8, instruction_,
      [](T val0, T val1) -> T { return val0 < val1 ? val0 : val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredminu32) {
  using T = uint32_t;
  SetSemanticFunction(&Vredminu);
  ReductionOpTestHelper<T, T>(
      "Vredminu", /*sew*/ sizeof(T) * 8, instruction_,
      [](T val0, T val1) -> T { return val0 < val1 ? val0 : val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredminu64) {
  using T = uint64_t;
  SetSemanticFunction(&Vredminu);
  ReductionOpTestHelper<T, T>(
      "Vredminu", /*sew*/ sizeof(T) * 8, instruction_,
      [](T val0, T val1) -> T { return val0 < val1 ? val0 : val1; });
}

// Vector signed min reduction.
TEST_F(RiscVVectorReductionInstructionsTest, Vredmin8) {
  using T = int8_t;
  SetSemanticFunction(&Vredmin);
  ReductionOpTestHelper<T, T>(
      "Vredmin", /*sew*/ sizeof(T) * 8, instruction_,
      [](T val0, T val1) -> T { return val0 < val1 ? val0 : val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredmin16) {
  using T = int16_t;
  SetSemanticFunction(&Vredmin);
  ReductionOpTestHelper<T, T>(
      "Vredmin", /*sew*/ sizeof(T) * 8, instruction_,
      [](T val0, T val1) -> T { return val0 < val1 ? val0 : val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredmin32) {
  using T = int32_t;
  SetSemanticFunction(&Vredmin);
  ReductionOpTestHelper<T, T>(
      "Vredmin", /*sew*/ sizeof(T) * 8, instruction_,
      [](T val0, T val1) -> T { return val0 < val1 ? val0 : val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredmin64) {
  using T = int64_t;
  SetSemanticFunction(&Vredmin);
  ReductionOpTestHelper<T, T>(
      "Vredmin", /*sew*/ sizeof(T) * 8, instruction_,
      [](T val0, T val1) -> T { return val0 < val1 ? val0 : val1; });
}

// Vector unsigned max reduction.
TEST_F(RiscVVectorReductionInstructionsTest, Vredmaxu8) {
  using T = uint8_t;
  SetSemanticFunction(&Vredmaxu);
  ReductionOpTestHelper<T, T>(
      "Vredmaxu", /*sew*/ sizeof(T) * 8, instruction_,
      [](T val0, T val1) -> T { return val0 > val1 ? val0 : val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredmaxu16) {
  using T = uint16_t;
  SetSemanticFunction(&Vredmaxu);
  ReductionOpTestHelper<T, T>(
      "Vredmaxu", /*sew*/ sizeof(T) * 8, instruction_,
      [](T val0, T val1) -> T { return val0 > val1 ? val0 : val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredmaxu32) {
  using T = uint32_t;
  SetSemanticFunction(&Vredmaxu);
  ReductionOpTestHelper<T, T>(
      "Vredmaxu", /*sew*/ sizeof(T) * 8, instruction_,
      [](T val0, T val1) -> T { return val0 > val1 ? val0 : val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredmaxu64) {
  using T = uint64_t;
  SetSemanticFunction(&Vredmaxu);
  ReductionOpTestHelper<T, T>(
      "Vredmaxu", /*sew*/ sizeof(T) * 8, instruction_,
      [](T val0, T val1) -> T { return val0 > val1 ? val0 : val1; });
}

// Vector signed max reduction.
TEST_F(RiscVVectorReductionInstructionsTest, Vredmax8) {
  using T = int8_t;
  SetSemanticFunction(&Vredmax);
  ReductionOpTestHelper<T, T>(
      "Vredmax", /*sew*/ sizeof(T) * 8, instruction_,
      [](T val0, T val1) -> T { return val0 > val1 ? val0 : val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredmax16) {
  using T = int16_t;
  SetSemanticFunction(&Vredmax);
  ReductionOpTestHelper<T, T>(
      "Vredmax", /*sew*/ sizeof(T) * 8, instruction_,
      [](T val0, T val1) -> T { return val0 > val1 ? val0 : val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredmax32) {
  using T = int32_t;
  SetSemanticFunction(&Vredmax);
  ReductionOpTestHelper<T, T>(
      "Vredmax", /*sew*/ sizeof(T) * 8, instruction_,
      [](T val0, T val1) -> T { return val0 > val1 ? val0 : val1; });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vredmax64) {
  using T = int64_t;
  SetSemanticFunction(&Vredmax);
  ReductionOpTestHelper<T, T>(
      "Vredmax", /*sew*/ sizeof(T) * 8, instruction_,
      [](T val0, T val1) -> T { return val0 > val1 ? val0 : val1; });
}

// Vector widening unsigned sum reduction.
TEST_F(RiscVVectorReductionInstructionsTest, Vwredsumu8) {
  using T = uint8_t;
  using WT = WideType<T>::type;
  SetSemanticFunction(&Vwredsumu);
  ReductionOpTestHelper<WT, T>(
      "Vredsumu", /*sew*/ sizeof(T) * 8, instruction_,
      [](WT val0, T val1) -> WT { return val0 + static_cast<WT>(val1); });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vwredsumu16) {
  using T = uint16_t;
  using WT = WideType<T>::type;
  SetSemanticFunction(&Vwredsumu);
  ReductionOpTestHelper<WT, T>(
      "Vredsumu", /*sew*/ sizeof(T) * 8, instruction_,
      [](WT val0, T val1) -> WT { return val0 + static_cast<WT>(val1); });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vwredsumu32) {
  using T = uint32_t;
  using WT = WideType<T>::type;
  SetSemanticFunction(&Vwredsumu);
  ReductionOpTestHelper<WT, T>(
      "Vredsumu", /*sew*/ sizeof(T) * 8, instruction_,
      [](WT val0, T val1) -> WT { return val0 + static_cast<WT>(val1); });
}

// Vector widening signed sum reduction.

TEST_F(RiscVVectorReductionInstructionsTest, Vwredsum8) {
  using T = int8_t;
  using WT = WideType<T>::type;
  SetSemanticFunction(&Vwredsum);
  ReductionOpTestHelper<WT, T>(
      "Vredsum", /*sew*/ sizeof(T) * 8, instruction_,
      [](WT val0, T val1) -> WT { return val0 + static_cast<WT>(val1); });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vwredsum16) {
  using T = int16_t;
  using WT = WideType<T>::type;
  SetSemanticFunction(&Vwredsum);
  ReductionOpTestHelper<WT, T>(
      "Vredsum", /*sew*/ sizeof(T) * 8, instruction_,
      [](WT val0, T val1) -> WT { return val0 + static_cast<WT>(val1); });
}
TEST_F(RiscVVectorReductionInstructionsTest, Vwredsum32) {
  using T = int32_t;
  using WT = WideType<T>::type;
  SetSemanticFunction(&Vwredsum);
  ReductionOpTestHelper<WT, T>(
      "Vredsum", /*sew*/ sizeof(T) * 8, instruction_,
      [](WT val0, T val1) -> WT { return val0 + static_cast<WT>(val1); });
}

}  // namespace
