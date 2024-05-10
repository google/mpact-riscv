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

#include "riscv/riscv_vector_fp_reduction_instructions.h"

#include <algorithm>
#include <cstring>
#include <ios>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/random/random.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/instruction.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_vector_state.h"
#include "riscv/test/riscv_vector_fp_test_utilities.h"
#include "riscv/test/riscv_vector_instructions_test_base.h"

namespace {

using Instruction = ::mpact::sim::generic::Instruction;

// Functions to test.

using ::mpact::sim::riscv::Vfredmax;
using ::mpact::sim::riscv::Vfredmin;
using ::mpact::sim::riscv::Vfredosum;
using ::mpact::sim::riscv::Vfwredosum;

using ::absl::Span;
using ::mpact::sim::riscv::FPRoundingMode;
using ::mpact::sim::riscv::RVFpRegister;
using ::mpact::sim::riscv::ScopedFPStatus;

using ::mpact::sim::riscv::test::FPCompare;
using ::mpact::sim::riscv::test::FPTypeInfo;
using ::mpact::sim::riscv::test::kA5Mask;
using ::mpact::sim::riscv::test::kLmul8Values;
using ::mpact::sim::riscv::test::kLmulSettings;
using ::mpact::sim::riscv::test::kSewSettingsByByteSize;
using ::mpact::sim::riscv::test::kVd;
using ::mpact::sim::riscv::test::kVectorLengthInBytes;
using ::mpact::sim::riscv::test::kVmask;
using ::mpact::sim::riscv::test::kVmaskName;
using ::mpact::sim::riscv::test::kVs1;
using ::mpact::sim::riscv::test::kVs2;

// Test fixture for binary fp instructions.
class RiscVFPReductionInstructionsTest
    : public ::mpact::sim::riscv::test::RiscVFPInstructionsTestBase {
 public:
  // Helper function for floating point reduction operations.
  template <typename Vd, typename Vs2, typename Vs1>
  void ReductionOpFPTestHelper(absl::string_view name, int sew,
                               Instruction *inst, int delta_position,
                               std::function<Vd(Vs1, Vs2)> operation) {
    int byte_sew = sew / 8;
    if (byte_sew != sizeof(Vd) && byte_sew != sizeof(Vs2) &&
        byte_sew != sizeof(Vs1)) {
      FAIL() << name << ": selected element width != any operand types"
             << "sew: " << sew << " Vd: " << sizeof(Vd)
             << " Vs2: " << sizeof(Vs2) << " Vs1: " << sizeof(Vs1);
      return;
    }
    // Number of elements per vector register.
    constexpr int vs2_size = kVectorLengthInBytes / sizeof(Vs2);
    constexpr int vs1_size = kVectorLengthInBytes / sizeof(Vs1);
    // Input values for 8 registers.
    Vs2 vs2_value[vs2_size * 8];
    auto vs2_span = Span<Vs2>(vs2_value);
    Vs1 vs1_value[vs1_size * 8];
    auto vs1_span = Span<Vs1>(vs1_value);
    AppendVectorRegisterOperands({kVs2, kVs1, kVmask}, {kVd});
    auto mask_span = Span<const uint8_t>(kA5Mask);
    SetVectorRegisterValues<uint8_t>({{kVmaskName, mask_span}});
    // Iterate across the different lmul values.
    for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
      // Initialize input values.
      FillArrayWithRandomFPValues<Vs2>(vs2_span);
      FillArrayWithRandomFPValues<Vs1>(vs1_span);
      for (int i = 0; i < 8; i++) {
        auto vs2_name = absl::StrCat("v", kVs2 + i);
        auto vs1_name = absl::StrCat("v", kVs1 + i);
        SetVectorRegisterValues<Vs2>(
            {{vs2_name, vs2_span.subspan(vs2_size * i, vs2_size)}});
        SetVectorRegisterValues<Vs1>(
            {{vs1_name, vs1_span.subspan(vs1_size * i, vs1_size)}});
      }
      for (int vlen_count = 0; vlen_count < 4; vlen_count++) {
        int lmul8 = kLmul8Values[lmul_index];
        int lmul8_vs2 = lmul8 * sizeof(Vs2) / byte_sew;
        int lmul8_vs1 = lmul8 * sizeof(Vs1) / byte_sew;
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

        // Iterate across rounding modes.
        for (int rm : {0, 1, 2, 3, 4}) {
          rv_fp_->SetRoundingMode(static_cast<FPRoundingMode>(rm));

          ClearVectorRegisterGroup(kVd, 8);

          inst->Execute();

          if (lmul8_vd < 1 || lmul8_vd > 64) {
            EXPECT_TRUE(rv_vector_->vector_exception())
                << "lmul8: vd: " << lmul8_vd;
            rv_vector_->clear_vector_exception();
            continue;
          }

          if (lmul8_vs2 < 1 || lmul8_vs2 > 64) {
            EXPECT_TRUE(rv_vector_->vector_exception())
                << "lmul8: vs2: " << lmul8_vs2;
            rv_vector_->clear_vector_exception();
            continue;
          }

          if (lmul8_vs1 < 1 || lmul8_vs1 > 64) {
            EXPECT_TRUE(rv_vector_->vector_exception())
                << "lmul8: vs1: " << lmul8_vs1;
            rv_vector_->clear_vector_exception();
            continue;
          }
          EXPECT_FALSE(rv_vector_->vector_exception());
          // Initialize the accumulator with the value from vs1[0].
          Vd accumulator = static_cast<Vd>(vs1_span[0]);
          ScopedFPStatus set_fpstatus(rv_fp_->host_fp_interface());
          for (int i = 0; i < num_values; i++) {
            int mask_index = i >> 3;
            int mask_offset = i & 0b111;
            bool mask_value = (mask_span[mask_index] >> mask_offset) & 0b1;
            if (mask_value) {
              accumulator = operation(accumulator, vs2_span[i]);
            }
          }
          auto reg_val = vreg_[kVd]->data_buffer()->Get<Vd>(0);
          FPCompare<Vd>(accumulator, reg_val, delta_position, "");
        }
      }
    }
  }
};

// Test vector floating point sum reduction.
TEST_F(RiscVFPReductionInstructionsTest, Vfredosum) {
  SetSemanticFunction(&Vfredosum);
  ReductionOpFPTestHelper<float, float, float>(
      "Vfredosum_32", /*sew*/ 32, instruction_, /*delta_position*/ 32,
      [](float val0, float val1) -> float { return val0 + val1; });
  ResetInstruction();
  SetSemanticFunction(&Vfredosum);
  ReductionOpFPTestHelper<double, double, double>(
      "Vfredosum_64", /*sew*/ 64, instruction_, /*delta_position*/ 64,
      [](double val0, double val1) -> double { return val0 + val1; });
}

// Test vector floating point widening sum reduction.
TEST_F(RiscVFPReductionInstructionsTest, Vfwredosum) {
  SetSemanticFunction(&Vfwredosum);
  ReductionOpFPTestHelper<double, float, double>(
      "Vfwredosum_32", /*sew*/ 32, instruction_, /*delta_position*/ 64,
      [](double val0, float val1) -> double {
        return val0 + static_cast<double>(val1);
      });
}

template <typename T>
T MaxMinHelper(T vs2, T vs1, std::function<T(T, T)> operation) {
  using UInt = typename FPTypeInfo<T>::IntType;
  UInt vs2_uint = *reinterpret_cast<UInt *>(&vs2);
  UInt vs1_uint = *reinterpret_cast<UInt *>(&vs1);
  UInt mask = 1ULL << (FPTypeInfo<T>::kSigSize - 1);
  bool nan_vs2 = std::isnan(vs2);
  bool nan_vs1 = std::isnan(vs1);
  if ((nan_vs2 && ((mask & vs2_uint) == 0)) ||
      (nan_vs1 && ((mask & vs1_uint) == 0)) || (nan_vs2 && nan_vs1)) {
    // Canonical NaN.
    UInt canonical = ((1ULL << (FPTypeInfo<T>::kExpSize + 1)) - 1)
                     << (FPTypeInfo<T>::kSigSize - 1);
    T canonical_fp = *reinterpret_cast<T *>(&canonical);
    return canonical_fp;
  }
  if (nan_vs2) return vs1;
  if (nan_vs1) return vs2;
  return operation(vs2, vs1);
}

// Test vector floating point min reduction.
TEST_F(RiscVFPReductionInstructionsTest, Vfredmin) {
  SetSemanticFunction(&Vfredmin);
  ReductionOpFPTestHelper<float, float, float>(
      "Vfredmin_32", /*sew*/ 32, instruction_, /*delta_position*/ 32,
      [](float val0, float val1) -> float {
        return MaxMinHelper<float>(val0, val1,
                                   [](float val0, float val1) -> float {
                                     return (val0 > val1) ? val1 : val0;
                                   });
      });
  ResetInstruction();
  SetSemanticFunction(&Vfredmin);
  ReductionOpFPTestHelper<double, double, double>(
      "Vfredmin_64", /*sew*/ 64, instruction_, /*delta_position*/ 64,
      [](double val0, double val1) -> double {
        return MaxMinHelper<double>(val0, val1,
                                    [](double val0, double val1) -> double {
                                      return (val0 > val1) ? val1 : val0;
                                    });
      });
}

// Test vector floating point max reduction.
TEST_F(RiscVFPReductionInstructionsTest, Vfredmax) {
  SetSemanticFunction(&Vfredmax);
  ReductionOpFPTestHelper<float, float, float>(
      "Vfredmin_32", /*sew*/ 32, instruction_, /*delta_position*/ 32,
      [](float val0, float val1) -> float {
        return MaxMinHelper<float>(val0, val1,
                                   [](float val0, float val1) -> float {
                                     return (val0 < val1) ? val1 : val0;
                                   });
      });
  ResetInstruction();
  SetSemanticFunction(&Vfredmax);
  ReductionOpFPTestHelper<double, double, double>(
      "Vfredmin_64", /*sew*/ 64, instruction_, /*delta_position*/ 64,
      [](double val0, double val1) -> double {
        return MaxMinHelper<double>(val0, val1,
                                    [](double val0, double val1) -> double {
                                      return (val0 < val1) ? val1 : val0;
                                    });
      });
}

}  // namespace
