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

#include "riscv/riscv_vector_fp_compare_instructions.h"

#include <algorithm>
#include <cstdint>
#include <functional>

#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/instruction.h"
#include "riscv/riscv_fp_host.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_register.h"
#include "riscv/test/riscv_vector_fp_test_utilities.h"
#include "riscv/test/riscv_vector_instructions_test_base.h"

// This file contains the tests of the instruction semantic functions for
// RiscV vector floating point compare instructions.
namespace {

using Instruction = ::mpact::sim::generic::Instruction;

// The semantic functions.
using ::mpact::sim::riscv::Vmfeq;
using ::mpact::sim::riscv::Vmfge;
using ::mpact::sim::riscv::Vmfgt;
using ::mpact::sim::riscv::Vmfle;
using ::mpact::sim::riscv::Vmflt;
using ::mpact::sim::riscv::Vmfne;

// Needed types.
using ::absl::Span;
using ::mpact::sim::riscv::RVFpRegister;
using ::mpact::sim::riscv::ScopedFPStatus;

// Test utilities.
using ::mpact::sim::riscv::test::FPTypeInfo;
using ::mpact::sim::riscv::test::kA5Mask;
using ::mpact::sim::riscv::test::kFs1Name;
using ::mpact::sim::riscv::test::kLmul8Values;
using ::mpact::sim::riscv::test::kLmulSettings;
using ::mpact::sim::riscv::test::kSewSettingsByByteSize;
using ::mpact::sim::riscv::test::kVd;
using ::mpact::sim::riscv::test::kVectorLengthInBytes;
using ::mpact::sim::riscv::test::kVmask;
using ::mpact::sim::riscv::test::kVmaskName;
using ::mpact::sim::riscv::test::kVs1;
using ::mpact::sim::riscv::test::kVs2;
using ::mpact::sim::riscv::test::NaNBox;

class RiscVFPCompareInstructionsTest
    : public ::mpact::sim::riscv::test::RiscVFPInstructionsTestBase {
 public:
  // Helper function for testing binary mask vector-vector instructions that
  // use the mask bit.
  template <typename Vs2, typename Vs1>
  void BinaryMaskFPOpWithMaskTestHelperVV(
      absl::string_view name, int sew, Instruction *inst,
      std::function<uint8_t(Vs2, Vs1, bool)> operation) {
    int byte_sew = sew / 8;
    if (byte_sew != sizeof(Vs2) && byte_sew != sizeof(Vs1)) {
      FAIL() << name << ": selected element width != any operand types"
             << "sew: " << sew << " Vs2: " << sizeof(Vs2)
             << " Vs1: " << sizeof(Vs1);
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
    // Initialize input values.
    FillArrayWithRandomFPValues<Vs2>(vs2_span);
    FillArrayWithRandomFPValues<Vs1>(vs1_span);
    // Overwrite the first few values of the input data with infinities,
    // zeros, denormals and NaNs.
    using Vs2Int = typename FPTypeInfo<Vs2>::IntType;
    *reinterpret_cast<Vs2Int *>(&vs2_span[0]) = FPTypeInfo<Vs2>::kQNaN;
    *reinterpret_cast<Vs2Int *>(&vs2_span[1]) = FPTypeInfo<Vs2>::kSNaN;
    *reinterpret_cast<Vs2Int *>(&vs2_span[2]) = FPTypeInfo<Vs2>::kPosInf;
    *reinterpret_cast<Vs2Int *>(&vs2_span[3]) = FPTypeInfo<Vs2>::kNegInf;
    *reinterpret_cast<Vs2Int *>(&vs2_span[4]) = FPTypeInfo<Vs2>::kPosZero;
    *reinterpret_cast<Vs2Int *>(&vs2_span[5]) = FPTypeInfo<Vs2>::kNegZero;
    *reinterpret_cast<Vs2Int *>(&vs2_span[6]) = FPTypeInfo<Vs2>::kPosDenorm;
    *reinterpret_cast<Vs2Int *>(&vs2_span[7]) = FPTypeInfo<Vs2>::kNegDenorm;
    // Make every third value the same (at least if the types are same sized).
    for (int i = 0; i < std::min(vs1_size, vs2_size); i += 3) {
      vs1_span[i] = static_cast<Vs1>(vs2_span[i]);
    }

    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});
    // Modify the first mask bits to use each of the special floating point
    // values.
    vreg_[kVmask]->data_buffer()->Set<uint8_t>(0, 0xff);

    for (int i = 0; i < 8; i++) {
      auto vs2_name = absl::StrCat("v", kVs2 + i);
      SetVectorRegisterValues<Vs2>(
          {{vs2_name, vs2_span.subspan(vs2_size * i, vs2_size)}});
      auto vs1_name = absl::StrCat("v", kVs1 + i);
      SetVectorRegisterValues<Vs1>(
          {{vs1_name, vs1_span.subspan(vs1_size * i, vs1_size)}});
    }
    for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
      for (int vstart_count = 0; vstart_count < 4; vstart_count++) {
        for (int vlen_count = 0; vlen_count < 4; vlen_count++) {
          ClearVectorRegisterGroup(kVd, 8);
          int lmul8 = kLmul8Values[lmul_index];
          int lmul8_vs2 = lmul8 * sizeof(Vs2) / byte_sew;
          int num_values = lmul8 * kVectorLengthInBytes / (8 * byte_sew);
          int vstart = 0;
          if (vstart_count > 0) {
            vstart = absl::Uniform(absl::IntervalOpen, bitgen_, 0, num_values);
          }
          // Set vlen, but leave vlen high at least once.
          int vlen = 1024;
          if (vlen_count > 0) {
            vlen = absl::Uniform(absl::IntervalOpenClosed, bitgen_, vstart,
                                 num_values);
          }
          num_values = std::min(num_values, vlen);
          ASSERT_TRUE(vlen > vstart);
          // Configure vector unit for different lmul settings.
          uint32_t vtype = (kSewSettingsByByteSize[byte_sew] << 3) |
                           kLmulSettings[lmul_index];
          ConfigureVectorUnit(vtype, vlen);
          rv_vector_->set_vstart(vstart);

          inst->Execute();
          if ((lmul8_vs2 < 1) || (lmul8_vs2 > 64)) {
            EXPECT_TRUE(rv_vector_->vector_exception());
            rv_vector_->clear_vector_exception();
            continue;
          }

          EXPECT_FALSE(rv_vector_->vector_exception());
          EXPECT_EQ(rv_vector_->vstart(), 0);
          auto dest_span = vreg_[kVd]->data_buffer()->Get<uint8_t>();
          for (int i = 0; i < kVectorLengthInBytes * 8; i++) {
            int mask_index = i >> 3;
            int mask_offset = i & 0b111;
            bool mask_value = true;
            if (mask_index > 0) {
              mask_value = ((kA5Mask[mask_index] >> mask_offset) & 0b1) != 0;
            }
            uint8_t inst_value = dest_span[i >> 3];
            inst_value = (inst_value >> mask_offset) & 0b1;
            if ((i >= vstart) && (i < num_values)) {
              // Set rounding mode and perform the computation.
              ScopedFPStatus set_fpstatus(rv_fp_->host_fp_interface());
              uint8_t expected_value =
                  operation(vs2_value[i], vs1_value[i], mask_value);
              auto int_vs2_val =
                  *reinterpret_cast<typename FPTypeInfo<Vs2>::IntType *>(
                      &vs2_value[i]);
              auto int_vs1_val =
                  *reinterpret_cast<typename FPTypeInfo<Vs1>::IntType *>(
                      &vs1_value[i]);
              EXPECT_EQ(expected_value, inst_value)
                  << absl::StrCat(name, "[", i, "] op(", vs2_value[i], "[0x",
                                  absl::Hex(int_vs2_val), "], ", vs1_value[i],
                                  "[0x", absl::Hex(int_vs1_val), "])");
            } else {
              EXPECT_EQ(0, inst_value) << absl::StrCat(
                  name, "[", i, "]  0 != reg[][", i, "]   lmul8(", lmul8,
                  ")  vstart(", vstart, ")  num_values(", num_values, ")");
            }
          }
          if (HasFailure()) return;
        }
      }
    }
  }

  // Helper function for testing binary mask vector-vector instructions that do
  // not use the mask bit.
  template <typename Vs2, typename Vs1>
  void BinaryMaskFPOpTestHelperVV(absl::string_view name, int sew,
                                  Instruction *inst,
                                  std::function<uint8_t(Vs2, Vs1)> operation) {
    BinaryMaskFPOpWithMaskTestHelperVV<Vs2, Vs1>(
        name, sew, inst,
        [operation](Vs2 vs2, Vs1 vs1, bool mask_value) -> uint8_t {
          if (mask_value) {
            return operation(vs2, vs1);
          }
          return 0;
        });
  }

  // Helper function for testing mask vector-scalar/immediate instructions that
  // use the mask bit.
  template <typename Vs2, typename Fs1>
  void BinaryMaskFPOpWithMaskTestHelperVX(
      absl::string_view name, int sew, Instruction *inst,
      std::function<uint8_t(Vs2, Fs1, bool)> operation) {
    int byte_sew = sew / 8;
    if (byte_sew != sizeof(Vs2) && byte_sew != sizeof(Fs1)) {
      FAIL() << name << ": selected element width != any operand types"
             << "sew: " << sew << " Vs2: " << sizeof(Vs2)
             << " Rs1: " << sizeof(Fs1);
      return;
    }
    // Number of elements per vector register.
    constexpr int vs2_size = kVectorLengthInBytes / sizeof(Vs2);
    // Input values for 8 registers.
    Vs2 vs2_value[vs2_size * 8];
    auto vs2_span = Span<Vs2>(vs2_value);
    AppendVectorRegisterOperands({kVs2}, {});
    AppendRegisterOperands<RVFpRegister>({kFs1Name}, {});
    AppendVectorRegisterOperands({kVmask}, {kVd});
    // Initialize input values.
    FillArrayWithRandomValues<Vs2>(vs2_span);
    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});
    for (int i = 0; i < 8; i++) {
      auto vs2_name = absl::StrCat("v", kVs2 + i);
      SetVectorRegisterValues<Vs2>(
          {{vs2_name, vs2_span.subspan(vs2_size * i, vs2_size)}});
    }
    for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
      for (int vstart_count = 0; vstart_count < 4; vstart_count++) {
        for (int vlen_count = 0; vlen_count < 4; vlen_count++) {
          ClearVectorRegisterGroup(kVd, 8);
          int lmul8 = kLmul8Values[lmul_index];
          int lmul8_vs2 = lmul8 * sizeof(Vs2) / byte_sew;
          int num_values = lmul8 * kVectorLengthInBytes / (8 * byte_sew);
          int vstart = 0;
          if (vstart_count > 0) {
            vstart = absl::Uniform(absl::IntervalOpen, bitgen_, 0, num_values);
          }
          // Set vlen, but leave vlen high at least once.
          int vlen = 1024;
          if (vlen_count > 0) {
            vlen = absl::Uniform(absl::IntervalOpenClosed, bitgen_, vstart,
                                 num_values);
          }
          num_values = std::min(num_values, vlen);
          ASSERT_TRUE(vlen > vstart);
          // Configure vector unit for different lmul settings.
          uint32_t vtype = (kSewSettingsByByteSize[byte_sew] << 3) |
                           kLmulSettings[lmul_index];
          ConfigureVectorUnit(vtype, vlen);
          rv_vector_->set_vstart(vstart);

          // Generate a new rs1 value.
          Fs1 fs1_value = RandomFPValue<Fs1>();
          // Need to NaN box the value, that is, if the register value type is
          // wider than the data type for a floating point value, the upper bits
          // are all set to 1's.
          typename RVFpRegister::ValueType fs1_reg_value =
              NaNBox<Fs1, typename RVFpRegister::ValueType>(fs1_value);
          SetRegisterValues<typename RVFpRegister::ValueType, RVFpRegister>(
              {{kFs1Name, fs1_reg_value}});

          inst->Execute();
          if ((lmul8_vs2 < 1) || (lmul8_vs2 > 64)) {
            EXPECT_TRUE(rv_vector_->vector_exception());
            rv_vector_->clear_vector_exception();
            continue;
          }

          EXPECT_FALSE(rv_vector_->vector_exception());
          EXPECT_EQ(rv_vector_->vstart(), 0);
          auto dest_span = vreg_[kVd]->data_buffer()->Get<uint8_t>();
          for (int i = 0; i < kVectorLengthInBytes * 8; i++) {
            int mask_index = i >> 3;
            int mask_offset = i & 0b111;
            bool mask_value = ((kA5Mask[mask_index] >> mask_offset) & 0b1) != 0;
            uint8_t inst_value = dest_span[i >> 3];
            inst_value = (inst_value >> mask_offset) & 0b1;
            if ((i >= vstart) && (i < num_values)) {
              // Set rounding mode and perform the computation.

              ScopedFPStatus set_fpstatus(rv_fp_->host_fp_interface());
              uint8_t expected_value =
                  operation(vs2_value[i], fs1_value, mask_value);
              auto int_vs2_val =
                  *reinterpret_cast<typename FPTypeInfo<Vs2>::IntType *>(
                      &vs2_value[i]);
              auto int_fs1_val =
                  *reinterpret_cast<typename FPTypeInfo<Fs1>::IntType *>(
                      &fs1_value);
              EXPECT_EQ(expected_value, inst_value)
                  << absl::StrCat(name, "[", i, "] op(", vs2_value[i], "[0x",
                                  absl::Hex(int_vs2_val), "], ", fs1_value,
                                  "[0x", absl::Hex(int_fs1_val), "])");
            } else {
              EXPECT_EQ(0, inst_value) << absl::StrCat(
                  name, "  0 != reg[0][", i, "]   lmul8(", lmul8, ")");
            }
          }
          if (HasFailure()) return;
        }
      }
    }
  }

  // Helper function for testing mask vector-vector instructions that do not
  // use the mask bit.
  template <typename Vs2, typename Fs1>
  void BinaryMaskFPOpTestHelperVX(absl::string_view name, int sew,
                                  Instruction *inst,
                                  std::function<uint8_t(Vs2, Fs1)> operation) {
    BinaryMaskFPOpWithMaskTestHelperVX<Vs2, Fs1>(
        name, sew, inst,
        [operation](Vs2 vs2, Fs1 fs1, bool mask_value) -> uint8_t {
          if (mask_value) {
            return operation(vs2, fs1);
          }
          return 0;
        });
  }
};

// Testing vector floating point compare instructions.

// Vector floating point compare equal.
TEST_F(RiscVFPCompareInstructionsTest, Vmfeq) {
  SetSemanticFunction(&Vmfeq);
  BinaryMaskFPOpTestHelperVV<float, float>(
      "Vmfeq_vv32", /*sew*/ 32, instruction_,
      [](float vs2, float vs1) -> uint8_t { return (vs2 == vs1) ? 1 : 0; });
  ResetInstruction();
  SetSemanticFunction(&Vmfeq);
  BinaryMaskFPOpTestHelperVX<float, float>(
      "Vmfeq_vx32", /*sew*/ 32, instruction_,
      [](float vs2, float vs1) -> uint8_t { return (vs2 == vs1) ? 1 : 0; });
  ResetInstruction();
  SetSemanticFunction(&Vmfeq);
  BinaryMaskFPOpTestHelperVV<double, double>(
      "Vmfeq_vv64", /*sew*/ 64, instruction_,
      [](double vs2, double vs1) -> uint8_t { return (vs2 == vs1) ? 1 : 0; });
  ResetInstruction();
  SetSemanticFunction(&Vmfeq);
  BinaryMaskFPOpTestHelperVX<double, double>(
      "Vmfeq_vx64", /*sew*/ 64, instruction_,
      [](double vs2, double vs1) -> uint8_t { return (vs2 == vs1) ? 1 : 0; });
}

// Vector floating point compare less than or equal.
TEST_F(RiscVFPCompareInstructionsTest, Vmfle) {
  SetSemanticFunction(&Vmfle);
  BinaryMaskFPOpTestHelperVV<float, float>(
      "Vmfle_vv32", /*sew*/ 32, instruction_,
      [](float vs2, float vs1) -> uint8_t { return (vs2 <= vs1) ? 1 : 0; });
  ResetInstruction();
  SetSemanticFunction(&Vmfle);
  BinaryMaskFPOpTestHelperVX<float, float>(
      "Vmfle_vx32", /*sew*/ 32, instruction_,
      [](float vs2, float vs1) -> uint8_t { return (vs2 <= vs1) ? 1 : 0; });
  ResetInstruction();
  SetSemanticFunction(&Vmfle);
  BinaryMaskFPOpTestHelperVV<double, double>(
      "Vmfle_vv64", /*sew*/ 64, instruction_,
      [](double vs2, double vs1) -> uint8_t { return (vs2 <= vs1) ? 1 : 0; });
  ResetInstruction();
  SetSemanticFunction(&Vmfle);
  BinaryMaskFPOpTestHelperVX<double, double>(
      "Vmfle_vx64", /*sew*/ 64, instruction_,
      [](double vs2, double vs1) -> uint8_t { return (vs2 <= vs1) ? 1 : 0; });
}

// Vector floating point compare less than.
TEST_F(RiscVFPCompareInstructionsTest, Vmflt) {
  SetSemanticFunction(&Vmflt);
  BinaryMaskFPOpTestHelperVV<float, float>(
      "Vmflt_vv32", /*sew*/ 32, instruction_,
      [](float vs2, float vs1) -> uint8_t { return (vs2 < vs1) ? 1 : 0; });
  ResetInstruction();
  SetSemanticFunction(&Vmflt);
  BinaryMaskFPOpTestHelperVX<float, float>(
      "Vmflt_vx32", /*sew*/ 32, instruction_,
      [](float vs2, float vs1) -> uint8_t { return (vs2 < vs1) ? 1 : 0; });
  ResetInstruction();
  SetSemanticFunction(&Vmflt);
  BinaryMaskFPOpTestHelperVV<double, double>(
      "Vmflt_vv64", /*sew*/ 64, instruction_,
      [](double vs2, double vs1) -> uint8_t { return (vs2 < vs1) ? 1 : 0; });
  ResetInstruction();
  SetSemanticFunction(&Vmflt);
  BinaryMaskFPOpTestHelperVX<double, double>(
      "Vmflt_vx64", /*sew*/ 64, instruction_,
      [](double vs2, double vs1) -> uint8_t { return (vs2 < vs1) ? 1 : 0; });
}

// Vector floating point compare not equal.
TEST_F(RiscVFPCompareInstructionsTest, Vmfne) {
  SetSemanticFunction(&Vmfne);
  BinaryMaskFPOpTestHelperVV<float, float>(
      "Vmfne_vv32", /*sew*/ 32, instruction_,
      [](float vs2, float vs1) -> uint8_t { return (vs2 != vs1) ? 1 : 0; });
  ResetInstruction();
  SetSemanticFunction(&Vmfne);
  BinaryMaskFPOpTestHelperVX<float, float>(
      "Vmfne_vx32", /*sew*/ 32, instruction_,
      [](float vs2, float vs1) -> uint8_t { return (vs2 != vs1) ? 1 : 0; });
  ResetInstruction();
  SetSemanticFunction(&Vmfne);
  BinaryMaskFPOpTestHelperVV<double, double>(
      "Vmfne_vv64", /*sew*/ 64, instruction_,
      [](double vs2, double vs1) -> uint8_t { return (vs2 != vs1) ? 1 : 0; });
  ResetInstruction();
  SetSemanticFunction(&Vmfne);
  BinaryMaskFPOpTestHelperVX<double, double>(
      "Vmfne_vx64", /*sew*/ 64, instruction_,
      [](double vs2, double vs1) -> uint8_t { return (vs2 != vs1) ? 1 : 0; });
}

// Vector floating point compare greater than (used for Vector-Scalar
// comparisons).
TEST_F(RiscVFPCompareInstructionsTest, Vmfgt) {
  SetSemanticFunction(&Vmfgt);
  BinaryMaskFPOpTestHelperVX<float, float>(
      "Vmfgt_vx32", /*sew*/ 32, instruction_,
      [](float vs2, float vs1) -> uint8_t { return (vs2 > vs1) ? 1 : 0; });
  ResetInstruction();
  SetSemanticFunction(&Vmfgt);
  BinaryMaskFPOpTestHelperVX<double, double>(
      "Vmfgt_vx64", /*sew*/ 64, instruction_,
      [](double vs2, double vs1) -> uint8_t { return (vs2 > vs1) ? 1 : 0; });
}

// Vector floating point compare greater than or equal (used for Vector-Scalar
// comparisons).
TEST_F(RiscVFPCompareInstructionsTest, Vmfge) {
  SetSemanticFunction(&Vmfge);
  BinaryMaskFPOpTestHelperVX<float, float>(
      "Vmfge_vx32", /*sew*/ 32, instruction_,
      [](float vs2, float vs1) -> uint8_t { return (vs2 >= vs1) ? 1 : 0; });
  ResetInstruction();
  SetSemanticFunction(&Vmfge);
  BinaryMaskFPOpTestHelperVX<double, double>(
      "Vmfge_vx64", /*sew*/ 64, instruction_,
      [](double vs2, double vs1) -> uint8_t { return (vs2 >= vs1) ? 1 : 0; });
}

}  // namespace
