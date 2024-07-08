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

#include "riscv/riscv_vector_fp_unary_instructions.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <ios>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/operand_interface.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_fp_host.h"
#include "riscv/riscv_fp_info.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_vector_state.h"
#include "riscv/test/riscv_vector_fp_test_utilities.h"
#include "riscv/test/riscv_vector_instructions_test_base.h"

namespace {

using Instruction = ::mpact::sim::generic::Instruction;
using ::mpact::sim::generic::operator*;
using ::mpact::sim::riscv::FPExceptions;

// Functions to test.
using ::mpact::sim::riscv::Vfclassv;
using ::mpact::sim::riscv::Vfcvtfxuv;
using ::mpact::sim::riscv::Vfcvtfxv;
using ::mpact::sim::riscv::Vfcvtrtzxfv;
using ::mpact::sim::riscv::Vfcvtrtzxufv;
using ::mpact::sim::riscv::Vfcvtxfv;
using ::mpact::sim::riscv::Vfcvtxufv;
using ::mpact::sim::riscv::Vfmvfs;
using ::mpact::sim::riscv::Vfmvsf;
using ::mpact::sim::riscv::Vfncvtffw;
using ::mpact::sim::riscv::Vfncvtfxuw;
using ::mpact::sim::riscv::Vfncvtfxw;
using ::mpact::sim::riscv::Vfncvtrodffw;
using ::mpact::sim::riscv::Vfncvtrtzxfw;
using ::mpact::sim::riscv::Vfncvtrtzxufw;
using ::mpact::sim::riscv::Vfncvtxfw;
using ::mpact::sim::riscv::Vfncvtxufw;
using ::mpact::sim::riscv::Vfrec7v;
using ::mpact::sim::riscv::Vfrsqrt7v;
using ::mpact::sim::riscv::Vfsqrtv;
using ::mpact::sim::riscv::Vfwcvtffv;
using ::mpact::sim::riscv::Vfwcvtfxuv;
using ::mpact::sim::riscv::Vfwcvtfxv;
using ::mpact::sim::riscv::Vfwcvtrtzxfv;
using ::mpact::sim::riscv::Vfwcvtrtzxufv;
using ::mpact::sim::riscv::Vfwcvtxfv;
using ::mpact::sim::riscv::Vfwcvtxufv;

using ::absl::Span;
using ::mpact::sim::riscv::FPRoundingMode;
using ::mpact::sim::riscv::RiscVCsrInterface;
using ::mpact::sim::riscv::RiscVFPState;
using ::mpact::sim::riscv::RVFpRegister;
using ::mpact::sim::riscv::ScopedFPStatus;

using ::mpact::sim::riscv::test::FPCompare;
using ::mpact::sim::riscv::test::FPTypeInfo;
using ::mpact::sim::riscv::test::kA5Mask;
using ::mpact::sim::riscv::test::kLmul8Values;
using ::mpact::sim::riscv::test::kLmulSettingByLogSize;
using ::mpact::sim::riscv::test::kLmulSettings;
using ::mpact::sim::riscv::test::kSewSettingsByByteSize;
using ::mpact::sim::riscv::test::kVd;
using ::mpact::sim::riscv::test::kVectorLengthInBytes;
using ::mpact::sim::riscv::test::kVmask;
using ::mpact::sim::riscv::test::kVmaskName;
using ::mpact::sim::riscv::test::kVs2;
using ::mpact::sim::riscv::test::kVs2Name;

constexpr char kFs1Name[] = "f4";
constexpr int kFs1 = 4;

// Test fixture that extends the Base fixture for unary floating point
// instructions.
class RiscVFPUnaryInstructionsTest
    : public ::mpact::sim::riscv::test::RiscVVectorInstructionsTestBase {
 public:
  RiscVFPUnaryInstructionsTest() {
    rv_fp_ = new mpact::sim::riscv::RiscVFPState(state_);
    state_->set_rv_fp(rv_fp_);
  }
  ~RiscVFPUnaryInstructionsTest() override {
    state_->set_rv_fp(nullptr);
    delete rv_fp_;
  }

  // This method uses random values for each field in the fp number.
  template <typename T>
  void FillArrayWithRandomFPValues(absl::Span<T> span) {
    using UInt = typename FPTypeInfo<T>::IntType;
    for (auto &val : span) {
      UInt sign = absl::Uniform(absl::IntervalClosed, bitgen_, 0ULL, 1ULL);
      UInt exp = absl::Uniform(absl::IntervalClosedOpen, bitgen_, 0ULL,
                               1ULL << FPTypeInfo<T>::kExpSize);
      UInt sig = absl::Uniform(absl::IntervalClosedOpen, bitgen_, 0ULL,
                               1ULL << FPTypeInfo<T>::kSigSize);
      UInt value = (sign & 1) << (FPTypeInfo<T>::kBitSize - 1) |
                   (exp << FPTypeInfo<T>::kSigSize) | sig;
      val = *reinterpret_cast<T *>(&value);
    }
  }

  // Floating point test needs to ensure to use the fp special values (inf, NaN
  // etc.) during testing, not just random values.
  template <typename Vd, typename Vs2>
  void UnaryOpFPTestHelperV(absl::string_view name, int sew, Instruction *inst,
                            int delta_position,
                            std::function<Vd(Vs2)> operation) {
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
    AppendVectorRegisterOperands({kVs2, kVmask}, {kVd});
    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});
    // Iterate across different lmul values.
    for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
      // Initialize input values.
      FillArrayWithRandomFPValues<Vs2>(vs2_span);
      using Vs2Int = typename FPTypeInfo<Vs2>::IntType;
      // Overwrite the first few values of the input data with infinities,
      // zeros, denormals and NaNs.
      *reinterpret_cast<Vs2Int *>(&vs2_span[0]) = FPTypeInfo<Vs2>::kQNaN;
      *reinterpret_cast<Vs2Int *>(&vs2_span[1]) = FPTypeInfo<Vs2>::kSNaN;
      *reinterpret_cast<Vs2Int *>(&vs2_span[2]) = FPTypeInfo<Vs2>::kPosInf;
      *reinterpret_cast<Vs2Int *>(&vs2_span[3]) = FPTypeInfo<Vs2>::kNegInf;
      *reinterpret_cast<Vs2Int *>(&vs2_span[4]) = FPTypeInfo<Vs2>::kPosZero;
      *reinterpret_cast<Vs2Int *>(&vs2_span[5]) = FPTypeInfo<Vs2>::kNegZero;
      *reinterpret_cast<Vs2Int *>(&vs2_span[6]) = FPTypeInfo<Vs2>::kPosDenorm;
      *reinterpret_cast<Vs2Int *>(&vs2_span[7]) = FPTypeInfo<Vs2>::kNegDenorm;
      *reinterpret_cast<Vs2Int *>(&vs2_span[8]) = 0x0119515e;
      *reinterpret_cast<Vs2Int *>(&vs2_span[9]) = 0x0007fea3;
      *reinterpret_cast<Vs2Int *>(&vs2_span[10]) = 0x800bc58f;
      // Modify the first mask bits to use each of the special floating point
      // values.
      vreg_[kVmask]->data_buffer()->Set<uint8_t>(0, 0xff);
      vreg_[kVmask]->data_buffer()->Set<uint8_t>(1, 0xa5 | 0x3);
      // Set values for all 8 vector registers in the vector register group.
      for (int i = 0; i < 8; i++) {
        auto vs2_name = absl::StrCat("v", kVs2 + i);
        SetVectorRegisterValues<Vs2>(
            {{vs2_name, vs2_span.subspan(vs2_size * i, vs2_size)}});
      }
      int lmul8 = kLmul8Values[lmul_index];
      int lmul8_vd = lmul8 * sizeof(Vd) / byte_sew;
      int lmul8_vs2 = lmul8 * sizeof(Vs2) / byte_sew;
      int num_reg_values = lmul8 * kVectorLengthInBytes / (8 * byte_sew);
      // Configure vector unit for different lmul settings.
      uint32_t vtype =
          (kSewSettingsByByteSize[byte_sew] << 3) | kLmulSettings[lmul_index];
      int vstart = 0;
      // Try different vstart values (updated at the bottom of the loop).
      for (int vstart_count = 0; vstart_count < 4; vstart_count++) {
        int vlen = 1024;
        // Try different vector lengths (updated at the bottom of the loop).
        for (int vlen_count = 0; vlen_count < 4; vlen_count++) {
          ASSERT_TRUE(vlen > vstart);
          int num_values = std::min(num_reg_values, vlen);
          ConfigureVectorUnit(vtype, vlen);
          // Iterate across rounding modes.
          for (int rm : {0, 1, 2, 3, 4}) {
            rv_fp_->SetRoundingMode(static_cast<FPRoundingMode>(rm));
            rv_vector_->set_vstart(vstart);
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

            EXPECT_FALSE(rv_vector_->vector_exception());
            EXPECT_EQ(rv_vector_->vstart(), 0);
            int count = 0;
            for (int reg = kVd; reg < kVd + 8; reg++) {
              for (int i = 0; i < kVectorLengthInBytes / sizeof(Vd); i++) {
                int mask_index = count >> 3;
                int mask_offset = count & 0b111;
                bool mask_value = true;
                // The first 10 bits of the mask are set to true above, so only
                // read the mask value for the entries after that.
                if (count >= 10) {
                  mask_value =
                      ((kA5Mask[mask_index] >> mask_offset) & 0b1) != 0;
                }
                auto reg_val = vreg_[reg]->data_buffer()->Get<Vd>(i);
                if ((count >= vstart) && mask_value && (count < num_values)) {
                  auto op_val = operation(vs2_value[count]);
                  // Do separate comparison if the result is a NaN.
                  auto int_reg_val =
                      *reinterpret_cast<typename FPTypeInfo<Vd>::IntType *>(
                          &reg_val);
                  auto int_op_val =
                      *reinterpret_cast<typename FPTypeInfo<Vd>::IntType *>(
                          &op_val);
                  auto int_vs2_val =
                      *reinterpret_cast<typename FPTypeInfo<Vs2>::IntType *>(
                          &vs2_value[count]);
                  FPCompare<Vd>(
                      op_val, reg_val, delta_position,
                      absl::StrCat(name, "[", count, "] op(", vs2_value[count],
                                   "[0x", absl::Hex(int_vs2_val),
                                   "]) = ", absl::Hex(int_op_val), " != reg[",
                                   reg, "][", i, "]  (", reg_val, " [0x",
                                   absl::Hex(int_reg_val), "]) lmul8(", lmul8,
                                   ") rm = ", *(rv_fp_->GetRoundingMode())));
                } else {
                  EXPECT_EQ(0, reg_val) << absl::StrCat(
                      name, "  0 != reg[", reg, "][", i, "]  (", reg_val,
                      " [0x", absl::Hex(reg_val), "]) lmul8(", lmul8, ")");
                }
                count++;
              }
            }
          }
          if (HasFailure()) {
            return;
          }
          vlen = absl::Uniform(absl::IntervalOpenClosed, bitgen_, vstart,
                               num_reg_values);
        }
        vstart = absl::Uniform(absl::IntervalOpen, bitgen_, 0, num_reg_values);
      }
    }
  }

  // Floating point test needs to ensure to use the fp special values (inf, NaN
  // etc.) during testing, not just random values.
  template <typename Vd, typename Vs2>
  void UnaryOpWithFflagsFPTestHelperV(
      absl::string_view name, int sew, Instruction *inst, int delta_position,
      std::function<std::tuple<Vd, uint32_t>(Vs2)> operation) {
    using VdInt = typename FPTypeInfo<Vd>::IntType;
    using Vs2Int = typename FPTypeInfo<Vs2>::IntType;
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
    AppendVectorRegisterOperands({kVs2, kVmask}, {kVd});
    auto *op = rv_fp_->fflags()->CreateSetDestinationOperand(0, "fflags");
    instruction_->AppendDestination(op);
    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});
    // Iterate across different lmul values.
    for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
      // Initialize input values.
      FillArrayWithRandomFPValues<Vs2>(vs2_span);
      // Overwrite the first few values of the input data with infinities,
      // zeros, denormals and NaNs.
      *reinterpret_cast<Vs2Int *>(&vs2_span[0]) = FPTypeInfo<Vs2>::kQNaN;
      *reinterpret_cast<Vs2Int *>(&vs2_span[1]) = FPTypeInfo<Vs2>::kSNaN;
      *reinterpret_cast<Vs2Int *>(&vs2_span[2]) = FPTypeInfo<Vs2>::kPosInf;
      *reinterpret_cast<Vs2Int *>(&vs2_span[3]) = FPTypeInfo<Vs2>::kNegInf;
      *reinterpret_cast<Vs2Int *>(&vs2_span[4]) = FPTypeInfo<Vs2>::kPosZero;
      *reinterpret_cast<Vs2Int *>(&vs2_span[5]) = FPTypeInfo<Vs2>::kNegZero;
      *reinterpret_cast<Vs2Int *>(&vs2_span[6]) = FPTypeInfo<Vs2>::kPosDenorm;
      *reinterpret_cast<Vs2Int *>(&vs2_span[7]) = FPTypeInfo<Vs2>::kNegDenorm;
      *reinterpret_cast<Vs2Int *>(&vs2_span[8]) = 0x0119515e;
      *reinterpret_cast<Vs2Int *>(&vs2_span[9]) = 0x0007fea3;
      *reinterpret_cast<Vs2Int *>(&vs2_span[10]) = 0x800bc58f;
      // Modify the first mask bits to use each of the special floating point
      // values.
      vreg_[kVmask]->data_buffer()->Set<uint8_t>(0, 0xff);
      vreg_[kVmask]->data_buffer()->Set<uint8_t>(1, 0xa5 | 0x3);
      // Set values for all 8 vector registers in the vector register group.
      for (int i = 0; i < 8; i++) {
        auto vs2_name = absl::StrCat("v", kVs2 + i);
        SetVectorRegisterValues<Vs2>(
            {{vs2_name, vs2_span.subspan(vs2_size * i, vs2_size)}});
      }
      int lmul8 = kLmul8Values[lmul_index];
      int lmul8_vd = lmul8 * sizeof(Vd) / byte_sew;
      int lmul8_vs2 = lmul8 * sizeof(Vs2) / byte_sew;
      int num_reg_values = lmul8 * kVectorLengthInBytes / (8 * byte_sew);
      // Configure vector unit for different lmul settings.
      uint32_t vtype =
          (kSewSettingsByByteSize[byte_sew] << 3) | kLmulSettings[lmul_index];
      int vstart = 0;
      // Try different vstart values (updated at the bottom of the loop).
      for (int vstart_count = 0; vstart_count < 4; vstart_count++) {
        int vlen = 1024;
        // Try different vector lengths (updated at the bottom of the loop).
        for (int vlen_count = 0; vlen_count < 4; vlen_count++) {
          ASSERT_TRUE(vlen > vstart);
          int num_values = std::min(num_reg_values, vlen);
          ConfigureVectorUnit(vtype, vlen);
          // Iterate across rounding modes.
          for (int rm : {0, 1, 2, 3, 4}) {
            rv_vector_->set_vstart(vstart);
            ClearVectorRegisterGroup(kVd, 8);
            // Set rounding mode and clear flags.
            rv_fp_->SetRoundingMode(static_cast<FPRoundingMode>(rm));
            rv_fp_->fflags()->Write(0U);

            inst->Execute();

            // Get the flags for the instruction execution.
            uint32_t fflags = rv_fp_->fflags()->AsUint32();

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

            EXPECT_FALSE(rv_vector_->vector_exception());
            EXPECT_EQ(rv_vector_->vstart(), 0);
            int count = 0;
            // Clear flags for the test execution.
            rv_fp_->fflags()->Write(0U);
            uint32_t fflags_test = 0;
            for (int reg = kVd; reg < kVd + 8; reg++) {
              for (int i = 0; i < kVectorLengthInBytes / sizeof(Vd); i++) {
                int mask_index = count >> 3;
                int mask_offset = count & 0b111;
                bool mask_value = true;
                // The first 10 bits of the mask are set to true above, so only
                // read the mask value for the entries after that.
                if (count >= 10) {
                  mask_value =
                      ((kA5Mask[mask_index] >> mask_offset) & 0b1) != 0;
                }
                auto reg_val = vreg_[reg]->data_buffer()->Get<Vd>(i);
                if ((count >= vstart) && mask_value && (count < num_values)) {
                  Vd op_val;
                  uint32_t flag;
                  {
                    ScopedFPStatus set_fp_status(rv_fp_->host_fp_interface());
                    auto [op_val_tmp, flag_tmp] = operation(vs2_value[count]);
                    op_val = op_val_tmp;
                    flag = flag_tmp;
                  }
                  fflags_test |= (rv_fp_->fflags()->AsUint32() | flag);
                  // Do separate comparison if the result is a NaN.
                  auto int_reg_val = *reinterpret_cast<VdInt *>(&reg_val);
                  auto int_op_val = *reinterpret_cast<VdInt *>(&op_val);
                  auto int_vs2_val =
                      *reinterpret_cast<Vs2Int *>(&vs2_value[count]);
                  FPCompare<Vd>(
                      op_val, reg_val, delta_position,
                      absl::StrCat(name, "[", count, "] op(", vs2_value[count],
                                   "[0x", absl::Hex(int_vs2_val),
                                   "]) = ", absl::Hex(int_op_val), " != reg[",
                                   reg, "][", i, "]  (", reg_val, " [0x",
                                   absl::Hex(int_reg_val), "]) lmul8(", lmul8,
                                   ") rm = ", *(rv_fp_->GetRoundingMode())));
                } else {
                  EXPECT_EQ(0, reg_val) << absl::StrCat(
                      name, "  0 != reg[", reg, "][", i, "]  (", reg_val,
                      " [0x", absl::Hex(reg_val), "]) lmul8(", lmul8, ")");
                }
                count++;
              }
            }
            EXPECT_EQ(fflags, fflags_test) << name;
          }
          if (HasFailure()) {
            return;
          }
          vlen = absl::Uniform(absl::IntervalOpenClosed, bitgen_, vstart,
                               num_reg_values);
        }
        vstart = absl::Uniform(absl::IntervalOpen, bitgen_, 0, num_reg_values);
      }
    }
  }

 protected:
  mpact::sim::riscv::RiscVFPState *rv_fp_ = nullptr;
  RiscVCsrInterface *fflags_ = nullptr;
};

// Templated helper function for classifying fp numbers.
template <typename T>
typename FPTypeInfo<T>::IntType VfclassVHelper(T val) {
  auto fp_class = std::fpclassify(val);
  switch (fp_class) {
    case FP_INFINITE:
      return std::signbit(val) ? 1 : 1 << 7;
    case FP_NAN: {
      auto uint_val =
          *reinterpret_cast<typename FPTypeInfo<T>::IntType *>(&val);
      bool quiet_nan = (uint_val >> (FPTypeInfo<T>::kSigSize - 1)) & 1;
      return quiet_nan ? 1 << 9 : 1 << 8;
    }
    case FP_ZERO:
      return std::signbit(val) ? 1 << 3 : 1 << 4;
    case FP_SUBNORMAL:
      return std::signbit(val) ? 1 << 2 : 1 << 5;
    case FP_NORMAL:
      return std::signbit(val) ? 1 << 1 : 1 << 6;
  }
  return 0;
}

// Test fp classify.
TEST_F(RiscVFPUnaryInstructionsTest, Vfclassv) {
  SetSemanticFunction(&Vfclassv);
  UnaryOpFPTestHelperV<uint32_t, float>(
      "Vfclassv_32", /*sew*/ 32, instruction_, /*delta_position*/ 32,
      [](float vs2) -> uint32_t { return VfclassVHelper(vs2); });
  ResetInstruction();
  SetSemanticFunction(&Vfclassv);
  UnaryOpFPTestHelperV<uint64_t, double>(
      "Vfclassv_64", /*sew*/ 64, instruction_, /*delta_position*/ 64,
      [](double vs2) -> uint64_t { return VfclassVHelper(vs2); });
}

// Test convert from unsigned integer to fp.
TEST_F(RiscVFPUnaryInstructionsTest, Vfcvtfxuv) {
  SetSemanticFunction(&Vfcvtfxuv);
  UnaryOpTestHelperV<float, uint32_t>(
      "Vfcvt.f.xu.v_32", /*sew*/ 32, instruction_,
      [](uint32_t vs2) -> float { return static_cast<float>(vs2); });
  ResetInstruction();
  SetSemanticFunction(&Vfcvtfxuv);
  UnaryOpTestHelperV<double, uint64_t>(
      "Vfcvt.f.xu.v_64", /*sew*/ 64, instruction_,
      [](uint64_t vs2) -> double { return static_cast<double>(vs2); });
}

// Test convert from signed integer to fp.
TEST_F(RiscVFPUnaryInstructionsTest, Vfcvtfxv) {
  SetSemanticFunction(&Vfcvtfxv);
  UnaryOpTestHelperV<float, int32_t>(
      "Vfcvt.f.x.v_32", /*sew*/ 32, instruction_,
      [](int32_t vs2) -> float { return static_cast<float>(vs2); });
  ResetInstruction();
  SetSemanticFunction(&Vfcvtfxv);
  UnaryOpTestHelperV<double, int64_t>(
      "Vfcvt.f.x.v_64", /*sew*/ 64, instruction_,
      [](int64_t vs2) -> double { return static_cast<double>(vs2); });
}

// Helper function for fp to integer conversions.
template <typename F, typename I>
std::tuple<I, uint32_t> ConvertHelper(F value, RiscVFPState *fp_state) {
  constexpr F kMin = static_cast<F>(std::numeric_limits<I>::min());
  constexpr F kMax = static_cast<F>(std::numeric_limits<I>::max());
  ScopedFPStatus status(fp_state->host_fp_interface());
  auto fp_class = std::fpclassify(value);
  switch (fp_class) {
    case FP_INFINITE:
      return std::make_tuple(std::signbit(value)
                                 ? std::numeric_limits<I>::min()
                                 : std::numeric_limits<I>::max(),
                             static_cast<uint32_t>(FPExceptions::kInvalidOp));
    case FP_NAN:
      return std::make_tuple(std::numeric_limits<I>::max(),
                             static_cast<uint32_t>(FPExceptions::kInvalidOp));
    case FP_ZERO:
      return std::make_tuple(0, 0);
    case FP_SUBNORMAL:
    case FP_NORMAL:
      if (value > kMax) {
        return std::make_tuple(std::numeric_limits<I>::max(),
                               static_cast<uint32_t>(FPExceptions::kInvalidOp));
      }
      if (value < kMin) {
        if (std::is_unsigned<I>::value) {
          if ((value > -1.0) &&
              (static_cast<typename std::make_signed<I>::type>(value) == 0)) {
            return std::make_tuple(
                0, static_cast<uint32_t>(FPExceptions::kInexact));
          }
        }
        if (value < kMin) {
          return std::make_tuple(
              std::numeric_limits<I>::min(),
              static_cast<uint32_t>(FPExceptions::kInvalidOp));
        }
      }
  }
  return std::make_tuple(static_cast<I>(value), 0);
}

// Test convert from fp to signed integer with truncation.
TEST_F(RiscVFPUnaryInstructionsTest, Vfcvtrtzxfv) {
  SetSemanticFunction(&Vfcvtrtzxfv);
  rv_fp_->SetRoundingMode(FPRoundingMode::kRoundTowardsZero);
  UnaryOpWithFflagsFPTestHelperV<int32_t, float>(
      "Vfcvt.rtz.x.f.v_32", /*sew*/ 32, instruction_, /*delta_position*/ 32,
      [this](float vs2) -> std::tuple<int32_t, uint32_t> {
        return ConvertHelper<float, int32_t>(vs2, this->rv_fp_);
      });
  ResetInstruction();
  SetSemanticFunction(&Vfcvtrtzxfv);
  rv_fp_->SetRoundingMode(FPRoundingMode::kRoundTowardsZero);
  UnaryOpWithFflagsFPTestHelperV<int64_t, double>(
      "Vfcvt.rtz.x.f.v_64", /*sew*/ 64, instruction_, /*delta_position*/ 64,
      [this](double vs2) -> std::tuple<int64_t, uint32_t> {
        return ConvertHelper<double, int64_t>(vs2, this->rv_fp_);
      });
}

// Test convert from fp to unsigned integer with truncation.
TEST_F(RiscVFPUnaryInstructionsTest, Vfcvtrtzxufv) {
  SetSemanticFunction(&Vfcvtrtzxufv);
  rv_fp_->SetRoundingMode(FPRoundingMode::kRoundTowardsZero);
  UnaryOpWithFflagsFPTestHelperV<uint32_t, float>(
      "Vfcvt.rtz.xu.f.v_32", /*sew*/ 32, instruction_, /*delta_position*/ 32,
      [this](float vs2) -> std::tuple<uint32_t, uint32_t> {
        return ConvertHelper<float, uint32_t>(vs2, this->rv_fp_);
      });
  ResetInstruction();
  SetSemanticFunction(&Vfcvtrtzxufv);
  rv_fp_->SetRoundingMode(FPRoundingMode::kRoundTowardsZero);
  UnaryOpWithFflagsFPTestHelperV<uint64_t, double>(
      "Vfcvt.rtz.xu.f.v_64", /*sew*/ 64, instruction_, /*delta_position*/ 64,
      [this](double vs2) -> std::tuple<uint64_t, uint32_t> {
        return ConvertHelper<double, uint64_t>(vs2, this->rv_fp_);
      });
}

// Test convert from fp to signed integer with rounding.
TEST_F(RiscVFPUnaryInstructionsTest, Vfcvtxfv) {
  SetSemanticFunction(&Vfcvtxfv);
  UnaryOpWithFflagsFPTestHelperV<int32_t, float>(
      "Vfcvt.x.f.v_32", /*sew*/ 32, instruction_, /*delta_position*/ 32,
      [this](float vs2) -> std::tuple<int32_t, uint32_t> {
        return ConvertHelper<float, int32_t>(vs2, this->rv_fp_);
      });
  ResetInstruction();
  SetSemanticFunction(&Vfcvtxfv);
  UnaryOpWithFflagsFPTestHelperV<int64_t, double>(
      "Vfcvt.x.f.v_64", /*sew*/ 64, instruction_, /*delta_position*/ 64,
      [this](double vs2) -> std::tuple<int64_t, uint32_t> {
        return ConvertHelper<double, int64_t>(vs2, this->rv_fp_);
      });
}

// Test convert from fp to unsigned integer with rounding.
TEST_F(RiscVFPUnaryInstructionsTest, Vfcvtxufv) {
  SetSemanticFunction(&Vfcvtxufv);
  UnaryOpWithFflagsFPTestHelperV<uint32_t, float>(
      "Vfcvt.xu.f.v_32", /*sew*/ 32, instruction_, /*delta_position*/ 32,
      [this](float vs2) -> std::tuple<uint32_t, uint32_t> {
        return ConvertHelper<float, uint32_t>(vs2, this->rv_fp_);
      });
  ResetInstruction();
  SetSemanticFunction(&Vfcvtxufv);
  UnaryOpWithFflagsFPTestHelperV<uint64_t, double>(
      "Vfcvt.xu.f.v_64", /*sew*/ 64, instruction_, /*delta_position*/ 64,
      [this](double vs2) -> std::tuple<uint64_t, uint32_t> {
        return ConvertHelper<double, uint64_t>(vs2, this->rv_fp_);
      });
}

// Test vfmv.f.s instruction - move element 0 to scalar fp register.
TEST_F(RiscVFPUnaryInstructionsTest, VfmvToScalar) {
  SetSemanticFunction(&Vfmvfs);
  AppendRegisterOperands({}, {kFs1Name});
  AppendVectorRegisterOperands({kVs2}, {});
  for (int byte_sew : {1, 2, 4, 8}) {
    int vlen = kVectorLengthInBytes / byte_sew;
    uint32_t vtype =
        (kSewSettingsByByteSize[byte_sew] << 3) | kLmulSettingByLogSize[4];
    ConfigureVectorUnit(vtype, vlen);
    if (byte_sew < 4) {
      instruction_->Execute();
      EXPECT_TRUE(rv_vector_->vector_exception());
      continue;
    }
    // Test 10 different values.
    for (int i = 0; i < 10; i++) {
      uint64_t value;
      switch (byte_sew) {
        case 4: {
          auto val32 = RandomValue<uint32_t>();
          value = 0xffff'ffff'0000'0000ULL | static_cast<uint64_t>(val32);
          SetVectorRegisterValues<uint32_t>(
              {{kVs2Name, absl::Span<const uint32_t>(&val32, 1)}});
          break;
        }
        case 8: {
          value = RandomValue<uint64_t>();
          SetVectorRegisterValues<uint64_t>(
              {{kVs2Name, absl::Span<const uint64_t>(&value, 1)}});
          break;
        }
      }
      instruction_->Execute();
      EXPECT_EQ(freg_[kFs1]->data_buffer()->Get<RVFpRegister::ValueType>(0),
                static_cast<RVFpRegister::ValueType>(value));
    }
  }
}

// Test vfmv.f.s instruction - move scalar fp register to element 0.
TEST_F(RiscVFPUnaryInstructionsTest, VfmvFromScalar) {
  SetSemanticFunction(&Vfmvsf);
  AppendRegisterOperands({kFs1Name}, {});
  AppendVectorRegisterOperands({}, {kVd});
  for (int byte_sew : {1, 2, 4, 8}) {
    int vlen = kVectorLengthInBytes / byte_sew;
    uint32_t vtype =
        (kSewSettingsByByteSize[byte_sew] << 3) | kLmulSettingByLogSize[4];
    ConfigureVectorUnit(vtype, vlen);
    if (byte_sew < 4) {
      instruction_->Execute();
      EXPECT_TRUE(rv_vector_->vector_exception());
      continue;
    }
    // Test 10 different values.
    for (int i = 0; i < 10; i++) {
      auto value = RandomValue<RVFpRegister::ValueType>();
      freg_[kFs1]->data_buffer()->Set<RVFpRegister::ValueType>(0, value);
      instruction_->Execute();
      switch (byte_sew) {
        case 4:
          EXPECT_EQ(vreg_[kVd]->data_buffer()->Get<uint32_t>(0),
                    static_cast<uint32_t>(value));
          break;
        case 8:
          EXPECT_EQ(vreg_[kVd]->data_buffer()->Get<uint64_t>(0),
                    static_cast<uint64_t>(value));
          break;
      }
    }
  }
}

// Test narrowing convert from unsigned integer to fp.
TEST_F(RiscVFPUnaryInstructionsTest, Vfncvtfxuw) {
  SetSemanticFunction(&Vfncvtfxuw);
  UnaryOpTestHelperV<float, uint64_t>(
      "Vfncvt.f.xu.w_64", /*sew*/ 64, instruction_,
      [](uint64_t vs2) -> float { return static_cast<float>(vs2); });
}

// Test narrowing convert from signed integer to fp.
TEST_F(RiscVFPUnaryInstructionsTest, Vfncvtfxw) {
  SetSemanticFunction(&Vfncvtfxw);
  UnaryOpTestHelperV<float, int64_t>(
      "Vfncvt.f.x.w_64", /*sew*/ 64, instruction_,
      [](int64_t vs2) -> float { return static_cast<float>(vs2); });
}

// Test narrowing convert fp to fp.
TEST_F(RiscVFPUnaryInstructionsTest, Vfncvtffw) {
  SetSemanticFunction(&Vfncvtffw);
  UnaryOpTestHelperV<float, double>(
      "Vfncvt.f.f.w_64", /*sew*/ 64, instruction_,
      [](double vs2) -> float { return static_cast<float>(vs2); });
}

// Test narrowing convert fp to fp with round to odd.
TEST_F(RiscVFPUnaryInstructionsTest, Vfncvtrodffw) {
  SetSemanticFunction(&Vfncvtrodffw);
  UnaryOpTestHelperV<float, double>(
      "Vfncvt.rod.f.f.w_64", /*sew*/ 64, instruction_, [](double vs2) -> float {
        if (std::isnan(vs2) || std::isinf(vs2)) {
          return static_cast<float>(vs2);
        }
        using UIntD = typename FPTypeInfo<double>::IntType;
        using UIntF = typename FPTypeInfo<float>::IntType;
        UIntD uval = *reinterpret_cast<UIntD *>(&vs2);
        int diff = FPTypeInfo<double>::kSigSize - FPTypeInfo<float>::kSigSize;
        UIntF bit = (uval & (FPTypeInfo<double>::kSigMask >> diff)) != 0;
        float res = static_cast<float>(vs2);
        // The narrowing conversion may have generated an infinity, so check
        // for infinity before doing rounding.
        if (std::isinf(res)) return res;
        UIntF ures = *reinterpret_cast<UIntF *>(&res) | bit;
        return *reinterpret_cast<float *>(&ures);
      });
}

// Test narrowing convert from fp to signed integer with truncation.
TEST_F(RiscVFPUnaryInstructionsTest, Vfncvtrtzxfw) {
  SetSemanticFunction(&Vfncvtrtzxfw);
  rv_fp_->SetRoundingMode(FPRoundingMode::kRoundTowardsZero);
  UnaryOpWithFflagsFPTestHelperV<int16_t, float>(
      "Vfncvt.rtz.x.f.w_32", /*sew*/ 32, instruction_, /*delta_position*/ 32,
      [this](float vs2) -> std::tuple<int16_t, uint32_t> {
        return ConvertHelper<float, int16_t>(vs2, this->rv_fp_);
      });
  ResetInstruction();
  SetSemanticFunction(&Vfncvtrtzxfw);
  rv_fp_->SetRoundingMode(FPRoundingMode::kRoundTowardsZero);
  UnaryOpWithFflagsFPTestHelperV<int32_t, double>(
      "Vfncvt.rtz.x.f.w_64", /*sew*/ 64, instruction_, /*delta_position*/ 64,
      [this](double vs2) -> std::tuple<int32_t, uint32_t> {
        return ConvertHelper<double, int32_t>(vs2, this->rv_fp_);
      });
}

// Test narrowing convert from fp to unsigned integer with truncation.
TEST_F(RiscVFPUnaryInstructionsTest, Vfncvtrtzxufw) {
  SetSemanticFunction(&Vfncvtrtzxufw);
  rv_fp_->SetRoundingMode(FPRoundingMode::kRoundTowardsZero);
  UnaryOpWithFflagsFPTestHelperV<uint16_t, float>(
      "Vfncvt.rtz.xu.f.w_32", /*sew*/ 32, instruction_, /*delta_position*/ 32,
      [this](float vs2) -> std::tuple<uint16_t, uint32_t> {
        return ConvertHelper<float, uint16_t>(vs2, this->rv_fp_);
      });
  ResetInstruction();
  SetSemanticFunction(&Vfncvtrtzxufw);
  rv_fp_->SetRoundingMode(FPRoundingMode::kRoundTowardsZero);
  UnaryOpWithFflagsFPTestHelperV<uint32_t, double>(
      "Vfncvt.rtz.xu.f.w_64", /*sew*/ 64, instruction_, /*delta_position*/ 64,
      [this](double vs2) -> std::tuple<uint32_t, uint32_t> {
        return ConvertHelper<double, uint32_t>(vs2, this->rv_fp_);
      });
}

// Test narrowing convert fp to signed integer with rounding.
TEST_F(RiscVFPUnaryInstructionsTest, Vfncvtxfw) {
  SetSemanticFunction(&Vfncvtxfw);
  UnaryOpWithFflagsFPTestHelperV<int16_t, float>(
      "Vfncvt.x.f.w_32", /*sew*/ 32, instruction_, /*delta_position*/ 32,
      [this](float vs2) -> std::tuple<int16_t, uint32_t> {
        return ConvertHelper<float, int16_t>(vs2, this->rv_fp_);
      });
  ResetInstruction();
  SetSemanticFunction(&Vfncvtxfw);
  UnaryOpWithFflagsFPTestHelperV<int32_t, double>(
      "Vfncvt.x.f.w_64", /*sew*/ 64, instruction_, /*delta_position*/ 32,
      [this](double vs2) -> std::tuple<int32_t, uint32_t> {
        return ConvertHelper<double, int32_t>(vs2, this->rv_fp_);
      });
}

// Test narrowing convert fp to unsigned integer with rounding.
TEST_F(RiscVFPUnaryInstructionsTest, Vfncvtxufw) {
  SetSemanticFunction(&Vfncvtxufw);
  UnaryOpWithFflagsFPTestHelperV<uint16_t, float>(
      "Vfncvt.xu.f.w_32", /*sew*/ 32, instruction_, /*delta_position*/ 32,
      [this](float vs2) -> std::tuple<uint16_t, uint32_t> {
        return ConvertHelper<float, uint16_t>(vs2, this->rv_fp_);
      });
  ResetInstruction();
  SetSemanticFunction(&Vfncvtxufw);
  UnaryOpWithFflagsFPTestHelperV<uint32_t, double>(
      "Vfncvt.xu.f.w_64", /*sew*/ 64, instruction_, /*delta_position*/ 64,
      [this](double vs2) -> std::tuple<uint32_t, uint32_t> {
        return ConvertHelper<double, uint32_t>(vs2, this->rv_fp_);
      });
}

// Helper function for testing approximate reciprocal instruction.
template <typename T>
inline T Vrecip7vTestHelper(T vs2, RiscVFPState *rv_fp) {
  using UInt = typename FPTypeInfo<T>::IntType;
  if (FPTypeInfo<T>::IsNaN(vs2)) {
    auto nan_value = FPTypeInfo<T>::kCanonicalNaN;
    return *reinterpret_cast<T *>(&nan_value);
  }
  if (std::isinf(vs2)) {
    return std::signbit(vs2) ? -0.0 : 0.0;
  }
  if (vs2 == 0.0) {
    UInt value =
        std::signbit(vs2) ? FPTypeInfo<T>::kNegInf : FPTypeInfo<T>::kPosInf;
    return *reinterpret_cast<T *>(&value);
  }
  UInt uint_vs2 = *reinterpret_cast<UInt *>(&vs2);
  auto exp = (uint_vs2 & FPTypeInfo<T>::kExpMask) >> FPTypeInfo<T>::kSigSize;
  auto sig2 =
      (uint_vs2 & FPTypeInfo<T>::kSigMask) >> (FPTypeInfo<T>::kSigSize - 2);
  auto rm = rv_fp->GetRoundingMode();
  if ((exp == 0) && (sig2 == 0)) {  // Denormal number.
    if (std::signbit(vs2)) {
      if ((rm == FPRoundingMode::kRoundTowardsZero) ||
          (rm == FPRoundingMode::kRoundUp)) {
        return std::numeric_limits<T>::lowest();
      } else {
        UInt value = FPTypeInfo<T>::kNegInf;
        return *reinterpret_cast<T *>(&value);
      }
    } else {
      if ((rm == FPRoundingMode::kRoundTowardsZero) ||
          (rm == FPRoundingMode::kRoundDown)) {
        return std::numeric_limits<T>::max();
      } else {
        UInt value = FPTypeInfo<T>::kPosInf;
        return *reinterpret_cast<T *>(&value);
      }
    }
  }
  ScopedFPStatus status(rv_fp->host_fp_interface(),
                        FPRoundingMode::kRoundTowardsZero);
  T value = 1.0 / vs2;
  UInt uint_val = *reinterpret_cast<UInt *>(&value);
  UInt mask = FPTypeInfo<T>::kSigMask >> 7;
  uint_val = uint_val & ~mask;
  return *reinterpret_cast<T *>(&uint_val);
}

// Test approximate reciprocal instruction.
TEST_F(RiscVFPUnaryInstructionsTest, Vfrec7v) {
  SetSemanticFunction(&Vfrec7v);
  UnaryOpFPTestHelperV<float, float>(
      "Vfrec7.v_32", /*sew*/ 32, instruction_, /*delta_position*/ 7,
      [this](float vs2) -> float {
        return Vrecip7vTestHelper(vs2, this->rv_fp_);
      });
}

// Helper function for testing approximate reciprocal square root instruction.
template <typename T>
inline T Vfrsqrt7vTestHelper(T vs2, RiscVFPState *rv_fp) {
  using UInt = typename FPTypeInfo<T>::IntType;
  if (FPTypeInfo<T>::IsNaN(vs2) || (vs2 < 0.0)) {
    auto nan_value = FPTypeInfo<T>::kCanonicalNaN;
    return *reinterpret_cast<T *>(&nan_value);
  }
  if (vs2 == 0.0) {
    UInt value =
        std::signbit(vs2) ? FPTypeInfo<T>::kNegInf : FPTypeInfo<T>::kPosInf;
    return *reinterpret_cast<T *>(&value);
  }
  if (std::isinf(vs2)) {
    return 0.0;
  }
  ScopedFPStatus status(rv_fp->host_fp_interface(),
                        FPRoundingMode::kRoundTowardsZero);
  T value = 1.0 / sqrt(vs2);
  UInt uint_val = *reinterpret_cast<UInt *>(&value);
  UInt mask = FPTypeInfo<T>::kSigMask >> 7;
  uint_val = uint_val & ~mask;
  return *reinterpret_cast<T *>(&uint_val);
}

// Test approximate reciprocal square root.
TEST_F(RiscVFPUnaryInstructionsTest, Vfrsqrt7v) {
  SetSemanticFunction(&Vfrsqrt7v);
  UnaryOpFPTestHelperV<float, float>(
      "Vfrsqrt7.v_32", /*sew*/ 32, instruction_, /*delta_position*/ 7,
      [this](float vs2) -> float {
        return Vfrsqrt7vTestHelper(vs2, this->rv_fp_);
      });
  ResetInstruction();
  SetSemanticFunction(&Vfrsqrt7v);
  UnaryOpFPTestHelperV<double, double>(
      "Vfrsqrt7.v_64", /*sew*/ 64, instruction_, /*delta_position*/ 7,
      [this](double vs2) -> double {
        return Vfrsqrt7vTestHelper(vs2, this->rv_fp_);
      });
}

// Helper function for testing square root instruction.
template <typename T>
inline std::tuple<T, uint32_t> VfsqrtvTestHelper(T vs2) {
  T res = sqrt(vs2);
  if (std::isnan(res)) {
    return std::make_tuple(
        *reinterpret_cast<const float *>(&FPTypeInfo<float>::kCanonicalNaN),
        (uint32_t)FPExceptions::kInvalidOp);
  }

  return std::make_tuple(res, 0);
}

// Test square root instruction.
TEST_F(RiscVFPUnaryInstructionsTest, Vfsqrtv) {
  SetSemanticFunction(&Vfsqrtv);
  UnaryOpWithFflagsFPTestHelperV<float, float>(
      "Vfsqrt.v_32", /*sew*/ 32, instruction_,
      /*delta_position*/ 32, [](float vs2) -> std::tuple<float, uint32_t> {
        return VfsqrtvTestHelper(vs2);
      });
  ResetInstruction();
  SetSemanticFunction(&Vfsqrtv);
  UnaryOpWithFflagsFPTestHelperV<double, double>(
      "Vfsqrt.v_64", /*sew*/ 64, instruction_, /*delta_position*/ 64,
      [](double vs2) -> std::tuple<double, uint32_t> {
        return VfsqrtvTestHelper(vs2);
      });
}

// Test widening convert fp to fp.
TEST_F(RiscVFPUnaryInstructionsTest, Vfwcvtffv) {
  SetSemanticFunction(&Vfwcvtffv);
  UnaryOpTestHelperV<double, float>(
      "Vfwcvt.f.f.v_32", /*sew*/ 32, instruction_,
      [](float vs2) -> double { return static_cast<double>(vs2); });
}

// Test widening convert unsigned integer to fp.
TEST_F(RiscVFPUnaryInstructionsTest, Vfwcvtfxuv) {
  SetSemanticFunction(&Vfwcvtfxuv);
  UnaryOpTestHelperV<float, uint16_t>(
      "Vfwcvt.f.xu.v_16", /*sew*/ 16, instruction_,
      [](uint16_t vs2) -> float { return static_cast<float>(vs2); });
  ResetInstruction();
  SetSemanticFunction(&Vfwcvtfxuv);
  UnaryOpTestHelperV<double, uint32_t>(
      "Vfwcvt.f.xu.v_32", /*sew*/ 32, instruction_,
      [](uint32_t vs2) -> double { return static_cast<double>(vs2); });
}

// Test widening convert signed integer to fp.
TEST_F(RiscVFPUnaryInstructionsTest, Vfwcvtfxv) {
  SetSemanticFunction(&Vfwcvtfxv);
  UnaryOpTestHelperV<float, int16_t>(
      "Vfwcvt.f.x.v_16", /*sew*/ 16, instruction_,
      [](int16_t vs2) -> float { return static_cast<float>(vs2); });
  ResetInstruction();
  SetSemanticFunction(&Vfwcvtfxv);
  UnaryOpTestHelperV<double, int32_t>(
      "Vfwcvt.f.x.v_32", /*sew*/ 32, instruction_,
      [](int32_t vs2) -> double { return static_cast<double>(vs2); });
}

// Test widening convert fp to signed integer with truncation (round to zero).
TEST_F(RiscVFPUnaryInstructionsTest, Vfwcvtrtzxfv) {
  SetSemanticFunction(&Vfwcvtrtzxfv);
  rv_fp_->SetRoundingMode(FPRoundingMode::kRoundTowardsZero);
  UnaryOpWithFflagsFPTestHelperV<int64_t, float>(
      "Vfwcvt.rtz.xu.f.v_32", /*sew*/ 32, instruction_, /*delta_position*/ 64,
      [this](float vs2) -> std::tuple<int64_t, uint32_t> {
        return ConvertHelper<float, int64_t>(vs2, this->rv_fp_);
      });
}

// Test widening convert fp to unsigned integer with truncation (round to zero).
TEST_F(RiscVFPUnaryInstructionsTest, Vfwcvtrtzxufv) {
  SetSemanticFunction(&Vfwcvtrtzxufv);
  rv_fp_->SetRoundingMode(FPRoundingMode::kRoundTowardsZero);
  UnaryOpWithFflagsFPTestHelperV<uint64_t, float>(
      "Vfwcvt.rtz.xu.f.v_32", /*sew*/ 32, instruction_, /*delta_position*/ 64,
      [this](float vs2) -> std::tuple<uint64_t, uint32_t> {
        return ConvertHelper<float, uint64_t>(vs2, this->rv_fp_);
      });
}

// Test widening convert fp to signed integer with rounding.
TEST_F(RiscVFPUnaryInstructionsTest, Vfwcvtxfv) {
  SetSemanticFunction(&Vfwcvtxfv);
  UnaryOpWithFflagsFPTestHelperV<int64_t, float>(
      "Vfwcvt.rtz.xu.f.v_32", /*sew*/ 32, instruction_, /*delta_position*/ 64,
      [this](float vs2) -> std::tuple<int64_t, uint32_t> {
        return ConvertHelper<float, int64_t>(vs2, this->rv_fp_);
      });
}

// Test widening convert fp to unsigned integer with rounding.
TEST_F(RiscVFPUnaryInstructionsTest, Vfwcvtxufv) {
  SetSemanticFunction(&Vfwcvtxufv);
  UnaryOpWithFflagsFPTestHelperV<uint64_t, float>(
      "Vfwcvt.rtz.xu.f.v_32", /*sew*/ 32, instruction_, /*delta_position*/ 64,
      [this](float vs2) -> std::tuple<uint64_t, uint32_t> {
        return ConvertHelper<float, uint64_t>(vs2, this->rv_fp_);
      });
}

}  // namespace
