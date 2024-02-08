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

#ifndef MPACT_RISCV_RISCV_TEST_RISCV_VECTOR_FP_TEST_UTILITIES_H_
#define MPACT_RISCV_RISCV_TEST_RISCV_VECTOR_FP_TEST_UTILITIES_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <tuple>
#include <type_traits>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "riscv/riscv_fp_host.h"
#include "riscv/riscv_fp_info.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_register.h"
#include "riscv/test/riscv_vector_instructions_test_base.h"
#include "mpact/sim/generic/type_helpers.h"

namespace mpact {
namespace sim {
namespace riscv {
namespace test {

using ::mpact::sim::generic::operator*;

constexpr char kFs1Name[] = "f4";
constexpr int kFs1 = 4;

// Templated helper structs to provide information about floating point types.
template <typename T>
struct FPTypeInfo {
  using IntType = typename std::make_unsigned<T>::type;
  static const int kBitSize = 8 * sizeof(T);
  static const int kExpSize = 0;
  static const int kSigSize = 0;
  static bool IsNaN(T value) { return false; }
};

template <>
struct FPTypeInfo<float> {
  using T = float;
  using IntType = uint32_t;
  static const int kBitSize = sizeof(float) << 3;
  static const int kExpSize = 8;
  static const int kSigSize = kBitSize - kExpSize - 1;
  static const IntType kExpMask = ((1ULL << kExpSize) - 1) << kSigSize;
  static const IntType kSigMask = (1ULL << kSigSize) - 1;
  static const IntType kQNaN = kExpMask | (1ULL << (kSigSize - 1)) | 1;
  static const IntType kSNaN = kExpMask | 1;
  static const IntType kPosInf = kExpMask;
  static const IntType kNegInf = kExpMask | (1ULL << (kBitSize - 1));
  static const IntType kPosZero = 0;
  static const IntType kNegZero = 1ULL << (kBitSize - 1);
  static const IntType kPosDenorm = 1ULL << (kSigSize - 2);
  static const IntType kNegDenorm =
      (1ULL << (kBitSize - 1)) | (1ULL << (kSigSize - 2));
  static const IntType kCanonicalNaN = 0x7fc0'0000ULL;
  static bool IsNaN(T value) { return std::isnan(value); }
};

template <>
struct FPTypeInfo<double> {
  using T = double;
  using IntType = uint64_t;
  static const int kBitSize = sizeof(double) << 3;
  static const int kExpSize = 11;
  static const int kSigSize = kBitSize - kExpSize - 1;
  static const IntType kExpMask = ((1ULL << kExpSize) - 1) << kSigSize;
  static const IntType kSigMask = (1ULL << kSigSize) - 1;
  static const IntType kQNaN = kExpMask | (1ULL << (kSigSize - 1)) | 1;
  static const IntType kSNaN = kExpMask | 1;
  static const IntType kPosInf = kExpMask;
  static const IntType kNegInf = kExpMask | (1ULL << (kBitSize - 1));
  static const IntType kPosZero = 0;
  static const IntType kNegZero = 1ULL << (kBitSize - 1);
  static const IntType kPosDenorm = 1ULL << (kSigSize - 2);
  static const IntType kNegDenorm =
      (1ULL << (kBitSize - 1)) | (1ULL << (kSigSize - 2));
  static const IntType kCanonicalNaN = 0x7ff8'0000'0000'0000ULL;
  static bool IsNaN(T value) { return std::isnan(value); }
};

// These templated functions allow for comparison of values with a tolerance
// given for floating point types. The tolerance is stated as the bit position
// in the mantissa of the op, with 0 being the msb of the mantissa. If the
// bit position is beyond the mantissa, a comparison of equal is performed.
template <typename T>
inline void FPCompare(T op, T reg, int, absl::string_view str) {
  EXPECT_EQ(reg, op) << str;
}

template <>
inline void FPCompare<float>(float op, float reg, int delta_position,
                             absl::string_view str) {
  using T = float;
  using UInt = typename FPTypeInfo<T>::IntType;
  if (!std::isnan(op) && !std::isinf(op) &&
      delta_position < FPTypeInfo<T>::kSigSize) {
    T delta;
    UInt exp = FPTypeInfo<T>::kExpMask >> FPTypeInfo<T>::kSigSize;
    if (exp > delta_position) {
      exp -= delta_position;
      UInt udelta = exp << FPTypeInfo<T>::kSigSize;
      delta = *reinterpret_cast<T *>(&udelta);
    } else {
      // Becomes a denormal
      int diff = delta_position - exp;
      UInt udelta = 1ULL << (FPTypeInfo<T>::kSigSize - 1 - diff);
      delta = *reinterpret_cast<T *>(&udelta);
    }
    EXPECT_THAT(reg, testing::NanSensitiveFloatNear(op, delta)) << str;
  } else {
    EXPECT_THAT(reg, testing::NanSensitiveFloatEq(op)) << str;
  }
}

template <>
inline void FPCompare<double>(double op, double reg, int delta_position,
                              absl::string_view str) {
  using T = double;
  using UInt = typename FPTypeInfo<T>::IntType;
  if (!std::isnan(op) && !std::isinf(op) &&
      delta_position < FPTypeInfo<T>::kSigSize) {
    T delta;
    UInt exp = FPTypeInfo<T>::kExpMask >> FPTypeInfo<T>::kSigSize;
    if (exp > delta_position) {
      exp -= delta_position;
      UInt udelta = exp << FPTypeInfo<T>::kSigSize;
      delta = *reinterpret_cast<T *>(&udelta);
    } else {
      // Becomes a denormal
      int diff = delta_position - exp;
      UInt udelta = 1ULL << (FPTypeInfo<T>::kSigSize - 1 - diff);
      delta = *reinterpret_cast<T *>(&udelta);
    }
    EXPECT_THAT(reg, testing::NanSensitiveDoubleNear(op, delta)) << str;
  } else {
    EXPECT_THAT(reg, testing::NanSensitiveDoubleEq(op)) << str;
  }
}

template <typename FP>
FP OptimizationBarrier(FP op) {
  asm volatile("" : "+X"(op));
  return op;
}

namespace internal {

// These are predicates used in the following NaNBox function definitions, as
// part of the enable_if construct.
template <typename S, typename D>
struct EqualSize {
  static const bool value = sizeof(S) == sizeof(D) &&
                            std::is_floating_point<S>::value &&
                            std::is_integral<D>::value;
};

template <typename S, typename D>
struct GreaterSize {
  static const bool value =
      sizeof(S) > sizeof(D) &&
      std::is_floating_point<S>::value &&std::is_integral<D>::value;
};

template <typename S, typename D>
struct LessSize {
  static const bool value = sizeof(S) < sizeof(D) &&
                            std::is_floating_point<S>::value &&
                            std::is_integral<D>::value;
};

}  // namespace internal

// Template functions to NaN box a floating point value when being assigned
// to a wider register. The first version places a smaller floating point value
// in a NaN box (all upper bits in the word are set to 1).

// Enable_if is used to select the proper implementation for different S and D
// type combinations. It uses the SFINAE (substitution failure is not an error)
// "feature" of C++ to hide the implementation that don't match the predicate
// from being resolved.

template <typename S, typename D>
inline typename std::enable_if<internal::LessSize<S, D>::value, D>::type NaNBox(
    S value) {
  using SInt = typename FPTypeInfo<S>::IntType;
  SInt sval = *reinterpret_cast<SInt *>(&value);
  D dval = (~static_cast<D>(0) << (sizeof(S) * 8)) | sval;
  return *reinterpret_cast<D *>(&dval);
}

// This version does a straight copy - as the data types are the same size.
template <typename S, typename D>
inline typename std::enable_if<internal::EqualSize<S, D>::value, D>::type
NaNBox(S value) {
  return *reinterpret_cast<D *>(&value);
}

// Signal error if the register is smaller than the floating point value.
template <typename S, typename D>
inline typename std::enable_if<internal::GreaterSize<S, D>::value, D>::type
NaNBox(S value) {
  // No return statement, so error will be reported.
}

// Test fixture for binary fp instructions.
class RiscVFPInstructionsTestBase
    : public ::mpact::sim::riscv::test::RiscVVectorInstructionsTestBase {
 public:
  RiscVFPInstructionsTestBase() {
    rv_fp_ = new mpact::sim::riscv::RiscVFPState(state_);
    state_->set_rv_fp(rv_fp_);
  }
  ~RiscVFPInstructionsTestBase() override {
    state_->set_rv_fp(nullptr);
    delete rv_fp_;
  }

  // Construct a random FP value by separately generating integer values for
  // sign, exponent and mantissa.
  template <typename T>
  T RandomFPValue() {
    using UInt = typename FPTypeInfo<T>::IntType;
    UInt sign = absl::Uniform(absl::IntervalClosed, bitgen_, 0ULL, 1ULL);
    UInt exp = absl::Uniform(absl::IntervalClosedOpen, bitgen_, 0ULL,
                             1ULL << FPTypeInfo<T>::kExpSize);
    UInt sig = absl::Uniform(absl::IntervalClosedOpen, bitgen_, 0ULL,
                             1ULL << FPTypeInfo<T>::kSigSize);
    UInt value = (sign & 1) << (FPTypeInfo<T>::kBitSize - 1) |
                 (exp << FPTypeInfo<T>::kSigSize) | sig;
    T val = *reinterpret_cast<T *>(&value);
    return val;
  }

  // This method uses random values for each field in the fp number.
  template <typename T>
  void FillArrayWithRandomFPValues(absl::Span<T> span) {
    for (auto &val : span) {
      val = RandomFPValue<T>();
    }
  }

  template <typename Vs2, typename Vs1>
  void InitializeInputs(absl::Span<Vs2> vs2_span, absl::Span<Vs1> vs1_span,
                        absl::Span<uint8_t> mask_span, int count) {
    // Initialize input values.
    FillArrayWithRandomFPValues<Vs2>(vs2_span);
    FillArrayWithRandomFPValues<Vs1>(vs1_span);
    using Vs2Int = typename FPTypeInfo<Vs2>::IntType;
    using Vs1Int = typename FPTypeInfo<Vs1>::IntType;
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
    if (count == 4) {
      *reinterpret_cast<Vs1Int *>(&vs1_span[0]) = FPTypeInfo<Vs1>::kQNaN;
      *reinterpret_cast<Vs1Int *>(&vs1_span[1]) = FPTypeInfo<Vs1>::kSNaN;
      *reinterpret_cast<Vs1Int *>(&vs1_span[2]) = FPTypeInfo<Vs1>::kPosInf;
      *reinterpret_cast<Vs1Int *>(&vs1_span[3]) = FPTypeInfo<Vs1>::kNegInf;
      *reinterpret_cast<Vs1Int *>(&vs1_span[4]) = FPTypeInfo<Vs1>::kPosZero;
      *reinterpret_cast<Vs1Int *>(&vs1_span[5]) = FPTypeInfo<Vs1>::kNegZero;
      *reinterpret_cast<Vs1Int *>(&vs1_span[6]) = FPTypeInfo<Vs1>::kPosDenorm;
      *reinterpret_cast<Vs1Int *>(&vs1_span[7]) = FPTypeInfo<Vs1>::kNegDenorm;
    } else if (count == 5) {
      *reinterpret_cast<Vs1Int *>(&vs1_span[7]) = FPTypeInfo<Vs1>::kQNaN;
      *reinterpret_cast<Vs1Int *>(&vs1_span[6]) = FPTypeInfo<Vs1>::kSNaN;
      *reinterpret_cast<Vs1Int *>(&vs1_span[5]) = FPTypeInfo<Vs1>::kPosInf;
      *reinterpret_cast<Vs1Int *>(&vs1_span[4]) = FPTypeInfo<Vs1>::kNegInf;
      *reinterpret_cast<Vs1Int *>(&vs1_span[3]) = FPTypeInfo<Vs1>::kPosZero;
      *reinterpret_cast<Vs1Int *>(&vs1_span[2]) = FPTypeInfo<Vs1>::kNegZero;
      *reinterpret_cast<Vs1Int *>(&vs1_span[1]) = FPTypeInfo<Vs1>::kPosDenorm;
      *reinterpret_cast<Vs1Int *>(&vs1_span[0]) = FPTypeInfo<Vs1>::kNegDenorm;
    } else if (count == 6) {
      *reinterpret_cast<Vs1Int *>(&vs1_span[0]) = FPTypeInfo<Vs1>::kQNaN;
      *reinterpret_cast<Vs1Int *>(&vs1_span[1]) = FPTypeInfo<Vs1>::kSNaN;
      *reinterpret_cast<Vs1Int *>(&vs1_span[2]) = FPTypeInfo<Vs1>::kNegInf;
      *reinterpret_cast<Vs1Int *>(&vs1_span[3]) = FPTypeInfo<Vs1>::kPosInf;
      *reinterpret_cast<Vs1Int *>(&vs1_span[4]) = FPTypeInfo<Vs1>::kNegZero;
      *reinterpret_cast<Vs1Int *>(&vs1_span[5]) = FPTypeInfo<Vs1>::kPosZero;
      *reinterpret_cast<Vs1Int *>(&vs1_span[6]) = FPTypeInfo<Vs1>::kNegDenorm;
      *reinterpret_cast<Vs1Int *>(&vs1_span[7]) = FPTypeInfo<Vs1>::kPosDenorm;
    }
    // Modify the first mask bits to use each of the special floating
    // point values.
    mask_span[0] = 0xff;
  }

  // Floating point test needs to ensure to use the fp special values (inf,
  // NaN etc.) during testing, not just random values.
  template <typename Vd, typename Vs2, typename Vs1>
  void BinaryOpFPTestHelperVV(absl::string_view name, int sew,
                              Instruction *inst, int delta_position,
                              std::function<Vd(Vs2, Vs1)> operation) {
    int byte_sew = sew / 8;
    if (byte_sew != sizeof(Vd) && byte_sew != sizeof(Vs2) &&
        byte_sew != sizeof(Vs1)) {
      FAIL() << name << ": selected element width != any operand types"
             << " sew: " << sew << " Vd: " << sizeof(Vd)
             << " Vs2: " << sizeof(Vs2) << " Vs1: " << sizeof(Vs1);
      return;
    }
    // Number of elements per vector register.
    constexpr int vs2_size = kVectorLengthInBytes / sizeof(Vs2);
    constexpr int vs1_size = kVectorLengthInBytes / sizeof(Vs1);
    // Input values for 8 registers.
    Vs2 vs2_value[vs2_size * 8];
    Vs1 vs1_value[vs1_size * 8];
    auto vs2_span = Span<Vs2>(vs2_value);
    auto vs1_span = Span<Vs1>(vs1_value);
    AppendVectorRegisterOperands({kVs2, kVs1, kVmask}, {kVd});
    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});
    // Iterate across different lmul values.
    for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
      InitializeInputs<Vs2, Vs1>(vs2_span, vs1_span,
                                 vreg_[kVmask]->data_buffer()->Get<uint8_t>(),
                                 lmul_index);
      // Set values for all 8 vector registers in the vector register group.
      for (int i = 0; i < 8; i++) {
        auto vs2_name = absl::StrCat("v", kVs2 + i);
        auto vs1_name = absl::StrCat("v", kVs1 + i);
        SetVectorRegisterValues<Vs2>(
            {{vs2_name, vs2_span.subspan(vs2_size * i, vs2_size)}});
        SetVectorRegisterValues<Vs1>(
            {{vs1_name, vs1_span.subspan(vs1_size * i, vs1_size)}});
      }
      int lmul8 = kLmul8Values[lmul_index];
      int lmul8_vd = lmul8 * sizeof(Vd) / byte_sew;
      int lmul8_vs2 = lmul8 * sizeof(Vs2) / byte_sew;
      int lmul8_vs1 = lmul8 * sizeof(Vs1) / byte_sew;
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

            if (lmul8_vs1 < 1 || lmul8_vs1 > 64) {
              EXPECT_TRUE(rv_vector_->vector_exception())
                  << "lmul8: vs1: " << lmul8_vs1;
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
                // The first 8 bits of the mask are set to true above, so
                // only read the mask value after the first byte.
                if (mask_index > 0) {
                  mask_value =
                      ((kA5Mask[mask_index] >> mask_offset) & 0b1) != 0;
                }
                auto reg_val = vreg_[reg]->data_buffer()->Get<Vd>(i);
                auto int_reg_val =
                    *reinterpret_cast<typename FPTypeInfo<Vd>::IntType *>(
                        &reg_val);
                if ((count >= vstart) && mask_value && (count < num_values)) {
                  ScopedFPStatus set_fpstatus(rv_fp_->host_fp_interface());
                  auto op_val = operation(vs2_value[count], vs1_value[count]);
                  auto int_op_val =
                      *reinterpret_cast<typename FPTypeInfo<Vd>::IntType *>(
                          &op_val);
                  auto int_vs2_val =
                      *reinterpret_cast<typename FPTypeInfo<Vs2>::IntType *>(
                          &vs2_value[count]);
                  auto int_vs1_val =
                      *reinterpret_cast<typename FPTypeInfo<Vs1>::IntType *>(
                          &vs1_value[count]);
                  FPCompare<Vd>(
                      op_val, reg_val, delta_position,
                      absl::StrCat(name, "[", count, "] op(", vs2_value[count],
                                   "[0x", absl::Hex(int_vs2_val), "], ",
                                   vs1_value[count], "[0x",
                                   absl::Hex(int_vs1_val),
                                   "]) = ", absl::Hex(int_op_val), " != reg[",
                                   reg, "][", i, "]  (", reg_val, " [0x",
                                   absl::Hex(int_reg_val), "]) lmul8(", lmul8,
                                   ") rm = ", *(rv_fp_->GetRoundingMode())));
                } else {
                  EXPECT_EQ(0, reg_val) << absl::StrCat(
                      name, "  0 != reg[", reg, "][", i, "]  (", reg_val,
                      " [0x", absl::Hex(int_reg_val), "]) lmul8(", lmul8, ")");
                }
                count++;
              }
              if (HasFailure()) return;
            }
          }
          vlen = absl::Uniform(absl::IntervalOpenClosed, bitgen_, vstart,
                               num_reg_values);
        }
        vstart = absl::Uniform(absl::IntervalOpen, bitgen_, 0, num_reg_values);
      }
    }
  }

  // Floating point test needs to ensure to use the fp special values (inf,
  // NaN etc.) during testing, not just random values.
  template <typename Vd, typename Vs2, typename Vs1>
  void BinaryOpWithFflagsFPTestHelperVV(
      absl::string_view name, int sew, Instruction *inst, int delta_position,
      std::function<std::tuple<Vd, uint32_t>(Vs2, Vs1)> operation) {
    using VdInt = typename FPTypeInfo<Vd>::IntType;
    using Vs2Int = typename FPTypeInfo<Vs2>::IntType;
    using Vs1Int = typename FPTypeInfo<Vs1>::IntType;
    int byte_sew = sew / 8;
    if (byte_sew != sizeof(Vd) && byte_sew != sizeof(Vs2) &&
        byte_sew != sizeof(Vs1)) {
      FAIL() << name << ": selected element width != any operand types"
             << " sew: " << sew << " Vd: " << sizeof(Vd)
             << " Vs2: " << sizeof(Vs2) << " Vs1: " << sizeof(Vs1);
      return;
    }
    // Number of elements per vector register.
    constexpr int vs2_size = kVectorLengthInBytes / sizeof(Vs2);
    constexpr int vs1_size = kVectorLengthInBytes / sizeof(Vs1);
    // Input values for 8 registers.
    Vs2 vs2_value[vs2_size * 8];
    Vs1 vs1_value[vs1_size * 8];
    auto vs2_span = Span<Vs2>(vs2_value);
    auto vs1_span = Span<Vs1>(vs1_value);
    AppendVectorRegisterOperands({kVs2, kVs1, kVmask}, {kVd});
    auto *flag_op = rv_fp_->fflags()->CreateSetDestinationOperand(0, "fflags");
    instruction_->AppendDestination(flag_op);
    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});
    // Iterate across different lmul values.
    for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
      InitializeInputs<Vs2, Vs1>(vs2_span, vs1_span,
                                 vreg_[kVmask]->data_buffer()->Get<uint8_t>(),
                                 lmul_index);

      // Set values for all 8 vector registers in the vector register group.
      for (int i = 0; i < 8; i++) {
        auto vs2_name = absl::StrCat("v", kVs2 + i);
        auto vs1_name = absl::StrCat("v", kVs1 + i);
        SetVectorRegisterValues<Vs2>(
            {{vs2_name, vs2_span.subspan(vs2_size * i, vs2_size)}});
        SetVectorRegisterValues<Vs1>(
            {{vs1_name, vs1_span.subspan(vs1_size * i, vs1_size)}});
      }
      int lmul8 = kLmul8Values[lmul_index];
      int lmul8_vd = lmul8 * sizeof(Vd) / byte_sew;
      int lmul8_vs2 = lmul8 * sizeof(Vs2) / byte_sew;
      int lmul8_vs1 = lmul8 * sizeof(Vs1) / byte_sew;
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
            rv_fp_->fflags()->Write(static_cast<uint32_t>(0));

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
            EXPECT_EQ(rv_vector_->vstart(), 0);
            int count = 0;
            uint32_t fflags_test = 0;
            for (int reg = kVd; reg < kVd + 8; reg++) {
              for (int i = 0; i < kVectorLengthInBytes / sizeof(Vd); i++) {
                int mask_index = count >> 3;
                int mask_offset = count & 0b111;
                bool mask_value = true;
                // The first 8 bits of the mask are set to true above, so
                // only read the mask value after the first byte.
                if (mask_index > 0) {
                  mask_value =
                      ((kA5Mask[mask_index] >> mask_offset) & 0b1) != 0;
                }
                auto reg_val = vreg_[reg]->data_buffer()->Get<Vd>(i);
                auto int_reg_val = *reinterpret_cast<VdInt *>(&reg_val);
                if ((count >= vstart) && mask_value && (count < num_values)) {
                  Vd op_val;
                  uint32_t flag;
                  {
                    ScopedFPStatus set_fpstatus(rv_fp_->host_fp_interface());
                    auto [op_val_tmp, flag_tmp] =
                        operation(vs2_value[count], vs1_value[count]);
                    op_val = op_val_tmp;
                    flag = flag_tmp;
                  }
                  auto int_op_val = *reinterpret_cast<VdInt *>(&op_val);
                  auto int_vs2_val =
                      *reinterpret_cast<Vs2Int *>(&vs2_value[count]);
                  auto int_vs1_val =
                      *reinterpret_cast<Vs1Int *>(&vs1_value[count]);
                  FPCompare<Vd>(
                      op_val, reg_val, delta_position,
                      absl::StrCat(name, "[", count, "] op(", vs2_value[count],
                                   "[0x", absl::Hex(int_vs2_val), "], ",
                                   vs1_value[count], "[0x",
                                   absl::Hex(int_vs1_val),
                                   "]) = ", absl::Hex(int_op_val), " != reg[",
                                   reg, "][", i, "]  (", reg_val, " [0x",
                                   absl::Hex(int_reg_val), "]) lmul8(", lmul8,
                                   ") rm = ", *(rv_fp_->GetRoundingMode())));
                  fflags_test |= flag;
                } else {
                  EXPECT_EQ(0, reg_val) << absl::StrCat(
                      name, "  0 != reg[", reg, "][", i, "]  (", reg_val,
                      " [0x", absl::Hex(int_reg_val), "]) lmul8(", lmul8, ")");
                }
                count++;
              }
            }
            uint32_t fflags = rv_fp_->fflags()->AsUint32();
            EXPECT_EQ(fflags, fflags_test) << name;
            if (HasFailure()) return;
          }
          vlen = absl::Uniform(absl::IntervalOpenClosed, bitgen_, vstart,
                               num_reg_values);
        }
        vstart = absl::Uniform(absl::IntervalOpen, bitgen_, 0, num_reg_values);
      }
    }
  }

  // Floating point test needs to ensure to use the fp special values (inf,
  // NaN etc.) during testing, not just random values.
  template <typename Vd, typename Vs2, typename Fs1>
  void BinaryOpWithFflagsFPTestHelperVX(
      absl::string_view name, int sew, Instruction *inst, int delta_position,
      std::function<std::tuple<Vd, uint32_t>(Vs2, Fs1)> operation) {
    using VdInt = typename FPTypeInfo<Vd>::IntType;
    using Vs2Int = typename FPTypeInfo<Vs2>::IntType;
    using Fs1Int = typename FPTypeInfo<Fs1>::IntType;
    int byte_sew = sew / 8;
    if (byte_sew != sizeof(Vd) && byte_sew != sizeof(Vs2) &&
        byte_sew != sizeof(Fs1)) {
      FAIL() << name << ": selected element width != any operand types"
             << " sew: " << sew << " Vd: " << sizeof(Vd)
             << " Vs2: " << sizeof(Vs2) << " Fs1: " << sizeof(Fs1);
      return;
    }
    // Number of elements per vector register.
    constexpr int vs2_size = kVectorLengthInBytes / sizeof(Vs2);
    // Input values for 8 registers.
    Vs2 vs2_value[vs2_size * 8];
    auto vs2_span = Span<Vs2>(vs2_value);
    AppendVectorRegisterOperands({kVs2}, {kVd});
    AppendRegisterOperands({kFs1Name}, {});
    auto *flag_op = rv_fp_->fflags()->CreateSetDestinationOperand(0, "fflags");
    instruction_->AppendDestination(flag_op);
    AppendVectorRegisterOperands({kVmask}, {});
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
      // Modify the first mask bits to use each of the special floating
      // point values.
      vreg_[kVmask]->data_buffer()->Set<uint8_t>(0, 0xff);

      // Set values for all 8 vector registers in the vector register group.
      for (int i = 0; i < 8; i++) {
        auto vs2_name = absl::StrCat("v", kVs2 + i);
        SetVectorRegisterValues<Vs2>(
            {{vs2_name, vs2_span.subspan(vs2_size * i, vs2_size)}});
      }
      int lmul8 = kLmul8Values[lmul_index];
      int lmul8_vd = lmul8 * sizeof(Vd) / byte_sew;
      int lmul8_vs2 = lmul8 * sizeof(Vs2) / byte_sew;
      int lmul8_vs1 = lmul8 * sizeof(Fs1) / byte_sew;
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
          // Generate a new rs1 value.
          Fs1 fs1_value = RandomFPValue<Fs1>();
          // Need to NaN box the value, that is, if the register value type
          // is wider than the data type for a floating point value, the
          // upper bits are all set to 1's.
          typename RVFpRegister::ValueType fs1_reg_value =
              NaNBox<Fs1, typename RVFpRegister::ValueType>(fs1_value);
          SetRegisterValues<typename RVFpRegister::ValueType, RVFpRegister>(
              {{kFs1Name, fs1_reg_value}});
          // Iterate across rounding modes.
          for (int rm : {0, 1, 2, 3, 4}) {
            rv_fp_->SetRoundingMode(static_cast<FPRoundingMode>(rm));
            rv_vector_->set_vstart(vstart);
            ClearVectorRegisterGroup(kVd, 8);
            rv_fp_->fflags()->Write(static_cast<uint32_t>(0));

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
            EXPECT_EQ(rv_vector_->vstart(), 0);
            int count = 0;
            uint32_t fflags_test = 0;
            for (int reg = kVd; reg < kVd + 8; reg++) {
              for (int i = 0; i < kVectorLengthInBytes / sizeof(Vd); i++) {
                int mask_index = count >> 3;
                int mask_offset = count & 0b111;
                bool mask_value = true;
                // The first 8 bits of the mask are set to true above, so
                // only read the mask value after the first byte.
                if (mask_index > 0) {
                  mask_value =
                      ((kA5Mask[mask_index] >> mask_offset) & 0b1) != 0;
                }
                auto reg_val = vreg_[reg]->data_buffer()->Get<Vd>(i);
                auto int_reg_val = *reinterpret_cast<VdInt *>(&reg_val);
                if ((count >= vstart) && mask_value && (count < num_values)) {
                  Vd op_val;
                  uint32_t flag;
                  {
                    ScopedFPStatus set_fpstatus(rv_fp_->host_fp_interface());
                    auto [op_val_tmp, flag_tmp] =
                        operation(vs2_value[count], fs1_value);
                    op_val = op_val_tmp;
                    flag = flag_tmp;
                  }
                  auto int_op_val = *reinterpret_cast<VdInt *>(&op_val);
                  auto int_vs2_val =
                      *reinterpret_cast<Vs2Int *>(&vs2_value[count]);
                  auto int_fs1_val = *reinterpret_cast<Fs1Int *>(&fs1_value);
                  FPCompare<Vd>(
                      op_val, reg_val, delta_position,
                      absl::StrCat(name, "[", count, "] op(", vs2_value[count],
                                   "[0x", absl::Hex(int_vs2_val), "], ",
                                   fs1_value, "[0x", absl::Hex(int_fs1_val),
                                   "]) = ", absl::Hex(int_op_val), " != reg[",
                                   reg, "][", i, "]  (", reg_val, " [0x",
                                   absl::Hex(int_reg_val), "]) lmul8(", lmul8,
                                   ") rm = ", *(rv_fp_->GetRoundingMode())));
                  fflags_test |= flag;
                } else {
                  EXPECT_EQ(0, reg_val) << absl::StrCat(
                      name, "  0 != reg[", reg, "][", i, "]  (", reg_val,
                      " [0x", absl::Hex(int_reg_val), "]) lmul8(", lmul8, ")");
                }
                count++;
              }
            }
            uint32_t fflags = rv_fp_->fflags()->AsUint32();
            EXPECT_EQ(fflags, fflags_test) << name;
            if (HasFailure()) return;
          }
          vlen = absl::Uniform(absl::IntervalOpenClosed, bitgen_, vstart,
                               num_reg_values);
        }
        vstart = absl::Uniform(absl::IntervalOpen, bitgen_, 0, num_reg_values);
      }
    }
  }

  // Floating point test needs to ensure to use the fp special values (inf,
  // NaN etc.) during testing, not just random values. This function handles
  // vector scalar instructions.
  template <typename Vd, typename Vs2, typename Fs1>
  void BinaryOpFPWithMaskTestHelperVX(
      absl::string_view name, int sew, Instruction *inst, int delta_position,
      std::function<Vd(Vs2, Fs1, bool)> operation) {
    using VdInt = typename FPTypeInfo<Vd>::IntType;
    using Vs2Int = typename FPTypeInfo<Vs2>::IntType;
    using Fs1Int = typename FPTypeInfo<Fs1>::IntType;
    int byte_sew = sew / 8;
    if (byte_sew != sizeof(Vd) && byte_sew != sizeof(Vs2) &&
        byte_sew != sizeof(Fs1)) {
      FAIL() << name << ": selected element width != any operand types"
             << "sew: " << sew << " Vd: " << sizeof(Vd)
             << " Vs2: " << sizeof(Vs2) << " Fs1: " << sizeof(Fs1);
      return;
    }
    // Number of elements per vector register.
    constexpr int vs2_size = kVectorLengthInBytes / sizeof(Vs2);
    // Input values for 8 registers.
    Vs2 vs2_value[vs2_size * 8];
    auto vs2_span = Span<Vs2>(vs2_value);
    AppendVectorRegisterOperands({kVs2}, {kVd});
    AppendRegisterOperands({kFs1Name}, {});
    AppendVectorRegisterOperands({kVmask}, {});
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
      // Modify the first mask bits to use each of the special floating
      // point values.
      vreg_[kVmask]->data_buffer()->Set<uint8_t>(0, 0xff);
      // Set values for all 8 vector registers in the vector register group.
      for (int i = 0; i < 8; i++) {
        auto vs2_name = absl::StrCat("v", kVs2 + i);
        SetVectorRegisterValues<Vs2>(
            {{vs2_name, vs2_span.subspan(vs2_size * i, vs2_size)}});
      }
      int lmul8 = kLmul8Values[lmul_index];
      int lmul8_vd = lmul8 * sizeof(Vd) / byte_sew;
      int lmul8_vs2 = lmul8 * sizeof(Vs2) / byte_sew;
      int lmul8_vs1 = lmul8 * sizeof(Fs1) / byte_sew;
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
          // Generate a new rs1 value.
          Fs1 fs1_value = RandomFPValue<Fs1>();
          // Need to NaN box the value, that is, if the register value type
          // is wider than the data type for a floating point value, the
          // upper bits are all set to 1's.
          typename RVFpRegister::ValueType fs1_reg_value =
              NaNBox<Fs1, typename RVFpRegister::ValueType>(fs1_value);
          SetRegisterValues<typename RVFpRegister::ValueType, RVFpRegister>(
              {{kFs1Name, fs1_reg_value}});
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

            if (lmul8_vs1 < 1 || lmul8_vs1 > 64) {
              EXPECT_TRUE(rv_vector_->vector_exception())
                  << "lmul8: vs1: " << lmul8_vs1;
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
                // The first 8 bits of the mask are set to true above, so
                // only read the mask value after the first byte.
                if (mask_index > 0) {
                  mask_value =
                      ((kA5Mask[mask_index] >> mask_offset) & 0b1) != 0;
                }
                auto reg_val = vreg_[reg]->data_buffer()->Get<Vd>(i);
                auto int_reg_val = *reinterpret_cast<VdInt *>(&reg_val);
                if ((count >= vstart) && (count < num_values)) {
                  ScopedFPStatus set_fpstatus(rv_fp_->host_fp_interface());
                  auto op_val =
                      operation(vs2_value[count], fs1_value, mask_value);
                  auto int_op_val = *reinterpret_cast<VdInt *>(&op_val);
                  auto int_vs2_val =
                      *reinterpret_cast<Vs2Int *>(&vs2_value[count]);
                  auto int_fs1_val = *reinterpret_cast<Fs1Int *>(&fs1_value);
                  FPCompare<Vd>(
                      op_val, reg_val, delta_position,
                      absl::StrCat(name, "[", count, "] op(", vs2_value[count],
                                   "[0x", absl::Hex(int_vs2_val), "], ",
                                   fs1_value, "[0x", absl::Hex(int_fs1_val),
                                   "]) = ", absl::Hex(int_op_val), " != reg[",
                                   reg, "][", i, "]  (", reg_val, " [0x",
                                   absl::Hex(int_reg_val), "]) lmul8(", lmul8,
                                   ") rm = ", *(rv_fp_->GetRoundingMode())));
                } else {
                  EXPECT_EQ(0, reg_val) << absl::StrCat(
                      name, "  0 != reg[", reg, "][", i, "]  (", reg_val,
                      " [0x", absl::Hex(int_reg_val), "]) lmul8(", lmul8, ")");
                }
                count++;
              }
              if (HasFailure()) return;
            }
          }
          vlen = absl::Uniform(absl::IntervalOpenClosed, bitgen_, vstart,
                               num_reg_values);
        }
        vstart = absl::Uniform(absl::IntervalOpen, bitgen_, 0, num_reg_values);
      }
    }
  }

  // Templated helper function that tests FP vector-scalar instructions that do
  // not use the value of the mask bit.
  template <typename Vd, typename Vs2, typename Vs1>
  void BinaryOpFPTestHelperVX(absl::string_view name, int sew,
                              Instruction *inst, int delta_position,
                              std::function<Vd(Vs2, Vs1)> operation) {
    BinaryOpFPWithMaskTestHelperVX<Vd, Vs2, Vs1>(
        name, sew, inst, delta_position,
        [operation](Vs2 vs2, Vs1 vs1, bool mask_value) -> Vd {
          if (mask_value) {
            return operation(vs2, vs1);
          }
          return 0;
        });
  }

 protected:
  mpact::sim::riscv::RiscVFPState *rv_fp_;
};

}  // namespace test
}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_TEST_RISCV_VECTOR_FP_TEST_UTILITIES_H_
