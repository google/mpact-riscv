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

#include "riscv/riscv_f_instructions.h"

#include <cmath>
#include <cstdint>
#include <tuple>

#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_fp_info.h"
#include "riscv/test/riscv_fp_test_base.h"

namespace {

using ::mpact::sim::riscv::FPExceptions;
using ::mpact::sim::riscv::test::FPTypeInfo;
using ::mpact::sim::riscv::test::RiscVFPInstructionTestBase;
using ::mpact::sim::generic::operator*;  // NOLINT: is used below.

using ::mpact::sim::riscv::RiscVFAdd;
using ::mpact::sim::riscv::RiscVFCvtSw;
using ::mpact::sim::riscv::RiscVFCvtSwu;
using ::mpact::sim::riscv::RiscVFDiv;
using ::mpact::sim::riscv::RiscVFMadd;
using ::mpact::sim::riscv::RiscVFMax;
using ::mpact::sim::riscv::RiscVFMin;
using ::mpact::sim::riscv::RiscVFMsub;
using ::mpact::sim::riscv::RiscVFMul;
using ::mpact::sim::riscv::RiscVFNmadd;
using ::mpact::sim::riscv::RiscVFNmsub;
using ::mpact::sim::riscv::RiscVFSgnj;
using ::mpact::sim::riscv::RiscVFSgnjn;
using ::mpact::sim::riscv::RiscVFSgnjx;
using ::mpact::sim::riscv::RiscVFSqrt;
using ::mpact::sim::riscv::RiscVFSub;
using ::mpact::sim::riscv::RV32::RiscVFClass;
using ::mpact::sim::riscv::RV32::RiscVFCmpeq;
using ::mpact::sim::riscv::RV32::RiscVFCmple;
using ::mpact::sim::riscv::RV32::RiscVFCmplt;
using ::mpact::sim::riscv::RV32::RiscVFCvtWs;
using ::mpact::sim::riscv::RV32::RiscVFCvtWus;
class RV32FInstructionTest : public RiscVFPInstructionTestBase {};

static bool is_snan(float a) {
  if (!std::isnan(a)) return false;
  uint32_t ua = *reinterpret_cast<uint32_t *>(&a);
  if ((ua & (1 << (FPTypeInfo<float>::kSigSize - 1))) == 0) return true;
  return false;
}

// Test basic arithmetic instructions.
TEST_F(RV32FInstructionTest, RiscVFadd) {
  SetSemanticFunction(&RiscVFAdd);
  BinaryOpFPTestHelper<float, float, float>(
      "fadd", instruction_, {"f", "f", "f"}, 32,
      [](float lhs, float rhs) -> float { return lhs + rhs; });
}

TEST_F(RV32FInstructionTest, RiscVFsub) {
  SetSemanticFunction(&RiscVFSub);
  BinaryOpFPTestHelper<float, float, float>(
      "fsub", instruction_, {"f", "f", "f"}, 32,
      [](float lhs, float rhs) -> float { return lhs - rhs; });
}

TEST_F(RV32FInstructionTest, RiscVFmul) {
  SetSemanticFunction(&RiscVFMul);
  BinaryOpFPTestHelper<float, float, float>(
      "fmul", instruction_, {"f", "f", "f"}, 32,
      [](float lhs, float rhs) -> float { return lhs * rhs; });
}

TEST_F(RV32FInstructionTest, RiscVFdiv) {
  SetSemanticFunction(&RiscVFDiv);
  BinaryOpFPTestHelper<float, float, float>(
      "fdiv", instruction_, {"f", "f", "f"}, 32,
      [](float lhs, float rhs) -> float { return lhs / rhs; });
}

// Test square root.
TEST_F(RV32FInstructionTest, RiscVFsqrt) {
  SetSemanticFunction(&RiscVFSqrt);
  UnaryOpWithFflagsFPTestHelper<float, float>(
      "fsqrt", instruction_, {"f", "f"}, 32,
      [](float lhs, int rm) -> std::tuple<float, uint32_t> {
        uint32_t flags = 0;
        if (lhs == 0) return std::tie(lhs, flags);
        float res;
        if (lhs > 0) {
          res = sqrt(lhs);
          double dres = sqrt(static_cast<double>(lhs));
          if (static_cast<double>(res) != dres) {
            flags = *FPExceptions::kInexact;
          }
          return std::tie(res, flags);
        }
        if (!FPTypeInfo<float>::IsQNaN(lhs)) {
          flags = *FPExceptions::kInvalidOp;
        }
        uint32_t val = FPTypeInfo<float>::kCanonicalNaN;
        res = *reinterpret_cast<const float *>(&val);
        return std::tie(res, flags);
      });
}

// Test Min/Max.
TEST_F(RV32FInstructionTest, RiscVFmin) {
  SetSemanticFunction(&RiscVFMin);
  BinaryOpWithFflagsFPTestHelper<float, float, float>(
      "fmin", instruction_, {"f", "f", "f"}, 32,
      [](float lhs, float rhs) -> std::tuple<float, uint32_t> {
        uint32_t flag = 0;
        if (is_snan(lhs) || is_snan(rhs)) {
          flag = static_cast<uint32_t>(FPExceptions::kInvalidOp);
        }
        if (std::isnan(lhs) && std::isnan(rhs)) {
          uint32_t val = FPTypeInfo<float>::kCanonicalNaN;
          return std::tie(*reinterpret_cast<const float *>(&val), flag);
        }
        if (std::isnan(lhs)) return std::tie(rhs, flag);
        if (std::isnan(rhs)) return std::tie(lhs, flag);
        if ((lhs == 0.0) && (rhs == 0.0)) {
          return std::tie(std::signbit(lhs) ? lhs : rhs, flag);
        }
        return std::tie(lhs > rhs ? rhs : lhs, flag);
      });
}

TEST_F(RV32FInstructionTest, RiscVFmax) {
  SetSemanticFunction(&RiscVFMax);
  BinaryOpWithFflagsFPTestHelper<float, float, float>(
      "fmax", instruction_, {"f", "f", "f"}, 32,
      [](float lhs, float rhs) -> std::tuple<float, uint32_t> {
        uint32_t flag = 0;
        if (is_snan(lhs) || is_snan(rhs)) {
          flag = static_cast<uint32_t>(FPExceptions::kInvalidOp);
        }
        if (std::isnan(lhs) && std::isnan(rhs)) {
          uint32_t val = FPTypeInfo<float>::kCanonicalNaN;
          return std::tie(*reinterpret_cast<const float *>(&val), flag);
        }
        if (std::isnan(lhs)) return std::tie(rhs, flag);
        if (std::isnan(rhs)) return std::tie(lhs, flag);
        if ((lhs == 0.0) && (rhs == 0.0)) {
          return std::tie(std::signbit(lhs) ? rhs : lhs, flag);
        }
        return std::tie(lhs < rhs ? rhs : lhs, flag);
      });
}

// Test MAC versions.
TEST_F(RV32FInstructionTest, RiscVFMadd) {
  SetSemanticFunction(&RiscVFMadd);
  TernaryOpWithFflagsFPTestHelper<float, float, float, float>(
      "fmadd", instruction_, {"f", "f", "f", "f"}, 32,
      [](float lhs, float mhs, float rhs) -> float {
        return fma(lhs, mhs, rhs);
      });
}
TEST_F(RV32FInstructionTest, RiscVFMsub) {
  SetSemanticFunction(&RiscVFMsub);
  TernaryOpWithFflagsFPTestHelper<float, float, float, float>(
      "fmsub", instruction_, {"f", "f", "f", "f"}, 32,
      [](float lhs, float mhs, float rhs) -> float {
        return fma(lhs, mhs, -rhs);
      });
}
TEST_F(RV32FInstructionTest, RiscVFNmadd) {
  SetSemanticFunction(&RiscVFNmadd);
  TernaryOpWithFflagsFPTestHelper<float, float, float, float>(
      "fnmadd", instruction_, {"f", "f", "f", "f"}, 32,
      [](float lhs, float mhs, float rhs) -> float {
        return fma(-lhs, mhs, -rhs);
      });
}
TEST_F(RV32FInstructionTest, RiscVFNmsub) {
  SetSemanticFunction(&RiscVFNmsub);
  TernaryOpWithFflagsFPTestHelper<float, float, float, float>(
      "fnmsub", instruction_, {"f", "f", "f", "f"}, 32,
      [](float lhs, float mhs, float rhs) -> float {
        return fma(-lhs, mhs, rhs);
      });
}

// Test conversion instructions.
// Double to signed 32 bit integer.
TEST_F(RV32FInstructionTest, RiscVFCvtWs) {
  SetSemanticFunction(&RiscVFCvtWs);
  UnaryOpWithFflagsFPTestHelper<int32_t, float>(
      "fcvt.w.s", instruction_, {"f", "x"}, 32,
      [&](float lhs, uint32_t rm) -> std::tuple<int32_t, uint32_t> {
        uint32_t flag = 0;
        const int32_t val = RoundToInteger<float, int32_t>(lhs, rm, flag);
        return std::make_tuple(val, flag);
      });
}

// Signed 32 bit integer to float.
TEST_F(RV32FInstructionTest, RiscVFCvtSw) {
  SetSemanticFunction(&RiscVFCvtSw);
  UnaryOpFPTestHelper<float, int32_t>(
      "fcvt.s.w", instruction_, {"x", "f"}, 32,
      [](int32_t lhs) -> float { return static_cast<float>(lhs); });
}

// Double to unsigned 32 bit integer.
TEST_F(RV32FInstructionTest, RiscVFCvtWus) {
  SetSemanticFunction(&RiscVFCvtWus);
  UnaryOpWithFflagsFPTestHelper<uint32_t, float>(
      "fcvt.wu.s", instruction_, {"f", "x"}, 64,
      [&](float lhs, uint32_t rm) -> std::tuple<uint32_t, uint32_t> {
        uint32_t flag = 0;
        const int32_t val = RoundToInteger<float, uint32_t>(lhs, rm, flag);
        return std::make_tuple(val, flag);
      });
}

// Unsigned 32 bit integer to float.
TEST_F(RV32FInstructionTest, RiscVFCvtSwu) {
  SetSemanticFunction(&RiscVFCvtSwu);
  UnaryOpFPTestHelper<float, uint32_t>(
      "fcvt.s.w", instruction_, {"x", "f"}, 32,
      [](uint32_t lhs) -> float { return static_cast<float>(lhs); });
}

// Test sign manipulation instructions.
TEST_F(RV32FInstructionTest, RiscVFSgnj) {
  SetSemanticFunction(&RiscVFSgnj);
  BinaryOpFPTestHelper<float, float, float>(
      "fsgnj", instruction_, {"f", "f", "f"}, 32,
      [](float lhs, float rhs) -> float { return copysign(abs(lhs), rhs); });
}

TEST_F(RV32FInstructionTest, RiscVFSgnjn) {
  SetSemanticFunction(&RiscVFSgnjn);
  BinaryOpFPTestHelper<float, float, float>(
      "fsgnjn", instruction_, {"f", "f", "f"}, 32,
      [](float lhs, float rhs) -> float { return copysign(abs(lhs), -rhs); });
}

TEST_F(RV32FInstructionTest, RiscVFSgnjx) {
  SetSemanticFunction(&RiscVFSgnjx);
  BinaryOpFPTestHelper<float, float, float>(
      "fsgnjn", instruction_, {"f", "f", "f"}, 32,
      [](float lhs, float rhs) -> float {
        auto lhs_u = *reinterpret_cast<uint32_t *>(&lhs);
        auto rhs_u = *reinterpret_cast<uint32_t *>(&rhs);
        auto res_u = (lhs_u ^ rhs_u) & 0x8000'0000;
        auto res = *reinterpret_cast<float *>(&res_u);
        return copysign(abs(lhs), res);
      });
}

// Test compare instructions.
TEST_F(RV32FInstructionTest, RiscVFCmpeq) {
  SetSemanticFunction(&RiscVFCmpeq);
  BinaryOpWithFflagsFPTestHelper<uint32_t, float, float>(
      "fcmpeq", instruction_, {"f", "f", "x"}, 32,
      [](float lhs, float rhs) -> std::tuple<uint32_t, uint32_t> {
        uint32_t flag = 0;
        if (is_snan(lhs) || is_snan(rhs)) {
          flag = static_cast<uint32_t>(FPExceptions::kInvalidOp);
        }
        return std::make_tuple(lhs == rhs, flag);
      });
}

TEST_F(RV32FInstructionTest, RiscVFCmplt) {
  SetSemanticFunction(&RiscVFCmplt);
  BinaryOpWithFflagsFPTestHelper<uint32_t, float, float>(
      "fcmplt", instruction_, {"f", "f", "x"}, 32,
      [](float lhs, float rhs) -> std::tuple<uint32_t, uint32_t> {
        uint32_t flag = 0;
        if (std::isnan(lhs) || std::isnan(rhs)) {
          flag = static_cast<uint32_t>(FPExceptions::kInvalidOp);
        }
        return std::make_tuple(lhs < rhs, flag);
      });
}

TEST_F(RV32FInstructionTest, RiscVFCmple) {
  SetSemanticFunction(&RiscVFCmple);
  BinaryOpWithFflagsFPTestHelper<uint32_t, float, float>(
      "fcmple", instruction_, {"f", "f", "x"}, 32,
      [](float lhs, float rhs) -> std::tuple<uint32_t, uint32_t> {
        uint32_t flag = 0;
        if (std::isnan(lhs) || std::isnan(rhs)) {
          flag = static_cast<uint32_t>(FPExceptions::kInvalidOp);
        }
        return std::make_tuple(lhs <= rhs, flag);
      });
}

// Test class instruction.
TEST_F(RV32FInstructionTest, RiscVFClass) {
  SetSemanticFunction(&RiscVFClass);
  UnaryOpFPTestHelper<uint32_t, float>(
      "fclass.d", instruction_, {"f", "x"}, 32, [](float lhs) -> uint32_t {
        auto fp_class = std::fpclassify(lhs);
        switch (fp_class) {
          case FP_INFINITE:
            return std::signbit(lhs) ? 1 : 1 << 7;
          case FP_NAN: {
            auto uint_val =
                *reinterpret_cast<typename FPTypeInfo<float>::IntType *>(&lhs);
            bool quiet_nan =
                (uint_val >> (FPTypeInfo<float>::kSigSize - 1)) & 1;
            return quiet_nan ? 1 << 9 : 1 << 8;
          }
          case FP_ZERO:
            return std::signbit(lhs) ? 1 << 3 : 1 << 4;
          case FP_SUBNORMAL:
            return std::signbit(lhs) ? 1 << 2 : 1 << 5;
          case FP_NORMAL:
            return std::signbit(lhs) ? 1 << 1 : 1 << 6;
        }
        return 0;
      });
}

}  // namespace
