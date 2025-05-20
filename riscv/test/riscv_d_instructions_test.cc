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

#include "riscv/riscv_d_instructions.h"

#include <cmath>
#include <cstdint>
#include <tuple>

#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_fp_info.h"
#include "riscv/riscv_register.h"
#include "riscv/test/riscv_fp_test_base.h"

namespace {

using ::mpact::sim::riscv::FPExceptions;
using ::mpact::sim::riscv::test::FPTypeInfo;
using ::mpact::sim::riscv::test::RiscVFPInstructionTestBase;
using ::mpact::sim::generic::operator*;  // NOLINT: is used below.

using ::mpact::sim::riscv::RiscVDAdd;
using ::mpact::sim::riscv::RiscVDCvtDs;
using ::mpact::sim::riscv::RiscVDCvtDw;
using ::mpact::sim::riscv::RiscVDCvtDwu;
using ::mpact::sim::riscv::RiscVDCvtSd;
using ::mpact::sim::riscv::RiscVDDiv;
using ::mpact::sim::riscv::RiscVDMadd;
using ::mpact::sim::riscv::RiscVDMax;
using ::mpact::sim::riscv::RiscVDMin;
using ::mpact::sim::riscv::RiscVDMsub;
using ::mpact::sim::riscv::RiscVDMul;
using ::mpact::sim::riscv::RiscVDNmadd;
using ::mpact::sim::riscv::RiscVDNmsub;
using ::mpact::sim::riscv::RiscVDSgnj;
using ::mpact::sim::riscv::RiscVDSgnjn;
using ::mpact::sim::riscv::RiscVDSgnjx;
using ::mpact::sim::riscv::RiscVDSqrt;
using ::mpact::sim::riscv::RiscVDSub;
using ::mpact::sim::riscv::RV32::RiscVDClass;
using ::mpact::sim::riscv::RV32::RiscVDCmpeq;
using ::mpact::sim::riscv::RV32::RiscVDCmple;
using ::mpact::sim::riscv::RV32::RiscVDCmplt;
using ::mpact::sim::riscv::RV32::RiscVDCvtWd;
using ::mpact::sim::riscv::RV32::RiscVDCvtWud;

class RV32DInstructionTest
    : public RiscVFPInstructionTestBase<mpact::sim::riscv::RV32Register> {};

static bool is_snan(double a) {
  if (!std::isnan(a)) return false;
  auto ua = *reinterpret_cast<uint64_t *>(&a);
  if ((ua & (1ULL << (FPTypeInfo<double>::kSigSize - 1))) == 0) return true;
  return false;
}

// Test basic arithmetic instructions.
TEST_F(RV32DInstructionTest, RiscVDadd) {
  SetSemanticFunction(&RiscVDAdd);
  BinaryOpFPTestHelper<double, double, double>(
      "dadd", instruction_, {"d", "d", "d"}, 64,
      [](double lhs, double rhs) -> double { return lhs + rhs; });
}

TEST_F(RV32DInstructionTest, RiscVDsub) {
  SetSemanticFunction(&RiscVDSub);
  BinaryOpFPTestHelper<double, double, double>(
      "dsub", instruction_, {"d", "d", "d"}, 64,
      [](double lhs, double rhs) -> double { return lhs - rhs; });
}

TEST_F(RV32DInstructionTest, RiscVDmul) {
  SetSemanticFunction(&RiscVDMul);
  BinaryOpFPTestHelper<double, double, double>(
      "dmul", instruction_, {"d", "d", "d"}, 64,
      [](double lhs, double rhs) -> double { return lhs * rhs; });
}

TEST_F(RV32DInstructionTest, RiscVDdiv) {
  SetSemanticFunction(&RiscVDDiv);
  BinaryOpFPTestHelper<double, double, double>(
      "ddiv", instruction_, {"d", "d", "d"}, 64,
      [](double lhs, double rhs) -> double { return lhs / rhs; });
}

// Test square root.
TEST_F(RV32DInstructionTest, RiscVDsqrt) {
  SetSemanticFunction(&RiscVDSqrt);
  UnaryOpWithFflagsFPTestHelper<double, double>(
      "dsqrt", instruction_, {"d", "d"}, 64,
      [](double lhs, int rm) -> std::tuple<double, uint32_t> {
        uint32_t flags = 0;
        if (lhs == 0) return std::tie(lhs, flags);
        double res;
        if (lhs > 0) {
          res = sqrt(lhs);
          uint64_t dhls = *reinterpret_cast<uint64_t *>(&lhs);
          // Get exponent of source value.
          int exp = (dhls & FPTypeInfo<double>::kExpMask) >>
                    FPTypeInfo<double>::kSigSize;
          exp -= FPTypeInfo<double>::kExpBias;
          // Get significand of result.
          uint64_t dres = *reinterpret_cast<uint64_t *>(&res);
          uint64_t sig = dres & FPTypeInfo<double>::kSigMask;
          bool is_square;
          // Slightly different test based on whether the exponent of the source
          // is even or odd.
          if (exp & 0x1) {
            is_square = ((sig & 0x1ff'ffff) == 0) && (res * res == lhs);
          } else {
            is_square = (sig & 0x3ff'ffff) == 0;
          }
          if (!is_square) {
            flags = *FPExceptions::kInexact;
          }
          return std::tie(res, flags);
        }
        if (!FPTypeInfo<double>::IsQNaN(lhs)) {
          flags = *FPExceptions::kInvalidOp;
        }
        uint64_t val = FPTypeInfo<double>::kCanonicalNaN;
        res = *reinterpret_cast<const double *>(&val);
        return std::tie(res, flags);
      });
}

// Test Min/Max.
TEST_F(RV32DInstructionTest, RiscVDmin) {
  SetSemanticFunction(&RiscVDMin);
  BinaryOpWithFflagsFPTestHelper<double, double, double>(
      "dmin", instruction_, {"d", "d", "d"}, 64,
      [](double lhs, double rhs) -> std::tuple<double, uint32_t> {
        uint32_t flag = 0;
        if (is_snan(lhs) || is_snan(rhs)) {
          flag = static_cast<uint32_t>(FPExceptions::kInvalidOp);
        }
        if (std::isnan(lhs) && std::isnan(rhs)) {
          uint64_t val = FPTypeInfo<double>::kCanonicalNaN;
          return std::tie(*reinterpret_cast<const double *>(&val), flag);
        }
        if (std::isnan(lhs)) return std::tie(rhs, flag);
        if (std::isnan(rhs)) return std::tie(lhs, flag);
        if ((lhs == 0.0) && (rhs == 0.0)) {
          return std::tie(std::signbit(lhs) ? lhs : rhs, flag);
        }
        return std::tie(lhs > rhs ? rhs : lhs, flag);
      });
}

TEST_F(RV32DInstructionTest, RiscVDmax) {
  SetSemanticFunction(&RiscVDMax);
  BinaryOpWithFflagsFPTestHelper<double, double, double>(
      "dmax", instruction_, {"d", "d", "d"}, 64,
      [](double lhs, double rhs) -> std::tuple<double, uint32_t> {
        uint32_t flag = 0;
        if (is_snan(lhs) || is_snan(rhs)) {
          flag = static_cast<uint32_t>(FPExceptions::kInvalidOp);
        }
        if (std::isnan(lhs) && std::isnan(rhs)) {
          uint64_t val = FPTypeInfo<double>::kCanonicalNaN;
          return std::tie(*reinterpret_cast<const double *>(&val), flag);
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
TEST_F(RV32DInstructionTest, RiscVDMadd) {
  SetSemanticFunction(&RiscVDMadd);
  TernaryOpFPTestHelper<double, double, double, double>(
      "dmadd", instruction_, {"d", "d", "d", "d"},
      FPTypeInfo<double>::kSigSize - 1,
      [](double lhs, double mhs, double rhs) -> double {
        return fma(lhs, mhs, rhs);
      });
}
TEST_F(RV32DInstructionTest, RiscVDMsub) {
  SetSemanticFunction(&RiscVDMsub);
  TernaryOpFPTestHelper<double, double, double, double>(
      "dmsub", instruction_, {"d", "d", "d", "d"},
      FPTypeInfo<double>::kSigSize - 1,
      [](double lhs, double mhs, double rhs) -> double {
        return fma(lhs, mhs, -rhs);
      });
}
TEST_F(RV32DInstructionTest, RiscVDNmadd) {
  SetSemanticFunction(&RiscVDNmadd);
  TernaryOpFPTestHelper<double, double, double, double>(
      "dnmadd", instruction_, {"d", "d", "d", "d"},
      FPTypeInfo<double>::kSigSize - 1,
      [](double lhs, double mhs, double rhs) -> double {
        return -fma(lhs, mhs, rhs);
      });
}
TEST_F(RV32DInstructionTest, RiscVDNmsub) {
  SetSemanticFunction(&RiscVDNmsub);
  TernaryOpFPTestHelper<double, double, double, double>(
      "dnmsub", instruction_, {"d", "d", "d", "d"},
      FPTypeInfo<double>::kSigSize - 1,
      [](double lhs, double mhs, double rhs) -> double {
        return -fma(lhs, mhs, -rhs);
      });
}

// Test conversion instructions.
// Double to signed 32 bit integer.
TEST_F(RV32DInstructionTest, RiscVDCvtWd) {
  SetSemanticFunction(&RiscVDCvtWd);
  UnaryOpWithFflagsFPTestHelper<int32_t, double>(
      "dcvt.w.d", instruction_, {"d", "x"}, 64,
      [&](double lhs, uint32_t rm) -> std::tuple<int32_t, uint32_t> {
        uint32_t flag = 0;
        const int32_t val = RoundToInteger<double, int32_t>(lhs, rm, flag);
        return std::make_tuple(val, flag);
      });
}

// Signed 32 bit integer to double.
TEST_F(RV32DInstructionTest, RiscVDCvtDw) {
  SetSemanticFunction(&RiscVDCvtDw);
  UnaryOpFPTestHelper<double, int32_t>(
      "dcvt.d.w", instruction_, {"x", "d"}, 64,
      [](int32_t lhs) -> double { return static_cast<double>(lhs); });
}

// Double to unsigned 32 bit integer.
TEST_F(RV32DInstructionTest, RiscVDCvtWud) {
  SetSemanticFunction(&RiscVDCvtWud);
  UnaryOpWithFflagsFPTestHelper<uint32_t, double>(
      "dcvt.wu.d", instruction_, {"d", "x"}, 64,
      [&](double lhs, uint32_t rm) -> std::tuple<uint32_t, uint32_t> {
        uint32_t flag = 0;
        const uint32_t val = RoundToInteger<double, uint32_t>(lhs, rm, flag);
        return std::make_tuple(val, flag);
      });
}

// Unsigned 32 bit integer to double.
TEST_F(RV32DInstructionTest, RiscVDCvtDwu) {
  SetSemanticFunction(&RiscVDCvtDwu);
  UnaryOpFPTestHelper<double, uint32_t>(
      "dcvt.d.w", instruction_, {"x", "d"}, 64,
      [](uint32_t lhs) -> double { return static_cast<double>(lhs); });
}

// Double to single precision.
TEST_F(RV32DInstructionTest, RiscVDCvtSd) {
  SetSemanticFunction(&RiscVDCvtSd);
  UnaryOpFPTestHelper<float, double>(
      "dcvt.s.d", instruction_, {"d", "x"}, 64,
      [](double lhs) -> float { return static_cast<float>(lhs); });
}

// Single precision to double.
TEST_F(RV32DInstructionTest, RiscVDCvtDs) {
  SetSemanticFunction(&RiscVDCvtDs);
  UnaryOpFPTestHelper<double, float>(
      "dcvt.d.s", instruction_, {"x", "d"}, 64,
      [](float lhs) -> double { return static_cast<double>(lhs); });
}

// Test sign manipulation instructions.
TEST_F(RV32DInstructionTest, RiscVDSgnj) {
  SetSemanticFunction(&RiscVDSgnj);
  BinaryOpFPTestHelper<double, double, double>(
      "dsgnj", instruction_, {"d", "d", "d"}, 64,
      [](double lhs, double rhs) -> double { return copysign(abs(lhs), rhs); });
}

TEST_F(RV32DInstructionTest, RiscVDSgnjn) {
  SetSemanticFunction(&RiscVDSgnjn);
  BinaryOpFPTestHelper<double, double, double>(
      "dsgnjn", instruction_, {"d", "d", "d"}, 64,
      [](double lhs, double rhs) -> double {
        return copysign(abs(lhs), -rhs);
      });
}

TEST_F(RV32DInstructionTest, RiscVDSgnjx) {
  SetSemanticFunction(&RiscVDSgnjx);
  BinaryOpFPTestHelper<double, double, double>(
      "dsgnjn", instruction_, {"d", "d", "d"}, 64,
      [](double lhs, double rhs) -> double {
        auto lhs_u = *reinterpret_cast<uint64_t *>(&lhs);
        auto rhs_u = *reinterpret_cast<uint64_t *>(&rhs);
        auto res_u = (lhs_u ^ rhs_u) & 0x8000'0000'0000'0000;
        auto res = *reinterpret_cast<double *>(&res_u);
        return copysign(abs(lhs), res);
      });
}

// Test compare instructions.
TEST_F(RV32DInstructionTest, RiscVDCmpeq) {
  SetSemanticFunction(&RiscVDCmpeq);
  BinaryOpWithFflagsFPTestHelper<uint32_t, double, double>(
      "dcmpeq", instruction_, {"d", "d", "x"}, 64,
      [](double lhs, double rhs) -> std::tuple<uint32_t, uint32_t> {
        int flag = 0;
        if (is_snan(lhs) || is_snan(rhs)) {
          flag = static_cast<uint32_t>(FPExceptions::kInvalidOp);
        }
        return std::make_tuple(lhs == rhs, flag);
      });
}

TEST_F(RV32DInstructionTest, RiscVDCmplt) {
  SetSemanticFunction(&RiscVDCmplt);
  BinaryOpWithFflagsFPTestHelper<uint32_t, double, double>(
      "dcmplt", instruction_, {"d", "d", "x"}, 64,
      [](double lhs, double rhs) -> std::tuple<uint32_t, uint32_t> {
        int flag = 0;
        if (std::isnan(lhs) || std::isnan(rhs)) {
          flag = static_cast<uint32_t>(FPExceptions::kInvalidOp);
        }
        return std::make_tuple(lhs < rhs, flag);
      });
}

TEST_F(RV32DInstructionTest, RiscVDCmple) {
  SetSemanticFunction(&RiscVDCmple);
  BinaryOpWithFflagsFPTestHelper<uint32_t, double, double>(
      "dcmple", instruction_, {"d", "d", "x"}, 64,
      [](double lhs, double rhs) -> std::tuple<uint32_t, uint32_t> {
        int flag = 0;
        if (std::isnan(lhs) || std::isnan(rhs)) {
          flag = static_cast<uint32_t>(FPExceptions::kInvalidOp);
        }
        return std::make_tuple(lhs <= rhs, flag);
      });
}

// Test class instruction.
TEST_F(RV32DInstructionTest, RiscVDClass) {
  SetSemanticFunction(&RiscVDClass);
  UnaryOpFPTestHelper<uint32_t, double>(
      "fclass.d", instruction_, {"d", "x"}, 64, [](double lhs) -> uint32_t {
        auto fp_class = std::fpclassify(lhs);
        switch (fp_class) {
          case FP_INFINITE:
            return std::signbit(lhs) ? 1 : 1 << 7;
          case FP_NAN: {
            auto uint_val =
                *reinterpret_cast<typename FPTypeInfo<double>::IntType *>(&lhs);
            bool quiet_nan =
                (uint_val >> (FPTypeInfo<double>::kSigSize - 1)) & 1;
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
