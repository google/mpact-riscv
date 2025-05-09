// Copyright 2025 Google LLC
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

#include "riscv/riscv_zfh_instructions.h"

#include <sys/types.h>

#include <algorithm>
#include <any>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <ios>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "absl/base/casts.h"
#include "absl/random/distributions.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/operand_interface.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_fp_host.h"
#include "riscv/riscv_fp_info.h"
#include "riscv/riscv_i_instructions.h"
#include "riscv/riscv_register.h"
#include "riscv/test/riscv_fp_test_base.h"

// This file contains the unit tests for the RiscV ZFH extension instructions.
// Testing is focused on the instruction semantic functions.

namespace {

using ::mpact::sim::generic::operator*;  // NOLINT: is used below.
using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::generic::HalfFP;
using ::mpact::sim::generic::ImmediateOperand;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::FPRoundingMode;
using ::mpact::sim::riscv::RiscVZfhCvtDh;
using ::mpact::sim::riscv::RiscVZfhCvtHd;
using ::mpact::sim::riscv::RiscVZfhCvtHs;
using ::mpact::sim::riscv::RiscVZfhCvtSh;
using ::mpact::sim::riscv::RiscVZfhFadd;
using ::mpact::sim::riscv::RiscVZfhFdiv;
using ::mpact::sim::riscv::RiscVZfhFlhChild;
using ::mpact::sim::riscv::RiscVZfhFmax;
using ::mpact::sim::riscv::RiscVZfhFmin;
using ::mpact::sim::riscv::RiscVZfhFmul;
using ::mpact::sim::riscv::RiscVZfhFMvhx;
using ::mpact::sim::riscv::RiscVZfhFsgnj;
using ::mpact::sim::riscv::RiscVZfhFsgnjn;
using ::mpact::sim::riscv::RiscVZfhFsgnjx;
using ::mpact::sim::riscv::RiscVZfhFsqrt;
using ::mpact::sim::riscv::RiscVZfhFsub;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RV64Register;
using ::mpact::sim::riscv::RVFpRegister;
using ::mpact::sim::riscv::ScopedFPRoundingMode;
using ::mpact::sim::riscv::RV32::RiscVILhu;
using ::mpact::sim::riscv::RV32::RiscVZfhCvtHw;
using ::mpact::sim::riscv::RV32::RiscVZfhCvtHwu;
using ::mpact::sim::riscv::RV32::RiscVZfhCvtWh;
using ::mpact::sim::riscv::RV32::RiscVZfhCvtWuh;
using ::mpact::sim::riscv::RV32::RiscVZfhFclass;
using ::mpact::sim::riscv::RV32::RiscVZfhFcmpeq;
using ::mpact::sim::riscv::RV32::RiscVZfhFcmple;
using ::mpact::sim::riscv::RV32::RiscVZfhFcmplt;
using ::mpact::sim::riscv::RV32::RiscVZfhFMvxh;

using ::mpact::sim::riscv::test::FloatingPointToString;
using ::mpact::sim::riscv::test::FPCompare;
using ::mpact::sim::riscv::test::FpConversionsTestHelper;
using ::mpact::sim::riscv::test::FPTypeInfo;
using ::mpact::sim::riscv::test::kTestValueLength;
using ::mpact::sim::riscv::test::RiscVFPInstructionTestBase;

const int kRoundingModeRoundToNearest =
    static_cast<int>(FPRoundingMode::kRoundToNearest);
const int kRoundingModeRoundTowardsZero =
    static_cast<int>(FPRoundingMode::kRoundTowardsZero);
const int kRoundingModeRoundDown = static_cast<int>(FPRoundingMode::kRoundDown);
const int kRoundingModeRoundUp = static_cast<int>(FPRoundingMode::kRoundUp);

class RVZfhInstructionTestBase : public RiscVFPInstructionTestBase {
 protected:
  template <typename AddressType, typename ValueType>
  void SetupMemory(AddressType, ValueType);

  template <typename ReturnType, typename IntegerRegister>
  ReturnType LoadHalfHelper(typename IntegerRegister::ValueType, int16_t);

  template <typename DestRegisterType, typename LhsRegisterType, typename R,
            typename LHS>
  void UnaryOpWithFflagsMixedTestHelper(
      absl::string_view name, Instruction *inst,
      absl::Span<const absl::string_view> reg_prefixes, int delta_position,
      std::function<std::tuple<R, uint32_t>(LHS, uint32_t)> operation);
};

template <typename AddressType, typename ValueType>
void RVZfhInstructionTestBase::SetupMemory(AddressType address,
                                           ValueType value) {
  DataBuffer *mem_db = state_->db_factory()->Allocate<ValueType>(1);
  mem_db->Set<ValueType>(0, value);
  state_->StoreMemory(instruction_, address, mem_db);
  mem_db->DecRef();
}

template <typename ReturnType, typename IntegerRegister>
ReturnType RVZfhInstructionTestBase::LoadHalfHelper(
    typename IntegerRegister::ValueType base, int16_t offset) {
  // Technically the offset for FL* is a 12 bit signed integer but we'll use 16
  // bits for testing.
  const std::string kRs1Name("x1");
  const std::string kFrdName("f5");
  AppendRegisterOperands<IntegerRegister>({kRs1Name}, {});
  AppendRegisterOperands<RVFpRegister>(child_instruction_, {}, {kFrdName});

  ImmediateOperand<int16_t> *offset_source_operand =
      new ImmediateOperand<int16_t>(offset);
  instruction_->AppendSource(offset_source_operand);

  SetRegisterValues<typename IntegerRegister::ValueType, IntegerRegister>(
      {{kRs1Name, static_cast<IntegerRegister::ValueType>(base)}});
  SetRegisterValues<uint64_t, RVFpRegister>({{kFrdName, 0}});

  instruction_->Execute(nullptr);

  ReturnType observed_val = state_->GetRegister<RVFpRegister>(kFrdName)
                                .first->data_buffer()
                                ->template Get<ReturnType>(0);
  return observed_val;
}

// Helper for unary instructions that go between floats and integers.
template <typename DestRegisterType, typename LhsRegisterType, typename R,
          typename LHS>
void RVZfhInstructionTestBase::UnaryOpWithFflagsMixedTestHelper(
    absl::string_view name, Instruction *inst,
    absl::Span<const absl::string_view> reg_prefixes, int delta_position,
    std::function<std::tuple<R, uint32_t>(LHS, uint32_t)> operation) {
  using LhsInt = typename FPTypeInfo<LHS>::IntType;
  using RInt = typename FPTypeInfo<R>::IntType;
  LHS lhs_values[kTestValueLength];
  auto lhs_span = absl::Span<LHS>(lhs_values);
  const std::string kR1Name = absl::StrCat(reg_prefixes[0], 1);
  const std::string kRdName = absl::StrCat(reg_prefixes[1], 5);
  // This is used for the rounding mode operand.
  const std::string kRmName = absl::StrCat("x", 10);
  if (kR1Name[0] == 'x') {
    AppendRegisterOperands<RV32Register>({kR1Name}, {});
  } else {
    AppendRegisterOperands<RVFpRegister>({kR1Name}, {});
  }
  if (kRdName[0] == 'x') {
    AppendRegisterOperands<RV32Register>({}, {kRdName});
  } else {
    AppendRegisterOperands<RVFpRegister>({}, {kRdName});
  }
  AppendRegisterOperands<RV32Register>({kRmName}, {});
  auto *flag_op = rv_fp_->fflags()->CreateSetDestinationOperand(0, "fflags");
  instruction_->AppendDestination(flag_op);
  if constexpr (std::is_integral<LHS>::value) {
    for (auto &lhs : lhs_span) {
      lhs = absl::Uniform(absl::IntervalClosed, bitgen_,
                          std::numeric_limits<LHS>::min(),
                          std::numeric_limits<LHS>::max());
    }
    *reinterpret_cast<LHS *>(&lhs_span[0]) = 0;
    *reinterpret_cast<LHS *>(&lhs_span[1]) = 1;
    *reinterpret_cast<LHS *>(&lhs_span[2]) = 2;
    *reinterpret_cast<LHS *>(&lhs_span[3]) = 4;
    *reinterpret_cast<LHS *>(&lhs_span[4]) = 8;
    *reinterpret_cast<LHS *>(&lhs_span[5]) = 16;
    *reinterpret_cast<LHS *>(&lhs_span[6]) = 1024;
    *reinterpret_cast<LHS *>(&lhs_span[7]) = 65000;
  } else {
    FillArrayWithRandomFPValues<LHS>(lhs_span);
    *reinterpret_cast<LhsInt *>(&lhs_span[0]) = FPTypeInfo<LHS>::kQNaN;
    *reinterpret_cast<LhsInt *>(&lhs_span[1]) = FPTypeInfo<LHS>::kSNaN;
    *reinterpret_cast<LhsInt *>(&lhs_span[2]) = FPTypeInfo<LHS>::kPosInf;
    *reinterpret_cast<LhsInt *>(&lhs_span[3]) = FPTypeInfo<LHS>::kNegInf;
    *reinterpret_cast<LhsInt *>(&lhs_span[4]) = FPTypeInfo<LHS>::kPosZero;
    *reinterpret_cast<LhsInt *>(&lhs_span[5]) = FPTypeInfo<LHS>::kNegZero;
    *reinterpret_cast<LhsInt *>(&lhs_span[6]) = FPTypeInfo<LHS>::kPosDenorm;
    *reinterpret_cast<LhsInt *>(&lhs_span[7]) = FPTypeInfo<LHS>::kNegDenorm;
  }
  for (int i = 0; i < kTestValueLength; i++) {
    if constexpr (std::is_integral<LHS>::value) {
      SetRegisterValues<LHS, LhsRegisterType>({{kR1Name, lhs_span[i]}});
    } else {
      SetNaNBoxedRegisterValues<LHS, LhsRegisterType>({{kR1Name, lhs_span[i]}});
    }

    for (int rm : {0, 1, 2, 3, 4}) {
      rv_fp_->SetRoundingMode(static_cast<FPRoundingMode>(rm));
      rv_fp_->fflags()->Write(static_cast<uint32_t>(0));
      SetRegisterValues<int, RV32Register>({{kRmName, rm}, {}});
      SetRegisterValues<typename DestRegisterType::ValueType, DestRegisterType>(
          {{kRdName, 0}});

      inst->Execute(nullptr);
      auto instruction_fflags = rv_fp_->fflags()->GetUint32();

      R op_val;
      uint32_t test_operation_fflags;
      {
        ScopedFPRoundingMode scoped_rm(rv_fp_->host_fp_interface(), rm);
        std::tie(op_val, test_operation_fflags) = operation(lhs_span[i], rm);
      }

      auto reg_val = state_->GetRegister<DestRegisterType>(kRdName)
                         .first->data_buffer()
                         ->template Get<R>(0);
      FPCompare<R>(
          op_val, reg_val, delta_position,
          absl::StrCat(name, "  ", i, ": ",
                       FloatingPointToString<LHS>(lhs_span[i]), " rm: ", rm));
      LhsInt lhs_uint = absl::bit_cast<LhsInt>(lhs_span[i]);
      RInt op_val_uint = absl::bit_cast<RInt>(op_val);
      EXPECT_EQ(test_operation_fflags, instruction_fflags)
          << name << "(" << FloatingPointToString<LHS>(lhs_span[i]) << ")  "
          << std::hex << name << "(0x" << lhs_uint
          << ") == " << FloatingPointToString<R>(op_val) << std::hex << "  0x"
          << op_val_uint << " rm: " << rm;
    }
  }
}

// A source operand that is used to set the rounding mode. This is less
// confusing than using a register source operand since the rounding mode is
// part of the instruction encoding.
class TestRoundingModeSourceOperand
    : public mpact::sim::generic::SourceOperandInterface {
 public:
  explicit TestRoundingModeSourceOperand()
      : rounding_mode_(FPRoundingMode::kRoundToNearest) {}

  void SetRoundingMode(FPRoundingMode rounding_mode) {
    rounding_mode_ = rounding_mode;
  }

  bool AsBool(int) override { return static_cast<bool>(rounding_mode_); }
  int8_t AsInt8(int) override { return static_cast<int8_t>(rounding_mode_); }
  uint8_t AsUint8(int) override { return static_cast<uint8_t>(rounding_mode_); }
  int16_t AsInt16(int) override { return static_cast<int16_t>(rounding_mode_); }
  uint16_t AsUint16(int) override {
    return static_cast<uint16_t>(rounding_mode_);
  }
  int32_t AsInt32(int) override { return static_cast<int32_t>(rounding_mode_); }
  uint32_t AsUint32(int) override {
    return static_cast<uint32_t>(rounding_mode_);
  }
  int64_t AsInt64(int) override { return static_cast<int64_t>(rounding_mode_); }
  uint64_t AsUint64(int) override {
    return static_cast<uint64_t>(rounding_mode_);
  }

  std::vector<int> shape() const override { return {1}; }
  std::string AsString() const override { return std::string(""); }
  std::any GetObject() const override { return std::any(); }

 protected:
  FPRoundingMode rounding_mode_;
};

class RV32ZfhInstructionTest : public RVZfhInstructionTestBase {
 protected:
  // Test conversion instructions. The instance variable semantic_function_ is
  // used to set the semantic function for the instruction and should be set
  // before calling this function.
  template <typename T, typename U, int rm = 0>
  T ConversionHelper(U input_val) {
    // Initialize a fresh instruction.
    ResetInstruction();
    assert(semantic_function_);
    SetSemanticFunction(semantic_function_);

    // Configure source and destination operands for the instruction.
    AppendRegisterOperands<RVFpRegister>({"f1"}, {"f5"});
    instruction_->AppendSource(new TestRoundingModeSourceOperand());
    auto *flag_op = rv_fp_->fflags()->CreateSetDestinationOperand(0, "fflags");
    instruction_->AppendDestination(flag_op);
    assert(instruction_->SourcesSize() == 2);
    assert(instruction_->DestinationsSize() == 2);

    // Set all operands to known values before executing the instruction.
    static_cast<TestRoundingModeSourceOperand *>(instruction_->Source(1))
        ->SetRoundingMode(static_cast<FPRoundingMode>(rm));
    rv_fp_->SetRoundingMode(static_cast<FPRoundingMode>(rm));
    SetNaNBoxedRegisterValues<U, RVFpRegister>({{"f1", input_val}});
    SetRegisterValues<int, RVFpRegister>({{"f5", 0xDEAFBEEFDEADBEEF}});
    rv_fp_->fflags()->Write(static_cast<uint32_t>(0));

    instruction_->Execute(nullptr);
    T reg_val = state_->GetRegister<RVFpRegister>("f5")
                    .first->data_buffer()
                    ->template Get<T>(0);
    return reg_val;
  }

  template <FPRoundingMode>
  void RoundingConversionTestHelper(uint32_t, uint16_t, uint32_t &, uint32_t,
                                    uint16_t, uint32_t &);

  template <FPRoundingMode rm>
  void RoundingPointTest(uint16_t);

  Instruction::SemanticFunction semantic_function_ = nullptr;
};

template <FPRoundingMode rm>
void RV32ZfhInstructionTest::RoundingConversionTestHelper(
    uint32_t float_uint_before, uint16_t half_uint_before,
    uint32_t &first_expected_fflags, uint32_t float_uint_after,
    uint16_t half_uint_after, uint32_t &second_expected_fflags) {
  float input_val;
  HalfFP expected_val;
  HalfFP actual_val;

  input_val = absl::bit_cast<float>(float_uint_before);
  expected_val = {.value = half_uint_before};
  actual_val = ConversionHelper<HalfFP, float, static_cast<int>(rm)>(input_val);
  EXPECT_EQ(expected_val.value, actual_val.value)
      << "expected: " << std::hex << expected_val.value
      << ", actual: " << std::hex << actual_val.value
      << ", float_uint: " << std::hex << float_uint_before
      << ", rounding_mode: " << static_cast<int>(rm);
  EXPECT_EQ(first_expected_fflags, rv_fp_->fflags()->GetUint32())
      << "while converting: " << std::hex << float_uint_before
      << " to:" << std::hex << actual_val.value
      << " with rounding mode: " << static_cast<int>(rm);

  input_val = absl::bit_cast<float>(float_uint_after);
  expected_val = {.value = half_uint_after};
  actual_val = ConversionHelper<HalfFP, float, static_cast<int>(rm)>(input_val);
  EXPECT_EQ(expected_val.value, actual_val.value)
      << "expected: " << std::hex << expected_val.value
      << ", actual: " << std::hex << actual_val.value
      << ", float_uint: " << std::hex << float_uint_after
      << ", rounding_mode: " << static_cast<int>(rm);
  EXPECT_EQ(second_expected_fflags, rv_fp_->fflags()->GetUint32())
      << "while converting: " << std::hex << float_uint_after
      << " to:" << std::hex << actual_val.value
      << " with rounding mode: " << static_cast<int>(rm);
}

// Test the FP16 load instruction. The semantic functions should match the isa
// file.
TEST_F(RV32ZfhInstructionTest, RiscVFlh) {
  SetSemanticFunction(&RiscVILhu);
  SetChildInstruction();
  SetChildSemanticFunction(&RiscVZfhFlhChild);

  SetupMemory<uint32_t, uint16_t>(0xFF, 0xBEEF);

  HalfFP observed_val =
      LoadHalfHelper<HalfFP, RV32Register>(/* base */ 0x0, /* offset */ 0xFF);
  EXPECT_EQ(observed_val.value, 0xBEEF);
}

// Test the FP16 load instruction. When looking at the register contents as a
// float, it should be NaN.
TEST_F(RV32ZfhInstructionTest, RiscVFlh_float_nanbox) {
  SetSemanticFunction(&RiscVILhu);
  SetChildInstruction();
  SetChildSemanticFunction(&RiscVZfhFlhChild);

  SetupMemory<uint32_t, uint16_t>(0xFF, 0xBEEF);

  float observed_val =
      LoadHalfHelper<float, RV32Register>(/* base */ 0xFF, /* offset */ 0);
  EXPECT_TRUE(std::isnan(observed_val));
}

// Test the FP16 load instruction. When looking at the register contents as a
// double, it should be NaN.
TEST_F(RV32ZfhInstructionTest, RiscVFlh_double_nanbox) {
  SetSemanticFunction(&RiscVILhu);
  SetChildInstruction();
  SetChildSemanticFunction(&RiscVZfhFlhChild);

  SetupMemory<uint32_t, uint16_t>(0xFF, 0xBEEF);

  double observed_val = LoadHalfHelper<double, RV32Register>(
      /* base */ 0x0100, /* offset */ -1);
  EXPECT_TRUE(std::isnan(observed_val));
}

// Move half precision from a float register to an integer register. The IEEE754
// encoding is preserved in the integer register.
TEST_F(RV32ZfhInstructionTest, RiscVZfhFMvxh) {
  SetSemanticFunction(&RiscVZfhFMvxh);
  UnaryOpFPTestHelper<uint32_t, HalfFP>(
      "fmv.x.h", instruction_, {"f", "x"}, 32, [](HalfFP half_fp) -> uint32_t {
        bool sign = 1 & (half_fp.value >> (FPTypeInfo<HalfFP>::kBitSize - 1));
        // Fill the upper XLEN-16 bits with the sign bit as per the spec.
        uint32_t result = sign ? 0xFFFF'0000 : 0;
        result |= static_cast<uint32_t>(half_fp.value);
        return result;
      });
}

// Move half precision from an integer register (lower 16 bits) to a float
// register.
TEST_F(RV32ZfhInstructionTest, RiscVZfhFMvhx) {
  SetSemanticFunction(&RiscVZfhFMvhx);
  UnaryOpFPTestHelper<HalfFP, uint64_t>(
      "fmv.h.x", instruction_, {"x", "f"}, 32, [](uint64_t scalar) -> HalfFP {
        return HalfFP{.value = static_cast<uint16_t>(scalar)};
      });
}

// Half precision to single precision conversion.
TEST_F(RV32ZfhInstructionTest, RiscVZfhCvtSh) {
  SetSemanticFunction(&RiscVZfhCvtSh);
  UnaryOpWithFflagsFPTestHelper<float, HalfFP>(
      "fcvt.s.h", instruction_, {"f", "f"}, 32,
      [](HalfFP half_fp, int rm) -> std::tuple<float, uint32_t> {
        uint32_t fflags = 0;
        float float_result =
            FpConversionsTestHelper(half_fp).ConvertWithFlags<float>(
                fflags, static_cast<FPRoundingMode>(rm));
        return std::make_tuple(float_result, fflags);
      });
}

// Single precision to half precision conversion.
TEST_F(RV32ZfhInstructionTest, RiscVZfhCvtHs) {
  SetSemanticFunction(&RiscVZfhCvtHs);
  UnaryOpWithFflagsFPTestHelper<HalfFP, float>(
      "fcvt.h.s", instruction_, {"f", "f"}, 32,
      [](float input_float, int rm) -> std::tuple<HalfFP, uint32_t> {
        uint32_t fflags = 0;
        HalfFP half_result = FpConversionsTestHelper(input_float)
                                 .ConvertWithFlags<HalfFP>(
                                     fflags, static_cast<FPRoundingMode>(rm));
        return std::make_tuple(half_result, fflags);
      });
}

// Find the nearest floats that convert to the given half precision values.
std::tuple<uint32_t, uint32_t> GetRoundingPoints(uint16_t first,
                                                 uint16_t second,
                                                 FPRoundingMode rm) {
  uint16_t upper_uhalf = std::max(first, second);
  uint32_t upper_ufloat = absl::bit_cast<uint32_t>(
      FpConversionsTestHelper(upper_uhalf).Convert<float>(rm));

  uint16_t lower_uhalf = std::min(first, second);
  uint32_t lower_ufloat = absl::bit_cast<uint32_t>(
      FpConversionsTestHelper(lower_uhalf).Convert<float>(rm));
  while (upper_ufloat - lower_ufloat > 1) {
    uint32_t udelta = upper_ufloat - lower_ufloat;
    uint32_t mid_ufloat = lower_ufloat + (udelta >> 1);
    HalfFP mid_half = FpConversionsTestHelper(mid_ufloat).Convert<HalfFP>(rm);
    uint16_t mid_uhalf = mid_half.value;
    if (upper_uhalf == mid_uhalf) {
      upper_ufloat = mid_ufloat;
    } else if (lower_uhalf == mid_uhalf) {
      lower_ufloat = mid_ufloat;
    }
  }
  if (first > second) {
    return std::make_tuple(upper_ufloat, lower_ufloat);
  }
  return std::make_tuple(lower_ufloat, upper_ufloat);
}

template <FPRoundingMode rm>
void RV32ZfhInstructionTest::RoundingPointTest(uint16_t base_uhalf) {
  uint16_t first_uhalf = base_uhalf, second_uhalf = base_uhalf + 1;
  uint32_t first_ufloat, second_ufloat;
  uint32_t first_expected_fflags = 0, second_expected_fflags = 0;

  std::tie(first_ufloat, second_ufloat) =
      GetRoundingPoints(first_uhalf, second_uhalf, rm);
  // Get the expected fflags
  FpConversionsTestHelper(first_ufloat)
      .ConvertWithFlags<HalfFP>(first_expected_fflags, rm);
  FpConversionsTestHelper(second_ufloat)
      .ConvertWithFlags<HalfFP>(second_expected_fflags, rm);
  RoundingConversionTestHelper<rm>(first_ufloat, first_uhalf,
                                   first_expected_fflags, second_ufloat,
                                   second_uhalf, second_expected_fflags);
}

// Verify that the rounding points in the semantic functions match the
// rounding points in the test helpers. Test across all rounding modes.
TEST_F(RV32ZfhInstructionTest,
       RiscVZfhCvtHs_conversion_rounding_points_first_nonzero) {
  semantic_function_ = &RiscVZfhCvtHs;
  // The first float that converts to a non zero half precision value after
  // rounding. Zero before, denormal after.
  uint16_t pos_uhalf = 0b0'00000'0000000000;
  RoundingPointTest<FPRoundingMode::kRoundToNearest>(pos_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundTowardsZero>(pos_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundDown>(pos_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundUp>(pos_uhalf);

  uint16_t neg_uhalf = 0b1'00000'0000000000;
  RoundingPointTest<FPRoundingMode::kRoundToNearest>(neg_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundTowardsZero>(neg_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundDown>(neg_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundUp>(neg_uhalf);
}

TEST_F(RV32ZfhInstructionTest,
       RiscVZfhCvtHs_conversion_rounding_points_denrom_denorm) {
  semantic_function_ = &RiscVZfhCvtHs;

  // Rounding denormal before and denormal after
  uint16_t pos_uhalf = 0b0'00000'0000000001;
  RoundingPointTest<FPRoundingMode::kRoundToNearest>(pos_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundTowardsZero>(pos_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundDown>(pos_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundUp>(pos_uhalf);

  uint16_t neg_uhalf = 0b1'00000'0000000001;
  RoundingPointTest<FPRoundingMode::kRoundToNearest>(neg_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundTowardsZero>(neg_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundDown>(neg_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundUp>(neg_uhalf);
}

TEST_F(RV32ZfhInstructionTest,
       RiscVZfhCvtHs_conversion_rounding_points_denorm_normal) {
  semantic_function_ = &RiscVZfhCvtHs;

  // The rounding overflows the significand and should increase the exponent.
  // Denormal before, normal after.
  uint16_t pos_uhalf = 0b0'00000'1111111111;
  RoundingPointTest<FPRoundingMode::kRoundToNearest>(pos_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundTowardsZero>(pos_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundDown>(pos_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundUp>(pos_uhalf);

  uint16_t neg_uhalf = 0b1'00000'1111111111;
  RoundingPointTest<FPRoundingMode::kRoundToNearest>(neg_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundTowardsZero>(neg_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundDown>(neg_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundUp>(neg_uhalf);
}

TEST_F(RV32ZfhInstructionTest,
       RiscVZfhCvtHs_conversion_rounding_points_normal_normal) {
  semantic_function_ = &RiscVZfhCvtHs;

  // Rounding normal before and normal after.
  uint16_t pos_uhalf = 0b0'11110'1111111110;
  RoundingPointTest<FPRoundingMode::kRoundToNearest>(pos_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundTowardsZero>(pos_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundDown>(pos_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundUp>(pos_uhalf);

  uint16_t neg_uhalf = 0b1'11110'1111111110;
  RoundingPointTest<FPRoundingMode::kRoundToNearest>(neg_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundTowardsZero>(neg_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundDown>(neg_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundUp>(neg_uhalf);
}

TEST_F(RV32ZfhInstructionTest, RiscVZfhCvtHs_conversion_rounding_points_inf) {
  semantic_function_ = &RiscVZfhCvtHs;

  // Rounding normal before and infinity after.
  uint16_t pos_uhalf = 0b0'11110'1111111111;
  RoundingPointTest<FPRoundingMode::kRoundToNearest>(pos_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundTowardsZero>(pos_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundDown>(pos_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundUp>(pos_uhalf);

  uint16_t neg_uhalf = 0b1'11110'1111111111;
  RoundingPointTest<FPRoundingMode::kRoundToNearest>(neg_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundTowardsZero>(neg_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundDown>(neg_uhalf);
  RoundingPointTest<FPRoundingMode::kRoundUp>(neg_uhalf);
}

// Half precision to double precision conversion.
TEST_F(RV32ZfhInstructionTest, RiscVZfhCvtDh) {
  SetSemanticFunction(&RiscVZfhCvtDh);
  UnaryOpWithFflagsFPTestHelper<double, HalfFP>(
      "fcvt.d.h", instruction_, {"f", "f"}, 32,
      [](HalfFP half_fp, int rm) -> std::tuple<double, uint32_t> {
        uint32_t fflags = 0;
        double double_result =
            FpConversionsTestHelper(half_fp).ConvertWithFlags<double>(
                fflags, static_cast<FPRoundingMode>(rm));
        return std::make_tuple(double_result, fflags);
      });
}

// Double precision to half precision conversion.
TEST_F(RV32ZfhInstructionTest, RiscVZfhCvtHd) {
  SetSemanticFunction(&RiscVZfhCvtHd);
  UnaryOpWithFflagsFPTestHelper<HalfFP, double>(
      "fcvt.h.d", instruction_, {"f", "f"}, 32,
      [](double input_double, int rm) -> std::tuple<HalfFP, uint32_t> {
        uint32_t fflags = 0;
        HalfFP half_result = FpConversionsTestHelper(input_double)
                                 .ConvertWithFlags<HalfFP>(
                                     fflags, static_cast<FPRoundingMode>(rm));
        return std::make_tuple(half_result, fflags);
      });
}

// A test to make sure +0 and -0 are converted correctly. Mutation testing
// inspired this test.
TEST_F(RV32ZfhInstructionTest, RiscVZfhCvtSh_strict_zeros) {
  semantic_function_ = &RiscVZfhCvtSh;
  uint32_t expected_p0 = FPTypeInfo<float>::kPosZero;
  uint32_t expected_n0 = FPTypeInfo<float>::kNegZero;
  float actual_p0;
  float actual_n0;
  HalfFP pos_zero = {.value = FPTypeInfo<HalfFP>::kPosZero};
  HalfFP neg_zero = {.value = FPTypeInfo<HalfFP>::kNegZero};

  actual_p0 =
      ConversionHelper<float, HalfFP, kRoundingModeRoundToNearest>(pos_zero);
  actual_n0 =
      ConversionHelper<float, HalfFP, kRoundingModeRoundToNearest>(neg_zero);
  EXPECT_EQ(absl::bit_cast<uint32_t>(actual_p0), expected_p0);
  EXPECT_EQ(absl::bit_cast<uint32_t>(actual_n0), expected_n0);

  actual_p0 =
      ConversionHelper<float, HalfFP, kRoundingModeRoundTowardsZero>(pos_zero);
  actual_n0 =
      ConversionHelper<float, HalfFP, kRoundingModeRoundTowardsZero>(neg_zero);
  EXPECT_EQ(absl::bit_cast<uint32_t>(actual_p0), expected_p0);
  EXPECT_EQ(absl::bit_cast<uint32_t>(actual_n0), expected_n0);

  actual_p0 = ConversionHelper<float, HalfFP, kRoundingModeRoundDown>(pos_zero);
  actual_n0 = ConversionHelper<float, HalfFP, kRoundingModeRoundDown>(neg_zero);
  EXPECT_EQ(absl::bit_cast<uint32_t>(actual_p0), expected_p0);
  EXPECT_EQ(absl::bit_cast<uint32_t>(actual_n0), expected_n0);

  actual_p0 = ConversionHelper<float, HalfFP, kRoundingModeRoundUp>(pos_zero);
  actual_n0 = ConversionHelper<float, HalfFP, kRoundingModeRoundUp>(neg_zero);
  EXPECT_EQ(absl::bit_cast<uint32_t>(actual_p0), expected_p0);
  EXPECT_EQ(absl::bit_cast<uint32_t>(actual_n0), expected_n0);
}

// A test to make sure +0 and -0 are converted correctly. Mutation testing
// inspired this test.
TEST_F(RV32ZfhInstructionTest, RiscVZfhCvtHs_strict_zeros) {
  semantic_function_ = &RiscVZfhCvtHs;
  uint16_t expected_p0 = FPTypeInfo<HalfFP>::kPosZero;
  uint16_t expected_n0 = FPTypeInfo<HalfFP>::kNegZero;
  HalfFP actual_p0;
  HalfFP actual_n0;
  float pos_zero = absl::bit_cast<float>(FPTypeInfo<float>::kPosZero);
  float neg_zero = absl::bit_cast<float>(FPTypeInfo<float>::kNegZero);
  actual_p0 =
      ConversionHelper<HalfFP, float, kRoundingModeRoundToNearest>(pos_zero);
  actual_n0 =
      ConversionHelper<HalfFP, float, kRoundingModeRoundToNearest>(neg_zero);
  EXPECT_EQ(absl::bit_cast<uint16_t>(actual_p0), expected_p0);
  EXPECT_EQ(absl::bit_cast<uint16_t>(actual_n0), expected_n0);

  actual_p0 =
      ConversionHelper<HalfFP, float, kRoundingModeRoundTowardsZero>(pos_zero);
  actual_n0 =
      ConversionHelper<HalfFP, float, kRoundingModeRoundTowardsZero>(neg_zero);
  EXPECT_EQ(absl::bit_cast<uint16_t>(actual_p0), expected_p0);
  EXPECT_EQ(absl::bit_cast<uint16_t>(actual_n0), expected_n0);

  actual_p0 = ConversionHelper<HalfFP, float, kRoundingModeRoundDown>(pos_zero);
  actual_n0 = ConversionHelper<HalfFP, float, kRoundingModeRoundDown>(neg_zero);
  EXPECT_EQ(absl::bit_cast<uint16_t>(actual_p0), expected_p0);
  EXPECT_EQ(absl::bit_cast<uint16_t>(actual_n0), expected_n0);

  actual_p0 = ConversionHelper<HalfFP, float, kRoundingModeRoundUp>(pos_zero);
  actual_n0 = ConversionHelper<HalfFP, float, kRoundingModeRoundUp>(neg_zero);
  EXPECT_EQ(absl::bit_cast<uint16_t>(actual_p0), expected_p0);
  EXPECT_EQ(absl::bit_cast<uint16_t>(actual_n0), expected_n0);
}

// Add half precision values. Generate the expected result by using a natively
// supported float datatype for the operation.
TEST_F(RV32ZfhInstructionTest, RiscVZfhFadd) {
  SetSemanticFunction(&RiscVZfhFadd);
  BinaryOpWithFflagsFPTestHelper<HalfFP, HalfFP, HalfFP>(
      "fadd.h", instruction_, {"f", "f", "f"}, 32,
      [this](HalfFP a, HalfFP b) -> std::tuple<HalfFP, uint32_t> {
        FPRoundingMode rm = rv_fp_->GetRoundingMode();
        uint32_t fflags = 0;
        double a_f =
            FpConversionsTestHelper(a).ConvertWithFlags<double>(fflags);
        double b_f =
            FpConversionsTestHelper(b).ConvertWithFlags<double>(fflags);
        HalfFP result =
            FpConversionsTestHelper(a_f + b_f).ConvertWithFlags<HalfFP>(fflags,
                                                                        rm);
        // inf + -inf, -inf + inf are both invalid.
        if (std::isinf(a_f) && std::isinf(b_f) && a.value != b.value) {
          fflags |= static_cast<uint32_t>(
              mpact::sim::riscv::FPExceptions::kInvalidOp);
          result.value = FPTypeInfo<HalfFP>::kCanonicalNaN;
        }
        return std::make_tuple(result, fflags);
      });
}

// Subtract half precision values. Generate the expected result by using a
// natively supported float datatype for the operation.
TEST_F(RV32ZfhInstructionTest, RiscVZfhFsub) {
  SetSemanticFunction(&RiscVZfhFsub);
  BinaryOpWithFflagsFPTestHelper<HalfFP, HalfFP, HalfFP>(
      "fsub.h", instruction_, {"f", "f", "f"}, 32,
      [this](HalfFP a, HalfFP b) -> std::tuple<HalfFP, uint32_t> {
        FPRoundingMode rm = rv_fp_->GetRoundingMode();
        uint32_t fflags = 0;
        double a_f =
            FpConversionsTestHelper(a).ConvertWithFlags<double>(fflags);
        double b_f =
            FpConversionsTestHelper(b).ConvertWithFlags<double>(fflags);
        HalfFP result =
            FpConversionsTestHelper(a_f - b_f).ConvertWithFlags<HalfFP>(fflags,
                                                                        rm);
        // inf - inf and -inf - -inf are both invalid.
        if (std::isinf(a_f) && std::isinf(b_f) && a.value == b.value) {
          fflags |= static_cast<uint32_t>(
              mpact::sim::riscv::FPExceptions::kInvalidOp);
          result.value = FPTypeInfo<HalfFP>::kCanonicalNaN;
        }
        return std::make_tuple(result, fflags);
      });
}

// Multiply half precision values. Generate the expected result by using a
// natively supported float datatype for the operation.
TEST_F(RV32ZfhInstructionTest, RiscVZfhFmul) {
  SetSemanticFunction(&RiscVZfhFmul);
  BinaryOpWithFflagsFPTestHelper<HalfFP, HalfFP, HalfFP>(
      "fmul.h", instruction_, {"f", "f", "f"}, 32,
      [this](HalfFP a, HalfFP b) -> std::tuple<HalfFP, uint32_t> {
        FPRoundingMode rm = rv_fp_->GetRoundingMode();
        uint32_t fflags = 0;
        double a_f =
            FpConversionsTestHelper(a).ConvertWithFlags<double>(fflags);
        double b_f =
            FpConversionsTestHelper(b).ConvertWithFlags<double>(fflags);
        HalfFP result =
            FpConversionsTestHelper(a_f * b_f).ConvertWithFlags<HalfFP>(fflags,
                                                                        rm);
        // Multiplying infinity and zero is invalid.
        if ((std::isinf(a_f) && b_f == 0) || (a_f == 0 && std::isinf(b_f))) {
          fflags |= static_cast<uint32_t>(
              mpact::sim::riscv::FPExceptions::kInvalidOp);
          result.value = FPTypeInfo<HalfFP>::kCanonicalNaN;
        }
        return std::make_tuple(result, fflags);
      });
}

// Divide half precision values. Generate the expected result by using a
// natively supported float datatype for the operation.
TEST_F(RV32ZfhInstructionTest, RiscVZfhFdiv) {
  SetSemanticFunction(&RiscVZfhFdiv);
  BinaryOpWithFflagsFPTestHelper<HalfFP, HalfFP, HalfFP>(
      "fdiv.h", instruction_, {"f", "f", "f"}, 32,
      [this](HalfFP a, HalfFP b) -> std::tuple<HalfFP, uint32_t> {
        FPRoundingMode rm = rv_fp_->GetRoundingMode();
        uint32_t fflags = 0;
        double a_f =
            FpConversionsTestHelper(a).ConvertWithFlags<double>(fflags);
        double b_f =
            FpConversionsTestHelper(b).ConvertWithFlags<double>(fflags);
        HalfFP result =
            FpConversionsTestHelper(a_f / b_f).ConvertWithFlags<HalfFP>(fflags,
                                                                        rm);
        if ((a_f == 0 && b_f == 0) || (std::isinf(a_f) && std::isinf(b_f))) {
          // 0 / 0 and inf / inf are both invalid.
          fflags |= static_cast<uint32_t>(
              mpact::sim::riscv::FPExceptions::kInvalidOp);
          result.value = FPTypeInfo<HalfFP>::kCanonicalNaN;
        } else if (!FPTypeInfo<HalfFP>::IsNaN(a) &&
                   !FPTypeInfo<HalfFP>::IsInf(a) &&
                   (b.value == FPTypeInfo<HalfFP>::kPosZero ||
                    b.value == FPTypeInfo<HalfFP>::kNegZero)) {
          // Dividing by zero requires an exception for non-NaN dividend values.
          result.value = ((a.value ^ b.value) & FPTypeInfo<HalfFP>::kNegZero) |
                         (FPTypeInfo<HalfFP>::kPosInf);
          fflags |= static_cast<uint32_t>(
              mpact::sim::riscv::FPExceptions::kDivByZero);
        }
        return std::make_tuple(result, fflags);
      });
}

// Find the minimum of two half precision values. Generate the expected result
// by using a natively supported float datatype for the operation.
TEST_F(RV32ZfhInstructionTest, RiscVZfhFmin) {
  SetSemanticFunction(&RiscVZfhFmin);
  BinaryOpWithFflagsFPTestHelper<HalfFP, HalfFP, HalfFP>(
      "fmin.h", instruction_, {"f", "f", "f"}, 32,
      [this](HalfFP a, HalfFP b) -> std::tuple<HalfFP, uint32_t> {
        FPRoundingMode rm = rv_fp_->GetRoundingMode();
        uint32_t fflags = 0;
        double a_f =
            FpConversionsTestHelper(a).ConvertWithFlags<double>(fflags);
        double b_f =
            FpConversionsTestHelper(b).ConvertWithFlags<double>(fflags);
        double min_f = 0;
        if (a_f == 0 && b_f == 0 && (a.value != b.value)) {
          // Special case for -0 vs +0.
          min_f = absl::bit_cast<double>(FPTypeInfo<double>::kNegZero);
        } else if (std::isnan(a_f) && !std::isnan(b_f)) {
          min_f = b_f;  // pick the non-NaN value
        } else if (!std::isnan(a_f) && std::isnan(b_f)) {
          min_f = a_f;  // pick the non-NaN value
        } else if (std::isnan(a_f) && std::isnan(b_f)) {
          min_f = absl::bit_cast<double>(FPTypeInfo<double>::kCanonicalNaN);
        } else if (std::isinf(a_f) && !std::isinf(b_f)) {
          min_f = a_f < 0 ? a_f : b_f;  // min(+/-inf, x)
        } else if (!std::isinf(a_f) && std::isinf(b_f)) {
          min_f = b_f < 0 ? b_f : a_f;  // min(x, +/-inf)
        } else {
          if (a_f < b_f) {
            min_f = a_f;
          } else {
            min_f = b_f;
          }
        }
        HalfFP result =
            FpConversionsTestHelper(min_f).ConvertWithFlags<HalfFP>(fflags, rm);
        return std::make_tuple(result, fflags);
      });
}

// Find the maximum of two half precision values. Generate the expected result
// by using a natively supported float datatype for the operation.
TEST_F(RV32ZfhInstructionTest, RiscVZfhFmax) {
  SetSemanticFunction(&RiscVZfhFmax);
  BinaryOpWithFflagsFPTestHelper<HalfFP, HalfFP, HalfFP>(
      "fmax.h", instruction_, {"f", "f", "f"}, 32,
      [this](HalfFP a, HalfFP b) -> std::tuple<HalfFP, uint32_t> {
        FPRoundingMode rm = rv_fp_->GetRoundingMode();
        uint32_t fflags = 0;
        double a_f =
            FpConversionsTestHelper(a).ConvertWithFlags<double>(fflags);
        double b_f =
            FpConversionsTestHelper(b).ConvertWithFlags<double>(fflags);
        double max_f = 0;
        if (a_f == 0 && b_f == 0 && (a.value != b.value)) {
          // Special case for -0 vs +0.
          max_f = absl::bit_cast<double>(FPTypeInfo<double>::kPosZero);
        } else if (std::isnan(a_f) && !std::isnan(b_f)) {
          max_f = b_f;  // pick the non-NaN value
        } else if (!std::isnan(a_f) && std::isnan(b_f)) {
          max_f = a_f;  // pick the non-NaN value
        } else if (std::isnan(a_f) && std::isnan(b_f)) {
          max_f = absl::bit_cast<double>(FPTypeInfo<double>::kCanonicalNaN);
        } else if (std::isinf(a_f) && !std::isinf(b_f)) {
          max_f = a_f > 0 ? a_f : b_f;  // max(+/-inf, x)
        } else if (!std::isinf(a_f) && std::isinf(b_f)) {
          max_f = b_f > 0 ? b_f : a_f;  // max(x, +/-inf)
        } else {
          if (a_f > b_f) {
            max_f = a_f;
          } else {
            max_f = b_f;
          }
        }
        HalfFP result =
            FpConversionsTestHelper(max_f).ConvertWithFlags<HalfFP>(fflags, rm);
        return std::make_tuple(result, fflags);
      });
}

// Test sign injection for half precision values.
TEST_F(RV32ZfhInstructionTest, RiscVZfhFsgnj) {
  SetSemanticFunction(&RiscVZfhFsgnj);
  BinaryOpWithFflagsFPTestHelper<HalfFP, HalfFP, HalfFP>(
      "fsgnj.h", instruction_, {"f", "f", "f"}, 32,
      [](HalfFP a, HalfFP b) -> std::tuple<HalfFP, uint32_t> {
        HalfFP result{.value = 0};
        result.value |= a.value & (FPTypeInfo<HalfFP>::kExpMask |
                                   FPTypeInfo<HalfFP>::kSigMask);
        result.value |= b.value & ~(FPTypeInfo<HalfFP>::kExpMask |
                                    FPTypeInfo<HalfFP>::kSigMask);
        return std::make_tuple(result, 0);
      });
}

// Test sign injection for half precision values with the opposite sign bit.
TEST_F(RV32ZfhInstructionTest, RiscVZfhFsgnjn) {
  SetSemanticFunction(&RiscVZfhFsgnjn);
  BinaryOpWithFflagsFPTestHelper<HalfFP, HalfFP, HalfFP>(
      "fsgnjn.h", instruction_, {"f", "f", "f"}, 32,
      [](HalfFP a, HalfFP b) -> std::tuple<HalfFP, uint32_t> {
        HalfFP result{.value = 0};
        result.value |= a.value & (FPTypeInfo<HalfFP>::kExpMask |
                                   FPTypeInfo<HalfFP>::kSigMask);
        result.value |= (~b.value) & ~(FPTypeInfo<HalfFP>::kExpMask |
                                       FPTypeInfo<HalfFP>::kSigMask);
        return std::make_tuple(result, 0);
      });
}

// Test sign injection for half precision values with the xor sign bit..
TEST_F(RV32ZfhInstructionTest, RiscVZfhFsgnjx) {
  SetSemanticFunction(&RiscVZfhFsgnjx);
  BinaryOpWithFflagsFPTestHelper<HalfFP, HalfFP, HalfFP>(
      "fsgnjn.h", instruction_, {"f", "f", "f"}, 32,
      [](HalfFP a, HalfFP b) -> std::tuple<HalfFP, uint32_t> {
        HalfFP result{.value = 0};
        result.value |= a.value & (FPTypeInfo<HalfFP>::kExpMask |
                                   FPTypeInfo<HalfFP>::kSigMask);
        result.value |= (a.value ^ b.value) & ~(FPTypeInfo<HalfFP>::kExpMask |
                                                FPTypeInfo<HalfFP>::kSigMask);
        return std::make_tuple(result, 0);
      });
}

// Test square root for half precision values.
TEST_F(RV32ZfhInstructionTest, RiscVZfhFsqrt) {
  SetSemanticFunction(&RiscVZfhFsqrt);
  UnaryOpWithFflagsFPTestHelper<HalfFP, HalfFP>(
      "fsqrt.h", instruction_, {"f", "f"}, 32,
      [](HalfFP input_half, int rm) -> std::tuple<HalfFP, uint32_t> {
        uint32_t fflags = 0;
        double input_double_f = FpConversionsTestHelper(input_half)
                                    .ConvertWithFlags<double>(fflags);
        HalfFP result;
        if (FPTypeInfo<HalfFP>::IsNaN(input_half)) {
          result.value = FPTypeInfo<HalfFP>::kCanonicalNaN;
          if (!FPTypeInfo<HalfFP>::IsQNaN(input_half)) {
            fflags |= static_cast<uint32_t>(
                mpact::sim::riscv::FPExceptions::kInvalidOp);
          }
        } else if (std::isinf(input_double_f) && input_double_f > 0) {
          result.value = FPTypeInfo<HalfFP>::kPosInf;
        } else if (input_double_f < 0) {
          result.value = FPTypeInfo<HalfFP>::kCanonicalNaN;
          fflags |= static_cast<uint32_t>(
              mpact::sim::riscv::FPExceptions::kInvalidOp);
        } else {
          result = FpConversionsTestHelper(std::sqrt(input_double_f))
                       .ConvertWithFlags<HalfFP>(
                           fflags, static_cast<FPRoundingMode>(rm));
        }
        return std::make_tuple(result, fflags);
      });
}

// Test conversion from signed 32 bit integer to half precision.
TEST_F(RV32ZfhInstructionTest, RiscVZfhCvtHw) {
  SetSemanticFunction(&RiscVZfhCvtHw);
  UnaryOpWithFflagsMixedTestHelper<RVFpRegister, RV32Register, HalfFP, int32_t>(
      "fcvt.h.w", instruction_, {"x", "f"}, 32,
      [](int32_t input_int, int rm) -> std::tuple<HalfFP, uint32_t> {
        uint32_t fflags = 0;
        HalfFP result = FpConversionsTestHelper(static_cast<double>(input_int))
                            .ConvertWithFlags<HalfFP>(
                                fflags, static_cast<FPRoundingMode>(rm));
        return std::make_tuple(result, fflags);
      });
}

// Test conversion from half precision to signed 32 bit integer.
TEST_F(RV32ZfhInstructionTest, RiscVZfhCvtWh) {
  SetSemanticFunction(&RiscVZfhCvtWh);
  UnaryOpWithFflagsMixedTestHelper<RV32Register, RVFpRegister, int32_t, HalfFP>(
      "fcvt.w.h", instruction_, {"f", "x"}, 32,
      [this](HalfFP input, int rm) -> std::tuple<int32_t, uint32_t> {
        uint32_t fflags = 0;
        double input_double =
            FpConversionsTestHelper(input).ConvertWithFlags<double>(fflags);
        const int32_t val =
            RoundToInteger<double, int32_t>(input_double, rm, fflags);
        return std::make_tuple(val, fflags);
      });
}

// Test conversion from unsigned 32 bit integer to half precision.
TEST_F(RV32ZfhInstructionTest, RiscVZfhCvtHwu) {
  SetSemanticFunction(&RiscVZfhCvtHwu);
  UnaryOpWithFflagsMixedTestHelper<RVFpRegister, RV32Register, HalfFP,
                                   uint32_t>(
      "fcvt.h.wu", instruction_, {"x", "f"}, 32,
      [](uint32_t input_int, int rm) -> std::tuple<HalfFP, uint32_t> {
        uint32_t fflags = 0;
        HalfFP result = FpConversionsTestHelper(static_cast<double>(input_int))
                            .ConvertWithFlags<HalfFP>(
                                fflags, static_cast<FPRoundingMode>(rm));
        return std::make_tuple(result, fflags);
      });
}

// Test conversion from half precision to unsigned 32 bit integer.
TEST_F(RV32ZfhInstructionTest, RiscVZfhCvtWuh) {
  SetSemanticFunction(&RiscVZfhCvtWuh);
  UnaryOpWithFflagsMixedTestHelper<RV32Register, RVFpRegister, uint32_t,
                                   HalfFP>(
      "fcvt.wu.h", instruction_, {"f", "x"}, 32,
      [this](HalfFP input, int rm) -> std::tuple<uint32_t, uint32_t> {
        uint32_t fflags = 0;
        double input_double =
            FpConversionsTestHelper(input).ConvertWithFlags<double>(fflags);
        const uint32_t val =
            RoundToInteger<double, uint32_t>(input_double, rm, fflags);
        return std::make_tuple(val, fflags);
      });
}

// Test equality comparison for half precision values.
TEST_F(RV32ZfhInstructionTest, RiscVZfhFcmpeq) {
  SetSemanticFunction(&RiscVZfhFcmpeq);
  BinaryOpWithFflagsFPTestHelper<uint64_t, HalfFP, HalfFP>(
      "feq.h", instruction_, {"f", "f", "x"}, 32,
      [](HalfFP a, HalfFP b) -> std::tuple<uint64_t, uint32_t> {
        uint32_t fflags = 0;
        double a_f =
            FpConversionsTestHelper(a).ConvertWithFlags<double>(fflags);
        double b_f =
            FpConversionsTestHelper(b).ConvertWithFlags<double>(fflags);
        uint64_t result = a_f == b_f ? 1 : 0;
        if (std::isnan(a_f) || std::isnan(b_f)) {
          result = 0;
        }
        return std::make_tuple(result, fflags);
      });
}

// Test less than comparison for half precision values.
TEST_F(RV32ZfhInstructionTest, RiscVZfhFcmplt) {
  SetSemanticFunction(&RiscVZfhFcmplt);
  BinaryOpWithFflagsFPTestHelper<uint64_t, HalfFP, HalfFP>(
      "flt.h", instruction_, {"f", "f", "x"}, 32,
      [](HalfFP a, HalfFP b) -> std::tuple<uint64_t, uint32_t> {
        uint32_t fflags = 0;
        double a_f =
            FpConversionsTestHelper(a).ConvertWithFlags<double>(fflags);
        double b_f =
            FpConversionsTestHelper(b).ConvertWithFlags<double>(fflags);
        uint64_t result = a_f < b_f ? 1 : 0;
        if (std::isnan(a_f) || std::isnan(b_f)) {
          result = 0;
          // LT is a signaling comparison, so the invalid operation flag is
          // set.
          fflags |= static_cast<uint32_t>(
              mpact::sim::riscv::FPExceptions::kInvalidOp);
        }
        return std::make_tuple(result, fflags);
      });
}

// Test less than or equal to comparison for half precision values.
TEST_F(RV32ZfhInstructionTest, RiscVZfhFcmple) {
  SetSemanticFunction(&RiscVZfhFcmple);
  BinaryOpWithFflagsFPTestHelper<uint64_t, HalfFP, HalfFP>(
      "fle.h", instruction_, {"f", "f", "x"}, 32,
      [](HalfFP a, HalfFP b) -> std::tuple<uint64_t, uint32_t> {
        uint32_t fflags = 0;
        double a_f =
            FpConversionsTestHelper(a).ConvertWithFlags<double>(fflags);
        double b_f =
            FpConversionsTestHelper(b).ConvertWithFlags<double>(fflags);
        uint64_t result = a_f <= b_f ? 1 : 0;
        if (std::isnan(a_f) || std::isnan(b_f)) {
          result = 0;
          // LE is a signaling comparison, so the invalid operation flag is
          // set.
          fflags |= static_cast<uint32_t>(
              mpact::sim::riscv::FPExceptions::kInvalidOp);
        }
        return std::make_tuple(result, fflags);
      });
}

// Test classification of half precision values.
TEST_F(RV32ZfhInstructionTest, RiscVZfhFclass) {
  SetSemanticFunction(&RiscVZfhFclass);
  UnaryOpWithFflagsMixedTestHelper<RV32Register, RVFpRegister, uint32_t,
                                   HalfFP>(
      "fclass.h", instruction_, {"f", "x"}, 32,
      [](HalfFP input, int rm) -> std::tuple<uint32_t, uint32_t> {
        uint16_t sign_mask =
            ~(FPTypeInfo<HalfFP>::kExpMask | FPTypeInfo<HalfFP>::kSigMask);
        uint16_t sign = input.value & sign_mask;
        int shift = -1;

        switch (input.value & FPTypeInfo<HalfFP>::kExpMask) {
          case 0:
            if (input.value & FPTypeInfo<HalfFP>::kSigMask) {
              if (sign) {
                shift = 2;  // Negative subnormal
              } else {
                shift = 5;  // Positive subnormal
              }
            } else {
              if (sign) {
                shift = 3;  // Negative zero
              } else {
                shift = 4;  // Positive zero
              }
            }
            break;
          case FPTypeInfo<HalfFP>::kExpMask:
            if (input.value & FPTypeInfo<HalfFP>::kSigMask) {
              if (FPTypeInfo<HalfFP>::IsQNaN(input)) {
                shift = 9;  // Quiet NaN
              } else {
                shift = 8;  // Signaling NaN
              }
            } else {  // Inf
              if (sign) {
                shift = 0;  // Negative infinity
              } else {
                shift = 7;  // Positive infinity
              }
            }
            break;
          default:
            if (sign) {
              shift = 1;  // Negative normal
            } else {
              shift = 6;  // Positive normal
            }
            break;
        }
        EXPECT_GE(shift, 0) << "The test didn't set the expected result.";
        return std::make_tuple(1 << shift, 0);
      });
}

class RV64ZfhInstructionTest : public RVZfhInstructionTestBase {};

// Move half precision from a float register to an integer register. The IEEE754
// encoding is preserved in the integer register.
TEST_F(RV64ZfhInstructionTest, RiscVZfhFMvxh) {
  SetSemanticFunction(&mpact::sim::riscv::RV64::RiscVZfhFMvxh);
  UnaryOpFPTestHelper<uint64_t, HalfFP>(
      "fmv.x.h", instruction_, {"f", "x"}, 32, [](HalfFP half_fp) -> uint64_t {
        bool sign = 1 & (half_fp.value >> (FPTypeInfo<HalfFP>::kBitSize - 1));
        // Fill the upper XLEN-16 bits with the sign bit as per the spec.
        uint64_t result = sign ? 0xFFFF'FFFF'FFFF'0000 : 0;
        result |= static_cast<uint64_t>(half_fp.value);
        return result;
      });
}

// Test the FP16 load instruction. The semantic functions should match the isa
// file.
TEST_F(RV64ZfhInstructionTest, RiscVFlh) {
  SetSemanticFunction(&RiscVILhu);
  SetChildInstruction();
  SetChildSemanticFunction(&RiscVZfhFlhChild);

  SetupMemory<uint64_t, uint16_t>(0xFF, 0xBEEF);

  HalfFP observed_val =
      LoadHalfHelper<HalfFP, RV64Register>(/* base */ 0x0, /* offset */ 0xFF);
  EXPECT_EQ(observed_val.value, 0xBEEF);
}

// Test the FP16 load instruction. When looking at the register contents as a
// float, it should be NaN.
TEST_F(RV64ZfhInstructionTest, RiscVFlh_float_nanbox) {
  SetSemanticFunction(&RiscVILhu);
  SetChildInstruction();
  SetChildSemanticFunction(&RiscVZfhFlhChild);

  SetupMemory<uint64_t, uint16_t>(0xFF, 0xBEEF);

  float observed_val =
      LoadHalfHelper<float, RV64Register>(/* base */ 0xFF, /* offset */ 0);
  EXPECT_TRUE(std::isnan(observed_val));
}

// Test the FP16 load instruction. When looking at the register contents as a
// double, it should be NaN.
TEST_F(RV64ZfhInstructionTest, RiscVFlh_double_nanbox) {
  SetSemanticFunction(&RiscVILhu);
  SetChildInstruction();
  SetChildSemanticFunction(&RiscVZfhFlhChild);

  SetupMemory<uint64_t, uint16_t>(0xFF, 0xBEEF);

  double observed_val = LoadHalfHelper<double, RV64Register>(
      /* base */ 0x0100, /* offset */ -1);
  EXPECT_TRUE(std::isnan(observed_val));
}

}  // namespace
