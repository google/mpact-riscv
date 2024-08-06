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

#ifndef MPACT_RISCV_RISCV_TEST_RISCV_FP_TEST_BASE_H_
#define MPACT_RISCV_RISCV_TEST_RISCV_FP_TEST_BASE_H_

#include <cmath>
#include <cstdint>
#include <functional>
#include <ios>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "riscv/riscv_fp_host.h"
#include "riscv/riscv_fp_info.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {
namespace test {

using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::FPRoundingMode;
using ::mpact::sim::util::FlatDemandMemory;

constexpr int kTestValueLength = 256;

constexpr uint32_t kInstAddress = 0x1000;
constexpr uint32_t kDataLoadAddress = 0x1'0000;
constexpr uint32_t kDataStoreAddress = 0x2'0000;

// Templated helper structs to provide information about floating point types.
template <typename T>
struct FPTypeInfo {
  using IntType = T;
  static const int kBitSize = 8 * sizeof(T);
  static const int kExpSize = 0;
  static const int kSigSize = 0;
  static bool IsNaN(T value) { return false; }
  static constexpr IntType kQNaN = 0;
  static constexpr IntType kSNaN = 0;
  static constexpr IntType kPosInf = std::numeric_limits<T>::max();
  static constexpr IntType kNegInf = std::numeric_limits<T>::min();
  static constexpr IntType kPosZero = 0;
  static constexpr IntType kNegZero = 0;
  static constexpr IntType kPosDenorm = 0;
  static constexpr IntType kNegDenorm = 0;
};

template <>
struct FPTypeInfo<float> {
  using T = float;
  using IntType = uint32_t;
  static const int kExpBias = 127;
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
  static bool IsQNaN(T value) {
    IntType uint_val = *reinterpret_cast<IntType *>(&value);
    return IsNaN(value) && (((1ULL << (kSigSize - 1)) & uint_val) != 0);
  }
};

template <>
struct FPTypeInfo<double> {
  using T = double;
  using IntType = uint64_t;
  static const int kExpBias = 1023;
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
  static bool IsQNaN(T value) {
    IntType uint_val = *reinterpret_cast<IntType *>(&value);
    return IsNaN(value) && (((1ULL << (kSigSize - 1)) & uint_val) != 0);
  }
};

// Templated helper function for classifying fp numbers.
template <typename T>
typename FPTypeInfo<T>::IntType VfclassVHelper(T val) {
  auto fp_class = fpclassify(val);
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
  UInt u_op = *reinterpret_cast<UInt *>(&op);
  UInt u_reg = *reinterpret_cast<UInt *>(&reg);
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
    EXPECT_THAT(reg, testing::NanSensitiveFloatNear(op, delta))
        << str << "  op: " << std::hex << u_op << "  reg: " << std::hex
        << u_reg;
  } else {
    EXPECT_THAT(reg, testing::NanSensitiveFloatEq(op))
        << str << "  op: " << std::hex << u_op << "  reg: " << std::hex
        << u_reg;
  }
}

template <>
inline void FPCompare<double>(double op, double reg, int delta_position,
                              absl::string_view str) {
  using T = double;
  using UInt = typename FPTypeInfo<T>::IntType;
  UInt u_op = *reinterpret_cast<UInt *>(&op);
  UInt u_reg = *reinterpret_cast<UInt *>(&reg);
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
    EXPECT_THAT(reg, testing::NanSensitiveDoubleNear(op, delta))
        << str << "  op: " << std::hex << u_op << "  reg: " << std::hex
        << u_reg;
  } else {
    EXPECT_THAT(reg, testing::NanSensitiveDoubleEq(op))
        << str << "  op: " << std::hex << u_op << "  reg: " << std::hex
        << u_reg;
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

class RiscVFPInstructionTestBase : public testing::Test {
 public:
  RiscVFPInstructionTestBase() {
    memory_ = new FlatDemandMemory(0);
    state_ = new RiscVState("test", RiscVXlen::RV32, memory_);
    rv_fp_ = new mpact::sim::riscv::RiscVFPState(state_->csr_set(), state_);
    state_->set_rv_fp(rv_fp_);
    instruction_ = new Instruction(kInstAddress, state_);
    instruction_->set_size(4);
    child_instruction_ = new Instruction(kInstAddress, state_);
    child_instruction_->set_size(4);
    // Initialize a portion of memory with a known pattern.
    auto *db = state_->db_factory()->Allocate(8192);
    auto span = db->Get<uint8_t>();
    for (int i = 0; i < 8192; i++) {
      span[i] = i & 0xff;
    }
    memory_->Store(kDataLoadAddress - 4096, db);
    db->DecRef();
    for (int i = 1; i < 32; i++) {
      xreg_[i] = state_->GetRegister<RV32Register>(absl::StrCat("x", i)).first;
    }
    for (int i = 1; i < 32; i++) {
      freg_[i] = state_->GetRegister<RVFpRegister>(absl::StrCat("f", i)).first;
    }
    for (int i = 1; i < 32; i++) {
      dreg_[i] = state_->GetRegister<RVFpRegister>(absl::StrCat("d", i)).first;
    }
  }
  ~RiscVFPInstructionTestBase() override {
    delete rv_fp_;
    state_->set_rv_fp(nullptr);
    delete state_;
    instruction_->DecRef();
    child_instruction_->DecRef();
    delete memory_;
  }

  // Clear the instruction instance and allocate a new one.
  void ResetInstruction() {
    instruction_->DecRef();
    instruction_ = new Instruction(kInstAddress, state_);
    instruction_->set_size(4);
  }

  // Creates source and destination scalar register operands for the registers
  // named in the two vectors and append them to the given instruction.
  void AppendRegisterOperands(Instruction *inst,
                              const std::vector<std::string> &sources,
                              const std::vector<std::string> &destinations) {
    for (auto &reg_name : sources) {
      auto *reg = state_->GetRegister<RV32Register>(reg_name).first;
      inst->AppendSource(reg->CreateSourceOperand());
    }
    for (auto &reg_name : destinations) {
      auto *reg = state_->GetRegister<RV32Register>(reg_name).first;
      inst->AppendDestination(reg->CreateDestinationOperand(0));
    }
  }

  // Creates source and destination scalar register operands for the registers
  // named in the two vectors and append them to the default instruction.
  void AppendRegisterOperands(const std::vector<std::string> &sources,
                              const std::vector<std::string> &destinations) {
    AppendRegisterOperands(instruction_, sources, destinations);
  }

  // named register and sets it to the corresponding value.
  template <typename T, typename RegisterType = RV32Register>
  void SetRegisterValues(
      const std::vector<std::tuple<std::string, const T>> &values) {
    for (auto &[reg_name, value] : values) {
      auto *reg = state_->GetRegister<RegisterType>(reg_name).first;
      auto *db =
          state_->db_factory()->Allocate<typename RegisterType::ValueType>(1);
      db->template Set<T>(0, value);
      reg->SetDataBuffer(db);
      db->DecRef();
    }
  }

  // Initializes the semantic function of the instruction object.
  void SetSemanticFunction(Instruction *inst,
                           Instruction::SemanticFunction fcn) {
    inst->set_semantic_function(fcn);
  }

  // Initializes the semantic function for the default instruction.
  void SetSemanticFunction(Instruction::SemanticFunction fcn) {
    instruction_->set_semantic_function(fcn);
  }

  // Sets the default child instruction as the child of the default instruction.
  void SetChildInstruction() { instruction_->AppendChild(child_instruction_); }

  // Initializes the semantic function for the default child instruction.
  void SetChildSemanticFunction(Instruction::SemanticFunction fcn) {
    child_instruction_->set_semantic_function(fcn);
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

  template <typename R, typename LHS>
  void UnaryOpFPTestHelper(absl::string_view name, Instruction *inst,
                           absl::Span<const absl::string_view> reg_prefixes,
                           int delta_position,
                           std::function<R(LHS)> operation) {
    using LhsRegisterType = RVFpRegister;
    using DestRegisterType = RVFpRegister;
    LHS lhs_values[kTestValueLength];
    auto lhs_span = absl::Span<LHS>(lhs_values);
    const std::string kR1Name = absl::StrCat(reg_prefixes[0], 1);
    const std::string kRdName = absl::StrCat(reg_prefixes[1], 5);
    // This is used for the rounding mode operand.
    const std::string kRmName = absl::StrCat("x", 10);
    AppendRegisterOperands({kR1Name, kRmName}, {kRdName});
    FillArrayWithRandomFPValues<LHS>(lhs_span);
    using LhsInt = typename FPTypeInfo<LHS>::IntType;
    *reinterpret_cast<LhsInt *>(&lhs_span[0]) = FPTypeInfo<LHS>::kQNaN;
    *reinterpret_cast<LhsInt *>(&lhs_span[1]) = FPTypeInfo<LHS>::kSNaN;
    *reinterpret_cast<LhsInt *>(&lhs_span[2]) = FPTypeInfo<LHS>::kPosInf;
    *reinterpret_cast<LhsInt *>(&lhs_span[3]) = FPTypeInfo<LHS>::kNegInf;
    *reinterpret_cast<LhsInt *>(&lhs_span[4]) = FPTypeInfo<LHS>::kPosZero;
    *reinterpret_cast<LhsInt *>(&lhs_span[5]) = FPTypeInfo<LHS>::kNegZero;
    *reinterpret_cast<LhsInt *>(&lhs_span[6]) = FPTypeInfo<LHS>::kPosDenorm;
    *reinterpret_cast<LhsInt *>(&lhs_span[7]) = FPTypeInfo<LHS>::kNegDenorm;
    for (int i = 0; i < kTestValueLength; i++) {
      SetRegisterValues<LHS, LhsRegisterType>({{kR1Name, lhs_span[i]}});

      for (int rm : {0, 1, 2, 3, 4}) {
        rv_fp_->SetRoundingMode(static_cast<FPRoundingMode>(rm));
        SetRegisterValues<int, RV32Register>({{kRmName, rm}});
        SetRegisterValues<R, DestRegisterType>({{kRdName, 0}});

        inst->Execute(nullptr);

        R op_val;
        {
          ScopedFPStatus set_fpstatus(rv_fp_->host_fp_interface());
          op_val = operation(lhs_span[i]);
        }
        auto reg_val = state_->GetRegister<DestRegisterType>(kRdName)
                           .first->data_buffer()
                           ->template Get<R>(0);
        FPCompare<R>(op_val, reg_val, delta_position,
                     absl::StrCat(name, "  ", i, ": ", lhs_span[i]));
      }
    }
  }

  // Tester for unary instructions that produce an exception flag value.
  template <typename R, typename LHS>
  void UnaryOpWithFflagsFPTestHelper(
      absl::string_view name, Instruction *inst,
      absl::Span<const absl::string_view> reg_prefixes, int delta_position,
      std::function<std::tuple<R, uint32_t>(LHS, uint32_t)> operation) {
    using LhsRegisterType = RVFpRegister;
    using DestRegisterType = RVFpRegister;
    using LhsInt = typename FPTypeInfo<LHS>::IntType;
    using RInt = typename FPTypeInfo<R>::IntType;
    LHS lhs_values[kTestValueLength];
    auto lhs_span = absl::Span<LHS>(lhs_values);
    const std::string kR1Name = absl::StrCat(reg_prefixes[0], 1);
    const std::string kRdName = absl::StrCat(reg_prefixes[1], 5);
    // This is used for the rounding mode operand.
    const std::string kRmName = absl::StrCat("x", 10);
    AppendRegisterOperands({kR1Name, kRmName}, {kRdName});
    auto *flag_op = rv_fp_->fflags()->CreateSetDestinationOperand(0, "fflags");
    instruction_->AppendDestination(flag_op);
    FillArrayWithRandomFPValues<LHS>(lhs_span);
    *reinterpret_cast<LhsInt *>(&lhs_span[0]) = FPTypeInfo<LHS>::kQNaN;
    *reinterpret_cast<LhsInt *>(&lhs_span[1]) = FPTypeInfo<LHS>::kSNaN;
    *reinterpret_cast<LhsInt *>(&lhs_span[2]) = FPTypeInfo<LHS>::kPosInf;
    *reinterpret_cast<LhsInt *>(&lhs_span[3]) = FPTypeInfo<LHS>::kNegInf;
    *reinterpret_cast<LhsInt *>(&lhs_span[4]) = FPTypeInfo<LHS>::kPosZero;
    *reinterpret_cast<LhsInt *>(&lhs_span[5]) = FPTypeInfo<LHS>::kNegZero;
    *reinterpret_cast<LhsInt *>(&lhs_span[6]) = FPTypeInfo<LHS>::kPosDenorm;
    *reinterpret_cast<LhsInt *>(&lhs_span[7]) = FPTypeInfo<LHS>::kNegDenorm;
    for (int i = 0; i < kTestValueLength; i++) {
      SetRegisterValues<LHS, LhsRegisterType>({{kR1Name, lhs_span[i]}});

      for (int rm : {0, 1, 2, 3, 4}) {
        rv_fp_->SetRoundingMode(static_cast<FPRoundingMode>(rm));
        rv_fp_->fflags()->Write(static_cast<uint32_t>(0));
        SetRegisterValues<int, RV32Register>({{kRmName, rm}, {}});
        SetRegisterValues<R, DestRegisterType>({{kRdName, 0}});

        inst->Execute(nullptr);
        auto fflags = rv_fp_->fflags()->GetUint32();

        R op_val;
        uint32_t flag;
        {
          ScopedFPRoundingMode scoped_rm(rv_fp_->host_fp_interface(), rm);
          std::tie(op_val, flag) = operation(lhs_span[i], rm);
        }

        auto reg_val = state_->GetRegister<DestRegisterType>(kRdName)
                           .first->data_buffer()
                           ->template Get<R>(0);
        FPCompare<R>(
            op_val, reg_val, delta_position,
            absl::StrCat(name, "  ", i, ": ", lhs_span[i], " rm: ", rm));
        auto lhs_uint = *reinterpret_cast<LhsInt *>(&lhs_span[i]);
        auto op_val_uint = *reinterpret_cast<RInt *>(&op_val);
        EXPECT_EQ(flag, fflags)
            << name << "(" << lhs_span[i] << ")  " << std::hex << name << "(0x"
            << lhs_uint << ") == " << op_val << std::hex << "  0x"
            << op_val_uint << " rm: " << rm;
      }
    }
  }

  // Test helper for binary fp instructions.
  template <typename R, typename LHS, typename RHS>
  void BinaryOpFPTestHelper(absl::string_view name, Instruction *inst,
                            absl::Span<const absl::string_view> reg_prefixes,
                            int delta_position,
                            std::function<R(LHS, RHS)> operation) {
    using LhsRegisterType = RVFpRegister;
    using RhsRegisterType = RVFpRegister;
    using DestRegisterType = RVFpRegister;
    LHS lhs_values[kTestValueLength];
    RHS rhs_values[kTestValueLength];
    auto lhs_span = absl::Span<LHS>(lhs_values);
    auto rhs_span = absl::Span<RHS>(rhs_values);
    const std::string kR1Name = absl::StrCat(reg_prefixes[0], 1);
    const std::string kR2Name = absl::StrCat(reg_prefixes[1], 2);
    const std::string kRdName = absl::StrCat(reg_prefixes[2], 5);
    // This is used for the rounding mode operand.
    const std::string kRmName = absl::StrCat("x", 10);
    AppendRegisterOperands({kR1Name, kR2Name, kRmName}, {kRdName});
    auto *flag_op = rv_fp_->fflags()->CreateSetDestinationOperand(0, "fflags");
    instruction_->AppendDestination(flag_op);
    FillArrayWithRandomFPValues<LHS>(lhs_span);
    FillArrayWithRandomFPValues<RHS>(rhs_span);
    using LhsInt = typename FPTypeInfo<LHS>::IntType;
    *reinterpret_cast<LhsInt *>(&lhs_span[0]) = FPTypeInfo<LHS>::kQNaN;
    *reinterpret_cast<LhsInt *>(&lhs_span[1]) = FPTypeInfo<LHS>::kSNaN;
    *reinterpret_cast<LhsInt *>(&lhs_span[2]) = FPTypeInfo<LHS>::kPosInf;
    *reinterpret_cast<LhsInt *>(&lhs_span[3]) = FPTypeInfo<LHS>::kNegInf;
    *reinterpret_cast<LhsInt *>(&lhs_span[4]) = FPTypeInfo<LHS>::kPosZero;
    *reinterpret_cast<LhsInt *>(&lhs_span[5]) = FPTypeInfo<LHS>::kNegZero;
    *reinterpret_cast<LhsInt *>(&lhs_span[6]) = FPTypeInfo<LHS>::kPosDenorm;
    *reinterpret_cast<LhsInt *>(&lhs_span[7]) = FPTypeInfo<LHS>::kNegDenorm;
    for (int i = 0; i < kTestValueLength; i++) {
      SetRegisterValues<LHS, LhsRegisterType>({{kR1Name, lhs_span[i]}});
      SetRegisterValues<RHS, RhsRegisterType>({{kR2Name, rhs_span[i]}});

      for (int rm : {0, 1, 2, 3, 4}) {
        rv_fp_->SetRoundingMode(static_cast<FPRoundingMode>(rm));
        SetRegisterValues<int, RV32Register>({{kRmName, rm}});
        SetRegisterValues<R, DestRegisterType>({{kRdName, 0}});

        inst->Execute(nullptr);

        R op_val;
        {
          ScopedFPStatus set_fpstatus(rv_fp_->host_fp_interface());
          op_val = operation(lhs_span[i], rhs_span[i]);
        }
        auto reg_val = state_->GetRegister<DestRegisterType>(kRdName)
                           .first->data_buffer()
                           ->template Get<R>(0);
        FPCompare<R>(op_val, reg_val, delta_position,
                     absl::StrCat(name, "  ", i, ": ", lhs_span[i], "  ",
                                  rhs_span[i], " rm: ", rm));
      }
      if (HasFailure()) return;
    }
  }

  // Test helper for binary instructions that also produce an exception flag
  // value.
  template <typename R, typename LHS, typename RHS>
  void BinaryOpWithFflagsFPTestHelper(
      absl::string_view name, Instruction *inst,
      absl::Span<const absl::string_view> reg_prefixes, int delta_position,
      std::function<std::tuple<R, uint32_t>(LHS, RHS)> operation) {
    using LhsRegisterType = RVFpRegister;
    using RhsRegisterType = RVFpRegister;
    using DestRegisterType = RVFpRegister;
    using LhsUInt = typename FPTypeInfo<LHS>::IntType;
    using RhsUInt = typename FPTypeInfo<RHS>::IntType;
    LHS lhs_values[kTestValueLength];
    RHS rhs_values[kTestValueLength];
    auto lhs_span = absl::Span<LHS>(lhs_values);
    auto rhs_span = absl::Span<RHS>(rhs_values);
    const std::string kR1Name = absl::StrCat(reg_prefixes[0], 1);
    const std::string kR2Name = absl::StrCat(reg_prefixes[1], 2);
    const std::string kRdName = absl::StrCat(reg_prefixes[2], 5);
    // This is used for the rounding mode operand.
    const std::string kRmName = absl::StrCat("x", 10);
    AppendRegisterOperands({kR1Name, kR2Name, kRmName}, {kRdName});
    auto *flag_op = rv_fp_->fflags()->CreateSetDestinationOperand(0, "fflags");
    instruction_->AppendDestination(flag_op);
    FillArrayWithRandomFPValues<LHS>(lhs_span);
    FillArrayWithRandomFPValues<RHS>(rhs_span);
    using LhsInt = typename FPTypeInfo<LHS>::IntType;
    *reinterpret_cast<LhsInt *>(&lhs_span[0]) = FPTypeInfo<LHS>::kQNaN;
    *reinterpret_cast<LhsInt *>(&lhs_span[1]) = FPTypeInfo<LHS>::kSNaN;
    *reinterpret_cast<LhsInt *>(&lhs_span[2]) = FPTypeInfo<LHS>::kPosInf;
    *reinterpret_cast<LhsInt *>(&lhs_span[3]) = FPTypeInfo<LHS>::kNegInf;
    *reinterpret_cast<LhsInt *>(&lhs_span[4]) = FPTypeInfo<LHS>::kPosZero;
    *reinterpret_cast<LhsInt *>(&lhs_span[5]) = FPTypeInfo<LHS>::kNegZero;
    *reinterpret_cast<LhsInt *>(&lhs_span[6]) = FPTypeInfo<LHS>::kPosDenorm;
    *reinterpret_cast<LhsInt *>(&lhs_span[7]) = FPTypeInfo<LHS>::kNegDenorm;
    for (int i = 0; i < kTestValueLength; i++) {
      SetRegisterValues<LHS, LhsRegisterType>({{kR1Name, lhs_span[i]}});
      SetRegisterValues<RHS, RhsRegisterType>({{kR2Name, rhs_span[i]}});

      for (int rm : {0, 1, 2, 3, 4}) {
        rv_fp_->SetRoundingMode(static_cast<FPRoundingMode>(rm));
        rv_fp_->fflags()->Write(static_cast<uint32_t>(0));
        SetRegisterValues<int, RV32Register>({{kRmName, rm}, {}});
        SetRegisterValues<R, DestRegisterType>({{kRdName, 0}});

        inst->Execute(nullptr);

        R op_val;
        uint32_t flag;
        {
          ScopedFPStatus set_fpstatus(rv_fp_->host_fp_interface());
          std::tie(op_val, flag) = operation(lhs_span[i], rhs_span[i]);
        }
        auto reg_val = state_->GetRegister<DestRegisterType>(kRdName)
                           .first->data_buffer()
                           ->template Get<R>(0);
        FPCompare<R>(
            op_val, reg_val, delta_position,
            absl::StrCat(name, "  ", i, ": ", lhs_span[i], "  ", rhs_span[i]));
        auto lhs_uint = *reinterpret_cast<LhsUInt *>(&lhs_span[i]);
        auto rhs_uint = *reinterpret_cast<RhsUInt *>(&rhs_span[i]);
        auto fflags = rv_fp_->fflags()->GetUint32();
        EXPECT_EQ(flag, fflags)
            << std::hex << name << "(" << lhs_uint << ", " << rhs_uint << ")";
      }
    }
  }

  template <typename R, typename LHS, typename MHS, typename RHS>
  void TernaryOpFPTestHelper(absl::string_view name, Instruction *inst,
                             absl::Span<const absl::string_view> reg_prefixes,
                             int delta_position,
                             std::function<R(LHS, MHS, RHS)> operation) {
    using LhsRegisterType = RVFpRegister;
    using MhsRegisterType = RVFpRegister;
    using RhsRegisterType = RVFpRegister;
    using DestRegisterType = RVFpRegister;
    LHS lhs_values[kTestValueLength];
    MHS mhs_values[kTestValueLength];
    RHS rhs_values[kTestValueLength];
    auto lhs_span = absl::Span<LHS>(lhs_values);
    auto mhs_span = absl::Span<MHS>(mhs_values);
    auto rhs_span = absl::Span<RHS>(rhs_values);
    const std::string kR1Name = absl::StrCat(reg_prefixes[0], 1);
    const std::string kR2Name = absl::StrCat(reg_prefixes[1], 2);
    const std::string kR3Name = absl::StrCat(reg_prefixes[2], 3);
    const std::string kRdName = absl::StrCat(reg_prefixes[3], 5);
    // This is used for the rounding mode operand.
    const std::string kRmName = absl::StrCat("x", 10);
    AppendRegisterOperands({kR1Name, kR2Name, kR3Name, kRmName}, {kRdName});
    FillArrayWithRandomFPValues<LHS>(lhs_span);
    FillArrayWithRandomFPValues<MHS>(mhs_span);
    FillArrayWithRandomFPValues<RHS>(rhs_span);
    using LhsInt = typename FPTypeInfo<LHS>::IntType;
    *reinterpret_cast<LhsInt *>(&lhs_span[0]) = FPTypeInfo<LHS>::kQNaN;
    *reinterpret_cast<LhsInt *>(&lhs_span[1]) = FPTypeInfo<LHS>::kSNaN;
    *reinterpret_cast<LhsInt *>(&lhs_span[2]) = FPTypeInfo<LHS>::kPosInf;
    *reinterpret_cast<LhsInt *>(&lhs_span[3]) = FPTypeInfo<LHS>::kNegInf;
    *reinterpret_cast<LhsInt *>(&lhs_span[4]) = FPTypeInfo<LHS>::kPosZero;
    *reinterpret_cast<LhsInt *>(&lhs_span[5]) = FPTypeInfo<LHS>::kNegZero;
    *reinterpret_cast<LhsInt *>(&lhs_span[6]) = FPTypeInfo<LHS>::kPosDenorm;
    *reinterpret_cast<LhsInt *>(&lhs_span[7]) = FPTypeInfo<LHS>::kNegDenorm;
    for (int i = 0; i < kTestValueLength; i++) {
      SetRegisterValues<LHS, LhsRegisterType>({{kR1Name, lhs_span[i]}});
      SetRegisterValues<MHS, MhsRegisterType>({{kR2Name, mhs_span[i]}});
      SetRegisterValues<RHS, RhsRegisterType>({{kR3Name, rhs_span[i]}});

      for (int rm : {0, 1, 2, 3, 4}) {
        rv_fp_->SetRoundingMode(static_cast<FPRoundingMode>(rm));
        SetRegisterValues<int, RV32Register>({{kRmName, rm}});
        SetRegisterValues<R, DestRegisterType>({{kRdName, 0}});

        inst->Execute(nullptr);

        R op_val;
        {
          ScopedFPStatus set_fpstatus(rv_fp_->host_fp_interface());
          op_val = operation(lhs_span[i], mhs_span[i], rhs_span[i]);
        }
        auto reg_val = state_->GetRegister<DestRegisterType>(kRdName)
                           .first->data_buffer()
                           ->template Get<R>(0);
        FPCompare<R>(op_val, reg_val, delta_position,
                     absl::StrCat(name, "  ", i, ": ", lhs_span[i], "  ",
                                  mhs_span[i], " ", rhs_span[i]));
      }
    }
  }

  template <typename R, typename LHS, typename MHS, typename RHS>
  void TernaryOpWithFflagsFPTestHelper(
      absl::string_view name, Instruction *inst,
      absl::Span<const absl::string_view> reg_prefixes, int delta_position,
      std::function<R(LHS, MHS, RHS)> operation) {
    using LhsRegisterType = RVFpRegister;
    using MhsRegisterType = RVFpRegister;
    using RhsRegisterType = RVFpRegister;
    using DestRegisterType = RVFpRegister;
    LHS lhs_values[kTestValueLength];
    MHS mhs_values[kTestValueLength];
    RHS rhs_values[kTestValueLength];
    auto lhs_span = absl::Span<LHS>(lhs_values);
    auto mhs_span = absl::Span<MHS>(mhs_values);
    auto rhs_span = absl::Span<RHS>(rhs_values);
    const std::string kR1Name = absl::StrCat(reg_prefixes[0], 1);
    const std::string kR2Name = absl::StrCat(reg_prefixes[1], 2);
    const std::string kR3Name = absl::StrCat(reg_prefixes[2], 3);
    const std::string kRdName = absl::StrCat(reg_prefixes[3], 5);
    // This is used for the rounding mode operand.
    const std::string kRmName = absl::StrCat("x", 10);
    AppendRegisterOperands({kR1Name, kR2Name, kR3Name, kRmName}, {kRdName});
    auto *flag_op = rv_fp_->fflags()->CreateSetDestinationOperand(0, "fflags");
    instruction_->AppendDestination(flag_op);
    FillArrayWithRandomFPValues<LHS>(lhs_span);
    FillArrayWithRandomFPValues<MHS>(mhs_span);
    FillArrayWithRandomFPValues<RHS>(rhs_span);
    using LhsInt = typename FPTypeInfo<LHS>::IntType;
    *reinterpret_cast<LhsInt *>(&lhs_span[0]) = FPTypeInfo<LHS>::kQNaN;
    *reinterpret_cast<LhsInt *>(&lhs_span[1]) = FPTypeInfo<LHS>::kSNaN;
    *reinterpret_cast<LhsInt *>(&lhs_span[2]) = FPTypeInfo<LHS>::kPosInf;
    *reinterpret_cast<LhsInt *>(&lhs_span[3]) = FPTypeInfo<LHS>::kNegInf;
    *reinterpret_cast<LhsInt *>(&lhs_span[4]) = FPTypeInfo<LHS>::kPosZero;
    *reinterpret_cast<LhsInt *>(&lhs_span[5]) = FPTypeInfo<LHS>::kNegZero;
    *reinterpret_cast<LhsInt *>(&lhs_span[6]) = FPTypeInfo<LHS>::kPosDenorm;
    *reinterpret_cast<LhsInt *>(&lhs_span[7]) = FPTypeInfo<LHS>::kNegDenorm;
    for (int i = 0; i < kTestValueLength; i++) {
      SetRegisterValues<LHS, LhsRegisterType>({{kR1Name, lhs_span[i]}});
      SetRegisterValues<MHS, MhsRegisterType>({{kR2Name, mhs_span[i]}});
      SetRegisterValues<RHS, RhsRegisterType>({{kR3Name, rhs_span[i]}});

      for (int rm : {0, 1, 2, 3, 4}) {
        rv_fp_->SetRoundingMode(static_cast<FPRoundingMode>(rm));
        rv_fp_->fflags()->Write(static_cast<uint32_t>(0));
        SetRegisterValues<int, RV32Register>({{kRmName, rm}});
        SetRegisterValues<R, DestRegisterType>({{kRdName, 0}});

        inst->Execute(nullptr);
        // Get the fflags for the instruction execution.
        auto fflags = rv_fp_->fflags()->GetUint32();
        rv_fp_->fflags()->Write(static_cast<uint32_t>(0));
        R op_val;
        {
          ScopedFPStatus set_fpstatus(rv_fp_->host_fp_interface());
          op_val = operation(lhs_span[i], mhs_span[i], rhs_span[i]);
        }
        auto reg_val = state_->GetRegister<DestRegisterType>(kRdName)
                           .first->data_buffer()
                           ->template Get<R>(0);
        FPCompare<R>(op_val, reg_val, delta_position,
                     absl::StrCat(name, "  ", i, ": ", lhs_span[i], "  ",
                                  mhs_span[i], " ", rhs_span[i]));

        auto flag = rv_fp_->fflags()->GetUint32();
        EXPECT_EQ(flag, fflags)
            << absl::StrCat(name, "  ", i, ": ", lhs_span[i], "  ", mhs_span[i],
                            " ", rhs_span[i]);
      }
    }
  }

  absl::Span<RV32Register *> xreg() {
    return absl::Span<RV32Register *>(xreg_);
  }

  absl::Span<RVFpRegister *> freg() {
    return absl::Span<RVFpRegister *>(freg_);
  }

  absl::Span<RV64Register *> dreg() {
    return absl::Span<RV64Register *>(dreg_);
  }
  absl::BitGen &bitgen() { return bitgen_; }
  Instruction *instruction() { return instruction_; }

  template <typename From, typename To>
  To RoundToInteger(From val, uint32_t rm, uint32_t &flags) {
    constexpr To kMax = std::numeric_limits<To>::max();
    constexpr To kMin = std::numeric_limits<To>::min();
    To value = 0;
    if (FPTypeInfo<From>::IsNaN(val)) {
      value = std::numeric_limits<To>::max();
      flags = (uint32_t)FPExceptions::kInvalidOp;
    } else if (val == 0.0) {
      value = 0;
    } else {
      using FromUint = typename FPTypeInfo<From>::IntType;
      auto constexpr kBias = FPTypeInfo<From>::kExpBias;
      auto constexpr kExpMask = FPTypeInfo<From>::kExpMask;
      auto constexpr kSigSize = FPTypeInfo<From>::kSigSize;
      auto constexpr kSigMask = FPTypeInfo<From>::kSigMask;
      auto constexpr kBitSize = FPTypeInfo<From>::kBitSize;
      FromUint val_u = *reinterpret_cast<FromUint *>(&val);
      FromUint exp = kExpMask & val_u;
      const bool sign = (val_u & (1ULL << (kBitSize - 1))) != 0;
      int exp_value = exp >> kSigSize;
      int unbiased_exp = exp_value - kBias;
      FromUint sig = kSigMask & val_u;
      // Turn the value into a denormal.

      int right_shift = exp ? kSigSize - 1 - unbiased_exp : 1;
      uint64_t base = sig;
      bool rightshift_compressed = 0;
      if (exp == 0) {
        // Denormalized value.
        // Format: (-1)^sign * 2 ^(exp - bias + 1) * 0.{sig}
        // fraction part is too small keep it as 1 bits if not zero.
        rightshift_compressed = base != 0;
        base = 0;
        flags = (uint32_t)FPExceptions::kInexact;
      } else {
        // Normalized value.
        // Format: (-1)^sign * 2 ^(exp - bias) * 1.{sig
        // base = 1.{sig} (total (1 + `kSigSize`) bits)
        base |= 1ULL << kSigSize;

        // Right shift all base part out.
        if (right_shift > (kBitSize - 1)) {
          rightshift_compressed = base != 0;
          flags = (uint32_t)FPExceptions::kInexact;
          base = 0;
        } else if (right_shift > 0) {
          // Right shift to leave only 1 bit in the fraction part, compressed
          // the right-shift eliminated number.
          right_shift = std::min(right_shift, kBitSize);
          uint64_t right_shifted_sig_mask = (1ULL << right_shift) - 1;
          rightshift_compressed = (base & right_shifted_sig_mask) != 0;
          base >>= right_shift;
        }
      }

      // Handle fraction part rounding.
      if (right_shift >= 0) {
        switch ((FPRoundingMode)rm) {
          case FPRoundingMode::kRoundToNearest:
            // 0.5, tie condition
            if (rightshift_compressed == 0 && base & 0b1) {
              // <odd>.5 -> <odd>.5 + 0.5 = even
              if ((base & 0b11) == 0b11) {
                base += 0b01;
                flags = (uint32_t)FPExceptions::kInexact;
              }
            } else if (base & 0b1 || rightshift_compressed) {
              // not tie condition, round to nearest integer, it equals to add
              // 0.5(=base + 0b01) and eliminate the fraction part.
              base += 0b01;
            }
            break;
          case FPRoundingMode::kRoundTowardsZero:
            // Round towards zero will eliminate the fraction part.
            // Do nothing on fraction part.
            // 1.2 -> 1.0, -1.5 -> -1.0, -0.7 -> 0.0
            break;
          case FPRoundingMode::kRoundDown:
            // Positive float will eliminate the fraction part.
            // Negative float with fraction part will subtract 1(= base + 0b10),
            // and eliminate the fraction part.
            // e.g., 1.2 -> 1.0, -1.5 -> -2.0, -0.7 -> -1.0
            if (sign && (base & 0b1 || rightshift_compressed)) {
              base += 0b10;
            }
            break;
          case FPRoundingMode::kRoundUp:
            // Positive float will add 1(= base + 0b10), and eliminate the
            // fraction part.
            // Negative float will eliminate the fraction part.
            // e.g., 1.2 -> 2.0, -1.5 -> -1.0, -0.7 -> 0.0
            if (!sign && (base & 0b1 || rightshift_compressed)) {
              base += 0b10;
            }
            break;
          case FPRoundingMode::kRoundToNearestTiesToMax:
            // Round to nearest integer that is far from zero.
            // e.g., 1.2 -> 2.0, -1.5 -> -2.0, -0.7 -> -1.0
            if (base & 0b1 || rightshift_compressed) {
              base += 0b1;
            }
            break;
          default:
            LOG(ERROR) << "Invalid rounding mode";
            return To();
        }
      }
      uint64_t unsigned_value;
      // Handle base with fraction part and store it to `unsigned_value`.
      if (right_shift >= 0) {
        // Set inexact flag if floating value has fraction part.
        if (base & 0b1 || rightshift_compressed) {
          flags = (uint32_t)FPExceptions::kInexact;
        }
        unsigned_value = base >> 1;
      } else {
        // Handle base without fraction part but need to left shift.
        int left_shift = -right_shift - 1;
        auto prev = unsigned_value = base;
        while (left_shift) {
          unsigned_value <<= 1;
          // Check if overflow happened and set the flag.
          if (prev > unsigned_value) {
            flags = (uint32_t)FPExceptions::kInvalidOp;
            unsigned_value = sign ? kMin : kMax;
            break;
          }
          prev = unsigned_value;
          --left_shift;
        }
      }

      // Handle the case that value is out of range, and final convert to value
      // with sign.
      if (std::is_signed<To>::value) {
        // Positive value but exceeds the max value.
        if (!sign && unsigned_value > kMax) {
          flags = (uint32_t)FPExceptions::kInvalidOp;
          value = kMax;
        } else if (sign && (unsigned_value > 0 && -unsigned_value < kMin)) {
          // Negative value but exceeds the min value.
          flags = (uint32_t)FPExceptions::kInvalidOp;
          value = kMin;
        } else {
          value = sign ? -((To)unsigned_value) : unsigned_value;
        }
      } else {
        // Positive value but exceeds the max value.
        if (unsigned_value > kMax) {
          flags = (uint32_t)FPExceptions::kInvalidOp;
          value = sign ? kMin : kMax;
        } else if (sign && unsigned_value != 0) {
          // float is negative value this is out of range of valid unsigned
          // value.
          flags = (uint32_t)FPExceptions::kInvalidOp;
          value = kMin;
        } else {
          value = sign ? -((To)unsigned_value) : unsigned_value;
        }
      }
    }
    return value;
  }

 protected:
  RV32Register *xreg_[32];
  RV64Register *dreg_[32];
  RVFpRegister *freg_[32];
  RiscVState *state_;
  Instruction *instruction_;
  Instruction *child_instruction_;
  FlatDemandMemory *memory_;
  RiscVFPState *rv_fp_;
  absl::BitGen bitgen_;
};

}  // namespace test
}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_TEST_RISCV_FP_TEST_BASE_H_
