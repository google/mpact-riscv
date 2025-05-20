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

#include <sys/stat.h>
#include <sys/types.h>

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
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
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

using ::mpact::sim::generic::ConvertHalfToSingle;
using ::mpact::sim::generic::FloatingPointToString;
using ::mpact::sim::generic::HalfFP;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::generic::IsMpactFp;
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
    IntType uint_val = absl::bit_cast<IntType>(value);
    return IsNaN(value) && (((1ULL << (kSigSize - 1)) & uint_val) != 0);
  }
  static bool IsInf(T value) { return std::isinf(value); }
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
    IntType uint_val = absl::bit_cast<IntType>(value);
    return IsNaN(value) && (((1ULL << (kSigSize - 1)) & uint_val) != 0);
  }
  static bool IsInf(T value) { return std::isinf(value); }
};

template <>
struct FPTypeInfo<HalfFP> {
  using T = HalfFP;
  using IntType = uint16_t;
  static const int kExpBias = 15;
  static const int kBitSize = sizeof(HalfFP) << 3;
  static const int kExpSize = 5;
  static const int kSigSize = kBitSize - kExpSize - 1;  // 10 from the spec.
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
  static const IntType kCanonicalNaN = 0x7e00;
  // std::isnan won't work for half precision.
  static bool IsNaN(T wrapper) {
    IntType exp = (wrapper.value & kExpMask) >> kSigSize;
    IntType sig = wrapper.value & kSigMask;
    return (exp == (1 << kExpSize) - 1) && (sig != 0);
  }
  static bool IsQNaN(T value) {
    IntType uint_val = absl::bit_cast<IntType>(value);
    IntType significand_msb = (uint_val >> (kSigSize - 1)) & 1;
    return IsNaN(value) && (significand_msb != 0);
  }
  // std::isinf won't work for half precision.
  static bool IsInf(T wrapper) {
    IntType exp = (wrapper.value & kExpMask) >> kSigSize;
    IntType sig = wrapper.value & kSigMask;
    return (exp == (1 << kExpSize) - 1) && (sig == 0);
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
      auto uint_val = absl::bit_cast<typename FPTypeInfo<T>::IntType>(val);
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
  UInt u_op = absl::bit_cast<UInt>(op);
  UInt u_reg = absl::bit_cast<UInt>(reg);
  if (!std::isnan(op) && !std::isinf(op) &&
      delta_position < FPTypeInfo<T>::kSigSize) {
    T delta;
    UInt exp = FPTypeInfo<T>::kExpMask >> FPTypeInfo<T>::kSigSize;
    if (exp > delta_position) {
      exp -= delta_position;
      UInt udelta = exp << FPTypeInfo<T>::kSigSize;
      delta = absl::bit_cast<T>(udelta);
    } else {
      // Becomes a denormal
      int diff = delta_position - exp;
      UInt udelta = 1ULL << (FPTypeInfo<T>::kSigSize - 1 - diff);
      delta = absl::bit_cast<T>(udelta);
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
  UInt u_op = absl::bit_cast<UInt>(op);
  UInt u_reg = absl::bit_cast<UInt>(reg);
  if (!std::isnan(op) && !std::isinf(op) &&
      delta_position < FPTypeInfo<T>::kSigSize) {
    T delta;
    UInt exp = FPTypeInfo<T>::kExpMask >> FPTypeInfo<T>::kSigSize;
    if (exp > delta_position) {
      exp -= delta_position;
      UInt udelta = exp << FPTypeInfo<T>::kSigSize;
      delta = absl::bit_cast<T>(udelta);
    } else {
      // Becomes a denormal
      int diff = delta_position - exp;
      UInt udelta = 1ULL << (FPTypeInfo<T>::kSigSize - 1 - diff);
      delta = absl::bit_cast<T>(udelta);
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

template <>
inline void FPCompare<HalfFP>(HalfFP op, HalfFP reg, int delta_position,
                              absl::string_view str) {
  float op_float = ConvertHalfToSingle(op);
  float reg_float = ConvertHalfToSingle(reg);
  FPCompare<float>(op_float, reg_float, delta_position, str);
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
  static const bool value = sizeof(S) == sizeof(D) && IsMpactFp<S>::value &&
                            std::is_integral<D>::value;
};

template <typename S, typename D>
struct GreaterSize {
  static const bool value =
      sizeof(S) > sizeof(D) && IsMpactFp<S>::value &&std::is_integral<D>::value;
};

template <typename S, typename D>
struct LessSize {
  static const bool value = sizeof(S) < sizeof(D) && IsMpactFp<S>::value &&
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
  SInt sval = absl::bit_cast<SInt>(value);
  D dval = (~static_cast<D>(0) << (sizeof(S) * 8)) | sval;
  return absl::bit_cast<D>(dval);
}

// This version does a straight copy - as the data types are the same size.
template <typename S, typename D>
inline typename std::enable_if<internal::EqualSize<S, D>::value, D>::type
NaNBox(S value) {
  return absl::bit_cast<D>(value);
}

// Signal error if the register is smaller than the floating point value.
template <typename S, typename D>
inline typename std::enable_if<internal::GreaterSize<S, D>::value, D>::type
NaNBox(S value) {
  // No return statement, so error will be reported.
}

template <typename XRegister = RV32Register>
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
      xreg_[i] = state_->GetRegister<XRegister>(absl::StrCat("x", i)).first;
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
  template <typename T>
  void AppendRegisterOperands(Instruction *inst,
                              const std::vector<std::string> &sources,
                              const std::vector<std::string> &destinations) {
    for (auto &reg_name : sources) {
      auto *reg = state_->GetRegister<T>(reg_name).first;
      inst->AppendSource(reg->CreateSourceOperand());
    }
    for (auto &reg_name : destinations) {
      auto *reg = state_->GetRegister<T>(reg_name).first;
      inst->AppendDestination(reg->CreateDestinationOperand(0));
    }
  }

  // Creates source and destination scalar register operands for the registers
  // named in the two vectors and append them to the default instruction.
  template <typename T>
  void AppendRegisterOperands(const std::vector<std::string> &sources,
                              const std::vector<std::string> &destinations) {
    AppendRegisterOperands<T>(instruction_, sources, destinations);
  }

  // named register and sets it to the corresponding value.
  template <typename T, typename RegisterType = XRegister>
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

  template <typename T, typename RegisterType = XRegister>
  void SetNaNBoxedRegisterValues(
      const std::vector<std::tuple<std::string, const T>> &values) {
    for (auto &[reg_name, value] : values) {
      typename RegisterType::ValueType reg_value =
          NaNBox<T, typename RegisterType::ValueType>(value);
      auto *reg = state_->GetRegister<RegisterType>(reg_name).first;
      auto *db =
          state_->db_factory()->Allocate<typename RegisterType::ValueType>(1);
      db->template Set<typename RegisterType::ValueType>(0, reg_value);
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
    return absl::bit_cast<T>(value);
    ;
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
    if (kR1Name[0] == 'x') {
      AppendRegisterOperands<XRegister>({kR1Name}, {});
    } else {
      AppendRegisterOperands<RVFpRegister>({kR1Name}, {});
    }
    if (kRdName[0] == 'x') {
      AppendRegisterOperands<XRegister>({}, {kRdName});
    } else {
      AppendRegisterOperands<RVFpRegister>({}, {kRdName});
    }
    AppendRegisterOperands<XRegister>({kRmName}, {});
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
      if constexpr (std::is_integral<LHS>::value) {
        SetRegisterValues<LHS, LhsRegisterType>({{kR1Name, lhs_span[i]}});
      } else {
        SetNaNBoxedRegisterValues<LHS, LhsRegisterType>(
            {{kR1Name, lhs_span[i]}});
      }

      for (int rm : {0, 1, 2, 3, 4}) {
        rv_fp_->SetRoundingMode(static_cast<FPRoundingMode>(rm));
        SetRegisterValues<int, XRegister>({{kRmName, rm}});
        SetRegisterValues<DestRegisterType::ValueType, DestRegisterType>(
            {{kRdName, 0}});

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
                     absl::StrCat(name, "  ", i, ": ",
                                  FloatingPointToString<LHS>(lhs_span[i])));
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
    if (kR1Name[0] == 'x') {
      AppendRegisterOperands<XRegister>({kR1Name}, {});
    } else {
      AppendRegisterOperands<RVFpRegister>({kR1Name}, {});
    }
    if (kRdName[0] == 'x') {
      AppendRegisterOperands<XRegister>({}, {kRdName});
    } else {
      AppendRegisterOperands<RVFpRegister>({}, {kRdName});
    }
    AppendRegisterOperands<XRegister>({kRmName}, {});
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
      if constexpr (std::is_integral<LHS>::value) {
        SetRegisterValues<LHS, LhsRegisterType>({{kR1Name, lhs_span[i]}});
      } else {
        SetNaNBoxedRegisterValues<LHS, LhsRegisterType>(
            {{kR1Name, lhs_span[i]}});
      }

      for (int rm : {0, 1, 2, 3, 4}) {
        rv_fp_->SetRoundingMode(static_cast<FPRoundingMode>(rm));
        rv_fp_->fflags()->Write(static_cast<uint32_t>(0));
        SetRegisterValues<int, XRegister>({{kRmName, rm}, {}});
        SetRegisterValues<DestRegisterType::ValueType, DestRegisterType>(
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
    if (kR1Name[0] == 'x') {
      AppendRegisterOperands<XRegister>({kR1Name}, {});
    } else {
      AppendRegisterOperands<RVFpRegister>({kR1Name}, {});
    }
    if (kR2Name[0] == 'x') {
      AppendRegisterOperands<XRegister>({kR2Name}, {});
    } else {
      AppendRegisterOperands<RVFpRegister>({kR2Name}, {});
    }
    if (kRdName[0] == 'x') {
      AppendRegisterOperands<XRegister>({}, {kRdName});
    } else {
      AppendRegisterOperands<RVFpRegister>({}, {kRdName});
    }
    AppendRegisterOperands<XRegister>({kRmName}, {});
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
      SetNaNBoxedRegisterValues<LHS, LhsRegisterType>({{kR1Name, lhs_span[i]}});
      SetNaNBoxedRegisterValues<RHS, RhsRegisterType>({{kR2Name, rhs_span[i]}});

      for (int rm : {0, 1, 2, 3, 4}) {
        rv_fp_->SetRoundingMode(static_cast<FPRoundingMode>(rm));
        SetRegisterValues<int, XRegister>({{kRmName, rm}});
        SetRegisterValues<DestRegisterType::ValueType, DestRegisterType>(
            {{kRdName, 0}});

        inst->Execute(nullptr);

        R op_val;
        {
          ScopedFPStatus set_fpstatus(rv_fp_->host_fp_interface());
          op_val = operation(lhs_span[i], rhs_span[i]);
        }
        auto reg_val = state_->GetRegister<DestRegisterType>(kRdName)
                           .first->data_buffer()
                           ->template Get<R>(0);
        FPCompare<R>(
            op_val, reg_val, delta_position,
            absl::StrCat(name, "  ", i, ": ",
                         FloatingPointToString<LHS>(lhs_span[i]), "  ",
                         FloatingPointToString<RHS>(rhs_span[i]), " rm: ", rm));
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
    if (kR1Name[0] == 'x') {
      AppendRegisterOperands<XRegister>({kR1Name}, {});
    } else {
      AppendRegisterOperands<RVFpRegister>({kR1Name}, {});
    }
    if (kR2Name[0] == 'x') {
      AppendRegisterOperands<XRegister>({kR2Name}, {});
    } else {
      AppendRegisterOperands<RVFpRegister>({kR2Name}, {});
    }
    if (kRdName[0] == 'x') {
      AppendRegisterOperands<XRegister>({}, {kRdName});
    } else {
      AppendRegisterOperands<RVFpRegister>({}, {kRdName});
    }
    AppendRegisterOperands<XRegister>({kRmName}, {});
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
      SetNaNBoxedRegisterValues<LHS, LhsRegisterType>({{kR1Name, lhs_span[i]}});
      SetNaNBoxedRegisterValues<RHS, RhsRegisterType>({{kR2Name, rhs_span[i]}});

      for (int rm : {0, 1, 2, 3, 4}) {
        rv_fp_->SetRoundingMode(static_cast<FPRoundingMode>(rm));
        rv_fp_->fflags()->Write(static_cast<uint32_t>(0));
        SetRegisterValues<int, XRegister>({{kRmName, rm}, {}});
        SetRegisterValues<DestRegisterType::ValueType, DestRegisterType>(
            {{kRdName, 0}});

        inst->Execute(nullptr);

        auto instruction_fflags = rv_fp_->fflags()->GetUint32();

        R op_val;
        uint32_t test_operation_fflags;
        {
          ScopedFPStatus set_fpstatus(rv_fp_->host_fp_interface());
          std::tie(op_val, test_operation_fflags) =
              operation(lhs_span[i], rhs_span[i]);
        }
        auto reg_val = state_->GetRegister<DestRegisterType>(kRdName)
                           .first->data_buffer()
                           ->template Get<R>(0);
        FPCompare<R>(op_val, reg_val, delta_position,
                     absl::StrCat(name, "  ", i, ": ",
                                  FloatingPointToString<LHS>(lhs_span[i]), "  ",
                                  FloatingPointToString<RHS>(rhs_span[i]),
                                  " (rm: ", rm, ") "));
        LhsUInt lhs_uint = absl::bit_cast<LhsUInt>(lhs_span[i]);
        RhsUInt rhs_uint = absl::bit_cast<RhsUInt>(rhs_span[i]);
        EXPECT_EQ(test_operation_fflags, instruction_fflags)
            << std::hex << name << "(" << lhs_uint << ", " << rhs_uint << ")"
            << " rm: " << rm;
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
    if (kR1Name[0] == 'x') {
      AppendRegisterOperands<XRegister>({kR1Name}, {});
    } else {
      AppendRegisterOperands<RVFpRegister>({kR1Name}, {});
    }
    if (kR2Name[0] == 'x') {
      AppendRegisterOperands<XRegister>({kR2Name}, {});
    } else {
      AppendRegisterOperands<RVFpRegister>({kR2Name}, {});
    }
    if (kR3Name[0] == 'x') {
      AppendRegisterOperands<XRegister>({kR3Name}, {});
    } else {
      AppendRegisterOperands<RVFpRegister>({kR3Name}, {});
    }
    if (kRdName[0] == 'x') {
      AppendRegisterOperands<XRegister>({}, {kRdName});
    } else {
      AppendRegisterOperands<RVFpRegister>({}, {kRdName});
    }
    AppendRegisterOperands<XRegister>({kRmName}, {});
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
      SetNaNBoxedRegisterValues<LHS, LhsRegisterType>({{kR1Name, lhs_span[i]}});
      SetNaNBoxedRegisterValues<MHS, MhsRegisterType>({{kR2Name, mhs_span[i]}});
      SetNaNBoxedRegisterValues<RHS, RhsRegisterType>({{kR3Name, rhs_span[i]}});

      for (int rm : {0, 1, 2, 3, 4}) {
        rv_fp_->SetRoundingMode(static_cast<FPRoundingMode>(rm));
        SetRegisterValues<int, XRegister>({{kRmName, rm}});
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
                     absl::StrCat(name, "  ", i, ": ",
                                  FloatingPointToString<LHS>(lhs_span[i]), "  ",
                                  FloatingPointToString<MHS>(mhs_span[i]), "  ",
                                  FloatingPointToString<RHS>(rhs_span[i])));
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
    if (kR1Name[0] == 'x') {
      AppendRegisterOperands<XRegister>({kR1Name}, {});
    } else {
      AppendRegisterOperands<RVFpRegister>({kR1Name}, {});
    }
    if (kR2Name[0] == 'x') {
      AppendRegisterOperands<XRegister>({kR2Name}, {});
    } else {
      AppendRegisterOperands<RVFpRegister>({kR2Name}, {});
    }
    if (kR3Name[0] == 'x') {
      AppendRegisterOperands<XRegister>({kR3Name}, {});
    } else {
      AppendRegisterOperands<RVFpRegister>({kR3Name}, {});
    }
    if (kRdName[0] == 'x') {
      AppendRegisterOperands<XRegister>({}, {kRdName});
    } else {
      AppendRegisterOperands<RVFpRegister>({}, {kRdName});
    }
    AppendRegisterOperands<XRegister>({kRmName}, {});
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
      SetNaNBoxedRegisterValues<LHS, LhsRegisterType>({{kR1Name, lhs_span[i]}});
      SetNaNBoxedRegisterValues<MHS, MhsRegisterType>({{kR2Name, mhs_span[i]}});
      SetNaNBoxedRegisterValues<RHS, RhsRegisterType>({{kR3Name, rhs_span[i]}});

      for (int rm : {0, 1, 2, 3, 4}) {
        rv_fp_->SetRoundingMode(static_cast<FPRoundingMode>(rm));
        rv_fp_->fflags()->Write(static_cast<uint32_t>(0));
        SetRegisterValues<int, XRegister>({{kRmName, rm}});
        SetRegisterValues<DestRegisterType::ValueType, DestRegisterType>(
            {{kRdName, 0}});

        inst->Execute(nullptr);
        // Get the fflags for the instruction execution.
        auto instruction_fflags = rv_fp_->fflags()->GetUint32();
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
                     absl::StrCat(name, "  ", i, ": ",
                                  FloatingPointToString<LHS>(lhs_span[i]), "  ",
                                  FloatingPointToString<MHS>(mhs_span[i]), "  ",
                                  FloatingPointToString<RHS>(rhs_span[i])));

        auto test_operation_fflags = rv_fp_->fflags()->GetUint32();
        EXPECT_EQ(test_operation_fflags, instruction_fflags) << absl::StrCat(
            name, "  ", i, ": ", FloatingPointToString<LHS>(lhs_span[i]), " ",
            FloatingPointToString<MHS>(mhs_span[i]), " ",
            FloatingPointToString<RHS>(rhs_span[i]));
      }
    }
  }

  absl::Span<XRegister *> xreg() { return absl::Span<XRegister *>(xreg_); }

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
      FromUint val_u = absl::bit_cast<FromUint>(val);
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
  XRegister *xreg_[32];
  RV64Register *dreg_[32];
  RVFpRegister *freg_[32];
  RiscVState *state_;
  Instruction *instruction_;
  Instruction *child_instruction_;
  FlatDemandMemory *memory_;
  RiscVFPState *rv_fp_;
  absl::BitGen bitgen_;
};

namespace internal {

template <typename T>
struct UnsignedToFpType {};

template <>
struct UnsignedToFpType<uint16_t> {
  using FpType = HalfFP;
};

template <>
struct UnsignedToFpType<uint32_t> {
  using FpType = float;
};

template <>
struct UnsignedToFpType<uint64_t> {
  using FpType = double;
};

template <typename T>
double ToDouble(T input) {
  using IntType = typename FPTypeInfo<T>::IntType;
  using FpType = typename internal::UnsignedToFpType<IntType>::FpType;

  IntType uint_val = absl::bit_cast<IntType>(input);
  FpType fp_val = absl::bit_cast<FpType>(uint_val);
  if (FPTypeInfo<FpType>::IsNaN(fp_val)) {
    return std::numeric_limits<double>::quiet_NaN();
  } else if (FPTypeInfo<FpType>::IsInf(fp_val)) {
    return std::numeric_limits<double>::infinity();
  }
  IntType exp =
      (uint_val & FPTypeInfo<FpType>::kExpMask) >> FPTypeInfo<FpType>::kSigSize;
  IntType sig = uint_val & FPTypeInfo<FpType>::kSigMask;
  int32_t unbiased_exponent =
      exp ? static_cast<int32_t>(exp) - FPTypeInfo<FpType>::kExpBias
          : 1 - static_cast<int32_t>(FPTypeInfo<FpType>::kExpBias);

  double exponent_factor = std::pow(2.0, unbiased_exponent);
  double significand_factor = static_cast<double>(sig);
  double precision_factor =
      std::pow(2.0, -static_cast<int32_t>(FPTypeInfo<FpType>::kSigSize));
  double implicit_bit_adjustment = exp ? 1.0 : 0.0;
  double sign_factor =
      std::pow(-1.0, uint_val >> (FPTypeInfo<FpType>::kBitSize - 1));
  return ((significand_factor * precision_factor) + implicit_bit_adjustment) *
         exponent_factor * sign_factor;
}

// sign, lsb, guard, round, sticky --- rm = 0
inline constexpr int kRoundToNearestTable[] = {
    0, /*00000*/
    0, /*00001*/
    0, /*00010*/
    0, /*00011*/
    0, /*00100*/
    1, /*00101*/
    1, /*00110*/
    1, /*00111*/
    0, /*01000*/
    0, /*01001*/
    0, /*01010*/
    0, /*01011*/
    1, /*01100*/
    1, /*01101*/
    1, /*01110*/
    1, /*01111*/
    0, /*10000*/
    0, /*10001*/
    0, /*10010*/
    0, /*10011*/
    0, /*10100*/
    1, /*10101*/
    1, /*10110*/
    1, /*10111*/
    0, /*11000*/
    0, /*11001*/
    0, /*11010*/
    0, /*11011*/
    1, /*11100*/
    1, /*11101*/
    1, /*11110*/
    1, /*11111*/
};
// sign, lsb, guard, round, sticky --- rm = 1
inline constexpr int kRoundTowardsZeroTable[] = {
    0, /*00000*/
    0, /*00001*/
    0, /*00010*/
    0, /*00011*/
    0, /*00100*/
    0, /*00101*/
    0, /*00110*/
    0, /*00111*/
    0, /*01000*/
    0, /*01001*/
    0, /*01010*/
    0, /*01011*/
    0, /*01100*/
    0, /*01101*/
    0, /*01110*/
    0, /*01111*/
    0, /*10000*/
    0, /*10001*/
    0, /*10010*/
    0, /*10011*/
    0, /*10100*/
    0, /*10101*/
    0, /*10110*/
    0, /*10111*/
    0, /*11000*/
    0, /*11001*/
    0, /*11010*/
    0, /*11011*/
    0, /*11100*/
    0, /*11101*/
    0, /*11110*/
    0, /*11111*/
};
// sign, lsb, guard, round, sticky --- rm = 2
inline constexpr int kRoundDownTable[] = {
    0, /*00000*/
    0, /*00001*/
    0, /*00010*/
    0, /*00011*/
    0, /*00100*/
    0, /*00101*/
    0, /*00110*/
    0, /*00111*/
    0, /*01000*/
    0, /*01001*/
    0, /*01010*/
    0, /*01011*/
    0, /*01100*/
    0, /*01101*/
    0, /*01110*/
    0, /*01111*/
    0, /*10000*/
    1, /*10001*/
    1, /*10010*/
    1, /*10011*/
    1, /*10100*/
    1, /*10101*/
    1, /*10110*/
    1, /*10111*/
    0, /*11000*/
    1, /*11001*/
    1, /*11010*/
    1, /*11011*/
    1, /*11100*/
    1, /*11101*/
    1, /*11110*/
    1, /*11111*/
};
// sign, lsb, guard, round, sticky --- rm = 3
inline constexpr int kRoundUpTable[] = {
    0, /*00000*/
    1, /*00001*/
    1, /*00010*/
    1, /*00011*/
    1, /*00100*/
    1, /*00101*/
    1, /*00110*/
    1, /*00111*/
    0, /*01000*/
    1, /*01001*/
    1, /*01010*/
    1, /*01011*/
    1, /*01100*/
    1, /*01101*/
    1, /*01110*/
    1, /*01111*/
    0, /*10000*/
    0, /*10001*/
    0, /*10010*/
    0, /*10011*/
    0, /*10100*/
    0, /*10101*/
    0, /*10110*/
    0, /*10111*/
    0, /*11000*/
    0, /*11001*/
    0, /*11010*/
    0, /*11011*/
    0, /*11100*/
    0, /*11101*/
    0, /*11110*/
    0, /*11111*/
};

}  // namespace internal

template <typename T>
class FpConversionsTestHelper {
  using IntType = typename FPTypeInfo<T>::IntType;
  using FpType = typename internal::UnsignedToFpType<IntType>::FpType;

 public:
  // The conversion helper can be used with a float in its floating point format
  // or with its unsigned integer representation.
  FpConversionsTestHelper(T value) : fflags_(0) {
    if constexpr (std::is_same_v<T, IntType>) {
      unsigned_value_ = value;
      fp_value_ = absl::bit_cast<FpType>(unsigned_value_);
    } else if constexpr (std::is_same_v<T, FpType>) {
      fp_value_ = value;
      unsigned_value_ = absl::bit_cast<IntType>(fp_value_);
    }
  }

  template <typename U>
  U Convert(FPRoundingMode rm = FPRoundingMode::kRoundToNearest);

  template <typename U>
  U ConvertWithFlags(uint32_t &fflags,
                     FPRoundingMode rm = FPRoundingMode::kRoundToNearest) {
    fflags_ = 0;
    U ret = Convert<U>(rm);
    fflags |= fflags_;
    return ret;
  }

 protected:
  FpType fp_value_;
  IntType unsigned_value_;
  uint32_t fflags_;

  bool sign() {
    return (unsigned_value_ & (1ULL << (FPTypeInfo<FpType>::kBitSize - 1))) !=
           0;
  }

  template <typename IntReturnType, typename FpReturnType>
  void NarrowingConversionMakeExponentAndSignificand(FPRoundingMode,
                                                     IntReturnType &,
                                                     IntReturnType &);

  template <typename IntReturnType, typename FpReturnType>
  IntReturnType NarrowingConversionHandleInfinity(FPRoundingMode);

  template <typename IntReturnType, typename FpReturnType>
  IntReturnType NarrowingConversion(FPRoundingMode);

  template <typename U>
  U RoundingRightShift(U value, int32_t shift_amt, FPRoundingMode rm) {
    bool guard = 0;
    bool round = 0;
    bool sticky = 0;
    for (int i = 0; i < shift_amt; ++i) {
      sticky |= round;
      round = guard;
      guard = value & 1;
      value >>= 1;
    }

    bool lsb = value & 1;
    uint8_t key = sign() << 4 | lsb << 3 | guard << 2 | round << 1 | sticky;
    value += GetRoundingTable(rm)[key];
    return value;
  }

  const int *GetRoundingTable(FPRoundingMode rm) {
    switch (rm) {
      case FPRoundingMode::kRoundToNearest:
        return static_cast<const int *>(internal::kRoundToNearestTable);
      case FPRoundingMode::kRoundTowardsZero:
        return static_cast<const int *>(internal::kRoundTowardsZeroTable);
      case FPRoundingMode::kRoundDown:
        return static_cast<const int *>(internal::kRoundDownTable);
      case FPRoundingMode::kRoundUp:
        return static_cast<const int *>(internal::kRoundUpTable);
      default:
        return static_cast<const int *>(internal::kRoundToNearestTable);
    }
  }
};  // class FpConversionsTestHelper

template <typename T>
template <typename U>
U FpConversionsTestHelper<T>::Convert(FPRoundingMode rm) {
  using IntReturnType = typename FPTypeInfo<U>::IntType;
  using FpReturnType =
      typename internal::UnsignedToFpType<IntReturnType>::FpType;

  if constexpr (std::is_same_v<U, IntType>) {
    return unsigned_value_;
  } else if constexpr (std::is_same_v<U, FpType>) {
    return fp_value_;
  }

  if (FPTypeInfo<FpType>::IsNaN(fp_value_) &&
      !FPTypeInfo<FpType>::IsQNaN(fp_value_)) {
    fflags_ |= static_cast<uint32_t>(FPExceptions::kInvalidOp);
  }

  if (FPTypeInfo<FpType>::IsNaN(fp_value_)) {
    IntReturnType return_bits =
        sign() ? 1ULL << (FPTypeInfo<FpReturnType>::kBitSize - 1) : 0;
    return_bits |= FPTypeInfo<FpReturnType>::kCanonicalNaN;
    return absl::bit_cast<U>(return_bits);
  }

  if (FPTypeInfo<FpType>::kPosInf == unsigned_value_) {
    return absl::bit_cast<U>(FPTypeInfo<FpReturnType>::kPosInf);
  } else if (FPTypeInfo<FpType>::kNegInf == unsigned_value_) {
    return absl::bit_cast<U>(FPTypeInfo<FpReturnType>::kNegInf);
  } else if (FPTypeInfo<FpType>::kPosZero == unsigned_value_) {
    return absl::bit_cast<U>(FPTypeInfo<FpReturnType>::kPosZero);
  } else if (FPTypeInfo<FpType>::kNegZero == unsigned_value_) {
    return absl::bit_cast<U>(FPTypeInfo<FpReturnType>::kNegZero);
  }

  if constexpr (std::numeric_limits<IntReturnType>::digits >
                std::numeric_limits<IntType>::digits) {
    // The return type is larger so the conversion is simple.
    FpReturnType mantissa = static_cast<FpReturnType>(
        unsigned_value_ & FPTypeInfo<FpType>::kSigMask);
    FpReturnType precision_factor =
        std::pow(2.0, -static_cast<FpReturnType>(FPTypeInfo<FpType>::kSigSize));
    IntType biased_exponent =
        (unsigned_value_ & FPTypeInfo<FpType>::kExpMask) >>
        FPTypeInfo<FpType>::kSigSize;
    int32_t unbiased_exponent =
        (biased_exponent ? static_cast<int32_t>(biased_exponent) : 1) -
        FPTypeInfo<FpType>::kExpBias;

    // Use the formula from the IEEE754 section 3.4 that details moving from
    // the binary format to the number being represented.
    FpReturnType implicit_bit_adjustment = biased_exponent ? 1.0 : 0.0;
    FpReturnType exponent_factor = std::pow(2.0, unbiased_exponent);
    FpReturnType unsigned_result =
        ((mantissa * precision_factor) + implicit_bit_adjustment) *
        exponent_factor;
    IntReturnType result = absl::bit_cast<IntReturnType>(unsigned_result) |
                           (static_cast<IntReturnType>(sign())
                            << (FPTypeInfo<FpReturnType>::kBitSize - 1));
    return absl::bit_cast<U>(result);
  }

  // If the return type is smaller then call through to the narrowing
  // conversion.
  return absl::bit_cast<U>(
      NarrowingConversion<IntReturnType, FpReturnType>(rm));
}

template <typename T>
template <typename IntReturnType, typename FpReturnType>
void FpConversionsTestHelper<T>::NarrowingConversionMakeExponentAndSignificand(
    FPRoundingMode rm, IntReturnType &out_exponent,
    IntReturnType &out_significand) {
  int32_t e_max = FPTypeInfo<FpReturnType>::kExpBias;
  int32_t e_min = 1 - e_max;
  IntType in_exponent = (unsigned_value_ & FPTypeInfo<FpType>::kExpMask) >>
                        FPTypeInfo<FpType>::kSigSize;
  IntType in_significand = unsigned_value_ & FPTypeInfo<FpType>::kSigMask;
  // Add the implicit bit to the significand.
  if (in_exponent) {
    in_significand |= 1ULL << FPTypeInfo<FpType>::kSigSize;
  }
  int32_t exponent_bias_diff =
      FPTypeInfo<FpType>::kExpBias - FPTypeInfo<FpReturnType>::kExpBias;
  int32_t unbiased_exponent =
      static_cast<int32_t>(in_exponent) - FPTypeInfo<FpType>::kExpBias;
  int32_t significand_size_diff =
      FPTypeInfo<FpType>::kSigSize - FPTypeInfo<FpReturnType>::kSigSize;

  if (unbiased_exponent < e_min) {
    // The destination float will be subnormal.
    out_exponent = 0;
    int shift_amt = significand_size_diff + (e_min - unbiased_exponent);
    out_significand = RoundingRightShift(in_significand, shift_amt, rm);
  } else if (unbiased_exponent > e_max) {
    // The destination float will be infinity.
    out_exponent = FPTypeInfo<FpReturnType>::kExpMask >>
                   FPTypeInfo<FpReturnType>::kSigSize;
    out_significand = 0;
  } else {
    // The destination float will be normal.
    out_exponent = in_exponent - exponent_bias_diff;
    out_significand =
        RoundingRightShift(in_significand & FPTypeInfo<FpType>::kSigMask,
                           significand_size_diff, rm);
  }
  // Rounding can cause the significand to overflow. Remask and increment the
  // exponent to fix.
  if ((out_significand & FPTypeInfo<FpReturnType>::kSigMask) !=
      out_significand) {
    out_exponent =
        std::min(out_exponent + 1, FPTypeInfo<FpReturnType>::kExpMask >>
                                       FPTypeInfo<FpReturnType>::kSigSize);
    out_significand &= FPTypeInfo<FpReturnType>::kSigMask;
  }
}

template <typename T>
template <typename IntReturnType, typename FpReturnType>
IntReturnType FpConversionsTestHelper<T>::NarrowingConversionHandleInfinity(
    FPRoundingMode rm) {
  IntReturnType out_uint = sign() ? FPTypeInfo<FpReturnType>::kNegInf
                                  : FPTypeInfo<FpReturnType>::kPosInf;
  int32_t e_max = FPTypeInfo<FpReturnType>::kExpBias;
  double fp_value_double = internal::ToDouble(fp_value_);
  double largest_non_inf_double = internal::ToDouble<IntReturnType>(
      (sign() ? FPTypeInfo<FpReturnType>::kNegInf - 1
              : FPTypeInfo<FpReturnType>::kPosInf - 1));
  // To handle the cases near infinity, we need to consider what the
  // conversion would have been if the exponent was unbounded.
  double first_out_of_range_double =
      std::pow(2.0, e_max + 1) * std::pow(-1.0, sign());

  // Figure out if the input is closer to the largest non-inf or the unbounded
  // number.
  double distance_to_largest_non_inf =
      std::abs(fp_value_double - largest_non_inf_double);
  double distance_to_first_out_of_range =
      std::abs(fp_value_double - first_out_of_range_double);
  switch (rm) {
    case FPRoundingMode::kRoundToNearest:
      if (distance_to_largest_non_inf < distance_to_first_out_of_range) {
        out_uint = sign() ? FPTypeInfo<FpReturnType>::kNegInf - 1
                          : FPTypeInfo<FpReturnType>::kPosInf - 1;
      }
      break;
    case FPRoundingMode::kRoundTowardsZero:
      out_uint = sign() ? FPTypeInfo<FpReturnType>::kNegInf - 1
                        : FPTypeInfo<FpReturnType>::kPosInf - 1;
      break;
    case FPRoundingMode::kRoundDown:
      out_uint = sign() ? FPTypeInfo<FpReturnType>::kNegInf
                        : FPTypeInfo<FpReturnType>::kPosInf - 1;
      break;
    case FPRoundingMode::kRoundUp:
      out_uint = sign() ? FPTypeInfo<FpReturnType>::kNegInf - 1
                        : FPTypeInfo<FpReturnType>::kPosInf;
      break;
    default:
      break;
  }
  fflags_ |= static_cast<uint32_t>(FPExceptions::kOverflow);
  fflags_ |= static_cast<uint32_t>(FPExceptions::kInexact);
  return out_uint;
}

template <typename T>
template <typename IntReturnType, typename FpReturnType>
IntReturnType FpConversionsTestHelper<T>::NarrowingConversion(
    FPRoundingMode rm) {
  int32_t e_min = 1 - FPTypeInfo<FpReturnType>::kExpBias;
  IntReturnType out_exponent = 0;
  IntReturnType out_significand = 0;
  NarrowingConversionMakeExponentAndSignificand<IntReturnType, FpReturnType>(
      rm, out_exponent, out_significand);

  IntReturnType out_uint = sign() ? FPTypeInfo<FpReturnType>::kNegZero
                                  : FPTypeInfo<FpReturnType>::kPosZero;
  out_uint |= out_significand & FPTypeInfo<FpReturnType>::kSigMask;
  out_uint |= (out_exponent << FPTypeInfo<FpReturnType>::kSigSize) &
              FPTypeInfo<FpReturnType>::kExpMask;

  if (out_uint == FPTypeInfo<FpReturnType>::kPosInf ||
      out_uint == FPTypeInfo<FpReturnType>::kNegInf) {
    // Handle rounding and flags for infinity.
    out_uint =
        NarrowingConversionHandleInfinity<IntReturnType, FpReturnType>(rm);
  } else {
    // Handle the flags not related to infinity.
    double fp_value_double = internal::ToDouble<FpType>(fp_value_);
    double result = internal::ToDouble<IntReturnType>(out_uint);

    if (result != fp_value_double) {
      double b_emin = std::pow(2.0, e_min);
      if (std::abs(result) < b_emin || std::abs(fp_value_double) < b_emin) {
        fflags_ |= static_cast<uint32_t>(FPExceptions::kUnderflow);
      }
      fflags_ |= static_cast<uint32_t>(FPExceptions::kInexact);
    }
  }
  return out_uint;
}

template <typename T>
FpConversionsTestHelper(T) -> FpConversionsTestHelper<T>;

}  // namespace test
}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_TEST_RISCV_FP_TEST_BASE_H_
