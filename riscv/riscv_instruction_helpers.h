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

#ifndef MPACT_RISCV_RISCV_RISCV_INSTRUCTION_HELPERS_H_
#define MPACT_RISCV_RISCV_RISCV_INSTRUCTION_HELPERS_H_

#include <limits>
#include <tuple>

#include "absl/log/log.h"
#include "mpact/sim/generic/arch_state.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/operand_interface.h"
#include "mpact/sim/generic/register.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

// Templated helper function for convert instruction semantic functions.
template <typename From, typename To>
inline std::tuple<To, uint32_t> CvtHelper(From value) {
  constexpr From kMax = static_cast<From>(std::numeric_limits<To>::max());
  constexpr From kMin = static_cast<From>(std::numeric_limits<To>::min());

  if (FPTypeInfo<From>::IsNaN(value)) {
    return std::make_tuple(std::numeric_limits<To>::max(),
                           *FPExceptions::kInvalidOp);
  }
  if (value > kMax) {
    return std::make_tuple(std::numeric_limits<To>::max(),
                           *FPExceptions::kInvalidOp);
  }
  if (value < kMin) {
    if (std::is_unsigned<To>::value && (value > -1.0)) {
      using SignedTo = typename std::make_signed<To>::type;
      SignedTo signed_val = static_cast<SignedTo>(value);
      if (signed_val == 0) {
        return std::make_tuple(0, *FPExceptions::kInexact);
      }
    }
    return std::make_tuple(std::numeric_limits<To>::min(),
                           *FPExceptions::kInvalidOp);
  }

  auto output_value = static_cast<To>(value);
  return std::make_tuple(output_value, 0);
}

// Generic helper function for floating op instructions that do not require
// NaN boxing since they produce non fp-values, but set fflags.
template <typename Result, typename From, typename To>
inline void RiscVConvertFloatWithFflagsOp(const Instruction *instruction) {
  constexpr From kMax = static_cast<From>(std::numeric_limits<To>::max());
  constexpr From kMin = static_cast<From>(std::numeric_limits<To>::min());

  From lhs = generic::GetInstructionSource<From>(instruction, 0);

  uint32_t flags = 0;
  uint32_t rm = generic::GetInstructionSource<uint32_t>(instruction, 1);
  To value = 0;
  if (FPTypeInfo<From>::IsNaN(lhs)) {
    value = std::numeric_limits<To>::max();
    flags = *FPExceptions::kInvalidOp;
  } else if (lhs >= kMax) {
    value = std::numeric_limits<To>::max();
    flags = *FPExceptions::kInvalidOp;
  } else if (lhs < kMin) {
    bool is_set = false;
    if (std::is_unsigned<To>::value && (lhs > -1.0)) {
      using SignedTo = typename std::make_signed<To>::type;
      SignedTo signed_val = static_cast<SignedTo>(lhs);
      if (signed_val == 0) {
        value = 0;
        flags = *FPExceptions::kInexact;
        is_set = true;
      }
    }
    if (!is_set) {
      value = std::numeric_limits<To>::min();
      flags = *FPExceptions::kInvalidOp;
      is_set = true;
    }
  } else if (lhs == 0.0) {
    value = 0;
  } else {
    // static_cast<>() doesn't necessarily round, so will have to force
    // rounding before converting to the integer type if necessary.
    using FromUint = typename FPTypeInfo<From>::UIntType;
    auto constexpr kBias = FPTypeInfo<From>::kExpBias;
    auto constexpr kExpMask = FPTypeInfo<From>::kExpMask;
    auto constexpr kSigSize = FPTypeInfo<From>::kSigSize;
    auto constexpr kSigMask = FPTypeInfo<From>::kSigMask;
    FromUint lhs_u = *reinterpret_cast<FromUint *>(&lhs);
    FromUint exp = kExpMask & lhs_u;
    int exp_value = exp >> kSigSize;
    int unbiased_exp = exp_value - kBias;
    FromUint sig = kSigMask & lhs_u;
    // If the number of bits in the significand is greater or equal to
    // unbiased exponent, and there is a 1 among the extra bits, we need to
    // perform rounding.
    if (unbiased_exp < 0) {
      flags = *FPExceptions::kInexact;
    } else if (unbiased_exp <= kSigSize) {
      FromUint mask = (1ULL << (kSigSize - unbiased_exp)) - 1;
      if ((sig & mask) != 0) {
        flags = *FPExceptions::kInexact;
        FromUint sign = lhs_u & (1ULL << (FPTypeInfo<From>::kBitSize - 1));
        // Turn the value into a denormal.
        constexpr FromUint hidden = 1ULL << (kSigSize - 1);
        FromUint tmp_u = sign | hidden | (sig >> 1);
        From tmp = *reinterpret_cast<From *>(&tmp_u);
        // Divide so that only the bits we care about are left in the
        // significand.
        int shift = kBias + kSigSize - exp_value - 1;
        FromUint div_exp = shift + kBias;
        FromUint div_u = div_exp << kSigSize;
        From div = *reinterpret_cast<From *>(&div_u);
        auto *rv_fp = static_cast<RiscVState *>(instruction->state())->rv_fp();
        {
          // The rounding happens during this division.
          ScopedFPRoundingMode set_fp_rm(rv_fp, rm);
          tmp /= div;
        }
        // Convert back to normalized number, by using the original sign
        // and exponent, and the normalized and significand from the division.
        tmp_u = *reinterpret_cast<FromUint *>(&tmp);
        lhs_u = sign | exp | ((tmp_u << (shift + 1)) & kSigMask);
        lhs = *reinterpret_cast<From *>(&lhs_u);
      }
    }
    value = static_cast<To>(lhs);
  }
  using SignedTo = typename std::make_signed<To>::type;
  auto *dest = instruction->Destination(0);
  auto *reg_dest =
      static_cast<generic::RegisterDestinationOperand<Result> *>(dest);
  auto *reg = reg_dest->GetRegister();
  // The final value is sign-extended to the register width, even if it's
  // conversion to an unsigned value.
  SignedTo signed_value = static_cast<SignedTo>(value);
  Result dest_value = static_cast<Result>(signed_value);
  reg->data_buffer()->template Set<Result>(0, dest_value);
  if (flags) {
    auto *flag_db = instruction->Destination(1)->AllocateDataBuffer();
    flag_db->Set<uint32_t>(0, flags);
    flag_db->Submit();
  }
}

// Helper function to read a NaN boxed source value, converting it to NaN if
// it isn't formatted properly.
template <typename RegValue, typename Argument>
inline Argument GetNaNBoxedSource(const Instruction *instruction, int arg) {
  if (sizeof(RegValue) <= sizeof(Argument)) {
    return generic::GetInstructionSource<Argument>(instruction, arg);
  } else {
    using SInt = typename std::make_signed<RegValue>::type;
    using UInt = typename std::make_unsigned<RegValue>::type;
    SInt val = generic::GetInstructionSource<SInt>(instruction, arg);
    UInt uval = static_cast<UInt>(val);
    UInt mask = std::numeric_limits<UInt>::max() << (sizeof(Argument) * 8);
    if (((mask & uval) != mask)) {
      return *reinterpret_cast<const Argument *>(
          &FPTypeInfo<Argument>::kCanonicalNaN);
    }
    return generic::GetInstructionSource<Argument>(instruction, arg);
  }
}

// Generic helper function for binary instructions.
template <typename Register, typename Result, typename Argument>
inline void RiscVBinaryOp(const Instruction *instruction,
                          std::function<Result(Argument, Argument)> operation) {
  using RegValue = typename Register::ValueType;
  Argument lhs = generic::GetInstructionSource<Argument>(instruction, 0);
  Argument rhs = generic::GetInstructionSource<Argument>(instruction, 1);
  Result dest_value = operation(lhs, rhs);
  auto *reg = static_cast<generic::RegisterDestinationOperand<RegValue> *>(
                  instruction->Destination(0))
                  ->GetRegister();
  reg->data_buffer()->template Set<Result>(0, dest_value);
}

// Generic helper function for unary instructions.
template <typename Register, typename Result, typename Argument>
inline void RiscVUnaryOp(const Instruction *instruction,
                         std::function<Result(Argument)> operation) {
  using RegValue = typename Register::ValueType;
  auto lhs = generic::GetInstructionSource<Argument>(instruction, 0);
  Result dest_value = operation(lhs);
  auto *reg = static_cast<generic::RegisterDestinationOperand<RegValue> *>(
                  instruction->Destination(0))
                  ->GetRegister();
  reg->data_buffer()->template Set<Result>(0, dest_value);
}

// Helper function for conditional branches.
template <typename RegisterType, typename ValueType>
static inline void BranchConditional(
    const Instruction *instruction,
    std::function<bool(ValueType, ValueType)> cond) {
  using UIntType =
      typename std::make_unsigned<typename RegisterType::ValueType>::type;
  ValueType a = generic::GetInstructionSource<ValueType>(instruction, 0);
  ValueType b = generic::GetInstructionSource<ValueType>(instruction, 1);
  if (cond(a, b)) {
    UIntType offset = generic::GetInstructionSource<UIntType>(instruction, 2);
    UIntType target = offset + instruction->address();
    auto *db = instruction->Destination(0)->AllocateDataBuffer();
    db->SetSubmit<UIntType>(0, target);
  }
}

// Generic helper function for load instructions.
template <typename Register, typename ValueType>
inline void RVLoad(const Instruction *instruction) {
  using RegVal = typename Register::ValueType;
  using URegVal = typename std::make_unsigned<RegVal>::type;
  URegVal base = generic::GetInstructionSource<URegVal>(instruction, 0);
  RegVal offset = generic::GetInstructionSource<RegVal>(instruction, 1);
  URegVal address = base + offset;
  auto *value_db =
      instruction->state()->db_factory()->Allocate(sizeof(ValueType));
  value_db->set_latency(0);
  auto *context = new LoadContext(value_db);
  auto *state = static_cast<RiscVState *>(instruction->state());
  state->LoadMemory(instruction, address, value_db, instruction->child(),
                    context);
  context->DecRef();
}

// Generic helper function for load instructions' "child instruction".
template <typename Register, typename ValueType>
inline void RVLoadChild(const Instruction *instruction) {
  using RegVal = typename Register::ValueType;
  using URegVal = typename std::make_unsigned<RegVal>::type;
  using SRegVal = typename std::make_signed<URegVal>::type;
  LoadContext *context = static_cast<LoadContext *>(instruction->context());
  auto *reg = static_cast<generic::RegisterDestinationOperand<RegVal> *>(
                  instruction->Destination(0))
                  ->GetRegister();
  if (std::is_signed<ValueType>::value) {
    SRegVal value = static_cast<SRegVal>(context->value_db->Get<ValueType>(0));
    reg->data_buffer()->template Set<SRegVal>(0, value);
  } else {
    URegVal value = static_cast<URegVal>(context->value_db->Get<ValueType>(0));
    reg->data_buffer()->template Set<URegVal>(0, value);
  }
}

// Generic helper function for store instructions.
template <typename RegisterType, typename ValueType>
inline void RVStore(const Instruction *instruction) {
  using URegVal =
      typename std::make_unsigned<typename RegisterType::ValueType>::type;
  using SRegVal = typename std::make_signed<URegVal>::type;
  URegVal base = generic::GetInstructionSource<URegVal>(instruction, 0);
  SRegVal offset = generic::GetInstructionSource<SRegVal>(instruction, 1);
  URegVal address = base + offset;
  ValueType value = generic::GetInstructionSource<ValueType>(instruction, 2);
  auto *state = static_cast<RiscVState *>(instruction->state());
  auto *db = state->db_factory()->Allocate(sizeof(ValueType));
  db->Set<ValueType>(0, value);
  state->StoreMemory(instruction, address, db);
  db->DecRef();
}

// Generic helper function for binary instructions with NaN boxing. This is
// used for those instructions that produce results in fp registers, but are
// not really executing an fp operation that requires rounding.
template <typename RegValue, typename Result, typename Argument>
inline void RiscVBinaryNaNBoxOp(
    const Instruction *instruction,
    std::function<Result(Argument, Argument)> operation) {
  Argument lhs = GetNaNBoxedSource<RegValue, Argument>(instruction, 0);
  Argument rhs = GetNaNBoxedSource<RegValue, Argument>(instruction, 1);
  Result dest_value = operation(lhs, rhs);
  auto *reg = static_cast<generic::RegisterDestinationOperand<RegValue> *>(
                  instruction->Destination(0))
                  ->GetRegister();
  // Check to see if we need to NaN box the result.
  if (sizeof(RegValue) > sizeof(Result)) {
    // If the floating point value is narrower than the register, the upper
    // bits have to be set to all ones.
    using UReg = typename std::make_unsigned<RegValue>::type;
    using UInt = typename FPTypeInfo<Result>::UIntType;
    auto dest_u_value = *reinterpret_cast<UInt *>(&dest_value);
    UReg reg_value = std::numeric_limits<UReg>::max();
    int shift = 8 * (sizeof(RegValue) - sizeof(Result));
    reg_value = (reg_value << shift) | dest_u_value;
    reg->data_buffer()->template Set<RegValue>(0, reg_value);
    return;
  }
  reg->data_buffer()->template Set<Result>(0, dest_value);
}

// Generic helper function for unary instructions with NaN boxing.
template <typename DstRegValue, typename SrcRegValue, typename Result,
          typename Argument>
inline void RiscVUnaryNaNBoxOp(const Instruction *instruction,
                               std::function<Result(Argument)> operation) {
  Argument lhs = GetNaNBoxedSource<SrcRegValue, Argument>(instruction, 0);
  Result dest_value = operation(lhs);
  auto *reg = static_cast<generic::RegisterDestinationOperand<DstRegValue> *>(
                  instruction->Destination(0))
                  ->GetRegister();
  // Check to see if we need to NaN box the result.
  if (sizeof(DstRegValue) > sizeof(Result)) {
    // If the floating point value is narrower than the register, the upper
    // bits have to be set to all ones.
    using UReg = typename std::make_unsigned<DstRegValue>::type;
    using UInt = typename FPTypeInfo<Result>::UIntType;
    auto dest_u_value = *reinterpret_cast<UInt *>(&dest_value);
    UReg reg_value = std::numeric_limits<UReg>::max();
    int shift = 8 * (sizeof(DstRegValue) - sizeof(Result));
    reg_value = (reg_value << shift) | dest_u_value;
    reg->data_buffer()->template Set<DstRegValue>(0, reg_value);
    return;
  }
  reg->data_buffer()->template Set<Result>(0, dest_value);
}

// Generic helper function for unary floating point instructions. The main
// difference is that it handles rounding mode and performs NaN boxing.
template <typename DstRegValue, typename SrcRegValue, typename Result,
          typename Argument>
inline void RiscVUnaryFloatNaNBoxOp(const Instruction *instruction,
                                    std::function<Result(Argument)> operation) {
  using ResUint = typename FPTypeInfo<Result>::UIntType;
  Argument lhs = GetNaNBoxedSource<SrcRegValue, Argument>(instruction, 0);
  // Get the rounding mode.
  int rm_value = generic::GetInstructionSource<int>(instruction, 1);

  // If the rounding mode is dynamic, read it from the current state.
  auto *rv_fp = static_cast<RiscVState *>(instruction->state())->rv_fp();
  if (rm_value == *FPRoundingMode::kDynamic) {
    if (!rv_fp->rounding_mode_valid()) {
      LOG(ERROR) << "Invalid rounding mode";
      return;
    }
    rm_value = *(rv_fp->GetRoundingMode());
  }
  Result dest_value;
  {
    ScopedFPStatus set_fp_status(rv_fp, rm_value);
    dest_value = operation(lhs);
  }
  if (std::isnan(dest_value) && std::signbit(dest_value)) {
    ResUint res_value = *reinterpret_cast<ResUint *>(&dest_value);
    res_value &= FPTypeInfo<Result>::kInfMask;
    dest_value = *reinterpret_cast<Result *>(&res_value);
  }
  auto *dest = instruction->Destination(0);
  auto *reg_dest =
      static_cast<generic::RegisterDestinationOperand<DstRegValue> *>(dest);
  auto *reg = reg_dest->GetRegister();
  // Check to see if we need to NaN box the result.
  if (sizeof(DstRegValue) > sizeof(Result)) {
    // If the floating point Value is narrower than the register, the upper
    // bits have to be set to all ones.
    using UReg = typename std::make_unsigned<DstRegValue>::type;
    using UInt = typename FPTypeInfo<Result>::UIntType;
    auto dest_u_value = *reinterpret_cast<UInt *>(&dest_value);
    UReg reg_value = std::numeric_limits<UReg>::max();
    int shift = 8 * (sizeof(DstRegValue) - sizeof(Result));
    reg_value = (reg_value << shift) | dest_u_value;
    reg->data_buffer()->template Set<DstRegValue>(0, reg_value);
    return;
  }
  reg->data_buffer()->template Set<Result>(0, dest_value);
}

// Generic helper function for floating op instructions that do not require
// NaN boxing since they produce non fp-values.
template <typename Result, typename Argument>
inline void RiscVUnaryFloatOp(const Instruction *instruction,
                              std::function<Result(Argument)> operation) {
  Argument lhs = generic::GetInstructionSource<Argument>(instruction, 0);
  // Get the rounding mode.
  int rm_value = generic::GetInstructionSource<int>(instruction, 1);

  auto *rv_fp = static_cast<RiscVState *>(instruction->state())->rv_fp();
  // If the rounding mode is dynamic, read it from the current state.
  if (rm_value == *FPRoundingMode::kDynamic) {
    if (!rv_fp->rounding_mode_valid()) {
      LOG(ERROR) << "Invalid rounding mode";
      return;
    }
    rm_value = *rv_fp->GetRoundingMode();
  }
  Result dest_value;
  {
    ScopedFPStatus set_fp_status(rv_fp, rm_value);
    dest_value = operation(lhs);
  }
  auto *dest = instruction->Destination(0);
  using UInt = typename FPTypeInfo<Result>::UIntType;
  auto *reg_dest =
      static_cast<generic::RegisterDestinationOperand<UInt> *>(dest);
  auto *reg = reg_dest->GetRegister();
  reg->data_buffer()->template Set<Result>(0, dest_value);
}

// Generic helper function for floating op instructions that do not require
// NaN boxing since they produce non fp-values, but set fflags.
template <typename Result, typename Argument>
inline void RiscVUnaryFloatWithFflagsOp(
    const Instruction *instruction,
    std::function<Result(Argument, uint32_t &)> operation) {
  Argument lhs = generic::GetInstructionSource<Argument>(instruction, 0);
  // Get the rounding mode.
  int rm_value = generic::GetInstructionSource<int>(instruction, 1);

  auto *rv_fp = static_cast<RiscVState *>(instruction->state())->rv_fp();
  // If the rounding mode is dynamic, read it from the current state.
  if (rm_value == *FPRoundingMode::kDynamic) {
    if (!rv_fp->rounding_mode_valid()) {
      LOG(ERROR) << "Invalid rounding mode";
      return;
    }
    rm_value = *rv_fp->GetRoundingMode();
  }
  uint32_t flag = 0;
  Result dest_value;
  {
    ScopedFPStatus set_fp_status(rv_fp, rm_value);
    dest_value = operation(lhs, flag);
  }
  auto *dest = instruction->Destination(0);
  using UInt = typename FPTypeInfo<Result>::UIntType;
  auto *reg_dest =
      static_cast<generic::RegisterDestinationOperand<UInt> *>(dest);
  auto *reg = reg_dest->GetRegister();
  reg->data_buffer()->template Set<Result>(0, dest_value);
  auto *flag_db = instruction->Destination(1)->AllocateDataBuffer();
  flag_db->Set<uint32_t>(0, flag);
  flag_db->Submit();
}

// Generic helper function for binary floating point instructions. The main
// difference is that it handles rounding mode.
template <typename Register, typename Result, typename Argument>
inline void RiscVBinaryFloatNaNBoxOp(
    const Instruction *instruction,
    std::function<Result(Argument, Argument)> operation) {
  Argument lhs = GetNaNBoxedSource<Register, Argument>(instruction, 0);
  Argument rhs = GetNaNBoxedSource<Register, Argument>(instruction, 1);
  // Argument lhs = generic::GetInstructionSource<Argument>(instruction, 0);
  // Argument rhs = generic::GetInstructionSource<Argument>(instruction, 1);

  // Get the rounding mode.
  int rm_value = generic::GetInstructionSource<int>(instruction, 2);

  auto *rv_fp = static_cast<RiscVState *>(instruction->state())->rv_fp();
  // If the rounding mode is dynamic, read it from the current state.
  if (rm_value == *FPRoundingMode::kDynamic) {
    if (!rv_fp->rounding_mode_valid()) {
      LOG(ERROR) << "Invalid rounding mode";
      return;
    }
    rm_value = *rv_fp->GetRoundingMode();
  }
  Result dest_value;
  {
    ScopedFPStatus fp_status(rv_fp, rm_value);
    dest_value = operation(lhs, rhs);
  }
  if (std::isnan(dest_value)) {
    *reinterpret_cast<typename FPTypeInfo<Result>::UIntType *>(&dest_value) =
        FPTypeInfo<Result>::kCanonicalNaN;
  }
  auto *reg = static_cast<generic::RegisterDestinationOperand<Register> *>(
                  instruction->Destination(0))
                  ->GetRegister();
  // Check to see if we need to NaN box the result.
  if (sizeof(Register) > sizeof(Result)) {
    // If the floating point value is narrower than the register, the upper
    // bits have to be set to all ones.
    using UReg = typename std::make_unsigned<Register>::type;
    using UInt = typename FPTypeInfo<Result>::UIntType;
    auto dest_u_value = *reinterpret_cast<UInt *>(&dest_value);
    UReg reg_value = std::numeric_limits<UReg>::max();
    int shift = 8 * (sizeof(Register) - sizeof(Result));
    reg_value = (reg_value << shift) | dest_u_value;
    reg->data_buffer()->template Set<Register>(0, reg_value);
    return;
  }
  reg->data_buffer()->template Set<Result>(0, dest_value);
}

// Generic helper function for ternary floating point instructions.
template <typename Register, typename Result, typename Argument>
inline void RiscVTernaryFloatNaNBoxOp(
    const Instruction *instruction,
    std::function<Result(Argument, Argument, Argument)> operation) {
  Argument rs1 = generic::GetInstructionSource<Argument>(instruction, 0);
  Argument rs2 = generic::GetInstructionSource<Argument>(instruction, 1);
  Argument rs3 = generic::GetInstructionSource<Argument>(instruction, 2);
  // Get the rounding mode.
  int rm_value = generic::GetInstructionSource<int>(instruction, 3);

  auto *rv_fp = static_cast<RiscVState *>(instruction->state())->rv_fp();
  // If the rounding mode is dynamic, read it from the current state.
  if (rm_value == *FPRoundingMode::kDynamic) {
    if (!rv_fp->rounding_mode_valid()) {
      LOG(ERROR) << "Invalid rounding mode";
      return;
    }
    rm_value = *rv_fp->GetRoundingMode();
  }
  Result dest_value;
  {
    ScopedFPStatus fp_status(rv_fp, rm_value);
    dest_value = operation(rs1, rs2, rs3);
  }
  auto *reg = static_cast<generic::RegisterDestinationOperand<Register> *>(
                  instruction->Destination(0))
                  ->GetRegister();
  // Check to see if we need to NaN box the result.
  if (sizeof(Register) > sizeof(Result)) {
    // If the floating point value is narrower than the register, the upper
    // bits have to be set to all ones.
    using UReg = typename std::make_unsigned<Register>::type;
    using UInt = typename FPTypeInfo<Result>::UIntType;
    auto dest_u_value = *reinterpret_cast<UInt *>(&dest_value);
    UReg reg_value = std::numeric_limits<UReg>::max();
    int shift = 8 * (sizeof(Register) - sizeof(Result));
    reg_value = (reg_value << shift) | dest_u_value;
    reg->data_buffer()->template Set<Register>(0, reg_value);
    return;
  }
  reg->data_buffer()->template Set<Result>(0, dest_value);
}

// Helper function to classify floating point values.
template <typename T>
typename FPTypeInfo<T>::UIntType ClassifyFP(T val) {
  using UIntType = typename FPTypeInfo<T>::UIntType;
  auto int_value = *reinterpret_cast<UIntType *>(&val);
  UIntType sign = int_value >> (FPTypeInfo<T>::kBitSize - 1);
  UIntType exp_mask = (1 << FPTypeInfo<T>::kExpSize) - 1;
  UIntType exp = (int_value >> FPTypeInfo<T>::kSigSize) & exp_mask;
  UIntType sig =
      int_value & ((static_cast<UIntType>(1) << FPTypeInfo<T>::kSigSize) - 1);
  if (exp == 0) {    // The number is denormal or zero.
    if (sig == 0) {  // The number is zero.
      return sign ? 1 << 3 : 1 << 4;
    } else {  // subnormal.
      return sign ? 1 << 2 : 1 << 5;
    }
  } else if (exp == exp_mask) {  //  The number is infinity or NaN.
    if (sig == 0) {              // infinity
      return sign ? 1 : 1 << 7;
    } else {
      if ((sig >> (FPTypeInfo<T>::kSigSize - 1)) != 0) {  // Quiet NaN.
        return 1 << 9;
      } else {  // signaling NaN.
        return 1 << 8;
      }
    }
  }
  return sign ? 1 << 1 : 1 << 6;
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_INSTRUCTION_HELPERS_H_
