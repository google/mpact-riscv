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

#include <cstdint>
#include <functional>
#include <limits>
#include <tuple>
#include <type_traits>

#include "absl/log/log.h"
#include "mpact/sim/generic/arch_state.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/operand_interface.h"
#include "mpact/sim/generic/register.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_fp_host.h"
#include "riscv/riscv_fp_info.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::operator*;
using ::mpact::sim::generic::FPTypeInfo;

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
inline void RiscVConvertFloatWithFflagsOp(const Instruction* instruction) {
  constexpr To kMax = std::numeric_limits<To>::max();
  constexpr To kMin = std::numeric_limits<To>::min();

  From lhs = generic::GetInstructionSource<From>(instruction, 0);
  using FromUint = typename FPTypeInfo<From>::UIntType;
  FromUint lhs_u = *reinterpret_cast<FromUint*>(&lhs);
  auto constexpr kExpMask = FPTypeInfo<From>::kExpMask;
  auto constexpr kSigMask = FPTypeInfo<From>::kSigMask;
  uint32_t flags = 0;
  uint32_t rm = generic::GetInstructionSource<uint32_t>(instruction, 1);
  // Dynamic rounding mode will get rounding mode from the global state.
  if (rm == *FPRoundingMode::kDynamic) {
    auto* rv_fp = static_cast<RiscVState*>(instruction->state())->rv_fp();
    if (!rv_fp->rounding_mode_valid()) {
      LOG(ERROR) << "Invalid rounding mode";
      return;
    }
    rm = *rv_fp->GetRoundingMode();
  }
  To value = 0;
  if (FPTypeInfo<From>::IsNaN(lhs) || lhs_u == FPTypeInfo<From>::kPosInf) {
    value = std::numeric_limits<To>::max();
    flags = *FPExceptions::kInvalidOp;
  } else if (lhs_u == FPTypeInfo<From>::kNegInf) {
    value = std::numeric_limits<To>::min();
    flags = *FPExceptions::kInvalidOp;
  } else if ((lhs_u & (kExpMask | kSigMask)) == 0) {  // lhs == 0.0
    value = 0;
  } else {
    // static_cast<>() doesn't necessarily round, so will have to force
    // rounding before converting to the integer type if necessary.
    using FromUint = typename FPTypeInfo<From>::UIntType;
    auto constexpr kBias = FPTypeInfo<From>::kExpBias;
    auto constexpr kExpMask = FPTypeInfo<From>::kExpMask;
    auto constexpr kSigSize = FPTypeInfo<From>::kSigSize;
    auto constexpr kSigMask = FPTypeInfo<From>::kSigMask;
    auto constexpr kBitSize = FPTypeInfo<From>::kBitSize;
    FromUint lhs_u = *reinterpret_cast<FromUint*>(&lhs);
    const bool sign = (lhs_u & (1ULL << (kBitSize - 1))) != 0;
    FromUint exp = kExpMask & lhs_u;
    int exp_value = exp >> kSigSize;
    int unbiased_exp = exp_value - kBias;
    FromUint sig = kSigMask & lhs_u;

    // Get fraction part of the number, and right shift it to leave 1 bits
    // length of fraction part.
    // In forms of "<integer_value>.<fraction>"", where fraction is 2 bits.
    // e.g., 1.75 -> 0b1.11
    //  (float32) -> base = 0b110'0000'0000'0000'0000'0000'(23bits)
    //               exp = 127 (unbiased exp = 0)
    // After right shift to leave 1 bit in fraction part.
    // base = 0b1
    // rightshift_compressed = 1 (compress the right-shift eliminated number)
    int right_shift = exp ? kSigSize - 1 - unbiased_exp : 1;
    uint64_t base = sig;
    bool rightshift_compressed = 0;
    if (exp == 0) {
      // Denormalized value.
      // Format: (-1)^sign * 2 ^(exp - bias + 1) * 0.{sig}
      // fraction part is too small keep it as 1 bits if not zero.
      rightshift_compressed = base != 0;
      base = 0;
      flags = *FPExceptions::kInexact;
    } else {
      // Normalized value.
      // Format: (-1)^sign * 2 ^(exp - bias) * 1.{sig
      // base = 1.{sig} (total (1 + `kSigSize`) bits)
      base |= 1ULL << kSigSize;

      // Right shift all base part out.
      if (right_shift > (kBitSize - 1)) {
        rightshift_compressed = base != 0;
        flags = *FPExceptions::kInexact;
        base = 0;
      } else if (right_shift > 0) {
        // Right shift to leave only 1 bit in the fraction part, compressed the
        // right-shift eliminated number.
        right_shift = std::min(right_shift, kBitSize);
        uint64_t right_shifted_sig_mask = (1ULL << right_shift) - 1;
        rightshift_compressed = (base & right_shifted_sig_mask) != 0;
        base >>= right_shift;
      }
    }

    // Handle fraction part rounding.
    if (right_shift >= 0) {
      switch (rm) {
        case *FPRoundingMode::kRoundToNearest:
          // 0.5, tie condition
          if (rightshift_compressed == 0 && base & 0b1) {
            // <odd>.5 -> <odd>.5 + 0.5 = even
            if ((base & 0b11) == 0b11) {
              flags = *FPExceptions::kInexact;
              base += 0b01;
            }
          } else if (base & 0b1 || rightshift_compressed) {
            // not tie condition, round to nearest integer, it equals to add
            // 0.5(=base + 0b01) and eliminate the fraction part.
            base += 0b01;
          }
          break;
        case *FPRoundingMode::kRoundTowardsZero:
          // Round towards zero will eliminate the fraction part.
          // Do nothing on fraction part.
          // 1.2 -> 1.0, -1.5 -> -1.0, -0.7 -> 0.0
          break;
        case *FPRoundingMode::kRoundDown:
          // Positive float will eliminate the fraction part.
          // Negative float with fraction part will subtract 1(= base + 0b10),
          // and eliminate the fraction part.
          // e.g., 1.2 -> 1.0, -1.5 -> -2.0, -0.7 -> -1.0
          if (sign && (base & 0b1 || rightshift_compressed)) {
            base += 0b10;
          }
          break;
        case *FPRoundingMode::kRoundUp:
          // Positive float will add 1(= base + 0b10), and eliminate the
          // fraction part.
          // Negative float will eliminate the fraction part.
          // e.g., 1.2 -> 2.0, -1.5 -> -1.0, -0.7 -> 0.0
          if (!sign && (base & 0b1 || rightshift_compressed)) {
            base += 0b10;
          }
          break;
        case *FPRoundingMode::kRoundToNearestTiesToMax:
          // Round to nearest integer that is far from zero.
          // e.g., 1.2 -> 2.0, -1.5 -> -2.0, -0.7 -> -1.0
          if (base & 0b1 || rightshift_compressed) {
            base += 0b1;
          }
          break;
        default:
          LOG(ERROR) << "Invalid rounding mode";
          return;
      }
    }
    uint64_t unsigned_value;
    // Handle base with fraction part and store it to `unsigned_value`.
    if (right_shift >= 0) {
      // Set inexact flag if floating value has fraction part.
      if (base & 0b1 || rightshift_compressed) {
        flags = *FPExceptions::kInexact;
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
          flags = *FPExceptions::kInvalidOp;
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
        flags = *FPExceptions::kInvalidOp;
        value = kMax;
      } else if (sign && (unsigned_value > 0 && -unsigned_value < kMin)) {
        // Negative value but exceeds the min value.
        flags = *FPExceptions::kInvalidOp;
        value = kMin;
      } else {
        value = sign ? -((To)unsigned_value) : unsigned_value;
      }
    } else {
      // Positive value but exceeds the max value.
      if (unsigned_value > kMax) {
        flags = *FPExceptions::kInvalidOp;
        value = sign ? kMin : kMax;
      } else if (sign && unsigned_value != 0) {
        // float is negative value this is out of range of valid unsigned value.
        flags = *FPExceptions::kInvalidOp;
        value = kMin;
      } else {
        value = sign ? -((To)unsigned_value) : unsigned_value;
      }
    }
  }
  using SignedTo = typename std::make_signed<To>::type;
  auto* dest = instruction->Destination(0);
  auto* reg_dest =
      static_cast<generic::RegisterDestinationOperand<Result>*>(dest);
  auto* reg = reg_dest->GetRegister();
  // The final value is sign-extended to the register width, even if it's
  // conversion to an unsigned value.
  SignedTo signed_value = static_cast<SignedTo>(value);
  Result dest_value = static_cast<Result>(signed_value);
  reg->data_buffer()->template Set<Result>(0, dest_value);
  if (flags) {
    auto* flag_db = instruction->Destination(1)->AllocateDataBuffer();
    flag_db->Set<uint32_t>(0, flags);
    flag_db->Submit();
  }
}

// Helper function to read a NaN boxed source value, converting it to NaN if
// it isn't formatted properly.
template <typename RegValue, typename Argument>
inline Argument GetNaNBoxedSource(const Instruction* instruction, int arg) {
  if (sizeof(RegValue) <= sizeof(Argument)) {
    return generic::GetInstructionSource<Argument>(instruction, arg);
  } else {
    using UInt = typename std::make_unsigned<RegValue>::type;
    UInt uval = generic::GetInstructionSource<UInt>(instruction, arg);
    UInt mask = std::numeric_limits<UInt>::max() << (sizeof(Argument) * 8);
    if (((mask & uval) != mask)) {
      return *reinterpret_cast<const Argument*>(
          &FPTypeInfo<Argument>::kCanonicalNaN);
    }
    return generic::GetInstructionSource<Argument>(instruction, arg);
  }
}

// Generic helper function for binary instructions.
template <typename Register, typename Result, typename Argument>
inline void RiscVBinaryOp(const Instruction* instruction,
                          std::function<Result(Argument, Argument)> operation) {
  using RegValue = typename Register::ValueType;
  Argument lhs = generic::GetInstructionSource<Argument>(instruction, 0);
  Argument rhs = generic::GetInstructionSource<Argument>(instruction, 1);
  Result dest_value = operation(lhs, rhs);
  auto* reg = static_cast<generic::RegisterDestinationOperand<RegValue>*>(
                  instruction->Destination(0))
                  ->GetRegister();
  reg->data_buffer()->template Set<Result>(0, dest_value);
}

// Generic helper function for writing a value to a register by destination
// operand index.
template <typename Register, typename Value>
inline void RiscVWriteReg(const Instruction* instruction, int index,
                          Value value) {
  auto* reg = static_cast<generic::RegisterDestinationOperand<Value>*>(
                  instruction->Destination(index))
                  ->GetRegister();
  reg->data_buffer()->template Set<Value>(0, value);
}

// Generic helper function for unary instructions.
template <typename Register, typename Result, typename Argument>
inline void RiscVUnaryOp(const Instruction* instruction,
                         std::function<Result(Argument)> operation) {
  using RegValue = typename Register::ValueType;
  auto lhs = generic::GetInstructionSource<Argument>(instruction, 0);
  Result dest_value = operation(lhs);
  auto* reg = static_cast<generic::RegisterDestinationOperand<RegValue>*>(
                  instruction->Destination(0))
                  ->GetRegister();
  reg->data_buffer()->template Set<Result>(0, dest_value);
}

// Helper function for conditional branches.
template <typename RegisterType, typename ValueType>
static inline void BranchConditional(
    const Instruction* instruction,
    std::function<bool(ValueType, ValueType)> cond) {
  using UIntType =
      typename std::make_unsigned<typename RegisterType::ValueType>::type;
  ValueType a = generic::GetInstructionSource<ValueType>(instruction, 0);
  ValueType b = generic::GetInstructionSource<ValueType>(instruction, 1);
  if (cond(a, b)) {
    UIntType offset = generic::GetInstructionSource<UIntType>(instruction, 2);
    UIntType target = offset + instruction->address();
    // Check target for proper alignment.
    auto* state = static_cast<RiscVState*>(instruction->state());
    auto res =
        state->csr_set()->GetCsr(static_cast<uint64_t>(RiscVCsrEnum::kMIsa));
    bool has_c_extension = true;
    if (res.ok() || res.value() != nullptr) {
      has_c_extension =
          (res.value()->GetUint64() &
           static_cast<uint64_t>(IsaExtensions::kCompressed)) != 0;
    }
    if (!has_c_extension && ((target & 0x3) != 0)) {
      state->Trap(/*is_interrupt*/ false, instruction->address(),
                  *ExceptionCode::kInstructionAddressMisaligned,
                  instruction->address(), instruction);
      return;
    }
    auto* db = instruction->Destination(0)->AllocateDataBuffer();
    db->SetSubmit<UIntType>(0, target);
    state->set_branch(true);
  }
}

// Generic helper function for load instructions.
template <typename Register, typename ValueType>
inline void RVLoad(const Instruction* instruction) {
  using RegVal = typename Register::ValueType;
  using URegVal = typename std::make_unsigned<RegVal>::type;
  URegVal base = generic::GetInstructionSource<URegVal>(instruction, 0);
  RegVal offset = generic::GetInstructionSource<RegVal>(instruction, 1);
  URegVal address = base + offset;
  auto* value_db =
      instruction->state()->db_factory()->Allocate(sizeof(ValueType));
  value_db->set_latency(0);
  auto* context = new LoadContext(value_db);
  auto* state = static_cast<RiscVState*>(instruction->state());
  state->LoadMemory(instruction, address, value_db, instruction->child(),
                    context);
  context->DecRef();
}

// Generic helper function for load instructions' "child instruction".
template <typename Register, typename ValueType>
inline void RVLoadChild(const Instruction* instruction) {
  using RegVal = typename Register::ValueType;
  using URegVal = typename std::make_unsigned<RegVal>::type;
  using SRegVal = typename std::make_signed<URegVal>::type;
  LoadContext* context = static_cast<LoadContext*>(instruction->context());
  auto* reg = static_cast<generic::RegisterDestinationOperand<RegVal>*>(
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
inline void RVStore(const Instruction* instruction) {
  using URegVal =
      typename std::make_unsigned<typename RegisterType::ValueType>::type;
  using SRegVal = typename std::make_signed<URegVal>::type;
  URegVal base = generic::GetInstructionSource<URegVal>(instruction, 0);
  SRegVal offset = generic::GetInstructionSource<SRegVal>(instruction, 1);
  URegVal address = base + offset;
  ValueType value = generic::GetInstructionSource<ValueType>(instruction, 2);
  auto* state = static_cast<RiscVState*>(instruction->state());
  auto* db = state->db_factory()->Allocate(sizeof(ValueType));
  db->Set<ValueType>(0, value);
  state->StoreMemory(instruction, address, db);
  db->DecRef();
}

// Generic helper function for binary instructions with NaN boxing. This is
// used for those instructions that produce results in fp registers, but are
// not really executing an fp operation that requires rounding.
template <typename RegValue, typename Result, typename Argument>
inline void RiscVBinaryNaNBoxOp(
    const Instruction* instruction,
    std::function<Result(Argument, Argument)> operation) {
  Argument lhs = GetNaNBoxedSource<RegValue, Argument>(instruction, 0);
  Argument rhs = GetNaNBoxedSource<RegValue, Argument>(instruction, 1);
  Result dest_value = operation(lhs, rhs);
  auto* reg = static_cast<generic::RegisterDestinationOperand<RegValue>*>(
                  instruction->Destination(0))
                  ->GetRegister();
  // Check to see if we need to NaN box the result.
  if (sizeof(RegValue) > sizeof(Result)) {
    // If the floating point value is narrower than the register, the upper
    // bits have to be set to all ones.
    using UReg = typename std::make_unsigned<RegValue>::type;
    using UInt = typename FPTypeInfo<Result>::UIntType;
    auto dest_u_value = *reinterpret_cast<UInt*>(&dest_value);
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
inline void RiscVUnaryNaNBoxOp(const Instruction* instruction,
                               std::function<Result(Argument)> operation) {
  Argument lhs = GetNaNBoxedSource<SrcRegValue, Argument>(instruction, 0);
  Result dest_value = operation(lhs);
  auto* reg = static_cast<generic::RegisterDestinationOperand<DstRegValue>*>(
                  instruction->Destination(0))
                  ->GetRegister();
  // Check to see if we need to NaN box the result.
  if (sizeof(DstRegValue) > sizeof(Result)) {
    // If the floating point value is narrower than the register, the upper
    // bits have to be set to all ones.
    using UReg = typename std::make_unsigned<DstRegValue>::type;
    using UInt = typename FPTypeInfo<Result>::UIntType;
    auto dest_u_value = *reinterpret_cast<UInt*>(&dest_value);
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
inline void RiscVUnaryFloatNaNBoxOp(const Instruction* instruction,
                                    std::function<Result(Argument)> operation) {
  using ResUint = typename FPTypeInfo<Result>::UIntType;
  Argument lhs = GetNaNBoxedSource<SrcRegValue, Argument>(instruction, 0);
  // Get the rounding mode.
  int rm_value = generic::GetInstructionSource<int>(instruction, 1);

  // If the rounding mode is dynamic, read it from the current state.
  auto* rv_fp = static_cast<RiscVState*>(instruction->state())->rv_fp();
  if (rm_value == *FPRoundingMode::kDynamic) {
    if (!rv_fp->rounding_mode_valid()) {
      LOG(ERROR) << "Invalid rounding mode";
      return;
    }
    rm_value = *(rv_fp->GetRoundingMode());
  }
  Result dest_value;
  {
    ScopedFPStatus set_fp_status(rv_fp->host_fp_interface(), rm_value);
    dest_value = operation(lhs);
  }
  if (FPTypeInfo<Result>::IsNaN(dest_value) &&
      FPTypeInfo<Result>::SignBit(dest_value)) {
    ResUint res_value = *reinterpret_cast<ResUint*>(&dest_value);
    res_value &= FPTypeInfo<Result>::kInfMask;
    dest_value = *reinterpret_cast<Result*>(&res_value);
  }
  auto* dest = instruction->Destination(0);
  auto* reg_dest =
      static_cast<generic::RegisterDestinationOperand<DstRegValue>*>(dest);
  auto* reg = reg_dest->GetRegister();
  // Check to see if we need to NaN box the result.
  if (sizeof(DstRegValue) > sizeof(Result)) {
    // If the floating point Value is narrower than the register, the upper
    // bits have to be set to all ones.
    using UReg = typename std::make_unsigned<DstRegValue>::type;
    using UInt = typename FPTypeInfo<Result>::UIntType;
    auto dest_u_value = *reinterpret_cast<UInt*>(&dest_value);
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
inline void RiscVUnaryFloatOp(const Instruction* instruction,
                              std::function<Result(Argument)> operation) {
  Argument lhs = generic::GetInstructionSource<Argument>(instruction, 0);
  // Get the rounding mode.
  int rm_value = generic::GetInstructionSource<int>(instruction, 1);

  auto* rv_fp = static_cast<RiscVState*>(instruction->state())->rv_fp();
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
    ScopedFPStatus set_fp_status(rv_fp->host_fp_interface(), rm_value);
    dest_value = operation(lhs);
  }
  auto* dest = instruction->Destination(0);
  using UInt = typename FPTypeInfo<Result>::UIntType;
  auto* reg_dest =
      static_cast<generic::RegisterDestinationOperand<UInt>*>(dest);
  auto* reg = reg_dest->GetRegister();
  reg->data_buffer()->template Set<Result>(0, dest_value);
}

// Generic helper function for floating op instructions that do not require
// NaN boxing since they produce non fp-values, but set fflags.
template <typename Result, typename Argument>
inline void RiscVUnaryFloatWithFflagsOp(
    const Instruction* instruction,
    std::function<Result(Argument, uint32_t&)> operation) {
  Argument lhs = generic::GetInstructionSource<Argument>(instruction, 0);
  // Get the rounding mode.
  int rm_value = generic::GetInstructionSource<int>(instruction, 1);

  auto* rv_fp = static_cast<RiscVState*>(instruction->state())->rv_fp();
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
    ScopedFPStatus set_fp_status(rv_fp->host_fp_interface(), rm_value);
    dest_value = operation(lhs, flag);
  }
  auto* dest = instruction->Destination(0);
  using UInt = typename FPTypeInfo<Result>::UIntType;
  auto* reg_dest =
      static_cast<generic::RegisterDestinationOperand<UInt>*>(dest);
  auto* reg = reg_dest->GetRegister();
  reg->data_buffer()->template Set<Result>(0, dest_value);
  auto* flag_db = instruction->Destination(1)->AllocateDataBuffer();
  flag_db->Set<uint32_t>(0, flag);
  flag_db->Submit();
}

// Generic helper function for binary floating point instructions. The main
// difference is that it handles rounding mode.
template <typename Register, typename Result, typename Argument>
inline void RiscVBinaryFloatNaNBoxOp(
    const Instruction* instruction,
    std::function<Result(Argument, Argument)> operation) {
  Argument lhs = GetNaNBoxedSource<Register, Argument>(instruction, 0);
  Argument rhs = GetNaNBoxedSource<Register, Argument>(instruction, 1);

  // Get the rounding mode.
  int rm_value = generic::GetInstructionSource<int>(instruction, 2);

  auto* rv_fp = static_cast<RiscVState*>(instruction->state())->rv_fp();
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
    ScopedFPStatus fp_status(rv_fp->host_fp_interface(), rm_value);
    dest_value = operation(lhs, rhs);
  }
  if (FPTypeInfo<Result>::IsNaN(dest_value)) {
    *reinterpret_cast<typename FPTypeInfo<Result>::UIntType*>(&dest_value) =
        FPTypeInfo<Result>::kCanonicalNaN;
  }
  auto* reg = static_cast<generic::RegisterDestinationOperand<Register>*>(
                  instruction->Destination(0))
                  ->GetRegister();
  // Check to see if we need to NaN box the result.
  if (sizeof(Register) > sizeof(Result)) {
    // If the floating point value is narrower than the register, the upper
    // bits have to be set to all ones.
    using UReg = typename std::make_unsigned<Register>::type;
    using UInt = typename FPTypeInfo<Result>::UIntType;
    auto dest_u_value = *reinterpret_cast<UInt*>(&dest_value);
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
    const Instruction* instruction,
    std::function<Result(Argument, Argument, Argument)> operation) {
  Argument rs1 = generic::GetInstructionSource<Argument>(instruction, 0);
  Argument rs2 = generic::GetInstructionSource<Argument>(instruction, 1);
  Argument rs3 = generic::GetInstructionSource<Argument>(instruction, 2);
  // Get the rounding mode.
  int rm_value = generic::GetInstructionSource<int>(instruction, 3);

  auto* rv_fp = static_cast<RiscVState*>(instruction->state())->rv_fp();
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
    ScopedFPStatus fp_status(rv_fp->host_fp_interface(), rm_value);
    dest_value = operation(rs1, rs2, rs3);
  }
  auto* reg = static_cast<generic::RegisterDestinationOperand<Register>*>(
                  instruction->Destination(0))
                  ->GetRegister();
  // Check to see if we need to NaN box the result.
  if (sizeof(Register) > sizeof(Result)) {
    // If the floating point value is narrower than the register, the upper
    // bits have to be set to all ones.
    using UReg = typename std::make_unsigned<Register>::type;
    using UInt = typename FPTypeInfo<Result>::UIntType;
    auto dest_u_value = *reinterpret_cast<UInt*>(&dest_value);
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
  auto int_value = *reinterpret_cast<UIntType*>(&val);
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
