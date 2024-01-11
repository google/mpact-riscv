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
#include <functional>
#include <iostream>
#include <limits>
#include <tuple>
#include <type_traits>

#include "riscv/riscv_instruction_helpers.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/type_helpers.h"

namespace mpact {
namespace sim {
namespace riscv {

// The following instruction semantic functions implement the single precision
// floating point instructions in the RiscV architecture. They all utilize the
// templated helper functions in riscv_instruction_helpers.h to implement
// the boiler plate code.

using FPRegister = RVFpRegister;
using RegUInt = typename std::make_unsigned<RVFpRegister::ValueType>::type;

// These types are used instead of uint32_t and int32_t to represent the
// integer type of equal value to float when values of these types are
// really reinterpreted float values.
using FPUInt = FPTypeInfo<float>::UIntType;
using FPSInt = FPTypeInfo<float>::IntType;

// Note, for any SP operation on values in 64-bit DP registers, the input
// values have to be properly NaN-boxed. If not, the value is treated as
// a canonical NaN.

// Templated helper functions.

namespace internal {

// Convert float to signed 32 bit integer.
template <typename XInt>
static inline void RVFCvtWs(const Instruction *instruction) {
  RiscVConvertFloatWithFflagsOp<XInt, float, int32_t>(instruction);
}

// Convert float to unsigned 32 bit integer.
template <typename XInt>
static inline void RVFCvtWus(const Instruction *instruction) {
  RiscVConvertFloatWithFflagsOp<XInt, float, uint32_t>(instruction);
}

// Convert float to signed 64 bit integer.
template <typename XInt>
static inline void RVFCvtLs(const Instruction *instruction) {
  RiscVConvertFloatWithFflagsOp<XInt, float, int64_t>(instruction);
}

// Convert float to unsigned 64 bit integer.
template <typename XUint>
static inline void RVFCvtLus(const Instruction *instruction) {
  RiscVConvertFloatWithFflagsOp<XUint, float, uint64_t>(instruction);
}

// Single precision compare equal.
template <typename XRegister>
static inline void RVFCmpeq(const Instruction *instruction) {
  RiscVBinaryNaNBoxOp<typename XRegister::ValueType,
                      typename XRegister::ValueType, float>(
      instruction,
      [instruction](float a, float b) -> typename XRegister::ValueType {
        if (FPTypeInfo<float>::IsSNaN(a) || FPTypeInfo<float>::IsSNaN(b)) {
          auto *db = instruction->Destination(1)->AllocateDataBuffer();
          db->Set<uint32_t>(0, *FPExceptions::kInvalidOp);
          db->Submit();
        }
        return a == b;
      });
}

// Single precicion compare less than.
template <typename XRegister>
static inline void RVFCmplt(const Instruction *instruction) {
  RiscVBinaryNaNBoxOp<typename XRegister::ValueType,
                      typename XRegister::ValueType, float>(
      instruction,
      [instruction](float a, float b) -> typename XRegister::ValueType {
        if (FPTypeInfo<float>::IsNaN(a) || FPTypeInfo<float>::IsNaN(b)) {
          auto *db = instruction->Destination(1)->AllocateDataBuffer();
          db->Set<uint32_t>(0, *FPExceptions::kInvalidOp);
          db->Submit();
        }
        return a < b;
      });
}

// Single precision compare less than or equal.
template <typename XRegister>
static inline void RVFCmple(const Instruction *instruction) {
  RiscVBinaryNaNBoxOp<typename XRegister::ValueType,
                      typename XRegister::ValueType, float>(
      instruction,
      [instruction](float a, float b) -> typename XRegister::ValueType {
        if (FPTypeInfo<float>::IsNaN(a) || FPTypeInfo<float>::IsNaN(b)) {
          auto *db = instruction->Destination(1)->AllocateDataBuffer();
          db->Set<uint32_t>(0, *FPExceptions::kInvalidOp);
          db->Submit();
        }
        return a <= b;
      });
}

template <typename T>
static inline T CanonicalizeNaN(T value) {
  if (!std::isnan(value)) return value;
  auto nan_value = FPTypeInfo<T>::kCanonicalNaN;
  return *reinterpret_cast<T *>(&nan_value);
}

}  // namespace internal

// Load child instruction.
void RiscVIFlwChild(const Instruction *instruction) {
  LoadContext *context = static_cast<LoadContext *>(instruction->context());
  auto value = context->value_db->Get<FPUInt>(0);
  auto *reg =
      static_cast<generic::RegisterDestinationOperand<FPRegister::ValueType> *>(
          instruction->Destination(0))
          ->GetRegister();
  if (sizeof(FPRegister::ValueType) > sizeof(FPUInt)) {
    // NaN box the loaded value.
    auto reg_value = std::numeric_limits<FPRegister::ValueType>::max();
    reg_value <<= sizeof(FPUInt) * 8;
    reg_value |= value;
    reg->data_buffer()->Set<FPRegister::ValueType>(0, reg_value);
    return;
  }
  reg->data_buffer()->Set<FPRegister::ValueType>(0, value);
}

// Basic arithmetic instructions.
void RiscVFAdd(const Instruction *instruction) {
  RiscVBinaryFloatNaNBoxOp<FPRegister::ValueType, float, float>(
      instruction, [](float a, float b) { return a + b; });
}

void RiscVFSub(const Instruction *instruction) {
  RiscVBinaryFloatNaNBoxOp<FPRegister::ValueType, float, float>(
      instruction, [](float a, float b) { return a - b; });
}

void RiscVFMul(const Instruction *instruction) {
  RiscVBinaryFloatNaNBoxOp<FPRegister::ValueType, float, float>(
      instruction, [](float a, float b) { return a * b; });
}

void RiscVFDiv(const Instruction *instruction) {
  RiscVBinaryFloatNaNBoxOp<FPRegister::ValueType, float, float>(
      instruction, [](float a, float b) { return a / b; });
}

// Square root uses the library square root.
void RiscVFSqrt(const Instruction *instruction) {
  RiscVUnaryFloatNaNBoxOp<FPRegister::ValueType, FPRegister::ValueType, float,
                          float>(instruction, [](float a) -> float {
    float res = sqrt(a);
    if (std::isnan(res))
      return *reinterpret_cast<const float *>(
          &FPTypeInfo<float>::kCanonicalNaN);
    return res;
  });
}

// If either operand is NaN return the other.
void RiscVFMin(const Instruction *instruction) {
  RiscVBinaryOp<FPRegister, float, float>(
      instruction, [instruction](float a, float b) -> float {
        if (FPTypeInfo<float>::IsSNaN(a) || FPTypeInfo<float>::IsSNaN(b)) {
          auto *db = instruction->Destination(1)->AllocateDataBuffer();
          db->Set<uint32_t>(0, *FPExceptions::kInvalidOp);
          db->Submit();
        }
        if (FPTypeInfo<float>::IsNaN(a)) {
          if (FPTypeInfo<float>::IsNaN(b)) {
            FPTypeInfo<float>::UIntType not_a_number =
                FPTypeInfo<float>::kCanonicalNaN;
            return *reinterpret_cast<float *>(&not_a_number);
          }
          return b;
        }
        if (FPTypeInfo<float>::IsNaN(b)) return a;
        // If both are zero, return the negative zero if there is one.
        if ((a == 0.0) && (b == 0.0)) return (std::signbit(a)) ? a : b;
        return (a > b) ? b : a;
      });
}

// If either operand is NaN return the other.
void RiscVFMax(const Instruction *instruction) {
  RiscVBinaryOp<FPRegister, float, float>(
      instruction, [instruction](float a, float b) -> float {
        if (FPTypeInfo<float>::IsSNaN(a) || FPTypeInfo<float>::IsSNaN(b)) {
          auto *db = instruction->Destination(1)->AllocateDataBuffer();
          db->Set<uint32_t>(0, *FPExceptions::kInvalidOp);
          db->Submit();
        }
        if (FPTypeInfo<float>::IsNaN(a)) {
          if (FPTypeInfo<float>::IsNaN(b)) {
            FPTypeInfo<float>::UIntType not_a_number =
                FPTypeInfo<float>::kCanonicalNaN;
            return *reinterpret_cast<float *>(&not_a_number);
          }
          return b;
        }
        if (FPTypeInfo<float>::IsNaN(b)) return a;
        // If both are zero, return the positive zero if there is one.
        if ((a == 0.0) && (b == 0.0)) return (std::signbit(b)) ? a : b;
        return (a < b) ? b : a;
      });
}

// Four flavors of multiply-accumulate.
// Multiply-add (a * b) + c
// Multiply-subtract (a * b) - c
// Negated multiply-add -((a * b) + c)
// Negated multiply-subtract -((a * b) - c)

void RiscVFMadd(const Instruction *instruction) {
  using T = float;
  RiscVTernaryFloatNaNBoxOp<FPRegister::ValueType, T, T>(
      instruction, [instruction](T a, T b, T c) -> T {
        // Propagate any NaNs.
        if (FPTypeInfo<T>::IsNaN(a)) return internal::CanonicalizeNaN(a);
        if (FPTypeInfo<T>::IsNaN(b)) return internal::CanonicalizeNaN(b);
        if ((std::isinf(a) && (b == 0.0)) || ((std::isinf(b) && (a == 0.0)))) {
          auto *flag_db = instruction->Destination(1)->AllocateDataBuffer();
          flag_db->Set<uint32_t>(0, *FPExceptions::kInvalidOp);
          flag_db->Submit();
        }
        if (FPTypeInfo<T>::IsNaN(c)) return internal::CanonicalizeNaN(c);
        if (std::isinf(c) && !std::isinf(a) && !std::isinf(b)) return c;
        if (c == 0.0) {
          if ((a == 0.0 && !std::isinf(b)) || (b == 0.0 && !std::isinf(a))) {
            FPUInt c_sign = *reinterpret_cast<FPUInt *>(&c) >>
                            (FPTypeInfo<T>::kBitSize - 1);
            FPUInt ua = *reinterpret_cast<FPUInt *>(&a);
            FPUInt ub = *reinterpret_cast<FPUInt *>(&b);
            FPUInt prod_sign = (ua ^ ub) >> (FPTypeInfo<T>::kBitSize - 1);
            if (prod_sign != c_sign) return 0.0;
            return c;
          }
          return internal::CanonicalizeNaN(a * b);
        }
        return internal::CanonicalizeNaN((a * b) + c);
      });
}

void RiscVFMsub(const Instruction *instruction) {
  using T = float;
  RiscVTernaryFloatNaNBoxOp<FPRegister::ValueType, T, T>(
      instruction, [instruction](T a, T b, T c) -> T {
        if (FPTypeInfo<T>::IsNaN(a)) return internal::CanonicalizeNaN(a);
        if (FPTypeInfo<T>::IsNaN(b)) return internal::CanonicalizeNaN(b);
        if ((std::isinf(a) && (b == 0.0)) || ((std::isinf(b) && (a == 0.0)))) {
          auto *flag_db = instruction->Destination(1)->AllocateDataBuffer();
          flag_db->Set<uint32_t>(0, *FPExceptions::kInvalidOp);
          flag_db->Submit();
        }
        if (FPTypeInfo<T>::IsNaN(c)) return internal::CanonicalizeNaN(c);
        if (std::isinf(c) && !std::isinf(a) && !std::isinf(b)) return -c;
        if (c == 0.0) {
          if ((a == 0.0 && !std::isinf(b)) || (b == 0.0 && !std::isinf(a))) {
            FPUInt c_sign = -*reinterpret_cast<FPUInt *>(&c) >>
                            (FPTypeInfo<T>::kBitSize - 1);
            FPUInt ua = *reinterpret_cast<FPUInt *>(&a);
            FPUInt ub = *reinterpret_cast<FPUInt *>(&b);
            FPUInt prod_sign = (ua ^ ub) >> (FPTypeInfo<T>::kBitSize - 1);
            if (prod_sign == c_sign) return 0.0;
            return -c;
          }
          return internal::CanonicalizeNaN(a * b);
        }
        return internal::CanonicalizeNaN((a * b) - c);
      });
}

void RiscVFNmadd(const Instruction *instruction) {
  using T = float;
  RiscVTernaryFloatNaNBoxOp<FPRegister::ValueType, T, T>(
      instruction, [instruction](T a, T b, T c) -> T {
        if (FPTypeInfo<T>::IsNaN(a)) return internal::CanonicalizeNaN(a);
        if (FPTypeInfo<T>::IsNaN(b)) return internal::CanonicalizeNaN(b);
        if ((std::isinf(a) && (b == 0.0)) || ((std::isinf(b) && (a == 0.0)))) {
          auto *flag_db = instruction->Destination(1)->AllocateDataBuffer();
          flag_db->Set<uint32_t>(0, *FPExceptions::kInvalidOp);
          flag_db->Submit();
        }
        if (FPTypeInfo<T>::IsNaN(c)) return internal::CanonicalizeNaN(c);
        if (std::isinf(c) && !std::isinf(a) && !std::isinf(b)) return -c;
        if (c == 0.0) {
          if ((a == 0.0 && !std::isinf(b)) || (b == 0.0 && !std::isinf(a))) {
            FPUInt c_sign = *reinterpret_cast<FPUInt *>(&c) >>
                            (FPTypeInfo<T>::kBitSize - 1);
            FPUInt ua = *reinterpret_cast<FPUInt *>(&a);
            FPUInt ub = *reinterpret_cast<FPUInt *>(&b);
            FPUInt prod_sign = (ua ^ ub) >> (FPTypeInfo<T>::kBitSize - 1);
            if (prod_sign != c_sign) return 0.0;
            return -c;
          }
          return internal::CanonicalizeNaN(-a * b);
        }
        return internal::CanonicalizeNaN(-((a * b) + c));
      });
}

void RiscVFNmsub(const Instruction *instruction) {
  using T = float;
  RiscVTernaryFloatNaNBoxOp<FPRegister::ValueType, T, T>(
      instruction, [instruction](T a, T b, T c) -> T {
        if (FPTypeInfo<T>::IsNaN(a)) return internal::CanonicalizeNaN(a);
        if (FPTypeInfo<T>::IsNaN(b)) return internal::CanonicalizeNaN(b);
        if ((std::isinf(a) && (b == 0.0)) || ((std::isinf(b) && (a == 0.0)))) {
          auto *flag_db = instruction->Destination(1)->AllocateDataBuffer();
          flag_db->Set<uint32_t>(0, *FPExceptions::kInvalidOp);
          flag_db->Submit();
        }
        if (FPTypeInfo<T>::IsNaN(c)) return internal::CanonicalizeNaN(c);
        if (std::isinf(c) && !std::isinf(a) && !std::isinf(b)) return c;
        if (c == 0.0) {
          if ((a == 0.0 && !std::isinf(b)) || (b == 0.0 && !std::isinf(a))) {
            FPUInt c_sign = -*reinterpret_cast<FPUInt *>(&c) >>
                            (FPTypeInfo<T>::kBitSize - 1);
            FPUInt ua = *reinterpret_cast<FPUInt *>(&a);
            FPUInt ub = *reinterpret_cast<FPUInt *>(&b);
            FPUInt prod_sign = (ua ^ ub) >> (FPTypeInfo<T>::kBitSize - 1);
            if (prod_sign != c_sign) return 0.0;
            return c;
          }
          return internal::CanonicalizeNaN(-a * b);
        }
        return internal::CanonicalizeNaN(-((a * b) - c));
      });
}

// Set sign of the first operand to that of the second.
void RiscVFSgnj(const Instruction *instruction) {
  RiscVBinaryNaNBoxOp<FPRegister::ValueType, FPUInt, FPUInt>(
      instruction,
      [](FPUInt a, FPUInt b) { return (a & 0x7fff'ffff) | (b & 0x8000'0000); });
}

// Set the sign of the first operand to the opposite of the second.
void RiscVFSgnjn(const Instruction *instruction) {
  RiscVBinaryNaNBoxOp<FPRegister::ValueType, FPUInt, FPUInt>(
      instruction, [](FPUInt a, FPUInt b) {
        return (a & 0x7fff'ffff) | (~b & 0x8000'0000);
      });
}

// Set the sign of the first operand to the xor of the signs of the two
// operands.
void RiscVFSgnjx(const Instruction *instruction) {
  RiscVBinaryNaNBoxOp<FPRegister::ValueType, FPUInt, FPUInt>(
      instruction, [](FPUInt a, FPUInt b) {
        return (a & 0x7fff'ffff) | ((a ^ b) & 0x8000'0000);
      });
}

// Convert signed 32 bit integer to float.
void RiscVFCvtSw(const Instruction *instruction) {
  RiscVUnaryFloatNaNBoxOp<FPRegister::ValueType, uint32_t, float, int32_t>(
      instruction, [](int32_t a) -> float { return static_cast<float>(a); });
}

// Convert unsigned 32 bit integer to float.
void RiscVFCvtSwu(const Instruction *instruction) {
  RiscVUnaryFloatNaNBoxOp<FPRegister::ValueType, uint32_t, float, uint32_t>(
      instruction, [](uint32_t a) -> float { return static_cast<float>(a); });
}

// Convert signed 64 bit integer to float.
void RiscVFCvtSl(const Instruction *instruction) {
  RiscVUnaryFloatNaNBoxOp<FPRegister::ValueType, uint64_t, float, int64_t>(
      instruction, [](int64_t a) -> float { return static_cast<float>(a); });
}

// Convert unsigned 64 bit integer to float.
void RiscVFCvtSlu(const Instruction *instruction) {
  RiscVUnaryFloatNaNBoxOp<FPRegister::ValueType, uint64_t, float, uint64_t>(
      instruction, [](uint64_t a) -> float { return static_cast<float>(a); });
}

// Single precision move instruction from integer to fp register file.
void RiscVFMvwx(const Instruction *instruction) {
  RiscVUnaryNaNBoxOp<FPRegister::ValueType, uint32_t, uint32_t, uint32_t>(
      instruction, [](uint32_t a) -> uint32_t { return a; });
}

namespace RV32 {

using XRegister = RV32Register;
using XUint = typename std::make_unsigned<XRegister::ValueType>::type;
using XInt = typename std::make_signed<XRegister::ValueType>::type;

void RiscVFSw(const Instruction *instruction) {
  using T = uint32_t;
  auto *state = static_cast<RiscVState *>(instruction->state());
  if (state->mstatus()->fs() == 0) return;
  XUint base = generic::GetInstructionSource<XUint>(instruction, 0);
  XInt offset = generic::GetInstructionSource<XInt>(instruction, 1);
  XUint address = base + offset;
  T value = generic::GetInstructionSource<T>(instruction, 2);
  auto *db = state->db_factory()->Allocate(sizeof(T));
  db->Set<T>(0, value);
  state->StoreMemory(instruction, address, db);
  db->DecRef();
}

// Single precision conversion instructions.

// Convert float to signed 32 bit integer.
void RiscVFCvtWs(const Instruction *instruction) {
  internal::RVFCvtWs<XInt>(instruction);
}

// Convert float to unsigned 32 bit integer.
void RiscVFCvtWus(const Instruction *instruction) {
  internal::RVFCvtWus<XInt>(instruction);
}

// Single precision move instruction to integer register file, with
// sign-extension.
void RiscVFMvxw(const Instruction *instruction) {
  RiscVUnaryOp<XRegister, int32_t, int32_t>(instruction,
                                            [](int32_t a) { return a; });
}

// Single precision compare equal.
void RiscVFCmpeq(const Instruction *instruction) {
  internal::RVFCmpeq<XRegister>(instruction);
}

// Single precicion compare less than.
void RiscVFCmplt(const Instruction *instruction) {
  internal::RVFCmplt<XRegister>(instruction);
}

// Single precision compare less than or equal.
void RiscVFCmple(const Instruction *instruction) {
  internal::RVFCmple<XRegister>(instruction);
}

// Single precision fp class instruction.
void RiscVFClass(const Instruction *instruction) {
  RiscVUnaryOp<XRegister, uint32_t, float>(
      instruction,
      [](float a) -> uint32_t { return static_cast<uint32_t>(ClassifyFP(a)); });
}

}  // namespace RV32

namespace RV64 {

using XRegister = RV64Register;
using XUint = typename std::make_unsigned<XRegister::ValueType>::type;
using XInt = typename std::make_signed<XRegister::ValueType>::type;

void RiscVFSw(const Instruction *instruction) {
  using T = uint32_t;
  auto *state = static_cast<RiscVState *>(instruction->state());
  if (state->mstatus()->fs() == 0) return;
  XUint base = generic::GetInstructionSource<XUint>(instruction, 0);
  XInt offset = generic::GetInstructionSource<XInt>(instruction, 1);
  XUint address = base + offset;
  T value = generic::GetInstructionSource<T>(instruction, 2);
  auto *db = state->db_factory()->Allocate(sizeof(T));
  db->Set<T>(0, value);
  state->StoreMemory(instruction, address, db);
  db->DecRef();
}

// Convert float to signed 32 bit integer in a 64 bit register.
void RiscVFCvtWs(const Instruction *instruction) {
  internal::RVFCvtWs<XInt>(instruction);
}

// Convert float to unsigned 32 bit integer in a 64 bit register.
void RiscVFCvtWus(const Instruction *instruction) {
  internal::RVFCvtWus<XInt>(instruction);
}

// Convert float to signed 64 bit integer.
void RiscVFCvtLs(const Instruction *instruction) {
  internal::RVFCvtLs<XInt>(instruction);
}

// Convert float to unsigned 64 bit integer.
void RiscVFCvtLus(const Instruction *instruction) {
  internal::RVFCvtLus<XUint>(instruction);
}

// Single precision move instruction to integer register file, with
// sign-extension.
void RiscVFMvxw(const Instruction *instruction) {
  RiscVUnaryOp<XRegister, XInt, int32_t>(
      instruction, [](int32_t a) { return static_cast<XInt>(a); });
}

// Single precision move instruction from integer to fp register file.
void RiscVFMvwx(const Instruction *instruction) {
  RiscVUnaryNaNBoxOp<FPRegister::ValueType, uint32_t, uint32_t, uint32_t>(
      instruction, [](uint32_t a) -> uint32_t { return a; });
}

// Single precision compare equal.
void RiscVFCmpeq(const Instruction *instruction) {
  internal::RVFCmpeq<XRegister>(instruction);
}

// Single precicion compare less than.
void RiscVFCmplt(const Instruction *instruction) {
  internal::RVFCmplt<XRegister>(instruction);
}

// Single precision compare less than or equal.
void RiscVFCmple(const Instruction *instruction) {
  internal::RVFCmple<XRegister>(instruction);
}

// Single precision fp class instruction.
void RiscVFClass(const Instruction *instruction) {
  RiscVUnaryOp<XRegister, uint32_t, float>(
      instruction,
      [](float a) -> uint32_t { return static_cast<uint32_t>(ClassifyFP(a)); });
}

}  // namespace RV64

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
