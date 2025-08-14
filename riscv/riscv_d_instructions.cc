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
#include <type_traits>

#include "absl/log/log.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_fp_host.h"
#include "riscv/riscv_fp_info.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_instruction_helpers.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::Instruction;
using ::mpact::sim::generic::operator*;  // NOLINT: is used below.

// The following instruction semantic functions implement the double precision
// floating point instructions in the RiscV architecture. They all utilize the
// templated helper functions in riscv_instruction_helpers.h to implement
// the boiler plate code.

// These types are used instead of uint64_t and int64_t to represent the
// integer type of equal width to double when values of these types are
// really reinterpreted double values.
using UInt = FPTypeInfo<double>::UIntType;
using SInt = FPTypeInfo<double>::IntType;

using FPRegister = RVFpRegister;

namespace internal {

template <typename T>
static inline T CanonicalizeNaN(T value) {
  if (!std::isnan(value)) return value;
  auto nan_value = FPTypeInfo<T>::kCanonicalNaN;
  return *reinterpret_cast<T*>(&nan_value);
}

}  // namespace internal

// Basic arithmetic operations.
void RiscVDAdd(const Instruction* instruction) {
  RiscVBinaryFloatNaNBoxOp<RVFpRegister::ValueType, double, double>(
      instruction, [](double a, double b) { return a + b; });
}

void RiscVDSub(const Instruction* instruction) {
  RiscVBinaryFloatNaNBoxOp<RVFpRegister::ValueType, double, double>(
      instruction, [](double a, double b) { return a - b; });
}

void RiscVDMul(const Instruction* instruction) {
  RiscVBinaryFloatNaNBoxOp<RVFpRegister::ValueType, double, double>(
      instruction, [](double a, double b) { return a * b; });
}

void RiscVDDiv(const Instruction* instruction) {
  RiscVBinaryFloatNaNBoxOp<RVFpRegister::ValueType, double, double>(
      instruction, [](double a, double b) { return a / b; });
}

// Square root uses the library square root.
void RiscVDSqrt(const Instruction* instruction) {
  RiscVUnaryNaNBoxOp<FPRegister::ValueType, FPRegister::ValueType, double,
                     double>(instruction, [instruction](double a) -> double {
    // If the input value is NaN or less than zero, set the invalid op flag.
    if (FPTypeInfo<double>::IsNaN(a) || (a < 0.0)) {
      if (!FPTypeInfo<double>::IsQNaN(a)) {
        auto* flag_db = instruction->Destination(1)->AllocateDataBuffer();
        flag_db->Set<uint32_t>(0, *FPExceptions::kInvalidOp);
        flag_db->Submit();
      }
      return *reinterpret_cast<const double*>(
          &FPTypeInfo<double>::kCanonicalNaN);
    }

    // Square root of 0 returns 0, and of -0.0 returns -0.0.
    if (a == 0.0) return a;

    // For all other cases use the library sqrt.
    // Get the rounding mode.
    int rm_value = generic::GetInstructionSource<int>(instruction, 1);

    auto* rv_fp = static_cast<RiscVState*>(instruction->state())->rv_fp();
    // If the rounding mode is dynamic, read it from the current state.
    if (rm_value == *FPRoundingMode::kDynamic) {
      if (!rv_fp->rounding_mode_valid()) {
        LOG(ERROR) << "Invalid rounding mode";
        return *reinterpret_cast<const double*>(
            &FPTypeInfo<double>::kCanonicalNaN);
      }
      rm_value = *rv_fp->GetRoundingMode();
    }
    double res;
    {
      ScopedFPStatus set_fp_status(rv_fp->host_fp_interface(), rm_value);
      res = sqrt(a);
    }
    return res;
  });
}

// If either operand is NaN return the other.
void RiscVDMin(const Instruction* instruction) {
  RiscVBinaryOp<FPRegister, double, double>(
      instruction, [instruction](double a, double b) -> double {
        if (FPTypeInfo<double>::IsSNaN(a) || FPTypeInfo<double>::IsSNaN(b)) {
          auto* db = instruction->Destination(1)->AllocateDataBuffer();
          db->Set<uint32_t>(0, *FPExceptions::kInvalidOp);
          db->Submit();
        }
        if (FPTypeInfo<double>::IsNaN(a)) {
          if (FPTypeInfo<double>::IsNaN(b)) {
            UInt not_a_number = FPTypeInfo<double>::kCanonicalNaN;
            return *reinterpret_cast<double*>(&not_a_number);
          }
          return b;
        }
        if (FPTypeInfo<double>::IsNaN(b)) return a;
        // If both are zero, return the negative zero if there is one.
        if ((a == 0.0) && (b == 0.0)) return (std::signbit(a)) ? a : b;
        return (a > b) ? b : a;
      });
}

// If either operand is NaN return the other.
void RiscVDMax(const Instruction* instruction) {
  RiscVBinaryOp<FPRegister, double, double>(
      instruction, [instruction](double a, double b) {
        if (FPTypeInfo<double>::IsSNaN(a) || FPTypeInfo<double>::IsSNaN(b)) {
          auto* db = instruction->Destination(1)->AllocateDataBuffer();
          db->Set<uint32_t>(0, *FPExceptions::kInvalidOp);
          db->Submit();
        }
        if (FPTypeInfo<double>::IsNaN(a)) {
          if (FPTypeInfo<double>::IsNaN(b)) {
            UInt not_a_number = FPTypeInfo<double>::kCanonicalNaN;
            return *reinterpret_cast<double*>(&not_a_number);
          }
          return b;
        }
        if (FPTypeInfo<double>::IsNaN(b)) return a;
        // If both are zero, return the negative zero if there is one.
        if ((a == 0.0) && (b == 0.0)) return (std::signbit(a)) ? b : a;
        return (a < b) ? b : a;
      });
}

// Four flavors of multiply accumulate.
// Multiply-add (a * b) + c
// Multiply-subtract (a * b) - c
// Negated multiply-add -((a * b) + c)
// Negated multiply-subtract -((a * b) - c)

void RiscVDMadd(const Instruction* instruction) {
  using T = double;
  RiscVTernaryFloatNaNBoxOp<FPRegister::ValueType, T, T>(
      instruction, [instruction](T a, T b, T c) -> T {
        if (FPTypeInfo<T>::IsNaN(a)) return internal::CanonicalizeNaN(a);
        if (FPTypeInfo<T>::IsNaN(b)) return internal::CanonicalizeNaN(b);
        if ((std::isinf(a) && (b == 0.0)) || ((std::isinf(b) && (a == 0.0)))) {
          auto* flag_db = instruction->Destination(1)->AllocateDataBuffer();
          flag_db->Set<uint32_t>(0, *FPExceptions::kInvalidOp);
          flag_db->Submit();
        }
        if (FPTypeInfo<T>::IsNaN(c)) return internal::CanonicalizeNaN(c);
        if (std::isinf(c) && !std::isinf(a) && !std::isinf(b)) return c;
        if (c == 0.0) {
          if ((a == 0.0 && !std::isinf(b)) || (b == 0.0 && !std::isinf(a))) {
            UInt c_sign =
                *reinterpret_cast<UInt*>(&c) >> (FPTypeInfo<T>::kBitSize - 1);
            UInt ua = *reinterpret_cast<UInt*>(&a);
            UInt ub = *reinterpret_cast<UInt*>(&b);
            UInt prod_sign = (ua ^ ub) >> (FPTypeInfo<T>::kBitSize - 1);
            if (prod_sign != c_sign) return 0.0;
            return c;
          }
          return internal::CanonicalizeNaN(a * b);
        }
        return internal::CanonicalizeNaN(fma(a, b, c));
      });
}

void RiscVDMsub(const Instruction* instruction) {
  using T = double;
  RiscVTernaryFloatNaNBoxOp<FPRegister::ValueType, T, T>(
      instruction, [instruction](T a, T b, T c) -> T {
        if (FPTypeInfo<T>::IsNaN(a)) return internal::CanonicalizeNaN(a);
        if (FPTypeInfo<T>::IsNaN(b)) return internal::CanonicalizeNaN(b);
        if ((std::isinf(a) && (b == 0.0)) || ((std::isinf(b) && (a == 0.0)))) {
          auto* flag_db = instruction->Destination(1)->AllocateDataBuffer();
          flag_db->Set<uint32_t>(0, *FPExceptions::kInvalidOp);
          flag_db->Submit();
        }
        if (FPTypeInfo<T>::IsNaN(c)) return internal::CanonicalizeNaN(c);
        if (std::isinf(c) && !std::isinf(a) && !std::isinf(b)) return -c;
        if (c == 0.0) {
          if ((a == 0.0 && !std::isinf(b)) || (b == 0.0 && !std::isinf(a))) {
            UInt c_sign =
                -*reinterpret_cast<UInt*>(&c) >> (FPTypeInfo<T>::kBitSize - 1);
            UInt ua = *reinterpret_cast<UInt*>(&a);
            UInt ub = *reinterpret_cast<UInt*>(&b);
            UInt prod_sign = (ua ^ ub) >> (FPTypeInfo<T>::kBitSize - 1);
            if (prod_sign == c_sign) return 0.0;
            return -c;
          }
          return internal::CanonicalizeNaN(a * b);
        }
        return internal::CanonicalizeNaN(fma(a, b, -c));
      });
}

void RiscVDNmadd(const Instruction* instruction) {
  using T = double;
  RiscVTernaryFloatNaNBoxOp<FPRegister::ValueType, T, T>(
      instruction, [instruction](T a, T b, T c) -> T {
        if (FPTypeInfo<T>::IsNaN(a)) return internal::CanonicalizeNaN(a);
        if (FPTypeInfo<T>::IsNaN(b)) return internal::CanonicalizeNaN(b);
        if ((std::isinf(a) && (b == 0.0)) || ((std::isinf(b) && (a == 0.0)))) {
          auto* flag_db = instruction->Destination(1)->AllocateDataBuffer();
          flag_db->Set<uint32_t>(0, *FPExceptions::kInvalidOp);
          flag_db->Submit();
        }
        if (FPTypeInfo<T>::IsNaN(c)) return internal::CanonicalizeNaN(c);
        if (std::isinf(c) && !std::isinf(a) && !std::isinf(b)) return -c;
        if (c == 0.0) {
          if ((a == 0.0 && !std::isinf(b)) || (b == 0.0 && !std::isinf(a))) {
            UInt c_sign =
                *reinterpret_cast<UInt*>(&c) >> (FPTypeInfo<T>::kBitSize - 1);
            UInt ua = *reinterpret_cast<UInt*>(&a);
            UInt ub = *reinterpret_cast<UInt*>(&b);
            UInt prod_sign = (ua ^ ub) >> (FPTypeInfo<T>::kBitSize - 1);
            if (prod_sign != c_sign) return 0.0;
            return -c;
          }
          return internal::CanonicalizeNaN(-a * b);
        }
        return internal::CanonicalizeNaN(-fma(a, b, c));
      });
}

void RiscVDNmsub(const Instruction* instruction) {
  using T = double;
  RiscVTernaryFloatNaNBoxOp<FPRegister::ValueType, T, T>(
      instruction, [instruction](T a, T b, T c) -> T {
        if (FPTypeInfo<T>::IsNaN(a)) return internal::CanonicalizeNaN(a);
        if (FPTypeInfo<T>::IsNaN(b)) return internal::CanonicalizeNaN(b);
        // Illegal operation flag set if either a or b are infinite, and
        // the other is zero.
        if ((std::isinf(a) && (b == 0.0)) || ((std::isinf(b) && (a == 0.0)))) {
          auto* flag_db = instruction->Destination(1)->AllocateDataBuffer();
          flag_db->Set<uint32_t>(0, *FPExceptions::kInvalidOp);
          flag_db->Submit();
        }
        if (FPTypeInfo<T>::IsNaN(c)) return internal::CanonicalizeNaN(c);
        if (std::isinf(c) && !std::isinf(a) && !std::isinf(b)) return c;
        if (c == 0.0) {
          if ((a == 0.0 && !std::isinf(b)) || (b == 0.0 && !std::isinf(a))) {
            UInt c_sign =
                -*reinterpret_cast<UInt*>(&c) >> (FPTypeInfo<T>::kBitSize - 1);
            UInt ua = *reinterpret_cast<UInt*>(&a);
            UInt ub = *reinterpret_cast<UInt*>(&b);
            UInt prod_sign = (ua ^ ub) >> (FPTypeInfo<T>::kBitSize - 1);
            if (prod_sign != c_sign) return 0.0;
            return c;
          }
          return internal::CanonicalizeNaN(-a * b);
        }
        return internal::CanonicalizeNaN(-fma(a, b, -c));
      });
}

// Conversion instructions.

// Convert int to double.
void RiscVDCvtDw(const Instruction* instruction) {
  RiscVUnaryFloatOp<double, int32_t>(
      instruction, [](int32_t a) -> double { return static_cast<double>(a); });
}

// Convert unsigned word to double.
void RiscVDCvtDwu(const Instruction* instruction) {
  RiscVUnaryFloatOp<double, uint32_t>(
      instruction, [](uint32_t a) -> double { return static_cast<double>(a); });
}

// Convert double to float.
void RiscVDCvtSd(const Instruction* instruction) {
  RiscVUnaryFloatNaNBoxOp<RVFpRegister::ValueType, RVFpRegister::ValueType,
                          float, double>(instruction, [](double a) -> float {
    if (FPTypeInfo<double>::IsNaN(a)) {
      typename FPTypeInfo<float>::UIntType uint_value;
      uint_value = FPTypeInfo<float>::kCanonicalNaN;
      return *reinterpret_cast<float*>(&uint_value);
    }
    return static_cast<float>(a);
  });
}

// Convert float to double.
void RiscVDCvtDs(const Instruction* instruction) {
  RiscVUnaryFloatOp<double, float>(instruction, [](float a) -> double {
    if (FPTypeInfo<float>::IsNaN(a)) {
      typename FPTypeInfo<double>::UIntType uint_value;
      uint_value = FPTypeInfo<double>::kCanonicalNaN;
      return *reinterpret_cast<double*>(&uint_value);
    }
    return static_cast<double>(a);
  });
}

// Use sign of the second operand as the sign in the first.
void RiscVDSgnj(const Instruction* instruction) {
  RiscVBinaryOp<FPRegister, UInt, UInt>(
      instruction, [](UInt a, UInt b) -> UInt {
        return (a & 0x7fff'ffff'ffff'ffff) | (b & 0x8000'0000'0000'0000);
      });
}

// Use negation of the sign of the second operand as the sign in the first.
void RiscVDSgnjn(const Instruction* instruction) {
  RiscVBinaryOp<FPRegister, UInt, UInt>(
      instruction, [](UInt a, UInt b) -> UInt {
        return (a & 0x7fff'ffff'ffff'ffff) | (~b & 0x8000'0000'0000'0000);
      });
}

// Use the xor of the signs of the two operands as the sign in the first.
void RiscVDSgnjx(const Instruction* instruction) {
  RiscVBinaryOp<FPRegister, UInt, UInt>(
      instruction, [](UInt a, UInt b) -> UInt {
        return (a & 0x7fff'ffff'ffff'ffff) | ((a ^ b) & 0x8000'0000'0000'0000);
      });
}

namespace RV32 {

using XRegister = RV32Register;
using XInt = std::make_signed<RV32Register::ValueType>::type;
using XUInt = std::make_unsigned<RV32Register::ValueType>::type;

void RiscVDSd(const Instruction* instruction) {
  using T = uint64_t;
  auto* state = static_cast<RiscVState*>(instruction->state());
  if (state->mstatus()->fs() == 0) return;
  XUInt base = generic::GetInstructionSource<XUInt>(instruction, 0);
  XInt offset = generic::GetInstructionSource<XInt>(instruction, 1);
  XUInt address = base + offset;
  T value = generic::GetInstructionSource<T>(instruction, 2);
  auto* db = state->db_factory()->Allocate(sizeof(T));
  db->Set<T>(0, value);
  state->StoreMemory(instruction, address, db);
  db->DecRef();
}

// Convert double to int.
void RiscVDCvtWd(const Instruction* instruction) {
  RiscVConvertFloatWithFflagsOp<XInt, double, int32_t>(instruction);
}
// Convert double to unsigned word.
void RiscVDCvtWud(const Instruction* instruction) {
  RiscVConvertFloatWithFflagsOp<XUInt, double, uint32_t>(instruction);
}

// Double precision floating point compare equal.
void RiscVDCmpeq(const Instruction* instruction) {
  RiscVBinaryOp<XRegister, uint32_t, double>(
      instruction, [instruction](double a, double b) -> uint32_t {
        if (FPTypeInfo<double>::IsSNaN(a) || FPTypeInfo<double>::IsSNaN(b)) {
          auto* db = instruction->Destination(1)->AllocateDataBuffer();
          db->Set<uint32_t>(0, *FPExceptions::kInvalidOp);
          db->Submit();
        }
        return a == b;
      });
}

// Double precision floating point compare less.
void RiscVDCmplt(const Instruction* instruction) {
  RiscVBinaryOp<XRegister, uint32_t, double>(
      instruction, [instruction](double a, double b) -> uint32_t {
        if (FPTypeInfo<double>::IsNaN(a) || FPTypeInfo<double>::IsNaN(b)) {
          auto* db = instruction->Destination(1)->AllocateDataBuffer();
          db->Set<uint32_t>(0, *FPExceptions::kInvalidOp);
          db->Submit();
        }
        return a < b;
      });
}

// Double precision floating point compare less than or equal.
void RiscVDCmple(const Instruction* instruction) {
  RiscVBinaryOp<XRegister, uint32_t, double>(
      instruction, [instruction](double a, double b) -> uint32_t {
        if (FPTypeInfo<double>::IsNaN(a) || FPTypeInfo<double>::IsNaN(b)) {
          auto* db = instruction->Destination(1)->AllocateDataBuffer();
          db->Set<uint32_t>(0, *FPExceptions::kInvalidOp);
          db->Submit();
        }
        return a <= b;
      });
}

// Return the class attribute of the source operand.
void RiscVDClass(const Instruction* instruction) {
  RiscVUnaryOp<XRegister, XUInt, double>(instruction, [](double a) -> XUInt {
    auto res = static_cast<XUInt>(ClassifyFP(a));
    return res;
  });
}

}  // namespace RV32

namespace RV64 {

using XRegister = RV64Register;
using XInt = std::make_signed<RV64Register::ValueType>::type;
using XUInt = std::make_unsigned<RV64Register::ValueType>::type;

void RiscVDSd(const Instruction* instruction) {
  using T = uint64_t;
  auto* state = static_cast<RiscVState*>(instruction->state());
  if (state->mstatus()->fs() == 0) return;
  XUInt base = generic::GetInstructionSource<XUInt>(instruction, 0);
  XInt offset = generic::GetInstructionSource<XInt>(instruction, 1);
  XUInt address = base + offset;
  T value = generic::GetInstructionSource<T>(instruction, 2);
  auto* db = state->db_factory()->Allocate(sizeof(T));
  db->Set<T>(0, value);
  state->StoreMemory(instruction, address, db);
  db->DecRef();
}

// Convert double to int.
void RiscVDCvtWd(const Instruction* instruction) {
  RiscVConvertFloatWithFflagsOp<XInt, double, int32_t>(instruction);
}
// Convert double to unsigned word.
void RiscVDCvtWud(const Instruction* instruction) {
  RiscVConvertFloatWithFflagsOp<XUInt, double, uint32_t>(instruction);
}

// Double precision floating point compare equal.
void RiscVDCmpeq(const Instruction* instruction) {
  RiscVBinaryOp<XRegister, XUInt, double>(
      instruction, [instruction](double a, double b) -> XUInt {
        if (FPTypeInfo<double>::IsSNaN(a) || FPTypeInfo<double>::IsSNaN(b)) {
          auto* db = instruction->Destination(1)->AllocateDataBuffer();
          db->Set<uint32_t>(0, *FPExceptions::kInvalidOp);
          db->Submit();
        }
        return a == b;
      });
}

// Double precision floating point compare less.
void RiscVDCmplt(const Instruction* instruction) {
  RiscVBinaryOp<XRegister, XUInt, double>(
      instruction, [instruction](double a, double b) -> XUInt {
        if (FPTypeInfo<double>::IsNaN(a) || FPTypeInfo<double>::IsNaN(b)) {
          auto* db = instruction->Destination(1)->AllocateDataBuffer();
          db->Set<uint32_t>(0, *FPExceptions::kInvalidOp);
          db->Submit();
        }
        return a < b;
      });
}

// Double precision floating point compare less than or equal.
void RiscVDCmple(const Instruction* instruction) {
  RiscVBinaryOp<XRegister, XUInt, double>(
      instruction, [instruction](double a, double b) -> XUInt {
        if (FPTypeInfo<double>::IsNaN(a) || FPTypeInfo<double>::IsNaN(b)) {
          auto* db = instruction->Destination(1)->AllocateDataBuffer();
          db->Set<uint32_t>(0, *FPExceptions::kInvalidOp);
          db->Submit();
        }
        return a <= b;
      });
}

// Return the class attribute of the source operand.
void RiscVDClass(const Instruction* instruction) {
  RiscVUnaryOp<XRegister, UInt, double>(instruction, [](double a) -> UInt {
    auto res = static_cast<UInt>(ClassifyFP(a));
    return res;
  });
}

// Convert double to 64 bit signed integer.
void RiscVDCvtLd(const Instruction* instruction) {
  RiscVConvertFloatWithFflagsOp<XInt, double, int64_t>(instruction);
}

// Convert double to 64 bit unsigned integer.
void RiscVDCvtLud(const Instruction* instruction) {
  RiscVConvertFloatWithFflagsOp<XInt, double, uint64_t>(instruction);
}

// Convert signed 64 bit integer to double.
void RiscVDCvtDl(const Instruction* instruction) {
  RiscVUnaryFloatOp<double, int64_t>(
      instruction, [](int64_t a) -> double { return static_cast<double>(a); });
}

// Convert unsigned 64 bit integer to double.
void RiscVDCvtDlu(const Instruction* instruction) {
  RiscVUnaryFloatOp<double, uint64_t>(
      instruction, [](uint64_t a) -> double { return static_cast<double>(a); });
}

void RiscVDMvxd(const Instruction* instruction) {
  RiscVUnaryOp<XRegister, uint64_t, uint64_t>(
      instruction, [](uint64_t a) -> uint64_t { return a; });
}

void RiscVDMvdx(const Instruction* instruction) {
  RiscVUnaryOp<FPRegister, uint64_t, uint64_t>(
      instruction, [](uint64_t a) -> uint64_t { return a; });
}

}  // namespace RV64

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
