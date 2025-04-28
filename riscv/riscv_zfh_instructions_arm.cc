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

#include <sys/types.h>

#include <cassert>
#include <cmath>
#include <cstdint>

#include "absl/base/casts.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_fp_info.h"
#include "riscv/riscv_instruction_helpers.h"
#include "riscv/riscv_zfh_instructions.h"

namespace mpact {
namespace sim {
namespace riscv {

using HalfFP = ::mpact::sim::generic::HalfFP;

// TODO(b/401856759): Use arm intrinsics for fp32 -> fp16 and fp64 -> fp16
//                    conversions.

namespace {

// This is a soft conversion from a float or double to a half precision value.
// It is not a direct conversion from the floating point format to the half
// format. Instead, it uses the floating point hardware to do the conversion.
// This is done to get the correct rounding behavior for free from the FPU.
template <typename T>
HalfFP SoftConvertToHalfFP(T input_value, FPRoundingMode rm, uint32_t &fflags) {
  using UIntType = typename FPTypeInfo<T>::UIntType;
  using IntType = typename FPTypeInfo<T>::IntType;
  UIntType in_int = absl::bit_cast<UIntType>(input_value);
  HalfFP half_fp = {.value = 0x0000};

  // Extract the mantissa, exponent and sign.
  UIntType mantissa = in_int & FPTypeInfo<T>::kSigMask;
  UIntType exponent =
      (in_int & FPTypeInfo<T>::kExpMask) >> FPTypeInfo<T>::kSigSize;
  UIntType sign = in_int >> (FPTypeInfo<T>::kBitSize - 1);

  if (std::isnan(input_value)) {
    half_fp.value = FPTypeInfo<HalfFP>::kCanonicalNaN;
    if (FPTypeInfo<T>::IsSNaN(input_value)) {
      fflags |= static_cast<UIntType>(FPExceptions::kInvalidOp);
    }
    return half_fp;
  }

  if (std::isinf(input_value)) {
    half_fp.value = FPTypeInfo<HalfFP>::kPosInf;
    half_fp.value |= (sign & 1) << (FPTypeInfo<HalfFP>::kBitSize - 1);
    return half_fp;
  }

  if (in_int == 0 || in_int == 1ULL << (FPTypeInfo<T>::kBitSize - 1)) {
    half_fp.value =
        in_int >> (FPTypeInfo<T>::kBitSize - FPTypeInfo<HalfFP>::kBitSize);
    return half_fp;
  }

  IntType bias_diff = FPTypeInfo<T>::kExpBias - FPTypeInfo<HalfFP>::kExpBias;
  IntType unbounded_half_exponent = static_cast<IntType>(exponent) - bias_diff;
  IntType sig_size_diff =
      FPTypeInfo<T>::kSigSize - FPTypeInfo<HalfFP>::kSigSize;
  UIntType half_inf_exponent = ((1 << FPTypeInfo<HalfFP>::kExpSize) - 1);
  UIntType source_type_inf_exponent = ((1 << FPTypeInfo<T>::kExpSize) - 1);

  // Create a temp float with the smallest normal exponent and input mantissa.
  T ftmp = absl::bit_cast<T>(
      (sign << (FPTypeInfo<T>::kBitSize - 1)) |
      (static_cast<UIntType>(1ULL) << FPTypeInfo<T>::kSigSize) | mantissa);

  // Create a divisor float that will be used for shifting the mantissa in a
  // rounding aware way. The amount of shifting depends on if the result is
  // subnormal or normal.
  T fdiv = 0;
  UIntType default_fdiv_exp = FPTypeInfo<T>::kExpBias + sig_size_diff;
  UIntType fdiv_exp = default_fdiv_exp;
  if (unbounded_half_exponent > 0) {
    fdiv_exp = default_fdiv_exp;
  } else if (unbounded_half_exponent < 0) {
    // shift_count: emin - unbiased exponent
    IntType shift_count = 1 - static_cast<int>(exponent) + bias_diff;
    fdiv_exp = default_fdiv_exp + shift_count;
    fdiv_exp = std::min(fdiv_exp, source_type_inf_exponent - 1);
  } else {
    fdiv_exp = default_fdiv_exp + 1;
  }
  fdiv = absl::bit_cast<T>(fdiv_exp << FPTypeInfo<T>::kSigSize);

  // Shift right by doing division.
  T fres = ftmp / fdiv;
  UIntType res = absl::bit_cast<UIntType>(fres);

  // Shift left by doing multiplication.
  T fmultiply = absl::bit_cast<T>(default_fdiv_exp << FPTypeInfo<T>::kSigSize);
  T fres2 = fres * fmultiply;
  UIntType res2 = absl::bit_cast<UIntType>(fres2);

  // Update the exponent if rounding caused an increase.
  IntType exp_diff = static_cast<IntType>((res2 >> FPTypeInfo<T>::kSigSize) &
                                          source_type_inf_exponent) -
                     1;
  UIntType new_exponent = (exponent + exp_diff) & source_type_inf_exponent;

  UIntType half_exponent = 0;
  if (unbounded_half_exponent > 0) {
    half_exponent = new_exponent - bias_diff;
  } else if (unbounded_half_exponent < 0) {
    // Guaranteed subnormal. Nothing to do.
  } else {
    // This case could be normal or subnormal depending on the rounding result.
    half_exponent = (res2 >> FPTypeInfo<T>::kSigSize) & half_inf_exponent;
  }

  UIntType half_mantissa =
      (res2 >> sig_size_diff) & FPTypeInfo<HalfFP>::kSigMask;
  if (unbounded_half_exponent < 0) {  // Guaranteed Subnormal
    half_mantissa = (res & (1 << FPTypeInfo<HalfFP>::kSigSize))
                        ? ((res >> 1) & FPTypeInfo<HalfFP>::kSigMask)
                        : res & FPTypeInfo<HalfFP>::kSigMask;
  }

  // Handle the rules for overflowing to infinity depending on the rounding
  // mode.
  if (half_exponent >= half_inf_exponent) {
    fflags |= static_cast<uint32_t>(FPExceptions::kOverflow);
    fflags |= static_cast<uint32_t>(FPExceptions::kInexact);
    switch (rm) {
      case FPRoundingMode::kRoundToNearest:
        half_exponent = half_inf_exponent;
        half_mantissa = 0;
        break;
      case FPRoundingMode::kRoundTowardsZero:
        half_exponent = half_inf_exponent - 1;
        half_mantissa = FPTypeInfo<HalfFP>::kSigMask;
        break;
      case FPRoundingMode::kRoundDown:
        half_exponent = sign ? half_inf_exponent : half_inf_exponent - 1;
        half_mantissa = sign ? 0 : FPTypeInfo<HalfFP>::kSigMask;
        break;
      case FPRoundingMode::kRoundUp:
        half_exponent = sign ? half_inf_exponent - 1 : half_inf_exponent;
        half_mantissa = sign ? FPTypeInfo<HalfFP>::kSigMask : 0;
        break;
      default:
        half_exponent = half_inf_exponent;
        half_mantissa = 0;
        break;
    }
  }

  // Handle flags for the specific underflow case.
  if (unbounded_half_exponent < 0 ||
      (unbounded_half_exponent == 0 && fres2 != ftmp)) {
    fflags |= static_cast<uint32_t>(FPExceptions::kUnderflow);
  }

  // Handle flags for the specific inexact case.
  if (fres2 != ftmp) {
    fflags |= static_cast<uint32_t>(FPExceptions::kInexact);
  }

  // Construct the half float.
  half_fp.value = half_mantissa |
                  (half_exponent << FPTypeInfo<HalfFP>::kSigSize) |
                  (sign << (FPTypeInfo<HalfFP>::kBitSize - 1));

  // Do an arithmetic reconstruction of the float to check for exactness.
  T trailing_significand_float = static_cast<T>(half_mantissa);
  T precision_factor = std::pow(2.0, -1.0 * FPTypeInfo<HalfFP>::kSigSize);
  IntType unbiased_exponent =
      (half_exponent == 0 ? 1 : half_exponent) - FPTypeInfo<HalfFP>::kExpBias;
  T exponent_factor = std::pow(2.0, unbiased_exponent);
  T sign_factor = sign == 1 ? -1.0 : 1.0;
  T implicit_bit_adjustment = half_exponent == 0 ? 0.0 : 1.0;
  T reconstructed_value = ((trailing_significand_float * precision_factor) +
                           implicit_bit_adjustment) *
                          exponent_factor * sign_factor;

  if (reconstructed_value == input_value) {
    // Clear the flags for exact conversions.
    fflags &= ~(static_cast<uint32_t>(FPExceptions::kUnderflow) |
                static_cast<uint32_t>(FPExceptions::kInexact));
  }
  return half_fp;
}
}  // namespace

HalfFP ConvertSingleToHalfFP(float input_value, FPRoundingMode rm,
                             uint32_t &fflags) {
  return SoftConvertToHalfFP(input_value, rm, fflags);
}

HalfFP ConvertDoubleToHalfFP(double input_value, FPRoundingMode rm,
                             uint32_t &fflags) {
  return SoftConvertToHalfFP(input_value, rm, fflags);
}

namespace zfh_internal {
bool UseHostFlagsForConversion() { return false; }
}  // namespace zfh_internal

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
