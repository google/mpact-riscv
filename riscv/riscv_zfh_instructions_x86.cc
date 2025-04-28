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

#include <immintrin.h>
#include <sys/types.h>

#include <cstdint>

#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_fp_info.h"
#include "riscv/riscv_zfh_instructions.h"

namespace mpact {
namespace sim {
namespace riscv {

using HalfFP = ::mpact::sim::generic::HalfFP;

HalfFP ConvertSingleToHalfFP(float input_value, FPRoundingMode rm,
                             uint32_t &fflags) {
  HalfFP half_fp;

  // Get current MXCSR value. The simulator should have already configured the
  // rounding mode so we simply pass it along to the intrinsic.
  unsigned int mxcsr = _mm_getcsr();

  // Extract rounding control bits (bits 13 and 14)
  int rounding_control_bits = (mxcsr >> 13) & 0x3;

  switch (rounding_control_bits) {
    case 0x0:  // Round to nearest
      half_fp.value = _cvtss_sh(input_value, 0);
      break;
    case 0x1:  // Round down
      half_fp.value = _cvtss_sh(input_value, 1);
      break;
    case 0x2:  // Round up
      half_fp.value = _cvtss_sh(input_value, 2);
      break;
    case 0x3:  // Round towards zero
      half_fp.value = _cvtss_sh(input_value, 3);
      break;
    default:  // Default to nearest even if mode is not recognized
      half_fp.value = _cvtss_sh(input_value, 0);
      break;
  }

  return half_fp;
}

HalfFP ConvertDoubleToHalfFP(double input_value, FPRoundingMode rm,
                             uint32_t &fflags) {
  float input_float = static_cast<float>(input_value);
  return ConvertSingleToHalfFP(input_float, rm, fflags);
}

namespace zfh_internal {
bool UseHostFlagsForConversion() { return true; }
}  // namespace zfh_internal

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
