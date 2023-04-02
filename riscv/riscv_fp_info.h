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

#ifndef MPACT_RISCV_RISCV_RISCV_FP_INFO_H_
#define MPACT_RISCV_RISCV_RISCV_FP_INFO_H_

#include <cstdint>

namespace mpact {
namespace sim {
namespace riscv {

// RiscV floating point rounding modes. Note, for now, kRoundToNearestTiesToMax
// is treated as if it is kRRoundToNearest.
enum class FPRoundingMode : uint32_t {
  kRoundToNearest = 0b000,
  kRoundTowardsZero = 0b001,
  kRoundDown = 0b010,
  kRoundUp = 0b011,
  kRoundToNearestTiesToMax = 0b100,
  kDynamic = 0b111,  // kDynamic can only be used in the instruction rm field.
};

enum class FPExceptions : uint32_t {
  kInexact = 0b00001,
  kUnderflow = 0b00010,
  kOverflow = 0b00100,
  kDivByZero = 0b01000,
  kInvalidOp = 0b10000,
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_FP_INFO_H_
