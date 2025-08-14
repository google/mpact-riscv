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

#ifndef MPACT_RISCV_RISCV_RISCV_VECTOR_FP_UNARY_INSTRUCTIONS_H_
#define MPACT_RISCV_RISCV_RISCV_VECTOR_FP_UNARY_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

// This file lists the semantic functions for RiscV vector unary floating point
// instructions.

namespace mpact {
namespace sim {
namespace riscv {

using Instruction = ::mpact::sim::generic::Instruction;

// Move a single floating point value from vector[0] to vector[vl-1].
void Vfmvvf(const Instruction* inst);
// Move a single floating point value from vector[0] to scalar fp register.
void Vfmvsf(const Instruction* inst);
// Move single floating point value from scalar fp register to vector[0].
void Vfmvfs(const Instruction* inst);

// Each of the following semantic functions take 2 source operands and 1
// destination operand. Source operand 0 is a vector register group, source
// operand 1 is a vector mask register, and destination operand 0 is a vector
// register group.

// Vector element conversion instructions. These convert same sized values
// from/to signed/unsigned integer and floating point. The 'rtz' versions use
// truncation (round to zero), whereas the others use the dynamically set
// rounding mode.

// FP to unsigned integer.
void Vfcvtxufv(const Instruction* inst);
// FP to signed integer.
void Vfcvtxfv(const Instruction* inst);
// Unsigned integer to FP.
void Vfcvtfxuv(const Instruction* inst);
// Signed integer to FP.
void Vfcvtfxv(const Instruction* inst);
// FP to unsigned integer using round to zero.
void Vfcvtrtzxufv(const Instruction* inst);
// FP to signed integer using round to zero.
void Vfcvtrtzxfv(const Instruction* inst);

// Vector element widening conversion instructions. These convert values from/to
// signed/unsigned integer and floating point, where the resulting value is 2x
// the width of the source value. The 'rtz' versions use truncation (round to
// zero), whereas the others use the dynamically set rounding mode.

// FP to wider unsigned integer.
void Vfwcvtxufv(const Instruction* inst);
// FP to wider signed integer.
void Vfwcvtxfv(const Instruction* inst);
// FP to next wider FP.
void Vfwcvtffv(const Instruction* inst);
// Unsigned integer to wider FP.
void Vfwcvtfxuv(const Instruction* inst);
// Signed integer to wider FP.
void Vfwcvtfxv(const Instruction* inst);
// FP to wider unsigned integer using round to zero.
void Vfwcvtrtzxufv(const Instruction* inst);
// FP to wider signed integer using round to zero.
void Vfwcvtrtzxfv(const Instruction* inst);

// Vector element widening conversion instructions. These convert values from/to
// signed/unsigned integer and floating point, where the resulting value is 1/2x
// the width of the source value. The 'rtz' versions use truncation (round to
// zero), the 'rod' version uses 'round to odd', whereas the others use the
// dynamically set rounding mode.

// FP to narrower unsigned integer.
void Vfncvtxufw(const Instruction* inst);
// FP to narrower signed integer.
void Vfncvtxfw(const Instruction* inst);
// FP to next narrower FP.
void Vfncvtffw(const Instruction* inst);
// FP to next narrower FP with round to odd.
void Vfncvtrodffw(const Instruction* inst);
// Unsigned integer to narrower FP.
void Vfncvtfxuw(const Instruction* inst);
// Signed integer to narrower FP.
void Vfncvtfxw(const Instruction* inst);
// FP to narrower unsigned integer using round to zero.
void Vfncvtrtzxufw(const Instruction* inst);
// FP to narrower signed integer using round to zero.
void Vfncvtrtzxfw(const Instruction* inst);

// Vector element square root instruction.
void Vfsqrtv(const Instruction* inst);
// Vector element approximate reciprocal square root instruction.
void Vfrsqrt7v(const Instruction* inst);
// Vector element approximate reciprocal instruction.
void Vfrec7v(const Instruction* inst);
// Vector element floating point value classify instruction.
void Vfclassv(const Instruction* inst);

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_VECTOR_FP_UNARY_INSTRUCTIONS_H_
