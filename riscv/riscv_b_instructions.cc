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

#include "riscv/riscv_b_instructions.h"

#include <algorithm>
#include <type_traits>

#include "absl/numeric/bits.h"
#include "absl/types/span.h"
#include "mpact/sim/generic/instruction.h"
#include "riscv/riscv_instruction_helpers.h"

namespace mpact {
namespace sim {
namespace riscv {
namespace RV64 {
namespace {

using RegisterType = RV64Register;
using UIntReg =
    typename std::make_unsigned<typename RegisterType::ValueType>::type;
using IntReg = typename std::make_signed<UIntReg>::type;

// Handles instructions that perform a shift and add operation.
void ShNaddHelper(const generic::Instruction* instruction, int shift) {
  RiscVBinaryOp<RegisterType, IntReg, IntReg>(
      instruction, [shift](IntReg a, IntReg b) { return (a << shift) + b; });
}

// Handles instructions that perform an unsigned shift and add operation. Note
// that the least significant word (4 bytes) is masked before the operation is
// performed.
void ShNadduwHelper(const generic::Instruction* instruction, int shift) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [shift](UIntReg a, UIntReg b) {
        return ((a & 0xffff'ffff) << shift) + b;
      });
}

}  // namespace

void RiscVAddUw(const generic::Instruction* instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return (a & 0xffff'ffff) + b; });
}

void RiscVSh1add(const generic::Instruction* instruction) {
  ShNaddHelper(instruction, 1);
}

void RiscVSh2add(const generic::Instruction* instruction) {
  ShNaddHelper(instruction, 2);
}

void RiscVSh3add(const generic::Instruction* instruction) {
  ShNaddHelper(instruction, 3);
}

void RiscVSh1adduw(const generic::Instruction* instruction) {
  ShNadduwHelper(instruction, 1);
}

void RiscVSh2adduw(const generic::Instruction* instruction) {
  ShNadduwHelper(instruction, 2);
}

void RiscVSh3adduw(const generic::Instruction* instruction) {
  ShNadduwHelper(instruction, 3);
}

void RiscVSlliuw(const generic::Instruction* instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, int32_t>(
      instruction, [](UIntReg a, int32_t b) { return (a & 0xffff'ffff) << b; });
}

void RiscVAndn(const generic::Instruction* instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a & ~b; });
}

void RiscVOrn(const generic::Instruction* instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a | ~b; });
}

void RiscVXnor(const generic::Instruction* instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a ^ ~b; });
}

void RiscVClz(const generic::Instruction* instruction) {
  RiscVUnaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a) { return absl::countl_zero(a); });
}

void RiscVClzw(const generic::Instruction* instruction) {
  RiscVUnaryOp<RegisterType, UIntReg, UIntReg>(instruction, [](UIntReg a) {
    uint32_t value = static_cast<uint32_t>(a);
    return absl::countl_zero(value);
  });
}

void RiscVCtz(const generic::Instruction* instruction) {
  RiscVUnaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a) { return absl::countr_zero(a); });
}

void RiscVCtzw(const generic::Instruction* instruction) {
  RiscVUnaryOp<RegisterType, UIntReg, UIntReg>(instruction, [](UIntReg a) {
    uint32_t value = static_cast<uint32_t>(a);
    return absl::countr_zero(value);
  });
}

void RiscVCpop(const generic::Instruction* instruction) {
  RiscVUnaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a) { return absl::popcount(a); });
}

void RiscVCpopw(const generic::Instruction* instruction) {
  RiscVUnaryOp<RegisterType, UIntReg, UIntReg>(instruction, [](UIntReg a) {
    return absl::popcount(a & 0xffff'ffffULL);
  });
}

void RiscVMax(const generic::Instruction* instruction) {
  RiscVBinaryOp<RegisterType, IntReg, IntReg>(
      instruction, [](IntReg a, IntReg b) { return std::max(a, b); });
}

void RiscVMaxu(const generic::Instruction* instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return std::max(a, b); });
}

void RiscVMin(const generic::Instruction* instruction) {
  RiscVBinaryOp<RegisterType, IntReg, IntReg>(
      instruction, [](IntReg a, IntReg b) { return std::min(a, b); });
}

void RiscVMinu(const generic::Instruction* instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return std::min(a, b); });
}

void RiscVSexth(const generic::Instruction* instruction) {
  RiscVUnaryOp<RegisterType, IntReg, IntReg>(
      instruction, [](IntReg a) -> IntReg {
        if ((1 << 15) & a) {
          return a | 0xffff'ffff'ffff'0000;
        } else {
          return a & 0xffff;
        }
      });
}
void RiscVSextb(const generic::Instruction* instruction) {
  RiscVUnaryOp<RegisterType, IntReg, IntReg>(
      instruction, [](IntReg a) -> IntReg {
        if ((1 << 7) & a) {
          return a | 0xffff'ffff'ffff'ff00;
        } else {
          return a & 0xff;
        }
      });
}

void RiscVZextw(const generic::Instruction* instruction) {
  RiscVUnaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a) { return a & 0x0000'0000'ffff'ffff; });
}

void RiscVZexth(const generic::Instruction* instruction) {
  RiscVUnaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a) { return a & 0x0000'0000'0000'ffff; });
}

void RiscVRol(const generic::Instruction* instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction,
      [](UIntReg a, UIntReg b) { return absl::rotl(a, b & 0x3f); });
}

void RiscVRolw(const generic::Instruction* instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) {
        const uint32_t a32 = static_cast<uint32_t>(a);
        const uint32_t value = absl::rotl(a32, b & 0x1f);
        const uint64_t sign_extended_value =
            (value & 0x8000'0000)
                ? static_cast<uint64_t>(value) | 0xffff'ffff'0000'0000
                : static_cast<uint64_t>(value);
        return sign_extended_value;
      });
}

void RiscVRor(const generic::Instruction* instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction,
      [](UIntReg a, UIntReg b) { return absl::rotr(a, b & 0x3f); });
}

void RiscVRorw(const generic::Instruction* instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) {
        const uint32_t a32 = static_cast<uint32_t>(a);
        const uint32_t value = absl::rotr(a32, b & 0x1f);
        const uint64_t sign_extended_value =
            (value & 0x8000'0000)
                ? static_cast<uint64_t>(value) | 0xffff'ffff'0000'0000
                : static_cast<uint64_t>(value);
        return sign_extended_value;
      });
}

void RiscVRori(const generic::Instruction* instruction) {
  // Note: Handling of immediates and register values is identical.
  RiscVRor(instruction);
}

void RiscVRoriw(const generic::Instruction* instruction) {
  // Note: Handling of immediates and register values is identical.
  RiscVRorw(instruction);
}

void RiscVOrcb(const generic::Instruction* instruction) {
  RiscVUnaryOp<RegisterType, UIntReg, UIntReg>(instruction, [](UIntReg a) {
    UIntReg result = 0;
    for (int i = 0; i < sizeof(UIntReg); ++i) {
      const UIntReg mask = 0xFFULL << (8 * i);
      if (a & mask) {
        result |= mask;
      }
    }
    return result;
  });
}

void RiscVRev8(const generic::Instruction* instruction) {
  RiscVUnaryOp<RegisterType, UIntReg, UIntReg>(instruction, [](UIntReg a) {
    absl::Span<const uint8_t> input_span(reinterpret_cast<const uint8_t*>(&a),
                                         sizeof(UIntReg));

    UIntReg output = 0;
    absl::Span<uint8_t> output_span(reinterpret_cast<uint8_t*>(&output),
                                    sizeof(UIntReg));

    for (int i = 0; i < sizeof(UIntReg); ++i) {
      const int output_index = (sizeof(UIntReg) - 1) - i;
      output_span[output_index] = input_span[i];
    }
    return output;
  });
}

}  // namespace RV64
}  // namespace riscv
}  // namespace sim
}  // namespace mpact
