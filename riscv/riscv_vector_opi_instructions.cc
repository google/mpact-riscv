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

#include "riscv/riscv_vector_opi_instructions.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "absl/log/log.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_instruction_helpers.h"
#include "riscv/riscv_vector_state.h"

// This file contains the instruction semantic functions for most of the
// vector instructions in the OPIVV, OPIVX, and OPIVI encoding spaces. The
// exception is vector element permute instructions and a couple of reduction
// instructions.

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::MakeUnsigned;
using ::mpact::sim::generic::WideType;
using std::numeric_limits;

// Vector arithmetic operations.

// Vector add.
void Vadd(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t vs2, uint8_t vs1) -> uint8_t { return vs2 + vs1; });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2, uint16_t vs1) -> uint16_t { return vs2 + vs1; });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2, uint32_t vs1) -> uint32_t { return vs2 + vs1; });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2, uint64_t vs1) -> uint64_t { return vs2 + vs1; });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Vector subtract.
void Vsub(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t vs2, uint8_t vs1) -> uint8_t { return vs2 - vs1; });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2, uint16_t vs1) -> uint16_t { return vs2 - vs1; });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2, uint32_t vs1) -> uint32_t { return vs2 - vs1; });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2, uint64_t vs1) -> uint64_t { return vs2 - vs1; });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Vector reverse subtract.
void Vrsub(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t vs2, uint8_t vs1) -> uint8_t { return vs1 - vs2; });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2, uint16_t vs1) -> uint16_t { return vs1 - vs2; });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2, uint32_t vs1) -> uint32_t { return vs1 - vs2; });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2, uint64_t vs1) -> uint64_t { return vs1 - vs2; });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Vector logical operations.

// Vector and.
void Vand(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t vs2, uint8_t vs1) -> uint8_t { return vs2 & vs1; });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2, uint16_t vs1) -> uint16_t { return vs2 & vs1; });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2, uint32_t vs1) -> uint32_t { return vs2 & vs1; });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2, uint64_t vs1) -> uint64_t { return vs2 & vs1; });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Vector or.
void Vor(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t vs2, uint8_t vs1) -> uint8_t { return vs2 | vs1; });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2, uint16_t vs1) -> uint16_t { return vs2 | vs1; });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2, uint32_t vs1) -> uint32_t { return vs2 | vs1; });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2, uint64_t vs1) -> uint64_t { return vs2 | vs1; });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Vector xor.
void Vxor(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t vs2, uint8_t vs1) -> uint8_t { return vs2 ^ vs1; });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2, uint16_t vs1) -> uint16_t { return vs2 ^ vs1; });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2, uint32_t vs1) -> uint32_t { return vs2 ^ vs1; });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2, uint64_t vs1) -> uint64_t { return vs2 ^ vs1; });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Vector shift operations.

// Vector shift left logical.
void Vsll(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2, uint8_t vs1) -> uint8_t {
            return vs2 << (vs1 & 0b111);
          });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst, [](uint16_t vs2, uint16_t vs1) -> uint16_t {
            return vs2 << (vs1 & 0b1111);
          });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst, [](uint32_t vs2, uint32_t vs1) -> uint32_t {
            return vs2 << (vs1 & 0b1'1111);
          });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst, [](uint64_t vs2, uint64_t vs1) -> uint64_t {
            return vs2 << (vs1 & 0b11'1111);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Vector shift right logical.
void Vsrl(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2, uint8_t vs1) -> uint8_t {
            return vs2 >> (vs1 & 0b111);
          });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst, [](uint16_t vs2, uint16_t vs1) -> uint16_t {
            return vs2 >> (vs1 & 0b1111);
          });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst, [](uint32_t vs2, uint32_t vs1) -> uint32_t {
            return vs2 >> (vs1 & 0b1'1111);
          });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst, [](uint64_t vs2, uint64_t vs1) -> uint64_t {
            return vs2 >> (vs1 & 0b11'1111);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Vector shift right arithmetic.
void Vsra(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<int8_t, int8_t, int8_t>(
          rv_vector, inst, [](int8_t vs2, int8_t vs1) -> int8_t {
            return vs2 >> (vs1 & 0b111);
          });
    case 2:
      return RiscVBinaryVectorOp<int16_t, int16_t, int16_t>(
          rv_vector, inst, [](int16_t vs2, int16_t vs1) -> int16_t {
            return vs2 >> (vs1 & 0b1111);
          });
    case 4:
      return RiscVBinaryVectorOp<int32_t, int32_t, int32_t>(
          rv_vector, inst, [](int32_t vs2, int32_t vs1) -> int32_t {
            return vs2 >> (vs1 & 0b1'1111);
          });
    case 8:
      return RiscVBinaryVectorOp<int64_t, int64_t, int64_t>(
          rv_vector, inst, [](int64_t vs2, int64_t vs1) -> int64_t {
            return vs2 >> (vs1 & 0b11'1111);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Vector narrowing shift operations. Narrow from sew * 2 to sew.

// Vector narrowing shift right logical. Source op 0 is shifted right
// by source op 1 and the result is 1/2 the size of source op 0.
void Vnsrl(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  // LMUL8 cannot be 64.
  if (rv_vector->vector_length_multiplier() > 32) {
    rv_vector->set_vector_exception();
    LOG(ERROR) << "Vector length multiplier out of range for narrowing shift";
    return;
  }
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint16_t, uint8_t>(
          rv_vector, inst, [](uint16_t vs2, uint8_t vs1) -> uint8_t {
            return static_cast<uint8_t>(vs2 >> (vs1 & 0b1111));
          });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint32_t, uint16_t>(
          rv_vector, inst, [](uint32_t vs2, uint16_t vs1) -> uint16_t {
            return static_cast<uint16_t>(vs2 >> (vs1 & 0b1'1111));
          });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint64_t, uint32_t>(
          rv_vector, inst, [](uint64_t vs2, uint32_t vs1) -> uint32_t {
            return static_cast<uint32_t>(vs2 >> (vs1 & 0b11'1111));
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value for narrowing shift right: " << sew;
      return;
  }
}

// Vector narrowing shift right arithmetic. Source op 0 is shifted right
// by source op 1 and the result is 1/2 the size of source op 0.
void Vnsra(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  // If the vector length multiplier (x8) is greater than 32, that means that
  // the source values (sew * 2) would exceed the available register group.
  if (rv_vector->vector_length_multiplier() > 32) {
    rv_vector->set_vector_exception();
    LOG(ERROR) << "Vector length multiplier out of range for narrowing shift";
    return;
  }
  // Note, sew cannot be 64 bits, as there is no support for operations on
  // 128 bit quantities.
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<int8_t, int16_t, int8_t>(
          rv_vector, inst, [](int16_t vs2, int8_t vs1) -> int8_t {
            return vs2 >> (vs1 & 0b1111);
          });
    case 2:
      return RiscVBinaryVectorOp<int16_t, int32_t, int16_t>(
          rv_vector, inst, [](int32_t vs2, int16_t vs1) -> int16_t {
            return vs2 >> (vs1 & 0b1'1111);
          });
    case 4:
      return RiscVBinaryVectorOp<int32_t, int64_t, int32_t>(
          rv_vector, inst, [](int64_t vs2, int32_t vs1) -> int32_t {
            return vs2 >> (vs1 & 0b11'1111);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value for narrowing shift right: " << sew;
      return;
  }
}

// Vector unsigned min.
void Vminu(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2, uint8_t vs1) -> uint8_t {
            return std::min(vs2, vs1);
          });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst, [](uint16_t vs2, uint16_t vs1) -> uint16_t {
            return std::min(vs2, vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst, [](uint32_t vs2, uint32_t vs1) -> uint32_t {
            return std::min(vs2, vs1);
          });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst, [](uint64_t vs2, uint64_t vs1) -> uint64_t {
            return std::min(vs2, vs1);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Vector signed min.
void Vmin(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<int8_t, int8_t, int8_t>(
          rv_vector, inst,
          [](int8_t vs2, int8_t vs1) -> int8_t { return std::min(vs2, vs1); });
    case 2:
      return RiscVBinaryVectorOp<int16_t, int16_t, int16_t>(
          rv_vector, inst, [](int16_t vs2, int16_t vs1) -> int16_t {
            return std::min(vs2, vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<int32_t, int32_t, int32_t>(
          rv_vector, inst, [](int32_t vs2, int32_t vs1) -> int32_t {
            return std::min(vs2, vs1);
          });
    case 8:
      return RiscVBinaryVectorOp<int64_t, int64_t, int64_t>(
          rv_vector, inst, [](int64_t vs2, int64_t vs1) -> int64_t {
            return std::min(vs2, vs1);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Vector unsigned max.
void Vmaxu(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2, uint8_t vs1) -> uint8_t {
            return std::max(vs2, vs1);
          });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst, [](uint16_t vs2, uint16_t vs1) -> uint16_t {
            return std::max(vs2, vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst, [](uint32_t vs2, uint32_t vs1) -> uint32_t {
            return std::max(vs2, vs1);
          });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst, [](uint64_t vs2, uint64_t vs1) -> uint64_t {
            return std::max(vs2, vs1);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Vector signed max.
void Vmax(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<int8_t, int8_t, int8_t>(
          rv_vector, inst,
          [](int8_t vs2, int8_t vs1) -> int8_t { return std::max(vs2, vs1); });
    case 2:
      return RiscVBinaryVectorOp<int16_t, int16_t, int16_t>(
          rv_vector, inst, [](int16_t vs2, int16_t vs1) -> int16_t {
            return std::max(vs2, vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<int32_t, int32_t, int32_t>(
          rv_vector, inst, [](int32_t vs2, int32_t vs1) -> int32_t {
            return std::max(vs2, vs1);
          });
    case 8:
      return RiscVBinaryVectorOp<int64_t, int64_t, int64_t>(
          rv_vector, inst, [](int64_t vs2, int64_t vs1) -> int64_t {
            return std::max(vs2, vs1);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Set equal.
void Vmseq(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorMaskOp<uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t vs2, uint8_t vs1) -> bool { return (vs2 == vs1); });
    case 2:
      return RiscVBinaryVectorMaskOp<uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2, uint16_t vs1) -> bool { return (vs2 == vs1); });
    case 4:
      return RiscVBinaryVectorMaskOp<uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2, uint32_t vs1) -> bool { return (vs2 == vs1); });
    case 8:
      return RiscVBinaryVectorMaskOp<uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2, uint64_t vs1) -> bool { return (vs2 == vs1); });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Vector compare instructions.

// Set not equal.
void Vmsne(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorMaskOp<uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t vs2, uint8_t vs1) -> bool { return (vs2 != vs1); });
    case 2:
      return RiscVBinaryVectorMaskOp<uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2, uint16_t vs1) -> bool { return (vs2 != vs1); });
    case 4:
      return RiscVBinaryVectorMaskOp<uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2, uint32_t vs1) -> bool { return (vs2 != vs1); });
    case 8:
      return RiscVBinaryVectorMaskOp<uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2, uint64_t vs1) -> bool { return (vs2 != vs1); });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Set less than unsigned.
void Vmsltu(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorMaskOp<uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t vs2, uint8_t vs1) -> bool { return (vs2 < vs1); });
    case 2:
      return RiscVBinaryVectorMaskOp<uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2, uint16_t vs1) -> bool { return (vs2 < vs1); });
    case 4:
      return RiscVBinaryVectorMaskOp<uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2, uint32_t vs1) -> bool { return (vs2 < vs1); });
    case 8:
      return RiscVBinaryVectorMaskOp<uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2, uint64_t vs1) -> bool { return (vs2 < vs1); });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Set less than.
void Vmslt(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorMaskOp<int8_t, int8_t>(
          rv_vector, inst,
          [](int8_t vs2, int8_t vs1) -> bool { return (vs2 < vs1); });
    case 2:
      return RiscVBinaryVectorMaskOp<int16_t, int16_t>(
          rv_vector, inst,
          [](int16_t vs2, int16_t vs1) -> bool { return (vs2 < vs1); });
    case 4:
      return RiscVBinaryVectorMaskOp<int32_t, int32_t>(
          rv_vector, inst,
          [](int32_t vs2, int32_t vs1) -> bool { return (vs2 < vs1); });
    case 8:
      return RiscVBinaryVectorMaskOp<int64_t, int64_t>(
          rv_vector, inst,
          [](int64_t vs2, int64_t vs1) -> bool { return (vs2 < vs1); });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Set less than or equal unsigned.
void Vmsleu(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorMaskOp<uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t vs2, uint8_t vs1) -> bool { return (vs2 <= vs1); });
    case 2:
      return RiscVBinaryVectorMaskOp<uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2, uint16_t vs1) -> bool { return (vs2 <= vs1); });
    case 4:
      return RiscVBinaryVectorMaskOp<uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2, uint32_t vs1) -> bool { return (vs2 <= vs1); });
    case 8:
      return RiscVBinaryVectorMaskOp<uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2, uint64_t vs1) -> bool { return (vs2 <= vs1); });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Set less than or equal.
void Vmsle(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorMaskOp<int8_t, int8_t>(
          rv_vector, inst,
          [](int8_t vs2, int8_t vs1) -> bool { return (vs2 <= vs1); });
    case 2:
      return RiscVBinaryVectorMaskOp<int16_t, int16_t>(
          rv_vector, inst,
          [](int16_t vs2, int16_t vs1) -> bool { return (vs2 <= vs1); });
    case 4:
      return RiscVBinaryVectorMaskOp<int32_t, int32_t>(
          rv_vector, inst,
          [](int32_t vs2, int32_t vs1) -> bool { return (vs2 <= vs1); });
    case 8:
      return RiscVBinaryVectorMaskOp<int64_t, int64_t>(
          rv_vector, inst,
          [](int64_t vs2, int64_t vs1) -> bool { return (vs2 <= vs1); });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Set greater than unsigned.
void Vmsgtu(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorMaskOp<uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t vs2, uint8_t vs1) -> bool { return (vs2 > vs1); });
    case 2:
      return RiscVBinaryVectorMaskOp<uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2, uint16_t vs1) -> bool { return (vs2 > vs1); });
    case 4:
      return RiscVBinaryVectorMaskOp<uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2, uint32_t vs1) -> bool { return (vs2 > vs1); });
    case 8:
      return RiscVBinaryVectorMaskOp<uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2, uint64_t vs1) -> bool { return (vs2 > vs1); });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Set greater than.
void Vmsgt(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorMaskOp<int8_t, int8_t>(
          rv_vector, inst,
          [](int8_t vs2, int8_t vs1) -> bool { return (vs2 > vs1); });
    case 2:
      return RiscVBinaryVectorMaskOp<int16_t, int16_t>(
          rv_vector, inst,
          [](int16_t vs2, int16_t vs1) -> bool { return (vs2 > vs1); });
    case 4:
      return RiscVBinaryVectorMaskOp<int32_t, int32_t>(
          rv_vector, inst,
          [](int32_t vs2, int32_t vs1) -> bool { return (vs2 > vs1); });
    case 8:
      return RiscVBinaryVectorMaskOp<int64_t, int64_t>(
          rv_vector, inst,
          [](int64_t vs2, int64_t vs1) -> bool { return (vs2 > vs1); });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Saturated unsigned addition.
void Vsaddu(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [rv_vector](uint8_t vs2, uint8_t vs1) -> uint8_t {
            uint8_t sum = vs2 + vs1;
            if (sum < vs1) {
              sum = numeric_limits<uint8_t>::max();
              rv_vector->set_vxsat(true);
            }
            return sum;
          });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst, [rv_vector](uint16_t vs2, uint16_t vs1) -> uint16_t {
            uint16_t sum = vs2 + vs1;
            if (sum < vs1) {
              sum = numeric_limits<uint16_t>::max();
              rv_vector->set_vxsat(true);
            }
            return sum;
          });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst, [rv_vector](uint32_t vs2, uint32_t vs1) -> uint32_t {
            uint32_t sum = vs2 + vs1;
            if (sum < vs1) {
              sum = numeric_limits<uint32_t>::max();
              rv_vector->set_vxsat(true);
            }
            return sum;
          });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst, [rv_vector](uint64_t vs2, uint64_t vs1) -> uint64_t {
            uint64_t sum = vs2 + vs1;
            if (sum < vs1) {
              sum = numeric_limits<uint64_t>::max();
              rv_vector->set_vxsat(true);
            }
            return sum;
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Helper function for Vsadd.
// Uses unsigned arithmetic for the addition to avoid signed overflow, which,
// when compiled with --config=asan, will trigger an exception.
template <typename T>
inline T VsaddHelper(T vs2, T vs1, RiscVVectorState* rv_vector) {
  using UT = typename std::make_unsigned<T>::type;
  UT uvs2 = static_cast<UT>(vs2);
  UT uvs1 = static_cast<UT>(vs1);
  UT usum = uvs2 + uvs1;
  T sum = static_cast<T>(usum);
  if (((vs2 ^ vs1) >= 0) && ((sum ^ vs2) < 0)) {
    rv_vector->set_vxsat(true);
    return vs2 > 0 ? numeric_limits<T>::max() : numeric_limits<T>::min();
  }
  return sum;
}

// Saturated signed addition.
void Vsadd(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<int8_t, int8_t, int8_t>(
          rv_vector, inst, [rv_vector](int8_t vs2, int8_t vs1) -> int8_t {
            return VsaddHelper(vs2, vs1, rv_vector);
          });
    case 2:
      return RiscVBinaryVectorOp<int16_t, int16_t, int16_t>(
          rv_vector, inst, [rv_vector](int16_t vs2, int16_t vs1) -> int16_t {
            return VsaddHelper(vs2, vs1, rv_vector);
          });
    case 4:
      return RiscVBinaryVectorOp<int32_t, int32_t, int32_t>(
          rv_vector, inst, [rv_vector](int32_t vs2, int32_t vs1) -> int32_t {
            return VsaddHelper(vs2, vs1, rv_vector);
          });
    case 8:
      return RiscVBinaryVectorOp<int64_t, int64_t, int64_t>(
          rv_vector, inst, [rv_vector](int64_t vs2, int64_t vs1) -> int64_t {
            return VsaddHelper(vs2, vs1, rv_vector);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Saturated unsigned subtract.
void Vssubu(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [rv_vector](uint8_t vs2, uint8_t vs1) -> uint8_t {
            uint8_t diff = vs2 - vs1;
            if (vs2 < vs1) {
              diff = 0;
              rv_vector->set_vxsat(true);
            }
            return diff;
          });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst, [rv_vector](uint16_t vs2, uint16_t vs1) -> uint16_t {
            uint16_t diff = vs2 - vs1;
            if (vs2 < vs1) {
              diff = 0;
              rv_vector->set_vxsat(true);
            }
            return diff;
          });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst, [rv_vector](uint32_t vs2, uint32_t vs1) -> uint32_t {
            uint32_t diff = vs2 - vs1;
            if (vs2 < vs1) {
              diff = 0;
              rv_vector->set_vxsat(true);
            }
            return diff;
          });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst, [rv_vector](uint64_t vs2, uint64_t vs1) -> uint64_t {
            uint64_t diff = vs2 - vs1;
            if (vs2 < vs1) {
              diff = 0;
              rv_vector->set_vxsat(true);
            }
            return diff;
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

template <typename T>
T VssubHelper(T vs2, T vs1, RiscVVectorState* rv_vector) {
  using UT = typename std::make_unsigned<T>::type;
  UT uvs2 = static_cast<UT>(vs2);
  UT uvs1 = static_cast<UT>(vs1);
  UT udiff = uvs2 - uvs1;
  T diff = static_cast<T>(udiff);
  if (((vs2 ^ vs1) < 0) && ((diff ^ vs1) >= 0)) {
    rv_vector->set_vxsat(true);
    return vs1 < 0 ? numeric_limits<T>::max() : numeric_limits<T>::min();
  }
  return diff;
}

// Saturated signed subtract.
void Vssub(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<int8_t, int8_t, int8_t>(
          rv_vector, inst, [rv_vector](int8_t vs2, int8_t vs1) -> int8_t {
            return VssubHelper(vs2, vs1, rv_vector);
          });
    case 2:
      return RiscVBinaryVectorOp<int16_t, int16_t, int16_t>(
          rv_vector, inst, [rv_vector](int16_t vs2, int16_t vs1) -> int16_t {
            return VssubHelper(vs2, vs1, rv_vector);
          });
    case 4:
      return RiscVBinaryVectorOp<int32_t, int32_t, int32_t>(
          rv_vector, inst, [rv_vector](int32_t vs2, int32_t vs1) -> int32_t {
            return VssubHelper(vs2, vs1, rv_vector);
          });
    case 8:
      return RiscVBinaryVectorOp<int64_t, int64_t, int64_t>(
          rv_vector, inst, [rv_vector](int64_t vs2, int64_t vs1) -> int64_t {
            return VssubHelper(vs2, vs1, rv_vector);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Add/Subtract with carry, carry generation.
void Vadc(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVMaskBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2, uint8_t vs1, bool mask) -> uint8_t {
            return vs2 + vs1 + static_cast<uint8_t>(mask);
          });
    case 2:
      return RiscVMaskBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2, uint16_t vs1, bool mask) -> uint16_t {
            return vs2 + vs1 + static_cast<uint16_t>(mask);
          });
    case 4:
      return RiscVMaskBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2, uint32_t vs1, bool mask) -> uint32_t {
            return vs2 + vs1 + static_cast<uint32_t>(mask);
          });
    case 8:
      return RiscVMaskBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2, uint64_t vs1, bool mask) -> uint64_t {
            return vs2 + vs1 + static_cast<uint64_t>(mask);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Add with carry - carry generation.
void Vmadc(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVSetMaskBinaryVectorMaskOp<uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2, uint8_t vs1, bool mask) -> bool {
            uint16_t sum = static_cast<uint16_t>(vs2) +
                           static_cast<uint16_t>(vs1) +
                           static_cast<uint16_t>(mask);
            sum >>= 8;
            return sum;
          });
    case 2:
      return RiscVSetMaskBinaryVectorMaskOp<uint16_t, uint16_t>(
          rv_vector, inst, [](uint16_t vs2, uint16_t vs1, bool mask) -> bool {
            uint32_t sum = static_cast<uint32_t>(vs2) +
                           static_cast<uint32_t>(vs1) +
                           static_cast<uint32_t>(mask);
            sum >>= 16;
            return sum != 0;
          });
    case 4:
      return RiscVSetMaskBinaryVectorMaskOp<uint32_t, uint32_t>(
          rv_vector, inst, [](uint32_t vs2, uint32_t vs1, bool mask) -> bool {
            uint64_t sum = static_cast<uint64_t>(vs2) +
                           static_cast<uint64_t>(vs1) +
                           static_cast<uint64_t>(mask);
            sum >>= 32;
            return sum != 0;
          });
    case 8:
      return RiscVSetMaskBinaryVectorMaskOp<uint64_t, uint64_t>(
          rv_vector, inst, [](uint64_t vs2, uint64_t vs1, bool mask) -> bool {
            // Compute carry by doing two additions. First get the carry out
            // from adding the low byte.
            uint64_t carry =
                (vs1 & 0xff + vs2 & 0xff + static_cast<uint64_t>(mask)) >> 8;
            // Now add the high 7 bytes together with the carry from the low
            // byte addition.
            uint64_t sum = (vs1 >> 8) + (vs2 >> 8) + carry;
            // The carry out is in the high byte.
            sum >>= 56;
            return sum != 0;
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Subtract with borrow.
void Vsbc(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVMaskBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2, uint8_t vs1, bool mask) -> uint8_t {
            return vs2 - vs1 - static_cast<uint8_t>(mask);
          });
    case 2:
      return RiscVMaskBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2, uint16_t vs1, bool mask) -> uint16_t {
            return vs2 - vs1 - static_cast<uint16_t>(mask);
          });
    case 4:
      return RiscVMaskBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2, uint32_t vs1, bool mask) -> uint32_t {
            return vs2 - vs1 - static_cast<uint32_t>(mask);
          });
    case 8:
      return RiscVMaskBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2, uint64_t vs1, bool mask) -> uint64_t {
            return vs2 - vs1 - static_cast<uint64_t>(mask);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Subtract with borrow - borrow generation.
void Vmsbc(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVSetMaskBinaryVectorMaskOp<uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2, uint8_t vs1, bool mask) -> bool {
            return static_cast<uint16_t>(vs2) <
                   static_cast<uint16_t>(mask) + static_cast<uint16_t>(vs1);
          });
    case 2:
      return RiscVSetMaskBinaryVectorMaskOp<uint16_t, uint16_t>(
          rv_vector, inst, [](uint16_t vs2, uint16_t vs1, bool mask) -> bool {
            return static_cast<uint32_t>(vs2) <
                   static_cast<uint32_t>(mask) + static_cast<uint32_t>(vs1);
          });
    case 4:
      return RiscVSetMaskBinaryVectorMaskOp<uint32_t, uint32_t>(
          rv_vector, inst, [](uint32_t vs2, uint32_t vs1, bool mask) -> bool {
            return static_cast<uint64_t>(vs2) <
                   static_cast<uint64_t>(mask) + static_cast<uint64_t>(vs1);
          });
    case 8:
      return RiscVSetMaskBinaryVectorMaskOp<uint64_t, uint64_t>(
          rv_vector, inst, [](uint64_t vs2, uint64_t vs1, bool mask) -> bool {
            if (vs2 < vs1) return true;
            if (vs2 == vs1) return mask;
            return false;
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Vector merge.
void Vmerge(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVMaskBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2, uint8_t vs1, bool mask) -> uint8_t {
            return mask ? vs1 : vs2;
          });
    case 2:
      return RiscVMaskBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2, uint16_t vs1, bool mask) -> uint16_t {
            return mask ? vs1 : vs2;
          });
    case 4:
      return RiscVMaskBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2, uint32_t vs1, bool mask) -> uint32_t {
            return mask ? vs1 : vs2;
          });
    case 8:
      return RiscVMaskBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2, uint64_t vs1, bool mask) -> uint64_t {
            return mask ? vs1 : vs2;
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Vector move register(s).
void Vmvr(int num_regs, Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  if (rv_vector->vector_exception()) return;

  auto* src_op = static_cast<RV32VectorSourceOperand*>(inst->Source(0));
  auto* dest_op =
      static_cast<RV32VectorDestinationOperand*>(inst->Destination(0));
  if (src_op->size() < num_regs) {
    LOG(ERROR) << "Vmvr: source operand has fewer registers than requested";
    rv_vector->set_vector_exception();
    return;
  }
  if (dest_op->size() < num_regs) {
    LOG(ERROR)
        << "Vmvr: destination operand has fewer registers than requested";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  int num_elements_per_vector = rv_vector->vector_register_byte_length() / sew;
  int vstart = rv_vector->vstart();
  int start_reg = vstart / num_elements_per_vector;
  for (int i = start_reg; i < num_regs; i++) {
    auto* src_db = src_op->GetRegister(i)->data_buffer();
    auto* dest_db = dest_op->AllocateDataBuffer(i);
    std::memcpy(dest_db->raw_ptr(), src_db->raw_ptr(),
                dest_db->size<uint8_t>());
    dest_db->Submit();
  }
  rv_vector->clear_vstart();
}

// Templated helper function for shift right with rounding.
template <typename T>
T VssrHelper(RiscVVectorState* rv_vector, T vs2, T vs1) {
  using UT = typename MakeUnsigned<T>::type;
  int rm = rv_vector->vxrm();
  int max_shift = (sizeof(T) << 3) - 1;
  int shift_amount = static_cast<int>(vs1 & max_shift);
  // Create mask for the bits that will be shifted out + 1.
  UT round_bits = vs2;
  if (shift_amount < max_shift) {
    UT mask = numeric_limits<UT>::max();
    mask = ~(numeric_limits<UT>::max() << shift_amount + 1);
    round_bits = vs2 & mask;
  }
  vs2 >>= shift_amount;
  vs2 += static_cast<T>(GetRoundingBit(rm, round_bits, shift_amount + 1));
  return vs2;
}

// Logical shift right with rounding.
void Vssrl(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [rv_vector](uint8_t vs2, uint8_t vs1) -> uint8_t {
            return VssrHelper(rv_vector, vs2, vs1);
          });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst, [rv_vector](uint16_t vs2, uint16_t vs1) -> uint16_t {
            return VssrHelper(rv_vector, vs2, vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst, [rv_vector](uint32_t vs2, uint32_t vs1) -> uint32_t {
            return VssrHelper(rv_vector, vs2, vs1);
          });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst, [rv_vector](uint64_t vs2, uint64_t vs1) -> uint64_t {
            return VssrHelper(rv_vector, vs2, vs1);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Arithmetic shift right with rounding.
void Vssra(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<int8_t, int8_t, int8_t>(
          rv_vector, inst, [rv_vector](int8_t vs2, int8_t vs1) -> int8_t {
            return VssrHelper(rv_vector, vs2, vs1);
          });
    case 2:
      return RiscVBinaryVectorOp<int16_t, int16_t, int16_t>(
          rv_vector, inst, [rv_vector](int16_t vs2, int16_t vs1) -> int16_t {
            return VssrHelper(rv_vector, vs2, vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<int32_t, int32_t, int32_t>(
          rv_vector, inst, [rv_vector](int32_t vs2, int32_t vs1) -> int32_t {
            return VssrHelper(rv_vector, vs2, vs1);
          });
    case 8:
      return RiscVBinaryVectorOp<int64_t, int64_t, int64_t>(
          rv_vector, inst, [rv_vector](int64_t vs2, int64_t vs1) -> int64_t {
            return VssrHelper(rv_vector, vs2, vs1);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Templated helper function for shift right with rounding and saturation.
template <typename DT, typename WT, typename T>
T VnclipHelper(RiscVVectorState* rv_vector, WT vs2, T vs1) {
  using WUT = typename std::make_unsigned<WT>::type;
  int rm = rv_vector->vxrm();
  int max_shift = (sizeof(WT) << 3) - 1;
  int shift_amount = vs1 & ((sizeof(WT) << 3) - 1);
  // Create mask for the bits that will be shifted out + 1.
  WUT mask = vs2;
  if (shift_amount < max_shift) {
    mask = ~(numeric_limits<WUT>::max() << (shift_amount + 1));
  }
  WUT round_bits = vs2 & mask;
  // Perform the rounded shift.
  vs2 =
      (vs2 >> shift_amount) + GetRoundingBit(rm, round_bits, shift_amount + 1);
  // Saturate if needed.
  if (vs2 > numeric_limits<DT>::max()) {
    rv_vector->set_vxsat(true);
    return numeric_limits<DT>::max();
  }
  if (vs2 < numeric_limits<DT>::min()) {
    rv_vector->set_vxsat(true);
    return numeric_limits<DT>::min();
  }
  return static_cast<DT>(vs2);
}

// Arithmetic shift right and narrowing from 2*sew to sew with rounding and
// signed saturation.
void Vnclip(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int lmul8 = rv_vector->vector_length_multiplier();
  // This is a narrowing operation and sew is that of the narrow data type.
  // Thus if lmul > 32, then emul for the wider data type is illegal.
  if (lmul8 > 32) {
    LOG(ERROR) << "Illegal lmul value";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<int8_t, int16_t, int8_t>(
          rv_vector, inst, [rv_vector](int16_t vs2, int8_t vs1) -> int8_t {
            return VnclipHelper<int8_t, int16_t, int8_t>(rv_vector, vs2, vs1);
          });
    case 2:
      return RiscVBinaryVectorOp<int16_t, int32_t, int16_t>(
          rv_vector, inst, [rv_vector](int32_t vs2, int16_t vs1) -> int16_t {
            return VnclipHelper<int16_t, int32_t, int16_t>(rv_vector, vs2, vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<int32_t, int64_t, int32_t>(
          rv_vector, inst, [rv_vector](int64_t vs2, int32_t vs1) -> int32_t {
            return VnclipHelper<int32_t, int64_t, int32_t>(rv_vector, vs2, vs1);
          });
    case 8:
      // There is no valid sew * 2 = 16.
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Logical shift right and narrowing from 2*sew to sew with rounding and
// unsigned saturation.
void Vnclipu(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int lmul8 = rv_vector->vector_length_multiplier();
  // This is a narrowing operation and sew is that of the narrow data type.
  // Thus if lmul > 32, then emul for the wider data type is illegal.
  if (lmul8 > 32) {
    LOG(ERROR) << "Illegal lmul value";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint16_t, uint8_t>(
          rv_vector, inst, [rv_vector](uint16_t vs2, uint8_t vs1) -> uint8_t {
            return VnclipHelper<uint8_t, uint16_t, uint8_t>(rv_vector, vs2,
                                                            vs1);
          });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint32_t, uint16_t>(
          rv_vector, inst, [rv_vector](uint32_t vs2, uint16_t vs1) -> uint16_t {
            return VnclipHelper<uint16_t, uint32_t, uint16_t>(rv_vector, vs2,
                                                              vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint64_t, uint32_t>(
          rv_vector, inst, [rv_vector](uint64_t vs2, uint32_t vs1) -> uint32_t {
            return VnclipHelper<uint32_t, uint64_t, uint32_t>(rv_vector, vs2,
                                                              vs1);
          });
    case 8:
      // There is no valid sew * 2 = 16.
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Perform a signed multiply from T to wider int type. Shift that result
// right by sizeof(T) * 8 - 1 and round. Saturate if needed to fit into T.
template <typename T>
T VsmulHelper(RiscVVectorState* rv_vector, T vs2, T vs1) {
  using WT = typename WideType<T>::type;
  WT vd_w;
  WT vs2_w = static_cast<WT>(vs2);
  WT vs1_w = static_cast<WT>(vs1);
  vd_w = vs2_w * vs1_w;
  vd_w = VssrHelper<WT>(rv_vector, vd_w, sizeof(T) * 8 - 1);
  if (vd_w < numeric_limits<T>::min()) {
    rv_vector->set_vxsat(true);
    return numeric_limits<T>::min();
  }
  if (vd_w > numeric_limits<T>::max()) {
    rv_vector->set_vxsat(true);
    return numeric_limits<T>::max();
  }
  return static_cast<T>(vd_w);
}

// Vector fractional multiply with rounding and saturation.
void Vsmul(Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<int8_t, int8_t, int8_t>(
          rv_vector, inst, [rv_vector](int8_t vs2, int8_t vs1) -> int8_t {
            return VsmulHelper<int8_t>(rv_vector, vs2, vs1);
          });
    case 2:
      return RiscVBinaryVectorOp<int16_t, int16_t, int16_t>(
          rv_vector, inst, [rv_vector](int16_t vs2, int16_t vs1) -> int16_t {
            return VsmulHelper<int16_t>(rv_vector, vs2, vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<int32_t, int32_t, int32_t>(
          rv_vector, inst, [rv_vector](int32_t vs2, int32_t vs1) -> int32_t {
            return VsmulHelper<int32_t>(rv_vector, vs2, vs1);
          });
    case 8:
      return RiscVBinaryVectorOp<int64_t, int64_t, int64_t>(
          rv_vector, inst, [rv_vector](int64_t vs2, int64_t vs1) -> int64_t {
            return VsmulHelper<int64_t>(rv_vector, vs2, vs1);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
