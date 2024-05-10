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

#include "riscv/riscv_vector_opm_instructions.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <type_traits>

#include "absl/log/log.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_vector_instruction_helpers.h"
#include "riscv/riscv_vector_state.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::WideType;

// Helper function used to factor out some code from Vaadd* instructions.
template <typename T>
inline T VaaddHelper(RiscVVectorState *rv_vector, T vs2, T vs1) {
  // Perform the addition using a wider type, then shift and round.
  using WT = typename WideType<T>::type;
  WT vs2_w = static_cast<WT>(vs2);
  WT vs1_w = static_cast<WT>(vs1);
  auto res = RoundOff(rv_vector, vs2_w + vs1_w, 1);
  return static_cast<T>(res);
}

// Average unsigned add. The two sources are added, then shifted right by one
// and rounded.
void Vaaddu(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [rv_vector](uint8_t vs2, uint8_t vs1) -> uint8_t {
            return VaaddHelper(rv_vector, vs2, vs1);
          });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst, [rv_vector](uint16_t vs2, uint16_t vs1) -> uint16_t {
            return VaaddHelper(rv_vector, vs2, vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst, [rv_vector](uint32_t vs2, uint32_t vs1) -> uint32_t {
            return VaaddHelper(rv_vector, vs2, vs1);
          });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst, [rv_vector](uint64_t vs2, uint64_t vs1) -> uint64_t {
            return VaaddHelper(rv_vector, vs2, vs1);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Average signed add. The two sources are added, then shifted right by one and
// rounded.
void Vaadd(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<int8_t, int8_t, int8_t>(
          rv_vector, inst, [rv_vector](int8_t vs2, int8_t vs1) -> int8_t {
            return VaaddHelper(rv_vector, vs2, vs1);
          });
    case 2:
      return RiscVBinaryVectorOp<int16_t, int16_t, int16_t>(
          rv_vector, inst, [rv_vector](int16_t vs2, int16_t vs1) -> int16_t {
            return VaaddHelper(rv_vector, vs2, vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<int32_t, int32_t, int32_t>(
          rv_vector, inst, [rv_vector](int32_t vs2, int32_t vs1) -> int32_t {
            return VaaddHelper(rv_vector, vs2, vs1);
          });
    case 8:
      return RiscVBinaryVectorOp<int64_t, int64_t, int64_t>(
          rv_vector, inst, [rv_vector](int64_t vs2, int64_t vs1) -> int64_t {
            return VaaddHelper(rv_vector, vs2, vs1);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Helper function for Vasub* instructions. Subract using a wider type, then
// round.
template <typename T>
inline T VasubHelper(RiscVVectorState *rv_vector, T vs2, T vs1) {
  using WT = typename WideType<T>::type;
  WT vs2_w = static_cast<WT>(vs2);
  WT vs1_w = static_cast<WT>(vs1);
  auto res = RoundOff(rv_vector, vs2_w - vs1_w, 1);
  return static_cast<T>(res);
}

// Averaging unsigned subtract - subtract then shift right by 1 and round.
void Vasubu(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [rv_vector](uint8_t vs2, uint8_t vs1) -> uint8_t {
            return VasubHelper(rv_vector, vs2, vs1);
          });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst, [rv_vector](uint16_t vs2, uint16_t vs1) -> uint16_t {
            return VasubHelper(rv_vector, vs2, vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst, [rv_vector](uint32_t vs2, uint32_t vs1) -> uint32_t {
            return VasubHelper(rv_vector, vs2, vs1);
          });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst, [rv_vector](uint64_t vs2, uint64_t vs1) -> uint64_t {
            return VasubHelper(rv_vector, vs2, vs1);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Averaging signed subtract. Subtract then shift right by 1 and round.
void Vasub(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<int8_t, int8_t, int8_t>(
          rv_vector, inst, [rv_vector](int8_t vs2, int8_t vs1) -> int8_t {
            return VasubHelper(rv_vector, vs2, vs1);
          });
    case 2:
      return RiscVBinaryVectorOp<int16_t, int16_t, int16_t>(
          rv_vector, inst, [rv_vector](int16_t vs2, int16_t vs1) -> int16_t {
            return VasubHelper(rv_vector, vs2, vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<int32_t, int32_t, int32_t>(
          rv_vector, inst, [rv_vector](int32_t vs2, int32_t vs1) -> int32_t {
            return VasubHelper(rv_vector, vs2, vs1);
          });
    case 8:
      return RiscVBinaryVectorOp<int64_t, int64_t, int64_t>(
          rv_vector, inst, [rv_vector](int64_t vs2, int64_t vs1) -> int64_t {
            return VasubHelper(rv_vector, vs2, vs1);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Mask operands only operate on a single vector register. This helper function
// is used by the following bitwise mask manipulation instruction semantic
// functions.
static inline void BitwiseMaskBinaryOp(
    RiscVVectorState *rv_vector, const Instruction *inst,
    std::function<uint8_t(uint8_t, uint8_t)> op) {
  if (rv_vector->vector_exception()) return;
  int vstart = rv_vector->vstart();
  int vlen = rv_vector->vector_length();
  // Get spans for vector source and destination registers.
  auto *vs2_op = static_cast<RV32VectorSourceOperand *>(inst->Source(0));
  auto vs2_span = vs2_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  auto *vs1_op = static_cast<RV32VectorSourceOperand *>(inst->Source(1));
  auto vs1_span = vs1_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  auto *vd_op =
      static_cast<RV32VectorDestinationOperand *>(inst->Destination(0));
  auto *vd_db = vd_op->CopyDataBuffer();
  auto vd_span = vd_db->Get<uint8_t>();
  // Compute start and end locations.
  int start_byte = vstart / 8;
  int start_offset = vstart % 8;
  uint8_t start_mask = 0b1111'1111 << start_offset;
  int end_byte = (vlen - 1) / 8;
  int end_offset = (vlen - 1) % 8;
  uint8_t end_mask = 0b1111'1111 >> (7 - end_offset);
  // The start byte is computed first, applying a mask to mask out any preceding
  // bits.
  vd_span[start_byte] =
      (op(vs2_span[start_byte], vs1_span[start_byte]) & start_mask) |
      (vd_span[start_byte] & ~start_mask);
  // Perform the bitwise operation on each byte between start and end.
  for (int i = start_byte + 1; i < end_byte; i++) {
    vd_span[i] = op(vs2_span[i], vs1_span[i]);
  }
  // Perform the bitwise operation with a mask on the end byte.
  vd_span[end_byte] = (op(vs2_span[end_byte], vs1_span[end_byte]) & end_mask) |
                      (vd_span[end_byte] & ~end_mask);
  vd_db->Submit();
  rv_vector->clear_vstart();
}

// Bitwise vector mask instructions. The operation is clear by their name.
void Vmandnot(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  BitwiseMaskBinaryOp(rv_vector, inst, [](uint8_t vs2, uint8_t vs1) -> uint8_t {
    return vs2 & ~vs1;
  });
}

void Vmand(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  BitwiseMaskBinaryOp(rv_vector, inst, [](uint8_t vs2, uint8_t vs1) -> uint8_t {
    return vs2 & vs1;
  });
}
void Vmor(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  BitwiseMaskBinaryOp(rv_vector, inst, [](uint8_t vs2, uint8_t vs1) -> uint8_t {
    return vs2 | vs1;
  });
}
void Vmxor(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  BitwiseMaskBinaryOp(rv_vector, inst, [](uint8_t vs2, uint8_t vs1) -> uint8_t {
    return vs2 ^ vs1;
  });
}
void Vmornot(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  BitwiseMaskBinaryOp(rv_vector, inst, [](uint8_t vs2, uint8_t vs1) -> uint8_t {
    return vs2 | ~vs1;
  });
}
void Vmnand(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  BitwiseMaskBinaryOp(rv_vector, inst, [](uint8_t vs2, uint8_t vs1) -> uint8_t {
    return ~(vs2 & vs1);
  });
}
void Vmnor(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  BitwiseMaskBinaryOp(rv_vector, inst, [](uint8_t vs2, uint8_t vs1) -> uint8_t {
    return ~(vs2 | vs1);
  });
}
void Vmxnor(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  BitwiseMaskBinaryOp(rv_vector, inst, [](uint8_t vs2, uint8_t vs1) -> uint8_t {
    return ~(vs2 ^ vs1);
  });
}

// Vector unsigned divide. Note, just like the scalar divide instruction, a
// divide by zero does not cause an exception, instead it returns all 1s.
void Vdivu(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2, uint8_t vs1) -> uint8_t {
            if (vs1 == 0) return ~vs1;
            return vs2 / vs1;
          });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst, [](uint16_t vs2, uint16_t vs1) -> uint16_t {
            if (vs1 == 0) return ~vs1;
            return vs2 / vs1;
          });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst, [](uint32_t vs2, uint32_t vs1) -> uint32_t {
            if (vs1 == 0) return ~vs1;
            return vs2 / vs1;
          });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst, [](uint64_t vs2, uint64_t vs1) -> uint64_t {
            if (vs1 == 0) return ~vs1;
            return vs2 / vs1;
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Signed divide. Divide by 0 returns all 1s. If -1 is divided by the largest
// magnitude negative number, it returns that negative number.
void Vdiv(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<int8_t, int8_t, int8_t>(
          rv_vector, inst, [](int8_t vs2, int8_t vs1) -> int8_t {
            if (vs1 == 0) return static_cast<int8_t>(-1);
            if ((vs1 == -1) && (vs2 == std::numeric_limits<int8_t>::min())) {
              return std::numeric_limits<int8_t>::min();
            }
            return vs2 / vs1;
          });
    case 2:
      return RiscVBinaryVectorOp<int16_t, int16_t, int16_t>(
          rv_vector, inst, [](int16_t vs2, int16_t vs1) -> int16_t {
            if (vs1 == 0) return static_cast<int16_t>(-1);
            if ((vs1 == -1) && (vs2 == std::numeric_limits<int16_t>::min())) {
              return std::numeric_limits<int16_t>::min();
            }
            return vs2 / vs1;
          });
    case 4:
      return RiscVBinaryVectorOp<int32_t, int32_t, int32_t>(
          rv_vector, inst, [](int32_t vs2, int32_t vs1) -> int32_t {
            if (vs1 == 0) return static_cast<int32_t>(-1);
            if ((vs1 == -1) && (vs2 == std::numeric_limits<int32_t>::min())) {
              return std::numeric_limits<int32_t>::min();
            }
            return vs2 / vs1;
          });
    case 8:
      return RiscVBinaryVectorOp<int64_t, int64_t, int64_t>(
          rv_vector, inst, [](int64_t vs2, int64_t vs1) -> int64_t {
            if (vs1 == 0) return static_cast<int64_t>(-1);
            if ((vs1 == -1) && (vs2 == std::numeric_limits<int64_t>::min())) {
              return std::numeric_limits<int64_t>::min();
            }
            return vs2 / vs1;
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Unsigned remainder. If the denominator is 0, it returns the enumerator.
void Vremu(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2, uint8_t vs1) -> uint8_t {
            if (vs1 == 0) return vs2;
            return vs2 % vs1;
          });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst, [](uint16_t vs2, uint16_t vs1) -> uint16_t {
            if (vs1 == 0) return vs2;
            return vs2 % vs1;
          });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst, [](uint32_t vs2, uint32_t vs1) -> uint32_t {
            if (vs1 == 0) return vs2;
            return vs2 % vs1;
          });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst, [](uint64_t vs2, uint64_t vs1) -> uint64_t {
            if (vs1 == 0) return vs2;
            return vs2 % vs1;
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Signed remainder. If the denominator is 0, it returns the enumerator.
void Vrem(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<int8_t, int8_t, int8_t>(
          rv_vector, inst, [](int8_t vs2, int8_t vs1) -> int8_t {
            if (vs1 == 0) return vs2;
            return vs2 % vs1;
          });
    case 2:
      return RiscVBinaryVectorOp<int16_t, int16_t, int16_t>(
          rv_vector, inst, [](int16_t vs2, int16_t vs1) -> int16_t {
            if (vs1 == 0) return vs2;
            return vs2 % vs1;
          });
    case 4:
      return RiscVBinaryVectorOp<int32_t, int32_t, int32_t>(
          rv_vector, inst, [](int32_t vs2, int32_t vs1) -> int32_t {
            if (vs1 == 0) return vs2;
            return vs2 % vs1;
          });
    case 8:
      return RiscVBinaryVectorOp<int64_t, int64_t, int64_t>(
          rv_vector, inst, [](int64_t vs2, int64_t vs1) -> int64_t {
            if (vs1 == 0) return vs2;
            return vs2 % vs1;
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Helper function for multiply high. It promotes the to arguments to wider
// types, performs the multiplication, returns the high half of the result.
template <typename T>
inline T VmulHighHelper(T vs2, T vs1) {
  using WT = typename WideType<T>::type;
  WT vs2_w = static_cast<WT>(vs2);
  WT vs1_w = static_cast<WT>(vs1);
  WT prod = vs2_w * vs1_w;
  prod >>= sizeof(T) * 8;
  return static_cast<T>(prod);
}

// Multiply high, unsigned.
void Vmulhu(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2, uint8_t vs1) -> uint8_t {
            return VmulHighHelper(vs2, vs1);
          });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst, [](uint16_t vs2, uint16_t vs1) -> uint16_t {
            return VmulHighHelper(vs2, vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst, [](uint32_t vs2, uint32_t vs1) -> uint32_t {
            return VmulHighHelper(vs2, vs1);
          });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst, [](uint64_t vs2, uint64_t vs1) -> uint64_t {
            return VmulHighHelper(vs2, vs1);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Signed multiply. Note, that signed and unsigned multiply operations have the
// same result for the low half of the product.
void Vmul(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t vs2, uint8_t vs1) -> uint8_t { return vs2 * vs1; });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst, [](uint16_t vs2, uint16_t vs1) -> uint16_t {
            uint32_t vs2_32 = vs2;
            uint32_t vs1_32 = vs1;
            return static_cast<uint16_t>(vs2_32 * vs1_32);
          });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2, uint32_t vs1) -> uint32_t { return vs2 * vs1; });
    case 8:
      // The 64 bit version is treated a little differently. Because the vs1
      // operand may come from a register which may be 32 bits wide, it's first
      // converted to int64_t. Then the product is done on unsigned numbers to
      // avoid a signed multiply overflow, and returned as a signed number.
      return RiscVBinaryVectorOp<int64_t, int64_t, int64_t>(
          rv_vector, inst, [](int64_t vs2, int64_t vs1) -> int64_t {
            uint64_t vs2_u = vs2;
            uint64_t vs1_u = vs1;
            uint64_t prod = vs2_u * vs1_u;
            return prod;
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Helper for signed-unsigned multiplication return high half.
template <typename T>
inline typename std::make_signed<T>::type VmulHighSUHelper(
    typename std::make_signed<T>::type vs2,
    typename std::make_unsigned<T>::type vs1) {
  using WT = typename WideType<T>::type;
  using WST = typename WideType<typename std::make_signed<T>::type>::type;
  WST vs2_w = static_cast<WST>(vs2);
  WT vs1_w = static_cast<WT>(vs1);
  WST prod = vs2_w * vs1_w;
  prod >>= sizeof(T) * 8;
  return static_cast<typename std::make_signed<T>::type>(prod);
}

// Multiply signed unsigned and return the high half.
void Vmulhsu(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<int8_t, int8_t, uint8_t>(
          rv_vector, inst, [](int8_t vs2, uint8_t vs1) -> int8_t {
            return VmulHighSUHelper<int8_t>(vs2, vs1);
          });
    case 2:
      return RiscVBinaryVectorOp<int16_t, int16_t, uint16_t>(
          rv_vector, inst, [](int16_t vs2, uint16_t vs1) -> int16_t {
            return VmulHighSUHelper<int16_t>(vs2, vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<int32_t, int32_t, uint32_t>(
          rv_vector, inst, [](int32_t vs2, uint32_t vs1) -> int32_t {
            return VmulHighSUHelper<int32_t>(vs2, vs1);
          });
    case 8:
      return RiscVBinaryVectorOp<int64_t, int64_t, uint64_t>(
          rv_vector, inst, [](int64_t vs2, uint64_t vs1) -> int64_t {
            return VmulHighSUHelper<int64_t>(vs2, vs1);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Signed multiply, return high half.
void Vmulh(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<int8_t, int8_t, int8_t>(
          rv_vector, inst, [](int8_t vs2, int8_t vs1) -> int8_t {
            return VmulHighHelper(vs2, vs1);
          });
    case 2:
      return RiscVBinaryVectorOp<int16_t, int16_t, int16_t>(
          rv_vector, inst, [](int16_t vs2, int16_t vs1) -> int16_t {
            return VmulHighHelper(vs2, vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<int32_t, int32_t, int32_t>(
          rv_vector, inst, [](int32_t vs2, int32_t vs1) -> int32_t {
            return VmulHighHelper(vs2, vs1);
          });
    case 8:
      return RiscVBinaryVectorOp<int64_t, int64_t, int64_t>(
          rv_vector, inst, [](int64_t vs2, int64_t vs1) -> int64_t {
            return VmulHighHelper(vs2, vs1);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Multiply-add.
void Vmadd(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVTernaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2, uint8_t vs1, uint8_t vd) -> uint8_t {
            uint8_t prod = vs1 * vd;
            return prod + vs2;
          });
    case 2:
      return RiscVTernaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2, uint16_t vs1, uint16_t vd) -> uint16_t {
            uint32_t vs2_32 = vs2;
            uint32_t vs1_32 = vs1;
            uint32_t vd_32 = vd;
            return static_cast<uint16_t>(vs1_32 * vd_32 + vs2_32);
          });
    case 4:
      return RiscVTernaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2, uint32_t vs1, uint32_t vd) -> uint32_t {
            return vs1 * vd + vs2;
          });
    case 8:
      return RiscVTernaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2, uint64_t vs1, uint64_t vd) -> uint64_t {
            return vs1 * vd + vs2;
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Negated multiply and add.
void Vnmsub(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVTernaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2, uint8_t vs1, uint8_t vd) -> uint8_t {
            return -(vs1 * vd) + vs2;
          });
    case 2:
      return RiscVTernaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2, uint16_t vs1, uint16_t vd) -> uint16_t {
            uint32_t vs2_32 = vs2;
            uint32_t vs1_32 = vs1;
            uint32_t vd_32 = vd;
            return static_cast<uint16_t>(-(vs1_32 * vd_32) + vs2_32);
          });
    case 4:
      return RiscVTernaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2, uint32_t vs1, uint32_t vd) -> uint32_t {
            return -(vs1 * vd) + vs2;
          });
    case 8:
      return RiscVTernaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2, uint64_t vs1, uint64_t vd) -> uint64_t {
            return -(vs1 * vd) + vs2;
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Multiply add overwriting the sum.
void Vmacc(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVTernaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2, uint8_t vs1, uint8_t vd) -> uint8_t {
            return vs1 * vs2 + vd;
          });
    case 2:
      return RiscVTernaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2, uint16_t vs1, uint16_t vd) -> uint16_t {
            uint32_t vs2_32 = vs2;
            uint32_t vs1_32 = vs1;
            uint32_t vd_32 = vd;
            return static_cast<uint16_t>(vs1_32 * vs2_32 + vd_32);
          });
    case 4:
      return RiscVTernaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2, uint32_t vs1, uint32_t vd) -> uint32_t {
            return vs1 * vs2 + vd;
          });
    case 8:
      return RiscVTernaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2, uint64_t vs1, uint64_t vd) -> uint64_t {
            return vs1 * vs2 + vd;
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Negated multiply add, overwriting sum.
void Vnmsac(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVTernaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2, uint8_t vs1, uint8_t vd) -> uint8_t {
            return -(vs1 * vs2) + vd;
          });
    case 2:
      return RiscVTernaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2, uint16_t vs1, uint16_t vd) -> uint16_t {
            uint32_t vs2_32 = vs2;
            uint32_t vs1_32 = vs1;
            uint32_t vd_32 = vd;
            return static_cast<uint16_t>(-(vs1_32 * vs2_32) + vd_32);
          });
    case 4:
      return RiscVTernaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2, uint32_t vs1, uint32_t vd) -> uint32_t {
            return -(vs1 * vs2) + vd;
          });
    case 8:
      return RiscVTernaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2, uint64_t vs1, uint64_t vd) -> uint64_t {
            return -(vs1 * vs2) + vd;
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Widening unsigned add.
void Vwaddu(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint16_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2, uint8_t vs1) -> uint16_t {
            return static_cast<uint16_t>(vs2) + static_cast<uint16_t>(vs1);
          });
    case 2:
      return RiscVBinaryVectorOp<uint32_t, uint16_t, uint16_t>(
          rv_vector, inst, [](uint16_t vs2, uint16_t vs1) -> uint32_t {
            return static_cast<uint32_t>(vs2) + static_cast<uint32_t>(vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<uint64_t, uint32_t, uint32_t>(
          rv_vector, inst, [](uint32_t vs2, uint32_t vs1) -> uint64_t {
            return static_cast<uint64_t>(vs2) + static_cast<uint64_t>(vs1);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Widening unsigned subtract.
void Vwsubu(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint16_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2, uint8_t vs1) -> uint16_t {
            return static_cast<uint16_t>(vs2) - static_cast<uint16_t>(vs1);
          });
    case 2:
      return RiscVBinaryVectorOp<uint32_t, uint16_t, uint16_t>(
          rv_vector, inst, [](uint16_t vs2, uint16_t vs1) -> uint32_t {
            return static_cast<uint32_t>(vs2) - static_cast<uint32_t>(vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<uint64_t, uint32_t, uint32_t>(
          rv_vector, inst, [](uint32_t vs2, uint32_t vs1) -> uint64_t {
            return static_cast<uint64_t>(vs2) - static_cast<uint64_t>(vs1);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Widening signed add.
void Vwadd(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  // LMUL8 cannot be 64.
  if (rv_vector->vector_length_multiplier() > 32) {
    rv_vector->set_vector_exception();
    LOG(ERROR)
        << "Vector length multiplier out of range for widening operation";
    return;
  }
  // The values are first sign extended to the wide signed value, then
  // an unsigned addition is performed, for which overflow is not undefined,
  // as opposed to signed additions.
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint16_t, int8_t, int8_t>(
          rv_vector, inst, [](int8_t vs2, int8_t vs1) -> uint16_t {
            return static_cast<uint16_t>(static_cast<int16_t>(vs2)) +
                   static_cast<uint16_t>(static_cast<int16_t>(vs1));
          });
    case 2:
      return RiscVBinaryVectorOp<uint32_t, int16_t, int16_t>(
          rv_vector, inst, [](int16_t vs2, int16_t vs1) -> uint32_t {
            return static_cast<uint32_t>(static_cast<int32_t>(vs2)) +
                   static_cast<uint32_t>(static_cast<int32_t>(vs1));
          });
    case 4:
      return RiscVBinaryVectorOp<uint64_t, int32_t, int32_t>(
          rv_vector, inst, [](int32_t vs2, int32_t vs1) -> uint64_t {
            return static_cast<uint64_t>(static_cast<int64_t>(vs2)) +
                   static_cast<uint64_t>(static_cast<int64_t>(vs1));
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Widening signed subtract.
void Vwsub(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  // LMUL8 cannot be 64.
  if (rv_vector->vector_length_multiplier() > 32) {
    rv_vector->set_vector_exception();
    LOG(ERROR)
        << "Vector length multiplier out of range for widening operation";
    return;
  }  // The values are first sign extended to the wide signed value, then
  // an unsigned subtraction is performed, for which overflow is not undefined,
  // as opposed to signed subtraction.
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint16_t, int8_t, int8_t>(
          rv_vector, inst, [](int8_t vs2, int8_t vs1) -> uint16_t {
            return static_cast<uint16_t>(static_cast<int16_t>(vs2)) -
                   static_cast<uint16_t>(static_cast<int16_t>(vs1));
          });
    case 2:
      return RiscVBinaryVectorOp<uint32_t, int16_t, int16_t>(
          rv_vector, inst, [](int16_t vs2, int16_t vs1) -> uint32_t {
            return static_cast<uint32_t>(static_cast<int32_t>(vs2)) -
                   static_cast<uint32_t>(static_cast<int32_t>(vs1));
          });
    case 4:
      return RiscVBinaryVectorOp<uint64_t, int32_t, int32_t>(
          rv_vector, inst, [](int32_t vs2, int32_t vs1) -> uint64_t {
            return static_cast<uint64_t>(static_cast<int64_t>(vs2)) -
                   static_cast<uint64_t>(static_cast<int64_t>(vs1));
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Widening unsigned add with wide source.
void Vwadduw(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  // LMUL8 cannot be 64.
  if (rv_vector->vector_length_multiplier() > 32) {
    rv_vector->set_vector_exception();
    LOG(ERROR)
        << "Vector length multiplier out of range for widening operation";
    return;
  }
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint8_t>(
          rv_vector, inst, [](uint16_t vs2, uint8_t vs1) -> uint16_t {
            return vs2 + static_cast<uint16_t>(vs1);
          });
    case 2:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint16_t>(
          rv_vector, inst, [](uint32_t vs2, uint16_t vs1) -> uint32_t {
            return vs2 + static_cast<uint32_t>(vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint32_t>(
          rv_vector, inst, [](uint64_t vs2, uint32_t vs1) -> uint64_t {
            return vs2 + static_cast<uint64_t>(vs1);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Widening unsigned subtract with wide source.
void Vwsubuw(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  // LMUL8 cannot be 64.
  if (rv_vector->vector_length_multiplier() > 32) {
    rv_vector->set_vector_exception();
    LOG(ERROR)
        << "Vector length multiplier out of range for widening operation";
    return;
  }
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint8_t>(
          rv_vector, inst, [](uint16_t vs2, uint8_t vs1) -> uint16_t {
            return vs2 - static_cast<uint16_t>(vs1);
          });
    case 2:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint16_t>(
          rv_vector, inst, [](uint32_t vs2, uint16_t vs1) -> uint32_t {
            return vs2 - static_cast<uint32_t>(vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint32_t>(
          rv_vector, inst, [](uint64_t vs2, uint32_t vs1) -> uint64_t {
            return vs2 - static_cast<uint64_t>(vs1);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Widening signed add with wide source.
void Vwaddw(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  // LMUL8 cannot be 64.
  if (rv_vector->vector_length_multiplier() > 32) {
    rv_vector->set_vector_exception();
    LOG(ERROR)
        << "Vector length multiplier out of range for widening operation";
    return;
  }
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<int16_t, uint16_t, int8_t>(
          rv_vector, inst, [](uint16_t vs2, int8_t vs1) -> uint16_t {
            return vs2 + static_cast<uint16_t>(static_cast<int16_t>(vs1));
          });
    case 2:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, int16_t>(
          rv_vector, inst, [](uint32_t vs2, int16_t vs1) -> uint32_t {
            return vs2 + static_cast<uint32_t>(static_cast<int32_t>(vs1));
          });
    case 4:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, int32_t>(
          rv_vector, inst, [](uint64_t vs2, int32_t vs1) -> uint64_t {
            return vs2 + static_cast<uint64_t>(static_cast<int64_t>(vs1));
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Widening signed subtract with wide source.
void Vwsubw(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  // LMUL8 cannot be 64.
  if (rv_vector->vector_length_multiplier() > 32) {
    rv_vector->set_vector_exception();
    LOG(ERROR)
        << "Vector length multiplier out of range for widening operation";
    return;
  }
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, int8_t>(
          rv_vector, inst, [](uint16_t vs2, int8_t vs1) -> uint16_t {
            return vs2 - static_cast<uint16_t>(static_cast<int16_t>(vs1));
          });
    case 2:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, int16_t>(
          rv_vector, inst, [](uint32_t vs2, int16_t vs1) -> uint32_t {
            return vs2 - static_cast<uint32_t>(static_cast<int32_t>(vs1));
          });
    case 4:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, int32_t>(
          rv_vector, inst, [](uint64_t vs2, int32_t vs1) -> uint64_t {
            return vs2 - static_cast<uint64_t>(static_cast<int64_t>(vs1));
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Widening multiply helper function. Factors out some code.
template <typename T>
inline typename WideType<T>::type VwmulHelper(T vs2, T vs1) {
  using WT = typename WideType<T>::type;
  WT vs2_w = static_cast<WT>(vs2);
  WT vs1_w = static_cast<WT>(vs1);
  WT prod = vs2_w * vs1_w;
  return prod;
}

// Unsigned widening multiply.
void Vwmulu(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  // LMUL8 cannot be 64.
  if (rv_vector->vector_length_multiplier() > 32) {
    rv_vector->set_vector_exception();
    LOG(ERROR)
        << "Vector length multiplier out of range for widening operation";
    return;
  }
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint16_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2, int8_t vs1) -> uint16_t {
            return VwmulHelper<uint8_t>(vs2, vs1);
          });
    case 2:
      return RiscVBinaryVectorOp<uint32_t, uint16_t, uint16_t>(
          rv_vector, inst, [](uint16_t vs2, uint16_t vs1) -> uint32_t {
            return VwmulHelper<uint16_t>(vs2, vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<uint64_t, uint32_t, uint32_t>(
          rv_vector, inst, [](uint32_t vs2, uint32_t vs1) -> uint64_t {
            return VwmulHelper<uint32_t>(vs2, vs1);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Widening signed-unsigned multiply helper function.
template <typename T>
inline typename WideType<typename std::make_signed<T>::type>::type
VwmulSuHelper(typename std::make_signed<T>::type vs2,
              typename std::make_unsigned<T>::type vs1) {
  using WST = typename WideType<typename std::make_signed<T>::type>::type;
  using WT = typename WideType<typename std::make_unsigned<T>::type>::type;
  WST vs2_w = static_cast<WST>(vs2);
  WT vs1_w = static_cast<WT>(vs1);
  WST prod = vs2_w * vs1_w;
  return prod;
}

// Widening multiply signed-unsigned.
void Vwmulsu(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  // LMUL8 cannot be 64.
  if (rv_vector->vector_length_multiplier() > 32) {
    rv_vector->set_vector_exception();
    LOG(ERROR)
        << "Vector length multiplier out of range for widening operation";
    return;
  }
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<int16_t, int8_t, uint8_t>(
          rv_vector, inst, [](int8_t vs2, int8_t vs1) -> int16_t {
            return VwmulSuHelper<int8_t>(vs2, vs1);
          });
    case 2:
      return RiscVBinaryVectorOp<int32_t, int16_t, uint16_t>(
          rv_vector, inst, [](int16_t vs2, int16_t vs1) -> int32_t {
            return VwmulSuHelper<int16_t>(vs2, vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<int64_t, int32_t, uint32_t>(
          rv_vector, inst, [](int32_t vs2, int32_t vs1) -> int64_t {
            return VwmulSuHelper<int32_t>(vs2, vs1);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Widening signed multiply.
void Vwmul(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  // LMUL8 cannot be 64.
  if (rv_vector->vector_length_multiplier() > 32) {
    rv_vector->set_vector_exception();
    LOG(ERROR)
        << "Vector length multiplier out of range for widening operation";
    return;
  }
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<int16_t, int8_t, int8_t>(
          rv_vector, inst, [](int8_t vs2, int8_t vs1) -> int16_t {
            return VwmulHelper<int8_t>(vs2, vs1);
          });
    case 2:
      return RiscVBinaryVectorOp<int32_t, int16_t, int16_t>(
          rv_vector, inst, [](int16_t vs2, int16_t vs1) -> int32_t {
            return VwmulHelper<int16_t>(vs2, vs1);
          });
    case 4:
      return RiscVBinaryVectorOp<int64_t, int32_t, int32_t>(
          rv_vector, inst, [](int32_t vs2, int32_t vs1) -> int64_t {
            return VwmulHelper<int32_t>(vs2, vs1);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Widening multiply accumulate helper function.
template <typename Vd, typename Vs2, typename Vs1>
Vd VwmaccHelper(Vs2 vs2, Vs1 vs1, Vd vd) {
  Vd vs2_w = static_cast<Vd>(vs2);
  Vd vs1_w = static_cast<Vd>(vs1);
  Vd prod = vs2_w * vs1_w;
  using UVd = typename std::make_unsigned<Vd>::type;
  Vd res = absl::bit_cast<UVd>(prod) + absl::bit_cast<UVd>(vd);
  return res;
}

// Unsigned widening multiply and add.
void Vwmaccu(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  // LMUL8 cannot be 64.
  if (rv_vector->vector_length_multiplier() > 32) {
    rv_vector->set_vector_exception();
    LOG(ERROR)
        << "Vector length multiplier out of range for widening operation";
    return;
  }
  switch (sew) {
    case 1:
      return RiscVTernaryVectorOp<uint16_t, uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t vs2, uint8_t vs1, uint16_t vd) -> uint16_t {
            return VwmaccHelper(vs2, vs1, vd);
          });
    case 2:
      return RiscVTernaryVectorOp<uint32_t, uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2, uint16_t vs1, uint32_t vd) -> uint32_t {
            return VwmaccHelper(vs2, vs1, vd);
          });
    case 4:
      return RiscVTernaryVectorOp<uint64_t, uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2, uint32_t vs1, uint64_t vd) -> uint64_t {
            return VwmaccHelper(vs2, vs1, vd);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Widening signed multiply and add.
void Vwmacc(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  // LMUL8 cannot be 64.
  if (rv_vector->vector_length_multiplier() > 32) {
    rv_vector->set_vector_exception();
    LOG(ERROR)
        << "Vector length multiplier out of range for widening operation";
    return;
  }
  switch (sew) {
    case 1:
      return RiscVTernaryVectorOp<int16_t, int8_t, int8_t>(
          rv_vector, inst, [](int8_t vs2, int8_t vs1, int16_t vd) -> int16_t {
            return VwmaccHelper(vs2, vs1, vd);
          });
    case 2:
      return RiscVTernaryVectorOp<int32_t, int16_t, int16_t>(
          rv_vector, inst, [](int16_t vs2, int16_t vs1, int32_t vd) -> int32_t {
            return VwmaccHelper(vs2, vs1, vd);
          });
    case 4:
      return RiscVTernaryVectorOp<int64_t, int32_t, int32_t>(
          rv_vector, inst, [](int32_t vs2, int32_t vs1, int64_t vd) -> int64_t {
            return VwmaccHelper(vs2, vs1, vd);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Widening unsigned-signed multiply and add.
void Vwmaccus(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  // LMUL8 cannot be 64.
  if (rv_vector->vector_length_multiplier() > 32) {
    rv_vector->set_vector_exception();
    LOG(ERROR)
        << "Vector length multiplier out of range for widening operation";
    return;
  }
  switch (sew) {
    case 1:
      return RiscVTernaryVectorOp<int16_t, int8_t, uint8_t>(
          rv_vector, inst, [](int8_t vs2, uint8_t vs1, int16_t vd) -> int16_t {
            return VwmaccHelper(vs2, vs1, vd);
          });
    case 2:
      return RiscVTernaryVectorOp<int32_t, int16_t, uint16_t>(
          rv_vector, inst,
          [](int16_t vs2, uint16_t vs1, int32_t vd) -> int32_t {
            return VwmaccHelper(vs2, vs1, vd);
          });
    case 4:
      return RiscVTernaryVectorOp<int64_t, int32_t, uint32_t>(
          rv_vector, inst,
          [](int32_t vs2, uint32_t vs1, int64_t vd) -> int64_t {
            return VwmaccHelper(vs2, vs1, vd);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Widening signed-unsigned multiply and add.
void Vwmaccsu(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  // LMUL8 cannot be 64.
  if (rv_vector->vector_length_multiplier() > 32) {
    rv_vector->set_vector_exception();
    LOG(ERROR)
        << "Vector length multiplier out of range for widening operation";
    return;
  }
  switch (sew) {
    case 1:
      return RiscVTernaryVectorOp<int16_t, uint8_t, int8_t>(
          rv_vector, inst, [](uint8_t vs2, int8_t vs1, int16_t vd) -> int16_t {
            return VwmaccHelper(vs2, vs1, vd);
          });
    case 2:
      return RiscVTernaryVectorOp<int32_t, uint16_t, int16_t>(
          rv_vector, inst,
          [](uint16_t vs2, int16_t vs1, int32_t vd) -> int32_t {
            return VwmaccHelper(vs2, vs1, vd);
          });
    case 4:
      return RiscVTernaryVectorOp<int64_t, uint32_t, int32_t>(
          rv_vector, inst,
          [](uint32_t vs2, int32_t vs1, int64_t vd) -> int64_t {
            return VwmaccHelper(vs2, vs1, vd);
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
