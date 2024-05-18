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

#include "riscv/riscv_vector_unary_instructions.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <type_traits>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_instruction_helpers.h"
#include "riscv/riscv_vector_state.h"

namespace mpact {
namespace sim {
namespace riscv {

using SignedXregType =
    ::mpact::sim::generic::SameSignedType<RV32Register::ValueType,
                                          int64_t>::type;

// Move scalar to vector register.
void VmvToScalar(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (rv_vector->vstart()) return;
  if (rv_vector->vector_length() == 0) return;
  int sew = rv_vector->selected_element_width();
  auto *dest_db = inst->Destination(0)->AllocateDataBuffer();
  SignedXregType value;
  switch (sew) {
    case 1:
      value = static_cast<SignedXregType>(
          generic::GetInstructionSource<int8_t>(inst, 0));
      break;
    case 2:
      value = static_cast<SignedXregType>(
          generic::GetInstructionSource<int16_t>(inst, 0));
      break;
    case 4:
      value = static_cast<SignedXregType>(
          generic::GetInstructionSource<int32_t>(inst, 0));
      break;
    case 8:
      value = static_cast<SignedXregType>(
          generic::GetInstructionSource<int64_t>(inst, 0));
      break;
    default:
      LOG(ERROR) << absl::StrCat("Illegal SEW value (", sew, ") for Vmvxs");
      rv_vector->set_vector_exception();
      return;
  }
  dest_db->Set<SignedXregType>(0, value);
  dest_db->Submit();
}

void VmvFromScalar(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (rv_vector->vstart()) return;
  if (rv_vector->vector_length() == 0) return;
  int sew = rv_vector->selected_element_width();
  auto *dest_db = inst->Destination(0)->AllocateDataBuffer();
  std::memset(dest_db->raw_ptr(), 0, dest_db->size<uint8_t>());
  switch (sew) {
    case 1:
      dest_db->Set<int8_t>(0, generic::GetInstructionSource<int8_t>(inst, 0));
      break;
    case 2:
      dest_db->Set<int16_t>(0, generic::GetInstructionSource<int16_t>(inst, 0));
      break;
    case 4:
      dest_db->Set<int32_t>(0, generic::GetInstructionSource<int32_t>(inst, 0));
      break;
    case 8:
      dest_db->Set<int64_t>(0, generic::GetInstructionSource<int64_t>(inst, 0));
      break;
    default:
      LOG(ERROR) << absl::StrCat("Illegal SEW value (", sew, ") for Vmvxs");
      rv_vector->set_vector_exception();
      return;
  }
  dest_db->Submit();
}

// Population count of vector mask register. The value is written to a scalar
// register.
void Vcpop(Instruction *inst) {
  auto *rv_state = static_cast<RiscVState *>(inst->state());
  auto *rv_vector = rv_state->rv_vector();
  if (rv_vector->vstart()) {
    rv_vector->set_vector_exception();
    return;
  }
  int vlen = rv_vector->vector_length();
  auto src_op = static_cast<RV32VectorSourceOperand *>(inst->Source(0));
  auto src_span = src_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  auto mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(1));
  auto mask_span = mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  auto *dest_db = inst->Destination(0)->AllocateDataBuffer();
  uint64_t count = 0;
  for (int i = 0; i < vlen; i++) {
    int index = i >> 3;
    int offset = i & 0b111;
    int mask_value = (mask_span[index] >> offset);
    int src_value = (src_span[index] >> offset);
    count += mask_value & src_value & 0b1;
  }
  if (rv_state->xlen() == RiscVXlen::RV32) {
    dest_db->Set<RV32Register::ValueType>(0, count);
  } else if (rv_state->xlen() == RiscVXlen::RV64) {
    dest_db->Set<RV64Register::ValueType>(0, count);
  } else {
    LOG(ERROR) << absl::StreamFormat("Illegal XLEN value (%d) for Vcpop",
                                     rv_state->xlen());

    rv_vector->set_vector_exception();
    return;
  }

  dest_db->Submit();
}

// Find first set of vector mask register. The value is written to a scalar
// register.
void Vfirst(Instruction *inst) {
  auto *rv_state = static_cast<RiscVState *>(inst->state());
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (rv_vector->vstart()) {
    rv_vector->set_vector_exception();
    return;
  }
  auto src_op = static_cast<RV32VectorSourceOperand *>(inst->Source(0));
  auto src_span = src_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  auto mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(1));
  auto mask_span = mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  auto *dest_db = inst->Destination(0)->AllocateDataBuffer();
  // Initialize the element index to -1.
  uint64_t element_index = -1LL;
  int vlen = rv_vector->vector_length();
  for (int i = 0; i < vlen; i++) {
    int index = i >> 3;
    int offset = i & 0b111;
    int mask_value = (mask_span[index] >> offset);
    int src_value = (src_span[index] >> offset);
    if (mask_value & src_value & 0b1) {
      element_index = i;
      break;
    }
  }
  if (rv_state->xlen() == RiscVXlen::RV32) {
    dest_db->Set<RV32Register::ValueType>(0, element_index);
  } else if (rv_state->xlen() == RiscVXlen::RV64) {
    dest_db->Set<RV64Register::ValueType>(0, element_index);
  } else {
    LOG(ERROR) << absl::StreamFormat("Illegal XLEN value (%d) for Vcpop",
                                     rv_state->xlen());

    rv_vector->set_vector_exception();
    return;
  }
  dest_db->Submit();
}

// Vector integer sign and zero extension instructions.
void Vzext2(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 2:
      return RiscVUnaryVectorOp<uint16_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t vs2) -> uint16_t { return static_cast<uint16_t>(vs2); });
    case 4:
      return RiscVUnaryVectorOp<uint32_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2) -> uint32_t { return static_cast<uint32_t>(vs2); });
    case 8:
      return RiscVUnaryVectorOp<uint64_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2) -> uint64_t { return static_cast<uint64_t>(vs2); });
    default:
      LOG(ERROR) << absl::StrCat("Illegal SEW value (", sew, ") for Vzext2");
      rv_vector->set_vector_exception();
      return;
  }
}

void Vsext2(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 2:
      return RiscVUnaryVectorOp<int16_t, int8_t>(
          rv_vector, inst,
          [](int8_t vs2) -> int16_t { return static_cast<int16_t>(vs2); });
    case 4:
      return RiscVUnaryVectorOp<uint32_t, uint16_t>(
          rv_vector, inst,
          [](int16_t vs2) -> int32_t { return static_cast<int32_t>(vs2); });
    case 8:
      return RiscVUnaryVectorOp<int64_t, int32_t>(
          rv_vector, inst,
          [](int32_t vs2) -> int64_t { return static_cast<int64_t>(vs2); });
    default:
      LOG(ERROR) << absl::StrCat("Illegal SEW value (", sew, ") for Vsext2");
      rv_vector->set_vector_exception();
      return;
  }
}

void Vzext4(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVUnaryVectorOp<uint32_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t vs2) -> uint32_t { return static_cast<uint32_t>(vs2); });
    case 8:
      return RiscVUnaryVectorOp<uint64_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2) -> uint64_t { return static_cast<uint64_t>(vs2); });
    default:
      LOG(ERROR) << absl::StrCat("Illegal SEW value (", sew, ") for Vzext4");
      rv_vector->set_vector_exception();
      return;
  }
}

void Vsext4(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVUnaryVectorOp<uint32_t, uint8_t>(
          rv_vector, inst,
          [](int8_t vs2) -> int32_t { return static_cast<int32_t>(vs2); });
    case 8:
      return RiscVUnaryVectorOp<int64_t, int16_t>(
          rv_vector, inst,
          [](int16_t vs2) -> int64_t { return static_cast<int64_t>(vs2); });
    default:
      LOG(ERROR) << absl::StrCat("Illegal SEW value (", sew, ") for Vzext4");
      rv_vector->set_vector_exception();
      return;
  }
}

void Vzext8(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 8:
      return RiscVUnaryVectorOp<uint64_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t vs2) -> uint64_t { return static_cast<uint64_t>(vs2); });
    default:
      LOG(ERROR) << absl::StrCat("Illegal SEW value (", sew, ") for Vzext8");
      rv_vector->set_vector_exception();
      return;
  }
}

void Vsext8(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 8:
      return RiscVUnaryVectorOp<int64_t, int8_t>(
          rv_vector, inst,
          [](int8_t vs2) -> int64_t { return static_cast<int64_t>(vs2); });
    default:
      LOG(ERROR) << absl::StrCat("Illegal SEW value (", sew, ") for Vzext8");
      rv_vector->set_vector_exception();
      return;
  }
}

// Vector mask set-before-first mask bit.
void Vmsbf(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (rv_vector->vstart()) {
    rv_vector->set_vector_exception();
    return;
  }
  int vlen = rv_vector->vector_length();
  auto src_op = static_cast<RV32VectorSourceOperand *>(inst->Source(0));
  auto src_span = src_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  auto mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(1));
  auto mask_span = mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  auto dest_op =
      static_cast<RV32VectorDestinationOperand *>(inst->Destination(0));
  auto *dest_db = dest_op->CopyDataBuffer(0);
  auto dest_span = dest_db->Get<uint8_t>();
  bool before_first = true;
  int last = 0;
  // Set the bits before the first active 1.
  for (int i = 0; i < vlen; i++) {
    last = i;
    int index = i >> 3;
    int offset = i & 0b111;
    int mask_value = (mask_span[index] >> offset) & 0b1;
    int src_value = (src_span[index] >> offset) & 0b1;
    if (mask_value) {
      before_first = before_first && (src_value == 0);
      if (!before_first) break;

      dest_span[index] |= 1 << offset;
    }
  }
  // Clear the remaining bits.
  for (int i = last; !before_first && (i < vlen); i++) {
    int index = i >> 3;
    int offset = i & 0b111;
    dest_span[index] &= ~(1 << offset);
  }
  dest_db->Submit();
  rv_vector->clear_vstart();
}

// Vector mask set-including-first mask bit.
void Vmsif(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (rv_vector->vstart()) {
    rv_vector->set_vector_exception();
    return;
  }
  int vlen = rv_vector->vector_length();
  auto src_op = static_cast<RV32VectorSourceOperand *>(inst->Source(0));
  auto src_span = src_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  auto mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(1));
  auto mask_span = mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  auto dest_op =
      static_cast<RV32VectorDestinationOperand *>(inst->Destination(0));
  auto *dest_db = dest_op->CopyDataBuffer(0);
  auto dest_span = dest_db->Get<uint8_t>();
  uint8_t value = 1;
  for (int i = 0; i < vlen; i++) {
    int index = i >> 3;
    int offset = i & 0b111;
    int mask_value = (mask_span[index] >> offset) & 0b1;
    int src_value = (src_span[index] >> offset) & 0b1;
    if (mask_value) {
      if (value) {
        dest_span[index] |= 1 << offset;
      } else {
        dest_span[index] &= ~(1 << offset);
      }
      if (src_value) {
        value = 0;
      }
    }
  }
  dest_db->Submit();
  rv_vector->clear_vstart();
}

// Vector maks set-only-first mask bit.
void Vmsof(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (rv_vector->vstart()) {
    rv_vector->set_vector_exception();
    return;
  }
  int vlen = rv_vector->vector_length();
  auto src_op = static_cast<RV32VectorSourceOperand *>(inst->Source(0));
  auto src_span = src_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  auto mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(1));
  auto mask_span = mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  auto dest_op =
      static_cast<RV32VectorDestinationOperand *>(inst->Destination(0));
  auto *dest_db = dest_op->CopyDataBuffer(0);
  auto dest_span = dest_db->Get<uint8_t>();
  bool first = true;
  for (int i = 0; i < vlen; i++) {
    int index = i >> 3;
    int offset = i & 0b111;
    int mask_value = (mask_span[index] >> offset) & 0b1;
    int src_value = (src_span[index] >> offset) & 0b1;
    if (mask_value) {
      if (first & src_value) {
        dest_span[index] |= (1 << offset);
        first = false;
      } else {
        dest_span[index] &= ~(1 << offset);
      }
    }
  }
  dest_db->Submit();
  rv_vector->clear_vstart();
}

// Vector iota. This instruction reads a source vector mask register and
// writes to each element of the destination vector register group the sum
// of all bits of elements in the mask register whose index is less than the
// element. This is subject to masking.
void Viota(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  int count = 0;
  switch (sew) {
    case 1:
      return RiscVMaskNullaryVectorOp<uint8_t>(
          rv_vector, inst, [&count](bool mask) -> uint8_t {
            return mask ? static_cast<uint8_t>(count++)
                        : static_cast<uint8_t>(count);
          });
    case 2:
      return RiscVMaskNullaryVectorOp<uint16_t>(
          rv_vector, inst, [&count](bool mask) -> uint16_t {
            return mask ? static_cast<uint16_t>(count++)
                        : static_cast<uint16_t>(count);
          });
    case 4:
      return RiscVMaskNullaryVectorOp<uint32_t>(
          rv_vector, inst, [&count](bool mask) -> uint32_t {
            return mask ? static_cast<uint32_t>(count++)
                        : static_cast<uint32_t>(count);
          });
    case 8:
      return RiscVMaskNullaryVectorOp<uint64_t>(
          rv_vector, inst, [&count](bool mask) -> uint64_t {
            return mask ? static_cast<uint64_t>(count++)
                        : static_cast<uint64_t>(count);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Writes the index of each active (mask true) element to the destination
// vector elements.
void Vid(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  int index = 0;
  switch (sew) {
    case 1:
      return RiscVMaskNullaryVectorOp<uint8_t>(
          rv_vector, inst, [&index](bool mask) -> uint8_t {
            uint64_t ret = index++;
            return static_cast<uint8_t>(ret);
          });
    case 2:
      return RiscVMaskNullaryVectorOp<uint16_t>(
          rv_vector, inst, [&index](bool mask) -> uint16_t {
            uint64_t ret = index++;
            return static_cast<uint16_t>(ret);
          });
    case 4:
      return RiscVMaskNullaryVectorOp<uint32_t>(
          rv_vector, inst, [&index](bool mask) -> uint32_t {
            uint64_t ret = index++;
            return static_cast<uint32_t>(ret);
          });
    case 8:
      return RiscVMaskNullaryVectorOp<uint64_t>(
          rv_vector, inst, [&index](bool mask) -> uint64_t {
            uint64_t ret = index++;
            return static_cast<uint64_t>(ret);
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
