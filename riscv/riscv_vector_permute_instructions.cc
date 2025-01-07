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

#include "riscv/riscv_vector_permute_instructions.h"

#include <algorithm>
#include <cstdint>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_state.h"

namespace mpact {
namespace sim {
namespace riscv {

// This helper function handles the vector gather operations.
template <typename Vd, typename Vs2, typename Vs1>
void VrgatherHelper(RiscVVectorState *rv_vector, Instruction *inst) {
  if (rv_vector->vector_exception()) return;
  int num_elements = rv_vector->vector_length();
  int elements_per_vector =
      rv_vector->vector_register_byte_length() / sizeof(Vd);
  // Verify that the lmul is compatible with index size.
  int index_emul =
      rv_vector->vector_length_multiplier() * sizeof(Vs1) / sizeof(Vd);
  if (index_emul > 64) {
    rv_vector->set_vector_exception();
    return;
  }
  int max_regs = std::max(
      1, (num_elements + elements_per_vector - 1) / elements_per_vector);
  auto *dest_op =
      static_cast<RV32VectorDestinationOperand *>(inst->Destination(0));
  // Verify that there are enough registers in the destination operand.
  if (dest_op->size() < max_regs) {
    rv_vector->set_vector_exception();
    LOG(ERROR) << absl::StrCat(
        "Vector destination '", dest_op->AsString(), "' has fewer registers (",
        dest_op->size(), ") than required by the operation (", max_regs, ")");
    return;
  }
  // Get the vector mask.
  auto *mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(2));
  auto mask_span = mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  // Get the vector start element index and compute the where to start
  // the operation.
  int vector_index = rv_vector->vstart();
  int start_reg = vector_index / elements_per_vector;
  int item_index = vector_index % elements_per_vector;
  // Determine if it's vector-vector or vector-scalar.
  bool vector_scalar = inst->Source(1)->shape()[0] == 1;
  auto src0_op = static_cast<RV32VectorSourceOperand *>(inst->Source(0));
  int max_index = src0_op->size() * elements_per_vector;
  // Iterate over the number of registers to write.
  for (int reg = start_reg; (reg < max_regs) && (vector_index < num_elements);
       reg++) {
    // Allocate data buffer for the new register data.
    auto *dest_db = dest_op->CopyDataBuffer(reg);
    auto dest_span = dest_db->Get<Vd>();
    // Write data into register subject to masking.
    int element_count = std::min(elements_per_vector, num_elements);
    for (int i = item_index;
         (i < element_count) && (vector_index < num_elements); i++) {
      // Get the mask value.
      int mask_index = i >> 3;
      int mask_offset = i & 0b111;
      bool mask_value = ((mask_span[mask_index] >> mask_offset) & 0b1) != 0;
      if (mask_value) {
        // Compute result.
        RV32Register::ValueType vs1;
        if (vector_scalar) {
          vs1 = generic::GetInstructionSource<RV32Register::ValueType>(inst, 1,
                                                                       0);
        } else {
          vs1 = generic::GetInstructionSource<Vs1>(inst, 1, vector_index);
        }
        Vs2 vs2 = 0;
        if (vs1 < max_index) {
          vs2 = generic::GetInstructionSource<Vs2>(inst, 0, vs1);
        }
        dest_span[i] = vs2;
      }
      vector_index++;
    }
    // Submit the destination db .
    dest_db->Submit();
    item_index = 0;
  }
  rv_vector->clear_vstart();
}

// Vector register gather.
void Vrgather(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return VrgatherHelper<uint8_t, uint8_t, uint8_t>(rv_vector, inst);
    case 2:
      return VrgatherHelper<uint16_t, uint16_t, uint16_t>(rv_vector, inst);
    case 4:
      return VrgatherHelper<uint32_t, uint32_t, uint32_t>(rv_vector, inst);
    case 8:
      return VrgatherHelper<uint64_t, uint64_t, uint64_t>(rv_vector, inst);
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Vector register gather with 16 bit indices.
void Vrgatherei16(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return VrgatherHelper<uint8_t, uint8_t, uint16_t>(rv_vector, inst);
    case 2:
      return VrgatherHelper<uint16_t, uint16_t, uint16_t>(rv_vector, inst);
    case 4:
      return VrgatherHelper<uint32_t, uint32_t, uint16_t>(rv_vector, inst);
    case 8:
      return VrgatherHelper<uint64_t, uint64_t, uint16_t>(rv_vector, inst);
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// This helper function handles the vector slide up/down instructions.
template <typename Vd>
void VSlideHelper(RiscVVectorState *rv_vector, Instruction *inst, int offset) {
  if (rv_vector->vector_exception()) return;
  int num_elements = rv_vector->vector_length();
  int elements_per_vector =
      rv_vector->vector_register_byte_length() / sizeof(Vd);
  int max_regs = std::max(
      1, (num_elements + elements_per_vector - 1) / elements_per_vector);
  auto *dest_op =
      static_cast<RV32VectorDestinationOperand *>(inst->Destination(0));
  // Verify that there are enough registers in the destination operand.
  if (dest_op->size() < max_regs) {
    rv_vector->set_vector_exception();
    LOG(ERROR) << absl::StrCat(
        "Vector destination '", dest_op->AsString(), "' has fewer registers (",
        dest_op->size(), ") than required by the operation (", max_regs, ")");
    return;
  }
  // Get the vector mask.
  auto *mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(2));
  auto mask_span = mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  // Get the vector start element index and compute the where to start
  // the operation.
  int vector_index = rv_vector->vstart();
  int start_reg = vector_index / elements_per_vector;
  int item_index = vector_index % elements_per_vector;
  // Iterate over the number of registers to write.
  for (int reg = start_reg; (reg < max_regs) && (vector_index < num_elements);
       reg++) {
    // Allocate data buffer for the new register data.
    auto *dest_db = dest_op->CopyDataBuffer(reg);
    auto dest_span = dest_db->Get<Vd>();
    // Write data into register subject to masking.
    int element_count = std::min(elements_per_vector, num_elements);
    for (int i = item_index;
         (i < element_count) && (vector_index < num_elements); i++) {
      // Get the mask value.
      int mask_index = i >> 3;
      int mask_offset = i & 0b111;
      bool mask_value = ((mask_span[mask_index] >> mask_offset) & 0b1);
      int src_index = vector_index - offset;
      if ((src_index >= 0) && (mask_value)) {
        // Compute result.
        Vd src_value = 0;
        if (src_index < rv_vector->max_vector_length()) {
          src_value = generic::GetInstructionSource<Vd>(inst, 0, src_index);
        }
        dest_span[i] = src_value;
      }
      vector_index++;
    }
    // Submit the destination db .
    dest_db->Submit();
    item_index = 0;
  }
  rv_vector->clear_vstart();
}

void Vslideup(Instruction *inst) {
  using ValueType = RV32Register::ValueType;
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  auto offset = generic::GetInstructionSource<ValueType>(inst, 1, 0);
  int int_offset = static_cast<int>(offset);
  if (offset > rv_vector->max_vector_length()) return;
  // Slide up amount is positive.
  switch (sew) {
    case 1:
      return VSlideHelper<uint8_t>(rv_vector, inst, int_offset);
    case 2:
      return VSlideHelper<uint16_t>(rv_vector, inst, int_offset);
    case 4:
      return VSlideHelper<uint32_t>(rv_vector, inst, int_offset);
    case 8:
      return VSlideHelper<uint64_t>(rv_vector, inst, int_offset);
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

void Vslidedown(Instruction *inst) {
  using ValueType = RV32Register::ValueType;
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  auto offset = generic::GetInstructionSource<ValueType>(inst, 1, 0);
  // Slide down amount is negative.
  int int_offset = -static_cast<int>(offset);
  switch (sew) {
    case 1:
      return VSlideHelper<uint8_t>(rv_vector, inst, int_offset);
    case 2:
      return VSlideHelper<uint16_t>(rv_vector, inst, int_offset);
    case 4:
      return VSlideHelper<uint32_t>(rv_vector, inst, int_offset);
    case 8:
      return VSlideHelper<uint64_t>(rv_vector, inst, int_offset);
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// This helper function handles the vector slide up/down 1 instructions.
template <typename Vd>
void VSlide1Helper(RiscVVectorState *rv_vector, Instruction *inst, int offset) {
  if (rv_vector->vector_exception()) return;
  int num_elements = rv_vector->vector_length();
  int elements_per_vector =
      rv_vector->vector_register_byte_length() / sizeof(Vd);
  int max_regs = std::max(
      1, (num_elements + elements_per_vector - 1) / elements_per_vector);
  auto *dest_op =
      static_cast<RV32VectorDestinationOperand *>(inst->Destination(0));
  // Verify that there are enough registers in the destination operand.
  if (dest_op->size() < max_regs) {
    rv_vector->set_vector_exception();
    LOG(ERROR) << absl::StrCat(
        "Vector destination '", dest_op->AsString(), "' has fewer registers (",
        dest_op->size(), ") than required by the operation (", max_regs, ")");
    return;
  }
  // Get the vector mask.
  auto *mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(2));
  auto mask_span = mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  // Get the vector start element index and compute the where to start
  // the operation.
  int vector_index = rv_vector->vstart();
  int start_reg = vector_index / elements_per_vector;
  int item_index = vector_index % elements_per_vector;
  auto slide_value = generic::GetInstructionSource<Vd>(inst, 1, 0);
  // Iterate over the number of registers to write.
  for (int reg = start_reg; (reg < max_regs) && (vector_index < num_elements);
       reg++) {
    // Allocate data buffer for the new register data.
    auto *dest_db = dest_op->CopyDataBuffer(reg);
    auto dest_span = dest_db->Get<Vd>();
    // Write data into register subject to masking.
    int element_count = std::min(elements_per_vector, num_elements);
    for (int i = item_index;
         (i < element_count) && (vector_index < num_elements); i++) {
      // Get the mask value.
      int mask_index = i >> 3;
      int mask_offset = i & 0b111;
      bool mask_value = ((mask_span[mask_index] >> mask_offset) & 0b1) != 0;
      if (mask_value) {
        // Compute result.
        Vd src_value = slide_value;
        int src_index = vector_index - offset;
        if ((src_index > 0) && (src_index < rv_vector->max_vector_length())) {
          src_value = generic::GetInstructionSource<Vd>(inst, 0, src_index);
        }
        dest_span[i] = src_value;
      }
      vector_index++;
    }
    // Submit the destination db .
    dest_db->Submit();
    item_index = 0;
  }
  rv_vector->clear_vstart();
}

void Vslide1up(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return VSlide1Helper<uint8_t>(rv_vector, inst, 1);
    case 2:
      return VSlide1Helper<uint16_t>(rv_vector, inst, 1);
    case 4:
      return VSlide1Helper<uint32_t>(rv_vector, inst, 1);
    case 8:
      return VSlide1Helper<uint64_t>(rv_vector, inst, 1);
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

void Vslide1down(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return VSlide1Helper<uint8_t>(rv_vector, inst, -1);
    case 2:
      return VSlide1Helper<uint16_t>(rv_vector, inst, -1);
    case 4:
      return VSlide1Helper<uint32_t>(rv_vector, inst, -1);
    case 8:
      return VSlide1Helper<uint64_t>(rv_vector, inst, -1);
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

void Vfslide1up(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return VSlide1Helper<uint32_t>(rv_vector, inst, 1);
    case 8:
      return VSlide1Helper<uint64_t>(rv_vector, inst, 1);
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

void Vfslide1down(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return VSlide1Helper<uint32_t>(rv_vector, inst, -1);
    case 8:
      return VSlide1Helper<uint64_t>(rv_vector, inst, -1);
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

template <typename Vd>
void VCompressHelper(RiscVVectorState *rv_vector, Instruction *inst) {
  if (rv_vector->vector_exception()) return;
  int num_elements = rv_vector->vector_length();
  int elements_per_vector =
      rv_vector->vector_register_byte_length() / sizeof(Vd);
  int max_regs = std::max(
      1, (num_elements + elements_per_vector - 1) / elements_per_vector);
  auto *dest_op =
      static_cast<RV32VectorDestinationOperand *>(inst->Destination(0));
  // Verify that there are enough registers in the destination operand.
  if (dest_op->size() < max_regs) {
    rv_vector->set_vector_exception();
    LOG(ERROR) << absl::StrCat(
        "Vector destination '", dest_op->AsString(), "' has fewer registers (",
        dest_op->size(), ") than required by the operation (", max_regs, ")");
    return;
  }
  // Get the vector mask.
  auto *mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(1));
  auto mask_span = mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  // Get the vector start element index and compute the where to start
  // the operation.
  int vector_index = rv_vector->vstart();
  int dest_index = 0;
  int prev_reg = -1;
  absl::Span<Vd> dest_span;
  generic::DataBuffer *dest_db = nullptr;
  // Iterate over the input elements.
  for (int i = vector_index; i < num_elements; i++) {
    // Get mask value.
    int mask_index = i >> 3;
    int mask_offset = i & 0b111;
    bool mask_value = (mask_span[mask_index] >> mask_offset) & 0b1;
    if (mask_value) {
      // Compute destination register.
      int reg = dest_index / elements_per_vector;
      if (prev_reg != reg) {
        // Submit previous data buffer if needed.
        if (dest_db != nullptr) dest_db->Submit();
        // Allocate a data buffer.
        dest_db = dest_op->CopyDataBuffer(reg);
        dest_span = dest_db->Get<Vd>();
        prev_reg = reg;
      }
      // Copy the source value to the dest_index.
      Vd src_value = generic::GetInstructionSource<Vd>(inst, 0, i);
      dest_span[dest_index % elements_per_vector] = src_value;
      ++dest_index;
    }
  }
  if (dest_db != nullptr) dest_db->Submit();
  rv_vector->clear_vstart();
}

void Vcompress(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return VCompressHelper<uint8_t>(rv_vector, inst);
    case 2:
      return VCompressHelper<uint16_t>(rv_vector, inst);
    case 4:
      return VCompressHelper<uint32_t>(rv_vector, inst);
    case 8:
      return VCompressHelper<uint64_t>(rv_vector, inst);
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
