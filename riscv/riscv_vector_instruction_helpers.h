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

#ifndef MPACT_RISCV_RISCV_RISCV_VECTOR_INSTRUCTION_HELPERS_H_
#define MPACT_RISCV_RISCV_RISCV_VECTOR_INSTRUCTION_HELPERS_H_

#include <algorithm>
#include <limits>
#include <optional>
#include <type_traits>

#include "absl/log/log.h"
#include "mpact/sim/generic/instruction.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_state.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::GetInstructionSource;

// This helper function handles the case of instructions that target a vector
// mask and uses the mask value in the instruction, such as carry generation
// from add with carry.
template <typename Vs2, typename Vs1>
void RiscVMaskBinaryVectorMaskOp(RiscVVectorState *rv_vector,
                                 const Instruction *inst,
                                 std::function<bool(Vs2, Vs1, bool)> op) {
  if (rv_vector->vector_exception()) return;
  auto *dest_op =
      static_cast<RV32VectorDestinationOperand *>(inst->Destination(0));
  // Get the vector start element index and compute where to start
  // the operation.
  int num_elements = rv_vector->vector_length();
  int vector_index = rv_vector->vstart();
  // Allocate data buffer for the new register data.
  auto *dest_db = dest_op->CopyDataBuffer();
  auto dest_span = dest_db->Get<uint8_t>();
  // Determine if it's vector-vector or vector-scalar.
  bool vector_scalar = inst->Source(1)->shape()[0] == 1;
  // Get the vector mask.
  auto *mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(2));
  auto mask_span = mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  for (int i = vector_index; i < num_elements; i++) {
    int mask_index = i >> 3;
    int mask_offset = i & 0b111;
    bool mask_value = ((mask_span[mask_index] >> mask_offset) & 0b1) != 0;
    Vs2 vs2 = GetInstructionSource<Vs2>(inst, 0, i);
    Vs1 vs1 = GetInstructionSource<Vs1>(inst, 1, vector_scalar ? 0 : i);
    dest_span[mask_index] |= (op(vs2, vs1, mask_value) << mask_offset);
  }
  // Submit the destination db .
  dest_db->Submit();
  rv_vector->clear_vstart();
}

// This helper function handles the case of vector mask
// operations.
template <typename Vs2, typename Vs1>
void RiscVBinaryVectorMaskOp(RiscVVectorState *rv_vector,
                             const Instruction *inst,
                             std::function<bool(Vs2, Vs1)> op) {
  RiscVMaskBinaryVectorMaskOp<Vs2, Vs1>(
      rv_vector, inst, [op](Vs2 vs2, Vs1 vs1, bool mask_value) -> bool {
        if (mask_value) {
          return op(vs2, vs1);
        }
        return false;
      });
}

// This helper function handles the case of nullary vector
// operations. It implements all the checking necessary for both widening and
// narrowing operations.
template <typename Vd>
void RiscVMaskNullaryVectorOp(RiscVVectorState *rv_vector,
                              const Instruction *inst,
                              std::function<std::optional<Vd>(bool)> op) {
  if (rv_vector->vector_exception()) return;
  int num_elements = rv_vector->vector_length();
  int elements_per_vector =
      rv_vector->vector_register_byte_length() / sizeof(Vd);
  int max_regs = (num_elements + elements_per_vector - 1) / elements_per_vector;
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
  auto *mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(0));
  auto mask_span = mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  // Get the vector start element index and compute where to start
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
      int mask_index = vector_index >> 3;
      int mask_offset = vector_index & 0b111;
      bool mask_value = ((mask_span[mask_index] >> mask_offset) & 0b1) != 0;
      auto value = op(mask_value);
      if (value.has_value()) {
        dest_span[i] = value.value();
      }
      vector_index++;
    }
    // Submit the destination db .
    dest_db->Submit();
    item_index = 0;
  }
  rv_vector->clear_vstart();
}

// This helper function handles the case of unary vector
// operations. It implements all the checking necessary for both widening and
// narrowing operations.
template <typename Vd, typename Vs2>
void RiscVUnaryVectorOp(RiscVVectorState *rv_vector, const Instruction *inst,
                        std::function<Vd(Vs2)> op) {
  if (rv_vector->vector_exception()) return;
  int num_elements = rv_vector->vector_length();
  int lmul = rv_vector->vector_length_multiplier();
  int sew = rv_vector->selected_element_width();
  int lmul_vd = lmul * sizeof(Vd) / sew;
  int lmul_vs2 = lmul * sizeof(Vs2) / sew;
  if (lmul_vd > 64 || lmul_vd == 0) {
    rv_vector->set_vector_exception();
    LOG(ERROR) << "Illegal lmul value vd (" << lmul_vd << ")";
    return;
  }
  if (lmul_vs2 > 64 || lmul_vs2 == 0) {
    rv_vector->set_vector_exception();
    LOG(ERROR) << "Illegal lmul_value vs2 (" << lmul_vs2 << ")";
    return;
  }
  int elements_per_vector =
      rv_vector->vector_register_byte_length() / sizeof(Vd);
  int max_regs = (num_elements + elements_per_vector - 1) / elements_per_vector;
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
  // Get the vector start element index and compute where to start
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
      int mask_index = vector_index >> 3;
      int mask_offset = vector_index & 0b111;
      bool mask_value = ((mask_span[mask_index] >> mask_offset) & 0b1) != 0;
      if (mask_value) {
        // Compute result.
        Vs2 vs2 = GetInstructionSource<Vs2>(inst, 0, vector_index);
        dest_span[i] = op(vs2);
      }
      vector_index++;
    }
    // Submit the destination db .
    dest_db->Submit();
    item_index = 0;
  }
  rv_vector->clear_vstart();
}

// This helper function handles the case of unary vector operations that set
// fflags. It implements all the checking necessary for both widening and
// narrowing operations.
template <typename Vd, typename Vs2>
void RiscVUnaryVectorOpWithFflags(
    RiscVVectorState *rv_vector, const Instruction *inst,
    std::function<std::tuple<Vd, uint32_t>(Vs2)> op) {
  if (rv_vector->vector_exception()) return;
  int num_elements = rv_vector->vector_length();
  int lmul = rv_vector->vector_length_multiplier();
  int sew = rv_vector->selected_element_width();
  int lmul_vd = lmul * sizeof(Vd) / sew;
  int lmul_vs2 = lmul * sizeof(Vs2) / sew;
  if (lmul_vd > 64 || lmul_vd == 0) {
    rv_vector->set_vector_exception();
    LOG(ERROR) << "Illegal lmul value vd (" << lmul_vd << ")";
    return;
  }
  if (lmul_vs2 > 64 || lmul_vs2 == 0) {
    rv_vector->set_vector_exception();
    LOG(ERROR) << "Illegal lmul_value vs2 (" << lmul_vs2 << ")";
    return;
  }
  int elements_per_vector =
      rv_vector->vector_register_byte_length() / sizeof(Vd);
  int max_regs = (num_elements + elements_per_vector - 1) / elements_per_vector;
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
  // Get the vector start element index and compute where to start
  // the operation.
  int vector_index = rv_vector->vstart();
  int start_reg = vector_index / elements_per_vector;
  int item_index = vector_index % elements_per_vector;
  // Iterate over the number of registers to write.
  uint32_t fflags = 0;
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
      int mask_index = vector_index >> 3;
      int mask_offset = vector_index & 0b111;
      bool mask_value = ((mask_span[mask_index] >> mask_offset) & 0b1) != 0;
      if (mask_value) {
        // Compute result.
        Vs2 vs2 = GetInstructionSource<Vs2>(inst, 0, vector_index);
        auto [value, flag] = op(vs2);
        dest_span[i] = value;
        fflags |= flag;
      }
      vector_index++;
    }
    // Submit the destination db .
    dest_db->Submit();
    item_index = 0;
  }
  auto *flag_db = inst->Destination(1)->AllocateDataBuffer();
  flag_db->Set<uint32_t>(0, fflags);
  flag_db->Submit();
  rv_vector->clear_vstart();
}

// This helper function handles the case of mask + two source operand vector
// operations. It implements all the checking necessary for both widening and
// narrowing operations.
template <typename Vd, typename Vs2, typename Vs1>
void RiscVMaskBinaryVectorOp(
    RiscVVectorState *rv_vector, const Instruction *inst,
    std::function<std::optional<Vd>(Vs2, Vs1, bool)> op) {
  if (rv_vector->vector_exception()) return;
  int num_elements = rv_vector->vector_length();
  int lmul = rv_vector->vector_length_multiplier();
  int sew = rv_vector->selected_element_width();
  int lmul_vd = lmul * sizeof(Vd) / sew;
  int lmul_vs2 = lmul * sizeof(Vs2) / sew;
  int lmul_vs1 = lmul * sizeof(Vs1) / sew;
  if (lmul_vd > 64 || lmul_vs2 > 64 || lmul_vs1 > 64) {
    rv_vector->set_vector_exception();
    LOG(ERROR) << "Illegal lmul value";
    return;
  }
  if (lmul_vd == 0 || lmul_vs2 == 0 || lmul_vs1 == 0) {
    rv_vector->set_vector_exception();
    LOG(ERROR) << "Illegal lmul_value";
    return;
  }
  int elements_per_vector =
      rv_vector->vector_register_byte_length() / sizeof(Vd);
  int max_regs = (num_elements + elements_per_vector - 1) / elements_per_vector;
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
  // Get the vector start element index and compute where to start
  // the operation.
  int vector_index = rv_vector->vstart();
  int start_reg = vector_index / elements_per_vector;
  int item_index = vector_index % elements_per_vector;
  // Determine if it's vector-vector or vector-scalar.
  bool vector_scalar = inst->Source(1)->shape()[0] == 1;
  // Iterate over the number of registers to write.
  bool exception = false;
  for (int reg = start_reg;
       !exception && (reg < max_regs) && (vector_index < num_elements); reg++) {
    // Allocate data buffer for the new register data.
    auto *dest_db = dest_op->CopyDataBuffer(reg);
    auto dest_span = dest_db->Get<Vd>();
    // Write data into register subject to masking.
    int element_count = std::min(elements_per_vector, num_elements);
    for (int i = item_index;
         (i < element_count) && (vector_index < num_elements); i++) {
      // Get the mask value.
      int mask_index = vector_index >> 3;
      int mask_offset = vector_index & 0b111;
      bool mask_value = ((mask_span[mask_index] >> mask_offset) & 0b1) != 0;
      // Compute result.
      Vs2 vs2 = GetInstructionSource<Vs2>(inst, 0, vector_index);
      Vs1 vs1 = GetInstructionSource<Vs1>(inst, 1,
                                          (vector_scalar ? 0 : vector_index));
      auto value = op(vs2, vs1, mask_value);
      if (value.has_value()) {
        dest_span[i] = value.value();
      } else if (mask_value) {
        // If there is no value returned, but the mask_value is true, check
        // to see if there was an exception.
        if (rv_vector->vector_exception()) {
          rv_vector->set_vstart(vector_index);
          exception = true;
          break;
        }
      }
      vector_index++;
    }
    // Submit the destination db .
    dest_db->Submit();
    item_index = 0;
  }
  rv_vector->clear_vstart();
}

// This helper function handles the case of two source operand vector
// operations. It implements all the checking necessary for both widening and
// narrowing operations.
template <typename Vd, typename Vs2, typename Vs1>
void RiscVBinaryVectorOp(RiscVVectorState *rv_vector, const Instruction *inst,
                         std::function<Vd(Vs2, Vs1)> op) {
  RiscVMaskBinaryVectorOp<Vd, Vs2, Vs1>(
      rv_vector, inst,
      [op](Vs2 vs2, Vs1 vs1, bool mask_value) -> std::optional<Vd> {
        if (mask_value) {
          return op(vs2, vs1);
        }
        return std::nullopt;
      });
}

template <typename Vd, typename Vs2, typename Vs1>
void RiscVBinaryVectorOpWithFflags(
    RiscVVectorState *rv_vector, const Instruction *inst,
    std::function<std::tuple<Vd, uint32_t>(Vs2, Vs1)> op) {
  if (rv_vector->vector_exception()) return;
  int num_elements = rv_vector->vector_length();
  int lmul = rv_vector->vector_length_multiplier();
  int sew = rv_vector->selected_element_width();
  int lmul_vd = lmul * sizeof(Vd) / sew;
  int lmul_vs2 = lmul * sizeof(Vs2) / sew;
  int lmul_vs1 = lmul * sizeof(Vs1) / sew;
  if (lmul_vd > 64 || lmul_vs2 > 64 || lmul_vs1 > 64) {
    rv_vector->set_vector_exception();
    LOG(ERROR) << "Illegal lmul value";
    return;
  }
  if (lmul_vd == 0 || lmul_vs2 == 0 || lmul_vs1 == 0) {
    rv_vector->set_vector_exception();
    LOG(ERROR) << "Illegal lmul_value";
    return;
  }
  int elements_per_vector =
      rv_vector->vector_register_byte_length() / sizeof(Vd);
  int max_regs = (num_elements + elements_per_vector - 1) / elements_per_vector;
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
  // Get the vector start element index and compute where to start
  // the operation.
  int vector_index = rv_vector->vstart();
  int start_reg = vector_index / elements_per_vector;
  int item_index = vector_index % elements_per_vector;
  // Determine if it's vector-vector or vector-scalar.
  bool vector_scalar = inst->Source(1)->shape()[0] == 1;
  // Iterate over the number of registers to write.
  bool exception = false;
  uint32_t fflags = 0;
  for (int reg = start_reg;
       !exception && (reg < max_regs) && (vector_index < num_elements); reg++) {
    // Allocate data buffer for the new register data.
    auto *dest_db = dest_op->CopyDataBuffer(reg);
    auto dest_span = dest_db->Get<Vd>();
    // Write data into register subject to masking.
    int element_count = std::min(elements_per_vector, num_elements);
    for (int i = item_index;
         (i < element_count) && (vector_index < num_elements); i++) {
      // Get the mask value.
      int mask_index = vector_index >> 3;
      int mask_offset = vector_index & 0b111;
      bool mask_value = ((mask_span[mask_index] >> mask_offset) & 0b1) != 0;
      // Compute result.
      Vs2 vs2 = GetInstructionSource<Vs2>(inst, 0, vector_index);
      Vs1 vs1 = GetInstructionSource<Vs1>(inst, 1,
                                          (vector_scalar ? 0 : vector_index));
      if (mask_value) {
        auto [value, flag] = op(vs2, vs1);
        dest_span[i] = value;
        fflags |= flag;
        if (rv_vector->vector_exception()) {
          rv_vector->set_vstart(vector_index);
          exception = true;
          break;
        }
      }
      vector_index++;
    }
    // Submit the destination dbs.
    dest_db->Submit();
    item_index = 0;
  }
  auto *flag_db = inst->Destination(1)->AllocateDataBuffer();
  flag_db->Set<uint32_t>(0, fflags);
  flag_db->Submit();
  rv_vector->clear_vstart();
}

// This helper function handles three source operand vector operations. It
// implements all the checking necessary for both widening and narrowing
// operations.
template <typename Vd, typename Vs2, typename Vs1>
void RiscVTernaryVectorOp(RiscVVectorState *rv_vector, const Instruction *inst,
                          std::function<Vd(Vs2, Vs1, Vd)> op) {
  if (rv_vector->vector_exception()) return;
  int num_elements = rv_vector->vector_length();
  int lmul = rv_vector->vector_length_multiplier();
  int sew = rv_vector->selected_element_width();
  int lmul_vd = lmul * sizeof(Vd) / sew;
  int lmul_vs2 = lmul * sizeof(Vs2) / sew;
  int lmul_vs1 = lmul * sizeof(Vs1) / sew;
  if (lmul_vd > 64 || lmul_vs2 > 64 || lmul_vs1 > 64) {
    rv_vector->set_vector_exception();
    LOG(ERROR) << "Illegal lmul value";
    return;
  }
  if (lmul_vd == 0 || lmul_vs2 == 0 || lmul_vs1 == 0) {
    rv_vector->set_vector_exception();
    LOG(ERROR) << "Illegal lmul_value";
    return;
  }
  int elements_per_vector =
      rv_vector->vector_register_byte_length() / sizeof(Vd);
  int max_regs = (num_elements + elements_per_vector - 1) / elements_per_vector;
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
  auto *mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(3));
  auto mask_span = mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  // Get the vector start element index and compute where to start
  // the operation.
  int vector_index = rv_vector->vstart();
  int start_reg = vector_index / elements_per_vector;
  int item_index = vector_index % elements_per_vector;
  // Determine if it's vector-vector or vector-scalar.
  bool vector_scalar = inst->Source(1)->shape()[0] == 1;
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
      int mask_index = vector_index >> 3;
      int mask_offset = vector_index & 0b111;
      bool mask_value = ((mask_span[mask_index] >> mask_offset) & 0b1) != 0;
      // Compute result.
      Vs2 vs2 = GetInstructionSource<Vs2>(inst, 0, vector_index);
      Vs1 vs1 = GetInstructionSource<Vs1>(inst, 1,
                                          (vector_scalar ? 0 : vector_index));
      Vd vd = GetInstructionSource<Vd>(inst, 2, vector_index);
      if (mask_value) {
        dest_span[i] = op(vs2, vs1, vd);
      }
      vector_index++;
    }
    // Submit the destination db .
    dest_db->Submit();
    item_index = 0;
  }
  rv_vector->clear_vstart();
}

// The reduction instructions take Vs1[0], and all the elements (subject to
// masking) from Vs2 and apply the reduction operation to produce a single
// element that is written to Vd[0].
template <typename Vd, typename Vs2, typename Vs1>
void RiscVBinaryReductionVectorOp(RiscVVectorState *rv_vector,
                                  const Instruction *inst,
                                  std::function<Vd(Vd, Vs2)> op) {
  if (rv_vector->vector_exception()) return;
  if (rv_vector->vstart()) {
    rv_vector->vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  int lmul = rv_vector->vector_length_multiplier();
  int lmul_vd = lmul * sizeof(Vd) / sew;
  int lmul_vs2 = lmul * sizeof(Vs2) / sew;
  int lmul_vs1 = lmul * sizeof(Vs1) / sew;
  if (lmul_vd > 64 || lmul_vs2 > 64 || lmul_vs1 > 64) {
    rv_vector->set_vector_exception();
    LOG(ERROR) << "Illegal lmul value";
    return;
  }
  if (lmul_vd == 0 || lmul_vs2 == 0 || lmul_vs1 == 0) {
    rv_vector->set_vector_exception();
    LOG(ERROR) << "Illegal lmul_value";
    return;
  }
  int num_elements = rv_vector->vector_length();
  // Get the vector mask.
  auto *mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(2));
  auto mask_span = mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  Vd accumulator =
      static_cast<Vd>(generic::GetInstructionSource<Vs1>(inst, 1, 0));
  for (int i = 0; i < num_elements; i++) {
    int mask_index = i >> 3;
    int mask_offset = i & 0b111;
    bool mask_value = (mask_span[mask_index] >> mask_offset) & 0b1;
    if (mask_value) {
      accumulator =
          op(accumulator, generic::GetInstructionSource<Vs2>(inst, 0, i));
    }
  }
  auto *dest_op =
      static_cast<RV32VectorDestinationOperand *>(inst->Destination(0));
  auto dest_db = dest_op->CopyDataBuffer();
  dest_db->Set<Vd>(0, accumulator);
  dest_db->Submit();
  rv_vector->clear_vstart();
}

template <typename T>
T GetRoundingBit(int rounding_mode, T rounding_bits, int size) {
  switch (rounding_mode) {
    case 0:  // Round-to-nearest-up (add +0.5 lsb)
      if (size < 2) return 0;
      return (rounding_bits >> (size - 2)) & 0b1;
    case 1: {  // Round-to-nearest-event
      T v_d_minus_1 = (size < 2) ? 0 : (rounding_bits >> (size - 2)) & 0b1;
      T v_d = (size == 0) ? 0 : (rounding_bits >> (size - 1)) & 0b1;
      T v_d_minus_2_0 = (size < 3)
                            ? 0
                            : (rounding_bits & ~(std::numeric_limits<T>::max()
                                                 << (size - 2))) != 0;
      return v_d_minus_1 & (v_d_minus_2_0 | v_d);
    }
    case 2:  // Round-down (truncate).
      return 0;
    case 3: {  // Round-to-odd.
      T v_d_minus_1_0 = (size < 2)
                            ? 0
                            : (rounding_bits & ~(std::numeric_limits<T>::max()
                                                 << (size - 1))) != 0;
      T v_d = (rounding_bits >> (size - 1)) & 0b1;
      return (!v_d) & v_d_minus_1_0;
    }
    default:
      LOG(ERROR) << "GetRoundingBit: Invalid value for rounding mode";
      break;
  }
  return 0;
}

template <typename T>
T RoundOff(RiscVVectorState *rv_vector, T value, int size) {
  auto rm = rv_vector->vxrm();
  auto ret = (value >> size) + GetRoundingBit<T>(rm, value, size + 1);
  return ret;
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_VECTOR_INSTRUCTION_HELPERS_H_
