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

#include "riscv/riscv_vector_memory_instructions.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/register.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_state.h"

namespace mpact {
namespace sim {
namespace riscv {

using generic::GetInstructionSource;

// Helper function used by the load child instructions (non segment loads) that
// writes the loaded data into the registers.
template <typename T>
absl::Status WriteBackLoadData(int vector_register_byte_length,
                               const Instruction *inst) {
  // Get values from context.
  auto *context = static_cast<VectorLoadContext *>(inst->context());
  auto masks = context->mask_db->Get<bool>();
  auto values = context->value_db->Get<T>();
  int vector_start = context->vstart;
  int vector_length = context->vlength;

  int element_size = sizeof(T);
  int elements_per_vector = vector_register_byte_length / element_size;
  int max_regs =
      (vector_length + elements_per_vector - 1) / elements_per_vector;
  // Verify that the dest_op has enough registers. Else signal error.
  auto *dest_op =
      static_cast<RV32VectorDestinationOperand *>(inst->Destination(0));
  if (dest_op->size() < max_regs) {
    // TODO: signal error.
    return absl::InternalError("Not enough registers in destination operand");
  }
  // Compute the number of values to be written.
  int value_count = masks.size();
  if (vector_length - vector_start != value_count) {
    // TODO: signal error.
    return absl::InternalError(
        absl::StrCat("The number of mask elements (", value_count,
                     ") differs from the number of elements to write (",
                     vector_length - vector_start, ")"));
  }
  int load_data_index = 0;
  int start_reg = vector_start / elements_per_vector;
  int item_index = vector_start % elements_per_vector;
  // Iterate over the number of registers to write.
  for (int reg = start_reg; (reg < max_regs) && (value_count > 0); reg++) {
    // Allocate data buffer for the new register data.
    auto *dest_db = dest_op->CopyDataBuffer(reg);
    auto dest_span = dest_db->Get<T>();
    // Write data into register subject to masking.
    int count = std::min(elements_per_vector - item_index, value_count);
    for (int i = item_index; i < count; i++) {
      if (masks[load_data_index + i]) {
        dest_span[i] = values[load_data_index + i];
      }
    }
    value_count -= count;
    load_data_index += count;
    dest_db->Submit(0);
    item_index = 0;
  }
  return absl::OkStatus();
}

// Helper function used by the load child instructions (for segment loads) that
// writes the loaded data into the registers.
template <typename T>
absl::Status WriteBackSegmentLoadData(int vector_register_byte_length,
                                      const Instruction *inst) {
  // The number of fields in each segment.
  int num_fields = GetInstructionSource<uint32_t>(inst, 0) + 1;
  // Get values from context.
  auto *context = static_cast<VectorLoadContext *>(inst->context());
  auto masks = context->mask_db->Get<bool>();
  auto values = context->value_db->Get<T>();
  int start_segment = context->vstart;
  int vector_length = context->vlength;

  int element_size = sizeof(T);
  int num_segments = masks.size() / num_fields;
  // Number of registers written for each field.
  int max_elements_per_vector =
      std::min(vector_register_byte_length / element_size, num_segments);
  int num_regs =
      std::max(1, num_segments * element_size / vector_register_byte_length);
  // Total number of registers written.
  int total_regs = num_fields * num_regs;
  // Verify that the dest_op has enough registers. Else signal error.
  auto *dest_op =
      static_cast<RV32VectorDestinationOperand *>(inst->Destination(0));
  if (dest_op->size() < total_regs) {
    return absl::InternalError("Not enough registers in destination operand");
  }
  // Compute the number of segments to be written.
  if (vector_length - start_segment != num_segments) {
    return absl::InternalError(
        absl::StrCat("The number of mask elements (", num_segments,
                     ") differs from the number of elements to write (",
                     vector_length - start_segment, ")"));
  }
  int load_data_index = 0;
  // Data is organized by field. So write back in that order.
  for (int field = 0; field < num_fields; field++) {
    int start_reg =
        field * num_regs + (start_segment / max_elements_per_vector);
    int offset = start_segment % max_elements_per_vector;
    int remaining_data = num_segments;
    for (int reg = start_reg; reg < start_reg + num_regs; reg++) {
      auto *dest_db = dest_op->CopyDataBuffer(reg);
      auto span = dest_db->Get<T>();
      int max_entry =
          std::min(remaining_data + offset, max_elements_per_vector);
      for (int i = offset; i < max_entry; i++) {
        if (masks[load_data_index]) {
          span[i] = values[load_data_index];
        }
        load_data_index++;
        remaining_data--;
      }
      offset = 0;
      dest_db->Submit(0);
    }
  }
  return absl::OkStatus();
}

// This models the vsetvl set of instructions. The immediate versus register
// versions are all modeled by the same function. Flags are bound during decode
// to the two first parameters to specify if rd or rs1 are x0.
void Vsetvl(bool rd_zero, bool rs1_zero, const Instruction *inst) {
  auto *rv_state = static_cast<RiscVState *>(inst->state());
  auto *rv_vector = rv_state->rv_vector();
  uint32_t vtype = GetInstructionSource<uint32_t>(inst, 1) & 0b1'1'111'111;
  // Get previous vtype.
  uint32_t prev_vtype = rv_vector->vtype();
  // Get previous max length.
  int old_max_length = rv_vector->max_vector_length();
  // Set the new vector type.
  rv_vector->SetVectorType(vtype);
  auto new_max_length = rv_vector->max_vector_length();
  uint32_t vl = new_max_length;
  if (rs1_zero && rd_zero) {  // If rs1 and rd are both zero.
    // If max_length changed, then there's an error, otherwise, vector length
    // is now vl.
    if (old_max_length != new_max_length) {
      // ERROR: cannot change max_vector_length.
      // Revert, then set error flag.
      rv_vector->SetVectorType(prev_vtype);
      rv_vector->set_vector_exception();
      return;
    }
    rv_vector->set_vector_length(new_max_length);
    return;
  }
  if (!rs1_zero) {  // There is a requested vector length.
    uint32_t avl = GetInstructionSource<uint32_t>(inst, 0);
    // Unless the requested vl is less than 2 * max, set it to max.
    if (avl <= new_max_length) {
      // If the requested vl is less than max use it.
      vl = avl;
    }

    // The RISCV spec has the following constraint when VLMAX < AVL < 2 * VLMAX:
    //    ceil(AVL / 2) <= vl <= VLMAX
    //
    // This allows vl to be assigned to half of the requested AVL value, however
    // vl may be assigned to VLMAX instead. SiFive implementations of the RISCV
    // vector engine set vl to VLMAX in this case, which is the same approach
    // followed here.
  }
  rv_vector->set_vector_length(vl);
  if (!rd_zero) {  // Update register if there is a writable destination.
    auto *reg = static_cast<generic::RegisterDestinationOperand<uint32_t> *>(
                    inst->Destination(0))
                    ->GetRegister();
    if (rv_state->xlen() == RiscVXlen::RV32) {
      reg->data_buffer()->Set<uint32_t>(0, vl);
    } else {
      reg->data_buffer()->Set<uint64_t>(0, vl);
    }
  }
}

// Vector load - models both strided and unit stride. Strides can be positive,
// zero, or negative.

// Source(0): base address.
// Source(1): stride.
// Source(2): vector mask register, vector constant {1..} if not masked.
// Destination(0): vector destination register.
void VlStrided(int element_width, const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int start = rv_vector->vstart();
  uint64_t base = GetInstructionSource<uint64_t>(inst, 0);
  int64_t stride = GetInstructionSource<int64_t>(inst, 1);
  int emul = element_width * rv_vector->vector_length_multiplier() /
             rv_vector->selected_element_width();
  if ((emul > 64) || (emul == 0)) {
    // TODO: signal vector error.
    LOG(WARNING) << "EMUL (" << emul << ") out of range";
    return;
  }

  // Compute total number of elements to be loaded.
  int num_elements = rv_vector->vector_length();
  int num_elements_loaded = num_elements - start;

  // Allocate address data buffer.
  auto *db_factory = inst->state()->db_factory();
  auto *address_db = db_factory->Allocate<uint64_t>(num_elements_loaded);

  // Allocate the value data buffer that the loaded data is returned in.
  auto *value_db = db_factory->Allocate(num_elements_loaded * element_width);

  // Get the source mask (stored in a single vector register).
  auto *src_mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(2));
  auto src_masks = src_mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();

  // Allocate a byte mask data buffer for the load.
  auto *mask_db = db_factory->Allocate<bool>(num_elements_loaded);

  // Get the spans for addresses and masks.
  auto addresses = address_db->Get<uint64_t>();
  auto masks = mask_db->Get<bool>();

  // The vector mask in the vector register is a bit mask. The mask used in
  // the LoadMemory call is a bool mask so convert the bit masks to bool masks
  // and compute the element addresses.
  for (int i = start; i < num_elements; i++) {
    int index = i >> 3;
    int offset = i & 0b111;
    addresses[i - start] = base + i * stride * element_width;
    masks[i - start] = ((src_masks[index] >> offset) & 0b1) != 0;
  }

  // Set up the context, and submit the load.
  auto *context = new VectorLoadContext(value_db, mask_db, element_width, start,
                                        rv_vector->vector_length());
  auto *rv32_state = static_cast<RiscVState *>(inst->state());
  value_db->set_latency(0);
  rv32_state->LoadMemory(inst, address_db, mask_db, element_width, value_db,
                         inst->child(), context);
  // Release the context and address_db. The others will be released elsewhere.
  context->DecRef();
  address_db->DecRef();
  rv_vector->clear_vstart();
}

// Vector load vector-mask. This is simple, just a single register.

// Source(0): base address.
// Destination(0): vector destination register (for the child instruction).
void Vlm(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  // Compute base address.
  int start = rv_vector->vstart();
  uint64_t base = GetInstructionSource<uint64_t>(inst, 0) + start;
  // Compute the number of bytes to be loaded.
  int num_bytes = rv_vector->vector_register_byte_length() - start;
  // Allocate address data buffer.
  auto *db_factory = inst->state()->db_factory();
  auto *address_db = db_factory->Allocate<uint64_t>(num_bytes);
  // Allocate the value data buffer that the loaded data is returned in.
  auto *value_db = db_factory->Allocate<uint8_t>(num_bytes);
  // Allocate a byte mask data buffer.
  auto *mask_db = db_factory->Allocate<bool>(num_bytes);
  // Get the spans for addresses and masks.
  auto masks = mask_db->Get<bool>();
  auto addresses = address_db->Get<uint64_t>();
  // Set up addresses, mark all masks elements as true.
  for (int i = start; i < num_bytes; i++) {
    addresses[i - start] = base + i;
    masks[i - start] = true;
  }
  // Set up the context, and submit the load.
  auto *context =
      new VectorLoadContext(value_db, mask_db, sizeof(uint8_t), start,
                            rv_vector->vector_register_byte_length());
  auto *rv32_state = static_cast<RiscVState *>(inst->state());
  value_db->set_latency(0);
  rv32_state->LoadMemory(inst, address_db, mask_db, sizeof(uint8_t), value_db,
                         inst->child(), context);
  // Release the context and address db.
  address_db->DecRef();
  context->DecRef();
  rv_vector->clear_vstart();
}

// Vector load indexed (ordered and unordered). Index values are not scaled by
// element size, as the index values can also be treated as multiple base
// addresses with the base address acting as a common offset. Index values are
// treated as unsigned integers, and are zero extended from the element size to
// the internal address size (or truncated in case the internal XLEN is < index
// element size).

// Source(0) base address.
// Source(1) index vector.
// Source(2) masks.
// Destination(0): vector destination register (for the child instruction).
void VlIndexed(int index_width, const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int start = rv_vector->vstart();
  int element_width = rv_vector->selected_element_width();
  int lmul = rv_vector->vector_length_multiplier();
  uint64_t base = GetInstructionSource<uint64_t>(inst, 0);
  auto *index_op = static_cast<RV32VectorSourceOperand *>(inst->Source(1));
  int index_emul = index_width * lmul / element_width;
  // Validate that emul has a legal value.
  if ((index_emul > 64) || (index_emul == 0)) {
    // TODO: signal vector error.
    LOG(WARNING) << absl::StrCat(
        "Vector load indexed: emul (index) out of range: ", index_emul);
    rv_vector->set_vector_exception();
    return;
  }

  // Compute the number of bytes and elements to be loaded.
  int num_elements = rv_vector->vector_length();
  int num_elements_loaded = num_elements - start;
  int num_bytes_loaded = num_elements_loaded * element_width;

  // Allocate address data buffer.
  auto *db_factory = inst->state()->db_factory();
  auto *address_db = db_factory->Allocate<uint64_t>(num_elements_loaded);
  auto addresses = address_db->Get<uint64_t>();

  // Allocate the value data buffer that the loaded data is returned in.
  auto *value_db = db_factory->Allocate(num_bytes_loaded);

  // Get the source mask (stored in a single vector register).
  auto *src_mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(2));
  auto src_masks = src_mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();

  // Allocate a byte mask data buffer for the load.
  auto *mask_db = db_factory->Allocate<bool>(num_elements);
  auto masks = mask_db->Get<bool>();

  // Convert the bit masks to byte masks and compute the element addresses.
  // The index elements are treated as unsigned values.
  for (int i = start; i < num_elements; i++) {
    int mask_index = i >> 3;
    int mask_offset = i & 0b111;
    uint64_t offset;
    switch (index_width) {
      case 1:
        offset = index_op->AsUint8(i);
        break;
      case 2:
        offset = index_op->AsUint16(i);
        break;
      case 4:
        offset = index_op->AsUint32(i);
        break;
      case 8:
        offset = index_op->AsUint64(i);
        break;
      default:
        offset = 0;
        LOG(ERROR) << absl::StrCat("Illegal index width (", index_width, ")");
        rv_vector->set_vector_exception();
        break;
    }
    addresses[i - start] = base + offset;
    masks[i - start] = ((src_masks[mask_index] >> mask_offset) & 0b1) != 0;
  }

  // Set up context and submit load.
  auto *context = new VectorLoadContext(value_db, mask_db, element_width, start,
                                        rv_vector->vector_length());
  auto *rv32_state = static_cast<RiscVState *>(inst->state());
  value_db->set_latency(0);
  rv32_state->LoadMemory(inst, address_db, mask_db, element_width, value_db,
                         inst->child(), context);
  // Release the context and address db.
  address_db->DecRef();
  context->DecRef();
  rv_vector->clear_vstart();
}

// Vector load whole register(s). The number of registers is passed as
// a parameter to this function - bound to the called function object by the
// instruction decoder. Simple function, no masks, no diffrentiation between
// element sizes.
// Source(0): base address.
// Destination(0): vector destination register (for the child instruction).
void VlRegister(int num_regs, const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  // Get base address.
  uint64_t base = GetInstructionSource<uint64_t>(inst, 0);
  int num_elements =
      rv_vector->vector_register_byte_length() * num_regs / sizeof(uint64_t);
  // Allocate data buffers.
  auto *db_factory = inst->state()->db_factory();
  auto *data_db = db_factory->Allocate<uint64_t>(num_elements);
  auto *address_db = db_factory->Allocate<uint64_t>(num_elements);
  auto *mask_db = db_factory->Allocate<bool>(num_elements);
  // Get spans for addresses and masks.
  auto addresses = address_db->Get<uint64_t>();
  auto masks = mask_db->Get<bool>();
  // Compute addresses and set masks to true.
  int sew = rv_vector->selected_element_width();
  for (int i = 0; i < num_elements; i++) {
    addresses[i] = base + i * sew;
    masks[i] = true;
  }

  // Set up context and submit load.
  auto *context = new VectorLoadContext(data_db, mask_db, sizeof(uint64_t), 0,
                                        num_elements);
  auto *rv32_state = static_cast<RiscVState *>(inst->state());
  data_db->set_latency(0);
  rv32_state->LoadMemory(inst, address_db, mask_db, sizeof(uint64_t), data_db,
                         inst->child(), context);
  // Release the context and address db.
  address_db->DecRef();
  context->DecRef();
  rv_vector->clear_vstart();
}

// Vector load segment, unit stride. The stride is the size of each segment,
// i.e., number of fields * element size. The first field of each segment is
// loaded into the first register, the second into the second, etc. If there
// are more segments than elements in the vector register, adjacent vector
// registers are grouped together. So the first field goes in the first register
// group, etc.
// Source(0): base address
// Source(1): mask
// Source(2): number of fields - 1
// Destination(0): vector destination register (for the child instruction).
void VlSegment(int element_width, const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int start = rv_vector->vstart();
  uint64_t base = GetInstructionSource<uint64_t>(inst, 0);
  auto src_mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(1));
  auto src_masks = src_mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  int num_fields = GetInstructionSource<int32_t>(inst, 2) + 1;
  // Effective vector length multiplier.
  int emul = (element_width * rv_vector->vector_length_multiplier()) /
             rv_vector->selected_element_width();
  if (emul * num_fields > 64) {
    // This is a reserved encoding error.
    // If > 64, it means that the number of registers required is > 8.
    // TODO: signal error.
    rv_vector->set_vector_exception();
    return;
  }
  int num_segments = rv_vector->vector_length();
  int segment_stride = num_fields * element_width;
  int num_elements = num_fields * num_segments;
  // Set up data buffers.
  auto *db_factory = inst->state()->db_factory();
  auto *data_db = db_factory->Allocate(num_elements * element_width);
  auto *address_db = db_factory->Allocate<uint64_t>(num_elements);
  auto *mask_db = db_factory->Allocate<bool>(num_elements);
  // Get spans for addresses and masks.
  auto addresses = address_db->Get<uint64_t>();
  auto masks = mask_db->Get<bool>();

  for (int i = start; i < num_segments; i++) {
    int mask_index = i >> 3;
    int mask_offset = i & 0b111;
    bool mask_value = ((src_masks[mask_index] >> mask_offset) & 0x1) != 0;
    for (int field = 0; field < num_fields; field++) {
      masks[field * num_segments + i] = mask_value;
      addresses[field * num_segments + i] =
          base + i * segment_stride + field * element_width;
    }
  }
  auto *context = new VectorLoadContext(data_db, mask_db, element_width, start,
                                        num_segments);
  auto *rv32_state = static_cast<RiscVState *>(inst->state());
  data_db->set_latency(0);
  rv32_state->LoadMemory(inst, address_db, mask_db, element_width, data_db,
                         inst->child(), context);
  // Release the context and address db.
  address_db->DecRef();
  context->DecRef();
  rv_vector->clear_vstart();
}

// Vector load strided adds a byte address stride to the base address for each
// segment. Note, the stride offset is not scaled by the segment size.
// Source(0): base address
// Source(1): stride
// Source(2): mask
// Source(3): number of fields - 1
// Destination(0): vector destination register (for the child instruction).
void VlSegmentStrided(int element_width, const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int start = rv_vector->vstart();
  uint64_t base = GetInstructionSource<uint64_t>(inst, 0);
  int64_t segment_stride = GetInstructionSource<int64_t>(inst, 1);
  auto src_mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(2));
  auto src_masks = src_mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  int num_fields = GetInstructionSource<int32_t>(inst, 3) + 1;
  // Effective vector length multiplier.
  int emul = (element_width * rv_vector->vector_length_multiplier()) /
             rv_vector->selected_element_width();
  if (emul * num_fields > 64) {
    // This is a reserved encoding error.
    // If > 64, it means that the number of registers required is > 8.
    // TODO: signal error.
    rv_vector->set_vector_exception();
    return;
  }
  int num_segments = rv_vector->vector_length();
  int num_elements = num_fields * num_segments;
  // Set up data buffers.
  auto *db_factory = inst->state()->db_factory();
  auto *data_db = db_factory->Allocate(num_elements * element_width);
  auto *address_db = db_factory->Allocate<uint64_t>(num_elements);
  auto *mask_db = db_factory->Allocate<bool>(num_elements);
  // Get the spans for addresses and masks.
  auto addresses = address_db->Get<uint64_t>();
  auto masks = mask_db->Get<bool>();
  for (int i = start; i < num_segments; i++) {
    int mask_index = i >> 3;
    int mask_offset = i & 0b111;
    bool mask_value = ((src_masks[mask_index] >> mask_offset) & 0x1) != 0;
    for (int field = 0; field < num_fields; field++) {
      masks[field * num_segments + i] = mask_value;
      addresses[field * num_segments + i] =
          base + i * segment_stride + field * element_width;
    }
  }
  // Allocate the context and submit the load.
  auto *context = new VectorLoadContext(data_db, mask_db, element_width, start,
                                        num_segments);
  auto *rv32_state = static_cast<RiscVState *>(inst->state());
  data_db->set_latency(0);
  rv32_state->LoadMemory(inst, address_db, mask_db, element_width, data_db,
                         inst->child(), context);
  // Release the context and address db.
  address_db->DecRef();
  context->DecRef();
  rv_vector->clear_vstart();
}

// Vector load segment, indexed. Similar to the other segment loads, except
// that the offset to the base address comes from a vector of indices. Each
// offset is a byte address, and is not scaled by the segment size.
// Source(0): base address
// Source(1): index vector
// Source(2): mask
// Source(3): number of fields - 1
// Destination(0): vector destination register (for the child instruction).
void VlSegmentIndexed(int index_width, const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int start = rv_vector->vstart();
  uint64_t base = GetInstructionSource<uint64_t>(inst, 0);
  auto *index_op = static_cast<RV32VectorSourceOperand *>(inst->Source(1));
  auto src_mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(2));
  auto src_masks = src_mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  int num_fields = GetInstructionSource<int32_t>(inst, 3) + 1;
  int element_width = rv_vector->selected_element_width();
  // Effective vector length multiplier.
  int lmul8 = rv_vector->vector_length_multiplier();
  // Validate lmul.
  if (lmul8 * num_fields > 64) {
    LOG(WARNING) << "Vector segment load indexed: too many registers";
    rv_vector->set_vector_exception();
    return;
  }
  // Index lmul is scaled from the lmul by the relative size of the index
  // element to the SEW (selected element width).
  int index_emul = (element_width * lmul8) / element_width;
  // Validate that index_emul has a legal value.
  if ((index_emul > 64) || (index_emul == 0)) {
    // TODO: signal vector error.
    LOG(WARNING) << absl::StrCat(
        "Vector load indexed: emul (index) out of range: ", index_emul);
    rv_vector->set_vector_exception();
    return;
  }
  int num_segments = rv_vector->vector_length();
  int num_elements = num_fields * num_segments;

  // Set up data buffers.
  auto *db_factory = inst->state()->db_factory();
  auto *data_db = db_factory->Allocate(num_elements * element_width);
  auto *address_db = db_factory->Allocate<uint64_t>(num_elements);
  auto *mask_db = db_factory->Allocate<bool>(num_elements);
  // Get the spans for the addresses and masks.
  auto addresses = address_db->Get<uint64_t>();
  auto masks = mask_db->Get<bool>();

  for (int i = start; i < num_segments; i++) {
    // The mask value is per segment.
    int mask_index = i >> 3;
    int mask_offset = i & 0b111;
    bool mask_value = ((src_masks[mask_index] >> mask_offset) & 0x1) != 0;
    // Read the index value.
    uint64_t offset;
    switch (index_width) {
      case 1:
        offset = index_op->AsUint8(i);
        break;
      case 2:
        offset = index_op->AsUint16(i);
        break;
      case 4:
        offset = index_op->AsUint32(i);
        break;
      case 8:
        offset = index_op->AsUint64(i);
        break;
      default:
        offset = 0;
        // TODO: signal error.
        LOG(ERROR) << "Internal error - illegal value for index_width";
        rv_vector->set_vector_exception();
        return;
    }
    for (int field = 0; field < num_fields; field++) {
      masks[field * num_segments + i] = mask_value;
      addresses[field * num_segments + i] = base + offset + field;
    }
  }
  auto *context = new VectorLoadContext(data_db, mask_db, element_width, start,
                                        num_segments);
  auto *rv32_state = static_cast<RiscVState *>(inst->state());
  data_db->set_latency(0);
  rv32_state->LoadMemory(inst, address_db, mask_db, element_width, data_db,
                         inst->child(), context);
  // Release the context and address db.
  address_db->DecRef();
  context->DecRef();
  rv_vector->clear_vstart();
}

// Child instruction used for non-segment vector loads. This function really
// only is used to select a type specific version of the helper function to
// write back the load data.
void VlChild(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  absl::Status status;
  int byte_length = rv_vector->vector_register_byte_length();
  switch (static_cast<VectorLoadContext *>(inst->context())->element_width) {
    case 1:
      status = WriteBackLoadData<uint8_t>(byte_length, inst);
      break;
    case 2:
      status = WriteBackLoadData<uint16_t>(byte_length, inst);
      break;
    case 4:
      status = WriteBackLoadData<uint32_t>(byte_length, inst);
      break;
    case 8:
      status = WriteBackLoadData<uint64_t>(byte_length, inst);
      break;
    default:
      LOG(ERROR) << "Illegal element width";
      return;
  }
  if (!status.ok()) {
    LOG(WARNING) << status.message();
    rv_vector->set_vector_exception();
  }
}

// Child instruction used for segmen vector loads. This function really only is
// used to select a type specific version of the helper function to write back
// the load data.
void VlSegmentChild(int element_width, const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  absl::Status status;
  int byte_length = rv_vector->vector_register_byte_length();
  switch (static_cast<VectorLoadContext *>(inst->context())->element_width) {
    case 1:
      status = WriteBackSegmentLoadData<uint8_t>(byte_length, inst);
      break;
    case 2:
      status = WriteBackSegmentLoadData<uint16_t>(byte_length, inst);
      break;
    case 4:
      status = WriteBackSegmentLoadData<uint32_t>(byte_length, inst);
      break;
    case 8:
      status = WriteBackSegmentLoadData<uint64_t>(byte_length, inst);
      break;
    default:
      LOG(ERROR) << "Illegal element width";
      return;
  }
  if (!status.ok()) {
    LOG(WARNING) << status.message();
    rv_vector->set_vector_exception();
  }
}

// Templated helper function for vector stores.
template <typename T>
void StoreVectorStrided(int vector_length, int vstart, int emul,
                        const Instruction *inst) {
  uint64_t base = GetInstructionSource<uint64_t>(inst, 1);
  int64_t stride = GetInstructionSource<int64_t>(inst, 2);
  auto *src_mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(3));
  auto src_masks = src_mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();

  // Compute total number of elements to be stored.
  int num_elements = vector_length;
  // Allocate data buffers.
  auto *db_factory = inst->state()->db_factory();
  auto *address_db = db_factory->Allocate<uint64_t>(num_elements);
  auto addresses = address_db->Get<uint64_t>();
  auto *store_data_db = db_factory->Allocate(num_elements * sizeof(T));
  auto *mask_db = db_factory->Allocate<bool>(num_elements);

  // Get the spans for addresses and masks.
  auto store_data = store_data_db->Get<T>();
  auto masks = mask_db->Get<bool>();

  // Convert the bit masks to byte masks. Set up addresses.
  for (int i = vstart; i < num_elements; i++) {
    int mask_index = i >> 3;
    int mask_offset = i & 0b111;
    addresses[i - vstart] = base + i * stride * sizeof(T);
    masks[i - vstart] = ((src_masks[mask_index] >> mask_offset) & 0b1) != 0;
    store_data[i - vstart] = GetInstructionSource<T>(inst, 0, i);
  }
  // Perform the store.
  auto *rv32_state = static_cast<RiscVState *>(inst->state());
  rv32_state->StoreMemory(inst, address_db, mask_db, sizeof(T), store_data_db);
  address_db->DecRef();
  mask_db->DecRef();
  store_data_db->DecRef();
}

// Vector store - strided.
// Source(0): store data.
// Source(1): base address.
// Source(2): stride.
// Source(3): vector mask register, vector constant {1..} if not masked.
void VsStrided(int element_width, const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int emul = element_width * rv_vector->vector_length_multiplier() /
             rv_vector->selected_element_width();
  // Validate that emul has a legal value.
  if ((emul > 64) || (emul == 0)) {
    LOG(WARNING) << absl::StrCat("Illegal emul value for vector store (", emul,
                                 ")");
    rv_vector->set_vector_exception();
    return;
  }
  int vlength = rv_vector->vector_length();
  int vstart = rv_vector->vstart();
  switch (element_width) {
    case 1:
      StoreVectorStrided<uint8_t>(vlength, vstart, emul, inst);
      break;
    case 2:
      StoreVectorStrided<uint16_t>(vlength, vstart, emul, inst);
      break;
    case 4:
      StoreVectorStrided<uint32_t>(vlength, vstart, emul, inst);
      break;
    case 8:
      StoreVectorStrided<uint64_t>(vlength, vstart, emul, inst);
      break;
    default:
      break;
  }
  rv_vector->clear_vstart();
}

// Store vector mask. Single vector register store.
// Source(0): store data
// Source(1): base address
void Vsm(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  // Compute base address.
  int start = rv_vector->vstart();
  uint64_t base = GetInstructionSource<uint64_t>(inst, 1);
  // Compute the number of bytes and elements to be stored.
  int num_bytes = rv_vector->vector_register_byte_length();
  int num_bytes_stored = num_bytes - start;
  // Allocate address data buffer.
  auto *db_factory = inst->state()->db_factory();
  auto *address_db = db_factory->Allocate<uint64_t>(num_bytes_stored);
  auto *store_data_db = db_factory->Allocate(num_bytes_stored);
  auto *mask_db = db_factory->Allocate<uint8_t>(num_bytes_stored);
  // Get the spans for addresses, masks, and store data.
  auto addresses = address_db->Get<uint64_t>();
  auto masks = mask_db->Get<bool>();
  auto store_data = store_data_db->Get<uint8_t>();
  // Convert the bit masks to byte masks. Set up addresses.
  for (int i = start; i < num_bytes; i++) {
    addresses[i - start] = base + i;
    masks[i - start] = true;
    store_data[i - start] = GetInstructionSource<uint8_t>(inst, 0, i);
  }
  auto *rv32_state = static_cast<RiscVState *>(inst->state());
  rv32_state->StoreMemory(inst, address_db, mask_db, sizeof(uint8_t),
                          store_data_db);
  address_db->DecRef();
  mask_db->DecRef();
  store_data_db->DecRef();
  rv_vector->clear_vstart();
}

// Vector store indexed. Index values are not scaled by
// element size, as the index values can also be treated as multiple base
// addresses with the base address acting as a common offset. Index values are
// treated as unsigned integers, and are zero extended from the element size to
// the internal address size (or truncated in case the internal XLEN is < index
// element size).
// Source(0): store data.
// Source(1): base address.
// Source(2): offset vector.
// Source(3): mask.
void VsIndexed(int index_width, const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  // Compute base address.
  int start = rv_vector->vstart();
  int num_elements = rv_vector->vector_length() - start;
  int element_width = rv_vector->selected_element_width();
  int lmul8 = rv_vector->vector_length_multiplier();
  int index_emul = index_width * lmul8 / element_width;
  // Validate that emul has a legal value.
  if ((index_emul > 64) || (index_emul == 0)) {
    // TODO: signal vector error.
    rv_vector->set_vector_exception();
    return;
  }

  uint64_t base = GetInstructionSource<uint64_t>(inst, 1);
  auto *index_op = static_cast<RV32VectorSourceOperand *>(inst->Source(2));

  // Allocate data buffers.
  auto *db_factory = inst->state()->db_factory();
  auto *address_db = db_factory->Allocate<uint64_t>(num_elements);
  auto *value_db = db_factory->Allocate(num_elements * element_width);
  auto *mask_db = db_factory->Allocate<bool>(num_elements);

  // Get the source mask (stored in a single vector register).
  auto *src_mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(3));
  auto src_masks = src_mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();

  // Get the spans for addresses, masks, and data.
  auto masks = mask_db->Get<bool>();
  auto addresses = address_db->Get<uint64_t>();

  // Convert the bit masks to byte masks and compute the element addresses.
  for (int i = start; i < num_elements; i++) {
    int mask_index = i >> 3;
    int mask_offset = i & 0b111;
    uint64_t offset;
    switch (index_width) {
      case 1:
        offset = index_op->AsUint8(i);
        break;
      case 2:
        offset = index_op->AsUint16(i);
        break;
      case 4:
        offset = index_op->AsUint32(i);
        break;
      case 8:
        offset = index_op->AsUint64(i);
        break;
      default:
        offset = 0;
        // TODO: signal error.
        LOG(ERROR) << "Illegal value for index type width";
        return;
    }
    addresses[i - start] = base + offset;
    masks[i - start] = ((src_masks[mask_index] >> mask_offset) & 0b1) != 0;
    switch (element_width) {
      case 1:
        value_db->Set<uint8_t>(i, GetInstructionSource<uint8_t>(inst, 0, i));
        break;
      case 2:
        value_db->Set<uint16_t>(i, GetInstructionSource<uint16_t>(inst, 0, i));
        break;
      case 4:
        value_db->Set<uint32_t>(i, GetInstructionSource<uint32_t>(inst, 0, i));
        break;
      case 8:
        value_db->Set<uint64_t>(i, GetInstructionSource<uint64_t>(inst, 0, i));
        break;
      default:
        offset = 0;
        // TODO: signal error.
        LOG(ERROR) << "Illegal value for element width";
        break;
    }
  }

  // Set up context and submit store
  auto *rv32_state = static_cast<RiscVState *>(inst->state());
  rv32_state->StoreMemory(inst, address_db, mask_db, element_width, value_db);
  address_db->DecRef();
  mask_db->DecRef();
  value_db->DecRef();
  rv_vector->clear_vstart();
}

void VsRegister(int num_regs, const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  uint64_t base = GetInstructionSource<uint64_t>(inst, 1);
  int num_elements =
      rv_vector->vector_register_byte_length() * num_regs / sizeof(uint64_t);
  // Allocate data buffers.
  auto *db_factory = inst->state()->db_factory();
  auto *data_db = db_factory->Allocate<uint64_t>(num_elements);
  auto *address_db = db_factory->Allocate<uint64_t>(num_elements);
  auto *mask_db = db_factory->Allocate<bool>(num_elements);
  // Get the address, mask, and data spans.
  auto addresses = address_db->Get<uint64_t>();
  auto masks = mask_db->Get<bool>();
  auto data = data_db->Get<uint64_t>();
  for (int i = 0; i < num_elements; i++) {
    addresses[i] = base + i * sizeof(uint64_t);
    masks[i] = true;
    data[i] = GetInstructionSource<uint64_t>(inst, 0, i);
  }  // Submit store.
  auto *rv32_state = static_cast<RiscVState *>(inst->state());
  rv32_state->StoreMemory(inst, address_db, mask_db, sizeof(uint64_t), data_db);
  address_db->DecRef();
  mask_db->DecRef();
  data_db->DecRef();
  rv_vector->clear_vstart();
}

// Vector store segment (unit stride). This stores the segments contiguously
// in memory in a sequential manner.
void VsSegment(int element_width, const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int start = rv_vector->vstart();
  uint64_t base_address = GetInstructionSource<uint64_t>(inst, 1);
  auto src_mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(2));
  auto src_masks = src_mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  int num_fields = GetInstructionSource<int32_t>(inst, 3) + 1;
  // Effective vector length multiplier.
  int emul = (element_width * rv_vector->vector_length_multiplier()) /
             rv_vector->selected_element_width();
  if (emul * num_fields > 64) {
    // This is a reserved encoding error.
    // If > 64, it means that the number of registers required is > 8.
    // TODO: signal error.
    LOG(ERROR) << "Reserved encoding error";
    rv_vector->set_vector_exception();
    return;
  }
  int num_segments = rv_vector->vector_length();
  int num_elements = num_fields * num_segments;
  int num_elements_per_reg =
      rv_vector->vector_register_byte_length() / element_width;
  int reg_mul = std::max(1, emul / 8);
  // Set up data buffers.
  auto *db_factory = inst->state()->db_factory();
  auto *data_db = db_factory->Allocate(num_elements * element_width);
  auto *address_db = db_factory->Allocate<uint64_t>(num_elements);
  auto *mask_db = db_factory->Allocate<bool>(num_elements);
  // Get spans for addresses and masks.
  auto addresses = address_db->Get<uint64_t>();
  auto masks = mask_db->Get<bool>();
  auto data1 = data_db->Get<uint8_t>();
  auto data2 = data_db->Get<uint16_t>();
  auto data4 = data_db->Get<uint32_t>();
  auto data8 = data_db->Get<uint64_t>();

  auto *data_op = static_cast<RV32VectorSourceOperand *>(inst->Source(0));
  uint64_t address = base_address;
  int count = 0;
  for (int segment = start; segment < num_segments; segment++) {
    // Masks are applied on a segment basis.
    int mask_index = segment >> 3;
    int mask_offset = segment & 0b111;
    bool mask_value = ((src_masks[mask_index] >> mask_offset) & 0x1) != 0;
    // If the segments span multiple registers, compute the register offset
    // from the current segment number (upper bits).
    int reg_offset = segment / num_elements_per_reg;
    for (int field = 0; field < num_fields; field++) {
      // Compute register offset number within register group.
      int reg_no = field * reg_mul + reg_offset;
      // Compute element address and set mask value.
      addresses[count] = address;
      address += element_width;
      masks[count] = mask_value;
      if (!mask_value) {
        // If mask is false, just increment count and go to next field.
        count++;
        continue;
      }
      // Write store data from register db to data db.
      auto *reg_db = data_op->GetRegister(reg_no)->data_buffer();
      switch (element_width) {
        case 1:
          data1[count] = reg_db->Get<uint8_t>(segment % num_elements_per_reg);
          break;
        case 2:
          data2[count] = reg_db->Get<uint16_t>(segment % num_elements_per_reg);
          break;
        case 4:
          data4[count] = reg_db->Get<uint32_t>(segment % num_elements_per_reg);
          break;
        case 8:
          data8[count] = reg_db->Get<uint64_t>(segment % num_elements_per_reg);
          break;
        default:
          break;
      }
      count++;
    }
  }
  auto *rv32_state = static_cast<RiscVState *>(inst->state());
  rv32_state->StoreMemory(inst, address_db, mask_db, element_width, data_db);
  // Release the dbs.
  address_db->DecRef();
  mask_db->DecRef();
  data_db->DecRef();
  rv_vector->clear_vstart();
}

// Vector strided segment store. This stores each segment contiguously at
// locations separated by the segment stride.
void VsSegmentStrided(int element_width, const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int start = rv_vector->vstart();
  uint64_t base_address = GetInstructionSource<uint64_t>(inst, 1);
  int64_t segment_stride = GetInstructionSource<int64_t>(inst, 2);
  auto src_mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(3));
  auto src_masks = src_mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  int num_fields = GetInstructionSource<int32_t>(inst, 4) + 1;
  // Effective vector length multiplier.
  int emul = (element_width * rv_vector->vector_length_multiplier()) /
             rv_vector->selected_element_width();
  if (emul * num_fields > 64) {
    // This is a reserved encoding error.
    // If > 64, it means that the number of registers required is > 8.
    // TODO: signal error.
    LOG(ERROR) << "Reserved encoding error";
    rv_vector->set_vector_exception();
    return;
  }
  int num_segments = rv_vector->vector_length();
  int num_elements = num_fields * num_segments;
  int num_elements_per_reg =
      rv_vector->vector_register_byte_length() / element_width;
  int reg_mul = std::max(1, emul / 8);
  // Set up data buffers.
  auto *db_factory = inst->state()->db_factory();
  auto *data_db = db_factory->Allocate(num_elements * element_width);
  auto *address_db = db_factory->Allocate<uint64_t>(num_elements);
  auto *mask_db = db_factory->Allocate<bool>(num_elements);
  // Get spans for addresses and masks.
  auto addresses = address_db->Get<uint64_t>();
  auto masks = mask_db->Get<bool>();
  auto data1 = data_db->Get<uint8_t>();
  auto data2 = data_db->Get<uint16_t>();
  auto data4 = data_db->Get<uint32_t>();
  auto data8 = data_db->Get<uint64_t>();

  auto *data_op = static_cast<RV32VectorSourceOperand *>(inst->Source(0));
  uint64_t segment_address = base_address;
  int count = 0;
  for (int segment = start; segment < num_segments; segment++) {
    // Masks are applied on a segment basis.
    int mask_index = segment >> 3;
    int mask_offset = segment & 0b111;
    bool mask_value = ((src_masks[mask_index] >> mask_offset) & 0x1) != 0;
    // If the segments span multiple registers, compute the register offset
    // from the current segment number (upper bits).
    int reg_offset = segment / num_elements_per_reg;
    uint64_t field_address = segment_address;
    for (int field = 0; field < num_fields; field++) {
      // Compute register offset number within register group.
      int reg_no = field * reg_mul + reg_offset;
      // Compute element address and set mask value.
      addresses[count] = field_address;
      field_address += element_width;
      masks[count] = mask_value;
      if (!mask_value) {
        // If mask is false, just increment count and go to next field.
        count++;
        continue;
      }
      // Write store data from register db to data db.
      auto *reg_db = data_op->GetRegister(reg_no)->data_buffer();
      switch (element_width) {
        case 1:
          data1[count] = reg_db->Get<uint8_t>(segment % num_elements_per_reg);
          break;
        case 2:
          data2[count] = reg_db->Get<uint16_t>(segment % num_elements_per_reg);
          break;
        case 4:
          data4[count] = reg_db->Get<uint32_t>(segment % num_elements_per_reg);
          break;
        case 8:
          data8[count] = reg_db->Get<uint64_t>(segment % num_elements_per_reg);
          break;
        default:
          break;
      }
      count++;
    }
    segment_address += segment_stride;
  }
  auto *rv32_state = static_cast<RiscVState *>(inst->state());
  rv32_state->StoreMemory(inst, address_db, mask_db, element_width, data_db);
  // Release the dbs.
  address_db->DecRef();
  mask_db->DecRef();
  data_db->DecRef();
  rv_vector->clear_vstart();
}

// Vector indexted segment store. This instruction stores each segment
// contiguously at an address formed by adding the index value for that
// segment (from the index vector source operand) to the base address.
void VsSegmentIndexed(int index_width, const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int start = rv_vector->vstart();
  uint64_t base_address = GetInstructionSource<uint64_t>(inst, 1);
  auto src_mask_op = static_cast<RV32VectorSourceOperand *>(inst->Source(3));
  auto src_masks = src_mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>();
  int num_fields = GetInstructionSource<int32_t>(inst, 4) + 1;
  int element_width = rv_vector->selected_element_width();
  // Effective vector length multiplier.
  int lmul = rv_vector->vector_length_multiplier();
  int emul = index_width * lmul / element_width;
  if (lmul * num_fields > 64) {
    // This is a reserved encoding error.
    // If > 64, it means that the number of registers required is > 8.
    // TODO: signal error.
    LOG(ERROR) << "Reserved encoding error - lmul * num_fields out of range";
    rv_vector->set_vector_exception();
    return;
  }
  if (emul == 0 || emul > 64) {
    // This is a reserved encoding error.
    // If > 64, it means that the number of registers required is > 8.
    // TODO: signal error.
    LOG(ERROR) << "Reserved encoding error - emul out of range.";
    rv_vector->set_vector_exception();
    return;
  }
  int num_segments = rv_vector->vector_length();
  int num_elements = num_fields * num_segments;
  int num_elements_per_reg =
      rv_vector->vector_register_byte_length() / element_width;
  int reg_mul = std::max(1, lmul / 8);
  // Set up data buffers.
  auto *db_factory = inst->state()->db_factory();
  auto *data_db = db_factory->Allocate(num_elements * element_width);
  auto *address_db = db_factory->Allocate<uint64_t>(num_elements);
  auto *mask_db = db_factory->Allocate<bool>(num_elements);
  // Get spans for addresses and masks.
  auto addresses = address_db->Get<uint64_t>();
  auto masks = mask_db->Get<bool>();
  auto data1 = data_db->Get<uint8_t>();
  auto data2 = data_db->Get<uint16_t>();
  auto data4 = data_db->Get<uint32_t>();
  auto data8 = data_db->Get<uint64_t>();

  auto *data_op = static_cast<RV32VectorSourceOperand *>(inst->Source(0));
  int count = 0;
  for (int segment = start; segment < num_segments; segment++) {
    // Masks are applied on a segment basis.
    int mask_index = segment >> 3;
    int mask_offset = segment & 0b111;
    bool mask_value = ((src_masks[mask_index] >> mask_offset) & 0x1) != 0;
    // If the segments span multiple registers, compute the register offset
    // from the current segment number (upper bits).
    int reg_offset = segment / num_elements_per_reg;
    int64_t index_value;
    switch (index_width) {
      case 1:
        index_value = GetInstructionSource<int8_t>(inst, 2, segment);
        break;
      case 2:
        index_value = GetInstructionSource<int16_t>(inst, 2, segment);
        break;
      case 4:
        index_value = GetInstructionSource<int32_t>(inst, 2, segment);
        break;
      case 8:
        index_value = GetInstructionSource<int64_t>(inst, 2, segment);
        break;
      default:
        LOG(ERROR) << "Invalid index width: " << index_width << ".";
        rv_vector->set_vector_exception();
        return;
    }
    uint64_t field_address = base_address + index_value;
    for (int field = 0; field < num_fields; field++) {
      // Compute register offset number within register group.
      int reg_no = field * reg_mul + reg_offset;
      // Compute element address and set mask value.
      addresses[count] = field_address;
      field_address += element_width;
      masks[count] = mask_value;
      if (!mask_value) {
        // If mask is false, just increment count and go to next field.
        count++;
        continue;
      }
      // Write store data from register db to data db.
      auto *reg_db = data_op->GetRegister(reg_no)->data_buffer();
      switch (element_width) {
        case 1:
          data1[count] = reg_db->Get<uint8_t>(segment % num_elements_per_reg);
          break;
        case 2:
          data2[count] = reg_db->Get<uint16_t>(segment % num_elements_per_reg);
          break;
        case 4:
          data4[count] = reg_db->Get<uint32_t>(segment % num_elements_per_reg);
          break;
        case 8:
          data8[count] = reg_db->Get<uint64_t>(segment % num_elements_per_reg);
          break;
        default:
          LOG(ERROR) << "Invalid element width: " << element_width << ".";
          return;
      }
      count++;
    }
  }
  auto *rv32_state = static_cast<RiscVState *>(inst->state());
  rv32_state->StoreMemory(inst, address_db, mask_db, element_width, data_db);
  // Release the dbs.
  address_db->DecRef();
  mask_db->DecRef();
  data_db->DecRef();
  rv_vector->clear_vstart();
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
