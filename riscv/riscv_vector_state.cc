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

#include "riscv/riscv_vector_state.h"

#include "absl/log/log.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

// Constructor for the vector class. Need to pass in the parent RV32 state and
// the vector length in bytes.
RiscVVectorState::RiscVVectorState(RiscVState *state, int byte_length)
    : vector_register_byte_length_(byte_length) {
  state_ = state;
  state->set_rv_vector(this);
  state->set_vector_register_width(byte_length);
}

// This function parses the vector type, as used in the vset* instructions
// and sets the internal vector state accordingly.
void RiscVVectorState::SetVectorType(uint32_t vtype) {
  static const int lmul8_values[8] = {8, 16, 32, 64, 0, 1, 2, 4};
  static const int sew_values[8] = {8, 16, 32, 64, 0, 0, 0, 0};
  set_vtype(vtype);
  // The vtype field is divided into the following fields:
  // [2..0]: vector length multiplier.
  // [5..3]: element width specifier.
  // [6]:    vector tail agnostic bit.
  // [7]:    vector mask agnostic bit.
  // Extract the lmul.
  set_vector_length_multiplier(lmul8_values[(vtype & 0b111)]);
  // Extract the sew and convert from bits to bytes.
  set_selected_element_width(sew_values[(vtype >> 3) & 0b111] >> 3);
  // Extract the tail and mask agnostic flags.
  set_vector_tail_agnostic(static_cast<bool>((vtype >> 6) & 0b1));
  set_vector_mask_agnostic(static_cast<bool>((vtype >> 7) & 0b1));
  // Compute the new max vector length.
  max_vector_length_ = vector_register_byte_length() *
                       vector_length_multiplier() /
                       (8 * selected_element_width());
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
