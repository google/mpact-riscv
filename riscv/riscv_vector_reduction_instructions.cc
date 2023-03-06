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

#include "riscv/riscv_vector_reduction_instructions.h"

#include <algorithm>

#include "absl/log/log.h"
#include "riscv/riscv_vector_instruction_helpers.h"

namespace mpact {
namespace sim {
namespace riscv {

// Sum reduction.
void Vredsum(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryReductionVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t acc, uint8_t vs2) -> uint8_t { return acc + vs2; });
    case 2:
      return RiscVBinaryReductionVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t acc, uint16_t vs2) -> uint16_t { return acc + vs2; });
      return;
    case 4:
      return RiscVBinaryReductionVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t acc, uint32_t vs2) -> uint32_t { return acc + vs2; });
      return;
    case 8:
      return RiscVBinaryReductionVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t acc, uint64_t vs2) -> uint64_t { return acc + vs2; });
      return;
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// And reduction.
void Vredand(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryReductionVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t acc, uint8_t vs2) -> uint8_t { return acc & vs2; });
    case 2:
      return RiscVBinaryReductionVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t acc, uint16_t vs2) -> uint16_t { return acc & vs2; });
      return;
    case 4:
      return RiscVBinaryReductionVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t acc, uint32_t vs2) -> uint32_t { return acc & vs2; });
      return;
    case 8:
      return RiscVBinaryReductionVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t acc, uint64_t vs2) -> uint64_t { return acc & vs2; });
      return;
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Or reduction.
void Vredor(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryReductionVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t acc, uint8_t vs2) -> uint8_t { return acc | vs2; });
    case 2:
      return RiscVBinaryReductionVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t acc, uint16_t vs2) -> uint16_t { return acc | vs2; });
      return;
    case 4:
      return RiscVBinaryReductionVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t acc, uint32_t vs2) -> uint32_t { return acc | vs2; });
      return;
    case 8:
      return RiscVBinaryReductionVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t acc, uint64_t vs2) -> uint64_t { return acc | vs2; });
      return;
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Xor reduction.
void Vredxor(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryReductionVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t acc, uint8_t vs2) -> uint8_t { return acc ^ vs2; });
    case 2:
      return RiscVBinaryReductionVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t acc, uint16_t vs2) -> uint16_t { return acc ^ vs2; });
      return;
    case 4:
      return RiscVBinaryReductionVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t acc, uint32_t vs2) -> uint32_t { return acc ^ vs2; });
      return;
    case 8:
      return RiscVBinaryReductionVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t acc, uint64_t vs2) -> uint64_t { return acc ^ vs2; });
      return;
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Unsigned min reduction.
void Vredminu(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryReductionVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t acc, uint8_t vs2) -> uint8_t {
            return std::min(acc, vs2);
          });
    case 2:
      return RiscVBinaryReductionVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst, [](uint16_t acc, uint16_t vs2) -> uint16_t {
            return std::min(acc, vs2);
          });
      return;
    case 4:
      return RiscVBinaryReductionVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst, [](uint32_t acc, uint32_t vs2) -> uint32_t {
            return std::min(acc, vs2);
          });
      return;
    case 8:
      return RiscVBinaryReductionVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst, [](uint64_t acc, uint64_t vs2) -> uint64_t {
            return std::min(acc, vs2);
          });
      return;
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Signed min reduction.
void Vredmin(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryReductionVectorOp<int8_t, int8_t, int8_t>(
          rv_vector, inst,
          [](int8_t acc, int8_t vs2) -> int8_t { return std::min(acc, vs2); });
    case 2:
      return RiscVBinaryReductionVectorOp<int16_t, int16_t, int16_t>(
          rv_vector, inst, [](int16_t acc, int16_t vs2) -> int16_t {
            return std::min(acc, vs2);
          });
      return;
    case 4:
      return RiscVBinaryReductionVectorOp<int32_t, int32_t, int32_t>(
          rv_vector, inst, [](int32_t acc, int32_t vs2) -> int32_t {
            return std::min(acc, vs2);
          });
      return;
    case 8:
      return RiscVBinaryReductionVectorOp<int64_t, int64_t, int64_t>(
          rv_vector, inst, [](int64_t acc, int64_t vs2) -> int64_t {
            return std::min(acc, vs2);
          });
      return;
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Unsigned max reduction.
void Vredmaxu(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryReductionVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t acc, uint8_t vs2) -> uint8_t {
            return std::max(acc, vs2);
          });
    case 2:
      return RiscVBinaryReductionVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst, [](uint16_t acc, uint16_t vs2) -> uint16_t {
            return std::max(acc, vs2);
          });
      return;
    case 4:
      return RiscVBinaryReductionVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst, [](uint32_t acc, uint32_t vs2) -> uint32_t {
            return std::max(acc, vs2);
          });
      return;
    case 8:
      return RiscVBinaryReductionVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst, [](uint64_t acc, uint64_t vs2) -> uint64_t {
            return std::max(acc, vs2);
          });
      return;
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Signed max reduction.
void Vredmax(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryReductionVectorOp<int8_t, int8_t, int8_t>(
          rv_vector, inst,
          [](int8_t acc, int8_t vs2) -> int8_t { return std::max(acc, vs2); });
    case 2:
      return RiscVBinaryReductionVectorOp<int16_t, int16_t, int16_t>(
          rv_vector, inst, [](int16_t acc, int16_t vs2) -> int16_t {
            return std::max(acc, vs2);
          });
      return;
    case 4:
      return RiscVBinaryReductionVectorOp<int32_t, int32_t, int32_t>(
          rv_vector, inst, [](int32_t acc, int32_t vs2) -> int32_t {
            return std::max(acc, vs2);
          });
      return;
    case 8:
      return RiscVBinaryReductionVectorOp<int64_t, int64_t, int64_t>(
          rv_vector, inst, [](int64_t acc, int64_t vs2) -> int64_t {
            return std::max(acc, vs2);
          });
      return;
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Unsigned widening (SEW->SEW * 2) reduction.
void Vwredsumu(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryReductionVectorOp<uint16_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint16_t acc, uint8_t vs2) -> uint16_t {
            return acc + static_cast<uint16_t>(vs2);
          });
    case 2:
      return RiscVBinaryReductionVectorOp<uint32_t, uint16_t, uint16_t>(
          rv_vector, inst, [](uint32_t acc, uint16_t vs2) -> uint32_t {
            return acc + static_cast<uint32_t>(vs2);
          });
      return;
    case 4:
      return RiscVBinaryReductionVectorOp<uint64_t, uint32_t, uint32_t>(
          rv_vector, inst, [](uint64_t acc, uint32_t vs2) -> uint64_t {
            return acc + static_cast<uint64_t>(vs2);
          });
      return;
    case 8:
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Signed widening (SEW->SEW * 2) reduction.
void Vwredsum(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryReductionVectorOp<int16_t, int8_t, int8_t>(
          rv_vector, inst, [](int16_t acc, int8_t vs2) -> int16_t {
            return acc + static_cast<int16_t>(vs2);
          });
    case 2:
      return RiscVBinaryReductionVectorOp<int32_t, int16_t, int16_t>(
          rv_vector, inst, [](int32_t acc, int16_t vs2) -> int32_t {
            return acc + static_cast<int32_t>(vs2);
          });
      return;
    case 4:
      return RiscVBinaryReductionVectorOp<int64_t, int32_t, int32_t>(
          rv_vector, inst, [](int64_t acc, int32_t vs2) -> int64_t {
            return acc + static_cast<int64_t>(vs2);
          });
      return;
    case 8:
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
