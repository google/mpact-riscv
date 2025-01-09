// Copyright 2025 Google LLC
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

#include "riscv/riscv_vector_basic_bit_manipulation_instructions.h"

#include <cstdint>

#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/types/span.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_instruction_helpers.h"
#include "riscv/riscv_vector_state.h"

using ::mpact::sim::generic::operator*;  // NOLINT: is used below (clang error).

namespace mpact {
namespace sim {
namespace riscv {

void RV32VUnimplementedInstruction(const Instruction *inst) {
  auto *state = static_cast<RiscVState *>(inst->state());
  state->Trap(/*is_interrupt*/ false, /*trap_value*/ 0,
              *ExceptionCode::kIllegalInstruction,
              /*epc*/ inst->address(), inst);
}

namespace {
template <typename T>
T BitReverse(T input) {
  T result = 0;
  for (int i = 0; i < sizeof(T) * 8; ++i) {
    result <<= 1;
    result |= (input & 1);
    input >>= 1;
  }
  return result;
}

template <class T>
constexpr T ByteSwap(T input) {
  // TODO(julianmb): Once c++23 is supported, use std::byteswap.
  T result = 0;
  for (int i = 0; i < sizeof(T); ++i) {
    result |= ((input >> (i * 8)) & 0xFF) << ((sizeof(T) - 1 - i) * 8);
  }
  return result;
}
}  // namespace

void Vandn(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t vs2, uint8_t vs1) -> uint8_t { return vs2 & ~vs1; });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2, uint16_t vs1) -> uint16_t { return vs2 & ~vs1; });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2, uint32_t vs1) -> uint32_t { return vs2 & ~vs1; });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2, uint64_t vs1) -> uint64_t { return vs2 & ~vs1; });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

void Vbrev8(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVUnaryVectorOp<uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t vs2) -> uint8_t { return BitReverse(vs2); });
    case 2:
      return RiscVUnaryVectorOp<uint16_t, uint16_t>(
          rv_vector, inst, [](uint16_t vs2) -> uint16_t {
            absl::Span<uint8_t> span =
                absl::MakeSpan(reinterpret_cast<uint8_t *>(&vs2), sizeof(vs2));
            for (uint8_t &byte : span) {
              byte = BitReverse(byte);
            }
            return vs2;
          });
    case 4:
      return RiscVUnaryVectorOp<uint32_t, uint32_t>(
          rv_vector, inst, [](uint32_t vs2) -> uint32_t {
            absl::Span<uint8_t> span =
                absl::MakeSpan(reinterpret_cast<uint8_t *>(&vs2), sizeof(vs2));
            for (uint8_t &byte : span) {
              byte = BitReverse(byte);
            }
            return vs2;
          });
    case 8:
      return RiscVUnaryVectorOp<uint64_t, uint64_t>(
          rv_vector, inst, [](uint64_t vs2) -> uint64_t {
            absl::Span<uint8_t> span =
                absl::MakeSpan(reinterpret_cast<uint8_t *>(&vs2), sizeof(vs2));
            for (uint8_t &byte : span) {
              byte = BitReverse(byte);
            }
            return vs2;
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

void Vrev8(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVUnaryVectorOp<uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2) -> uint8_t { return vs2; });
    case 2:
      return RiscVUnaryVectorOp<uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2) -> uint16_t { return ByteSwap(vs2); });
    case 4:
      return RiscVUnaryVectorOp<uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2) -> uint32_t { return ByteSwap(vs2); });
    case 8:
      return RiscVUnaryVectorOp<uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2) -> uint64_t { return ByteSwap(vs2); });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

void Vrol(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2, uint8_t vs1) -> uint8_t {
            uint8_t rotate_amount = vs1 & 0b0000'0111;
            return absl::rotl(vs2, rotate_amount);
          });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst, [](uint16_t vs2, uint16_t vs1) -> uint16_t {
            uint8_t rotate_amount = vs1 & 0b0000'1111;
            return absl::rotl(vs2, rotate_amount);
          });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst, [](uint32_t vs2, uint32_t vs1) -> uint32_t {
            uint8_t rotate_amount = vs1 & 0b0001'1111;
            return absl::rotl(vs2, rotate_amount);
          });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst, [](uint64_t vs2, uint64_t vs1) -> uint64_t {
            uint8_t rotate_amount = vs1 & 0b0011'1111;
            return absl::rotl(vs2, rotate_amount);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

void Vror(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint8_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2, uint8_t vs1) -> uint8_t {
            uint8_t rotate_amount = vs1 & 0b0000'0111;
            return absl::rotr(vs2, rotate_amount);
          });
    case 2:
      return RiscVBinaryVectorOp<uint16_t, uint16_t, uint16_t>(
          rv_vector, inst, [](uint16_t vs2, uint16_t vs1) -> uint16_t {
            uint8_t rotate_amount = vs1 & 0b0000'1111;
            return absl::rotr(vs2, rotate_amount);
          });
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst, [](uint32_t vs2, uint32_t vs1) -> uint32_t {
            uint8_t rotate_amount = vs1 & 0b0001'1111;
            return absl::rotr(vs2, rotate_amount);
          });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst, [](uint64_t vs2, uint64_t vs1) -> uint64_t {
            uint8_t rotate_amount = vs1 & 0b0011'1111;
            return absl::rotr(vs2, rotate_amount);
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Instructions that are only in Zvbb

void Vbrev(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVUnaryVectorOp<uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t vs2) -> uint8_t { return BitReverse(vs2); });
    case 2:
      return RiscVUnaryVectorOp<uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2) -> uint16_t { return BitReverse(vs2); });
    case 4:
      return RiscVUnaryVectorOp<uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2) -> uint32_t { return BitReverse(vs2); });
    case 8:
      return RiscVUnaryVectorOp<uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2) -> uint64_t { return BitReverse(vs2); });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

void Vclz(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVUnaryVectorOp<uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t vs2) -> uint8_t { return absl::countl_zero(vs2); });
    case 2:
      return RiscVUnaryVectorOp<uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2) -> uint16_t { return absl::countl_zero(vs2); });
    case 4:
      return RiscVUnaryVectorOp<uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2) -> uint32_t { return absl::countl_zero(vs2); });
    case 8:
      return RiscVUnaryVectorOp<uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2) -> uint64_t { return absl::countl_zero(vs2); });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

void Vctz(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVUnaryVectorOp<uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t vs2) -> uint8_t { return absl::countr_zero(vs2); });
    case 2:
      return RiscVUnaryVectorOp<uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2) -> uint16_t { return absl::countr_zero(vs2); });
    case 4:
      return RiscVUnaryVectorOp<uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2) -> uint32_t { return absl::countr_zero(vs2); });
    case 8:
      return RiscVUnaryVectorOp<uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2) -> uint64_t { return absl::countr_zero(vs2); });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

void VectorVcpop(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVUnaryVectorOp<uint8_t, uint8_t>(
          rv_vector, inst,
          [](uint8_t vs2) -> uint8_t { return absl::popcount(vs2); });
    case 2:
      return RiscVUnaryVectorOp<uint16_t, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2) -> uint16_t { return absl::popcount(vs2); });
    case 4:
      return RiscVUnaryVectorOp<uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2) -> uint32_t { return absl::popcount(vs2); });
    case 8:
      return RiscVUnaryVectorOp<uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2) -> uint64_t { return absl::popcount(vs2); });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

void Vwsll(Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 1:
      return RiscVBinaryVectorOp<uint16_t, uint8_t, uint8_t>(
          rv_vector, inst, [](uint8_t vs2, uint8_t vs1) -> uint16_t {
            return static_cast<uint16_t>(vs2) << (vs1 & 0x0F);
          });
    case 2:
      return RiscVBinaryVectorOp<uint32_t, uint16_t, uint16_t>(
          rv_vector, inst, [](uint16_t vs2, uint16_t vs1) -> uint32_t {
            return static_cast<uint32_t>(vs2) << (vs1 & 0x1F);
          });
    case 4:
      return RiscVBinaryVectorOp<uint64_t, uint32_t, uint32_t>(
          rv_vector, inst, [](uint32_t vs2, uint32_t vs1) -> uint64_t {
            return static_cast<uint64_t>(vs2) << (vs1 & 0x3F);
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
