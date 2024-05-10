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

#include "riscv/riscv_a_instructions.h"

#include <cstdint>

#include "absl/status/status.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::operator*;  // NOLINT: is used below (clang error).

// This file contains the definitions of the semantic functions for the A, or
// atomic, subset of the RiscV architecture. Each semantic function calls the
// helper function which does all the heavy lifting.

using Operation = util::AtomicMemoryOpInterface::Operation;

// Helper function for the atomic memory operation semantic functions.
template <typename T>
static inline void AInstructionHelper(Instruction *inst, Operation op,
                                      bool has_store_value) {
  auto *state = static_cast<RiscVState *>(inst->state());
  auto *atomic = state->atomic_memory();
  // If the atomic memory operation interface is nullptr, this is an illegal
  // instruction.
  if (atomic == nullptr) {
    state->Trap(/*is_interrupt*/ false, /*trap_value*/ 0,
                *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  // Submit the memory operation.
  auto address = generic::GetInstructionSource<uint64_t>(inst, 0);
  auto *db = inst->state()->db_factory()->Allocate<T>(1);
  db->set_latency(0);
  // Only access the operand if there is a value to be read.
  if (has_store_value) {
    db->template Set<T>(0, generic::GetInstructionSource<T>(inst, 1));
  }
  // This transfers ownership of db to context. Don't DecRef.
  auto *context = new LoadContext(db);
  auto status = state->atomic_memory()->PerformMemoryOp(address, op, db,
                                                        inst->child(), context);
  // If the operation is unimplemented, this is an illegal instruction.
  if (absl::IsUnimplemented(status)) {
    state->Trap(/*is_interrupt*/ false, /*trap_value*/ 0,
                *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  context->DecRef();
}

void ALrw(Instruction *instruction) {
  AInstructionHelper<uint32_t>(instruction, Operation::kLoadLinked,
                               /*has_store_value*/ false);
}

void AScw(Instruction *instruction) {
  AInstructionHelper<uint32_t>(instruction, Operation::kStoreConditional,
                               /*has_store_value*/ true);
}

void AAmoswapw(Instruction *instruction) {
  AInstructionHelper<uint32_t>(instruction, Operation::kAtomicSwap,
                               /*has_store_value*/ true);
}

void AAmoaddw(Instruction *instruction) {
  AInstructionHelper<uint32_t>(instruction, Operation::kAtomicAdd,
                               /*has_store_value*/ true);
}

void AAmoandw(Instruction *instruction) {
  AInstructionHelper<uint32_t>(instruction, Operation::kAtomicAnd,
                               /*has_store_value*/ true);
}

void AAmoorw(Instruction *instruction) {
  AInstructionHelper<uint32_t>(instruction, Operation::kAtomicOr,
                               /*has_store_value*/ true);
}

void AAmoxorw(Instruction *instruction) {
  AInstructionHelper<uint32_t>(instruction, Operation::kAtomicXor,
                               /*has_store_value*/ true);
}

void AAmomaxw(Instruction *instruction) {
  AInstructionHelper<uint32_t>(instruction, Operation::kAtomicMax,
                               /*has_store_value*/ true);
}

void AAmomaxuw(Instruction *instruction) {
  AInstructionHelper<uint32_t>(instruction, Operation::kAtomicMaxu,
                               /*has_store_value*/ true);
}

void AAmominw(Instruction *instruction) {
  AInstructionHelper<uint32_t>(instruction, Operation::kAtomicMin,
                               /*has_store_value*/ true);
}

void AAmominuw(Instruction *instruction) {
  AInstructionHelper<uint32_t>(instruction, Operation::kAtomicMinu,
                               /*has_store_value*/ true);
}

void ALrd(Instruction *instruction) {
  AInstructionHelper<uint64_t>(instruction, Operation::kLoadLinked,
                               /*has_store_value*/ false);
}

void AScd(Instruction *instruction) {
  AInstructionHelper<uint64_t>(instruction, Operation::kStoreConditional,
                               /*has_store_value*/ true);
}

void AAmoswapd(Instruction *instruction) {
  AInstructionHelper<uint64_t>(instruction, Operation::kAtomicSwap,
                               /*has_store_value*/ true);
}

void AAmoaddd(Instruction *instruction) {
  AInstructionHelper<uint64_t>(instruction, Operation::kAtomicAdd,
                               /*has_store_value*/ true);
}

void AAmoandd(Instruction *instruction) {
  AInstructionHelper<uint64_t>(instruction, Operation::kAtomicAnd,
                               /*has_store_value*/ true);
}

void AAmoord(Instruction *instruction) {
  AInstructionHelper<uint64_t>(instruction, Operation::kAtomicOr,
                               /*has_store_value*/ true);
}

void AAmoxord(Instruction *instruction) {
  AInstructionHelper<uint64_t>(instruction, Operation::kAtomicXor,
                               /*has_store_value*/ true);
}

void AAmomaxd(Instruction *instruction) {
  AInstructionHelper<uint64_t>(instruction, Operation::kAtomicMax,
                               /*has_store_value*/ true);
}

void AAmomaxud(Instruction *instruction) {
  AInstructionHelper<uint64_t>(instruction, Operation::kAtomicMaxu,
                               /*has_store_value*/ true);
}

void AAmomind(Instruction *instruction) {
  AInstructionHelper<uint64_t>(instruction, Operation::kAtomicMin,
                               /*has_store_value*/ true);
}

void AAmominud(Instruction *instruction) {
  AInstructionHelper<uint64_t>(instruction, Operation::kAtomicMinu,
                               /*has_store_value*/ true);
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
