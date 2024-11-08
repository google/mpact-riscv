// Copyright 2024 Google LLC
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

#include "riscv/riscv_zc_instructions.h"

#include <cstdint>
#include <type_traits>

#include "mpact/sim/generic/instruction.h"
#include "riscv/riscv_instruction_helpers.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"

// This file implements the semantic functions for the Zcm* extensions.

namespace mpact::sim::riscv {

using ::mpact::sim::generic::Instruction;

namespace RV32 {

namespace {

constexpr int kStackAdjBase[] = {
    0, 0, 0, 0, 16, 16, 16, 16, 32, 32, 32, 32, 48, 48, 48, 64,
};

}  // namespace

using RegType = ::mpact::sim::riscv::RV32Register;
using UIntReg = typename std::make_unsigned<typename RegType::ValueType>::type;
using IntReg = typename std::make_signed<UIntReg>::type;

// Zcmp instructions.
void RiscVZCmpPush(const Instruction *inst) {
  RiscVState *state = static_cast<RiscVState *>(inst->state());
  int num_regs = inst->SourcesSize() - 3;
  auto *db = state->db_factory()->Allocate<UIntReg>(num_regs);
  auto db_span = db->Get<UIntReg>();
  // Get the register values and put them in the data buffer.
  for (int i = 0; i < num_regs; ++i) {
    auto value = generic::GetInstructionSource<UIntReg>(inst, i + 3);
    db_span[i] = value;
  }
  // Store the data buffer to memory.
  auto sp = generic::GetInstructionSource<UIntReg>(inst, 0);
  state->StoreMemory(inst, sp - sizeof(UIntReg) * num_regs, db);
  db->DecRef();
  // Compute the stack adjustment.
  auto spimm6 = generic::GetInstructionSource<UIntReg>(inst, 1);
  auto rlist = generic::GetInstructionSource<UIntReg>(inst, 2);
  auto sp_adjustment = spimm6 + kStackAdjBase[rlist];
  // Compute the new stack pointer.
  sp -= sp_adjustment;
  RiscVWriteReg<RegType, UIntReg>(inst, 0, sp);
}

namespace {

// This helper pops the number of registers specified into the appropriate
// destination operands, and then adjusts the stack pointer.
void RiscVZCmpPopHelper(const Instruction *inst, int size) {
  RiscVState *state = static_cast<RiscVState *>(inst->state());
  // Compute the stack adjustment.
  auto spimm6 = generic::GetInstructionSource<UIntReg>(inst, 1);
  auto rlist = generic::GetInstructionSource<UIntReg>(inst, 2);
  auto sp_adjustment = spimm6 + kStackAdjBase[rlist];
  auto *db = state->db_factory()->Allocate<UIntReg>(size);
  // Load registers from the stack.
  auto sp = generic::GetInstructionSource<UIntReg>(inst, 0);
  // Start address = sp + sp_adjustment - sizeof(UIntReg) * size;
  uint64_t start_address = sp + sp_adjustment - sizeof(UIntReg) * size;
  state->LoadMemory(inst, start_address, db, nullptr, nullptr);
  auto db_span = db->Get<UIntReg>();
  for (int i = 0; i < size; ++i) {
    RiscVWriteReg<RegType, UIntReg>(inst, i, db_span[i]);
  }
  // Write to the stack pointer register.
  db->DecRef();
  RiscVWriteReg<RegType, UIntReg>(inst, size, sp + sp_adjustment);
}

}  // namespace

void RiscVZCmpPop(const Instruction *inst) {
  // Size is the number of registers to pop.
  int size = inst->DestinationsSize() - 1;
  RiscVZCmpPopHelper(inst, size);
}

void RiscVZCmpPopRet(const Instruction *inst) {
  // Size is the number of registers to pop.
  int size = inst->DestinationsSize() - 2;  // x2 and next_pc.
  RiscVZCmpPopHelper(inst, size);
  // Now perform the return.
  UIntReg target = generic::GetInstructionSource<UIntReg>(inst, 3);
  auto *db = inst->Destination(size + 1)->AllocateDataBuffer();
  db->SetSubmit<UIntReg>(0, target);
  auto *state = static_cast<RiscVState *>(inst->state());
  state->set_branch(true);
}

void RiscVZCmpPopRetz(const Instruction *inst) {
  // Size is the number of registers to pop.
  int size = inst->DestinationsSize() - 3;  // x2, x10, and next_pc.
  RiscVZCmpPopHelper(inst, size);
  // Now clear a0.
  RiscVWriteReg<RegType, UIntReg>(inst, size + 1, 0);
  // Now perform the return.
  UIntReg target = generic::GetInstructionSource<UIntReg>(inst, 3);
  auto *db = inst->Destination(size + 2)->AllocateDataBuffer();
  db->SetSubmit<UIntReg>(0, target);
  auto *state = static_cast<RiscVState *>(inst->state());
  state->set_branch(true);
}

void RiscVZCmpMvTwoRegs(const Instruction *inst) {
  RiscVWriteReg<RegType, UIntReg>(
      inst, 0, generic::GetInstructionSource<UIntReg>(inst, 0));
  RiscVWriteReg<RegType, UIntReg>(
      inst, 1, generic::GetInstructionSource<UIntReg>(inst, 1));
}

// Zcmt instructions.
namespace {

void RiscVZCmtJtHelper(const Instruction *inst, int dest_index) {
  int index = generic::GetInstructionSource<UIntReg>(inst, 0);
  auto *state = static_cast<RiscVState *>(inst->state());
  auto jvt_value = state->jvt()->AsUint64();
  auto mode = jvt_value & 0x3f;
  if (mode != 0) {
    state->Trap(/*is_interrupt=*/false, 0, *ExceptionCode::kIllegalInstruction,
                inst->address(), inst);
    return;
  }
  // Load target address from the jump table.
  UIntReg entry_address = (jvt_value & ~0x3f) + (index * sizeof(UIntReg));
  auto *db = state->db_factory()->Allocate<UIntReg>(1);
  state->LoadMemory(inst, entry_address, db, nullptr, nullptr);
  UIntReg target_address = db->Get<UIntReg>(0);
  db->DecRef();
  // Write the target address to the next pc operand.
  auto *target_db = inst->Destination(0)->AllocateDataBuffer();
  target_db->SetSubmit<UIntReg>(0, target_address);
  state->set_branch(true);
}

}  // namespace

void RiscVZCmtJt(const Instruction *inst) { RiscVZCmtJtHelper(inst, 0); }

void RiscVZCmtJalt(const Instruction *inst) {
  RiscVZCmtJtHelper(inst, 1);
  // Write the return address to the x1 (ra) operand.
  RiscVWriteReg<RegType, UIntReg>(inst, 1, inst->address() + inst->size());
}

}  // namespace RV32

namespace RV64 {

namespace {

constexpr int kStackAdjBase[] = {
    0, 0, 0, 0, 16, 16, 32, 32, 48, 48, 64, 64, 80, 80, 96, 112,
};

}  // namespace

using RegType = ::mpact::sim::riscv::RV64Register;
using UIntReg = typename std::make_unsigned<typename RegType::ValueType>::type;
using IntReg = typename std::make_signed<UIntReg>::type;

// Zcmp instructions.
void RiscVZCmpPush(const Instruction *inst) {
  RiscVState *state = static_cast<RiscVState *>(inst->state());
  int num_regs = inst->SourcesSize() - 3;
  auto *db = state->db_factory()->Allocate<UIntReg>(num_regs);
  auto db_span = db->Get<UIntReg>();
  // Get the register values and put them in the data buffer.
  for (int i = 0; i < num_regs; ++i) {
    auto value = generic::GetInstructionSource<UIntReg>(inst, i + 3);
    db_span[i] = value;
  }
  // Store the data buffer to memory.
  auto sp = generic::GetInstructionSource<UIntReg>(inst, 0);
  state->StoreMemory(inst, sp - sizeof(UIntReg) * num_regs, db);
  db->DecRef();
  // Compute the stack adjustment.
  auto spimm6 = generic::GetInstructionSource<UIntReg>(inst, 1);
  auto rlist = generic::GetInstructionSource<UIntReg>(inst, 2);
  auto sp_adjustment = spimm6 + kStackAdjBase[rlist];
  // Compute the new stack pointer.
  sp -= sp_adjustment;
  RiscVWriteReg<RegType, UIntReg>(inst, 0, sp);
}

namespace {

// This helper pops the number of registers specified into the appropriate
// destination operands, and then adjusts the stack pointer.
void RiscVZCmpPopHelper(const Instruction *inst, int size) {
  RiscVState *state = static_cast<RiscVState *>(inst->state());
  // Compute the stack adjustment.
  auto spimm6 = generic::GetInstructionSource<UIntReg>(inst, 1);
  auto rlist = generic::GetInstructionSource<UIntReg>(inst, 2);
  auto sp_adjustment = spimm6 + kStackAdjBase[rlist];
  auto *db = state->db_factory()->Allocate<UIntReg>(size);
  // Load registers from the stack.
  auto sp = generic::GetInstructionSource<UIntReg>(inst, 0);
  // Start address = sp + sp_adjustment - sizeof(UIntReg) * size;
  uint64_t start_address = sp + sp_adjustment - sizeof(UIntReg) * size;
  state->LoadMemory(inst, start_address, db, nullptr, nullptr);
  auto db_span = db->Get<UIntReg>();
  for (int i = 0; i < size; ++i) {
    RiscVWriteReg<RegType, UIntReg>(inst, i, db_span[i]);
  }
  // Write to the stack pointer register.
  db->DecRef();
  RiscVWriteReg<RegType, UIntReg>(inst, size, sp + sp_adjustment);
}

}  // namespace

void RiscVZCmpPop(const Instruction *inst) {
  // Size is the number of registers to pop.
  int size = inst->DestinationsSize() - 1;
  RiscVZCmpPopHelper(inst, size);
}

void RiscVZCmpPopRet(const Instruction *inst) {
  // Size is the number of registers to pop.
  int size = inst->DestinationsSize() - 2;  // x2 and next_pc.
  RiscVZCmpPopHelper(inst, size);
  // Now perform the return.
  UIntReg target = generic::GetInstructionSource<UIntReg>(inst, 3);
  auto *db = inst->Destination(size + 1)->AllocateDataBuffer();
  db->SetSubmit<UIntReg>(0, target);
  auto *state = static_cast<RiscVState *>(inst->state());
  state->set_branch(true);
}

void RiscVZCmpPopRetz(const Instruction *inst) {
  // Size is the number of registers to pop.
  int size = inst->DestinationsSize() - 3;  // x2, x10, and next_pc.
  RiscVZCmpPopHelper(inst, size);
  // Now clear a0.
  RiscVWriteReg<RegType, UIntReg>(inst, size + 1, 0);
  // Now perform the return.
  UIntReg target = generic::GetInstructionSource<UIntReg>(inst, 3);
  auto *db = inst->Destination(size + 2)->AllocateDataBuffer();
  db->SetSubmit<UIntReg>(0, target);
  auto *state = static_cast<RiscVState *>(inst->state());
  state->set_branch(true);
}

void RiscVZCmpMvTwoRegs(const Instruction *inst) {
  RiscVWriteReg<RegType, UIntReg>(
      inst, 0, generic::GetInstructionSource<UIntReg>(inst, 0));
  RiscVWriteReg<RegType, UIntReg>(
      inst, 1, generic::GetInstructionSource<UIntReg>(inst, 1));
}

// Zcmt instructions.
namespace {

void RiscVZCmtJtHelper(const Instruction *inst, int dest_index) {
  int index = generic::GetInstructionSource<UIntReg>(inst, 0);
  auto *state = static_cast<RiscVState *>(inst->state());
  auto jvt_value = state->jvt()->AsUint64();
  auto mode = jvt_value & 0x3f;
  if (mode != 0) {
    state->Trap(/*is_interrupt=*/false, 0, *ExceptionCode::kIllegalInstruction,
                inst->address(), inst);
    return;
  }
  // Load target address from the jump table.
  UIntReg entry_address = (jvt_value & ~0x3f) + (index * sizeof(UIntReg));
  auto *db = state->db_factory()->Allocate<UIntReg>(1);
  state->LoadMemory(inst, entry_address, db, nullptr, nullptr);
  UIntReg target_address = db->Get<UIntReg>(0);
  db->DecRef();
  // Write the target address to the next pc operand.
  auto *target_db = inst->Destination(0)->AllocateDataBuffer();
  target_db->SetSubmit<UIntReg>(0, target_address);
  state->set_branch(true);
}

}  // namespace

void RiscVZCmtJt(const Instruction *inst) { RiscVZCmtJtHelper(inst, 0); }

void RiscVZCmtJalt(const Instruction *inst) {
  RiscVZCmtJtHelper(inst, 1);
  // Write the return address to the x1 (ra) operand.
  RiscVWriteReg<RegType, UIntReg>(inst, 1, inst->address() + inst->size());
}

}  // namespace RV64

}  // namespace mpact::sim::riscv
