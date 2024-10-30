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

#include "riscv/riscv_i_instructions.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <type_traits>

#include "mpact/sim/generic/arch_state.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/register.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_instruction_helpers.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

void RiscVIllegalInstruction(const Instruction *inst) {
  auto *state = static_cast<RiscVState *>(inst->state());
  state->Trap(/*is_interrupt*/ false, /*trap_value*/ 0,
              *ExceptionCode::kIllegalInstruction,
              /*epc*/ inst->address(), inst);
}

// The following instruction semantic functions implement basic alu operations.
// They are used for both register-register and register-immediate versions of
// the corresponding instructions.

namespace RV32 {

using RegisterType = RV32Register;

// Register width integer types. These are preferred to uint32_t, etc, for
// those instructions that operate on the entire instruction width.
using UIntReg =
    typename std::make_unsigned<typename RegisterType::ValueType>::type;
using IntReg = typename std::make_signed<UIntReg>::type;

void RiscVIAdd(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a + b; });
}

void RiscVISub(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a - b; });
}

void RiscVISlt(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, IntReg, IntReg>(
      instruction, [](IntReg a, IntReg b) { return a < b; });
}

void RiscVISltu(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a < b; });
}

void RiscVIAnd(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a & b; });
}

void RiscVIOr(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a | b; });
}

void RiscVIXor(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a ^ b; });
}

void RiscVISll(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a << (b & 0x1f); });
}

void RiscVISrl(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a >> (b & 0x1f); });
}

void RiscVISra(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, IntReg, IntReg>(
      instruction, [](IntReg a, IntReg b) { return a >> (b & 0x1f); });
}

// Load upper immediate. It is assumed that the decoder already shifted the
// immediate. Operates on 32 bit quantities, not XLEN bits.
void RiscVILui(const Instruction *instruction) {
  RiscVUnaryOp<RegisterType, uint32_t, uint32_t>(
      instruction, [](uint32_t lhs) { return lhs & ~0xfff; });
}

// Add upper immediate to PC (for PC relative addressing). It is assumed that
// the decoder already shifted the immediate. Operates on 32 bit quantities,
// not XLEN bits.
void RiscVIAuipc(const Instruction *instruction) {
  RiscVUnaryOp<RegisterType, uint32_t, uint32_t>(
      instruction, [instruction](uint32_t lhs) {
        return (lhs & ~0xfff) + instruction->address();
      });
}

// Jump and link instructions. One using a long offset, the other offset plus
// base.
void RiscVIJal(const Instruction *instruction) {
  UIntReg offset = generic::GetInstructionSource<UIntReg>(instruction, 0);
  UIntReg target = (offset + instruction->address()) & ~0x1;
  UIntReg return_address = instruction->address() + instruction->size();
  auto *db = instruction->Destination(0)->AllocateDataBuffer();
  db->SetSubmit<UIntReg>(0, target);
  auto *reg = static_cast<generic::RegisterDestinationOperand<UIntReg> *>(
                  instruction->Destination(1))
                  ->GetRegister();
  reg->data_buffer()->Set<UIntReg>(0, return_address);
}

void RiscVIJalr(const Instruction *instruction) {
  UIntReg reg_base = generic::GetInstructionSource<UIntReg>(instruction, 0);
  UIntReg offset = generic::GetInstructionSource<UIntReg>(instruction, 1);
  UIntReg target = (offset + reg_base) & ~0x1;
  UIntReg return_address = instruction->address() + instruction->size();
  auto *db = instruction->Destination(0)->AllocateDataBuffer();
  db->SetSubmit<UIntReg>(0, target);
  auto *reg = static_cast<generic::RegisterDestinationOperand<UIntReg> *>(
                  instruction->Destination(1))
                  ->GetRegister();
  reg->data_buffer()->Set<UIntReg>(0, return_address);
}

}  // namespace RV32

namespace RV64 {

using RegisterType = RV64Register;

// Register width integer types.
using UIntReg =
    typename std::make_unsigned<typename RegisterType::ValueType>::type;
using IntReg = typename std::make_signed<UIntReg>::type;
using NarrowIntReg = typename ::mpact::sim::generic::NarrowType<IntReg>::type;
using NarrowUIntReg = typename ::mpact::sim::generic::NarrowType<UIntReg>::type;

void RiscVIAdd(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a + b; });
}

void RiscVIAddw(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, IntReg, NarrowIntReg>(
      instruction, [](NarrowIntReg a, NarrowIntReg b) {
        IntReg c = static_cast<IntReg>(a + b);
        return c;
      });
}

void RiscVISub(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a - b; });
}

void RiscVISubw(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, IntReg, NarrowIntReg>(
      instruction, [](NarrowIntReg a, NarrowIntReg b) {
        IntReg c = static_cast<IntReg>(a - b);
        return c;
      });
}

void RiscVISlt(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, IntReg, IntReg>(
      instruction, [](IntReg a, IntReg b) { return a < b; });
}

void RiscVISltu(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a < b; });
}

void RiscVIAnd(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a & b; });
}

void RiscVIOr(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a | b; });
}

void RiscVIXor(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a ^ b; });
}

void RiscVISll(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a << (b & 0x3f); });
}

void RiscVISrl(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, UIntReg, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a >> (b & 0x3f); });
}

void RiscVISra(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, IntReg, IntReg>(
      instruction, [](IntReg a, IntReg b) { return a >> (b & 0x3f); });
}

void RiscVISllw(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, IntReg, NarrowIntReg>(
      instruction, [](NarrowIntReg a, NarrowIntReg b) -> IntReg {
        return static_cast<IntReg>(a << (b & 0x1f));
      });
}

void RiscVISrlw(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, IntReg, NarrowUIntReg>(
      instruction, [](NarrowUIntReg a, NarrowUIntReg b) -> IntReg {
        auto c = a >> (b & 0x1f);
        return static_cast<IntReg>(static_cast<NarrowIntReg>(c));
      });
}

void RiscVISraw(const Instruction *instruction) {
  RiscVBinaryOp<RegisterType, IntReg, NarrowIntReg>(
      instruction, [](NarrowIntReg a, NarrowIntReg b) -> IntReg {
        return static_cast<IntReg>(a >> (b & 0x1f));
      });
}

// Load upper immediate. It is assumed that the decoder already shifted the
// immediate.
void RiscVILui(const Instruction *instruction) {
  RiscVUnaryOp<RegisterType, IntReg, int32_t>(
      instruction, [](int32_t lhs) -> IntReg {
        auto lhs_w = static_cast<IntReg>(lhs);
        return lhs_w & ~0xfff;
      });
}

// Add upper immediate to PC (for PC relative addressing). It is assumed that
// the decoder already shifted the immediate.
void RiscVIAuipc(const Instruction *instruction) {
  RiscVUnaryOp<RegisterType, uint64_t, int32_t>(
      instruction, [instruction](int32_t lhs) {
        auto lhs_w = static_cast<IntReg>(lhs & ~0xfff);
        return lhs_w + instruction->address();
      });
}

// Jump and link instructions. One using a long offset, the other offset plus
// base.
void RiscVIJal(const Instruction *instruction) {
  UIntReg offset = generic::GetInstructionSource<UIntReg>(instruction, 0);
  UIntReg target = offset + instruction->address();
  target &= (std::numeric_limits<UIntReg>::max() << 1);
  UIntReg return_address = instruction->address() + instruction->size();
  auto *db = instruction->Destination(0)->AllocateDataBuffer();
  db->SetSubmit<UIntReg>(0, target);
  auto *reg = static_cast<generic::RegisterDestinationOperand<UIntReg> *>(
                  instruction->Destination(1))
                  ->GetRegister();
  reg->data_buffer()->Set<UIntReg>(0, return_address);
}

void RiscVIJalr(const Instruction *instruction) {
  UIntReg reg_base = generic::GetInstructionSource<UIntReg>(instruction, 0);
  UIntReg offset = generic::GetInstructionSource<UIntReg>(instruction, 1);
  UIntReg target = offset + reg_base;
  target &= (std::numeric_limits<UIntReg>::max() << 1);
  UIntReg return_address = instruction->address() + instruction->size();
  auto *db = instruction->Destination(0)->AllocateDataBuffer();
  db->SetSubmit<UIntReg>(0, target);
  auto *reg = static_cast<generic::RegisterDestinationOperand<UIntReg> *>(
                  instruction->Destination(1))
                  ->GetRegister();
  reg->data_buffer()->Set<UIntReg>(0, return_address);
}

}  // namespace RV64

void RiscVINop(const Instruction *instruction) {}

namespace RV32 {

using RegisterType = RV32Register;

// Register width integer types.
using UIntReg =
    typename std::make_unsigned<typename RegisterType::ValueType>::type;
using IntReg = typename std::make_signed<UIntReg>::type;

void RiscVIBeq(const Instruction *instruction) {
  BranchConditional<RegisterType, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a == b; });
}

void RiscVIBne(const Instruction *instruction) {
  BranchConditional<RegisterType, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a != b; });
}

void RiscVIBlt(const Instruction *instruction) {
  BranchConditional<RegisterType, IntReg>(
      instruction, [](IntReg a, IntReg b) { return a < b; });
}

void RiscVIBltu(const Instruction *instruction) {
  BranchConditional<RegisterType, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a < b; });
}

void RiscVIBge(const Instruction *instruction) {
  BranchConditional<RegisterType, IntReg>(
      instruction, [](IntReg a, IntReg b) { return a >= b; });
}

void RiscVIBgeu(const Instruction *instruction) {
  BranchConditional<RegisterType, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a >= b; });
}

}  // namespace RV32

namespace RV64 {

using RegisterType = RV64Register;

// Register width integer types.
using UIntReg =
    typename std::make_unsigned<typename RegisterType::ValueType>::type;
using IntReg = typename std::make_signed<UIntReg>::type;

void RiscVIBeq(const Instruction *instruction) {
  BranchConditional<RegisterType, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a == b; });
}

void RiscVIBne(const Instruction *instruction) {
  BranchConditional<RegisterType, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a != b; });
}

void RiscVIBlt(const Instruction *instruction) {
  BranchConditional<RegisterType, IntReg>(
      instruction, [](IntReg a, IntReg b) { return a < b; });
}

void RiscVIBltu(const Instruction *instruction) {
  BranchConditional<RegisterType, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a < b; });
}

void RiscVIBge(const Instruction *instruction) {
  BranchConditional<RegisterType, IntReg>(
      instruction, [](IntReg a, IntReg b) { return a >= b; });
}

void RiscVIBgeu(const Instruction *instruction) {
  BranchConditional<RegisterType, UIntReg>(
      instruction, [](UIntReg a, UIntReg b) { return a >= b; });
}

}  // namespace RV64

namespace RV32 {

using RegisterType = RV32Register;

void RiscVILd(const Instruction *instruction) {
  RVLoad<RegisterType, uint64_t>(instruction);
}

void RiscVILw(const Instruction *instruction) {
  RVLoad<RegisterType, int32_t>(instruction);
}

void RiscVILwChild(const Instruction *instruction) {
  RVLoadChild<RegisterType, int32_t>(instruction);
}

void RiscVILh(const Instruction *instruction) {
  RVLoad<RegisterType, int16_t>(instruction);
}

void RiscVILhChild(const Instruction *instruction) {
  RVLoadChild<RegisterType, int16_t>(instruction);
}

void RiscVILhu(const Instruction *instruction) {
  RVLoad<RegisterType, uint16_t>(instruction);
}

void RiscVILhuChild(const Instruction *instruction) {
  RVLoadChild<RegisterType, uint16_t>(instruction);
}

void RiscVILb(const Instruction *instruction) {
  RVLoad<RegisterType, int8_t>(instruction);
}

void RiscVILbChild(const Instruction *instruction) {
  RVLoadChild<RegisterType, int8_t>(instruction);
}

void RiscVILbu(const Instruction *instruction) {
  RVLoad<RegisterType, uint8_t>(instruction);
}

void RiscVILbuChild(const Instruction *instruction) {
  RVLoadChild<RegisterType, uint8_t>(instruction);
}

}  // namespace RV32

namespace RV64 {

using RegisterType = RV64Register;

void RiscVILd(const Instruction *instruction) {
  RVLoad<RegisterType, uint64_t>(instruction);
}

void RiscVILdChild(const Instruction *instruction) {
  RVLoadChild<RegisterType, int64_t>(instruction);
}

void RiscVILw(const Instruction *instruction) {
  RVLoad<RegisterType, int32_t>(instruction);
}

void RiscVILwChild(const Instruction *instruction) {
  RVLoadChild<RegisterType, int32_t>(instruction);
}

void RiscVILwu(const Instruction *instruction) {
  RVLoad<RegisterType, uint32_t>(instruction);
}

void RiscVILwuChild(const Instruction *instruction) {
  RVLoadChild<RegisterType, uint32_t>(instruction);
}

void RiscVILh(const Instruction *instruction) {
  RVLoad<RegisterType, int16_t>(instruction);
}

void RiscVILhChild(const Instruction *instruction) {
  RVLoadChild<RegisterType, int16_t>(instruction);
}

void RiscVILhu(const Instruction *instruction) {
  RVLoad<RegisterType, uint16_t>(instruction);
}

void RiscVILhuChild(const Instruction *instruction) {
  RVLoadChild<RegisterType, uint16_t>(instruction);
}

void RiscVILb(const Instruction *instruction) {
  RVLoad<RegisterType, int8_t>(instruction);
}

void RiscVILbChild(const Instruction *instruction) {
  RVLoadChild<RegisterType, int8_t>(instruction);
}

void RiscVILbu(const Instruction *instruction) {
  RVLoad<RegisterType, uint8_t>(instruction);
}

void RiscVILbuChild(const Instruction *instruction) {
  RVLoadChild<RegisterType, uint8_t>(instruction);
}

}  // namespace RV64

namespace RV32 {

using RegisterType = RV32Register;

void RiscVISd(const Instruction *instruction) {
  RVStore<RegisterType, uint64_t>(instruction);
}

void RiscVISw(const Instruction *instruction) {
  RVStore<RegisterType, uint32_t>(instruction);
}

void RiscVISh(const Instruction *instruction) {
  RVStore<RegisterType, uint16_t>(instruction);
}

void RiscVISb(const Instruction *instruction) {
  RVStore<RegisterType, uint8_t>(instruction);
}

}  // namespace RV32

namespace RV64 {

using RegisterType = RV64Register;

void RiscVISd(const Instruction *instruction) {
  RVStore<RegisterType, uint64_t>(instruction);
}

void RiscVISw(const Instruction *instruction) {
  RVStore<RegisterType, uint32_t>(instruction);
}

void RiscVISh(const Instruction *instruction) {
  RVStore<RegisterType, uint16_t>(instruction);
}

void RiscVISb(const Instruction *instruction) {
  RVStore<RegisterType, uint8_t>(instruction);
}

}  // namespace RV64

void RiscVIFence(const Instruction *instruction) {
  uint32_t bits = instruction->Source(0)->AsUint32(0);
  int fm = (bits >> 8) & 0xf;
  int predecessor = (bits >> 4) & 0xf;
  int successor = bits & 0xf;
  auto *state = static_cast<RiscVState *>(instruction->state());
  state->Fence(instruction, fm, predecessor, successor);
}

void RiscVIEcall(const Instruction *instruction) {
  auto *state = static_cast<RiscVState *>(instruction->state());
  state->ECall(instruction);
}

void RiscVIEbreak(const Instruction *instruction) {
  auto *state = static_cast<RiscVState *>(instruction->state());
  state->EBreak(instruction);
}

void RiscVWFI(const Instruction *instruction) {
  auto *state = static_cast<RiscVState *>(instruction->state());
  state->WFI(instruction);
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
