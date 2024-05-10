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

#include "riscv/riscv_zicsr_instructions.h"

#include <cstdint>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/register.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::operator*;  // NOLINT: is used below (clang error).

template <typename T>
inline T ReadCsr(RiscVCsrInterface *) {}

template <>
inline uint32_t ReadCsr<uint32_t>(RiscVCsrInterface *csr) {
  return csr->AsUint32();
}
template <>
inline uint64_t ReadCsr<uint64_t>(RiscVCsrInterface *csr) {
  return csr->AsUint64();
}

// Helper function to check that the CSR permission is valid. If not, throw
// an illegal instruction exception.
bool CheckCsrPermission(int csr_index, Instruction *instruction) {
  auto required_mode = (csr_index >> 8) & 0x3;
  auto *state = static_cast<RiscVState *>(instruction->state());
  auto current_mode = state->privilege_mode();
  // If the privilege mode is too low, throw an exception.
  if (*current_mode < required_mode) {
    state->Trap(/*is_interrupt*/ false, 0, *ExceptionCode::kIllegalInstruction,
                instruction->address(), instruction);
    return false;
  }
  // Check for any special case CSRs that have special access conditions.
  // Check for access to satp.
  if (current_mode == PrivilegeMode::kSupervisor &&
      (csr_index == *RiscVCsrEnum::kSAtp) && (state->mstatus()->tvm())) {
    state->Trap(/*is_interrupt*/ false, 0, *ExceptionCode::kIllegalInstruction,
                instruction->address(), instruction);
    return false;
  }
  return true;
}

// Templated helper functions.

// Read the CSR, write a new value back.
template <typename T>
static inline void RVZiCsrrw(Instruction *instruction) {
  // Get a handle to the state instance.
  auto state = static_cast<RiscVState *>(instruction->state());
  // Get the csr index.
  int csr_index = instruction->Source(1)->AsInt32(0);
  if (!CheckCsrPermission(csr_index, instruction)) return;
  // Read the csr.
  auto result = state->csr_set()->GetCsr(csr_index);
  if (!result.ok()) {
    // Signal error if it failed.
    LOG(ERROR) << absl::StrCat("Instruction at address 0x",
                               absl::Hex(instruction->address()),
                               " failed to read CSR 0x", absl::Hex(csr_index),
                               ": ", result.status().message());
    return;
  }
  // Get the new value.
  T new_value = generic::GetInstructionSource<T>(instruction, 0);
  auto *csr = result.value();
  // Update the register.
  auto *reg = static_cast<generic::RegisterDestinationOperand<T> *>(
                  instruction->Destination(0))
                  ->GetRegister();
  reg->data_buffer()->template Set<T>(0, ReadCsr<T>(csr));
  // Write the new value to the csr.
  csr->Write(new_value);
}

// Read the CSR, set the bits specified by the new value and write back.
template <typename T>
static inline void RVZiCsrrs(Instruction *instruction) {
  // Get a handle to the state instance.
  auto state = static_cast<RiscVState *>(instruction->state());
  // Get the csr index.
  int csr_index = instruction->Source(1)->AsInt32(0);
  if (!CheckCsrPermission(csr_index, instruction)) return;
  // Read the csr.
  auto result = state->csr_set()->GetCsr(csr_index);
  if (!result.ok()) {
    // Signal error if it failed.
    LOG(ERROR) << absl::StrCat("Instruction at address 0x",
                               absl::Hex(instruction->address()),
                               " failed to read CSR 0x", absl::Hex(csr_index),
                               ": ", result.status().message());
    return;
  }
  // Get the new value.
  T new_value = generic::GetInstructionSource<T>(instruction, 0);
  auto *csr = result.value();
  // Update the register.
  auto *reg = static_cast<generic::RegisterDestinationOperand<T> *>(
                  instruction->Destination(0))
                  ->GetRegister();
  reg->data_buffer()->template Set<T>(0, ReadCsr<T>(csr));
  // Write the new value to the csr.
  csr->SetBits(new_value);
}

// Read the CSR, clear the bits specified by the new value and write back.
template <typename T>
static inline void RVZiCsrrc(Instruction *instruction) {
  // Get a handle to the state instance.
  auto state = static_cast<RiscVState *>(instruction->state());
  // Get the csr index.
  int csr_index = instruction->Source(1)->AsInt32(0);
  if (!CheckCsrPermission(csr_index, instruction)) return;
  // Read the csr.
  auto result = state->csr_set()->GetCsr(csr_index);
  if (!result.ok()) {
    // Signal error if it failed.
    LOG(ERROR) << absl::StrCat("Instruction at address 0x",
                               absl::Hex(instruction->address()),
                               " failed to read CSR 0x", absl::Hex(csr_index),
                               ": ", result.status().message());
    return;
  }
  // Get the new value.
  T new_value = generic::GetInstructionSource<T>(instruction, 0);
  auto *csr = result.value();
  // Write the current value of the CSR to the destination register.
  auto *reg = static_cast<generic::RegisterDestinationOperand<T> *>(
                  instruction->Destination(0))
                  ->GetRegister();
  auto csr_val = ReadCsr<T>(csr);
  reg->data_buffer()->template Set<T>(0, csr_val);
  // Write the new value to the csr.
  csr->ClearBits(new_value);
}

// Do not read the CSR, just write the new value back.
template <typename T>
static inline void RVZiCsrrwNr(Instruction *instruction) {
  // Get a handle to the state instance.
  auto state = static_cast<RiscVState *>(instruction->state());
  // Get the csr index.
  int csr_index = instruction->Source(1)->AsInt32(0);
  if (!CheckCsrPermission(csr_index, instruction)) return;
  // Read the csr.
  auto result = state->csr_set()->GetCsr(csr_index);
  if (!result.ok()) {
    LOG(ERROR) << absl::StrCat("Instruction at address 0x",
                               absl::Hex(instruction->address()),
                               " failed to write CSR 0x", absl::Hex(csr_index),
                               ": ", result.status().message());
    return;
  }
  auto *csr = result.value();
  // Write the new value to the csr.
  T new_value = generic::GetInstructionSource<T>(instruction, 0);
  csr->Write(new_value);
}

// Do not write a value back to the CSR, just read it.
template <typename T>
static inline void RVZiCsrrNw(Instruction *instruction) {
  // Get a handle to the state instance.
  auto state = static_cast<RiscVState *>(instruction->state());
  // Get the csr index.
  int csr_index = instruction->Source(0)->AsInt32(0);
  if (!CheckCsrPermission(csr_index, instruction)) return;
  // Read the csr.
  auto result = state->csr_set()->GetCsr(csr_index);
  if (!result.ok()) {
    LOG(ERROR) << absl::StrCat("Instruction at address 0x",
                               absl::Hex(instruction->address()),
                               " failed to read CSR 0x", absl::Hex(csr_index),
                               ": ", result.status().message());
    return;
  }
  // Get the CSR object.
  auto *csr = result.value();
  auto *reg = static_cast<generic::RegisterDestinationOperand<T> *>(
                  instruction->Destination(0))
                  ->GetRegister();
  reg->data_buffer()->template Set<T>(0, ReadCsr<T>(csr));
}

namespace RV32 {

using RegisterType = RV32Register;
using UintReg = RegisterType::ValueType;

// Read the CSR, write a new value back.
void RiscVZiCsrrw(Instruction *instruction) { RVZiCsrrw<UintReg>(instruction); }

// Read the CSR, set the bits specified by the new value and write back.
void RiscVZiCsrrs(Instruction *instruction) { RVZiCsrrs<UintReg>(instruction); }

// Read the CSR, clear the bits specified by the new value and write back.
void RiscVZiCsrrc(Instruction *instruction) { RVZiCsrrc<UintReg>(instruction); }

// Do not read the CSR, just write the new value back.
void RiscVZiCsrrwNr(Instruction *instruction) {
  RVZiCsrrwNr<UintReg>(instruction);
}

// Do not write a value back to the CSR, just read it.
void RiscVZiCsrrNw(Instruction *instruction) {
  RVZiCsrrNw<UintReg>(instruction);
}

}  // namespace RV32

namespace RV64 {

using RegisterType = RV64Register;
using UintReg = RegisterType::ValueType;

// Read the CSR, write a new value back.
void RiscVZiCsrrw(Instruction *instruction) { RVZiCsrrw<UintReg>(instruction); }

// Read the CSR, set the bits specified by the new value and write back.
void RiscVZiCsrrs(Instruction *instruction) { RVZiCsrrs<UintReg>(instruction); }

// Read the CSR, clear the bits specified by the new value and write back.
void RiscVZiCsrrc(Instruction *instruction) { RVZiCsrrc<UintReg>(instruction); }

// Do not read the CSR, just write the new value back.
void RiscVZiCsrrwNr(Instruction *instruction) {
  RVZiCsrrwNr<UintReg>(instruction);
}

// Do not write a value back to the CSR, just read it.
void RiscVZiCsrrNw(Instruction *instruction) {
  RVZiCsrrNw<UintReg>(instruction);
}

}  // namespace RV64

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
