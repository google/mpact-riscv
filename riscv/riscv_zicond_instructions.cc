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

#include "riscv/riscv_zicond_instructions.h"

#include "mpact/sim/generic/instruction.h"
#include "riscv/riscv_instruction_helpers.h"
#include "riscv/riscv_register.h"

namespace mpact::sim::riscv {

namespace RV32 {

using RegType = RV32Register;
using RegValue = typename RegType::ValueType;

// If rs2 is zero, set the destination register to 0, otherwise copy rs1 to
// the destination register.
void RiscVCzeroEqz(const generic::Instruction* inst) {
  RiscVBinaryOp<RegType, RegValue, RegValue>(
      inst, [](RegValue rs1, RegValue rs2) { return rs2 == 0 ? 0 : rs1; });
}

// If rs2 is non-zero, set the destination register to 0, otherwise copy rs1 to
// the destination register.
void RiscVCzeroNez(const generic::Instruction* inst) {
  RiscVBinaryOp<RegType, RegValue, RegValue>(
      inst, [](RegValue rs1, RegValue rs2) { return rs2 != 0 ? 0 : rs1; });
}

}  // namespace RV32

namespace RV64 {

using RegType = RV64Register;
using RegValue = typename RegType::ValueType;

// If rs2 is zero, set the destination register to 0, otherwise copy rs1 to
// the destination register.
void RiscVCzeroEqz(const generic::Instruction* inst) {
  RiscVBinaryOp<RegType, RegValue, RegValue>(
      inst, [](RegValue rs1, RegValue rs2) { return rs2 == 0 ? 0 : rs1; });
}

// If rs2 is non-zero, set the destination register to 0, otherwise copy rs1 to
// the destination register.
void RiscVCzeroNez(const generic::Instruction* inst) {
  RiscVBinaryOp<RegType, RegValue, RegValue>(
      inst, [](RegValue rs1, RegValue rs2) { return rs2 != 0 ? 0 : rs1; });
}

}  // namespace RV64

}  // namespace mpact::sim::riscv
