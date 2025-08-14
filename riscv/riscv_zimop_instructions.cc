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

#include "riscv/riscv_zimop_instructions.h"

#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/register.h"
#include "riscv/riscv_register.h"

namespace mpact::sim::riscv {

using ::mpact::sim::generic::Instruction;

namespace RV32 {

using RegType = RV32Register;
using RegValue = typename RegType::ValueType;

void RiscVMop(const Instruction* inst) {
  // Get the destination register and set it to zero.
  auto* reg = static_cast<generic::RegisterDestinationOperand<RegValue>*>(
                  inst->Destination(0))
                  ->GetRegister();
  reg->data_buffer()->template Set<RegValue>(0, 0);
}

}  // namespace RV32

namespace RV64 {

using RegType = RV64Register;
using RegValue = typename RegType::ValueType;

void RiscVMop(const Instruction* inst) {
  // Get the destination register and set it to zero.
  auto* reg = static_cast<generic::RegisterDestinationOperand<RegValue>*>(
                  inst->Destination(0))
                  ->GetRegister();
  reg->data_buffer()->template Set<RegValue>(0, 0);
}

}  // namespace RV64

}  // namespace mpact::sim::riscv
