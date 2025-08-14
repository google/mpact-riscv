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

#ifndef MPACT_RISCV_RISCV_RISCV_A_INSTRUCTIONS_H_
#define MPACT_RISCV_RISCV_RISCV_A_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::Instruction;

// This file declares the set of semantic functions used in the A (or atomic)
// subset of the RiscV instruction set. The *w functions are used in the
// 32 bit version, and both *w and *d are used in the 64 bit version of RiscV.

// The following instruction takes 3 source operands and one destination
// operand. The first source is the register holding the address of the
// memory to access, the second is the acquire bit, and the third, the release
// bit. The instruction has a single destination operand which is the register
// to write the result to.
void ALrw(Instruction* instruction);
// The following instructions take 4 source operands and one destination
// operand. The first source is the register holding the address of the
// memory to access, the second is the register value used in the memory
// operation (to store, swap, etc), the third is the acquire bit, and the
// fourth, the release bit. The instruction has a single destination operand
// which is the register to write the result to.
void AScw(Instruction* instruction);
void AAmoswapw(Instruction* instruction);
void AAmoaddw(Instruction* instruction);
void AAmoandw(Instruction* instruction);
void AAmoorw(Instruction* instruction);
void AAmoxorw(Instruction* instruction);
void AAmomaxw(Instruction* instruction);
void AAmomaxuw(Instruction* instruction);
void AAmominw(Instruction* instruction);
void AAmominuw(Instruction* instruction);

// The following instruction takes 3 source operands and one destination
// operand. The first source is the register holding the address of the
// memory to access, the second is the acquire bit, and the third, the release
// bit. The instruction has a single destination operand which is the register
// to write the result to.
void ALrd(Instruction* instruction);
// The following instructions take 4 source operands and one destination
// operand. The first source is the register holding the address of the
// memory to access, the second is the register value used in the memory
// operation (to store, swap, etc), the third is the acquire bit, and the
// fourth, the release bit. The instruction has a single destination operand
// which is the register to write the result to.
void AScd(Instruction* instruction);
void AAmoswapd(Instruction* instruction);
void AAmoaddd(Instruction* instruction);
void AAmoandd(Instruction* instruction);
void AAmoord(Instruction* instruction);
void AAmoxord(Instruction* instruction);
void AAmomaxd(Instruction* instruction);
void AAmomaxud(Instruction* instruction);
void AAmomind(Instruction* instruction);
void AAmominud(Instruction* instruction);

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_A_INSTRUCTIONS_H_
