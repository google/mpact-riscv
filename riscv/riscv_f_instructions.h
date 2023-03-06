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

#ifndef RISCV_RISCV_F_INSTRUCTIONS_H_
#define RISCV_RISCV_F_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

// This file declares the semantic functions that implement the scalar
// single precision floating point instructions in the RiscV architecture.

namespace mpact {
namespace sim {
namespace riscv {

using generic::Instruction;

// Single precision arithmetic instructions each take 3 source operands and
// 1 destination register operand. Source 0 and 1 are the data sources to the
// operation, source 2 is the rounding mode specifier in the instruction.
void RiscVFAdd(const Instruction *instruction);
void RiscVFSub(const Instruction *instruction);
void RiscVFMul(const Instruction *instruction);
void RiscVFDiv(const Instruction *instruction);
// Single precision load child instruction takes a single destination operand.
void RiscVIFlwChild(const Instruction *instruction);
// Single precision square root takes 2 source operands, the source register
// and the rounding mode, and 1 destination register operand.
void RiscVFSqrt(const Instruction *instruction);
// The single precision Min and Max instructions each take 2 source register
// operands and 1 destination register operand.
void RiscVFMin(const Instruction *instruction);
void RiscVFMax(const Instruction *instruction);
// The following four multiply-accumulate instruction each take 3 source
// register operands (0, 1, 2) and the rounding mode (3), and one destination
// register operand.
void RiscVFMadd(const Instruction *instruction);
void RiscVFMsub(const Instruction *instruction);
void RiscVFNmadd(const Instruction *instruction);
void RiscVFNmsub(const Instruction *instruction);
// The single precision sign manipulation instructions take 2 source register
// operands and a single destination register operand.
void RiscVFSgnj(const Instruction *instruction);
void RiscVFSgnjn(const Instruction *instruction);
void RiscVFSgnjx(const Instruction *instruction);
// The single precision conversion instructions each take 1 source register
// operand, the rounding mode, and 1 destination register operand.
void RiscVFCvtSw(const Instruction *instruction);
void RiscVFCvtSwu(const Instruction *instruction);
void RiscVFCvtSl(const Instruction *instruction);
void RiscVFCvtSlu(const Instruction *instruction);
// The move instruction takes a single register source operand and a single
void RiscVFMvwx(const Instruction *instruction);

namespace RV32 {

// Store float instruction semantic function, source operand 0 is the base
// register, source operand 1 is the offset, while source operand 2 is the value
// to be stored referred to by rs2.
void RiscVFSw(const Instruction *instruction);
// Single precision load child instruction takes a single destination operand.
// The single precision conversion instructions each take 1 source register
// operand, the rounding mode, and 1 destination register operand.
void RiscVFCvtWs(const Instruction *instruction);
void RiscVFCvtWus(const Instruction *instruction);
// The move instruction takes a single register source operand and a single
// destination register operand.
void RiscVFMvxw(const Instruction *instruction);
// The single precision compare instructions take 2 source register operands
// and a single destination register operand.
void RiscVFCmpeq(const Instruction *instruction);
void RiscVFCmplt(const Instruction *instruction);
void RiscVFCmple(const Instruction *instruction);
// The single precision class instruction takes a single source register operand
// and a single destination register operand.
void RiscVFClass(const Instruction *instruction);

}  // namespace RV32

namespace RV64 {

// Store float instruction semantic function, source operand 0 is the base
// register, source operand 1 is the offset, while source operand 2 is the value
// to be stored referred to by rs2.
void RiscVFSw(const Instruction *instruction);
// Single precision load child instruction takes a single destination operand.
// The single precision conversion instructions each take 1 source register
// operand, the rounding mode, and 1 destination register operand.
void RiscVFCvtWs(const Instruction *instruction);
void RiscVFCvtWus(const Instruction *instruction);
void RiscVFCvtLs(const Instruction *instruction);
void RiscVFCvtLus(const Instruction *instruction);
// The move instruction takes a single register source operand and a single
// destination register operand.
void RiscVFMvxw(const Instruction *instruction);
// The single precision compare instructions take 2 source register operands
// and a single destination register operand.
void RiscVFCmpeq(const Instruction *instruction);
void RiscVFCmplt(const Instruction *instruction);
void RiscVFCmple(const Instruction *instruction);
// The single precision class instruction takes a single source register operand
// and a single destination register operand.
void RiscVFClass(const Instruction *instruction);

}  // namespace RV64

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // RISCV_RISCV_F_INSTRUCTIONS_H_
