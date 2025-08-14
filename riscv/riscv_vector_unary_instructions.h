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

#ifndef MPACT_RISCV_RISCV_RISCV_VECTOR_UNARY_INSTRUCTIONS_H_
#define MPACT_RISCV_RISCV_RISCV_VECTOR_UNARY_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

// This file declares the vector instruction semantic functions for the integer
// unary vector instructions.
namespace mpact {
namespace sim {
namespace riscv {

using Instruction = ::mpact::sim::generic::Instruction;

// VWXUNARY0
// Moves a value from index 0 element of a vector register to a scalar register.
// This instruction takes 1 source and 1 destination. One is a vector register,
// the other is a scalar (x) register.
void VmvToScalar(Instruction* inst);
// Moves a scalar to index 0 element of a vector register. This instruction
// takes 1 source and 1 destination. One is a vector register, the other is
// a scalar (x) register.
void VmvFromScalar(Instruction* inst);
// Does a population count on the vector mask in the source operand 0 (subject
// to masking), and writes the result to the scalar register in the destination
// operand. Operand 1 is a mask register. Only vlen bits are considered.
void Vcpop(Instruction* inst);
// Computes the index of the first set bit of the vector mask in the source
// operand 0 (subject ot masking), and writes the result to the scalar register
// in the destination operand. Operand 1 is a mask register. Only vlen bits
// are considered.
void Vfirst(Instruction* inst);

// VXUNARY0
// Element wide zero extend from SEW/2 to SEW. This instruction takes two source
// operands, and a vector destination operand. Source 0 is the vs2 vector
// source, and source 1 is a vector mask operand.
void Vzext2(Instruction* inst);
// Element wide sign extend from SEW/2 to SEW. This instruction takes two source
// operands, and a vector destination operand. Source 0 is the vs2 vector
// source, and source 1 is a vector mask operand.
void Vsext2(Instruction* inst);
// Element wide zero extend from SEW/4 to SEW. This instruction takes two source
// operands, and a vector destination operand. Source 0 is the vs2 vector
// source, and source 1 is a vector mask operand.
void Vzext4(Instruction* inst);
// Element wide sign extend from SEW/4 to SEW. This instruction takes two source
// operands, and a vector destination operand. Source 0 is the vs2 vector
// source, and source 1 is a vector mask operand.
void Vsext4(Instruction* inst);
// Element wide zero extend from SEW/8 to SEW. This instruction takes two source
// operands, and a vector destination operand. Source 0 is the vs2 vector
// source, and source 1 is a vector mask operand.
void Vzext8(Instruction* inst);
// Element wide sign extend from SEW/8 to SEW. This instruction takes two source
// operands, and a vector destination operand. Source 0 is the vs2 vector
// source, and source 1 is a vector mask operand.
void Vsext8(Instruction* inst);

// VMUNARY0
// Set before first mask bit. Takes a vector mask stored in a vector register
// and produces a mask register with all active bits before the first set active
// bit in the source mask set to 1. This instruction takes one vector source
// operands, a mask register, and a vector destination operand. Source 0 is the
// vs2 register source, source 1 is the vector mask operand.
void Vmsbf(Instruction* inst);
// Set only first mask bit. Takes a vector mask stored in a vector register
// and produces a mask register with only the bit set that corresponds to the
// the first set active bit in the source mask set to 1. This instruction takes
// one vector source operands, a mask register, and a vector destination
// operand. Source 0 is the vs2 register source, source 1 is the vector mask
// operand.
void Vmsof(Instruction* inst);
// Set including first mask bit. Takes a vector mask stored in a vector register
// and produces a mask register with all active bits before and including the
// first set active bit in the source mask set to 1. This instruction takes one
// vector source operands, a mask register, and a vector destination operand.
// Source 0 is the vs2 register source, source 1 is the vector mask operand.
void Vmsif(Instruction* inst);
// Vector Iota instruction. Takes a vector mask stored in a vector register
// and writes to each element of the destination vector register group the sum
// of all bits of elements in the mask register whose index is less than the
// element (parallel prefix sum). This instruction takes two sources and one
// destination. Source 0 is the vs2 register source, source 1 is the vector
// mask operand.
void Viota(Instruction* inst);
// Vector element index instruction. Writes the element index to the destination
// vector element group (masking does not change the value written to active
// elements, only which elements are written to). This instruction takes 1
// source (mask register) and one destination.
void Vid(Instruction* inst);
}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_VECTOR_UNARY_INSTRUCTIONS_H_
