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

#ifndef RISCV_RISCV_VECTOR_MEMORY_INSTRUCTIONS_H_
#define RISCV_RISCV_VECTOR_MEMORY_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

// This file declares the semantic functions used to implement RiscV vector
// load store instructions.

namespace mpact {
namespace sim {
namespace riscv {

using Instruction = ::mpact::sim::generic::Instruction;

// Set vector length. rd/rs1_ zero is true if the corresponding operand is
// register x0.
// The instruction takes 2 instruction source scalar operands: source operand 0
// is the requested vector length, source operand 1 is the requested vector
// configuration value. The destination operand is a scalar.
void Vsetvl(bool rd_zero, bool rs1_zero, const Instruction *inst);

// Vector load semantic functions.

// Load with constant stride, the parameter specifies the width of the vector
// elements. This instruction takes 3 source and 1 destination operands. Source
// operand 0 is a scalar base address, source operand 1 is a scalar stride,
// source operand 2 is the vector mask (either a vector register, or a
// constant). Destination operand 0 is assigned to the child instruction and
// is a vector register (group).
void VlStrided(int element_width, const Instruction *inst);
// Load vector mask. This instruction takes 1 source and 1 destination operand.
// The source operand is a scalar base address, the destination operand is the
// vector register to write the mask to.
void Vlm(const Instruction *inst);
// Indexed vector load (ordered and unordered). This instruction takes 3 source
// and 1 destination operands. Source operand 0 is a scalar base address, source
// operand 1 is a vector register (group) of indices, source operand 2 is the
// vector mask. Destination operand 0 is assigned to the child instruction and
// is a vector register (group).
void VlIndexed(int index_width, const Instruction *inst);
// Load vector register(s). Takes a parameter specifying how many registers to
// load. This instruction takes 1 source and 1 destination operand. Source
// operand 0 is a scalar base address. Destination operand 0 is assigned to the
// child instruction and is a vector register (group).
void VlRegister(int num_regs, const Instruction *inst);
// Child instruction semantic functions for non-segment loads responsible for
// writing load data back to the target register(s). It takes a single
// destination operand. Destination operand 0 is a vector register (group).
void VlChild(const Instruction *inst);
// Load segment, unit stride. The function takes one parameter that specifies
// the element width. The instruction takes 3 source operands and 1 destination
// operand. Source operand 0 is a scalar base address, source operand 1 is
// the vector mask, and source operand 2 is the number of fields - 1.
// Destination operand 0 is assigned to the child instruction and is a vector
// register (group).
void VlSegment(int element_width, const Instruction *inst);
// Load segment strided. The function takes one parameter that specifies
// the element width. The instruction takes 4 source operands and 1 destination
// operand. Source operand 0 is a scalar base address, source operand 1 is a
// scalar stride, source operand 2 is the vector mask, and source operand 3 is
// the number of fields - 1. Destination operand 0 is assigned to the child
// instruction and is a vector register (group).
void VlSegmentStrided(int element_width, const Instruction *inst);
// Load segment indexed. The function takes one parameter that specifies
// the index element width. The instruction takes 4 source operands and 1
// destination operand. Source operand 0 is a scalar base address, source
// operand 1 is a vector register (group) of indices, source operand 2 is the
// vector mask, and source operand 3 is the number of fields - 1. Destination
// operand 0 is assigned to the child instruction and is a vector register
// (group).
void VlSegmentIndexed(int index_width, const Instruction *inst);
// Child instruction semantic functions for segment loads responsible for
// writing load data back to the target register(s). It takes a single
// destination operand. Destination operand 0 is a vector register (group).
void VlSegmentChild(int element_width, const Instruction *inst);

// Vector store semantic functions.

// Store strided. The function takes one parameter that specifies the element
// width. The instruction takes 4 source parameters. Source 0 is the store data
// vector register (group), source 1 is the scalar base address, source 2 is the
// stride, and source 3 is the vector mask.
void VsStrided(int element_width, const Instruction *inst);
// Store vector mask. This instruction takes 2 source operands. Source 0 is the
// vector mask register to be stored, the second is the scalar base address.
void Vsm(const Instruction *inst);
// Store indexed. The function takes one parameter that specifies the element
// width. The instruction takes 4 source parameters. Source 0 is the store data
// vector register (group), source 1 is a vector (group) of indices, source 2 is
// the stride, and source 3 is the vector mask.
void VsIndexed(int index_width, const Instruction *inst);
// Store vector register (group). This function takes one parameter that
// specifies the number of registers to store. The instruction takes 2 source
// operands. Source 0 is the source vector register (group), the second is the
// scalar base address.
void VsRegister(int num_regs, const Instruction *inst);
// Store segment, unit stride. The function takes one parameter that specifies
// the element width. The instruction takes 4 source operands. Source operand 0
// is the store data, source operand 1 is the scalar base address, source
// operand 2 is the vector mask, and source operand 3 is the number of fields
// - 1.
void VsSegment(int element_width, const Instruction *inst);
// Store segment, unit stride. The function takes one parameter that specifies
// the element width. The instruction takes 5 source operands. Source operand 0
// is the store data, source operand 1 is the scalar base address, source
// operand 2 is the segment stride, source operand 3 is the vector mask, and
// source operand 4 is the number of fields
// - 1.
void VsSegmentStrided(int element_width, const Instruction *inst);
// Load segment indexed. The function takes one parameter that specifies
// the index element width. The instruction takes 5 source operands. Source
// operand 0 is the store data, source operand 1 is a scalar base address,
// source operand 2 is a vector register (group) of indices, source operand 3 is
// the vector mask, and source operand 4 is the number of fields - 1.
void VsSegmentIndexed(int index_width, const Instruction *inst);

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // RISCV_RISCV_VECTOR_MEMORY_INSTRUCTIONS_H_
