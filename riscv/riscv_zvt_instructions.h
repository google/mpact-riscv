// Copyright 2026 Google LLC
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

// Semantic functions for the RISC-V Vector Matrix Extension (Zvt / "VME",
// spec v0.3). This header declares the matrix configuration instructions
// (Zvtbase, spec 15.1.1.4). Tile move/load/store and the matrix-multiply
// instructions are added in their respective translation units.

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_ZVT_INSTRUCTIONS_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_ZVT_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

namespace mpact::sim::riscv {

using ::mpact::sim::generic::Instruction;

// KMAX (max inner dot-product depth) as a function of SEW in bits.
// Per spec 15.1.1.4: KMAX = max(1, 32 / SEW). int8->4, bf16(16)->2, fp32->1.
inline int RiscVZvtKmax(int sew_bits) {
  return sew_bits >= 32 ? 1 : (32 / sew_bits);
}

// Matrix configuration instructions (Zvtbase). All write the constrained
// result back to the scalar destination register where applicable.
// In the Zvt extension:
//  - M represents the number of rows of the accumulator tile.
//  - N represents the number of columns of the accumulator tile.
//  - K represents the inner dimension of the dot product.
//
//   msetmtype rs1, rs2 : sets mtype (rs1) and vtype (rs2), zeroes vl.
void RiscVZvtMsetMtype(const Instruction* inst);
//   msettn rd, rs1     : sets tn (== vl), bounded by the matrix config. (N is
//   num columns)
void RiscVZvtMsetTn(const Instruction* inst);
//   msettm rd, rs1     : sets mtype.tm. (M is num rows)
void RiscVZvtMsetTm(const Instruction* inst);
//   msettk rd, rs1     : sets mtype.tk. (K is inner dimension)
void RiscVZvtMsetTk(const Instruction* inst);

// Tile manipulation instructions (Zvtbase).
//   vtzero mtd         : zeroes the tm x tn submatrix of tile mtd.
void RiscVZvtVtZero(const Instruction* inst);
//   vtdiscard          : hints the tile state is dead (no-op in the ISS).
void RiscVZvtVtDiscard(const Instruction* inst);
//   vtmv.v.t vd, rs1   : moves a tile row/col subset (TSS in rs1) to vd.
void RiscVZvtVtmvVt(const Instruction* inst);
//   vtmv.t.v rs1, vs2  : moves vs2 to a tile row/col subset (TSS in rs1).
void RiscVZvtVtmvTv(const Instruction* inst);
//   vtle{8,16,32,64} rs2,(rs1) : loads a tile subset (TSS rs2) from memory.
void RiscVZvtVtle(int eew_bits, const Instruction* inst);
//   vtse{8,16,32,64} rs2,(rs1) : stores a tile subset (TSS rs2) to memory.
void RiscVZvtVtse(int eew_bits, const Instruction* inst);

// Matrix-multiply instructions: C[m,n] += sum_k A[k,m] * B[k,n], with C in a
// tile and A/B sourced from vector register groups (spec 15.1.1.8).
// M is the number of rows of C, N is the number of columns of C, and
// K is the inner dimension of the dot product.
enum class RiscVZvtMatmulKind {
  kInt8Signed,    // vtmms.tvv: int8 x int8 -> int32.
  kInt8Unsigned,  // vtmmu.tvv: uint8 x uint8 -> int32.
  kFp32,          // vtfmm.tvv: fp32 x fp32 -> fp32.
  kBf16,          // vtfmm.alt.tvv: bf16 x bf16 -> fp32.
};
void RiscVZvtMatmul(RiscVZvtMatmulKind kind, const Instruction* inst);

}  // namespace mpact::sim::riscv

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_ZVT_INSTRUCTIONS_H_
