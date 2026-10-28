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
// spec v0.4). This header declares the matrix configuration instructions
// (Zvtbase, spec 15.1.1.4). Tile move/load/store and the matrix-multiply
// instructions are added in their respective translation units.

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_ZVT_INSTRUCTIONS_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_ZVT_INSTRUCTIONS_H_

#include <cstdint>

#include "mpact/sim/generic/instruction.h"

namespace mpact::sim::riscv {

using ::mpact::sim::generic::Instruction;

// Architectural mstatus field shifts and bitmasks (RISC-V Privileged,
// Vector, and Matrix specs).
inline constexpr int kMstatusVsShift = 9;
inline constexpr uint64_t kMstatusVsMask = 0x3ULL << kMstatusVsShift;
inline constexpr int kMstatusFsShift = 13;
inline constexpr uint64_t kMstatusFsMask = 0x3ULL << kMstatusFsShift;
inline constexpr int kMstatusMsShift = 29;
inline constexpr uint64_t kMstatusMsMask = 0x3ULL << kMstatusMsShift;
inline constexpr int kMstatusSdShiftRv32 = 31;
inline constexpr uint64_t kMstatusSdMaskRv32 = 1ULL << kMstatusSdShiftRv32;
inline constexpr int kMstatusSdShiftRv64 = 63;
inline constexpr uint64_t kMstatusSdMaskRv64 = 1ULL << kMstatusSdShiftRv64;

// mstatus field state values (Off = 0, Initial = 1, Clean = 2, Dirty = 3).
inline constexpr uint32_t kMstatusStateOff = 0;
inline constexpr uint32_t kMstatusStateInitial = 1;
inline constexpr uint32_t kMstatusStateClean = 2;
inline constexpr uint32_t kMstatusStateDirty = 3;

// Returns KMAX (maximum inner dot-product depth) as a function of 'sew_bits'.
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

// Sets mtype (from rs1) and vtype (from rs2), and clears vl to 0.
//
// Source operands:
//   0: rs1: Scalar register holding requested mtype configuration.
//   1: rs2: Scalar register holding requested vtype configuration.
// Destination operands:
//   None. Directly updates matrix state, vector state, and CSRs.
void RiscVZvtMsetMtype(const Instruction* inst);

// Configures mtwiden, vsew, and altfmt from immediate instruction fields,
// and clears vl to 0.
//
// Source operands:
//   0: mtwiden: Immediate operand containing the matrix widening mode.
//   1: vsew: Immediate operand containing selected element width encoding.
//   2: altfmt: Immediate operand selecting standard vs alternate datatypes.
// Destination operands:
//   None. Directly updates matrix state, vector state, and CSRs.
void RiscVZvtMsetMtypei(const Instruction* inst);

// Sets the matrix column dimension (TN) and vector length (vl) bounded by
// the current matrix configuration.
//
// Source operands:
//   0: rs1: Scalar register specifying requested column dimension.
// Destination operands:
//   0: rd: Scalar register returning actual granted column dimension / vl.
void RiscVZvtMsetTn(const Instruction* inst);

// Sets the matrix row dimension (TM) bounded by the current matrix
// configuration.
//
// Source operands:
//   0: rs1: Scalar register specifying requested row dimension.
// Destination operands:
//   0: rd: Scalar register returning actual granted row dimension.
void RiscVZvtMsetTm(const Instruction* inst);

// Sets the matrix dot-product depth (TK) bounded by KMAX.
//
// Source operands:
//   0: rs1: Scalar register specifying requested dot-product depth.
// Destination operands:
//   0: rd: Scalar register returning actual granted dot-product depth.
void RiscVZvtMsetTk(const Instruction* inst);

// Tile manipulation instructions (Zvtbase).

// Zeroes the active TM x TN submatrix of the specified destination tile.
//
// Source operands:
//   0: mtile: Immediate operand specifying destination tile index.
// Destination operands:
//   None. Directly clears tile storage in matrix state.
void RiscVZvtVtZero(const Instruction* inst);

// Hints to the execution unit that active tile resources may be discarded.
//
// Source operands:
//   None.
// Destination operands:
//   None.
void RiscVZvtVtDiscard(const Instruction* inst);

// Moves a row or column subset of a tile into a vector destination register
// group.
//
// Source operands:
//   0: rs1: Scalar register holding the Tile Subset Specifier (TSS).
// Destination operands:
//   0: matVd: Vector register group receiving the extracted tile slice.
void RiscVZvtVtmvVt(const Instruction* inst);

// Moves data from a vector source register group into a row or column tile
// subset.
//
// Source operands:
//   0: matA: Vector register group holding the source data to write.
//   1: rs1: Scalar register holding the destination Tile Subset Specifier
//      (TSS).
// Destination operands:
//   None. Updates targeted tile elements in matrix state directly.
void RiscVZvtVtmvTv(const Instruction* inst);

// Loads a tile row/column subset from memory with explicit element width.
//
// Bound arguments:
//   eew_bits: Effective element width in bits (8, 16, 32, or 64).
// Source operands:
//   0: rs1: Scalar base address register.
//   1: rs2: Scalar register holding the Tile Subset Specifier (TSS).
// Destination operands:
//   None. Updates targeted tile storage directly from memory.
void RiscVZvtVtle(int eew_bits, const Instruction* inst);

// Stores a tile row/column subset to memory with explicit element width.
//
// Bound arguments:
//   eew_bits: Effective element width in bits (8, 16, 32, or 64).
// Source operands:
//   0: rs1: Scalar base address register.
//   1: rs2: Scalar register holding the Tile Subset Specifier (TSS).
// Destination operands:
//   None. Reads tile elements and writes directly to memory.
void RiscVZvtVtse(int eew_bits, const Instruction* inst);

// Specifies the element data types and accumulation mode for
// matrix-multiplication operations (spec 15.1.1.8):
//   C[m,n] += sum_k A[k,m] * B[k,n],
// with C in a tile and A/B sourced from vector register groups.
// M is the number of rows of C, N is the number of columns of C, and
// K is the inner dimension of the dot product.
enum class RiscVZvtMatmulKind {
  kInt8Signed,          // vtmms.tvv (altfmt=1): int8 x int8 -> int32.
  kInt8Unsigned,        // vtmmu.tvv (altfmt=0): uint8 x uint8 -> int32.
  kInt8UnsignedSigned,  // vtmmu.tvv (altfmt=1): uint8 x int8 -> int32.
  kInt8SignedUnsigned,  // vtmms.tvv (altfmt=0): int8 x uint8 -> int32.
  kFp8E4M3,             // vtfmm.tvv (altfmt=0): E4M3 x E4M3
  kFp8E4M3E5M2,         // vtfmm.tvv (altfmt=1): E4M3 x E5M2
  kFp8E5M2E4M3,         // vtfmm.alt.tvv (altfmt=0): E5M2 x E4M3
  kFp8E5M2,             // vtfmm.alt.tvv (altfmt=1): E5M2 x E5M2
  kFp16,                // vtfmm.tvv: fp16 x fp16 -> fp32
  kBf16,                // vtfmm.alt.tvv: bf16 x bf16 -> fp32.
  kFp32,                // vtfmm.tvv: fp32 x fp32 -> fp32.
  kFp64,                // vtfmm.tvv: fp64 x fp64 -> fp64.
};

// Dispatches matrix multiplication according to kind: C += A^T * B.
//
// Source operands:
//   0: matA: Vector register group holding matrix A (vs2).
//   1: matB: Vector register group holding matrix B (vs1).
//   2: mtile: Immediate operand specifying destination accumulator tile C.
//   3: vs2_idx: Immediate operand specifying the base register number of matrix
//      A.
//   4: vs1_idx: Immediate operand specifying the base register number of matrix
//      B.
// Destination operands:
//   0: fflags: Accrued floating-point exception flags (for floating-point
//      variants).
void RiscVZvtMatmul(RiscVZvtMatmulKind kind, const Instruction* inst);

// Performs signed integer matrix multiply-accumulate: C += A^T * B.
//
// Source operands:
//   0: matA: Vector register group holding matrix A (vs2).
//   1: matB: Vector register group holding matrix B (vs1).
//   2: mtile: Immediate operand specifying destination accumulator tile C.
//   3: vs2_idx: Immediate operand specifying the base register number of matrix
//      A.
//   4: vs1_idx: Immediate operand specifying the base register number of matrix
//      B.
// Destination operands:
//   None. Accumulates result directly into accumulator tile C.
void RiscVZvtVtmms(const Instruction* inst);

// Performs unsigned integer matrix multiply-accumulate: C += A^T * B.
//
// Source operands:
//   0: matA: Vector register group holding matrix A (vs2).
//   1: matB: Vector register group holding matrix B (vs1).
//   2: mtile: Immediate operand specifying destination accumulator tile C.
//   3: vs2_idx: Immediate operand specifying the base register number of matrix
//      A.
//   4: vs1_idx: Immediate operand specifying the base register number of matrix
//      B.
// Destination operands:
//   None. Accumulates result directly into accumulator tile C.
void RiscVZvtVtmmu(const Instruction* inst);

// Performs floating-point matrix multiply-accumulate: C += A^T * B.
//
// Source operands:
//   0: matA: Vector register group holding matrix A (vs2).
//   1: matB: Vector register group holding matrix B (vs1).
//   2: mtile: Immediate operand specifying destination accumulator tile C.
//   3: vs2_idx: Immediate operand specifying the base register number of matrix
//      A.
//   4: vs1_idx: Immediate operand specifying the base register number of matrix
//      B.
// Destination operands:
//   0: fflags: Accrued floating-point exception flags.
void RiscVZvtVtfmm(const Instruction* inst);

// Performs alternate floating-point (e.g. BF16 / FP8) matrix
// multiply-accumulate: C += A^T * B.
//
// Source operands:
//   0: matA: Vector register group holding matrix A (vs2).
//   1: matB: Vector register group holding matrix B (vs1).
//   2: mtile: Immediate operand specifying destination accumulator tile C.
//   3: vs2_idx: Immediate operand specifying the base register number of matrix
//      A.
//   4: vs1_idx: Immediate operand specifying the base register number of matrix
//      B.
// Destination operands:
//   0: fflags: Accrued floating-point exception flags.
void RiscVZvtVtfmmAlt(const Instruction* inst);

}  // namespace mpact::sim::riscv

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_ZVT_INSTRUCTIONS_H_
