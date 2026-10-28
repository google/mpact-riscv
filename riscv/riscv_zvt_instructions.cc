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

#include "riscv/riscv_zvt_instructions.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>

#include "absl/log/log.h"
#include "absl/types/span.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_fp_host.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_instruction_helpers.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_state.h"
#include "riscv/riscv_zvt_state.h"

namespace mpact::sim::riscv {

using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::generic::GetInstructionSource;
using ::mpact::sim::generic::operator*;  // NOLINT(misc-unused-using-decls)

using ::mpact::sim::riscv::RiscVVectorState;
using ::mpact::sim::riscv::RV32VectorDestinationOperand;
using ::mpact::sim::riscv::RV32VectorSourceOperand;

namespace {

// Returns whether the vill bit is set in vtype.
// TODO(b/575522744): Per RISC-V Vector Spec 1.0 (Section 3.1.2 & 3.4.4), vill
// is held at bit XLEN-1 (bit 31 for RV32, bit 63 for RV64). In
// RiscVVectorState, vtype is currently stored as a 32-bit integer with
// kVillBitMask = 1U << 31.
inline bool GetVtypeVill(
    const ::mpact::sim::riscv::RiscVVectorState* vector_state) {
  constexpr uint32_t kVillBitMask = 1U << 31;
  return (vector_state->vtype() & kVillBitMask) != 0;
}

// Returns the vector selected element width (SEW) in bits.
inline int GetSewBits(
    const ::mpact::sim::riscv::RiscVVectorState* vector_state) {
  switch (vector_state->selected_element_width()) {
    case 1:
      return 8;
    case 2:
      return 16;
    case 4:
      return 32;
    case 8:
      return 64;
    default:
      return 0;
  }
}

// Calculates the tile element width (TEW) in bits from 'sew_bits' and 'twiden'.
inline int CalculateTew(int sew_bits, int twiden) {
  switch (twiden) {
    case 1:
      return sew_bits;
    case 2:
      return sew_bits << 1;
    case 4:
      return sew_bits << 2;
    default:
      return 0;
  }
}

// ELEN for the Zve32{x,f} base.
constexpr int kElen = 32;

// Represents a decoded Tile Subset Specifier (TSS, per spec 15.1.1.5).
struct Tss {
  int tile;        // bits[30:27]
  int pattern;     // bits[26:24]: 0 = row, 1 = col.
  uint32_t index;  // bits[23:0].
};

// Decodes a 32-bit Tile Subset Specifier ('tss') for element width 'tew'.
Tss DecodeTss(const RiscVZvtMatrixState* matrix_state, uint32_t tss, int tew) {
  return Tss{
      .tile =
          matrix_state->MaskTssTile(static_cast<int>((tss >> 27) & 0xF), tew),
      .pattern = static_cast<int>((tss >> 24) & 0x7),
      .index = tss & 0xFFFFFF,
  };
}

constexpr int kTssPatternCol = 1;

// Updates vector and matrix geometry (vtype, vl, mtype) based on the
// current configuration.
void UpdateVectorMatrixGeometry(
    ::mpact::sim::riscv::RiscVVectorState* vector_state,
    ::mpact::sim::riscv::RiscVZvtMatrixState* matrix_state) {
  const int vill_shift = sizeof(vector_state->vtype()) * 8 - 1;
  const int sew_bits = GetSewBits(vector_state);
  if ((vector_state->vtype() >> vill_shift) || sew_bits == 0) {
    matrix_state->set_mtype(0);
    return;
  }
  const int twiden = matrix_state->twiden();
  if (twiden == 0) {
    vector_state->set_vector_exception();
    vector_state->SetVectorType(1ull << vill_shift);
    vector_state->set_vector_length(0);
    matrix_state->set_mtype(0);
    return;
  }
  const int tew = CalculateTew(sew_bits, twiden);
  const int effective_tile_elements = matrix_state->effective_tile_edge(tew);
  const int max_k = RiscVZvtKmax(sew_bits);

  int max_k_lmul_num = 8;
  int max_k_lmul_den = max_k;
  int ete_eve_lmul_num = effective_tile_elements;
  int ete_eve_lmul_den = vector_state->vector_register_byte_length() /
                         vector_state->selected_element_width();
  int ceil_ete_eve =
      (ete_eve_lmul_num + ete_eve_lmul_den - 1) / ete_eve_lmul_den;

  int vlmul_max_k = (8 * max_k_lmul_num) / max_k_lmul_den;
  int vlmul_ete_eve = 8 * ceil_ete_eve;

  // Per Zvt v0.4 spec 15.1.1.4: target LMUL is min(LMUL_max_k, LMUL_twiden,
  // LMUL_ete_eve). With kElen = 32, LMUL_twiden (64 / twiden) is mathematically
  // >= LMUL_max_k (64 / max_k = 2 * SEW) for all non-vill configurations
  // where SEW * twiden <= 32, making it strictly dominated by LMUL_max_k.
  int target_vlmul_scaled = std::min(vlmul_max_k, vlmul_ete_eve);
  if (target_vlmul_scaled < 1) target_vlmul_scaled = 1;
  if (target_vlmul_scaled > 64) target_vlmul_scaled = 64;

  uint32_t vlmul_enc = 0;
  switch (target_vlmul_scaled) {
    case 1:
      vlmul_enc = 5;
      break;
    case 2:
      vlmul_enc = 6;
      break;
    case 4:
      vlmul_enc = 7;
      break;
    case 8:
      vlmul_enc = 0;
      break;
    case 16:
      vlmul_enc = 1;
      break;
    case 32:
      vlmul_enc = 2;
      break;
    case 64:
      vlmul_enc = 3;
      break;
    default:
      vlmul_enc = 0;
      break;
  }

  uint32_t new_vtype =
      (vector_state->vtype() & ~0x7u) | vlmul_enc | (1u << 6) | (1u << 7);
  vector_state->SetVectorType(new_vtype);
  const int lmul_elements = vector_state->max_vector_length();

  const bool vill = tew > kElen;
  if (vill) {
    vector_state->set_vector_exception();
    vector_state->SetVectorType(1ull << vill_shift);
    matrix_state->set_mtype(0);
    return;
  }

  matrix_state->set_tile_k(std::min(matrix_state->tile_k(), max_k));
  matrix_state->set_tile_m(
      std::min(static_cast<int>(matrix_state->tile_m()),
               std::min(lmul_elements, effective_tile_elements)));
}

// Reads 'tew'-bit element bits from tile 'tile' at index [row, col].
uint64_t LoadTileBits(RiscVZvtMatrixState* matrix_state, int tile, int row,
                      int col, int tew) {
  const int effective_tile_edge = matrix_state->effective_tile_edge(tew);
  if (row < 0 || row >= effective_tile_edge || col < 0 ||
      col >= effective_tile_edge) {
    return 0;
  }
  switch (tew) {
    case 8:
      return matrix_state->GetElem<uint8_t>(tile, row, col, 8).value_or(0);
    case 16:
      return matrix_state->GetElem<uint16_t>(tile, row, col, 16).value_or(0);
    case 32:
      return matrix_state->GetElem<uint32_t>(tile, row, col, 32).value_or(0);
    case 64:
      return matrix_state->GetElem<uint64_t>(tile, row, col, 64).value_or(0);
    default:
      return 0;
  }
}

// Stores 'tew'-bit element bits into tile 'tile' at index [row, col].
void StoreTileBits(RiscVZvtMatrixState* matrix_state, int tile, int row,
                   int col, int tew, uint64_t value) {
  const int effective_tile_edge = matrix_state->effective_tile_edge(tew);
  if (row < 0 || row >= effective_tile_edge || col < 0 ||
      col >= effective_tile_edge) {
    return;
  }
  switch (tew) {
    case 8:
      matrix_state
          ->SetElem<uint8_t>(tile, row, col, 8, static_cast<uint8_t>(value))
          .IgnoreError();
      break;
    case 16:
      matrix_state
          ->SetElem<uint16_t>(tile, row, col, 16, static_cast<uint16_t>(value))
          .IgnoreError();
      break;
    case 32:
      matrix_state
          ->SetElem<uint32_t>(tile, row, col, 32, static_cast<uint32_t>(value))
          .IgnoreError();
      break;
    case 64:
      matrix_state->SetElem<uint64_t>(tile, row, col, 64, value).IgnoreError();
      break;
  }
}

// Writes a 32-bit value to the scalar destination operand at `dest_index`.
void WriteScalar(const Instruction* inst, int dest_index, uint32_t value) {
  auto* dest = inst->Destination(dest_index);
  DataBuffer* db = dest->AllocateDataBuffer();
  if (db->size<uint8_t>() == 8) {
    db->Set<uint64_t>(0, value);
  } else {
    db->Set<uint32_t>(0, value);
  }
  db->Submit();
}

// Traps with an Illegal Instruction if mstatus.MS is off.
bool TrapIfMatrixStateOff(const RiscVZvtMatrixState* matrix_state,
                          const Instruction* inst) {
  auto* state = matrix_state->riscv_state();

  bool trap = false;
  if ((state->mstatus()->GetUint32() & 0x60000000U) == 0) {
    trap = true;
  }

  if (trap) {
    state->Trap(
        /*is_interrupt=*/false, /*trap_value=*/0,
        *::mpact::sim::riscv::ExceptionCode::kIllegalInstruction,
        inst->address(), inst);
    return true;
  }
  return false;
}

// Traps with an Illegal Instruction if mstatus.FS is off.
bool TrapIfFpStateOff(const Instruction* inst) {
  auto* state = static_cast<RiscVState*>(inst->state());

  bool trap = false;
  if (state->mstatus()->fs() == 0) {
    trap = true;
  }

  if (trap) {
    state->Trap(
        /*is_interrupt=*/false, /*trap_value=*/0,
        *::mpact::sim::riscv::ExceptionCode::kIllegalInstruction,
        inst->address(), inst);
    return true;
  }
  return false;
}

// Sets mstatus.MS to Initial (1).
void SetMatrixStateInitial(const RiscVZvtMatrixState* matrix_state) {
  auto* state = matrix_state->riscv_state();
  if (auto mstatus_csr_or = state->csr_set()->GetCsr("mstatus");
      mstatus_csr_or.ok()) {
    auto* csr = mstatus_csr_or.value();
    uint64_t val =
        (csr->GetUint64() & ~kMstatusMsMask) |
        (static_cast<uint64_t>(kMstatusStateInitial) << kMstatusMsShift);
    csr->Set(val);
  }
}

// Sets mstatus.MS to Dirty (3) and updates mstatus.SD accordingly.
void SetMatrixStateDirty(const RiscVZvtMatrixState* matrix_state) {
  auto* state = matrix_state->riscv_state();
  if (auto mstatus_csr_or = state->csr_set()->GetCsr("mstatus");
      mstatus_csr_or.ok()) {
    auto* csr = mstatus_csr_or.value();
    uint64_t sd_mask = (state->xlen() == RiscVXlen::RV32) ? kMstatusSdMaskRv32
                                                          : kMstatusSdMaskRv64;
    uint64_t val =
        csr->GetUint64() |
        (static_cast<uint64_t>(kMstatusStateDirty) << kMstatusMsShift) |
        sd_mask;
    csr->Set(val);
  }
}

// Sets mstatus.VS to Dirty (3) and updates mstatus.SD accordingly.
void SetVectorStateDirty(const RiscVZvtMatrixState* matrix_state) {
  auto* state = matrix_state->riscv_state();
  if (auto mstatus_csr_or = state->csr_set()->GetCsr("mstatus");
      mstatus_csr_or.ok()) {
    auto* csr = mstatus_csr_or.value();
    uint64_t sd_mask = (state->xlen() == RiscVXlen::RV32) ? kMstatusSdMaskRv32
                                                          : kMstatusSdMaskRv64;
    uint64_t val =
        csr->GetUint64() |
        (static_cast<uint64_t>(kMstatusStateDirty) << kMstatusVsShift) |
        sd_mask;
    csr->Set(val);
  }
}

// Returns true and raises an IllegalInstruction exception if 'tss' is invalid.
bool TrapIfTssInvalid(RiscVZvtMatrixState* matrix_state, const Tss& tss,
                      int tew, const Instruction* inst) {
  if (tss.pattern > 1) {
    matrix_state->riscv_state()->Trap(
        /*is_interrupt=*/false, /*trap_value=*/0,
        *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return true;
  }
  const uint32_t effective_tile_edge =
      static_cast<uint32_t>(matrix_state->effective_tile_edge(tew));
  if (tss.index >= effective_tile_edge) {
    matrix_state->riscv_state()->Trap(
        /*is_interrupt=*/false, /*trap_value=*/0,
        *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return true;
  }
  return false;
}

// Returns the number of tile-subset elements processed:
// min(vl, EFFECTIVE_TILE_EDGE) for element width 'tew'.
int TileSubsetCount(RiscVZvtMatrixState* matrix_state,
                    RiscVVectorState* vector_state, int tew) {
  const int effective_tile_edge = matrix_state->effective_tile_edge(tew);
  return std::min(vector_state->vector_length(), effective_tile_edge);
}

}  // namespace

void RiscVZvtMsetMtypei(const Instruction* inst) {
  auto m_result = GetMatrixState(inst);
  if (!m_result.ok()) {
    LOG(ERROR) << "No matrix state found while executing Zvt instruction: "
               << m_result.status().message();
    auto* rv_state = static_cast<RiscVState*>(inst->state());
    rv_state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                   *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  auto* matrix_state = *m_result;
  auto* vector_state = matrix_state->riscv_state()->rv_vector();

  matrix_state->riscv_state()->mstatus()->Submit();

  const uint32_t mtwiden = GetInstructionSource<uint32_t>(inst, 0);
  const uint32_t vsew = GetInstructionSource<uint32_t>(inst, 1);
  const uint32_t altfmt = GetInstructionSource<uint32_t>(inst, 2) & 0x1;

  vector_state->set_vector_length(0);
  matrix_state->set_mtype(mtwiden << kMtypeMtwidenShift);

  // vtype fields: vsew at [5:3], altfmt at [8]
  uint32_t vtype = (vsew << 3) | ((altfmt & 1) << 8);

  vector_state->SetVectorType(vtype);

  if (matrix_state->mtwiden() == 0) {
    matrix_state->set_mtype(0);
    SetVectorStateDirty(matrix_state);
    return;
  }

  UpdateVectorMatrixGeometry(vector_state, matrix_state);
  SetVectorStateDirty(matrix_state);
  SetMatrixStateDirty(matrix_state);
}

void RiscVZvtMsetMtype(const Instruction* inst) {
  auto m_result = GetMatrixState(inst);
  if (!m_result.ok()) {
    LOG(ERROR) << "No matrix state found while executing Zvt instruction: "
               << m_result.status().message();
    auto* rv_state = static_cast<RiscVState*>(inst->state());
    rv_state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                   *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  auto* matrix_state = *m_result;
  auto* vector_state = matrix_state->riscv_state()->rv_vector();

  const uint32_t rs1 = GetInstructionSource<uint32_t>(inst, 0);  // rs1
  const uint32_t rs2 = GetInstructionSource<uint32_t>(inst, 1);  // rs2

  // Spec 15.1.1.4: vl = 0; mtype = rs1; then vtype = rs2 as if by vsetvl.
  vector_state->set_vector_length(0);
  matrix_state->set_mtype(rs1);
  vector_state->SetVectorType(rs2);

  if (matrix_state->mtwiden() == 0) {
    matrix_state->set_mtype(0);
    SetVectorStateDirty(matrix_state);
    return;
  }

  UpdateVectorMatrixGeometry(vector_state, matrix_state);
  SetVectorStateDirty(matrix_state);
  SetMatrixStateDirty(matrix_state);
}

void RiscVZvtMsetTn(const Instruction* inst) {
  auto m_result = GetMatrixState(inst);
  if (!m_result.ok()) {
    LOG(ERROR) << "No matrix state found while executing Zvt instruction: "
               << m_result.status().message();
    auto* rv_state = static_cast<RiscVState*>(inst->state());
    rv_state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                   *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  auto* matrix_state = *m_result;
  auto* vector_state = matrix_state->riscv_state()->rv_vector();

  const uint32_t rs1 = GetInstructionSource<uint32_t>(inst, 0);

  uint32_t tile_n;
  if (matrix_state->mtwiden() == 0) {
    // Degenerates to vsetvl using the current vtype: vl = min(rs1, VLMAX).
    tile_n = std::min<uint32_t>(rs1, vector_state->max_vector_length());
  } else {
    const int bound = std::min(vector_state->max_vector_length(),
                               matrix_state->effective_tile_edge());
    tile_n = std::min<uint32_t>(rs1, bound);
  }
  vector_state->set_vector_length(tile_n);
  WriteScalar(inst, 0, tile_n);
  SetVectorStateDirty(matrix_state);
  SetMatrixStateDirty(matrix_state);
}

void RiscVZvtMsetTm(const Instruction* inst) {
  auto m_result = GetMatrixState(inst);
  if (!m_result.ok()) {
    LOG(ERROR) << "No matrix state found while executing Zvt instruction: "
               << m_result.status().message();
    auto* rv_state = static_cast<RiscVState*>(inst->state());
    rv_state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                   *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  auto* matrix_state = *m_result;
  auto* vector_state = matrix_state->riscv_state()->rv_vector();

  const uint32_t rs1 = GetInstructionSource<uint32_t>(inst, 0);

  if (matrix_state->mtwiden() == 0) {
    matrix_state->set_tile_m(0);
    WriteScalar(inst, 0, 0);
    SetVectorStateDirty(matrix_state);
    return;
  }

  const int bound = std::min(vector_state->max_vector_length(),
                             matrix_state->effective_tile_edge());
  uint32_t tile_m = std::min<uint32_t>(rs1, bound);
  matrix_state->set_tile_m(tile_m);
  WriteScalar(inst, 0, tile_m);
  SetVectorStateDirty(matrix_state);
  SetMatrixStateDirty(matrix_state);
}

void RiscVZvtMsetTk(const Instruction* inst) {
  auto m_result = GetMatrixState(inst);
  if (!m_result.ok()) {
    LOG(ERROR) << "No matrix state found while executing Zvt instruction: "
               << m_result.status().message();
    auto* rv_state = static_cast<RiscVState*>(inst->state());
    rv_state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                   *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  auto* matrix_state = *m_result;
  auto* vector_state = matrix_state->riscv_state()->rv_vector();

  const uint32_t rs1 = GetInstructionSource<uint32_t>(inst, 0);

  if (matrix_state->mtwiden() == 0) {
    matrix_state->set_tile_k(0);
    WriteScalar(inst, 0, 0);
    SetVectorStateDirty(matrix_state);
    return;
  }

  const int sew_bits = GetSewBits(vector_state);
  uint32_t tile_k = std::min<uint32_t>(rs1, RiscVZvtKmax(sew_bits));
  matrix_state->set_tile_k(tile_k);
  WriteScalar(inst, 0, tile_k);
  SetVectorStateDirty(matrix_state);
  SetMatrixStateDirty(matrix_state);
}

// ----------------------------------------------------------------------------
// Tile manipulation
// ----------------------------------------------------------------------------

void RiscVZvtVtZero(const Instruction* inst) {
  auto m_result = GetMatrixState(inst);
  if (!m_result.ok()) {
    LOG(ERROR) << "No matrix state found while executing Zvt instruction: "
               << m_result.status().message();
    auto* rv_state = static_cast<RiscVState*>(inst->state());
    rv_state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                   *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  auto* matrix_state = *m_result;
  if (TrapIfMatrixStateOff(matrix_state, inst)) return;
  auto* vector_state = matrix_state->riscv_state()->rv_vector();
  const int tile = GetInstructionSource<uint32_t>(inst, 0);  // mtile
  const int tew = matrix_state->sew_bits() * matrix_state->twiden();
  if (matrix_state->twiden() != 0) {
    if (!matrix_state->IsValidTile(tile, tew)) {
      matrix_state->riscv_state()->Trap(
          /*is_interrupt=*/false, /*trap_value=*/0,
          *ExceptionCode::kIllegalInstruction, inst->address(), inst);
      return;
    }
  }
  const int tile_m = matrix_state->tile_m();
  const int effective_tile_edge = matrix_state->effective_tile_edge(tew);
  const int tile_n =
      std::min(vector_state->vector_length(), effective_tile_edge);
  for (int row = 0; row < tile_m; row++) {
    for (int col = 0; col < tile_n; col++) {
      StoreTileBits(matrix_state, tile, row, col, tew, 0);
    }
  }
  SetMatrixStateDirty(matrix_state);
  SetVectorStateDirty(matrix_state);
}

void RiscVZvtVtDiscard(const Instruction* inst) {
  // Architecturally hints the tile state is dead. No-op in the ISS.
  auto m_result = GetMatrixState(inst);
  if (!m_result.ok()) {
    LOG(ERROR) << "No matrix state found while executing Zvt instruction: "
               << m_result.status().message();
    auto* rv_state = static_cast<RiscVState*>(inst->state());
    rv_state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                   *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  auto* matrix_state = *m_result;
  if (TrapIfMatrixStateOff(matrix_state, inst)) return;

  auto* rv_vector = matrix_state->riscv_state()->rv_vector();
  if (GetVtypeVill(rv_vector)) {
    matrix_state->riscv_state()->Trap(
        /*is_interrupt=*/false, /*trap_value=*/0,
        *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }

  rv_vector->clear_vstart();
  SetMatrixStateInitial(matrix_state);
  SetVectorStateDirty(matrix_state);
}

void RiscVZvtVtmvVt(const Instruction* inst) {
  // Transfers a tile row or column subset to a vector register group
  // (vtmv.v.t).
  auto m_result = GetMatrixState(inst);
  if (!m_result.ok()) {
    LOG(ERROR) << "No matrix state found while executing Zvt instruction: "
               << m_result.status().message();
    auto* rv_state = static_cast<RiscVState*>(inst->state());
    rv_state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                   *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  auto* matrix_state = *m_result;
  if (TrapIfMatrixStateOff(matrix_state, inst)) return;
  auto* state = matrix_state->riscv_state();
  if (!matrix_state->configured()) {
    state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  auto* vector_state = state->rv_vector();
  const int sew_bytes = vector_state->selected_element_width();
  const int tew = sew_bytes * 8;
  const Tss tss =
      DecodeTss(matrix_state, GetInstructionSource<uint32_t>(inst, 0), tew);
  if (TrapIfTssInvalid(matrix_state, tss, tew, inst)) return;
  const int count = TileSubsetCount(matrix_state, vector_state, tew);
  const int start_element = vector_state->vstart();
  if (start_element >= count) {
    vector_state->clear_vstart();
    SetVectorStateDirty(matrix_state);
    return;
  }
  const int elements_per_vector =
      vector_state->vector_register_byte_length() / sew_bytes;
  auto* vd = static_cast<RV32VectorDestinationOperand*>(inst->Destination(0));
  const int num_regs = (count + elements_per_vector - 1) / elements_per_vector;
  const int start_reg = start_element / elements_per_vector;
  for (int r = start_reg; r < num_regs; r++) {
    // Copy the destination buffer to retain undisturbed elements.
    DataBuffer* db = vd->CopyDataBuffer(r);
    absl::Span<uint8_t> bytes = db->Get<uint8_t>();
    // Overwrite target elements in the active range [start_element, count).
    for (int e = 0; e < elements_per_vector; e++) {
      const int elem = r * elements_per_vector + e;
      if (elem < start_element) continue;
      if (elem >= count) {
        // Zero-fill tail elements past the active count.
        std::memset(&bytes[e * sew_bytes], 0, sew_bytes);
        continue;
      }
      const uint64_t val =
          (tss.pattern == kTssPatternCol)
              ? LoadTileBits(matrix_state, tss.tile, elem, tss.index, tew)
              : LoadTileBits(matrix_state, tss.tile, tss.index, elem, tew);
      std::memcpy(&bytes[e * sew_bytes], &val, sew_bytes);
    }
    db->Submit();
  }
  vector_state->clear_vstart();
  SetVectorStateDirty(matrix_state);
}

void RiscVZvtVtmvTv(const Instruction* inst) {
  // Transfers a vector register group to a tile row or column subset
  // (vtmv.t.v).
  auto m_result = GetMatrixState(inst);
  if (!m_result.ok()) {
    LOG(ERROR) << "No matrix state found while executing Zvt instruction: "
               << m_result.status().message();
    auto* rv_state = static_cast<RiscVState*>(inst->state());
    rv_state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                   *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  auto* matrix_state = *m_result;
  if (TrapIfMatrixStateOff(matrix_state, inst)) return;
  auto* state = matrix_state->riscv_state();
  if (!matrix_state->configured()) {
    state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  auto* vector_state = state->rv_vector();
  auto* vs = static_cast<RV32VectorSourceOperand*>(inst->Source(0));
  const int sew_bytes = vector_state->selected_element_width();
  const int tew = sew_bytes * 8;
  const Tss tss =
      DecodeTss(matrix_state, GetInstructionSource<uint32_t>(inst, 1), tew);
  if (TrapIfTssInvalid(matrix_state, tss, tew, inst)) return;
  const int count = TileSubsetCount(matrix_state, vector_state, tew);
  const int start_element = vector_state->vstart();
  if (start_element >= count) {
    vector_state->clear_vstart();
    SetVectorStateDirty(matrix_state);
    return;
  }
  const int elements_per_vector =
      vector_state->vector_register_byte_length() / sew_bytes;
  for (int elem = start_element; elem < count; elem++) {
    const int reg = elem / elements_per_vector;
    const int e = elem % elements_per_vector;
    auto bytes = vs->GetRegister(reg)->data_buffer()->Get<uint8_t>();
    uint64_t val = 0;
    std::memcpy(&val, &bytes[e * sew_bytes], sew_bytes);
    if (tss.pattern == kTssPatternCol) {
      StoreTileBits(matrix_state, tss.tile, elem, tss.index, tew, val);
    } else {
      StoreTileBits(matrix_state, tss.tile, tss.index, elem, tew, val);
    }
  }
  vector_state->clear_vstart();
  SetMatrixStateDirty(matrix_state);
}

void RiscVZvtVtle(int eew_bits, const Instruction* inst) {
  // Loads a tile row or column subset from memory into a tile register (vtle*).
  auto m_result = GetMatrixState(inst);
  if (!m_result.ok()) {
    LOG(ERROR) << "No matrix state found while executing Zvt instruction: "
               << m_result.status().message();
    auto* rv_state = static_cast<RiscVState*>(inst->state());
    rv_state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                   *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  auto* matrix_state = *m_result;
  if (TrapIfMatrixStateOff(matrix_state, inst)) return;

  if (!matrix_state->configured()) {
    matrix_state->riscv_state()->Trap(
        /*is_interrupt=*/false, /*trap_value=*/0,
        *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }

  auto* state = matrix_state->riscv_state();
  auto* vector_state = state->rv_vector();
  const uint64_t base = GetInstructionSource<uint64_t>(inst, 0);  // rs1
  const int element_size = eew_bits / 8;
  const int tew = eew_bits;
  const Tss tss = DecodeTss(
      matrix_state, GetInstructionSource<uint32_t>(inst, 1), tew);  // rs2
  if (TrapIfTssInvalid(matrix_state, tss, tew, inst)) return;
  const int count = TileSubsetCount(matrix_state, vector_state, tew);
  const int start_element = vector_state->vstart();
  if (start_element >= count) {
    vector_state->clear_vstart();
    SetVectorStateDirty(matrix_state);
    return;
  }
  if (element_size > 1 &&
      ((base + vector_state->vstart() * element_size) % element_size) != 0) {
    state->Trap(/*is_interrupt=*/false,
                /*trap_value=*/base + vector_state->vstart() * element_size,
                *ExceptionCode::kLoadAddressMisaligned, inst->address(), inst);
    return;
  }
  const int elements_to_load = count - start_element;
  DataBuffer* db = matrix_state->GetScratchDb(elements_to_load * element_size);
  state->memory()->Load(base + start_element * element_size, db, nullptr,
                        nullptr);
  if (state->branch()) return;
  absl::Span<uint8_t> bytes = db->Get<uint8_t>();
  for (int e = start_element; e < count; e++) {
    uint64_t val = 0;
    std::memcpy(&val, &bytes[(e - start_element) * element_size], element_size);
    if (tss.pattern == kTssPatternCol) {
      StoreTileBits(matrix_state, tss.tile, e, tss.index, eew_bits, val);
    } else {
      StoreTileBits(matrix_state, tss.tile, tss.index, e, eew_bits, val);
    }
  }
  if (vector_state->vector_tail_agnostic()) {
    const int effective_tile_edge = matrix_state->effective_tile_edge(eew_bits);
    for (int e = count; e < effective_tile_edge; e++) {
      if (tss.pattern == kTssPatternCol) {
        StoreTileBits(matrix_state, tss.tile, e, tss.index, eew_bits, ~0ULL);
      } else {
        StoreTileBits(matrix_state, tss.tile, tss.index, e, eew_bits, ~0ULL);
      }
    }
  }
  vector_state->clear_vstart();
  SetMatrixStateDirty(matrix_state);
}

void RiscVZvtVtse(int eew_bits, const Instruction* inst) {
  // Stores a tile row or column subset from a tile register into memory
  // (vtse*).
  auto m_result = GetMatrixState(inst);
  if (!m_result.ok()) {
    LOG(ERROR) << "No matrix state found while executing Zvt instruction: "
               << m_result.status().message();
    auto* rv_state = static_cast<RiscVState*>(inst->state());
    rv_state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                   *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  auto* matrix_state = *m_result;
  if (TrapIfMatrixStateOff(matrix_state, inst)) return;

  if (!matrix_state->configured()) {
    matrix_state->riscv_state()->Trap(
        /*is_interrupt=*/false, /*trap_value=*/0,
        *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }

  auto* state = matrix_state->riscv_state();
  auto* vector_state = state->rv_vector();
  const uint64_t base = GetInstructionSource<uint64_t>(inst, 0);  // rs1
  const int element_size = eew_bits / 8;
  const int tew = eew_bits;
  const Tss tss = DecodeTss(
      matrix_state, GetInstructionSource<uint32_t>(inst, 1), tew);  // rs2
  if (TrapIfTssInvalid(matrix_state, tss, tew, inst)) return;
  const int count = TileSubsetCount(matrix_state, vector_state, tew);
  const int start_element = vector_state->vstart();
  if (start_element >= count) {
    vector_state->clear_vstart();
    SetVectorStateDirty(matrix_state);
    return;
  }
  if (element_size > 1 &&
      ((base + vector_state->vstart() * element_size) % element_size) != 0) {
    state->Trap(/*is_interrupt=*/false,
                /*trap_value=*/base + vector_state->vstart() * element_size,
                *ExceptionCode::kStoreAddressMisaligned, inst->address(), inst);
    return;
  }
  const int elements_to_store = count - start_element;
  DataBuffer* db = matrix_state->GetScratchDb(elements_to_store * element_size);
  absl::Span<uint8_t> bytes = db->Get<uint8_t>();
  for (int e = start_element; e < count; e++) {
    const uint64_t val =
        (tss.pattern == kTssPatternCol)
            ? LoadTileBits(matrix_state, tss.tile, e, tss.index, eew_bits)
            : LoadTileBits(matrix_state, tss.tile, tss.index, e, eew_bits);
    std::memcpy(&bytes[(e - start_element) * element_size], &val, element_size);
  }
  state->memory()->Store(base + start_element * element_size, db);
  if (state->branch()) return;
  vector_state->clear_vstart();
  SetVectorStateDirty(matrix_state);
}

// ----------------------------------------------------------------------------
// Matrix multiply: C[m,n] += sum_k A[k,m] * B[k,n]
// ----------------------------------------------------------------------------

void RiscVZvtMatmul(RiscVZvtMatmulKind kind, const Instruction* inst) {
  auto m_result = GetMatrixState(inst);
  if (!m_result.ok()) {
    LOG(ERROR) << "No matrix state found while executing Zvt instruction: "
               << m_result.status().message();
    auto* rv_state = static_cast<RiscVState*>(inst->state());
    rv_state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                   *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  auto* matrix_state = *m_result;
  if (TrapIfMatrixStateOff(matrix_state, inst)) return;
  auto* vector_state = matrix_state->riscv_state()->rv_vector();
  if (vector_state->vstart() > 0) {
    matrix_state->riscv_state()->Trap(
        /*is_interrupt=*/false, /*trap_value=*/0,
        *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  if (!matrix_state->configured()) {
    matrix_state->riscv_state()->Trap(
        /*is_interrupt=*/false, /*trap_value=*/0,
        *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  const int sew_bytes = vector_state->selected_element_width();
  const int sew_bits = sew_bytes * 8;

  const int elements_per_vector =
      vector_state->vector_register_byte_length() / sew_bytes;  // elems/reg.
  const int stride = 8 / RiscVZvtKmax(sew_bits);  // regs between rows.
  const int tile_k =
      matrix_state->tile_k();  // K: the inner dimension of the dot product.
  const int tile_m =
      matrix_state->tile_m();  // M: the number of rows of the accumulator tile.
  const uint32_t raw_mtile = GetInstructionSource<uint32_t>(inst, 2);
  const int dest_tile = raw_mtile;
  const int tew = matrix_state->sew_bits() * matrix_state->twiden();
  const int effective_tile_edge = matrix_state->effective_tile_edge(tew);
  // N: the number of columns of the accumulator tile (derived from vector
  // length).
  const int tile_n =
      std::min(vector_state->vector_length(), effective_tile_edge);

  auto* a = static_cast<RV32VectorSourceOperand*>(inst->Source(0));
  auto* b = static_cast<RV32VectorSourceOperand*>(inst->Source(1));

  const uint32_t reg_a = GetInstructionSource<uint32_t>(inst, 3);
  const uint32_t reg_b = GetInstructionSource<uint32_t>(inst, 4);

  int lmul = std::max(1, vector_state->vector_length_multiplier() / 8);
  if (!matrix_state->IsValidTile(dest_tile, tew) || (reg_a % lmul != 0) ||
      (reg_b % lmul != 0) || (reg_a % 8 >= stride) || (reg_b % 8 >= stride)) {
    matrix_state->riscv_state()->Trap(
        /*is_interrupt=*/false, /*trap_value=*/0,
        *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }

  if (tile_m == 0 || tile_n == 0 || tile_k == 0) {
    return;
  }

  absl::Span<uint8_t> a_spans[32] = {};
  absl::Span<uint8_t> b_spans[32] = {};

  const bool is_fp =
      (kind == RiscVZvtMatmulKind::kFp32 || kind == RiscVZvtMatmulKind::kBf16 ||
       kind == RiscVZvtMatmulKind::kFp16 ||
       kind == RiscVZvtMatmulKind::kFp8E4M3 ||
       kind == RiscVZvtMatmulKind::kFp8E4M3E5M2 ||
       kind == RiscVZvtMatmulKind::kFp8E5M2E4M3 ||
       kind == RiscVZvtMatmulKind::kFp8E5M2 ||
       kind == RiscVZvtMatmulKind::kFp64);

  auto get_span = [](absl::Span<uint8_t>* spans, int reg,
                     RV32VectorSourceOperand* op) {
    if (spans[reg].empty()) {
      auto* reg_ptr = op->GetRegister(reg);
      if (reg_ptr) {
        spans[reg] = reg_ptr->data_buffer()->Get<uint8_t>();
      }
    }
    return spans[reg];
  };

  // Reads element `col` of row `kk` from a vector-group operand for matrix A.
  auto read_fp_a = [&get_span, kind, stride, elements_per_vector, sew_bytes](
                       RV32VectorSourceOperand* op, absl::Span<uint8_t>* spans,
                       int kk, int col) -> float {
    const int reg = kk * stride + col / elements_per_vector;
    const int e = col % elements_per_vector;
    absl::Span<uint8_t> span = get_span(spans, reg, op);
    if (kind == RiscVZvtMatmulKind::kBf16) {
      uint16_t u;
      std::memcpy(&u, &span[e * sew_bytes], sizeof(u));
      return ::mpact::sim::generic::ConvertBf16ToSingle(
          ::mpact::sim::generic::Bfloat16{u});
    }
    if (kind == RiscVZvtMatmulKind::kFp16) {
      uint16_t u;
      std::memcpy(&u, &span[e * sew_bytes], sizeof(u));
      return ::mpact::sim::generic::ConvertHalfToSingle(
          ::mpact::sim::generic::HalfFP{u});
    }
    if (kind == RiscVZvtMatmulKind::kFp8E4M3 ||
        kind == RiscVZvtMatmulKind::kFp8E4M3E5M2) {
      uint8_t u;
      std::memcpy(&u, &span[e * sew_bytes], sizeof(u));
      return ::mpact::sim::generic::ConvertFp8E4m3ToSingle(
          ::mpact::sim::generic::Fp8E4m3{u});
    }
    if (kind == RiscVZvtMatmulKind::kFp8E5M2E4M3 ||
        kind == RiscVZvtMatmulKind::kFp8E5M2) {
      uint8_t u;
      std::memcpy(&u, &span[e * sew_bytes], sizeof(u));
      return ::mpact::sim::generic::ConvertFp8E5m2ToSingle(
          ::mpact::sim::generic::Fp8E5m2{u});
    }
    float f;
    std::memcpy(&f, &span[e * sew_bytes], sizeof(f));
    return f;
  };

  // Reads element `col` of row `kk` from a vector-group operand for matrix B.
  auto read_fp_b = [&get_span, kind, stride, elements_per_vector, sew_bytes](
                       RV32VectorSourceOperand* op, absl::Span<uint8_t>* spans,
                       int kk, int col) -> float {
    const int reg = kk * stride + col / elements_per_vector;
    const int e = col % elements_per_vector;
    absl::Span<uint8_t> span = get_span(spans, reg, op);
    if (kind == RiscVZvtMatmulKind::kBf16) {
      uint16_t u;
      std::memcpy(&u, &span[e * sew_bytes], sizeof(u));
      return ::mpact::sim::generic::ConvertBf16ToSingle(
          ::mpact::sim::generic::Bfloat16{u});
    }
    if (kind == RiscVZvtMatmulKind::kFp16) {
      uint16_t u;
      std::memcpy(&u, &span[e * sew_bytes], sizeof(u));
      return ::mpact::sim::generic::ConvertHalfToSingle(
          ::mpact::sim::generic::HalfFP{u});
    }
    if (kind == RiscVZvtMatmulKind::kFp8E4M3 ||
        kind == RiscVZvtMatmulKind::kFp8E5M2E4M3) {
      uint8_t u;
      std::memcpy(&u, &span[e * sew_bytes], sizeof(u));
      return ::mpact::sim::generic::ConvertFp8E4m3ToSingle(
          ::mpact::sim::generic::Fp8E4m3{u});
    }
    if (kind == RiscVZvtMatmulKind::kFp8E4M3E5M2 ||
        kind == RiscVZvtMatmulKind::kFp8E5M2) {
      uint8_t u;
      std::memcpy(&u, &span[e * sew_bytes], sizeof(u));
      return ::mpact::sim::generic::ConvertFp8E5m2ToSingle(
          ::mpact::sim::generic::Fp8E5m2{u});
    }
    float f;
    std::memcpy(&f, &span[e * sew_bytes], sizeof(f));
    return f;
  };
  auto read_fp64 = [&get_span, stride, elements_per_vector, sew_bytes](
                       RV32VectorSourceOperand* op, absl::Span<uint8_t>* spans,
                       int kk, int col) -> double {
    const int reg = kk * stride + col / elements_per_vector;
    const int e = col % elements_per_vector;
    absl::Span<uint8_t> span = get_span(spans, reg, op);
    double d;
    std::memcpy(&d, &span[e * sew_bytes], sizeof(d));
    return d;
  };
  auto read_int_a = [&get_span, kind, stride, elements_per_vector](
                        RV32VectorSourceOperand* op, absl::Span<uint8_t>* spans,
                        int kk, int col) -> int32_t {
    const int reg = kk * stride + col / elements_per_vector;
    const int e = col % elements_per_vector;
    absl::Span<uint8_t> span = get_span(spans, reg, op);
    if (kind == RiscVZvtMatmulKind::kInt8Signed ||
        kind == RiscVZvtMatmulKind::kInt8SignedUnsigned) {
      return static_cast<int8_t>(span[e]);
    }
    return static_cast<uint8_t>(span[e]);
  };
  auto read_int_b = [&get_span, kind, stride, elements_per_vector](
                        RV32VectorSourceOperand* op, absl::Span<uint8_t>* spans,
                        int kk, int col) -> int32_t {
    const int reg = kk * stride + col / elements_per_vector;
    const int e = col % elements_per_vector;
    absl::Span<uint8_t> span = get_span(spans, reg, op);
    if (kind == RiscVZvtMatmulKind::kInt8Signed ||
        kind == RiscVZvtMatmulKind::kInt8UnsignedSigned) {
      return static_cast<int8_t>(span[e]);
    }
    return static_cast<uint8_t>(span[e]);
  };

  auto* fp_state = matrix_state->riscv_state()->rv_fp();
  uint32_t orig_fcsr = 0;
  if (is_fp && fp_state) {
    orig_fcsr = fp_state->fcsr()->AsUint32();
  }

  {
    std::optional<ScopedFPStatus> fp_status;
    if (is_fp && fp_state) {
      fp_status.emplace(fp_state->host_fp_interface());
    }

    for (int mm = 0; mm < tile_m; mm++) {
      for (int nn = 0; nn < tile_n; nn++) {
        if (is_fp) {
          if (kind == RiscVZvtMatmulKind::kFp64) {
            double acc = matrix_state->GetElem<double>(dest_tile, mm, nn, tew)
                             .value_or(0.0);
            for (int kk = 0; kk < tile_k; kk++) {
              double product =
                  read_fp64(a, a_spans, kk, mm) * read_fp64(b, b_spans, kk, nn);
              // For Fp64 (SEW=64, TWIDEN=1), TEW=64. The product is implicitly
              // rounded to Fp64.
              acc += product;
            }
            matrix_state->SetElem<double>(dest_tile, mm, nn, tew, acc)
                .IgnoreError();
          } else {
            double sum_products = 0.0;
            for (int kk = 0; kk < tile_k; kk++) {
              double product =
                  static_cast<double>(read_fp_a(a, a_spans, kk, mm)) *
                  static_cast<double>(read_fp_b(b, b_spans, kk, nn));

              if (kind == RiscVZvtMatmulKind::kFp32) {
                // For Fp32 (SEW=32, TWIDEN=1), TEW=32. Implement TEW-bit
                // intermediate rounding.
                product = static_cast<double>(static_cast<float>(product));
              }
              sum_products += product;
            }

            float sum_products_f;
            if (kind != RiscVZvtMatmulKind::kFp32) {
              sum_products_f = DoubleToFloatROD(sum_products, fp_state);
            } else {
              sum_products_f = static_cast<float>(sum_products);
            }

            float initial_acc =
                matrix_state->GetElem<float>(dest_tile, mm, nn, tew)
                    .value_or(0.0f);
            double final_acc_d = static_cast<double>(initial_acc) +
                                 static_cast<double>(sum_products_f);

            float final_acc;
            if (kind != RiscVZvtMatmulKind::kFp32) {
              final_acc = DoubleToFloatROD(final_acc_d, fp_state);
            } else {
              final_acc = static_cast<float>(final_acc_d);
            }

            matrix_state->SetElem<float>(dest_tile, mm, nn, tew, final_acc)
                .IgnoreError();
          }
        } else {
          int32_t acc = matrix_state->GetElem<int32_t>(dest_tile, mm, nn, tew)
                            .value_or(0);
          for (int kk = 0; kk < tile_k; kk++) {
            auto av = read_int_a(a, a_spans, kk, mm);
            auto bv = read_int_b(b, b_spans, kk, nn);
            acc += av * bv;
          }
          matrix_state->SetElem<int32_t>(dest_tile, mm, nn, tew, acc)
              .IgnoreError();
        }
      }
    }
  }

  if (is_fp && fp_state) {
    uint32_t current_fcsr = fp_state->fcsr()->AsUint32();
    constexpr uint32_t kFcsrDzBit = 1 << 3;
    constexpr uint32_t kFcsrUfBit = 1 << 1;
    constexpr uint32_t kFcsrNxBit = 1 << 0;
    uint32_t preservation_mask = kFcsrDzBit | kFcsrUfBit | kFcsrNxBit;
    uint32_t intended_fcsr =
        (current_fcsr & ~preservation_mask) | (orig_fcsr & preservation_mask);

    if (intended_fcsr != current_fcsr) {
      fp_state->fcsr()->Write(intended_fcsr);
    }

    if (intended_fcsr != orig_fcsr ||
        (tile_m > 0 && tile_n > 0 && tile_k > 0)) {
      auto* mstatus = matrix_state->riscv_state()->mstatus();
      if (mstatus != nullptr) {
        mstatus->set_fs(3);
        mstatus->Submit();
      }
    }
  }

  SetMatrixStateDirty(matrix_state);
}

void RiscVZvtVtmms(const Instruction* inst) {
  auto m_result = GetMatrixState(inst);
  if (!m_result.ok()) {
    LOG(ERROR) << "No matrix state found while executing Zvt instruction: "
               << m_result.status().message();
    auto* rv_state = static_cast<RiscVState*>(inst->state());
    rv_state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                   *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  auto* matrix_state = *m_result;
  auto* vector_state = matrix_state->riscv_state()->rv_vector();
  const int sew_bits = GetSewBits(vector_state);
  if (sew_bits != 8 || matrix_state->twiden() != 4) {
    matrix_state->riscv_state()->Trap(
        /*is_interrupt=*/false, /*trap_value=*/0,
        *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  const uint32_t altfmt = (vector_state->vtype() >> 8) & 1;
  RiscVZvtMatmul(altfmt ? RiscVZvtMatmulKind::kInt8Signed
                        : RiscVZvtMatmulKind::kInt8SignedUnsigned,
                 inst);
}

void RiscVZvtVtmmu(const Instruction* inst) {
  auto m_result = GetMatrixState(inst);
  if (!m_result.ok()) {
    LOG(ERROR) << "No matrix state found while executing Zvt instruction: "
               << m_result.status().message();
    auto* rv_state = static_cast<RiscVState*>(inst->state());
    rv_state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                   *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  auto* matrix_state = *m_result;
  auto* vector_state = matrix_state->riscv_state()->rv_vector();
  const int sew_bits = GetSewBits(vector_state);
  if (sew_bits != 8 || matrix_state->twiden() != 4) {
    matrix_state->riscv_state()->Trap(
        /*is_interrupt=*/false, /*trap_value=*/0,
        *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  const uint32_t altfmt = (vector_state->vtype() >> 8) & 1;
  RiscVZvtMatmul(altfmt ? RiscVZvtMatmulKind::kInt8UnsignedSigned
                        : RiscVZvtMatmulKind::kInt8Unsigned,
                 inst);
}

void RiscVZvtVtfmm(const Instruction* inst) {
  if (TrapIfFpStateOff(inst)) return;

  auto m_result = GetMatrixState(inst);
  if (!m_result.ok()) {
    LOG(ERROR) << "No matrix state found while executing Zvt instruction: "
               << m_result.status().message();
    auto* rv_state = static_cast<RiscVState*>(inst->state());
    rv_state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                   *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  auto* matrix_state = *m_result;
  auto* vector_state = matrix_state->riscv_state()->rv_vector();
  const uint32_t altfmt = (vector_state->vtype() >> 8) & 1;
  const int sew_bits = GetSewBits(vector_state);
  const uint32_t twiden = matrix_state->twiden();

  RiscVZvtMatmulKind kind = RiscVZvtMatmulKind::kFp32;
  if (sew_bits == 8 && twiden == 4) {
    kind = altfmt ? RiscVZvtMatmulKind::kFp8E4M3E5M2
                  : RiscVZvtMatmulKind::kFp8E4M3;
  } else if (sew_bits == 16 && twiden == 2 && altfmt == 0) {
    kind = RiscVZvtMatmulKind::kFp16;
  } else if (sew_bits == 32 && twiden == 1 && altfmt == 0) {
    kind = RiscVZvtMatmulKind::kFp32;
  } else if (sew_bits == 64 && twiden == 1 && altfmt == 0) {
    kind = RiscVZvtMatmulKind::kFp64;
  } else {
    auto* rv_state = static_cast<RiscVState*>(inst->state());
    rv_state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                   *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  RiscVZvtMatmul(kind, inst);
}

void RiscVZvtVtfmmAlt(const Instruction* inst) {
  if (TrapIfFpStateOff(inst)) return;

  auto m_result = GetMatrixState(inst);
  if (!m_result.ok()) {
    LOG(ERROR) << "No matrix state found while executing Zvt instruction: "
               << m_result.status().message();
    auto* rv_state = static_cast<RiscVState*>(inst->state());
    rv_state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                   *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  auto* matrix_state = *m_result;
  auto* vector_state = matrix_state->riscv_state()->rv_vector();
  const uint32_t altfmt = (vector_state->vtype() >> 8) & 1;
  const int sew_bits = GetSewBits(vector_state);
  const uint32_t twiden = matrix_state->twiden();

  RiscVZvtMatmulKind kind = RiscVZvtMatmulKind::kBf16;
  if (sew_bits == 8 && twiden == 4) {
    kind = altfmt ? RiscVZvtMatmulKind::kFp8E5M2
                  : RiscVZvtMatmulKind::kFp8E5M2E4M3;
  } else if (sew_bits == 16 && twiden == 2 && altfmt == 0) {
    kind = RiscVZvtMatmulKind::kBf16;
  } else {
    auto* rv_state = static_cast<RiscVState*>(inst->state());
    rv_state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                   *ExceptionCode::kIllegalInstruction, inst->address(), inst);
    return;
  }
  RiscVZvtMatmul(kind, inst);
}

}  // namespace mpact::sim::riscv
