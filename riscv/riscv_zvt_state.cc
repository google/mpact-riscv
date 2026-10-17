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

#include "riscv/riscv_zvt_state.h"

#include <cstdint>
#include <memory>

#include "absl/base/nullability.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_state.h"
#include "riscv/riscv_xstatus.h"

namespace mpact::sim::riscv {

namespace {
constexpr uint32_t kMstatusVsMask = 0x0600U;
}  // namespace

using ::mpact::sim::generic::operator*;  // NOLINT(misc-unused-using-decls)

uint32_t RiscVMtypeCsr::AsUint32() {
  auto* state = matrix_state_->riscv_state();
  if ((state->mstatus()->GetUint32() & kMstatusVsMask) == 0) {
    state->Trap(
        /*is_interrupt=*/false, 0,
        *::mpact::sim::riscv::ExceptionCode::kIllegalInstruction, 0, nullptr);
    return 0;
  }
  return matrix_state_->mtype();
}
uint64_t RiscVMtypeCsr::AsUint64() {
  auto* state = matrix_state_->riscv_state();
  if ((state->mstatus()->GetUint32() & kMstatusVsMask) == 0) {
    state->Trap(
        /*is_interrupt=*/false, 0,
        *::mpact::sim::riscv::ExceptionCode::kIllegalInstruction, 0, nullptr);
    return 0;
  }
  return matrix_state_->mtype();
}

void RiscVMtypeCsr::Write(uint32_t /*value*/) {
  matrix_state_->riscv_state()->Trap(
      /*is_interrupt=*/false, 0,
      *::mpact::sim::riscv::ExceptionCode::kIllegalInstruction, 0, nullptr);
}

void RiscVMtypeCsr::Write(uint64_t /*value*/) {
  matrix_state_->riscv_state()->Trap(
      /*is_interrupt=*/false, 0,
      *::mpact::sim::riscv::ExceptionCode::kIllegalInstruction, 0, nullptr);
}

absl::StatusOr<std::unique_ptr<RiscVZvtMatrixState>>
RiscVZvtMatrixState::Create(
    ::mpact::sim::riscv::RiscVState* /* absl_nonnull */ state,
    int tile_elements) {
  if (tile_elements < 4) {
    return absl::InvalidArgumentError("Tile element width must be >= 4");
  }
  if ((tile_elements & (tile_elements - 1)) != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("TE must be a power of two: ", tile_elements));
  }
  auto matrix = absl::WrapUnique(new RiscVZvtMatrixState(state, tile_elements));
  auto status = state->csr_set()->AddCsr(&matrix->mtype_csr_);
  if (!status.ok()) return status;
  return matrix;
}

RiscVZvtMatrixState::RiscVZvtMatrixState(
    ::mpact::sim::riscv::RiscVState* /* absl_nonnull */ state,
    int tile_elements)
    : state_(state), tile_elements_(tile_elements), mtype_csr_(this, state) {
  // TE must be a power of two in [4, VLEN/4]. VLEN/4 = 32 for VLEN=128b.
  tile_buffer_.reset(state_->db_factory()->Allocate<uint8_t>(
      16 * tile_elements * tile_elements));
  for (auto& byte : tile_buffer_->Get<uint8_t>()) {
    byte = 0;
  }
}

RiscVZvtMatrixState::~RiscVZvtMatrixState() = default;

::mpact::sim::generic::DataBuffer* RiscVZvtMatrixState::GetScratchDb(int size) {
  if (size >= scratch_dbs_.size()) scratch_dbs_.resize(size + 1);
  if (!scratch_dbs_[size]) {
    scratch_dbs_[size].reset(state_->db_factory()->Allocate<uint8_t>(size));
  }
  return scratch_dbs_[size].get();
}

int RiscVZvtMatrixState::sew_bytes() const {
  auto* rv_vector = state_->rv_vector();
  // selected_element_width() is in bytes; default to 1 (SEW=8) if no vector
  // state is attached.
  return rv_vector == nullptr ? 1 : rv_vector->selected_element_width();
}

// Transcribed verbatim from the Zvt v0.4 spec ("Tile Punning").
absl::StatusOr<int> RiscVZvtMatrixState::GetPunningOffset(
    int tile, int row, int col, int tile_element_width) const {
  const int kTileElements = tile_elements_;
  int ptile, minor_offset, major_offset;
  switch (tile_element_width) {
    case 8: {
      ptile = tile;
      minor_offset = (row % 4) * 4 + (col % 4);
      major_offset = (row / 4) * (kTileElements / 4) + (col / 4);
      break;
    }
    case 16: {
      ptile = tile + ((row & 2) >> 1);
      minor_offset = (row % 2) * 4 + (col % 2) * 2 + ((col / 2) % 2) * 8;
      major_offset = (row / 4) * (kTileElements / 4) + (col / 4);
      break;
    }
    case 32: {
      ptile = tile + 2 * (row / (kTileElements / 2)) + ((col & 2) >> 1);
      minor_offset = (row % 2) * 8 + (col % 2) * 4;
      major_offset =
          ((row / 2) % (kTileElements / 4)) * (kTileElements / 4) + (col / 4);
      break;
    }
    case 64: {
      ptile = tile + (row / (kTileElements / 4));
      minor_offset = (col % 2) * 8;
      major_offset =
          (row % (kTileElements / 4)) * (kTileElements / 4) + (col / 2);
      break;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid tile element width: ", tile_element_width));
  }
  return (ptile * kTileElements * kTileElements) + (major_offset * 16) +
         minor_offset;
}

namespace {
int GetTileAlignment(int tile_element_width_bits) {
  int ntiles = 16;
  if (tile_element_width_bits == 16) {
    ntiles = 8;
  } else if (tile_element_width_bits == 32) {
    ntiles = 4;
  } else if (tile_element_width_bits == 64) {
    ntiles = 8;
  }
  return 16 / ntiles;
}
}  // namespace

bool RiscVZvtMatrixState::IsValidTile(int tile, int tew_bits) const {
  return (tile % GetTileAlignment(tew_bits)) == 0;
}

int RiscVZvtMatrixState::MaskTssTile(int tile, int tile_element_width) const {
  return tile & ~(GetTileAlignment(tile_element_width) - 1);
}

absl::StatusOr<RiscVZvtMatrixState*> GetMatrixState(
    const ::mpact::sim::generic::Instruction* inst) {
  auto* rv_state = static_cast<::mpact::sim::riscv::RiscVState*>(inst->state());
  auto result = rv_state->csr_set()->GetCsr(kRiscVMtypeCsrIndex);
  if (!result.ok()) return result.status();
  return static_cast<RiscVMtypeCsr*>(*result)->matrix_state();
}

}  // namespace mpact::sim::riscv
