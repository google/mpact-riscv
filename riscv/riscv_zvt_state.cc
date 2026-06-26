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
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_state.h"

namespace mpact::sim::riscv {

uint32_t RiscVMtypeCsr::AsUint32() { return matrix_state_->mtype(); }
uint64_t RiscVMtypeCsr::AsUint64() { return matrix_state_->mtype(); }

absl::StatusOr<std::unique_ptr<RiscVZvtMatrixState>>
RiscVZvtMatrixState::Create(
    ::mpact::sim::riscv::RiscVState* /* absl_nonnull */ state,
    int tile_elements) {
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
  CHECK_GE(tile_elements_, 4) << "Tile element width must be >= 4";
  CHECK_EQ(tile_elements_ & (tile_elements_ - 1), 0)
      << "TE must be a power of two: " << tile_elements_;
  tile_buffer_ = state_->db_factory()->Allocate<uint8_t>(16 * tile_elements *
                                                         tile_elements);
  for (auto& byte : tile_buffer_->Get<uint8_t>()) {
    byte = 0;
  }
}

RiscVZvtMatrixState::~RiscVZvtMatrixState() {
  if (tile_buffer_ != nullptr) tile_buffer_->DecRef();
  for (auto* db : scratch_dbs_) {
    if (db != nullptr) db->DecRef();
  }
}

::mpact::sim::generic::DataBuffer* RiscVZvtMatrixState::GetScratchDb(int size) {
  if (size >= scratch_dbs_.size()) scratch_dbs_.resize(size + 1, nullptr);
  if (!scratch_dbs_[size]) {
    scratch_dbs_[size] = state_->db_factory()->Allocate<uint8_t>(size);
  }
  return scratch_dbs_[size];
}

int RiscVZvtMatrixState::sew_bytes() const {
  auto* rv_vector = state_->rv_vector();
  // selected_element_width() is in bytes; default to 1 (SEW=8) if no vector
  // state is attached.
  return rv_vector == nullptr ? 1 : rv_vector->selected_element_width();
}

// Transcribed verbatim from the Zvt v0.3 spec (15.1.1.1, "Tile Punning").
int RiscVZvtMatrixState::Punning(int tile, int row, int col,
                                 int tile_element_width) const {
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
      LOG(FATAL) << "Invalid tile element width: " << tile_element_width;
  }
  return (ptile * kTileElements * kTileElements) + (major_offset * 16) +
         minor_offset;
}

absl::StatusOr<RiscVZvtMatrixState*> GetMatrixState(
    const ::mpact::sim::generic::Instruction* inst) {
  auto* rv_state = static_cast<::mpact::sim::riscv::RiscVState*>(inst->state());
  auto result = rv_state->csr_set()->GetCsr(kRiscVMtypeCsrIndex);
  if (!result.ok()) return result.status();
  return static_cast<RiscVMtypeCsr*>(*result)->matrix_state();
}

}  // namespace mpact::sim::riscv
