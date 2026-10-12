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
#include <utility>

#include "absl/status/status.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/immediate_operand.h"  // NOLINT(misc-include-cleaner)
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_memory_instructions.h"
#include "riscv/riscv_vector_state.h"

/* start - helper test macro */
#define CONCAT_INNER(a, b) a##b
#define CONCAT(a, b) CONCAT_INNER(a, b)
#define ASSERT_OK(expression) ASSERT_TRUE((expression).ok())
#define ASSERT_OK_AND_ASSIGN(lhs, rexpr)            \
  auto CONCAT(status_or_line_, __LINE__) = (rexpr); \
  ASSERT_OK(CONCAT(status_or_line_, __LINE__));     \
  lhs = std::move(CONCAT(status_or_line_, __LINE__)).value();
/* end - helper test macro */

namespace {

using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::RiscVZvtMatrixState;
using ::mpact::sim::util::FlatDemandMemory;

TEST(RiscVZvtStateTest, MtypeLayoutConfiguration) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);
  ASSERT_OK_AND_ASSIGN(auto matrix_state,
                       RiscVZvtMatrixState::Create(state.get(), 32));

  // Test mtwiden (bits 0-1)
  matrix_state->set_mtype(0x3);
  EXPECT_EQ(matrix_state->mtwiden(), 3);
  EXPECT_EQ(matrix_state->twiden(), 4);
  EXPECT_TRUE(matrix_state->configured());

  // Test unconfigured (mtwiden = 0)
  matrix_state->set_mtype(0x0);
  EXPECT_EQ(matrix_state->mtwiden(), 0);
  EXPECT_EQ(matrix_state->twiden(), 0);
  EXPECT_FALSE(matrix_state->configured());

  // Test tk (bits 5-7)
  matrix_state->set_mtype(0x7 << 5);
  EXPECT_EQ(matrix_state->tile_k(), 7);

  // Test tm (bits 10-23)
  matrix_state->set_mtype(0x3FFF << 10);
  EXPECT_EQ(matrix_state->tile_m(), 0x3FFF);
}

TEST(RiscVZvtStateTest, MtypeCsrWriteTrapsIllegalInstruction) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);
  ASSERT_OK_AND_ASSIGN(auto matrix_state,
                       RiscVZvtMatrixState::Create(state.get(), 32));

  state->mstatus()->Set((state->mstatus()->GetUint32() & ~0x0600U) | (1U << 9));

  // Fetch CSR using the constant address.
  ASSERT_OK_AND_ASSIGN(auto* csr, state->csr_set()->GetCsr(
                                      mpact::sim::riscv::kRiscVMtypeCsrIndex));
  auto* mtype_csr = csr;

  // Set up the trap callback.
  bool trap_taken = false;
  uint64_t trap_code = 0;
  state->set_on_trap([&](bool, uint64_t, uint64_t exception_code, uint64_t,
                         const ::mpact::sim::generic::Instruction*) {
    trap_taken = true;
    trap_code = exception_code;
    return true;
  });

  // Attempt to write via CSR interface. This must trap since mtype is
  // read-only.
  mtype_csr->Write(static_cast<uint32_t>(0xDEADBEEF));
  EXPECT_TRUE(trap_taken);
  EXPECT_EQ(trap_code,
            static_cast<uint64_t>(
                ::mpact::sim::riscv::ExceptionCode::kIllegalInstruction));

  trap_taken = false;
  trap_code = 0;
  mtype_csr->Write(static_cast<uint64_t>(0xDEADBEEF));
  EXPECT_TRUE(trap_taken);
  EXPECT_EQ(trap_code,
            static_cast<uint64_t>(
                ::mpact::sim::riscv::ExceptionCode::kIllegalInstruction));
}

struct PunningOffsets {
  int ptile;
  int minor_offset;
  int major_offset;
  int expected_offset;
};

// Number of architectural tile-specifier steps between valid tiles for a given
// `tile_element_width` (Zvt v0.4 Section "Matrix Tile State"):
//   TEW=8  -> 16 tiles (step 1: mt0, mt1, ..., mt15)
//   TEW=16 ->  8 tiles (step 2: mt0, mt2, ..., mt14)
//   TEW=32 ->  4 tiles (step 4: mt0, mt4, mt8, mt12)
//   TEW=64 ->  8 tiles (step 2: mt0, mt2, ..., mt14)
int TileAlignmentForTew(int tile_element_width) {
  switch (tile_element_width) {
    case 8:
      return 1;
    case 16:
    case 64:
      return 2;
    case 32:
      return 4;
    default:
      return 1;
  }
}

// Effective number of elements along a tile edge (ETE) as a function of
// `tile_elements` (TE) and `tile_element_width` (TEW) (Zvt v0.4):
//   ETE = (TEW < 64) ? TE : (TE / 2).
int EffectiveTileEdge(int tile_elements, int tile_element_width) {
  return tile_element_width < 64 ? tile_elements : tile_elements / 2;
}

// Computes the Zvt v0.4 Section "Tile Punning" decomposition:
//   - `ptile`: physical TEW=8 tile index (0..15) in the 16*TE*TE-byte buffer
//   - `minor_offset`: byte offset (0..15) within the 16-byte minor block
//   - `major_offset`: index of the 16-byte minor block within `ptile`
//   - `expected_offset`: linear byte offset in `tile_buffer_`
PunningOffsets ComputeExpectedPunningOffsets(int tile_elements, int tile,
                                             int row, int col,
                                             int tile_element_width) {
  constexpr int kMinorBlockBytes = 16;
  constexpr int kSubblockEdge = 4;
  const int kSubblocksPerEdge = tile_elements / kSubblockEdge;
  int ptile = 0;
  int minor_offset = 0;
  int major_offset = 0;
  switch (tile_element_width) {
    case 8:
      ptile = tile;
      minor_offset =
          (row % kSubblockEdge) * kSubblockEdge + (col % kSubblockEdge);
      major_offset =
          (row / kSubblockEdge) * kSubblocksPerEdge + (col / kSubblockEdge);
      break;
    case 16:
      ptile = tile + ((row & 2) >> 1);
      minor_offset = (row % 2) * 4 + (col % 2) * 2 + ((col / 2) % 2) * 8;
      major_offset =
          (row / kSubblockEdge) * kSubblocksPerEdge + (col / kSubblockEdge);
      break;
    case 32:
      ptile = tile + 2 * (row / (tile_elements / 2)) + ((col & 2) >> 1);
      minor_offset = (row % 2) * 8 + (col % 2) * 4;
      major_offset = ((row / 2) % kSubblocksPerEdge) * kSubblocksPerEdge +
                     (col / kSubblockEdge);
      break;
    case 64:
      ptile = tile + (row / kSubblocksPerEdge);
      minor_offset = (col % 2) * 8;
      major_offset = (row % kSubblocksPerEdge) * kSubblocksPerEdge + (col / 2);
      break;
  }
  int expected_offset = (ptile * tile_elements * tile_elements) +
                        (major_offset * kMinorBlockBytes) + minor_offset;
  return {ptile, minor_offset, major_offset, expected_offset};
}

TEST(RiscVZvtStateTest, PunningTEW8) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);
  constexpr int kTileElements = 32;
  constexpr int kTileElementWidth = 8;
  const int kTileAlignment = TileAlignmentForTew(kTileElementWidth);
  const int kEffectiveTileEdge =
      EffectiveTileEdge(kTileElements, kTileElementWidth);
  ASSERT_OK_AND_ASSIGN(auto matrix_state,
                       RiscVZvtMatrixState::Create(state.get(), kTileElements));

  // 1) Origin corner: first valid tile (0), row 0, col 0.
  ASSERT_OK_AND_ASSIGN(
      int offset0,
      matrix_state->GetPunningOffset(/*tile=*/0, /*row=*/0, /*col=*/0,
                                     /*tile_element_width=*/kTileElementWidth));
  EXPECT_EQ(offset0, 0);

  // 2) Interior element: aligned tile specifier and interior (row, col) with
  // non-zero quotients and remainders modulo 2 and 4.
  const int kInteriorTile = 3 * kTileAlignment;
  constexpr int kInteriorRow = 10;
  constexpr int kInteriorCol = 7;
  auto [ptile1, minor_offset1, major_offset1, expected_offset1] =
      ComputeExpectedPunningOffsets(/*tile_elements=*/kTileElements,
                                    /*tile=*/kInteriorTile,
                                    /*row=*/kInteriorRow,
                                    /*col=*/kInteriorCol,
                                    /*tile_element_width=*/kTileElementWidth);
  EXPECT_LT(ptile1, 16);
  EXPECT_LT(minor_offset1, 16);
  EXPECT_LT(major_offset1, (kTileElements / 4) * (kTileElements / 4));
  ASSERT_OK_AND_ASSIGN(
      int offset1,
      matrix_state->GetPunningOffset(/*tile=*/kInteriorTile,
                                     /*row=*/kInteriorRow, /*col=*/kInteriorCol,
                                     /*tile_element_width=*/kTileElementWidth));
  EXPECT_EQ(offset1, expected_offset1);

  // 3) Upper-bound corner: last valid tile and last (row, col) within ETE.
  const int kMaxValidTile = 16 - kTileAlignment;
  const int kMaxRow = kEffectiveTileEdge - 1;
  const int kMaxCol = kEffectiveTileEdge - 1;
  auto [ptile2, minor_offset2, major_offset2, expected_offset2] =
      ComputeExpectedPunningOffsets(/*tile_elements=*/kTileElements,
                                    /*tile=*/kMaxValidTile,
                                    /*row=*/kMaxRow,
                                    /*col=*/kMaxCol,
                                    /*tile_element_width=*/kTileElementWidth);
  EXPECT_EQ(ptile2, 15);
  EXPECT_LT(minor_offset2, 16);
  EXPECT_LT(major_offset2, (kTileElements / 4) * (kTileElements / 4));
  ASSERT_OK_AND_ASSIGN(
      int offset2,
      matrix_state->GetPunningOffset(/*tile=*/kMaxValidTile, /*row=*/kMaxRow,
                                     /*col=*/kMaxCol,
                                     /*tile_element_width=*/kTileElementWidth));
  EXPECT_EQ(offset2, expected_offset2);
  EXPECT_EQ(offset2,
            matrix_state->tile_buffer_bytes() - (kTileElementWidth / 8));
}

TEST(RiscVZvtStateTest, PunningTEW16) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);
  constexpr int kTileElements = 32;
  constexpr int kTileElementWidth = 16;
  const int kTileAlignment = TileAlignmentForTew(kTileElementWidth);
  const int kEffectiveTileEdge =
      EffectiveTileEdge(kTileElements, kTileElementWidth);
  ASSERT_OK_AND_ASSIGN(auto matrix_state,
                       RiscVZvtMatrixState::Create(state.get(), kTileElements));

  // 1) Origin corner: first valid tile (0), row 0, col 0.
  ASSERT_OK_AND_ASSIGN(
      int offset0,
      matrix_state->GetPunningOffset(/*tile=*/0, /*row=*/0, /*col=*/0,
                                     /*tile_element_width=*/kTileElementWidth));
  EXPECT_EQ(offset0, 0);

  // 2) Interior element: aligned tile specifier and interior (row, col).
  const int kInteriorTile = 3 * kTileAlignment;
  constexpr int kInteriorRow = 10;
  constexpr int kInteriorCol = 7;
  auto [ptile1, minor_offset1, major_offset1, expected_offset1] =
      ComputeExpectedPunningOffsets(/*tile_elements=*/kTileElements,
                                    /*tile=*/kInteriorTile,
                                    /*row=*/kInteriorRow,
                                    /*col=*/kInteriorCol,
                                    /*tile_element_width=*/kTileElementWidth);
  EXPECT_LT(ptile1, 16);
  EXPECT_LT(minor_offset1, 16);
  EXPECT_LT(major_offset1, (kTileElements / 4) * (kTileElements / 4));
  ASSERT_OK_AND_ASSIGN(
      int offset1,
      matrix_state->GetPunningOffset(/*tile=*/kInteriorTile,
                                     /*row=*/kInteriorRow, /*col=*/kInteriorCol,
                                     /*tile_element_width=*/kTileElementWidth));
  EXPECT_EQ(offset1, expected_offset1);

  // 3) Upper-bound corner: last valid tile and last (row, col) within ETE.
  const int kMaxValidTile = 16 - kTileAlignment;
  const int kMaxRow = kEffectiveTileEdge - 1;
  const int kMaxCol = kEffectiveTileEdge - 1;
  auto [ptile2, minor_offset2, major_offset2, expected_offset2] =
      ComputeExpectedPunningOffsets(/*tile_elements=*/kTileElements,
                                    /*tile=*/kMaxValidTile,
                                    /*row=*/kMaxRow,
                                    /*col=*/kMaxCol,
                                    /*tile_element_width=*/kTileElementWidth);
  EXPECT_EQ(ptile2, 15);
  EXPECT_LT(minor_offset2, 16);
  EXPECT_LT(major_offset2, (kTileElements / 4) * (kTileElements / 4));
  ASSERT_OK_AND_ASSIGN(
      int offset2,
      matrix_state->GetPunningOffset(/*tile=*/kMaxValidTile, /*row=*/kMaxRow,
                                     /*col=*/kMaxCol,
                                     /*tile_element_width=*/kTileElementWidth));
  EXPECT_EQ(offset2, expected_offset2);
  EXPECT_EQ(offset2,
            matrix_state->tile_buffer_bytes() - (kTileElementWidth / 8));
}

TEST(RiscVZvtStateTest, PunningTEW32) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);
  constexpr int kTileElements = 32;
  constexpr int kTileElementWidth = 32;
  const int kTileAlignment = TileAlignmentForTew(kTileElementWidth);
  const int kEffectiveTileEdge =
      EffectiveTileEdge(kTileElements, kTileElementWidth);
  ASSERT_OK_AND_ASSIGN(auto matrix_state,
                       RiscVZvtMatrixState::Create(state.get(), kTileElements));

  // 1) Origin corner: first valid tile (0), row 0, col 0.
  ASSERT_OK_AND_ASSIGN(
      int offset0,
      matrix_state->GetPunningOffset(/*tile=*/0, /*row=*/0, /*col=*/0,
                                     /*tile_element_width=*/kTileElementWidth));
  EXPECT_EQ(offset0, 0);

  // 2) Interior element: aligned tile specifier and interior (row, col).
  const int kInteriorTile = 3 * kTileAlignment;
  constexpr int kInteriorRow = 10;
  constexpr int kInteriorCol = 7;
  auto [ptile1, minor_offset1, major_offset1, expected_offset1] =
      ComputeExpectedPunningOffsets(/*tile_elements=*/kTileElements,
                                    /*tile=*/kInteriorTile,
                                    /*row=*/kInteriorRow,
                                    /*col=*/kInteriorCol,
                                    /*tile_element_width=*/kTileElementWidth);
  EXPECT_LT(ptile1, 16);
  EXPECT_LT(minor_offset1, 16);
  EXPECT_LT(major_offset1, (kTileElements / 4) * (kTileElements / 4));
  ASSERT_OK_AND_ASSIGN(
      int offset1,
      matrix_state->GetPunningOffset(/*tile=*/kInteriorTile,
                                     /*row=*/kInteriorRow, /*col=*/kInteriorCol,
                                     /*tile_element_width=*/kTileElementWidth));
  EXPECT_EQ(offset1, expected_offset1);

  // 3) Upper-bound corner: last valid tile and last (row, col) within ETE.
  const int kMaxValidTile = 16 - kTileAlignment;
  const int kMaxRow = kEffectiveTileEdge - 1;
  const int kMaxCol = kEffectiveTileEdge - 1;
  auto [ptile2, minor_offset2, major_offset2, expected_offset2] =
      ComputeExpectedPunningOffsets(/*tile_elements=*/kTileElements,
                                    /*tile=*/kMaxValidTile,
                                    /*row=*/kMaxRow,
                                    /*col=*/kMaxCol,
                                    /*tile_element_width=*/kTileElementWidth);
  EXPECT_EQ(ptile2, 15);
  EXPECT_LT(minor_offset2, 16);
  EXPECT_LT(major_offset2, (kTileElements / 4) * (kTileElements / 4));
  ASSERT_OK_AND_ASSIGN(
      int offset2,
      matrix_state->GetPunningOffset(/*tile=*/kMaxValidTile, /*row=*/kMaxRow,
                                     /*col=*/kMaxCol,
                                     /*tile_element_width=*/kTileElementWidth));
  EXPECT_EQ(offset2, expected_offset2);
  EXPECT_EQ(offset2,
            matrix_state->tile_buffer_bytes() - (kTileElementWidth / 8));
}

TEST(RiscVZvtStateTest, PunningTEW64) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);
  constexpr int kTileElements = 32;
  constexpr int kTileElementWidth = 64;
  const int kTileAlignment = TileAlignmentForTew(kTileElementWidth);
  const int kEffectiveTileEdge =
      EffectiveTileEdge(kTileElements, kTileElementWidth);
  ASSERT_OK_AND_ASSIGN(auto matrix_state,
                       RiscVZvtMatrixState::Create(state.get(), kTileElements));

  // 1) Origin corner: first valid tile (0), row 0, col 0.
  ASSERT_OK_AND_ASSIGN(
      int offset0,
      matrix_state->GetPunningOffset(/*tile=*/0, /*row=*/0, /*col=*/0,
                                     /*tile_element_width=*/kTileElementWidth));
  EXPECT_EQ(offset0, 0);

  // 2) Interior element: aligned tile specifier and interior (row, col).
  const int kInteriorTile = 3 * kTileAlignment;
  constexpr int kInteriorRow = 10;
  constexpr int kInteriorCol = 7;
  auto [ptile1, minor_offset1, major_offset1, expected_offset1] =
      ComputeExpectedPunningOffsets(/*tile_elements=*/kTileElements,
                                    /*tile=*/kInteriorTile,
                                    /*row=*/kInteriorRow,
                                    /*col=*/kInteriorCol,
                                    /*tile_element_width=*/kTileElementWidth);
  EXPECT_LT(ptile1, 16);
  EXPECT_LT(minor_offset1, 16);
  EXPECT_LT(major_offset1, (kTileElements / 4) * (kTileElements / 4));
  ASSERT_OK_AND_ASSIGN(
      int offset1,
      matrix_state->GetPunningOffset(/*tile=*/kInteriorTile,
                                     /*row=*/kInteriorRow, /*col=*/kInteriorCol,
                                     /*tile_element_width=*/kTileElementWidth));
  EXPECT_EQ(offset1, expected_offset1);

  // 3) Upper-bound corner: last valid tile and last (row, col) within ETE.
  const int kMaxValidTile = 16 - kTileAlignment;
  const int kMaxRow = kEffectiveTileEdge - 1;
  const int kMaxCol = kEffectiveTileEdge - 1;
  auto [ptile2, minor_offset2, major_offset2, expected_offset2] =
      ComputeExpectedPunningOffsets(/*tile_elements=*/kTileElements,
                                    /*tile=*/kMaxValidTile,
                                    /*row=*/kMaxRow,
                                    /*col=*/kMaxCol,
                                    /*tile_element_width=*/kTileElementWidth);
  EXPECT_EQ(ptile2, 15);
  EXPECT_LT(minor_offset2, 16);
  EXPECT_LT(major_offset2, (kTileElements / 4) * (kTileElements / 4));
  ASSERT_OK_AND_ASSIGN(
      int offset2,
      matrix_state->GetPunningOffset(/*tile=*/kMaxValidTile, /*row=*/kMaxRow,
                                     /*col=*/kMaxCol,
                                     /*tile_element_width=*/kTileElementWidth));
  EXPECT_EQ(offset2, expected_offset2);
  EXPECT_EQ(offset2,
            matrix_state->tile_buffer_bytes() - (kTileElementWidth / 8));
}

TEST(RiscVZvtStateTest, TileSpecifierAlignmentAndMasking) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);
  ASSERT_OK_AND_ASSIGN(auto matrix_state,
                       RiscVZvtMatrixState::Create(state.get(), 32));

  // Without vector state, sew_bytes() defaults to 1 (SEW = 8 bits).
  // tew = sew(8) * twiden.
  // We manipulate twiden via mtwiden. twiden = 1 << (mtwiden - 1).

  // TEW=8 (mtwiden=1 => twiden=1). NTiles = 16. align = 1.
  matrix_state->set_mtype(1 << mpact::sim::riscv::kMtypeMtwidenShift);
  EXPECT_TRUE(matrix_state->IsValidTile(15));
  EXPECT_EQ(matrix_state->MaskTssTile(/*tile=*/15, /*tile_element_width=*/8),
            15);

  // TEW=16 (mtwiden=2 => twiden=2). NTiles = 8. align = 2.
  matrix_state->set_mtype(2 << mpact::sim::riscv::kMtypeMtwidenShift);
  EXPECT_TRUE(matrix_state->IsValidTile(0));
  EXPECT_TRUE(matrix_state->IsValidTile(2));
  EXPECT_FALSE(matrix_state->IsValidTile(3));
  EXPECT_EQ(matrix_state->MaskTssTile(/*tile=*/3, /*tile_element_width=*/16),
            2);
  EXPECT_EQ(matrix_state->MaskTssTile(/*tile=*/15, /*tile_element_width=*/16),
            14);

  // TEW=32 (mtwiden=3 => twiden=4). NTiles = 4. align = 4.
  matrix_state->set_mtype(3 << mpact::sim::riscv::kMtypeMtwidenShift);
  EXPECT_TRUE(matrix_state->IsValidTile(4));
  EXPECT_FALSE(matrix_state->IsValidTile(5));
  EXPECT_EQ(matrix_state->MaskTssTile(/*tile=*/7, /*tile_element_width=*/32),
            4);
  EXPECT_EQ(matrix_state->MaskTssTile(/*tile=*/15, /*tile_element_width=*/32),
            12);
}

TEST(RiscVZvtStateTest, TileSpecifierAlignmentAndMasking64) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);
  auto vector_state =
      std::make_unique<mpact::sim::riscv::RiscVVectorState>(state.get(), 16);
  // vtype[5:3] = 1 (SEW=16 bits)
  vector_state->SetVectorType(1 << 3);
  ASSERT_OK_AND_ASSIGN(auto matrix_state,
                       RiscVZvtMatrixState::Create(state.get(), 32));

  // TEW=64 (sew=16, mtwiden=3 => twiden=4). NTiles = 8. align = 2.
  matrix_state->set_mtype(3 << mpact::sim::riscv::kMtypeMtwidenShift);
  EXPECT_TRUE(matrix_state->IsValidTile(0));
  EXPECT_TRUE(matrix_state->IsValidTile(2));
  EXPECT_TRUE(matrix_state->IsValidTile(8));
  EXPECT_FALSE(matrix_state->IsValidTile(9));
  EXPECT_FALSE(matrix_state->IsValidTile(3));
  EXPECT_EQ(matrix_state->MaskTssTile(/*tile=*/15, /*tile_element_width=*/64),
            14);
  EXPECT_EQ(matrix_state->MaskTssTile(/*tile=*/7, /*tile_element_width=*/64),
            6);
}

TEST(RiscVZvtStateTest, VsetvlZeroesMtype) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);
  state->mstatus()->Set((state->mstatus()->GetUint32() & ~0x0600U) | (1U << 9));
  auto vector_state =
      std::make_unique<mpact::sim::riscv::RiscVVectorState>(state.get(), 16);
  vector_state->SetVectorType(0);
  ASSERT_OK_AND_ASSIGN(auto matrix_state,
                       RiscVZvtMatrixState::Create(state.get(), 32));

  matrix_state->set_mtype(0xFFFFFFFF);
  mpact::sim::generic::Instruction inst(0, state.get());
  inst.AppendSource(new mpact::sim::generic::ImmediateOperand<uint32_t>(32));
  inst.AppendSource(new mpact::sim::generic::ImmediateOperand<uint32_t>(0));
  auto* reg = state->GetRegister<mpact::sim::riscv::RV32Register>("x1").first;
  inst.AppendDestination(reg->CreateDestinationOperand(0));

  mpact::sim::riscv::Vsetvl(/*rd_zero=*/false, /*rs1_zero=*/false, &inst);
  EXPECT_EQ(matrix_state->mtype(), 0);
}

TEST(RiscVZvtStateTest, VsetvliZeroesMtype) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);
  state->mstatus()->Set((state->mstatus()->GetUint32() & ~0x0600U) | (1U << 9));
  auto vector_state =
      std::make_unique<mpact::sim::riscv::RiscVVectorState>(state.get(), 16);
  vector_state->SetVectorType(0);
  ASSERT_OK_AND_ASSIGN(auto matrix_state,
                       RiscVZvtMatrixState::Create(state.get(), 32));

  matrix_state->set_mtype(0xFFFFFFFF);
  mpact::sim::generic::Instruction inst(0, state.get());
  inst.AppendSource(new mpact::sim::generic::ImmediateOperand<uint32_t>(32));
  inst.AppendSource(new mpact::sim::generic::ImmediateOperand<uint32_t>(0));
  auto* reg = state->GetRegister<mpact::sim::riscv::RV32Register>("x1").first;
  inst.AppendDestination(reg->CreateDestinationOperand(0));

  mpact::sim::riscv::Vsetvl(/*rd_zero=*/false, /*rs1_zero=*/true, &inst);
  EXPECT_EQ(matrix_state->mtype(), 0);
}

TEST(RiscVZvtStateTest, VsetivliZeroesMtype) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);
  state->mstatus()->Set((state->mstatus()->GetUint32() & ~0x0600U) | (1U << 9));
  auto vector_state =
      std::make_unique<mpact::sim::riscv::RiscVVectorState>(state.get(), 16);
  vector_state->SetVectorType(0);
  ASSERT_OK_AND_ASSIGN(auto matrix_state,
                       RiscVZvtMatrixState::Create(state.get(), 32));

  matrix_state->set_mtype(0xFFFFFFFF);
  mpact::sim::generic::Instruction inst(0, state.get());
  inst.AppendSource(new mpact::sim::generic::ImmediateOperand<uint32_t>(32));
  inst.AppendSource(new mpact::sim::generic::ImmediateOperand<uint32_t>(0));
  auto* reg = state->GetRegister<mpact::sim::riscv::RV32Register>("x0").first;
  inst.AppendDestination(reg->CreateDestinationOperand(0));

  mpact::sim::riscv::Vsetvl(/*rd_zero=*/true, /*rs1_zero=*/true, &inst);
  EXPECT_EQ(matrix_state->mtype(), 0);
}

TEST(RiscVZvtStateTest, MtypeCsrTrapWhenVsOff) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);
  bool trap_taken = false;
  uint64_t trap_code = 0;
  state->set_on_trap([&trap_taken, &trap_code](
                         bool, uint64_t, uint64_t exception_code, uint64_t,
                         const mpact::sim::generic::Instruction*) {
    trap_taken = true;
    trap_code = exception_code;
    return true;
  });

  ASSERT_OK_AND_ASSIGN(auto matrix_state,
                       RiscVZvtMatrixState::Create(state.get(), 32));

  auto* mstatus = state->mstatus();
  mstatus->Set(mstatus->GetUint32() & ~0x0600U);  // VS bits [10:9] = 0 (Off)

  ASSERT_OK_AND_ASSIGN(
      auto* mtype,
      state->csr_set()->GetCsr(mpact::sim::riscv::kRiscVMtypeCsrIndex));

  trap_taken = false;
  mtype->AsUint32();
  EXPECT_TRUE(trap_taken);
  EXPECT_EQ(trap_code,
            static_cast<uint64_t>(
                ::mpact::sim::riscv::ExceptionCode::kIllegalInstruction));

  trap_taken = false;
  mtype->AsUint64();
  EXPECT_TRUE(trap_taken);
  EXPECT_EQ(trap_code,
            static_cast<uint64_t>(
                ::mpact::sim::riscv::ExceptionCode::kIllegalInstruction));
}

TEST(RiscVZvtStateTest, MtypeCsrSucceedsWhenMsOffAndVsOn) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);
  bool trap_taken = false;
  state->set_on_trap([&trap_taken](bool, uint64_t, uint64_t, uint64_t,
                                   const mpact::sim::generic::Instruction*) {
    trap_taken = true;
    return true;
  });

  ASSERT_OK_AND_ASSIGN(auto matrix_state,
                       RiscVZvtMatrixState::Create(state.get(), 32));

  auto* mstatus = state->mstatus();
  // VS bits [10:9] = 1 (Initial), MS bits [30:29] = 0 (Off).
  mstatus->Set((mstatus->GetUint32() & ~0x6000'0600U) | (1U << 9));

  ASSERT_OK_AND_ASSIGN(
      auto* mtype,
      state->csr_set()->GetCsr(mpact::sim::riscv::kRiscVMtypeCsrIndex));

  trap_taken = false;
  mtype->AsUint32();
  EXPECT_FALSE(trap_taken);

  mtype->AsUint64();
  EXPECT_FALSE(trap_taken);
}

TEST(RiscVZvtStateTest, CreateValidatesTileElements) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);
  auto matrix_state_res = RiscVZvtMatrixState::Create(state.get(), 0);
  EXPECT_FALSE(matrix_state_res.ok());
  EXPECT_EQ(matrix_state_res.status().code(),
            absl::StatusCode::kInvalidArgument);

  auto matrix_state_neg_res = RiscVZvtMatrixState::Create(state.get(), -4);
  EXPECT_FALSE(matrix_state_neg_res.ok());
  EXPECT_EQ(matrix_state_neg_res.status().code(),
            absl::StatusCode::kInvalidArgument);

  // Non-power-of-two tile_elements >= 4 must be rejected.
  auto matrix_state_non_pow2_res = RiscVZvtMatrixState::Create(state.get(), 6);
  EXPECT_FALSE(matrix_state_non_pow2_res.ok());
  EXPECT_EQ(matrix_state_non_pow2_res.status().code(),
            absl::StatusCode::kInvalidArgument);

  // Creating a second matrix state on the same RiscVState fails because the
  // mtype CSR is already registered.
  ASSERT_OK_AND_ASSIGN(auto matrix_state,
                       RiscVZvtMatrixState::Create(state.get(), 32));
  auto duplicate_res = RiscVZvtMatrixState::Create(state.get(), 32);
  EXPECT_FALSE(duplicate_res.ok());
}

TEST(RiscVZvtStateTest, GetScratchDbAllocatesAndCaches) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);
  ASSERT_OK_AND_ASSIGN(auto matrix_state,
                       RiscVZvtMatrixState::Create(state.get(), 32));

  // Allocate a scratch DataBuffer of size 16.
  auto* db16 = matrix_state->GetScratchDb(16);
  ASSERT_NE(db16, nullptr);
  EXPECT_EQ(db16->size<uint8_t>(), 16);

  // Requesting the same size again returns the cached DataBuffer.
  EXPECT_EQ(matrix_state->GetScratchDb(16), db16);

  // Requesting a smaller size that has not been allocated yet allocates it
  // without resizing scratch_dbs_.
  auto* db8 = matrix_state->GetScratchDb(8);
  ASSERT_NE(db8, nullptr);
  EXPECT_EQ(db8->size<uint8_t>(), 8);
  EXPECT_NE(db8, db16);
}

TEST(RiscVZvtStateTest, GetAndSetElemAndInvalidWidthErrorPaths) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);
  ASSERT_OK_AND_ASSIGN(auto matrix_state,
                       RiscVZvtMatrixState::Create(state.get(), 32));

  // Invalid tile element width in GetPunningOffset, SetElem, and GetElem.
  EXPECT_EQ(matrix_state
                ->GetPunningOffset(/*tile=*/0, /*row=*/0, /*col=*/0,
                                   /*tile_element_width=*/12)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(matrix_state
                ->SetElem<uint32_t>(/*tile=*/0, /*row=*/0, /*col=*/0,
                                    /*tile_element_width=*/12, /*val=*/42U)
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(matrix_state
                ->GetElem<uint32_t>(/*tile=*/0, /*row=*/0, /*col=*/0,
                                    /*tile_element_width=*/12)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);

  // Valid SetElem and GetElem round-trip.
  ASSERT_OK(matrix_state->SetElem<uint32_t>(
      /*tile=*/0, /*row=*/1, /*col=*/2, /*tile_element_width=*/32,
      /*val=*/0xCAFEBABEU));
  ASSERT_OK_AND_ASSIGN(uint32_t val, matrix_state->GetElem<uint32_t>(
                                         /*tile=*/0, /*row=*/1, /*col=*/2,
                                         /*tile_element_width=*/32));
  EXPECT_EQ(val, 0xCAFEBABEU);
}

TEST(RiscVZvtStateTest, GetMatrixStateLookup) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);
  mpact::sim::generic::Instruction inst(0, state.get());

  // Before RiscVZvtMatrixState::Create, mtype CSR is not registered.
  auto missing_res = mpact::sim::riscv::GetMatrixState(&inst);
  EXPECT_FALSE(missing_res.ok());

  ASSERT_OK_AND_ASSIGN(auto matrix_state,
                       RiscVZvtMatrixState::Create(state.get(), 32));
  ASSERT_OK_AND_ASSIGN(auto* found_state,
                       mpact::sim::riscv::GetMatrixState(&inst));
  EXPECT_EQ(found_state, matrix_state.get());
}

}  // namespace
