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

// Architectural state for the RISC-V Vector Matrix Extension (Zvt / "VME",
// spec v0.3, chapter 15.1). This is layered on top of the existing RISC-V
// vector state: the matrix unit sources its A/B operands from vector registers
// and accumulates results into the new square matrix "tile" state modeled here.
//
// NOTE: Zvt is a draft (v0.3) extension. The mtype CSR field positions below
// follow the v0.3 manual (15.1.1.2). They are isolated as named constants so
// they can be adjusted if the encoding firms up.

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_ZVT_STATE_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_ZVT_STATE_H_

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_state.h"

namespace mpact::sim::riscv {

// mtype CSR address (unprivileged, read-only). Spec 15.1.1.2.
inline constexpr uint64_t kRiscVMtypeCsrIndex = 0xC23;

// mtype field layout (see note above re: draft status).
//   [1:0]   mtwiden  (0 => unconfigured; otherwise TWIDEN = 1 << (mtwiden-1))
//   [7:5]   tk       (0..4; bounded by KMAX). K is the inner dimension of the
//   dot product.
//   [23:10] tm       (0..TE). M is the number of rows of the
//   accumulator tile.
inline constexpr uint32_t kMtypeMtwidenShift = 0;
inline constexpr uint32_t kMtypeMtwidenMask = 0x3;
inline constexpr uint32_t kMtypeTkShift = 5;
inline constexpr uint32_t kMtypeTkMask = 0x7;
inline constexpr uint32_t kMtypeTmShift = 10;
inline constexpr uint32_t kMtypeTmMask = 0x3FFF;

// Default tile-edge dimension (TE). The tile state is fixed at the
// implementation maximum; the runtime-usable extents are bounded by tk/tm/tn.
// Constraint (spec 15.1.1.1): TE a power of two, VLEN/4 >= TE >= 4. With
// VLEN = 128 bits (kRiscVV2VectorByteLength = 16 bytes), VLEN/4 = 32.
inline constexpr int kRiscVZvtDefaultTe = 32;

class RiscVZvtMatrixState;

// The read-only `mtype` CSR. The value lives in the owning matrix state; this
// CSR is a read-only view (it mirrors mtype the way RiscVVl/RiscVVtype mirror
// the vector state). Software cannot write it (write mask is 0 so `csrw mtype`
// is a no-op); mtype is mutated only by the matrix configuration instructions.
// The back-pointer also lets semantic functions recover the matrix state from
// the CSR set (see GetMatrixState()).
class RiscVMtypeCsr : public ::mpact::sim::riscv::RiscVSimpleCsr<uint32_t> {
 public:
  RiscVMtypeCsr(RiscVZvtMatrixState* matrix_state,
                ::mpact::sim::riscv::RiscVState* state)
      : ::mpact::sim::riscv::RiscVSimpleCsr<uint32_t>(
            "mtype", kRiscVMtypeCsrIndex, /*read_mask=*/0xFFFFFFFFu,
            /*write_mask=*/0x0u, state),
        matrix_state_(matrix_state) {}

  // Overrides: read the live value from the matrix state.
  uint32_t AsUint32() override;
  uint64_t AsUint64() override;

  [[nodiscard]] RiscVZvtMatrixState* matrix_state() const {
    return matrix_state_;
  }

 private:
  RiscVZvtMatrixState* matrix_state_;
};

// Matrix tile state. Owns the flat tile buffer (16 * TE * TE bytes) and the
// mtype CSR. All element access goes through Punning()/GetElem()/SetElem() so
// the tile-aliasing ("punning") layout is implemented exactly once.
class RiscVZvtMatrixState {
 public:
  static absl::StatusOr<std::unique_ptr<RiscVZvtMatrixState>> Create(
      ::mpact::sim::riscv::RiscVState* /* absl_nonnull */ state,
      int tile_elements = kRiscVZvtDefaultTe);

  ~RiscVZvtMatrixState();

  // Tile-edge dimension and total buffer size (bytes).
  [[nodiscard]] int tile_elements() const { return tile_elements_; }
  [[nodiscard]] int tile_buffer_bytes() const {
    return tile_buffer_->size<uint8_t>();
  }

  [[nodiscard]] ::mpact::sim::riscv::RiscVState* riscv_state() const {
    return state_;
  }

  // Byte offset into tile_buffer_ for element (tile,row,col) at the given tile
  // element width `tile_element_width` (in bits: 8/16/32/64). Transcribed
  // verbatim from the Zvt v0.3 spec (15.1.1.1, "Tile Punning"). `tile` is the
  // named tile index (0..15); `tile_element_width` selects the aliasing scheme.
  int Punning(int tile, int row, int col, int tile_element_width) const;

  template <typename T>
  T GetElem(int tile, int row, int col, int tile_element_width) const {
    T value;
    auto span = tile_buffer_->Get<uint8_t>();
    std::memcpy(&value, &span[Punning(tile, row, col, tile_element_width)],
                sizeof(T));
    return value;
  }

  template <typename T>
  void SetElem(int tile, int row, int col, int tile_element_width, T value) {
    auto span = tile_buffer_->Get<uint8_t>();
    std::memcpy(&span[Punning(tile, row, col, tile_element_width)], &value,
                sizeof(T));
  }

  // mtype accessors. The mtype value is held here; the CSR is a read-only view.
  [[nodiscard]] uint32_t mtype() const { return mtype_; }
  void set_mtype(uint32_t value) { mtype_ = value; }

  // Decoded mtype fields.
  [[nodiscard]] int mtwiden() const {
    return (mtype() >> kMtypeMtwidenShift) & kMtypeMtwidenMask;
  }
  // TWIDEN = 1 << (mtwiden - 1) when configured, else 0.
  [[nodiscard]] int twiden() const {
    return mtwiden() == 0 ? 0 : (1 << (mtwiden() - 1));
  }
  bool configured() const { return mtwiden() != 0; }
  // K is the inner dimension of the dot product.
  [[nodiscard]] int tile_k() const {
    return (mtype() >> kMtypeTkShift) & kMtypeTkMask;
  }
  // M is the number of rows of the accumulator tile.
  [[nodiscard]] int tile_m() const {
    return (mtype() >> kMtypeTmShift) & kMtypeTmMask;
  }

  void set_tile_k(int value) { set_field(kMtypeTkShift, kMtypeTkMask, value); }
  void set_tile_m(int value) { set_field(kMtypeTmShift, kMtypeTmMask, value); }

  // Returns a pre-allocated scratchpad DataBuffer of exactly the requested
  // size.
  ::mpact::sim::generic::DataBuffer* GetScratchDb(int size);

  // SEW (bytes/bits) from the current vector configuration.
  [[nodiscard]] int sew_bytes() const;
  [[nodiscard]] int sew_bits() const { return sew_bytes() * 8; }
  // TILE_ELEMENT_WIDTH (tile element width) in bits = SEW * TWIDEN.
  [[nodiscard]] int tile_element_width_bits() const {
    return sew_bits() * twiden();
  }
  // EFFECTIVE_TILE_EDGE: effective tile edge for the current
  // TILE_ELEMENT_WIDTH. = TE for TILE_ELEMENT_WIDTH<64, TE/2 else.
  [[nodiscard]] int effective_tile_edge() const {
    return tile_element_width_bits() < 64 ? tile_elements_ : tile_elements_ / 2;
  }

 private:
  explicit RiscVZvtMatrixState(
      ::mpact::sim::riscv::RiscVState* /* absl_nonnull */ state,
      int tile_elements);

  void set_field(uint32_t shift, uint32_t mask, int value) {
    uint32_t v = mtype();
    v &= ~(mask << shift);
    v |= (static_cast<uint32_t>(value) & mask) << shift;
    set_mtype(v);
  }

  ::mpact::sim::riscv::RiscVState* state_;
  int tile_elements_;
  ::mpact::sim::generic::DataBuffer* tile_buffer_;  // 16 * TE * TE bytes.
  std::vector<::mpact::sim::generic::DataBuffer*> scratch_dbs_;
  uint32_t mtype_ = 0;
  RiscVMtypeCsr mtype_csr_;
};

// Recovers the matrix state from an executing instruction by looking up the
// mtype CSR (cheap hash lookup; same idiom as the existing kCsr operand
// getter). Returns an error if the Zvt matrix state is not installed.
absl::StatusOr<RiscVZvtMatrixState*> GetMatrixState(
    const ::mpact::sim::generic::Instruction* inst);

}  // namespace mpact::sim::riscv

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_ZVT_STATE_H_
