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

// Functional and regression unit tests for the RISC-V Vector Matrix Extension
// (Zvt v0.4) specification in the MPACT simulator.
//
// This test suite verifies architectural compliance and functional correctness
// across the following instruction and subsystem categories:
//
// 1. Matrix Configuration Instructions:
//    - msetmtype, msetmtypei: dynamic tile element width (TEW), widening factor
//      (TWIDEN), vector length (VL) zeroing, tail/mask policy mapping, and
//      vill invalidation state machine.
//    - msettn, msettm, msettk: dimension configuration, bound clamping (KMAX,
//      MMAX, NMAX), and illegal instruction traps on invalid TWIDEN/SEW.
//
// 2. Tile Movement Instructions:
//    - vtmv.v.t, vtmv.t.v: bidirectional register transfers between vector
//      register groups and matrix tiles across SEW=8/16/32/64 and LMUL=1/2/4/8.
//    - vtzero, vtdiscard: tile clearing, canonical tile ID validation, and
//      status transitions (MS downgrade to Initial).
//
// 3. Tile Memory Instructions:
//    - vtle*, vtse*: strided and unit-stride tile loads/stores, row/column TSS
//      patterns, address alignment, exception prioritization (reserved TSS
//      trap priority over misaligned address trap), and vstart restart bounds.
//
// 4. Matrix Multiplication Instructions:
//    - vtmmu.tvv, vtmms.tvv: integer unsigned and signed matrix multiplication
//      across valid/invalid tile topologies.
//    - vtfmm.tvv, vtfmm.alt.tvv: floating-point matrix multiplication
//      supporting FP8 (E4M3, E5M2), FP16, BF16, FP32, and FP64 arithmetic,
//      tile alignment rules, and status tracking (FS dirtied on execution).
//
// 5. Architectural Context & Status Tracking:
//    - mstatus.MS (Matrix Status), mstatus.VS (Vector Status), and mstatus.FS
//      (Floating-Point Status) dirty/initial/off lifecycle state tracking,
//      vsstatus virtualization, and SD bit calculation across RV32 and RV64.

#include "riscv/riscv_zvt_instructions.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>  // IWYU pragma: keep
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"  // IWYU pragma: keep
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/operand_interface.h"
#include "mpact/sim/generic/register.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_fp_info.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_instruction_helpers.h"
#include "riscv/riscv_misa.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_memory_instructions.h"
#include "riscv/riscv_vector_state.h"
#include "riscv/riscv_xstatus.h"
#include "riscv/riscv_zvt_state.h"

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

using ::mpact::sim::generic::ArchState;
using ::mpact::sim::generic::operator*;  // NOLINT(misc-unused-using-decls)
using ::mpact::sim::generic::Bfloat16;
using ::mpact::sim::generic::ConvertFp8E4m3ToSingle;
using ::mpact::sim::generic::ConvertFp8E5m2ToSingle;
using ::mpact::sim::generic::DestinationOperandInterface;
using ::mpact::sim::generic::Fp8E4m3;
using ::mpact::sim::generic::Fp8E5m2;
using ::mpact::sim::generic::ImmediateOperand;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::generic::RegisterBase;
using ::mpact::sim::generic::SourceOperandInterface;
using ::mpact::sim::riscv::DoubleToFloatROD;
using ::mpact::sim::riscv::ExceptionCode;
using ::mpact::sim::riscv::FPExceptions;
using ::mpact::sim::riscv::FPRoundingMode;
using ::mpact::sim::riscv::kMstatusFsMask;
using ::mpact::sim::riscv::kMstatusFsShift;
using ::mpact::sim::riscv::kMstatusMsMask;
using ::mpact::sim::riscv::kMstatusMsShift;
using ::mpact::sim::riscv::kMstatusSdMaskRv32;
using ::mpact::sim::riscv::kMstatusSdMaskRv64;
using ::mpact::sim::riscv::kMstatusSdShiftRv32;
using ::mpact::sim::riscv::kMstatusSdShiftRv64;
using ::mpact::sim::riscv::kMstatusStateDirty;
using ::mpact::sim::riscv::kMstatusStateInitial;
using ::mpact::sim::riscv::kMstatusStateOff;
using ::mpact::sim::riscv::kMstatusVsMask;
using ::mpact::sim::riscv::kMstatusVsShift;
using ::mpact::sim::riscv::kMtypeMtwidenShift;
using ::mpact::sim::riscv::kMtypeTkMask;
using ::mpact::sim::riscv::kMtypeTkShift;
using ::mpact::sim::riscv::kMtypeTmMask;
using ::mpact::sim::riscv::kMtypeTmShift;
using ::mpact::sim::riscv::kRiscVMtypeCsrIndex;
using ::mpact::sim::riscv::RiscVCsrEnum;
using ::mpact::sim::riscv::RiscVFPState;
using ::mpact::sim::riscv::RiscVMIsa;
using ::mpact::sim::riscv::RiscVMStatus;
using ::mpact::sim::riscv::RiscVSimpleCsr;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVVectorState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::RiscVZvtMatrixState;
using ::mpact::sim::riscv::RiscVZvtMsetMtype;
using ::mpact::sim::riscv::RiscVZvtMsetMtypei;
using ::mpact::sim::riscv::RiscVZvtMsetTk;
using ::mpact::sim::riscv::RiscVZvtMsetTm;
using ::mpact::sim::riscv::RiscVZvtMsetTn;
using ::mpact::sim::riscv::RiscVZvtVtDiscard;
using ::mpact::sim::riscv::RiscVZvtVtfmm;
using ::mpact::sim::riscv::RiscVZvtVtfmmAlt;
using ::mpact::sim::riscv::RiscVZvtVtle;
using ::mpact::sim::riscv::RiscVZvtVtmms;
using ::mpact::sim::riscv::RiscVZvtVtmmu;
using ::mpact::sim::riscv::RiscVZvtVtmvTv;
using ::mpact::sim::riscv::RiscVZvtVtmvVt;
using ::mpact::sim::riscv::RiscVZvtVtse;
using ::mpact::sim::riscv::RiscVZvtVtZero;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RV32VectorDestinationOperand;
using ::mpact::sim::riscv::RV32VectorSourceOperand;
using ::mpact::sim::riscv::RV64Register;
using ::mpact::sim::riscv::RVVectorRegister;
using ::mpact::sim::riscv::Vsetvl;
using ::mpact::sim::util::FlatDemandMemory;
using ::mpact::sim::util::MemoryInterface;

// mstatus.XS (user extension state) field constants.
inline constexpr uint32_t kMstatusXsShift = 15;
inline constexpr uint32_t kMstatusXsMask = 0b11U << kMstatusXsShift;

class TestMStatus : public RiscVMStatus {
 public:
  TestMStatus(uint64_t initial_value, ArchState* state, RiscVMIsa* misa)
      : RiscVMStatus(initial_value, state, misa, ~0ULL, ~0ULL) {}
  uint32_t vs() { return (GetUint32() & kMstatusVsMask) >> kMstatusVsShift; }
  void set_vs(uint32_t value) {
    uint32_t v =
        (GetUint32() & ~kMstatusVsMask) | ((value & 0b11U) << kMstatusVsShift);
    UpdateSdAndSet(v);
  }
  uint32_t ms() { return (GetUint32() & kMstatusMsMask) >> kMstatusMsShift; }
  void set_ms(uint32_t value) {
    uint32_t v =
        (GetUint32() & ~kMstatusMsMask) | ((value & 0b11U) << kMstatusMsShift);
    UpdateSdAndSet(v);
  }
  uint32_t fs() { return (GetUint32() & kMstatusFsMask) >> kMstatusFsShift; }
  void set_fs(uint32_t value) {
    uint32_t v =
        (GetUint32() & ~kMstatusFsMask) | ((value & 0b11U) << kMstatusFsShift);
    UpdateSdAndSet(v);
  }
  uint32_t sd() {
    return ((vs() == kMstatusStateDirty) || (fs() == kMstatusStateDirty) ||
            (ms() == kMstatusStateDirty) ||
            (((GetUint32() & kMstatusXsMask) >> kMstatusXsShift) ==
             kMstatusStateDirty))
               ? 1
               : 0;
  }
  void Submit() {}

 private:
  void UpdateSdAndSet(uint32_t v) {
    bool is_sd =
        (((v & kMstatusMsMask) >> kMstatusMsShift) == kMstatusStateDirty) ||
        (((v & kMstatusVsMask) >> kMstatusVsShift) == kMstatusStateDirty) ||
        (((v & kMstatusFsMask) >> kMstatusFsShift) == kMstatusStateDirty) ||
        (((v & kMstatusXsMask) >> kMstatusXsShift) == kMstatusStateDirty);
    if (is_sd) {
      v |= kMstatusSdMaskRv32;
    } else {
      v &= ~kMstatusSdMaskRv32;
    }
    Set(v);
  }
};

// CSR index for vsstatus (virtual supervisor status).
inline constexpr uint32_t kVsstatusCsrIndex = 0x200;

class RiscVVSStatus : public RiscVSimpleCsr<uint64_t> {
 public:
  RiscVVSStatus(std::string name, RiscVCsrEnum csr_enum, uint64_t initial_value,
                ArchState* state, TestMStatus* mstatus)
      : RiscVSimpleCsr<uint64_t>(std::move(name), csr_enum, initial_value,
                                 ~0ULL, ~0ULL, state),
        mstatus_(mstatus) {}
  uint32_t vs() { return mstatus_->vs(); }
  void set_vs(uint32_t value) { mstatus_->set_vs(value); }
  uint32_t fs() { return mstatus_->fs(); }
  void set_fs(uint32_t value) { mstatus_->set_fs(value); }
  uint32_t ms() { return (GetUint32() & kMstatusMsMask) >> kMstatusMsShift; }
  void set_ms(uint32_t value) {
    Set((GetUint32() & ~kMstatusMsMask) | ((value & 0b11U) << kMstatusMsShift));
  }
  uint32_t sd() { return mstatus_->sd(); }
  void Submit() {}

 private:
  TestMStatus* mstatus_;
};

class TestRiscVState : public RiscVState {
 public:
  TestRiscVState(absl::string_view id, RiscVXlen xlen, MemoryInterface* memory)
      : RiscVState(id, xlen, memory) {
    uint64_t old_val = mstatus_->GetUint64();
    csr_set()->RemoveCsr(*RiscVCsrEnum::kMStatus).IgnoreError();
    test_mstatus_ = std::make_unique<TestMStatus>(old_val, this, misa());
    mstatus_ = test_mstatus_.get();
    csr_set()->AddCsr(mstatus_).IgnoreError();
    vsstatus_ = std::make_unique<RiscVVSStatus>(
        "vsstatus", static_cast<RiscVCsrEnum>(kVsstatusCsrIndex), 0, this,
        test_mstatus_.get());
    csr_set()->AddCsr(vsstatus_.get()).IgnoreError();
  }
  TestMStatus* mstatus() const { return test_mstatus_.get(); }
  RiscVVSStatus* vsstatus() const { return vsstatus_.get(); }
  bool v() const { return v_; }
  void set_v(bool v) { v_ = v; }

 private:
  bool v_ = false;
  std::unique_ptr<TestMStatus> test_mstatus_;
  std::unique_ptr<RiscVVSStatus> vsstatus_;
};

// Architectural mstatus SD bit states.
inline constexpr uint64_t kMstatusSdSet = 1ULL;
inline constexpr uint64_t kMstatusSdClean = 0ULL;

// RVV vtype bitfield encodings.
inline constexpr uint32_t kVsew8 = 0;   // 8-bit SEW (1 byte)
inline constexpr uint32_t kVsew16 = 1;  // 16-bit SEW (2 bytes)
inline constexpr uint32_t kVsew32 = 2;  // 32-bit SEW (4 bytes)
inline constexpr uint32_t kVsew64 = 3;  // 64-bit SEW (8 bytes)
inline constexpr uint32_t kVsewShift = 3;

inline constexpr uint32_t kVlmul1 = 0;   // LMUL = 1
inline constexpr uint32_t kVlmul2 = 1;   // LMUL = 2
inline constexpr uint32_t kVlmul4 = 2;   // LMUL = 4
inline constexpr uint32_t kVlmul8 = 3;   // LMUL = 8
inline constexpr uint32_t kVlmulF8 = 5;  // LMUL = 1/8
inline constexpr uint32_t kVlmulF4 = 6;  // LMUL = 1/4
inline constexpr uint32_t kVlmulF2 = 7;  // LMUL = 1/2
inline constexpr uint32_t kVlmulMask = 0x7;

inline constexpr uint32_t kVtaShift = 6;
inline constexpr uint32_t kVmaShift = 7;
inline constexpr uint32_t kVtypeAltfmtShift = 8;
inline constexpr uint32_t kVillMask = 1U << 31;

// Returns the alternate format (altfmt) bit from vtype.
inline uint32_t GetVtypeAltfmt(const RiscVVectorState* vector_state) {
  return (vector_state->vtype() >> kVtypeAltfmtShift) & 1U;
}

// Assembles an RVV / Zvt vtype configuration value from individual fields.
constexpr uint32_t EncodeVtype(uint32_t vsew, uint32_t vlmul, bool vta = false,
                               bool vma = false, bool altfmt = false) {
  return (vsew << kVsewShift) | (vlmul & kVlmulMask) |
         (static_cast<uint32_t>(vta) << kVtaShift) |
         (static_cast<uint32_t>(vma) << kVmaShift) |
         (static_cast<uint32_t>(altfmt) << kVtypeAltfmtShift);
}

// Zvt mtype mtwiden encodings.
inline constexpr uint32_t kMtwidenUnconfigured = 0;
inline constexpr uint32_t kMtwiden1 = 1;  // TWIDEN = 1 (no widening, TEW = SEW)
inline constexpr uint32_t kMtwiden2 = 2;  // TWIDEN = 2 (2x, TEW = 2 * SEW)
inline constexpr uint32_t kMtwiden4 = 3;  // TWIDEN = 4 (4x, TEW = 4 * SEW)

// Zvt msetmtypei altfmt encodings.
inline constexpr uint32_t kAltfmtStandard = 0;
inline constexpr uint32_t kAltfmtAlternate = 1;

// Assembles a Zvt mtype configuration value from mtwiden, tk, and tm fields.
constexpr uint32_t EncodeMtype(uint32_t mtwiden, uint32_t tk = 0,
                               uint32_t tm = 0) {
  return (mtwiden << kMtypeMtwidenShift) |
         ((tk & kMtypeTkMask) << kMtypeTkShift) |
         ((tm & kMtypeTmMask) << kMtypeTmShift);
}

// Zvt Tile Subset Specifier (TSS) encodings (spec 15.1.1.5).
inline constexpr uint32_t kTssPatternRow = 0;
inline constexpr uint32_t kTssPatternCol = 1;
inline constexpr uint32_t kTssPatternInvalid = 2;
inline constexpr uint32_t kTssPatternShift = 24;
inline constexpr uint32_t kTssTileShift = 27;

// FCSR rounding-mode bitfield offset.
inline constexpr uint32_t kFcsrFrmShift = 5;

// Assembles a 32-bit Tile Subset Specifier (TSS) value.
constexpr uint32_t EncodeTss(uint32_t tile, uint32_t pattern, uint32_t index) {
  return ((tile & 0xF) << kTssTileShift) |
         ((pattern & 0x7) << kTssPatternShift) | (index & 0xFFFFFF);
}

// Holds expected vector geometry scaling results.
struct ExpectedVectorGeometry {
  uint32_t vlmul = 0;
  uint32_t max_vector_length = 0;
};

// Calculates the expected architectural LMUL and VLMAX resulting from
// Zvt vector-matrix geometry scaling (Zvt v0.4 spec 15.1.1.4).
ExpectedVectorGeometry CalculateExpectedVectorGeometry(int vlen_bytes,
                                                       int tile_elements,
                                                       int sew_bits,
                                                       int twiden) {
  const int sew_bytes = sew_bits / 8;
  const int tew = sew_bits * twiden;
  // Effective tile elements scale inversely with element width above 32 bits.
  const int effective_tile_elements =
      (tew > 32) ? (tile_elements / (tew / 32)) : tile_elements;
  const int max_k = std::max(1, 32 / sew_bits);

  const int eve = vlen_bytes / sew_bytes;
  const int ceil_ete_eve = (effective_tile_elements + eve - 1) / eve;

  const int vlmul_max_k = 64 / max_k;
  const int vlmul_twiden = 64 / twiden;
  const int vlmul_ete_eve = 8 * ceil_ete_eve;

  int target_vlmul_scaled =
      std::min({vlmul_max_k, vlmul_twiden, vlmul_ete_eve});
  target_vlmul_scaled = std::clamp(target_vlmul_scaled, 1, 64);

  uint32_t vlmul_enc = kVlmul1;
  int lmul_num = 1;
  int lmul_den = 1;
  switch (target_vlmul_scaled) {
    case 1:
      vlmul_enc = kVlmulF8;
      lmul_den = 8;
      break;
    case 2:
      vlmul_enc = kVlmulF4;
      lmul_den = 4;
      break;
    case 4:
      vlmul_enc = kVlmulF2;
      lmul_den = 2;
      break;
    case 8:
      vlmul_enc = kVlmul1;
      lmul_num = 1;
      break;
    case 16:
      vlmul_enc = kVlmul2;
      lmul_num = 2;
      break;
    case 32:
      vlmul_enc = kVlmul4;
      lmul_num = 4;
      break;
    case 64:
      vlmul_enc = kVlmul8;
      lmul_num = 8;
      break;
    default:
      vlmul_enc = kVlmul1;
      lmul_num = 1;
      break;
  }
  const uint32_t max_vl = (eve * lmul_num) / lmul_den;
  return {.vlmul = vlmul_enc, .max_vector_length = max_vl};
}

// RAII test helper that intercepts trap callbacks on a RiscVState instance
// and records whether a trap was taken along with its exception code.
class ScopedTrapCapture {
 public:
  explicit ScopedTrapCapture(RiscVState* state) : state_(state) {
    state_->set_on_trap([this](bool, uint64_t trap_value, uint64_t code,
                               uint64_t, const Instruction*) {
      trap_taken_ = true;
      trap_code_ = code;
      trap_value_ = trap_value;
      return true;
    });
  }

  ~ScopedTrapCapture() {
    if (state_ != nullptr) {
      state_->set_on_trap(nullptr);
    }
  }

  ScopedTrapCapture(const ScopedTrapCapture&) = delete;
  ScopedTrapCapture& operator=(const ScopedTrapCapture&) = delete;

  bool trap_taken() const { return trap_taken_; }
  uint64_t trap_code() const { return trap_code_; }
  uint64_t trap_value() const { return trap_value_; }

 private:
  RiscVState* state_;
  bool trap_taken_ = false;
  uint64_t trap_code_ = 0;
  uint64_t trap_value_ = 0;
};

// Encapsulates a localized 64-bit RISC-V state, vector state, and matrix
// state along with mstatus status accessors for RV64-specific instruction
// tests.
struct Rv64TestContext {
  FlatDemandMemory memory;
  RiscVState state;
  RiscVVectorState vector_state;
  std::unique_ptr<RiscVZvtMatrixState> matrix_state;

  explicit Rv64TestContext(absl::string_view name)
      : state(name, RiscVXlen::RV64, &memory), vector_state(&state, 128) {
    state.set_rv_vector(&vector_state);
    vector_state.SetVectorType(EncodeVtype(kVsew8, kVlmul1));
  }

  static absl::StatusOr<std::unique_ptr<Rv64TestContext>> Create(
      absl::string_view name, int tile_elements = 32) {
    auto ctx = std::make_unique<Rv64TestContext>(name);
    auto ms_or = RiscVZvtMatrixState::Create(&ctx->state, tile_elements);
    if (!ms_or.ok()) return ms_or.status();
    ctx->matrix_state = *std::move(ms_or);
    return ctx;
  }

  void SetMstatusVs(uint32_t state_val) {
    uint64_t val = state.mstatus()->GetUint64() & ~kMstatusVsMask;
    val |= (static_cast<uint64_t>(state_val & 3U) << kMstatusVsShift);
    state.mstatus()->Set(val);
  }

  void SetMstatusMs(uint32_t state_val) {
    uint64_t val = state.mstatus()->GetUint64() & ~kMstatusMsMask;
    val |= (static_cast<uint64_t>(state_val & 3U) << kMstatusMsShift);
    state.mstatus()->Set(val);
  }

  uint32_t GetMstatusVs() const {
    return (state.mstatus()->GetUint32() >> kMstatusVsShift) &
           kMstatusStateDirty;
  }

  uint32_t GetMstatusMs() const {
    return (state.mstatus()->GetUint32() >> kMstatusMsShift) &
           kMstatusStateDirty;
  }

  uint64_t GetMstatusSd() const {
    return (state.mstatus()->GetUint64() >> kMstatusSdShiftRv64) & 1ULL;
  }
};

class RiscVZvtInstructionsTest : public testing::Test {
 protected:
  void SetUp() override {
    state_ =
        std::make_unique<TestRiscVState>("test", RiscVXlen::RV32, &memory_);
    fp_state_ = std::make_unique<RiscVFPState>(state_->csr_set(), state_.get());
    state_->set_rv_fp(fp_state_.get());
    state_->mstatus()->set_fs(kMstatusStateInitial);

    vector_state_ =
        std::make_unique<RiscVVectorState>(state_.get(), 128);  // 128 bytes
    state_->set_rv_vector(vector_state_.get());

    ASSERT_OK_AND_ASSIGN(matrix_state_,
                         RiscVZvtMatrixState::Create(state_.get(), 32));

    instruction_.reset(new Instruction(0x1000, state_.get()));
    instruction_->set_size(4);
  }

  ~RiscVZvtInstructionsTest() override = default;

  void ResetInstruction() {
    instruction_.reset(new Instruction(0x1000, state_.get()));
    instruction_->set_size(4);
  }

  // Appends a vector group source operand constructed from 'count' consecutive
  // vector registers starting at 'base_reg'.
  void AppendVectorGroupSource(int base_reg, int count = 8) {
    std::vector<RegisterBase*> reg_vec;
    reg_vec.reserve(count);
    for (int i = 0; i < count; ++i) {
      reg_vec.push_back(
          state_->GetRegister<RVVectorRegister>(absl::StrCat("v", base_reg + i))
              .first);
    }
    instruction_->AppendSource(new RV32VectorSourceOperand(
        absl::MakeSpan(reg_vec), absl::StrCat("v", base_reg)));
  }

  // Appends a vector group destination operand constructed from 'count'
  // consecutive vector registers starting at 'base_reg'.
  void AppendVectorGroupDestination(int base_reg, int latency = 0,
                                    int count = 8) {
    std::vector<RegisterBase*> reg_vec;
    reg_vec.reserve(count);
    for (int i = 0; i < count; ++i) {
      reg_vec.push_back(
          state_->GetRegister<RVVectorRegister>(absl::StrCat("v", base_reg + i))
              .first);
    }
    instruction_->AppendDestination(new RV32VectorDestinationOperand(
        absl::MakeSpan(reg_vec), latency, absl::StrCat("v", base_reg)));
  }

  // Sets the value of scalar register 'name' to 'value'.
  void SetRegisterValue(absl::string_view name, uint32_t value) {
    auto* reg = state_->GetRegister<RV32Register>(name).first;
    reg->data_buffer()->Set<uint32_t>(0, value);
  }

  // Returns the value of scalar register 'name'.
  uint32_t GetRegisterValue(absl::string_view name) const {
    return state_->GetRegister<RV32Register>(name)
        .first->data_buffer()
        ->Get<uint32_t>(0);
  }

  // Sets the mstatus.MS (matrix state) field to 'state_val'.
  void SetMstatusMs(uint32_t state_val) {
    uint64_t val = state_->mstatus()->GetUint64() & ~kMstatusMsMask;
    val |= (static_cast<uint64_t>(state_val & 3U) << kMstatusMsShift);
    state_->mstatus()->Set(val);
  }

  // Sets the mstatus.VS (vector state) field to 'state_val'.
  void SetMstatusVs(uint32_t state_val) {
    uint64_t val = state_->mstatus()->GetUint64() & ~kMstatusVsMask;
    val |= (static_cast<uint64_t>(state_val & 3U) << kMstatusVsShift);
    state_->mstatus()->Set(val);
  }

  // Returns the mstatus.VS (vector state) field value.
  uint32_t GetMstatusVs() const {
    return (state_->mstatus()->GetUint32() >> kMstatusVsShift) &
           kMstatusStateDirty;
  }

  // Returns the mstatus.MS (matrix state) field value.
  uint32_t GetMstatusMs() const {
    return (state_->mstatus()->GetUint32() >> kMstatusMsShift) &
           kMstatusStateDirty;
  }

  // Returns the mstatus.SD (state dirty) bit value.
  uint64_t GetMstatusSd(bool is_rv64 = false) const {
    int shift = is_rv64 ? kMstatusSdShiftRv64 : kMstatusSdShiftRv32;
    return (state_->mstatus()->GetUint64() >> shift) & 1ULL;
  }

  // Returns the vsstatus CSR pointer.
  RiscVVSStatus* vsstatus() const { return state_->vsstatus(); }

  // Creates a scalar source operand for register 'name'.
  SourceOperandInterface* CreateScalarSource(absl::string_view name) {
    return state_->GetRegister<RV32Register>(name).first->CreateSourceOperand();
  }

  // Creates a scalar destination operand for register 'name'.
  DestinationOperandInterface* CreateScalarDest(absl::string_view name) {
    return state_->GetRegister<RV32Register>(name)
        .first->CreateDestinationOperand(0);
  }

  // Creates a vector destination operand for vector register group 'reg_name'.
  RV32VectorDestinationOperand* CreateVectorDestOperand(
      absl::string_view reg_name) {
    auto* reg = state_->GetRegister<RVVectorRegister>(reg_name).first;
    RegisterBase* regs[] = {reg};
    return new RV32VectorDestinationOperand(absl::MakeSpan(regs), 0,
                                            std::string(reg_name));
  }

  // Creates a vector source operand for vector register group 'reg_name'.
  RV32VectorSourceOperand* CreateVectorSourceOperand(
      absl::string_view reg_name) {
    auto* reg = state_->GetRegister<RVVectorRegister>(reg_name).first;
    RegisterBase* regs[] = {reg};
    return new RV32VectorSourceOperand(absl::MakeSpan(regs),
                                       std::string(reg_name));
  }

  // Appends all 5 ISA source operands (matA, matB, mtile, vs2_idx, vs1_idx)
  // for Zvt matrix-multiply instructions.
  void AppendMatmulSources(uint32_t reg_a, uint32_t reg_b,
                           uint32_t dest_tile = 0) {
    std::vector<RegisterBase*> reg_vec_a;
    reg_vec_a.reserve(std::min<size_t>(8, 32 - reg_a));
    for (int i = 0; i < 8 && (reg_a + i) < 32; ++i) {
      reg_vec_a.push_back(
          state_->GetRegister<RVVectorRegister>(absl::StrCat("v", reg_a + i))
              .first);
    }
    instruction_->AppendSource(new RV32VectorSourceOperand(
        absl::MakeSpan(reg_vec_a), absl::StrCat("v", reg_a)));

    std::vector<RegisterBase*> reg_vec_b;
    reg_vec_b.reserve(std::min<size_t>(8, 32 - reg_b));
    for (int i = 0; i < 8 && (reg_b + i) < 32; ++i) {
      reg_vec_b.push_back(
          state_->GetRegister<RVVectorRegister>(absl::StrCat("v", reg_b + i))
              .first);
    }
    instruction_->AppendSource(new RV32VectorSourceOperand(
        absl::MakeSpan(reg_vec_b), absl::StrCat("v", reg_b)));

    instruction_->AppendSource(new ImmediateOperand<uint32_t>(dest_tile));
    instruction_->AppendSource(new ImmediateOperand<uint32_t>(reg_a));
    instruction_->AppendSource(new ImmediateOperand<uint32_t>(reg_b));
  }

  // Executes 'semantic_function' on a state lacking matrix extension support
  // and returns the captured trap code, or an error if no trap occurred.
  absl::StatusOr<uint64_t> ExecuteWithoutMatrixStateExpectTrap(
      void (*semantic_function)(const Instruction*)) {
    FlatDemandMemory memory;
    RiscVState rv_state("test_no_matrix", RiscVXlen::RV32, &memory);
    ScopedTrapCapture trap_capture(&rv_state);

    auto* inst = new Instruction(0x1000, &rv_state);
    inst->set_size(4);
    inst->set_semantic_function(semantic_function);
    inst->Execute(nullptr);
    inst->DecRef();

    if (!trap_capture.trap_taken()) {
      return absl::FailedPreconditionError("Expected trap was not taken");
    }
    return trap_capture.trap_code();
  }

  // Enables matrix, vector, and floating-point extensions in mstatus/vsstatus
  // and configures mtype, vtype, vl, and vstart for a test scenario.
  void ConfigureMatrixUnit(uint32_t mtwiden, uint32_t vsew, uint32_t vlmul,
                           uint32_t vl, uint32_t tk = 0, uint32_t tm = 0,
                           uint32_t vstart = 0, bool vta = false,
                           bool vma = false, bool altfmt = false) {
    state_->mstatus()->set_ms(kMstatusStateInitial);
    state_->mstatus()->set_vs(kMstatusStateInitial);
    state_->mstatus()->set_fs(kMstatusStateInitial);
    state_->mstatus()->Submit();
    vsstatus()->set_ms(kMstatusStateInitial);
    vsstatus()->set_fs(kMstatusStateInitial);

    matrix_state_->set_mtype(EncodeMtype(mtwiden, tk, tm));
    vector_state_->SetVectorType(EncodeVtype(vsew, vlmul, vta, vma, altfmt));
    vector_state_->set_vector_length(vl);
    vector_state_->set_vstart(vstart);
  }

  // Reads a single element at (tile, row, col) from the active matrix state.
  template <typename T>
  absl::StatusOr<T> GetTileElem(uint32_t tile, uint32_t row, uint32_t col,
                                int elem_bits = sizeof(T) * 8) {
    return matrix_state_->GetElem<T>(tile, row, col, elem_bits);
  }

  // Writes 'values' into row 'row' of 'tile' starting at column 0.
  template <typename T>
  absl::Status SetTileRow(uint32_t tile, uint32_t row,
                          absl::Span<const T> values,
                          int elem_bits = sizeof(T) * 8) {
    for (int c = 0; c < static_cast<int>(values.size()); ++c) {
      absl::Status status =
          matrix_state_->SetElem<T>(tile, row, c, elem_bits, values[c]);
      if (!status.ok()) return status;
    }
    return absl::OkStatus();
  }

  template <typename T>
  absl::Status SetTileRow(uint32_t tile, uint32_t row,
                          std::initializer_list<T> values,
                          int elem_bits = sizeof(T) * 8) {
    return SetTileRow(tile, row, absl::Span<const T>(values), elem_bits);
  }

  // Populates 'count' columns of 'row' in 'tile' using 'generator(col)'.
  template <typename T, typename GeneratorFn>
  absl::Status SetTileRowPattern(uint32_t tile, uint32_t row, int count,
                                 GeneratorFn generator,
                                 int elem_bits = sizeof(T) * 8) {
    for (int c = 0; c < count; ++c) {
      absl::Status status = matrix_state_->SetElem<T>(
          tile, row, c, elem_bits, static_cast<T>(generator(c)));
      if (!status.ok()) return status;
    }
    return absl::OkStatus();
  }

  // Populates a small 2D matrix literal starting at (0, 0) in 'tile'.
  template <typename T>
  absl::Status SetTileMatrix(
      uint32_t tile, std::initializer_list<std::initializer_list<T>> rows,
      int elem_bits = sizeof(T) * 8) {
    int r = 0;
    for (const auto& row : rows) {
      absl::Status status = SetTileRow<T>(tile, r++, row, elem_bits);
      if (!status.ok()) return status;
    }
    return absl::OkStatus();
  }

  // Writes 'values' into column 'col' of 'tile' starting at row 0.
  template <typename T>
  absl::Status SetTileColumn(uint32_t tile, uint32_t col,
                             absl::Span<const T> values,
                             int elem_bits = sizeof(T) * 8) {
    for (int r = 0; r < static_cast<int>(values.size()); ++r) {
      absl::Status status =
          matrix_state_->SetElem<T>(tile, r, col, elem_bits, values[r]);
      if (!status.ok()) return status;
    }
    return absl::OkStatus();
  }

  template <typename T>
  absl::Status SetTileColumn(uint32_t tile, uint32_t col,
                             std::initializer_list<T> values,
                             int elem_bits = sizeof(T) * 8) {
    return SetTileColumn(tile, col, absl::Span<const T>(values), elem_bits);
  }

  // Populates a rows x cols region of 'tile' using 'generator(row, col)'.
  template <typename T, typename GeneratorFn>
  absl::Status FillTilePattern(uint32_t tile, int rows, int cols,
                               GeneratorFn generator,
                               int elem_bits = sizeof(T) * 8) {
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        absl::Status status = matrix_state_->SetElem<T>(
            tile, r, c, elem_bits, static_cast<T>(generator(r, c)));
        if (!status.ok()) return status;
      }
    }
    return absl::OkStatus();
  }

  // Populates a rows x cols region of 'tile' with a constant value.
  template <typename T>
  absl::Status FillTileConstant(uint32_t tile, int rows, int cols, T value,
                                int elem_bits = sizeof(T) * 8) {
    return FillTilePattern<T>(
        tile, rows, cols, [value](int, int) { return value; }, elem_bits);
  }

  // Writes 'values' into the front of vector register 'reg_name'.
  template <typename T>
  void WriteVectorRegister(absl::string_view reg_name,
                           absl::Span<const T> values) {
    auto* reg = state_->GetRegister<RVVectorRegister>(reg_name).first;
    auto* db = state_->db_factory()->MakeCopyOf(reg->data_buffer());
    auto span = db->template Get<T>();
    std::copy(values.begin(), values.end(), span.begin());
    reg->SetDataBuffer(db);
    db->DecRef();
  }

  template <typename T>
  void WriteVectorRegister(absl::string_view reg_name,
                           std::initializer_list<T> values) {
    WriteVectorRegister(reg_name, absl::Span<const T>(values));
  }

  // Fills all elements of type T in vector register 'reg_name' with 'value'.
  template <typename T>
  void FillVectorRegister(absl::string_view reg_name, T value) {
    auto* reg = state_->GetRegister<RVVectorRegister>(reg_name).first;
    auto* db = state_->db_factory()->MakeCopyOf(reg->data_buffer());
    auto span = db->template Get<T>();
    std::fill(span.begin(), span.end(), value);
    reg->SetDataBuffer(db);
    db->DecRef();
  }

  // Returns a read-only view of the first 'count' elements in 'reg_name'.
  template <typename T>
  absl::Span<const T> ReadVectorRegister(absl::string_view reg_name,
                                         size_t count) const {
    auto* reg = state_->GetRegister<RVVectorRegister>(reg_name).first;
    return reg->data_buffer()->template Get<T>().subspan(0, count);
  }

  // Writes 'values' to 'memory_' starting at byte address 'addr'.
  template <typename T>
  void WriteMemory(uint64_t addr, absl::Span<const T> values) {
    auto* db = state_->db_factory()->Allocate(values.size() * sizeof(T));
    auto span = db->template Get<T>();
    std::copy(values.begin(), values.end(), span.begin());
    memory_.Store(addr, db);
    db->DecRef();
  }

  template <typename T>
  void WriteMemory(uint64_t addr, std::initializer_list<T> values) {
    WriteMemory(addr, absl::Span<const T>(values));
  }

  // Writes 'count' elements generated by 'generator(index)' to 'memory_'.
  template <typename T, typename GeneratorFn>
  void WriteMemoryPattern(uint64_t addr, size_t count, GeneratorFn generator) {
    auto* db = state_->db_factory()->Allocate(count * sizeof(T));
    auto span = db->template Get<T>();
    for (size_t i = 0; i < count; ++i) {
      span[i] = static_cast<T>(generator(i));
    }
    memory_.Store(addr, db);
    db->DecRef();
  }

  // Writes 'count' copies of 'value' to 'memory_' starting at 'addr'.
  template <typename T>
  void WriteMemoryConstant(uint64_t addr, size_t count, T value) {
    WriteMemoryPattern<T>(addr, count, [value](size_t) { return value; });
  }

  // Reads 'count' elements of type T from 'memory_' starting at 'addr'.
  template <typename T>
  std::vector<T> ReadMemory(uint64_t addr, size_t count) {
    auto* db = state_->db_factory()->Allocate(count * sizeof(T));
    memory_.Load(addr, db, nullptr, nullptr);
    auto span = db->template Get<T>();
    std::vector<T> result(span.begin(), span.begin() + count);
    db->DecRef();
    return result;
  }

  // Executes vtmv.v.t with encoded 'tss' into vector register group 'vd_reg'.
  void ExecuteVtmvVt(uint32_t tss, int vd_reg = 8) {
    instruction_->AppendSource(new ImmediateOperand<uint32_t>(tss));
    AppendVectorGroupDestination(vd_reg);
    instruction_->set_semantic_function(&RiscVZvtVtmvVt);
    instruction_->Execute(nullptr);
  }

  // Executes vtmv.t.v from vector register group 'vs_reg' into tile slice
  // 'tss'.
  void ExecuteVtmvTv(int vs_reg, uint32_t tss) {
    AppendVectorGroupSource(vs_reg);
    instruction_->AppendSource(new ImmediateOperand<uint32_t>(tss));
    instruction_->set_semantic_function(&RiscVZvtVtmvTv);
    instruction_->Execute(nullptr);
  }

  // Executes a Zvt tile memory semantic function (vtle or vtse) with EEW 'eew',
  // base address 'base_addr', and encoded TSS 'tss'.
  void ExecuteTileMemoryOp(void (*semantic_fn)(int, const Instruction*),
                           int eew, uint64_t base_addr, uint32_t tss) {
    instruction_->AppendSource(new ImmediateOperand<uint64_t>(base_addr));
    instruction_->AppendSource(new ImmediateOperand<uint32_t>(tss));
    instruction_->set_semantic_function(
        [semantic_fn, eew](const Instruction* inst) {
          semantic_fn(eew, inst);
        });
    instruction_->Execute(nullptr);
  }

  // Executes a Zvt matrix-multiply semantic function with source register
  // groups 'reg_a' and 'reg_b' accumulating into 'dest_tile'.
  void ExecuteMatmul(void (*semantic_fn)(const Instruction*),
                     uint32_t reg_a = 8, uint32_t reg_b = 16,
                     uint32_t dest_tile = 0) {
    AppendMatmulSources(reg_a, reg_b, dest_tile);
    instruction_->set_semantic_function(semantic_fn);
    instruction_->Execute(nullptr);
  }

  // Reads a rows x cols region of 'tile' in row-major order into a vector.
  template <typename T>
  absl::StatusOr<std::vector<T>> GetTileRegion(uint32_t tile, int rows,
                                               int cols,
                                               int elem_bits = sizeof(T) * 8) {
    std::vector<T> elements;
    elements.reserve(rows * cols);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        auto elem_or = matrix_state_->GetElem<T>(tile, r, c, elem_bits);
        if (!elem_or.ok()) return elem_or.status();
        elements.push_back(*elem_or);
      }
    }
    return elements;
  }

  // Reads 'count' elements from row 'row' of tile 'tile' into a vector.
  template <typename T>
  absl::StatusOr<std::vector<T>> GetTileRow(uint32_t tile, uint32_t row,
                                            int count,
                                            int elem_bits = sizeof(T) * 8) {
    std::vector<T> elements;
    elements.reserve(count);
    for (int c = 0; c < count; ++c) {
      auto elem_or = matrix_state_->GetElem<T>(tile, row, c, elem_bits);
      if (!elem_or.ok()) return elem_or.status();
      elements.push_back(*elem_or);
    }
    return elements;
  }

  // Reads 'count' elements from column 'col' of tile 'tile' into a vector.
  template <typename T>
  absl::StatusOr<std::vector<T>> GetTileColumn(uint32_t tile, uint32_t col,
                                               int count,
                                               int elem_bits = sizeof(T) * 8) {
    std::vector<T> elements;
    elements.reserve(count);
    for (int r = 0; r < count; ++r) {
      auto elem_or = matrix_state_->GetElem<T>(tile, r, col, elem_bits);
      if (!elem_or.ok()) return elem_or.status();
      elements.push_back(*elem_or);
    }
    return elements;
  }

  FlatDemandMemory memory_;
  std::unique_ptr<TestRiscVState> state_;
  std::unique_ptr<RiscVFPState> fp_state_;
  std::unique_ptr<RiscVVectorState> vector_state_;
  std::unique_ptr<RiscVZvtMatrixState> matrix_state_;
  struct InstructionDeleter {
    void operator()(Instruction* ptr) const {
      if (ptr) ptr->DecRef();
    }
  };
  std::unique_ptr<Instruction, InstructionDeleter> instruction_;
};

TEST_F(RiscVZvtInstructionsTest, MsetmtypeFunctionalDecoding) {
  // Configure matrix state mtype with TWIDEN = 1 (no widening, mtwiden = 1)
  // and vector state vtype with SEW = 32 bits, LMUL = 1/8.
  const uint32_t mtype_val = EncodeMtype(kMtwiden1);
  const uint32_t vtype_val = EncodeVtype(kVsew32, kVlmulF8);
  SetRegisterValue("x1", mtype_val);
  SetRegisterValue("x2", vtype_val);

  instruction_->AppendSource(CreateScalarSource("x1"));
  instruction_->AppendSource(CreateScalarSource("x2"));

  instruction_->set_semantic_function(&RiscVZvtMsetMtype);
  instruction_->Execute(nullptr);

  // Semantics state vl = 0; mtype = rs1; vtype = rs2 (as if by vsetvl)
  EXPECT_EQ(matrix_state_->mtype(), mtype_val);
  EXPECT_EQ(state_->rv_vector()->vector_length(), 0);
  EXPECT_EQ(state_->rv_vector()->selected_element_width(), 4);
}

TEST_F(RiscVZvtInstructionsTest, MsetmtypeDestinationZeroRegister) {
  // Verifies that msetmtype does not write to any destination register,
  // matching the architectural requirement that rd = 0b00000 (x0) and acts
  // only as a structural encoding constraint, not a true destination operand.
  const uint32_t mtype_val = EncodeMtype(kMtwiden1);
  const uint32_t vtype_val = EncodeVtype(kVsew32, kVlmulF8);
  SetRegisterValue("x1", mtype_val);
  SetRegisterValue("x2", vtype_val);

  instruction_->AppendSource(CreateScalarSource("x1"));
  instruction_->AppendSource(CreateScalarSource("x2"));

  instruction_->set_semantic_function(&RiscVZvtMsetMtype);
  instruction_->Execute(nullptr);

  // Assert no destination operands are appended or written by the semantic
  // function
  EXPECT_EQ(instruction_->DestinationsSize(), 0);
}

TEST_F(RiscVZvtInstructionsTest, MsetmtypeiFunctionalDecoding) {
  // mtwiden = 1 (TWIDEN = 1), vsew = 0 (SEW = 8 bits), altfmt = 0 (standard).
  instruction_->AppendSource(
      new ImmediateOperand<uint32_t>(/*mtwiden=*/kMtwiden1));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vsew=*/kVsew8));
  instruction_->AppendSource(
      new ImmediateOperand<uint32_t>(/*altfmt=*/kAltfmtStandard));

  instruction_->set_semantic_function(&RiscVZvtMsetMtypei);
  instruction_->Execute(nullptr);

  // Semantics state vl = 0; mtype.mtwiden = 1
  EXPECT_EQ(matrix_state_->mtwiden(), kMtwiden1);
  EXPECT_EQ(state_->rv_vector()->vector_length(), 0);
  EXPECT_EQ(state_->rv_vector()->selected_element_width(), 1);
  EXPECT_EQ(GetVtypeAltfmt(state_->rv_vector()), 0);
}

TEST_F(RiscVZvtInstructionsTest, MsetmtypeiImmediateDecoding) {
  // mtwiden = 1 (TWIDEN = 1), vsew = 2 (SEW = 32 bits), altfmt = 1 (alternate).
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(kMtwiden1));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(kVsew32));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(kAltfmtAlternate));

  instruction_->set_semantic_function(&RiscVZvtMsetMtypei);
  instruction_->Execute(nullptr);

  EXPECT_EQ(matrix_state_->mtwiden(), kMtwiden1);
  EXPECT_EQ(state_->rv_vector()->vector_length(), 0);
  EXPECT_EQ(state_->rv_vector()->selected_element_width(), 4);
  EXPECT_EQ(state_->rv_vector()->vtype() & kVlmulMask, kVlmul1);
  EXPECT_EQ(state_->rv_vector()->max_vector_length(), 32);
  EXPECT_EQ(GetVtypeAltfmt(state_->rv_vector()), 1);
}

TEST_F(RiscVZvtInstructionsTest, MsettnFunctionalDecoding) {
  // Case 1: mtwiden != 0, rs1 > bounds -> clamps to min(VLMAX,
  // effective_tile_edge)
  SetRegisterValue("x1", 500);  // Exceeds bounds
  instruction_->AppendSource(CreateScalarSource("x1"));
  instruction_->AppendDestination(CreateScalarDest("x2"));

  instruction_->set_semantic_function(&RiscVZvtMsetTn);
  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  vector_state_->SetVectorType(
      EncodeVtype(kVsew8, kVlmul1));  // SEW = 8 bits, kmax = 4, VLMAX = 128.

  instruction_->Execute(nullptr);
  // effective_tile_edge for 8-bit is 32. min(128, 32) = 32.
  EXPECT_EQ(state_->rv_vector()->vector_length(), 32);
  EXPECT_EQ(GetRegisterValue("x2"), 32);
  EXPECT_EQ(GetMstatusVs(), kMstatusStateDirty);
  EXPECT_EQ(GetMstatusMs(), kMstatusStateDirty);
  EXPECT_EQ(GetMstatusSd(), kMstatusSdSet);

  // Case 2: mtwiden == 0, rs1 > bounds -> clamps to VLMAX
  SetMstatusVs(kMstatusStateInitial);
  EXPECT_EQ(GetMstatusVs(), kMstatusStateInitial);

  matrix_state_->set_mtype(EncodeMtype(kMtwidenUnconfigured));
  instruction_->Execute(nullptr);
  EXPECT_EQ(state_->rv_vector()->vector_length(), 128);
  EXPECT_EQ(GetRegisterValue("x2"), 128);
  EXPECT_EQ(GetMstatusVs(), kMstatusStateDirty);
}

TEST_F(RiscVZvtInstructionsTest, MsettmFunctionalDecoding) {
  SetRegisterValue("x1", 500);
  instruction_->AppendSource(CreateScalarSource("x1"));
  instruction_->AppendDestination(CreateScalarDest("x2"));

  instruction_->set_semantic_function(&RiscVZvtMsetTm);

  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  vector_state_->SetVectorType(
      EncodeVtype(kVsew8, kVlmul1));  // SEW = 8 bits, VLMAX = 128, bound = 32.

  instruction_->Execute(nullptr);
  EXPECT_EQ(matrix_state_->tile_m(), 32);
  EXPECT_EQ(GetRegisterValue("x2"), 32);
  EXPECT_EQ(GetMstatusVs(), kMstatusStateDirty);
  EXPECT_EQ(GetMstatusMs(), kMstatusStateDirty);
  EXPECT_EQ(GetMstatusSd(), kMstatusSdSet);

  SetMstatusVs(kMstatusStateInitial);
  EXPECT_EQ(GetMstatusVs(), kMstatusStateInitial);

  // mtwiden = 0, but retain active TM in mtype to ensure msettm traps
  // with kIllegalInstruction and leaves state unchanged.
  matrix_state_->set_mtype(
      EncodeMtype(kMtwidenUnconfigured, /*tk=*/0, /*tm=*/32));
  ASSERT_EQ(matrix_state_->tile_m(), 32);
  ScopedTrapCapture trap_capture(state_.get());
  instruction_->Execute(nullptr);
  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
  EXPECT_EQ(matrix_state_->tile_m(), 32);
  EXPECT_EQ(GetRegisterValue("x2"), 32);
  EXPECT_EQ(GetMstatusVs(), kMstatusStateInitial);
}

TEST_F(RiscVZvtInstructionsTest, MsettkFunctionalDecoding) {
  SetRegisterValue("x1", 500);  // Exceeds Kmax
  instruction_->AppendSource(CreateScalarSource("x1"));
  instruction_->AppendDestination(CreateScalarDest("x2"));

  instruction_->set_semantic_function(&RiscVZvtMsetTk);

  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  vector_state_->SetVectorType(
      EncodeVtype(kVsew8, kVlmul1));  // SEW = 8 bits, Kmax = 32 / 8 = 4.

  instruction_->Execute(nullptr);
  EXPECT_EQ(matrix_state_->tile_k(), 4);
  EXPECT_EQ(GetRegisterValue("x2"), 4);
  EXPECT_EQ(GetMstatusVs(), kMstatusStateDirty);
  EXPECT_EQ(GetMstatusMs(), kMstatusStateDirty);
  EXPECT_EQ(GetMstatusSd(), kMstatusSdSet);

  SetMstatusVs(kMstatusStateInitial);
  EXPECT_EQ(GetMstatusVs(), kMstatusStateInitial);

  // mtwiden = 0, but retain active TK in mtype to ensure msettk traps
  // with kIllegalInstruction and leaves state unchanged.
  matrix_state_->set_mtype(
      EncodeMtype(kMtwidenUnconfigured, /*tk=*/4, /*tm=*/0));
  ASSERT_EQ(matrix_state_->tile_k(), 4);
  ScopedTrapCapture trap_capture(state_.get());
  instruction_->Execute(nullptr);
  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
  EXPECT_EQ(matrix_state_->tile_k(), 4);
  EXPECT_EQ(GetRegisterValue("x2"), 4);
  EXPECT_EQ(GetMstatusVs(), kMstatusStateInitial);
}

TEST_F(RiscVZvtInstructionsTest, DynamicVlmulResolution) {
  // mtwiden = 1 -> TWIDEN = 1
  uint32_t rs1_val = EncodeMtype(kMtwiden1);
  // overridden.
  uint32_t rs2_val = EncodeVtype(kVsew16, kVlmul8);

  SetRegisterValue("x1", rs1_val);
  SetRegisterValue("x2", rs2_val);

  auto* reg1 = state_->GetRegister<RV32Register>("x1").first;
  auto* reg2 = state_->GetRegister<RV32Register>("x2").first;

  instruction_->AppendSource(reg1->CreateSourceOperand());
  instruction_->AppendSource(reg2->CreateSourceOperand());

  instruction_->set_semantic_function(&RiscVZvtMsetMtype);

  instruction_->Execute(nullptr);

  // EXPECT_EQ LMUL=1 (represented internally as 8).
  EXPECT_EQ(state_->rv_vector()->vector_length_multiplier(), 8);
}

TEST_F(RiscVZvtInstructionsTest, MtypeTkKmaxBoundsClamping) {
  // tk=7, mtwiden=1
  uint32_t rs1_val = EncodeMtype(kMtwiden1, /*tk=*/7);
  uint32_t rs2_val = 0;

  SetRegisterValue("x1", rs1_val);
  SetRegisterValue("x2", rs2_val);

  auto* reg1 = state_->GetRegister<RV32Register>("x1").first;
  auto* reg2 = state_->GetRegister<RV32Register>("x2").first;

  instruction_->AppendSource(reg1->CreateSourceOperand());
  instruction_->AppendSource(reg2->CreateSourceOperand());

  instruction_->set_semantic_function(&RiscVZvtMsetMtype);
  instruction_->Execute(nullptr);

  // Assert clamped K to KMAX=4
  EXPECT_EQ(matrix_state_->tile_k(), 4);
}

TEST_F(RiscVZvtInstructionsTest, MtypeTmCapacityLayoutClamping) {
  // tm=500, mtwiden=1
  uint32_t rs1_val = EncodeMtype(kMtwiden1, /*tk=*/0, /*tm=*/500);
  uint32_t rs2_val = 0;

  SetRegisterValue("x1", rs1_val);
  SetRegisterValue("x2", rs2_val);

  auto* reg1 = state_->GetRegister<RV32Register>("x1").first;
  auto* reg2 = state_->GetRegister<RV32Register>("x2").first;

  instruction_->AppendSource(reg1->CreateSourceOperand());
  instruction_->AppendSource(reg2->CreateSourceOperand());

  instruction_->set_semantic_function(&RiscVZvtMsetMtype);
  instruction_->Execute(nullptr);

  // ETE for 8-bit * TWIDEN=1 is tile_elements(), which is 32.
  // LMUL=1 -> kLmulElements = 128 (max_vector_length in elements).
  // min(tmpl_m=500, kLmulElements=128, kEffectiveTileElements=32) = 32.
  EXPECT_EQ(matrix_state_->tile_m(), 32);
}

TEST_F(RiscVZvtInstructionsTest, MsetmtypeVmaTailMapping) {
  SetRegisterValue("x1", EncodeMtype(kMtwiden1));
  SetRegisterValue("x2",
                   EncodeVtype(kVsew8, kVlmul1, /*vta=*/false, /*vma=*/true));

  auto* reg1 = state_->GetRegister<RV32Register>("x1").first;
  auto* reg2 = state_->GetRegister<RV32Register>("x2").first;
  instruction_->AppendSource(reg1->CreateSourceOperand());
  instruction_->AppendSource(reg2->CreateSourceOperand());

  instruction_->set_semantic_function(&RiscVZvtMsetMtype);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(state_->rv_vector()->vector_mask_agnostic());
}

TEST_F(RiscVZvtInstructionsTest, MsetmtypeVtaMaskMapping) {
  SetRegisterValue("x1", EncodeMtype(kMtwiden1));
  SetRegisterValue("x2", EncodeVtype(kVsew8, kVlmul1, /*vta=*/true));

  auto* reg1 = state_->GetRegister<RV32Register>("x1").first;
  auto* reg2 = state_->GetRegister<RV32Register>("x2").first;
  instruction_->AppendSource(reg1->CreateSourceOperand());
  instruction_->AppendSource(reg2->CreateSourceOperand());

  instruction_->set_semantic_function(&RiscVZvtMsetMtype);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(state_->rv_vector()->vector_tail_agnostic());
}

TEST_F(RiscVZvtInstructionsTest, MsetmtypeUnconditionalVlZeroing) {
  vector_state_->set_vector_length(100);

  SetRegisterValue("x1", 1);
  SetRegisterValue("x2", 0);

  auto* reg1 = state_->GetRegister<RV32Register>("x1").first;
  auto* reg2 = state_->GetRegister<RV32Register>("x2").first;
  instruction_->AppendSource(reg1->CreateSourceOperand());
  instruction_->AppendSource(reg2->CreateSourceOperand());

  instruction_->set_semantic_function(&RiscVZvtMsetMtype);
  instruction_->Execute(nullptr);

  EXPECT_EQ(state_->rv_vector()->vector_length(), 0);
}

TEST_F(RiscVZvtInstructionsTest, MsetmtypeiUnconditionalVlZeroing) {
  vector_state_->set_vector_length(100);

  instruction_->AppendSource(
      new ImmediateOperand<uint32_t>(/*mtwiden=*/kMtwiden1));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vsew=*/kVsew8));
  instruction_->AppendSource(
      new ImmediateOperand<uint32_t>(/*altfmt=*/kAltfmtStandard));
  instruction_->set_semantic_function(&RiscVZvtMsetMtypei);
  instruction_->Execute(nullptr);

  EXPECT_EQ(state_->rv_vector()->vector_length(), 0);
}

TEST_F(RiscVZvtInstructionsTest, MstatusVsDirtyTrackingOnMsetmtype) {
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();

  SetRegisterValue("x1", EncodeMtype(kMtwiden1));
  SetRegisterValue("x2", 0);

  auto* reg1 = state_->GetRegister<RV32Register>("x1").first;
  auto* reg2 = state_->GetRegister<RV32Register>("x2").first;
  instruction_->AppendSource(reg1->CreateSourceOperand());
  instruction_->AppendSource(reg2->CreateSourceOperand());

  instruction_->set_semantic_function(&RiscVZvtMsetMtype);
  instruction_->Execute(nullptr);

  EXPECT_EQ(state_->mstatus()->vs(), kMstatusStateDirty);
}

TEST_F(RiscVZvtInstructionsTest, MstatusVsDirtyTrackingOnMsetmtypei) {
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();

  vsstatus()->set_vs(kMstatusStateInitial);

  instruction_->AppendSource(
      new ImmediateOperand<uint32_t>(/*mtwiden=*/kMtwiden1));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vsew=*/kVsew8));
  instruction_->AppendSource(
      new ImmediateOperand<uint32_t>(/*altfmt=*/kAltfmtStandard));
  instruction_->set_semantic_function(&RiscVZvtMsetMtypei);
  instruction_->Execute(nullptr);

  EXPECT_EQ(state_->mstatus()->vs(), kMstatusStateDirty);
  EXPECT_EQ(vsstatus()->vs(), kMstatusStateDirty);
}

TEST_F(RiscVZvtInstructionsTest, MsetExecuteWithoutTrapWhenMsOff) {
  state_->mstatus()->set_ms(kMstatusStateOff);

  ScopedTrapCapture trap_capture(state_.get());

  instruction_->AppendSource(
      new ImmediateOperand<uint32_t>(/*mtwiden=*/kMtwiden1));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vsew=*/kVsew8));
  instruction_->AppendSource(
      new ImmediateOperand<uint32_t>(/*altfmt=*/kAltfmtStandard));
  instruction_->set_semantic_function(&RiscVZvtMsetMtypei);
  instruction_->Execute(nullptr);

  EXPECT_FALSE(trap_capture.trap_taken());
}

TEST_F(RiscVZvtInstructionsTest, MstatusVsDirtyTrackingOnMsettn) {
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();

  vsstatus()->set_vs(kMstatusStateInitial);

  // Configure matrix state first
  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));

  SetRegisterValue("x1", 16);
  auto* reg1 = state_->GetRegister<RV32Register>("x1").first;
  auto* dest = state_->GetRegister<RV32Register>("x2").first;
  instruction_->AppendSource(reg1->CreateSourceOperand());
  instruction_->AppendDestination(dest->CreateDestinationOperand(0));

  instruction_->set_semantic_function(&RiscVZvtMsetTn);
  instruction_->Execute(nullptr);

  EXPECT_EQ(state_->mstatus()->vs(), kMstatusStateDirty);
  EXPECT_EQ(vsstatus()->vs(), kMstatusStateDirty);
}

TEST_F(RiscVZvtInstructionsTest, MstatusVsDirtyTrackingOnMsettm) {
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();

  vsstatus()->set_vs(kMstatusStateInitial);

  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));

  SetRegisterValue("x1", 16);
  auto* reg1 = state_->GetRegister<RV32Register>("x1").first;
  auto* dest = state_->GetRegister<RV32Register>("x2").first;
  instruction_->AppendSource(reg1->CreateSourceOperand());
  instruction_->AppendDestination(dest->CreateDestinationOperand(0));

  instruction_->set_semantic_function(&RiscVZvtMsetTm);
  instruction_->Execute(nullptr);

  EXPECT_EQ(state_->mstatus()->vs(), kMstatusStateDirty);
  EXPECT_EQ(vsstatus()->vs(), kMstatusStateDirty);
}

TEST_F(RiscVZvtInstructionsTest, MstatusVsDirtyTrackingOnMsettk) {
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();

  vsstatus()->set_vs(kMstatusStateInitial);

  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));

  SetRegisterValue("x1", 16);
  auto* reg1 = state_->GetRegister<RV32Register>("x1").first;
  auto* dest = state_->GetRegister<RV32Register>("x2").first;
  instruction_->AppendSource(reg1->CreateSourceOperand());
  instruction_->AppendDestination(dest->CreateDestinationOperand(0));

  instruction_->set_semantic_function(&RiscVZvtMsetTk);
  instruction_->Execute(nullptr);

  EXPECT_EQ(state_->mstatus()->vs(), kMstatusStateDirty);
  EXPECT_EQ(vsstatus()->vs(), kMstatusStateDirty);
}

TEST_F(RiscVZvtInstructionsTest, MsettmIllegalInstructionException) {
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();
  matrix_state_->set_mtype(kMtwidenUnconfigured);

  ScopedTrapCapture trap_capture(state_.get());

  SetRegisterValue("x1", 500);
  auto* reg1 = state_->GetRegister<RV32Register>("x1").first;
  auto* dest = state_->GetRegister<RV32Register>("x2").first;
  instruction_->AppendSource(reg1->CreateSourceOperand());
  instruction_->AppendDestination(dest->CreateDestinationOperand(0));

  instruction_->set_semantic_function(&RiscVZvtMsetTm);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
  EXPECT_EQ(state_->mstatus()->vs(), kMstatusStateInitial);
}

TEST_F(RiscVZvtInstructionsTest, GetMatrixStateIllegalInstruction) {
  // Remove the CSR to simulate GetMatrixState failure.
  ASSERT_OK(state_->csr_set()->RemoveCsr(kRiscVMtypeCsrIndex));

  ScopedTrapCapture trap_capture(state_.get());

  instruction_->set_semantic_function(&RiscVZvtMsetMtypei);
  instruction_->Execute(nullptr);
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, MsettkIllegalInstructionException) {
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();
  matrix_state_->set_mtype(kMtwidenUnconfigured);

  ScopedTrapCapture trap_capture(state_.get());

  SetRegisterValue("x1", 500);
  auto* reg1 = state_->GetRegister<RV32Register>("x1").first;
  auto* dest = state_->GetRegister<RV32Register>("x2").first;
  instruction_->AppendSource(reg1->CreateSourceOperand());
  instruction_->AppendDestination(dest->CreateDestinationOperand(0));

  instruction_->set_semantic_function(&RiscVZvtMsetTk);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
  EXPECT_EQ(state_->mstatus()->vs(), kMstatusStateInitial);
}

TEST_F(RiscVZvtInstructionsTest, MsettnVsetvlDegeneration) {
  matrix_state_->set_mtype(kMtwidenUnconfigured);
  vector_state_->SetVectorType(
      EncodeVtype(kVsew8, kVlmul1));  // vlmax = 128 elements

  SetRegisterValue("x1", 500);  // Greater than vlmax
  auto* reg1 = state_->GetRegister<RV32Register>("x1").first;
  auto* dest = state_->GetRegister<RV32Register>("x2").first;
  instruction_->AppendSource(reg1->CreateSourceOperand());
  instruction_->AppendDestination(dest->CreateDestinationOperand(0));

  instruction_->set_semantic_function(&RiscVZvtMsetTn);
  instruction_->Execute(nullptr);

  EXPECT_EQ(state_->rv_vector()->vector_length(),
            128);  // Degenerates to min(500, current vlmax)
}

TEST_F(RiscVZvtInstructionsTest, MsettnX0RegisterHandling) {
  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));

  SetRegisterValue("x0", 0);
  auto* reg0 = state_->GetRegister<RV32Register>("x0").first;
  auto* dest = state_->GetRegister<RV32Register>("x2").first;
  instruction_->AppendSource(reg0->CreateSourceOperand());
  instruction_->AppendDestination(dest->CreateDestinationOperand(0));

  instruction_->set_semantic_function(&RiscVZvtMsetTn);
  instruction_->Execute(nullptr);

  EXPECT_EQ(state_->rv_vector()->vector_length(), 0);
}

TEST_F(RiscVZvtInstructionsTest, MsettmX0RegisterHandling) {
  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));

  SetRegisterValue("x0", 0);
  auto* reg0 = state_->GetRegister<RV32Register>("x0").first;
  auto* dest = state_->GetRegister<RV32Register>("x2").first;
  instruction_->AppendSource(reg0->CreateSourceOperand());
  instruction_->AppendDestination(dest->CreateDestinationOperand(0));

  instruction_->set_semantic_function(&RiscVZvtMsetTm);
  instruction_->Execute(nullptr);

  EXPECT_EQ(matrix_state_->tile_m(), 0);
}

TEST_F(RiscVZvtInstructionsTest, MsettkX0RegisterHandling) {
  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));

  SetRegisterValue("x0", 0);
  auto* reg0 = state_->GetRegister<RV32Register>("x0").first;
  auto* dest = state_->GetRegister<RV32Register>("x2").first;
  instruction_->AppendSource(reg0->CreateSourceOperand());
  instruction_->AppendDestination(dest->CreateDestinationOperand(0));

  instruction_->set_semantic_function(&RiscVZvtMsetTk);
  instruction_->Execute(nullptr);

  EXPECT_EQ(matrix_state_->tile_k(), 0);
}

TEST_F(RiscVZvtInstructionsTest, VtypeVillInvalidationStateMachine) {
  // ELEN is 32. mtwiden = 1. SEW = 64bit -> 64 * 1 > 32 -> triggers vill
  // (exception).
  SetRegisterValue("x1", EncodeMtype(kMtwiden1));
  SetRegisterValue("x2", EncodeVtype(kVsew64, kVlmul1));

  auto* reg1 = state_->GetRegister<RV32Register>("x1").first;
  auto* reg2 = state_->GetRegister<RV32Register>("x2").first;
  instruction_->AppendSource(reg1->CreateSourceOperand());
  instruction_->AppendSource(reg2->CreateSourceOperand());

  instruction_->set_semantic_function(&RiscVZvtMsetMtype);
  instruction_->Execute(nullptr);

  // Assert vtype.vill escalation
  EXPECT_TRUE(state_->rv_vector()->vector_exception());
  EXPECT_EQ(state_->rv_vector()->vtype(), kVillMask);
  EXPECT_EQ(state_->rv_vector()->vector_length(), 0u);
  // Assert mtype zeroing
  EXPECT_EQ(matrix_state_->mtype(), 0u);
  // Assert vtype remaining fields zeroing
  EXPECT_EQ(state_->rv_vector()->vector_length_multiplier(), 0);
  EXPECT_EQ(state_->rv_vector()->selected_element_width(), 0);
  EXPECT_FALSE(state_->rv_vector()->vector_tail_agnostic());
  EXPECT_FALSE(state_->rv_vector()->vector_mask_agnostic());
}

TEST_F(RiscVZvtInstructionsTest, VillEscalationWhenSewTwidenExceedsElen) {
  // ELEN is 32.
  // mtwiden = 1. SEW = 64bit -> 64 * 1 > 32 -> triggers vill (exception).
  SetRegisterValue("x1", EncodeMtype(kMtwiden1));
  SetRegisterValue("x2", EncodeVtype(kVsew64, kVlmul1));

  auto* reg1 = state_->GetRegister<RV32Register>("x1").first;
  auto* reg2 = state_->GetRegister<RV32Register>("x2").first;
  instruction_->AppendSource(reg1->CreateSourceOperand());
  instruction_->AppendSource(reg2->CreateSourceOperand());

  instruction_->set_semantic_function(&RiscVZvtMsetMtype);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(state_->rv_vector()->vector_exception());
  EXPECT_EQ(state_->rv_vector()->vtype(), kVillMask);
  EXPECT_EQ(state_->rv_vector()->vector_length(), 0u);
  EXPECT_EQ(matrix_state_->mtype(), 0u);
}

TEST_F(RiscVZvtInstructionsTest, MatrixExceptionTrapWhenMsOff) {
  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));

  state_->mstatus()->set_ms(kMstatusStateOff);  // MS off
  state_->mstatus()->Submit();

  ScopedTrapCapture trap_capture(state_.get());

  instruction_->set_semantic_function(&RiscVZvtVtZero);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtDiscardTrapWhenMatrixStateOff) {
  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));

  state_->mstatus()->set_ms(kMstatusStateOff);  // MS off
  state_->mstatus()->Submit();

  ScopedTrapCapture trap_capture(state_.get());

  instruction_->set_semantic_function(&RiscVZvtVtDiscard);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtDiscardTrapWhenVill) {
  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  vector_state_->SetVectorType(kVillMask);

  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->Submit();
  ScopedTrapCapture trap_capture(state_.get());

  instruction_->set_semantic_function(&RiscVZvtVtDiscard);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtDiscardTrapWhenVillAndMtwidenZero) {
  matrix_state_->set_mtype(0);  // mtwiden = 0

  // Set vill = 1 (vector unconfigured)
  vector_state_->SetVectorType(kVillMask);

  state_->mstatus()->set_ms(kMstatusStateDirty);
  state_->mstatus()->Submit();

  ScopedTrapCapture trap_capture(state_.get());

  instruction_->set_semantic_function(&RiscVZvtVtDiscard);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtDiscardNoTrapWhenMtwidenZero) {
  matrix_state_->set_mtype(0);  // mtwiden = 0
  // Assert valid vector state (vill = 0) to avoid precondition trap
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));

  state_->mstatus()->set_ms(kMstatusStateDirty);
  state_->mstatus()->Submit();
  ScopedTrapCapture trap_capture(state_.get());

  instruction_->set_semantic_function(&RiscVZvtVtDiscard);
  instruction_->Execute(nullptr);

  EXPECT_FALSE(trap_capture.trap_taken());
  EXPECT_EQ(state_->mstatus()->ms(), kMstatusStateInitial);  // Initial
}

TEST_F(RiscVZvtInstructionsTest, MstatusMsDirtyTrackingOnDirectTileUpdate) {
  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));

  state_->mstatus()->set_ms(kMstatusStateInitial);  // Initial
  state_->mstatus()->Submit();

  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*mtd=*/0));
  instruction_->set_semantic_function(&RiscVZvtVtZero);
  instruction_->Execute(nullptr);

  EXPECT_EQ(state_->mstatus()->ms(), kMstatusStateDirty);  // Dirty
}

TEST_F(RiscVZvtInstructionsTest, MstatusVsDirtyTrackingOnGeneralInstruction) {
  state_->mstatus()->set_ms(kMstatusStateInitial);  // Initial
  state_->mstatus()->set_vs(kMstatusStateInitial);  // Initial
  state_->mstatus()->Submit();

  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));

  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*mtd=*/0));
  instruction_->set_semantic_function(&RiscVZvtVtZero);
  instruction_->Execute(nullptr);

  EXPECT_EQ(state_->mstatus()->vs(), kMstatusStateDirty);  // Dirty
}

TEST_F(RiscVZvtInstructionsTest, VsetvlDoesNotTrapWhenMatrixStateIsOff) {
  // mstatus.MS = Off
  state_->mstatus()->set_ms(kMstatusStateOff);
  state_->mstatus()->Submit();

  // Test vsetvl execution
  SetRegisterValue("x1", 10);
  SetRegisterValue("x2", 0x15);
  auto* reg1 = state_->GetRegister<RV32Register>("x1").first;
  auto* reg2 = state_->GetRegister<RV32Register>("x2").first;
  auto* dest = state_->GetRegister<RV32Register>("x3").first;

  instruction_->AppendSource(reg1->CreateSourceOperand());
  instruction_->AppendSource(reg2->CreateSourceOperand());
  instruction_->AppendDestination(dest->CreateDestinationOperand(0));

  ScopedTrapCapture trap_capture(state_.get());

  instruction_->set_semantic_function([](Instruction* inst) {
    Vsetvl(/*rd_zero=*/false, /*rs1_zero=*/false, inst);
  });

  instruction_->Execute(nullptr);

  // Expect no exception to be generated
  // Matrix state Off shouldn't cause kIllegalInstruction in vsetvl!
  EXPECT_FALSE(trap_capture.trap_taken());
}

TEST_F(RiscVZvtInstructionsTest, VtzeroStateZeroingLoop) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4, /*tk=*/0,
                      /*tm=*/4);
  constexpr uint32_t kTileId = 0;
  ScopedTrapCapture trap_capture(state_.get());

  // Populate tile 0 with non-zero values to verify zeroing.
  ASSERT_OK(FillTileConstant<uint32_t>(kTileId, 4, 4, 0xDEADC0DE));

  // Create `vtzero` instruction testing `mtd` = 0.
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(kTileId));  // mtd
  instruction_->set_semantic_function(&RiscVZvtVtZero);
  instruction_->Execute(nullptr);

  EXPECT_FALSE(trap_capture.trap_taken());
  ASSERT_OK_AND_ASSIGN(
      auto tile_vals, GetTileRegion<uint32_t>(kTileId, /*rows=*/4, /*cols=*/4));
  EXPECT_THAT(tile_vals, testing::Each(0));
}

TEST_F(RiscVZvtInstructionsTest, VtzeroTrapOnIllegalTile) {
  // mstatus.MS and mstatus.VS must be enabled
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();

  vsstatus()->set_ms(kMstatusStateInitial);

  // Set up mtype (mtwiden = 1) -> TWIDEN = 1
  uint32_t mtype = EncodeMtype(kMtwiden1, /*tk=*/0, /*tm=*/4);
  matrix_state_->set_mtype(mtype);

  // Set up vector state: vl = 4, sew = 32 bits (0b10 == 2, shifted by 3)
  vector_state_->set_vector_length(4);
  vector_state_->SetVectorType(EncodeVtype(kVsew32, kVlmul1));

  const int tile_id = 7;
  ScopedTrapCapture trap_capture(state_.get());

  // Create `vtzero` instruction testing `mtd` = 7.
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(tile_id));  // mtd
  instruction_->set_semantic_function(&RiscVZvtVtZero);
  instruction_->Execute(nullptr);

  // Expect trap since tile_id=7 is unaligned for TEW=32
  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtzeroNoTrapWhenMtwidenZero) {
  // mstatus.MS and mstatus.VS must be enabled
  state_->mstatus()->set_ms(kMstatusStateDirty);
  state_->mstatus()->set_vs(kMstatusStateDirty);
  state_->mstatus()->Submit();

  // Set up mtype (mtwiden = 0)
  matrix_state_->set_mtype(0);

  ScopedTrapCapture trap_capture(state_.get());

  const int tile_id =
      1;  // Unaligned tile index (e.g., usually would be mod 16/NTiles)

  instruction_->AppendSource(new ImmediateOperand<uint32_t>(tile_id));  // mtd
  instruction_->set_semantic_function(&RiscVZvtVtZero);
  instruction_->Execute(nullptr);

  // Must not trap when mtwiden == 0, even if the tile is otherwise unaligned.
  EXPECT_FALSE(trap_capture.trap_taken());
}

TEST_F(RiscVZvtInstructionsTest, VtzeroTrapWhenMtwidenNotZeroAndUnaligned) {
  state_->mstatus()->set_ms(kMstatusStateDirty);
  state_->mstatus()->set_vs(kMstatusStateDirty);
  state_->mstatus()->Submit();

  // Set up mtype (mtwiden = 1) -> TWIDEN = 1, NTiles = 8, tile align = 2
  uint32_t mtype = EncodeMtype(kMtwiden1, /*tk=*/0, /*tm=*/4);
  matrix_state_->set_mtype(mtype);

  vector_state_->set_vector_length(4);
  vector_state_->SetVectorType(EncodeVtype(kVsew32, kVlmul1));

  ScopedTrapCapture trap_capture(state_.get());

  const int tile_id = 1;  // Unaligned since align is 16/NTiles = 16/8 = 2

  instruction_->AppendSource(new ImmediateOperand<uint32_t>(tile_id));  // mtd
  instruction_->set_semantic_function(&RiscVZvtVtZero);
  instruction_->Execute(nullptr);

  // Must trap when mtwiden != 0 and tile is unaligned.
  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtmvVtGeneralizedRowElementExtraction) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4, /*tk=*/0,
                      /*tm=*/4);
  constexpr uint32_t kTileId = 4;
  const int te = matrix_state_->tile_elements();
  ASSERT_OK(FillTilePattern<uint32_t>(kTileId, te, te, [](int row, int col) {
    return 0xDEADC0DE + row * 10 + col;
  }));

  ExecuteVtmvVt(EncodeTss(kTileId, kTssPatternRow, /*index=*/2));

  EXPECT_THAT(ReadVectorRegister<uint32_t>("v8", 4),
              testing::ElementsAre(0xDEADC0DE + 20, 0xDEADC0DE + 21,
                                   0xDEADC0DE + 22, 0xDEADC0DE + 23));
}

TEST_F(RiscVZvtInstructionsTest, VtmvVtGeneralizedColumnElementExtraction) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4, /*tk=*/0,
                      /*tm=*/4);
  constexpr uint32_t kTileId = 4;
  const int te = matrix_state_->tile_elements();
  ASSERT_OK(FillTilePattern<uint32_t>(kTileId, te, te, [](int row, int col) {
    return 0xDEADC0DE + row * 10 + col;
  }));

  ExecuteVtmvVt(EncodeTss(kTileId, kTssPatternCol, /*index=*/2));

  EXPECT_THAT(ReadVectorRegister<uint32_t>("v8", 4),
              testing::ElementsAre(0xDEADC0DE + 2, 0xDEADC0DE + 12,
                                   0xDEADC0DE + 22, 0xDEADC0DE + 32));
}

TEST_F(RiscVZvtInstructionsTest, VtmvTvPatternDirectedRowVectorInsertion) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4, /*tk=*/0,
                      /*tm=*/4);
  constexpr uint32_t kTileId = 4;
  const int te = matrix_state_->tile_elements();
  ASSERT_OK(FillTileConstant<uint32_t>(kTileId, te, te, 0));
  WriteVectorRegister<uint32_t>(
      "v8", {0xBEAF0000, 0xBEAF0001, 0xBEAF0002, 0xBEAF0003});

  ExecuteVtmvTv(/*vs_reg=*/8, EncodeTss(kTileId, kTssPatternRow, /*index=*/2));

  ASSERT_OK_AND_ASSIGN(auto row_vals,
                       GetTileRow<uint32_t>(kTileId, /*row=*/2, /*count=*/4));
  EXPECT_THAT(row_vals, testing::ElementsAre(0xBEAF0000, 0xBEAF0001, 0xBEAF0002,
                                             0xBEAF0003));
  EXPECT_EQ(state_->mstatus()->ms(), kMstatusStateDirty);
}

TEST_F(RiscVZvtInstructionsTest, VtmvTvPatternDirectedColumnVectorInsertion) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4, /*tk=*/0,
                      /*tm=*/4);
  constexpr uint32_t kTileId = 4;
  const int te = matrix_state_->tile_elements();
  ASSERT_OK(FillTileConstant<uint32_t>(kTileId, te, te, 0));
  WriteVectorRegister<uint32_t>(
      "v8", {0xBEAF0000, 0xBEAF0001, 0xBEAF0002, 0xBEAF0003});

  ExecuteVtmvTv(/*vs_reg=*/8, EncodeTss(kTileId, kTssPatternCol, /*index=*/2));

  ASSERT_OK_AND_ASSIGN(
      auto col_vals, GetTileColumn<uint32_t>(kTileId, /*col=*/2, /*count=*/4));
  EXPECT_THAT(col_vals, testing::ElementsAre(0xBEAF0000, 0xBEAF0001, 0xBEAF0002,
                                             0xBEAF0003));
  EXPECT_EQ(state_->mstatus()->ms(), kMstatusStateDirty);
}

TEST_F(RiscVZvtInstructionsTest, VtmvTssPatternExceedsBoundsTrap) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteVtmvVt(EncodeTss(/*tile=*/4, kTssPatternInvalid, /*index=*/2));

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtmvTssIndexOverflowTrap) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  // Max tile elements for SEW=32 and TWIDEN=1 is 128 / 4 = 32; index 32 is out
  // of bounds (0..31).
  ExecuteVtmvVt(EncodeTss(/*tile=*/4, kTssPatternRow, /*index=*/32));

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtmvTssMisalignedTileMasking) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  // Mock data in tile 0 so we can observe that misaligned tile 1 masks to tile
  // 0.
  ASSERT_OK(SetTileRow<uint32_t>(
      /*tile=*/0, /*row=*/0, {0xFACEBEEF, 0xDEADC0DE, 0xBAADF00D, 0xCAFEBABE}));

  ExecuteVtmvVt(EncodeTss(/*tile=*/1, kTssPatternRow, /*index=*/0));

  EXPECT_FALSE(trap_capture.trap_taken());
  EXPECT_THAT(
      ReadVectorRegister<uint32_t>("v8", 4),
      testing::ElementsAre(0xFACEBEEF, 0xDEADC0DE, 0xBAADF00D, 0xCAFEBABE));
}

TEST_F(RiscVZvtInstructionsTest, VtleUnconfiguredTrap) {
  ConfigureMatrixUnit(kMtwidenUnconfigured, kVsew8, kVlmul1, /*vl=*/0);
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/32, /*base_addr=*/0, /*tss=*/0);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtseUnconfiguredTrap) {
  ConfigureMatrixUnit(kMtwidenUnconfigured, kVsew8, kVlmul1, /*vl=*/0);
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/32, /*base_addr=*/0, /*tss=*/0);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtmvVtVstartLowerBoundEnforcement) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4, /*tk=*/0,
                      /*tm=*/4, /*vstart=*/2);
  constexpr uint32_t kTileId = 4;
  const int te = matrix_state_->tile_elements();
  ASSERT_OK(FillTilePattern<uint32_t>(kTileId, te, te, [](int row, int col) {
    return 0xDEADC0DE + row * 10 + col;
  }));

  // Pre-fill v8 to verify elements < vstart are unchanged.
  WriteVectorRegister<uint32_t>(
      "v8", {0x11111111, 0x11111111, 0x11111111, 0x11111111});

  ExecuteVtmvVt(EncodeTss(kTileId, kTssPatternRow, /*index=*/2));

  // Elements < vstart remain untouched; elements [vstart..vl) are extracted.
  EXPECT_THAT(ReadVectorRegister<uint32_t>("v8", 4),
              testing::ElementsAre(0x11111111, 0x11111111, 0xDEADC0DE + 22,
                                   0xDEADC0DE + 23));
}

TEST_F(RiscVZvtInstructionsTest, VtmvTvVstartLowerBoundEnforcement) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4, /*tk=*/0,
                      /*tm=*/4, /*vstart=*/2);
  constexpr uint32_t kTileId = 4;
  const int te = matrix_state_->tile_elements();
  ASSERT_OK(FillTileConstant<uint32_t>(kTileId, te, te, 0x0));
  WriteVectorRegister<uint32_t>(
      "v8", {0x11111110, 0x11111111, 0x11111112, 0x11111113});

  ExecuteVtmvTv(/*vs_reg=*/8, EncodeTss(kTileId, kTssPatternRow, /*index=*/2));

  // Elements < vstart remain untouched (0x0); elements [vstart..vl) are
  // inserted.
  ASSERT_OK_AND_ASSIGN(auto row_vals,
                       GetTileRow<uint32_t>(kTileId, /*row=*/2, /*count=*/4));
  EXPECT_THAT(row_vals, testing::ElementsAre(0x0, 0x0, 0x11111112, 0x11111113));
}

TEST_F(RiscVZvtInstructionsTest, VtmvVtUnconfiguredTrap) {
  ConfigureMatrixUnit(kMtwidenUnconfigured, kVsew32, kVlmul1, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteVtmvVt(/*tss=*/0);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtmvTvUnconfiguredTrap) {
  ConfigureMatrixUnit(kMtwidenUnconfigured, kVsew32, kVlmul1, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteVtmvTv(/*vs_reg=*/8, /*tss=*/0);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtleVstartLowerBoundLimitation) {
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/4, /*tk=*/0, /*tm=*/0,
                      /*vstart=*/2);

  constexpr uint64_t kBaseAddr = 0x2000;
  WriteMemory<uint8_t>(kBaseAddr, {0x11, 0x22, 0x33, 0x44});

  constexpr uint32_t kTileId = 1;
  constexpr uint32_t kRowIndex = 2;
  // Pre-fill tile row to verify elements < vstart remain unchanged.
  ASSERT_OK(SetTileRowPattern<uint8_t>(kTileId, kRowIndex, /*count=*/4,
                                       [](int) { return 0xAA; }));

  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/8, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternRow, kRowIndex));

  // Elements < vstart remain untouched (0xAA); elements [vstart..vl) are
  // loaded.
  ASSERT_OK_AND_ASSIGN(auto row_vals,
                       GetTileRow<uint8_t>(kTileId, kRowIndex, /*count=*/4));
  EXPECT_THAT(row_vals, testing::ElementsAre(0xAA, 0xAA, 0x33, 0x44));
  EXPECT_EQ(vector_state_->vstart(), 0);
}

TEST_F(RiscVZvtInstructionsTest, Vtle8UnitStrideRow) {
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/4);

  constexpr uint64_t kBaseAddr = 0x2000;
  WriteMemory<uint8_t>(kBaseAddr, {0x11, 0x22, 0x33, 0x44});

  constexpr uint32_t kTileId = 1;
  constexpr uint32_t kRowIndex = 2;
  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/8, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternRow, kRowIndex));

  ASSERT_OK_AND_ASSIGN(auto row_vals,
                       GetTileRow<uint8_t>(kTileId, kRowIndex, /*count=*/4));
  EXPECT_THAT(row_vals, testing::ElementsAre(0x11, 0x22, 0x33, 0x44));
}

TEST_F(RiscVZvtInstructionsTest, Vtle16UnitStrideRow) {
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/4);

  constexpr uint64_t kBaseAddr = 0x2000;
  WriteMemory<uint16_t>(kBaseAddr, {0x1111, 0x2222, 0x3333, 0x4444});

  constexpr uint32_t kTileId = 2;
  constexpr uint32_t kRowIndex = 2;
  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/16, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternRow, kRowIndex));

  ASSERT_OK_AND_ASSIGN(auto row_vals,
                       GetTileRow<uint16_t>(kTileId, kRowIndex, /*count=*/4));
  EXPECT_THAT(row_vals, testing::ElementsAre(0x1111, 0x2222, 0x3333, 0x4444));
}

TEST_F(RiscVZvtInstructionsTest, Vtle32UnitStrideRow) {
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/4);

  constexpr uint64_t kBaseAddr = 0x2000;
  WriteMemory<uint32_t>(kBaseAddr,
                        {0x11111111, 0x22222222, 0x33333333, 0x44444444});

  constexpr uint32_t kTileId = 4;
  constexpr uint32_t kRowIndex = 2;
  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/32, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternRow, kRowIndex));

  ASSERT_OK_AND_ASSIGN(auto row_vals,
                       GetTileRow<uint32_t>(kTileId, kRowIndex, /*count=*/4));
  EXPECT_THAT(row_vals, testing::ElementsAre(0x11111111, 0x22222222, 0x33333333,
                                             0x44444444));
}

TEST_F(RiscVZvtInstructionsTest, Vtle64UnitStrideRow) {
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/2);

  constexpr uint64_t kBaseAddr = 0x2000;
  WriteMemory<uint64_t>(kBaseAddr,
                        {0x1111111122222222ULL, 0x3333333344444444ULL});

  constexpr uint32_t kTileId = 8;
  constexpr uint32_t kRowIndex = 2;
  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/64, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternRow, kRowIndex));

  ASSERT_OK_AND_ASSIGN(auto row_vals,
                       GetTileRow<uint64_t>(kTileId, kRowIndex, /*count=*/2));
  EXPECT_THAT(row_vals, testing::ElementsAre(0x1111111122222222ULL,
                                             0x3333333344444444ULL));
}

TEST_F(RiscVZvtInstructionsTest, Vtle8UnitStrideCol) {
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/4);

  constexpr uint64_t kBaseAddr = 0x2000;
  WriteMemory<uint8_t>(kBaseAddr, {0x11, 0x22, 0x33, 0x44});

  constexpr uint32_t kTileId = 1;
  constexpr uint32_t kColIndex = 2;
  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/8, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternCol, kColIndex));

  ASSERT_OK_AND_ASSIGN(auto col_vals,
                       GetTileColumn<uint8_t>(kTileId, kColIndex, /*count=*/4));
  EXPECT_THAT(col_vals, testing::ElementsAre(0x11, 0x22, 0x33, 0x44));
}

TEST_F(RiscVZvtInstructionsTest, Vtse8UnitStrideRow) {
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/4);

  constexpr uint32_t kTileId = 1;
  constexpr uint32_t kRowIndex = 2;
  ASSERT_OK(SetTileRow<uint8_t>(kTileId, kRowIndex, {0x11, 0x22, 0x33, 0x44}));

  constexpr uint64_t kBaseAddr = 0x2000;
  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/8, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternRow, kRowIndex));

  EXPECT_THAT(ReadMemory<uint8_t>(kBaseAddr, 4),
              testing::ElementsAre(0x11, 0x22, 0x33, 0x44));
}

TEST_F(RiscVZvtInstructionsTest, Vtse8UnitStrideCol) {
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/4);

  constexpr uint32_t kTileId = 1;
  constexpr uint32_t kColIndex = 2;
  ASSERT_OK(
      SetTileColumn<uint8_t>(kTileId, kColIndex, {0x11, 0x22, 0x33, 0x44}));

  constexpr uint64_t kBaseAddr = 0x2000;
  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/8, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternCol, kColIndex));

  EXPECT_THAT(ReadMemory<uint8_t>(kBaseAddr, 4),
              testing::ElementsAre(0x11, 0x22, 0x33, 0x44));
}

TEST_F(RiscVZvtInstructionsTest, Vtle16UnitStrideCol) {
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/4);

  constexpr uint64_t kBaseAddr = 0x2000;
  WriteMemory<uint16_t>(kBaseAddr, {0x1111, 0x2222, 0x3333, 0x4444});

  constexpr uint32_t kTileId = 2;
  constexpr uint32_t kColIndex = 2;
  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/16, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternCol, kColIndex));

  ASSERT_OK_AND_ASSIGN(
      auto col_vals, GetTileColumn<uint16_t>(kTileId, kColIndex, /*count=*/4));
  EXPECT_THAT(col_vals, testing::ElementsAre(0x1111, 0x2222, 0x3333, 0x4444));
}

TEST_F(RiscVZvtInstructionsTest, Vtse16UnitStrideRow) {
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/4);

  constexpr uint32_t kTileId = 2;
  constexpr uint32_t kRowIndex = 2;
  ASSERT_OK(SetTileRow<uint16_t>(kTileId, kRowIndex,
                                 {0x1111, 0x2222, 0x3333, 0x4444}));

  constexpr uint64_t kBaseAddr = 0x2000;
  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/16, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternRow, kRowIndex));

  EXPECT_THAT(ReadMemory<uint16_t>(kBaseAddr, 4),
              testing::ElementsAre(0x1111, 0x2222, 0x3333, 0x4444));
}

TEST_F(RiscVZvtInstructionsTest, Vtse16UnitStrideCol) {
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/4);

  constexpr uint32_t kTileId = 2;
  constexpr uint32_t kColIndex = 2;
  ASSERT_OK(SetTileColumn<uint16_t>(kTileId, kColIndex,
                                    {0x1111, 0x2222, 0x3333, 0x4444}));

  constexpr uint64_t kBaseAddr = 0x2000;
  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/16, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternCol, kColIndex));

  EXPECT_THAT(ReadMemory<uint16_t>(kBaseAddr, 4),
              testing::ElementsAre(0x1111, 0x2222, 0x3333, 0x4444));
}

TEST_F(RiscVZvtInstructionsTest, Vtle32UnitStrideCol) {
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/4);

  constexpr uint64_t kBaseAddr = 0x2000;
  WriteMemory<uint32_t>(kBaseAddr,
                        {0x11111111, 0x22222222, 0x33333333, 0x44444444});

  constexpr uint32_t kTileId = 4;
  constexpr uint32_t kColIndex = 2;
  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/32, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternCol, kColIndex));

  ASSERT_OK_AND_ASSIGN(
      auto col_vals, GetTileColumn<uint32_t>(kTileId, kColIndex, /*count=*/4));
  EXPECT_THAT(col_vals, testing::ElementsAre(0x11111111, 0x22222222, 0x33333333,
                                             0x44444444));
}

TEST_F(RiscVZvtInstructionsTest, Vtse32UnitStrideRow) {
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/4);

  constexpr uint32_t kTileId = 4;
  constexpr uint32_t kRowIndex = 2;
  ASSERT_OK(SetTileRow<uint32_t>(
      kTileId, kRowIndex, {0x11111111, 0x22222222, 0x33333333, 0x44444444}));

  constexpr uint64_t kBaseAddr = 0x2000;
  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/32, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternRow, kRowIndex));

  EXPECT_THAT(
      ReadMemory<uint32_t>(kBaseAddr, 4),
      testing::ElementsAre(0x11111111, 0x22222222, 0x33333333, 0x44444444));
}

TEST_F(RiscVZvtInstructionsTest, Vtse32UnitStrideCol) {
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/4);

  constexpr uint32_t kTileId = 4;
  constexpr uint32_t kColIndex = 2;
  ASSERT_OK(SetTileColumn<uint32_t>(
      kTileId, kColIndex, {0x11111111, 0x22222222, 0x33333333, 0x44444444}));

  constexpr uint64_t kBaseAddr = 0x2000;
  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/32, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternCol, kColIndex));

  EXPECT_THAT(
      ReadMemory<uint32_t>(kBaseAddr, 4),
      testing::ElementsAre(0x11111111, 0x22222222, 0x33333333, 0x44444444));
}

TEST_F(RiscVZvtInstructionsTest, Vtle64UnitStrideCol) {
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/2);

  constexpr uint64_t kBaseAddr = 0x2000;
  WriteMemory<uint64_t>(kBaseAddr,
                        {0x1111111122222222ULL, 0x3333333344444444ULL});

  constexpr uint32_t kTileId = 8;
  constexpr uint32_t kColIndex = 2;
  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/64, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternCol, kColIndex));

  ASSERT_OK_AND_ASSIGN(
      auto col_vals, GetTileColumn<uint64_t>(kTileId, kColIndex, /*count=*/2));
  EXPECT_THAT(col_vals, testing::ElementsAre(0x1111111122222222ULL,
                                             0x3333333344444444ULL));
}

TEST_F(RiscVZvtInstructionsTest, VtleActiveBodyBounds) {
  // Test that vtle loads min(vl, ETE) elements (EEW=8 => ETE=32, vl=40 => 32).
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/40);

  constexpr uint64_t kBaseAddr = 0x2000;
  WriteMemoryPattern<uint8_t>(kBaseAddr, /*count=*/40,
                              [](size_t i) { return i; });

  constexpr uint32_t kTileId = 0;
  constexpr uint32_t kRowIndex = 0;
  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/8, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternRow, kRowIndex));

  ASSERT_OK_AND_ASSIGN(auto row_vals,
                       GetTileRow<uint8_t>(kTileId, kRowIndex, /*count=*/32));
  std::vector<uint8_t> expected(32);
  std::iota(expected.begin(), expected.end(), 0);
  EXPECT_THAT(row_vals, testing::ElementsAreArray(expected));
}

TEST_F(RiscVZvtInstructionsTest, VtleActiveBodyBounds64) {
  // Test that vtle loads min(vl, ETE) elements for 64-bit
  // (EEW=64 => ETE=16, vl=20 => 16).
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/20);

  constexpr uint64_t kBaseAddr = 0x2000;
  WriteMemoryPattern<uint64_t>(kBaseAddr, /*count=*/20,
                               [](size_t i) { return i; });

  constexpr uint32_t kTileId = 8;
  constexpr uint32_t kRowIndex = 0;
  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/64, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternRow, kRowIndex));

  ASSERT_OK_AND_ASSIGN(auto row_vals,
                       GetTileRow<uint64_t>(kTileId, kRowIndex, /*count=*/16));
  std::vector<uint64_t> expected(16);
  std::iota(expected.begin(), expected.end(), 0);
  EXPECT_THAT(row_vals, testing::ElementsAreArray(expected));
}

TEST_F(RiscVZvtInstructionsTest, Vtse64UnitStrideRow) {
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/2);

  constexpr uint32_t kTileId = 8;
  constexpr uint32_t kRowIndex = 2;
  ASSERT_OK(SetTileRow<uint64_t>(
      kTileId, kRowIndex, {0x1111111122222222ULL, 0x3333333344444444ULL}));

  constexpr uint64_t kBaseAddr = 0x2000;
  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/64, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternRow, kRowIndex));

  EXPECT_THAT(
      ReadMemory<uint64_t>(kBaseAddr, 2),
      testing::ElementsAre(0x1111111122222222ULL, 0x3333333344444444ULL));
}

TEST_F(RiscVZvtInstructionsTest, Vtse64UnitStrideCol) {
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/2);

  constexpr uint32_t kTileId = 8;
  constexpr uint32_t kColIndex = 2;
  ASSERT_OK(SetTileColumn<uint64_t>(
      kTileId, kColIndex, {0x1111111122222222ULL, 0x3333333344444444ULL}));

  constexpr uint64_t kBaseAddr = 0x2000;
  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/64, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternCol, kColIndex));

  EXPECT_THAT(
      ReadMemory<uint64_t>(kBaseAddr, 2),
      testing::ElementsAre(0x1111111122222222ULL, 0x3333333344444444ULL));
}

TEST_F(RiscVZvtInstructionsTest, VtseActiveBodyBounds) {
  // Test that vtse stores min(vl, ETE) elements (EEW=8 => ETE=32, vl=40 => 32).
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/40);

  constexpr uint32_t kTileId = 0;
  constexpr uint32_t kRowIndex = 0;
  ASSERT_OK(SetTileRowPattern<uint8_t>(kTileId, kRowIndex, /*count=*/40,
                                       [](int i) { return i; }));

  constexpr uint64_t kBaseAddr = 0x2000;
  WriteMemoryConstant<uint8_t>(kBaseAddr, /*count=*/40, 0xFF);

  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/8, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternRow, kRowIndex));

  std::vector<uint8_t> expected(40, 0xFF);
  std::iota(expected.begin(), expected.begin() + 32, 0);
  EXPECT_THAT(ReadMemory<uint8_t>(kBaseAddr, 40),
              testing::ElementsAreArray(expected));
}

TEST_F(RiscVZvtInstructionsTest, VtseActiveBodyBounds64) {
  // Test that vtse stores min(vl, ETE) elements for 64-bit
  // (EEW=64 => ETE=16, vl=20 => 16).
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/20);

  constexpr uint32_t kTileId = 8;
  constexpr uint32_t kRowIndex = 0;
  ASSERT_OK(SetTileRowPattern<uint64_t>(kTileId, kRowIndex, /*count=*/20,
                                        [](int i) { return i; }));

  constexpr uint64_t kBaseAddr = 0x2000;
  WriteMemoryConstant<uint64_t>(kBaseAddr, /*count=*/20, 0xFFFFFFFFFFFFFFFFULL);

  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/64, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternRow, kRowIndex));

  std::vector<uint64_t> expected(20, 0xFFFFFFFFFFFFFFFFULL);
  std::iota(expected.begin(), expected.begin() + 16, 0);
  EXPECT_THAT(ReadMemory<uint64_t>(kBaseAddr, 20),
              testing::ElementsAreArray(expected));
}

TEST_F(RiscVZvtInstructionsTest, MatmulOperandLmulDivisibilityTrap) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul2, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  // Misaligned register group v9 for LMUL=2!
  ExecuteMatmul(&RiscVZvtVtfmm, /*reg_a=*/8, /*reg_b=*/9, /*dest_tile=*/0);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, MatmulOperandKmaxBoundsTrap) {
  ConfigureMatrixUnit(kMtwiden4, kVsew8, kVlmul1, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteMatmul(&RiscVZvtVtfmm, /*reg_a=*/10, /*reg_b=*/8, /*dest_tile=*/0);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, MatmulUint8ByUint8) {
  // Zvt spec v0.4: vtmmu.tvv (altfmt=0) uint8 x uint8 -> int32.
  ConfigureMatrixUnit(kMtwiden4, kVsew8, kVlmul1, /*vl=*/2, /*tk=*/2,
                      /*tm=*/2);

  // Pre-populate destination tile 0 with initial accumulator values.
  ASSERT_OK(SetTileMatrix<int32_t>(/*tile=*/0, {{50, 0}, {0, 0}}));

  // KMAX(8) = 4, kStride = 8 / 4 = 2.
  // Matrix A: kk=0 -> v8: [200, 10]; kk=1 -> v10: [150, 20].
  WriteVectorRegister<uint8_t>("v8", {200, 10});
  WriteVectorRegister<uint8_t>("v10", {150, 20});

  // Matrix B: kk=0 -> v16: [130, 5]; kk=1 -> v18: [140, 7].
  WriteVectorRegister<uint8_t>("v16", {130, 5});
  WriteVectorRegister<uint8_t>("v18", {140, 7});

  ExecuteMatmul(&RiscVZvtVtmmu);

  // C[0,0] = 50 + 200*130 + 150*140 = 47050, C[0,1] = 0 + 200*5 + 150*7 = 2050
  // C[1,0] = 0  + 10*130  + 20*140  = 4100,  C[1,1] = 0 + 10*5  + 20*7  = 190
  ASSERT_OK_AND_ASSIGN(
      auto result, GetTileRegion<int32_t>(/*tile=*/0, /*rows=*/2, /*cols=*/2));
  EXPECT_THAT(result, testing::ElementsAre(47050, 2050, 4100, 190));

  EXPECT_EQ(state_->mstatus()->ms(), kMstatusStateDirty);
  EXPECT_EQ(vsstatus()->ms(),
            state_->v() ? kMstatusStateDirty : kMstatusStateInitial);
}

TEST_F(RiscVZvtInstructionsTest, VtmmuDynamicDispatchAltfmt0) {
  // Zvt spec v0.4: vtmmu.tvv (altfmt=0) uint8 x uint8 -> int32.
  ConfigureMatrixUnit(kMtwiden4, kVsew8, kVlmul1, /*vl=*/2, /*tk=*/2,
                      /*tm=*/2);

  ASSERT_OK(SetTileMatrix<int32_t>(/*tile=*/0, {{50, 0}, {0, 0}}));
  WriteVectorRegister<uint8_t>("v8", {200, 10});
  WriteVectorRegister<uint8_t>("v10", {150, 20});
  WriteVectorRegister<uint8_t>("v16", {130, 5});
  WriteVectorRegister<uint8_t>("v18", {140, 7});

  ExecuteMatmul(&RiscVZvtVtmmu);

  ASSERT_OK_AND_ASSIGN(
      auto result, GetTileRegion<int32_t>(/*tile=*/0, /*rows=*/2, /*cols=*/2));
  EXPECT_THAT(result, testing::ElementsAre(47050, 2050, 4100, 190));
  EXPECT_EQ(state_->mstatus()->ms(), kMstatusStateDirty);
  EXPECT_EQ(vsstatus()->ms(),
            state_->v() ? kMstatusStateDirty : kMstatusStateInitial);
}

TEST_F(RiscVZvtInstructionsTest, VtmmuDynamicDispatchAltfmt1) {
  // Zvt spec v0.4: vtmmu.tvv (altfmt=1) uint8 x int8 -> int32.
  ConfigureMatrixUnit(kMtwiden4, kVsew8, kVlmul1, /*vl=*/2, /*tk=*/2,
                      /*tm=*/2, /*vstart=*/0, /*vta=*/false, /*vma=*/false,
                      /*altfmt=*/true);

  ASSERT_OK(SetTileMatrix<int32_t>(/*tile=*/0, {{50, 0}, {0, 0}}));
  WriteVectorRegister<uint8_t>("v8", {200, 10});
  WriteVectorRegister<uint8_t>("v10", {150, 20});
  WriteVectorRegister<uint8_t>("v16", {static_cast<uint8_t>(-126), 5});
  WriteVectorRegister<uint8_t>("v18", {static_cast<uint8_t>(-116), 7});

  ExecuteMatmul(&RiscVZvtVtmmu);

  ASSERT_OK_AND_ASSIGN(
      auto result, GetTileRegion<int32_t>(/*tile=*/0, /*rows=*/2, /*cols=*/2));
  EXPECT_THAT(result, testing::ElementsAre(-42550, 2050, -3580, 190));
}

TEST_F(RiscVZvtInstructionsTest, VtmmsDynamicDispatchAltfmt0) {
  // Zvt spec v0.4: vtmms.tvv (altfmt=0) int8 x uint8 -> int32.
  ConfigureMatrixUnit(kMtwiden4, kVsew8, kVlmul1, /*vl=*/2, /*tk=*/2,
                      /*tm=*/2);

  ASSERT_OK(SetTileMatrix<int32_t>(/*tile=*/0, {{100, -20}, {30, 40}}));
  WriteVectorRegister<uint8_t>("v8", {static_cast<uint8_t>(-50), 5});
  WriteVectorRegister<uint8_t>("v10", {40, static_cast<uint8_t>(-10)});
  WriteVectorRegister<uint8_t>("v16", {200, 10});
  WriteVectorRegister<uint8_t>("v18", {150, 20});

  ExecuteMatmul(&RiscVZvtVtmms);

  ASSERT_OK_AND_ASSIGN(
      auto result, GetTileRegion<int32_t>(/*tile=*/0, /*rows=*/2, /*cols=*/2));
  EXPECT_THAT(result, testing::ElementsAre(-3900, 280, -470, -110));
}

TEST_F(RiscVZvtInstructionsTest, MatmulUint8ByInt8) {
  // Zvt spec v0.4: vtmmu.tvv (altfmt=1) uint8 x int8 -> int32.
  ConfigureMatrixUnit(kMtwiden4, kVsew8, kVlmul1, /*vl=*/2, /*tk=*/2,
                      /*tm=*/2, /*vstart=*/0, /*vta=*/false, /*vma=*/false,
                      /*altfmt=*/true);

  ASSERT_OK(SetTileMatrix<int32_t>(/*tile=*/0, {{100, -20}, {30, 40}}));
  WriteVectorRegister<uint8_t>("v8", {200, 10});
  WriteVectorRegister<uint8_t>("v10", {150, 20});
  WriteVectorRegister<uint8_t>("v16", {static_cast<uint8_t>(-50), 5});
  WriteVectorRegister<uint8_t>("v18", {40, static_cast<uint8_t>(-10)});

  ExecuteMatmul(&RiscVZvtVtmmu);

  // C[0,0] = 100 + 200*(-50) + 150*40    = -3900
  // C[0,1] = -20 + 200*5     + 150*(-10) = -520
  // C[1,0] = 30  + 10*(-50)  + 20*40     = 330
  // C[1,1] = 40  + 10*5      + 20*(-10)  = -110
  ASSERT_OK_AND_ASSIGN(
      auto result, GetTileRegion<int32_t>(/*tile=*/0, /*rows=*/2, /*cols=*/2));
  EXPECT_THAT(result, testing::ElementsAre(-3900, -520, 330, -110));

  EXPECT_EQ(state_->mstatus()->ms(), kMstatusStateDirty);
  EXPECT_EQ(vsstatus()->ms(),
            state_->v() ? kMstatusStateDirty : kMstatusStateInitial);
}

TEST_F(RiscVZvtInstructionsTest, VtmmsTvvSignedUnsignedMatmul) {
  // Zvt spec v0.4: vtmms.tvv (altfmt=0) int8 x uint8 -> int32.
  ConfigureMatrixUnit(kMtwiden4, kVsew8, kVlmul1, /*vl=*/2, /*tk=*/2,
                      /*tm=*/2);

  ASSERT_OK(SetTileMatrix<int32_t>(/*tile=*/0, {{100, -20}, {30, 40}}));
  WriteVectorRegister<uint8_t>("v8", {static_cast<uint8_t>(-50), 5});
  WriteVectorRegister<uint8_t>("v10", {40, static_cast<uint8_t>(-10)});
  WriteVectorRegister<uint8_t>("v16", {200, 10});
  WriteVectorRegister<uint8_t>("v18", {150, 20});

  ExecuteMatmul(&RiscVZvtVtmms);

  // C[0,0] = 100 + (-50)*200 + 40*150    = -3900
  // C[0,1] = -20 + (-50)*10  + 40*20     = 280
  // C[1,0] = 30  + 5*200     + (-10)*150 = -470
  // C[1,1] = 40  + 5*10      + (-10)*20  = -110
  ASSERT_OK_AND_ASSIGN(
      auto result, GetTileRegion<int32_t>(/*tile=*/0, /*rows=*/2, /*cols=*/2));
  EXPECT_THAT(result, testing::ElementsAre(-3900, 280, -470, -110));

  EXPECT_EQ(state_->mstatus()->ms(), kMstatusStateDirty);
  EXPECT_EQ(vsstatus()->ms(),
            state_->v() ? kMstatusStateDirty : kMstatusStateInitial);
}

TEST_F(RiscVZvtInstructionsTest, VtmmsTvvSignedSignedMatmul) {
  // Zvt spec v0.4: vtmms.tvv (altfmt=1) int8 x int8 -> int32.
  ConfigureMatrixUnit(kMtwiden4, kVsew8, kVlmul1, /*vl=*/2, /*tk=*/2,
                      /*tm=*/2, /*vstart=*/0, /*vta=*/false, /*vma=*/false,
                      /*altfmt=*/true);

  ASSERT_OK(SetTileMatrix<int32_t>(/*tile=*/0, {{100, -20}, {30, 40}}));
  WriteVectorRegister<uint8_t>("v8", {static_cast<uint8_t>(-50), 5});
  WriteVectorRegister<uint8_t>("v10", {40, static_cast<uint8_t>(-10)});
  WriteVectorRegister<uint8_t>("v16", {static_cast<uint8_t>(-30), 10});
  WriteVectorRegister<uint8_t>("v18", {20, static_cast<uint8_t>(-15)});

  ExecuteMatmul(&RiscVZvtVtmms);

  // C[0,0] = 100 + (-50)*(-30) + 40*20     = 2400
  // C[0,1] = -20 + (-50)*10    + 40*(-15)  = -1120
  // C[1,0] = 30  + 5*(-30)     + (-10)*20  = -320
  // C[1,1] = 40  + 5*10        + (-10)*(-15) = 240
  ASSERT_OK_AND_ASSIGN(
      auto result, GetTileRegion<int32_t>(/*tile=*/0, /*rows=*/2, /*cols=*/2));
  EXPECT_THAT(result, testing::ElementsAre(2400, -1120, -320, 240));

  EXPECT_EQ(state_->mstatus()->ms(), kMstatusStateDirty);
  EXPECT_EQ(vsstatus()->ms(),
            state_->v() ? kMstatusStateDirty : kMstatusStateInitial);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmTvvFp32Matmul) {
  // Zvt spec v0.4: vtfmm.tvv FP32 x FP32 -> FP32.
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/2, /*tk=*/1,
                      /*tm=*/2);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{1.5f, -0.5f}, {0.0f, 2.25f}}));
  WriteVectorRegister<float>("v8", {2.0f, 0.5f});
  WriteVectorRegister<float>("v16", {3.0f, 1.0f});

  ExecuteMatmul(&RiscVZvtVtfmm);

  ASSERT_OK_AND_ASSIGN(
      auto result, GetTileRegion<float>(/*tile=*/0, /*rows=*/2, /*cols=*/2));
  EXPECT_THAT(result, testing::ElementsAre(7.5f, 1.5f, 1.5f, 2.75f));

  EXPECT_EQ(state_->mstatus()->ms(), kMstatusStateDirty);
  EXPECT_EQ(vsstatus()->ms(),
            state_->v() ? kMstatusStateDirty : kMstatusStateInitial);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmTvvFp32DynamicRoundingMode) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/1, /*tk=*/1,
                      /*tm=*/1);
  // Set FCSR frm to RUP - Round Up (towards +inf).
  fp_state_->fcsr()->Write(*FPRoundingMode::kRoundUp << kFcsrFrmShift);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{1.0f}}));
  WriteVectorRegister<uint32_t>("v8", {0x3f800000});   // 1.0f
  WriteVectorRegister<uint32_t>("v16", {0x33800000});  // 2^-24

  ExecuteMatmul(&RiscVZvtVtfmm);

  // In RUP, 1.0 + 2^-24 rounds up to 1.0 + 2^-23 (0x3f800001).
  ASSERT_OK_AND_ASSIGN(float result_rup,
                       GetTileElem<float>(/*tile=*/0, /*row=*/0, /*col=*/0));
  EXPECT_EQ(absl::bit_cast<uint32_t>(result_rup), 0x3f800001U);

  // Re-run with RDN - Round Down (towards -inf).
  fp_state_->fcsr()->Write(*FPRoundingMode::kRoundDown << kFcsrFrmShift);
  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{1.0f}}));
  instruction_->Execute(nullptr);

  ASSERT_OK_AND_ASSIGN(float result_rdn,
                       GetTileElem<float>(/*tile=*/0, /*row=*/0, /*col=*/0));
  EXPECT_EQ(absl::bit_cast<uint32_t>(result_rdn), 0x3f800000U);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmTvvFp64Matmul) {
  // Zvt spec v0.4: vtfmm.tvv FP64 x FP64 -> FP64.
  ConfigureMatrixUnit(kMtwiden1, kVsew64, kVlmul1, /*vl=*/2, /*tk=*/1,
                      /*tm=*/2);

  ASSERT_OK(SetTileMatrix<double>(/*tile=*/0, {{1.5, -0.5}, {0.0, 2.25}}));
  WriteVectorRegister<double>("v8", {2.0, 0.5});
  WriteVectorRegister<double>("v16", {3.0, 1.0});

  ExecuteMatmul(&RiscVZvtVtfmm);

  ASSERT_OK_AND_ASSIGN(
      auto result, GetTileRegion<double>(/*tile=*/0, /*rows=*/2, /*cols=*/2));
  EXPECT_THAT(result, testing::ElementsAre(7.5, 1.5, 1.5, 2.75));

  EXPECT_EQ(state_->mstatus()->ms(), kMstatusStateDirty);
  EXPECT_EQ(vsstatus()->ms(),
            state_->v() ? kMstatusStateDirty : kMstatusStateInitial);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmTvvFp16BulkNormalization) {
  // Zvt spec v0.4: vtfmm.tvv FP16 x FP16 -> FP32.
  ConfigureMatrixUnit(kMtwiden2, kVsew16, kVlmul1, /*vl=*/1, /*tk=*/2,
                      /*tm=*/1);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{1.0f}}));
  WriteVectorRegister<uint16_t>("v8", {0x7800});   // 2^15
  WriteVectorRegister<uint16_t>("v12", {0xF800});  // -2^15
  WriteVectorRegister<uint16_t>("v16", {0x7800});  // 2^15
  WriteVectorRegister<uint16_t>("v20", {0x7800});  // 2^15

  ExecuteMatmul(&RiscVZvtVtfmm);

  // With sequential FP32 addition: (1.0 + 2^30) - 2^30 = 0.0
  // With RVBNA bulk normalization: 1.0 + (2^30 - 2^30) = 1.0
  ASSERT_OK_AND_ASSIGN(float result,
                       GetTileElem<float>(/*tile=*/0, /*row=*/0, /*col=*/0));
  EXPECT_EQ(result, 1.0f);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmTvvFp16RoundToOddConversion) {
  // Zvt spec v0.4: vtfmm.tvv FP16 x FP16 -> FP32.
  ConfigureMatrixUnit(kMtwiden2, kVsew16, kVlmul1, /*vl=*/1, /*tk=*/2,
                      /*tm=*/1);
  fp_state_->fcsr()->Write(0U);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{0.0f}}));
  // Inputs yielding sum of products = 1.0 * 1.0 + 2^-24 * 2^-24.
  WriteVectorRegister<uint16_t>("v8", {0x3C00});   // 1.0
  WriteVectorRegister<uint16_t>("v12", {0x0001});  // 2^-24 (min subnormal)
  WriteVectorRegister<uint16_t>("v16", {0x3C00});  // 1.0
  WriteVectorRegister<uint16_t>("v20", {0x0001});  // 2^-24

  ExecuteMatmul(&RiscVZvtVtfmm);

  ASSERT_OK_AND_ASSIGN(float result,
                       GetTileElem<float>(/*tile=*/0, /*row=*/0, /*col=*/0));
  EXPECT_EQ(absl::bit_cast<uint32_t>(result),
            0x3F800001u);  // 1.0 + 2^-23 (ROD)
}

TEST_F(RiscVZvtInstructionsTest, VtfmmTvvFp16RoundToOddAddition) {
  // Zvt spec v0.4: vtfmm.tvv FP16 x FP16 -> FP32.
  ConfigureMatrixUnit(kMtwiden2, kVsew16, kVlmul1, /*vl=*/1, /*tk=*/2,
                      /*tm=*/1);
  fp_state_->fcsr()->Write(0U);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{1.0f}}));
  WriteVectorRegister<uint16_t>("v8", {0x0000});   // 0.0
  WriteVectorRegister<uint16_t>("v12", {0x0C00});  // 2^-12
  WriteVectorRegister<uint16_t>("v16", {0x0000});  // 0.0
  WriteVectorRegister<uint16_t>("v20", {0x0C00});  // 2^-12

  ExecuteMatmul(&RiscVZvtVtfmm);

  ASSERT_OK_AND_ASSIGN(float result,
                       GetTileElem<float>(/*tile=*/0, /*row=*/0, /*col=*/0));
  EXPECT_EQ(absl::bit_cast<uint32_t>(result),
            0x3F800001u);  // 1.0 + 2^-23 (ROD)
}

TEST_F(RiscVZvtInstructionsTest, VtfmmAltTvvBf16Matmul) {
  // Zvt spec v0.4: vtfmm.alt.tvv BF16 x BF16 -> FP32.
  ConfigureMatrixUnit(kMtwiden2, kVsew16, kVlmul1, /*vl=*/2, /*tk=*/2,
                      /*tm=*/2, /*vstart=*/0, /*vta=*/false, /*vma=*/false,
                      /*altfmt=*/true);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{1.5f, -0.5f}, {0.0f, 2.25f}}));
  WriteVectorRegister<uint16_t>("v8", {0x4000, 0x3F00});   // 2.0f, 0.5f
  WriteVectorRegister<uint16_t>("v12", {0x4100, 0x3E80});  // 8.0f, 0.25f
  WriteVectorRegister<uint16_t>("v16", {0x4040, 0x3F80});  // 3.0f, 1.0f
  WriteVectorRegister<uint16_t>("v20", {0x4080, 0x40C0});  // 4.0f, 6.0f

  ExecuteMatmul(&RiscVZvtVtfmmAlt);

  ASSERT_OK_AND_ASSIGN(
      auto result, GetTileRegion<float>(/*tile=*/0, /*rows=*/2, /*cols=*/2));
  EXPECT_THAT(result, testing::ElementsAre(39.5f, 49.5f, 2.5f, 4.25f));

  EXPECT_EQ(state_->mstatus()->ms(), kMstatusStateDirty);
  EXPECT_EQ(vsstatus()->ms(),
            state_->v() ? kMstatusStateDirty : kMstatusStateInitial);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmAltTvvSew32Trap) {
  ConfigureMatrixUnit(kMtwiden2, kVsew32, kVlmul1, /*vl=*/2, /*tk=*/1,
                      /*tm=*/2);
  ScopedTrapCapture trap_capture(state_.get());

  // SEW=32 is invalid for vtfmm.alt.tvv.
  ExecuteMatmul(&RiscVZvtVtfmmAlt);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmAltTvvSew64Trap) {
  ConfigureMatrixUnit(kMtwiden2, kVsew64, kVlmul1, /*vl=*/2, /*tk=*/1,
                      /*tm=*/2);
  ScopedTrapCapture trap_capture(state_.get());

  // SEW=64 is invalid for vtfmm.alt.tvv.
  ExecuteMatmul(&RiscVZvtVtfmmAlt);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmTvvFp16Matmul) {
  // Zvt spec v0.4: vtfmm.tvv FP16 x FP16 -> FP32.
  ConfigureMatrixUnit(kMtwiden2, kVsew16, kVlmul1, /*vl=*/2, /*tk=*/2,
                      /*tm=*/2);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{1.5f, -0.5f}, {0.0f, 2.25f}}));
  WriteVectorRegister<uint16_t>("v8", {0x4000, 0x3800});   // 2.0f, 0.5f
  WriteVectorRegister<uint16_t>("v12", {0x4800, 0x3400});  // 8.0f, 0.25f
  WriteVectorRegister<uint16_t>("v16", {0x4200, 0x3c00});  // 3.0f, 1.0f
  WriteVectorRegister<uint16_t>("v20", {0x4400, 0x4600});  // 4.0f, 6.0f

  ExecuteMatmul(&RiscVZvtVtfmm);

  // C = [[1.5, -0.5], [0.0, 2.25]] + [[38.0, 50.0], [2.5, 2.0]]
  ASSERT_OK_AND_ASSIGN(
      auto result, GetTileRegion<float>(/*tile=*/0, /*rows=*/2, /*cols=*/2));
  EXPECT_THAT(result, testing::ElementsAre(39.5f, 49.5f, 2.5f, 4.25f));

  EXPECT_EQ(state_->mstatus()->ms(), kMstatusStateDirty);
  EXPECT_EQ(vsstatus()->ms(),
            state_->v() ? kMstatusStateDirty : kMstatusStateInitial);
}

TEST_F(RiscVZvtInstructionsTest, MatmulDestTileModuloReservationTrap) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  // Invalid tile 1 since alignment is 4 (NTiles=4 when TEW=32, Align=16/4=4).
  ExecuteMatmul(&RiscVZvtVtfmm, /*reg_a=*/0, /*reg_b=*/8, /*dest_tile=*/1);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, MatmulDestTileModuloReservationTrapTEW64) {
  ConfigureMatrixUnit(kMtwiden1, kVsew64, kVlmul1, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  // Invalid tile 3 since alignment is 2 (NTiles=8 when TEW=64, Align=16/8=2).
  ExecuteMatmul(&RiscVZvtVtfmm, /*reg_a=*/0, /*reg_b=*/8, /*dest_tile=*/3);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, MatmulVstartNonZeroTrap) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4, /*tk=*/0,
                      /*tm=*/0, /*vstart=*/1);
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteMatmul(&RiscVZvtVtfmm, /*reg_a=*/0, /*reg_b=*/8, /*dest_tile=*/0);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, MatmulZeroDimBypass) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4, /*tk=*/4,
                      /*tm=*/0);

  ExecuteMatmul(&RiscVZvtVtfmm, /*reg_a=*/0, /*reg_b=*/8, /*dest_tile=*/0);

  // Status should remain 1 (initial), not 3 (dirty), due to bypass.
  EXPECT_EQ(state_->mstatus()->ms(), kMstatusStateInitial);
}

TEST_F(RiscVZvtInstructionsTest, CeilEteEveScalesVlmulCorrectly) {
  // Use a configuration where vector register byte length is 16 (128 bits)
  // and tile_elements is 32, so that effective_tile_elements / ete_eve_lmul_den
  // > 1.
  FlatDemandMemory memory;
  RiscVState rv_state("rv32_16B", RiscVXlen::RV32, &memory);
  RiscVVectorState vec_state(&rv_state, /*vector_register_byte_length=*/16);
  rv_state.set_rv_vector(&vec_state);

  ASSERT_OK_AND_ASSIGN(auto matrix_state, RiscVZvtMatrixState::Create(
                                              &rv_state, /*tile_elements=*/32));

  auto* inst = new Instruction(0x1000, &rv_state);
  inst->set_size(4);
  inst->AppendSource(new ImmediateOperand<uint32_t>(kMtwiden1));
  inst->AppendSource(new ImmediateOperand<uint32_t>(kVsew8));
  inst->AppendSource(new ImmediateOperand<uint32_t>(kAltfmtStandard));
  inst->set_semantic_function(&RiscVZvtMsetMtypei);

  inst->Execute(nullptr);
  inst->DecRef();

  const auto expected = CalculateExpectedVectorGeometry(
      /*vlen_bytes=*/16, /*tile_elements=*/32, /*sew_bits=*/8, /*twiden=*/1);
  EXPECT_EQ(vec_state.vtype() & kVlmulMask, expected.vlmul);
  EXPECT_EQ(vec_state.max_vector_length(), expected.max_vector_length);
}

TEST_F(RiscVZvtInstructionsTest, UpdateVectorMatrixGeometryEteEveDenScaling) {
  // Configuration: vector register length = 32 bytes (256 bits),
  // tile_elements = 32, sew = 16 bits (2 bytes), twiden = 1.
  FlatDemandMemory memory;
  RiscVState rv_state("rv32_32B", RiscVXlen::RV32, &memory);
  RiscVVectorState vec_state(&rv_state, /*vector_register_byte_length=*/32);
  rv_state.set_rv_vector(&vec_state);

  ASSERT_OK_AND_ASSIGN(auto matrix_state, RiscVZvtMatrixState::Create(
                                              &rv_state, /*tile_elements=*/32));

  auto* inst = new Instruction(0x1000, &rv_state);
  inst->set_size(4);
  inst->AppendSource(new ImmediateOperand<uint32_t>(kMtwiden1));
  inst->AppendSource(new ImmediateOperand<uint32_t>(kVsew16));
  inst->AppendSource(new ImmediateOperand<uint32_t>(kAltfmtStandard));
  inst->set_semantic_function(&RiscVZvtMsetMtypei);

  inst->Execute(nullptr);
  inst->DecRef();

  const auto expected = CalculateExpectedVectorGeometry(
      /*vlen_bytes=*/32, /*tile_elements=*/32, /*sew_bits=*/16, /*twiden=*/1);
  EXPECT_EQ(vec_state.vtype() & kVlmulMask, expected.vlmul);
  EXPECT_EQ(vec_state.max_vector_length(), expected.max_vector_length);
}

TEST_F(RiscVZvtInstructionsTest,
       UpdateVectorMatrixGeometryEteEveDenScalingPlusOne) {
  // Configuration: VLEN = 16 bytes (128 bits), TE = 32, SEW = 32 bits (4B).
  // twiden = 1 -> tew = 32 bits.
  FlatDemandMemory memory;
  RiscVState rv_state("rv32_16B_sew32", RiscVXlen::RV32, &memory);
  RiscVVectorState vec_state(&rv_state, /*vector_register_byte_length=*/16);
  rv_state.set_rv_vector(&vec_state);

  ASSERT_OK_AND_ASSIGN(auto matrix_state, RiscVZvtMatrixState::Create(
                                              &rv_state, /*tile_elements=*/32));

  auto* inst = new Instruction(0x1000, &rv_state);
  inst->set_size(4);
  inst->AppendSource(new ImmediateOperand<uint32_t>(kMtwiden1));
  inst->AppendSource(new ImmediateOperand<uint32_t>(kVsew32));
  inst->AppendSource(new ImmediateOperand<uint32_t>(kAltfmtStandard));
  inst->set_semantic_function(&RiscVZvtMsetMtypei);

  inst->Execute(nullptr);
  inst->DecRef();

  const auto expected = CalculateExpectedVectorGeometry(
      /*vlen_bytes=*/16, /*tile_elements=*/32, /*sew_bits=*/32, /*twiden=*/1);
  EXPECT_EQ(vec_state.vtype() & kVlmulMask, expected.vlmul);
  EXPECT_EQ(vec_state.max_vector_length(), expected.max_vector_length);
}

TEST_F(RiscVZvtInstructionsTest, UpdateVectorMatrixGeometryMaxKScaling) {
  // Configuration: VLEN = 8 bytes (64 bits), TE = 32, SEW = 8 bits (1B).
  // twiden = 1 -> tew = 8 bits.
  FlatDemandMemory memory;
  RiscVState rv_state("rv32_8B_sew8", RiscVXlen::RV32, &memory);
  RiscVVectorState vec_state(&rv_state, /*vector_register_byte_length=*/8);
  rv_state.set_rv_vector(&vec_state);

  ASSERT_OK_AND_ASSIGN(auto matrix_state, RiscVZvtMatrixState::Create(
                                              &rv_state, /*tile_elements=*/32));

  auto* inst = new Instruction(0x1000, &rv_state);
  inst->set_size(4);
  inst->AppendSource(new ImmediateOperand<uint32_t>(kMtwiden1));
  inst->AppendSource(new ImmediateOperand<uint32_t>(kVsew8));
  inst->AppendSource(new ImmediateOperand<uint32_t>(kAltfmtStandard));
  inst->set_semantic_function(&RiscVZvtMsetMtypei);

  inst->Execute(nullptr);
  inst->DecRef();

  const auto expected = CalculateExpectedVectorGeometry(
      /*vlen_bytes=*/8, /*tile_elements=*/32, /*sew_bits=*/8, /*twiden=*/1);
  EXPECT_EQ(vec_state.vtype() & kVlmulMask, expected.vlmul);
  EXPECT_EQ(vec_state.max_vector_length(), expected.max_vector_length);
}

TEST_F(RiscVZvtInstructionsTest, UpdateVectorMatrixGeometryTwidenScaling) {
  // Configuration: VLEN = 16 bytes (128 bits), TE = 32, SEW = 16 bits (2B).
  // mtwiden = 2 (twiden = 2) -> tew = 32 bits (<= 32, no vill).
  FlatDemandMemory memory;
  RiscVState rv_state("rv32_16B_sew16_twiden2", RiscVXlen::RV32, &memory);
  RiscVVectorState vec_state(&rv_state, /*vector_register_byte_length=*/16);
  rv_state.set_rv_vector(&vec_state);

  ASSERT_OK_AND_ASSIGN(auto matrix_state, RiscVZvtMatrixState::Create(
                                              &rv_state, /*tile_elements=*/32));

  auto* inst = new Instruction(0x1000, &rv_state);
  inst->set_size(4);
  inst->AppendSource(new ImmediateOperand<uint32_t>(kMtwiden2));
  inst->AppendSource(new ImmediateOperand<uint32_t>(kVsew16));
  inst->AppendSource(new ImmediateOperand<uint32_t>(kAltfmtStandard));
  inst->set_semantic_function(&RiscVZvtMsetMtypei);

  inst->Execute(nullptr);
  inst->DecRef();

  const auto expected = CalculateExpectedVectorGeometry(
      /*vlen_bytes=*/16, /*tile_elements=*/32, /*sew_bits=*/16, /*twiden=*/2);
  EXPECT_EQ(vec_state.vtype() & kVlmulMask, expected.vlmul);
  EXPECT_EQ(vec_state.max_vector_length(), expected.max_vector_length);
}

TEST_F(RiscVZvtInstructionsTest, SetVectorStateDirtyRv32UpdatesMstatusVsAndSd) {
  // Clear mstatus.VS to Initial (1) and clear SD.
  SetMstatusVs(kMstatusStateInitial);
  state_->mstatus()->Set(state_->mstatus()->GetUint64() & ~kMstatusSdMaskRv32);

  EXPECT_EQ(GetMstatusVs(), kMstatusStateInitial);
  EXPECT_EQ(GetMstatusSd(/*is_rv64=*/false), kMstatusSdClean);

  // Execute msetmtypei, which calls SetVectorStateDirty.
  ResetInstruction();
  instruction_->AppendSource(
      new ImmediateOperand<uint32_t>(/*mtwiden=*/kMtwiden1));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vsew=*/kVsew8));
  instruction_->AppendSource(
      new ImmediateOperand<uint32_t>(/*altfmt=*/kAltfmtStandard));
  instruction_->set_semantic_function(&RiscVZvtMsetMtypei);

  instruction_->Execute(nullptr);

  // Must set VS to Dirty (3) and SD (bit 31 on RV32) to 1.
  // If state != nullptr return; was mutated, neither bit is set.
  // If sd_bit was mutated to 63 on RV32, bit 31 is not set.
  EXPECT_EQ(GetMstatusVs(), kMstatusStateDirty);
  EXPECT_EQ(GetMstatusSd(/*is_rv64=*/false), kMstatusSdSet);
  EXPECT_EQ(GetMstatusSd(/*is_rv64=*/true), kMstatusSdClean);
}

TEST_F(RiscVZvtInstructionsTest, SetVectorStateDirtyRv64UpdatesMstatusVsAndSd) {
  ASSERT_OK_AND_ASSIGN(auto rv64, Rv64TestContext::Create("rv64_test"));

  // Clear VS and SD (bit 63 on RV64).
  rv64->SetMstatusVs(kMstatusStateInitial);
  rv64->state.mstatus()->Set(rv64->state.mstatus()->GetUint64() &
                             ~kMstatusSdMaskRv64);

  EXPECT_EQ(rv64->GetMstatusVs(), kMstatusStateInitial);
  EXPECT_EQ(rv64->GetMstatusSd(), kMstatusSdClean);

  auto* inst = new Instruction(0x1000, &rv64->state);
  inst->set_size(4);
  inst->AppendSource(new ImmediateOperand<uint32_t>(kMtwiden1));
  inst->AppendSource(new ImmediateOperand<uint32_t>(kVsew8));
  inst->AppendSource(new ImmediateOperand<uint32_t>(kAltfmtStandard));
  inst->set_semantic_function(&RiscVZvtMsetMtypei);

  inst->Execute(nullptr);
  inst->DecRef();

  // Must set VS to Dirty (3) and SD (bit 63 on RV64) to 1.
  // If sd_bit was mutated to 31 on RV64, bit 63 is not set.
  EXPECT_EQ(rv64->GetMstatusVs(), kMstatusStateDirty);
  EXPECT_EQ(rv64->GetMstatusSd(), kMstatusSdSet);
}

TEST_F(RiscVZvtInstructionsTest, MsetmtypeInvalidSewSetsVill) {
  ResetInstruction();
  // SEW = 64 bits (kVsew64 > kElen of 32 bits for Zve32x base).
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(kMtwiden1));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(kVsew64));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(kAltfmtStandard));
  instruction_->set_semantic_function(&RiscVZvtMsetMtypei);

  instruction_->Execute(nullptr);

  // Per spec, when tew > kElen, vill is set and vl is set to 0.
  EXPECT_NE(vector_state_->vtype() & kVillMask, 0U);
  EXPECT_EQ(vector_state_->vector_length(), 0);
}

TEST_F(RiscVZvtInstructionsTest,
       VtZeroClearsActiveTileElementsAndDirtiesState) {
  // Configure matrix state: TM = 4, TN (vl) = 4, TEW = 8.
  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));
  matrix_state_->set_tile_m(4);
  vector_state_->set_vector_length(4);

  // Populate tile 0 with non-zero bytes.
  ASSERT_OK(
      FillTileConstant<uint8_t>(/*tile=*/0, /*rows=*/4, /*cols=*/4, 0xAB));

  // Clear mstatus MS and VS.
  SetMstatusMs(kMstatusStateInitial);
  SetMstatusVs(kMstatusStateInitial);

  ResetInstruction();
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*tile=*/0));

  instruction_->set_semantic_function(&RiscVZvtVtZero);

  instruction_->Execute(nullptr);

  // All active elements in tile 0 must be 0.
  ASSERT_OK_AND_ASSIGN(
      auto tile_elements,
      GetTileRegion<uint8_t>(/*tile=*/0, /*rows=*/4, /*cols=*/4));
  EXPECT_THAT(tile_elements, testing::Each(0));

  // MS and VS must both be Dirty.
  EXPECT_EQ(GetMstatusMs(), kMstatusStateDirty);
  EXPECT_EQ(GetMstatusVs(), kMstatusStateDirty);
}

TEST_F(RiscVZvtInstructionsTest, VtDiscardHintsTileDeadAndDirtiesVectorState) {
  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));
  vector_state_->set_vstart(2);

  // Set VS = initial (1) and MS = dirty (3).
  SetMstatusVs(kMstatusStateInitial);
  SetMstatusMs(kMstatusStateDirty);
  state_->mstatus()->Set(state_->mstatus()->GetUint64() | kMstatusSdMaskRv32);

  ResetInstruction();
  instruction_->set_semantic_function(&RiscVZvtVtDiscard);
  instruction_->Execute(nullptr);

  // vstart is cleared, matrix state reset to initial (1), vector state dirtied
  // (3). If SetMatrixStateInitial's csr->Set(val) was removed, MS remains dirty
  // (3).
  EXPECT_EQ(vector_state_->vstart(), 0);
  EXPECT_EQ(GetMstatusMs(), kMstatusStateInitial);
  EXPECT_EQ(GetMstatusVs(), kMstatusStateDirty);
  EXPECT_EQ(GetMstatusSd(/*is_rv64=*/false), kMstatusSdSet);
}

TEST_F(RiscVZvtInstructionsTest, VtDiscardTrapsWhenVtypeVill) {
  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  vector_state_->SetVectorType(kVillMask);

  // Turn on MS and VS in mstatus so trap is specifically due to vill.
  SetMstatusVs(kMstatusStateInitial);
  SetMstatusMs(kMstatusStateInitial);

  ScopedTrapCapture trap_capture(state_.get());

  ResetInstruction();
  instruction_->set_semantic_function(&RiscVZvtVtDiscard);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtmvVtTransfersTileSliceToVectorRegister) {
  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));
  matrix_state_->set_tile_m(4);
  vector_state_->set_vector_length(4);

  // Ensure MS is on (initial).
  SetMstatusMs(kMstatusStateInitial);

  // Populate row 1 of tile 0: elements at (0, 1, c) = c + 0x10.
  ASSERT_OK(SetTileRowPattern<uint8_t>(
      /*tile=*/0, /*row=*/1, /*count=*/4,
      [](int c) { return static_cast<uint8_t>(c + 0x10); }));

  // TSS for row 1 of tile 0: pattern = row, tile = 0, index = 1.
  SetRegisterValue("x1", EncodeTss(/*tile=*/0, kTssPatternRow, /*index=*/1));

  ResetInstruction();
  instruction_->AppendSource(CreateScalarSource("x1"));
  instruction_->AppendDestination(CreateVectorDestOperand("v2"));
  instruction_->set_semantic_function(&RiscVZvtVtmvVt);

  instruction_->Execute(nullptr);

  // Vector register v2 must receive the row data.
  auto* v2 = state_->GetRegister<RVVectorRegister>("v2").first;
  auto span = v2->data_buffer()->Get<uint8_t>();
  EXPECT_THAT(span.subspan(0, 4), testing::ElementsAre(0x10, 0x11, 0x12, 0x13));
  // VS must be dirty.
  EXPECT_EQ(GetMstatusVs(), kMstatusStateDirty);
}

TEST_F(RiscVZvtInstructionsTest, VtmvTvTransfersVectorRegisterToTileSlice) {
  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));
  matrix_state_->set_tile_m(4);
  vector_state_->set_vector_length(4);

  // Ensure MS is on (initial).
  SetMstatusMs(kMstatusStateInitial);

  // Set up source vector register v4 with 4 bytes: 0x51, 0x52, 0x53, 0x54.
  auto* v4 = state_->GetRegister<RVVectorRegister>("v4").first;
  auto* db = v4->data_buffer();
  auto span = db->Get<uint8_t>();
  span[0] = 0x51;
  span[1] = 0x52;
  span[2] = 0x53;
  span[3] = 0x54;

  // TSS for row 2 of tile 0: pattern = row, tile = 0, index = 2.
  SetRegisterValue("x1", EncodeTss(/*tile=*/0, kTssPatternRow, /*index=*/2));

  ResetInstruction();
  instruction_->AppendSource(CreateVectorSourceOperand("v4"));
  instruction_->AppendSource(CreateScalarSource("x1"));
  instruction_->set_semantic_function(&RiscVZvtVtmvTv);

  instruction_->Execute(nullptr);

  // Check tile elements at (0, 2, c).
  ASSERT_OK_AND_ASSIGN(auto row_elements,
                       GetTileRow<uint8_t>(/*tile=*/0, /*row=*/2, /*count=*/4));
  EXPECT_THAT(row_elements, testing::ElementsAre(0x51, 0x52, 0x53, 0x54));
  // MS must be dirty.
  EXPECT_EQ(GetMstatusMs(), kMstatusStateDirty);
}

TEST_F(RiscVZvtInstructionsTest, VtmvVtResetsVstartToZero) {
  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));
  matrix_state_->set_tile_m(4);
  vector_state_->set_vector_length(4);
  vector_state_->set_vstart(1);

  // Ensure MS is on (initial).
  SetMstatusMs(kMstatusStateInitial);

  // Populate row 1 of tile 0.
  ASSERT_OK(SetTileRowPattern<uint8_t>(
      /*tile=*/0, /*row=*/1, /*count=*/4,
      [](int c) { return static_cast<uint8_t>(c + 0x10); }));

  // TSS for row 1 of tile 0: pattern = row, tile = 0, index = 1.
  SetRegisterValue("x1", EncodeTss(/*tile=*/0, kTssPatternRow, /*index=*/1));
  auto* v2 = state_->GetRegister<RVVectorRegister>("v2").first;
  // Initialize v2 elements: element 0 will be undisturbed since vstart = 1.
  v2->data_buffer()->Get<uint8_t>()[0] = 0xAA;

  ResetInstruction();
  instruction_->AppendSource(CreateScalarSource("x1"));
  instruction_->AppendDestination(CreateVectorDestOperand("v2"));
  instruction_->set_semantic_function(&RiscVZvtVtmvVt);

  instruction_->Execute(nullptr);

  auto span = v2->data_buffer()->Get<uint8_t>();
  EXPECT_THAT(span.subspan(0, 2), testing::ElementsAre(0xAA, 0x11));
  EXPECT_EQ(vector_state_->vstart(), 0);
}

TEST_F(RiscVZvtInstructionsTest, VtmvTvResetsVstartToZero) {
  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));
  matrix_state_->set_tile_m(4);
  vector_state_->set_vector_length(4);
  vector_state_->set_vstart(1);

  // Ensure MS is on (initial).
  SetMstatusMs(kMstatusStateInitial);

  auto* v4 = state_->GetRegister<RVVectorRegister>("v4").first;
  auto* db = v4->data_buffer();
  auto span = db->Get<uint8_t>();
  span[0] = 0x51;
  span[1] = 0x52;
  span[2] = 0x53;
  span[3] = 0x54;

  // Pre-populate tile 0, row 2, element 0 with 0xEE to verify undisturbed.
  ASSERT_OK(matrix_state_->SetElem<uint8_t>(/*tile=*/0, /*row=*/2, /*col=*/0,
                                            /*element_width_in_bits=*/8, 0xEE));

  // TSS for row 2 of tile 0: pattern = row, tile = 0, index = 2.
  SetRegisterValue("x1", EncodeTss(/*tile=*/0, kTssPatternRow, /*index=*/2));

  ResetInstruction();
  instruction_->AppendSource(CreateVectorSourceOperand("v4"));
  instruction_->AppendSource(CreateScalarSource("x1"));
  instruction_->set_semantic_function(&RiscVZvtVtmvTv);

  instruction_->Execute(nullptr);

  ASSERT_OK_AND_ASSIGN(auto row_elements,
                       GetTileRow<uint8_t>(/*tile=*/0, /*row=*/2, /*count=*/2));
  EXPECT_THAT(row_elements, testing::ElementsAre(0xEE, 0x52));
  EXPECT_EQ(vector_state_->vstart(), 0);
}

TEST_F(RiscVZvtInstructionsTest, WriteScalarRv64ClearsUpperBits) {
  // Test WriteScalar on RV64: writes a 32-bit scalar value to an 8-byte
  // destination register operand, ensuring the upper 32 bits are cleared.
  // Ensures WriteScalar clears upper 32 bits on 64-bit registers rather than
  // leaving them untouched.
  ASSERT_OK_AND_ASSIGN(auto rv64, Rv64TestContext::Create("rv64_scalar_test"));

  // Turn on MS in mstatus so instructions don't trap.
  rv64->SetMstatusMs(kMstatusStateInitial);

  // Initialize x1 register with high bits set.
  auto* reg1 = rv64->state.GetRegister<RV64Register>("x1").first;
  reg1->data_buffer()->Set<uint64_t>(0, 0x1234'5678'9ABC'DEF0ULL);

  // Pre-seed a recycled 8-byte DataBuffer with non-zero upper bits to
  // guarantee that WriteScalar writes the full 64 bits and does not leave
  // upper bits intact.
  auto* recycled_db = rv64->state.db_factory()->Allocate<uint64_t>(1);
  recycled_db->Set<uint64_t>(0, 0xDEAD'BEEF'CAFE'BABEULL);
  recycled_db->DecRef();

  auto* inst = new Instruction(0x1000, &rv64->state);
  inst->set_size(4);
  inst->AppendSource(new ImmediateOperand<uint32_t>(/*col=*/4));

  inst->AppendDestination(reg1->CreateDestinationOperand(0));
  inst->set_semantic_function(&RiscVZvtMsetTn);

  inst->Execute(nullptr);
  inst->DecRef();

  // Entire 64 bits must be written (upper 32 bits zeroed).
  EXPECT_EQ(reg1->data_buffer()->Get<uint64_t>(0), 4ULL);
  EXPECT_EQ(reg1->data_buffer()->Get<uint64_t>(0) >> 32, 0ULL);
}

TEST_F(RiscVZvtInstructionsTest, SetMatrixStateDirtyRv64UpdatesMstatusMsAndSd) {
  ASSERT_OK_AND_ASSIGN(auto rv64,
                       Rv64TestContext::Create("rv64_matrix_dirty_test"));

  // Set MS = Initial (1) so instructions don't trap, and clear SD (bit 63).
  rv64->SetMstatusMs(kMstatusStateInitial);
  rv64->state.mstatus()->Set(rv64->state.mstatus()->GetUint64() &
                             ~kMstatusSdMaskRv64);

  EXPECT_EQ(rv64->GetMstatusMs(), kMstatusStateInitial);
  EXPECT_EQ(rv64->GetMstatusSd(), kMstatusSdClean);

  auto* inst = new Instruction(0x1000, &rv64->state);
  inst->set_size(4);
  inst->AppendSource(new ImmediateOperand<uint32_t>(/*tile=*/0));

  inst->set_semantic_function(&RiscVZvtVtZero);

  inst->Execute(nullptr);
  inst->DecRef();

  // Must set MS to Dirty (3) and SD (bit 63 on RV64) to 1.
  EXPECT_EQ(rv64->GetMstatusMs(), kMstatusStateDirty);
  EXPECT_EQ(rv64->GetMstatusSd(), kMstatusSdSet);
  uint64_t mstatus_val = rv64->state.mstatus()->GetUint64();
  EXPECT_EQ((mstatus_val >> kMstatusSdShiftRv32) & 1ULL, kMstatusSdClean);
}

TEST_F(RiscVZvtInstructionsTest, DoubleToFloatRODInexactJamsLsb) {
  // 1.0f in IEEE 754 is 0x3f800000 (even mantissa, LSB is 0).
  // 1.0 + 1e-12 in double precision rounds towards zero to 1.0f.
  // Inexact check triggers, and bit 0 is jammed to 1, producing 0x3f800001.
  double inexact_d = 1.0 + 1e-12;
  float res = DoubleToFloatROD(inexact_d, fp_state_.get());
  EXPECT_EQ(absl::bit_cast<uint32_t>(res), 0x3f800001U);

  // Exact representation: 1.0 is exact in float, so LSB is not jammed.
  float exact_res = DoubleToFloatROD(1.0, fp_state_.get());
  EXPECT_EQ(absl::bit_cast<uint32_t>(exact_res), 0x3f800000U);

  // Null fp_state falls back to standard static_cast.
  float null_res = DoubleToFloatROD(inexact_d, nullptr);
  EXPECT_EQ(null_res, static_cast<float>(inexact_d));
}

TEST_F(RiscVZvtInstructionsTest,
       MsetmtypeiZeroMtwidenResetsMtypeAndDirtiesVectorState) {
  // Set VS = Initial (1) and clear SD.
  SetMstatusVs(kMstatusStateInitial);
  state_->mstatus()->Set(state_->mstatus()->GetUint64() & ~kMstatusSdMaskRv32);

  ResetInstruction();
  // mtwiden = 0 (unconfigured), vsew = 16 bits, altfmt = 0 (standard).
  instruction_->AppendSource(
      new ImmediateOperand<uint32_t>(kMtwidenUnconfigured));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(kVsew16));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(kAltfmtStandard));
  instruction_->set_semantic_function(&RiscVZvtMsetMtypei);

  instruction_->Execute(nullptr);

  // mtwiden must be reset to 0.
  EXPECT_EQ(matrix_state_->mtwiden(), 0);
  // SetVectorStateDirty must be called: VS becomes Dirty (3) and SD becomes 1.
  EXPECT_EQ(GetMstatusVs(), kMstatusStateDirty);
  EXPECT_EQ(GetMstatusSd(/*is_rv64=*/false), kMstatusSdSet);
  // UpdateVectorMatrixGeometry must NOT be called: no exception, vill=0.
  EXPECT_FALSE(state_->rv_vector()->vector_exception());
  EXPECT_EQ(state_->rv_vector()->vtype() & kVillMask, 0U);
}

TEST_F(RiscVZvtInstructionsTest,
       MsetmtypeZeroMtwidenResetsMtypeAndDirtiesVectorState) {
  // Set VS = Initial (1) and clear SD.
  SetMstatusVs(kMstatusStateInitial);
  state_->mstatus()->Set(state_->mstatus()->GetUint64() & ~kMstatusSdMaskRv32);

  ResetInstruction();
  // rs1: mtwiden = 0 (unconfigured), rs2: vsew = 16 bits, vill = 0.
  instruction_->AppendSource(
      new ImmediateOperand<uint32_t>(EncodeMtype(kMtwidenUnconfigured)));
  instruction_->AppendSource(
      new ImmediateOperand<uint32_t>(EncodeVtype(kVsew16, kVlmul1)));
  instruction_->set_semantic_function(&RiscVZvtMsetMtype);

  instruction_->Execute(nullptr);

  // mtwiden must be reset to 0.
  EXPECT_EQ(matrix_state_->mtwiden(), 0);
  // SetVectorStateDirty must be called: VS becomes Dirty (3) and SD becomes 1.
  EXPECT_EQ(GetMstatusVs(), kMstatusStateDirty);
  EXPECT_EQ(GetMstatusSd(/*is_rv64=*/false), kMstatusSdSet);
  // UpdateVectorMatrixGeometry must NOT be called when mtwiden is zero:
  // no exception, vill=0.
  EXPECT_FALSE(state_->rv_vector()->vector_exception());
  EXPECT_EQ(state_->rv_vector()->vtype() & kVillMask, 0U);
  EXPECT_EQ(state_->rv_vector()->selected_element_width(), 2);
}

TEST_F(RiscVZvtInstructionsTest, MsetTnTrapsWhenMatrixStateMissing) {
  ASSERT_OK_AND_ASSIGN(uint64_t trap_code,
                       ExecuteWithoutMatrixStateExpectTrap(&RiscVZvtMsetTn));
  EXPECT_EQ(trap_code, *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, MsetTmTrapsWhenMatrixStateMissing) {
  ASSERT_OK_AND_ASSIGN(uint64_t trap_code,
                       ExecuteWithoutMatrixStateExpectTrap(&RiscVZvtMsetTm));
  EXPECT_EQ(trap_code, *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, MsetTkTrapsWhenMatrixStateMissing) {
  ASSERT_OK_AND_ASSIGN(uint64_t trap_code,
                       ExecuteWithoutMatrixStateExpectTrap(&RiscVZvtMsetTk));
  EXPECT_EQ(trap_code, *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, MsetMtypeTrapsWhenMatrixStateMissing) {
  ASSERT_OK_AND_ASSIGN(uint64_t trap_code,
                       ExecuteWithoutMatrixStateExpectTrap(&RiscVZvtMsetMtype));
  EXPECT_EQ(trap_code, *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, MsetMtypeiTrapsWhenMatrixStateMissing) {
  ASSERT_OK_AND_ASSIGN(uint64_t trap_code, ExecuteWithoutMatrixStateExpectTrap(
                                               &RiscVZvtMsetMtypei));
  EXPECT_EQ(trap_code, *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, MsetTnClampsToEffectiveTileEdgeForTew64) {
  // Configure matrix_state with SEW = 32 and twiden = 2 (tew = 64).
  // ETE for tew = 64 is tile_elements_ / 2 = 32 / 2 = 16.
  // If mutated to (sew_bits * twiden) - 1 = 63, ETE becomes 32.
  matrix_state_->set_mtype(EncodeMtype(kMtwiden2));
  // SEW = 32 bits, LMUL = 1 -> max_vl = 32.
  vector_state_->SetVectorType(EncodeVtype(kVsew32, kVlmul1));
  ASSERT_EQ(vector_state_->max_vector_length(), 32);

  SetRegisterValue("x1", 32);

  ResetInstruction();
  instruction_->AppendSource(CreateScalarSource("x1"));
  instruction_->AppendDestination(CreateScalarDest("x1"));
  instruction_->set_semantic_function(&RiscVZvtMsetTn);

  instruction_->Execute(nullptr);

  // Must clamp to effective_tile_edge(64) = 16.
  EXPECT_EQ(GetRegisterValue("x1"), 16U);
  EXPECT_EQ(vector_state_->vector_length(), 16U);
}

TEST_F(RiscVZvtInstructionsTest,
       VtmvVtTransfersColumnTileSliceToVectorRegister) {
  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));
  matrix_state_->set_tile_m(4);
  vector_state_->set_vector_length(4);

  // Ensure MS is on (initial).
  SetMstatusMs(kMstatusStateInitial);

  // Populate asymmetric matrix data in tile 0:
  // element at (r, c) = (r << 4) | c.
  ASSERT_OK(FillTilePattern<uint8_t>(
      /*tile=*/0, /*rows=*/4, /*cols=*/4,
      [](int r, int c) { return static_cast<uint8_t>((r << 4) | c); }));

  // TSS for column 1 of tile 0: pattern = column, tile = 0, index = 1.
  SetRegisterValue("x1", EncodeTss(/*tile=*/0, kTssPatternCol, /*index=*/1));

  ResetInstruction();
  instruction_->AppendSource(CreateScalarSource("x1"));
  instruction_->AppendDestination(CreateVectorDestOperand("v2"));
  instruction_->set_semantic_function(&RiscVZvtVtmvVt);

  instruction_->Execute(nullptr);

  // Vector register v2 must receive column 1 elements:
  // row 0: 0x01, row 1: 0x11, row 2: 0x21, row 3: 0x31.
  // If mutated to (false), row 1 is loaded instead: 0x10, 0x11, 0x12, 0x13.
  auto* v2 = state_->GetRegister<RVVectorRegister>("v2").first;
  auto span = v2->data_buffer()->Get<uint8_t>();
  EXPECT_THAT(span.subspan(0, 4), testing::ElementsAre(0x01, 0x11, 0x21, 0x31));
  EXPECT_EQ(GetMstatusVs(), kMstatusStateDirty);
}

TEST_F(RiscVZvtInstructionsTest, MsetTmClampsToEffectiveTileEdgeForTew64) {
  // Configure matrix_state with SEW = 32 and twiden = 2 (tew = 64).
  // ETE for tew = 64 is tile_elements_ / 2 = 32 / 2 = 16.
  // If mutated to (sew_bits * twiden) - 1 = 63, ETE becomes 32.
  matrix_state_->set_mtype(EncodeMtype(kMtwiden2));
  // SEW = 32 bits, LMUL = 1 -> max_vl = 32.
  vector_state_->SetVectorType(EncodeVtype(kVsew32, kVlmul1));
  ASSERT_EQ(vector_state_->max_vector_length(), 32);

  SetRegisterValue("x1", 32);

  ResetInstruction();
  instruction_->AppendSource(CreateScalarSource("x1"));
  instruction_->AppendDestination(CreateScalarDest("x1"));
  instruction_->set_semantic_function(&RiscVZvtMsetTm);

  instruction_->Execute(nullptr);

  // Must clamp to effective_tile_edge(64) = 16.
  EXPECT_EQ(GetRegisterValue("x1"), 16U);
  EXPECT_EQ(matrix_state_->tile_m(), 16U);
}

TEST_F(RiscVZvtInstructionsTest,
       VtmvTvTransfersVectorRegisterToColumnTileSlice) {
  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));
  matrix_state_->set_tile_m(4);
  vector_state_->set_vector_length(4);

  // Ensure MS is on (initial).
  SetMstatusMs(kMstatusStateInitial);

  // Set up source vector register v4 with 4 bytes: 0xA1, 0xB2, 0xC3, 0xD4.
  auto* v4 = state_->GetRegister<RVVectorRegister>("v4").first;
  auto span = v4->data_buffer()->Get<uint8_t>();
  span[0] = 0xA1;
  span[1] = 0xB2;
  span[2] = 0xC3;
  span[3] = 0xD4;

  // TSS for column 1 of tile 0: pattern = column, tile = 0, index = 1.
  ResetInstruction();
  instruction_->AppendSource(CreateVectorSourceOperand("v4"));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(
      EncodeTss(/*tile=*/0, kTssPatternCol, /*index=*/1)));
  instruction_->set_semantic_function(&RiscVZvtVtmvTv);

  instruction_->Execute(nullptr);

  // Column 1 must receive the elements: (r, 1) for r = 0..3.
  ASSERT_OK_AND_ASSIGN(
      auto col_elements,
      GetTileColumn<uint8_t>(/*tile=*/0, /*col=*/1, /*count=*/4));
  EXPECT_THAT(col_elements, testing::ElementsAre(0xA1, 0xB2, 0xC3, 0xD4));

  // If mutated to if (false), StoreTileBits stores into row 1 instead.
  // Check that row 1 was not written: (1, 0) should remain 0.
  ASSERT_OK_AND_ASSIGN(auto elem_1_0, matrix_state_->GetElem<uint8_t>(
                                          /*tile=*/0, /*row=*/1, /*col=*/0,
                                          /*element_width_in_bits=*/8));
  EXPECT_EQ(elem_1_0, 0x00);
  // MS must be dirty.
  EXPECT_EQ(GetMstatusMs(), kMstatusStateDirty);
}

TEST_F(RiscVZvtInstructionsTest,
       MsetmtypeSetsVectorExceptionWhenTewExceedsElen) {
  // SEW = 32 (kVsew32), twiden = 2 (kMtwiden2) -> tew = 64 > kElen (32).
  // vill triggers: sets vector exception, vill in vtype, vl = 0, mtype = 0.
  // Set vl and mtype non-zero to verify set_vector_length(0) and set_mtype(0).
  state_->rv_vector()->set_vector_length(16);
  matrix_state_->set_mtype(123);

  instruction_->AppendSource(new ImmediateOperand<uint32_t>(kMtwiden2));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(kVsew32));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(kAltfmtStandard));

  instruction_->set_semantic_function(&RiscVZvtMsetMtypei);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(state_->rv_vector()->vector_exception());
  EXPECT_EQ(state_->rv_vector()->vector_length(), 0);
  EXPECT_EQ(matrix_state_->mtype(), 0);
}

TEST_F(RiscVZvtInstructionsTest, MsetTmTrapsWhenMtwidenIsZero) {
  SetRegisterValue("x1", 500);
  SetRegisterValue("x2", 99);
  instruction_->AppendSource(CreateScalarSource("x1"));
  instruction_->AppendDestination(CreateScalarDest("x2"));
  instruction_->set_semantic_function(&RiscVZvtMsetTm);

  // mtwiden = 0 with TM = 24 in mtype.
  matrix_state_->set_mtype(
      EncodeMtype(kMtwidenUnconfigured, /*tk=*/0, /*tm=*/24));
  ASSERT_EQ(matrix_state_->tile_m(), 24);

  SetMstatusVs(kMstatusStateOff);
  ASSERT_EQ(GetMstatusVs(), kMstatusStateOff);

  ScopedTrapCapture trap_capture(state_.get());
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
  EXPECT_EQ(matrix_state_->tile_m(), 24);
  EXPECT_EQ(GetRegisterValue("x2"), 99);
  EXPECT_EQ(GetMstatusVs(), kMstatusStateOff);
}

TEST_F(RiscVZvtInstructionsTest, MsetTkTrapsWhenMtwidenIsZero) {
  SetRegisterValue("x1", 500);
  SetRegisterValue("x2", 77);
  instruction_->AppendSource(CreateScalarSource("x1"));
  instruction_->AppendDestination(CreateScalarDest("x2"));
  instruction_->set_semantic_function(&RiscVZvtMsetTk);

  // mtwiden = 0 with TK = 4 in mtype.
  matrix_state_->set_mtype(
      EncodeMtype(kMtwidenUnconfigured, /*tk=*/4, /*tm=*/0));
  ASSERT_EQ(matrix_state_->tile_k(), 4);

  SetMstatusVs(kMstatusStateOff);
  ASSERT_EQ(GetMstatusVs(), kMstatusStateOff);

  ScopedTrapCapture trap_capture(state_.get());
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
  EXPECT_EQ(matrix_state_->tile_k(), 4);
  EXPECT_EQ(GetRegisterValue("x2"), 77);
  EXPECT_EQ(GetMstatusVs(), kMstatusStateOff);
}

TEST_F(RiscVZvtInstructionsTest, VtmvVtRespectsVectorRegisterBoundary) {
  // Use a localized state with 8-byte vector registers (VLEN = 64 bits).
  // With SEW = 8 bits, elements_per_vector = 8 / 1 = 8.
  FlatDemandMemory memory;
  RiscVState state("boundary_vt_test", RiscVXlen::RV32, &memory);
  RiscVVectorState vec_state(&state, 8);
  state.set_rv_vector(&vec_state);
  vec_state.SetVectorType(EncodeVtype(kVsew8, kVlmul1));
  vec_state.set_vector_length(8);
  vec_state.set_vstart(2);

  ASSERT_OK_AND_ASSIGN(auto matrix_state,
                       RiscVZvtMatrixState::Create(&state, 16));
  matrix_state->set_mtype(EncodeMtype(kMtwiden1));
  matrix_state->set_tile_m(8);

  // Turn on MS in mstatus so instructions don't trap.
  ASSERT_OK_AND_ASSIGN(auto* mstatus_csr, state.csr_set()->GetCsr("mstatus"));
  mstatus_csr->Set(static_cast<uint64_t>(
      mstatus_csr->GetUint64() |
      (static_cast<uint64_t>(kMstatusStateInitial) << kMstatusMsShift)));

  // Populate row 0 of tile 0 with 8 distinct bytes: 0x10 .. 0x17.
  absl::Status init_status;
  for (int c = 0; c < 8; ++c) {
    init_status.Update(matrix_state->SetElem<uint8_t>(
        /*tile=*/0, /*row=*/0, c, /*element_width_in_bits=*/8, 0x10 + c));
  }
  ASSERT_OK(init_status);

  auto* v2 = state.GetRegister<RVVectorRegister>("v2").first;
  auto* v3 = state.GetRegister<RVVectorRegister>("v3").first;
  v2->data_buffer()->Get<uint8_t>()[7] = 0xAA;
  v3->data_buffer()->Get<uint8_t>()[0] = 0xBB;

  std::vector<RegisterBase*> vd_group = {v2, v3};
  auto* vd = new RV32VectorDestinationOperand(
      absl::Span<RegisterBase*>(vd_group), 0, "v2");

  auto* inst = new Instruction(0x1000, &state);
  inst->set_size(4);
  // TSS: row 0 of tile 0 -> pattern = row, tile = 0, index = 0.
  inst->AppendSource(new ImmediateOperand<uint32_t>(
      EncodeTss(/*tile=*/0, kTssPatternRow, /*index=*/0)));
  inst->AppendDestination(vd);
  inst->set_semantic_function(&RiscVZvtVtmvVt);

  inst->Execute(nullptr);
  inst->DecRef();

  // Clean code: elements_per_vector = 8, num_regs = 1.
  // Writes row 0 elements [0..7] into v2[0..7]. v2[7] becomes 0x17.
  // v3 is untouched and retains 0xBB.
  // Mutated code (elements_per_vector = 7):
  // num_regs = 2. r = 0 writes elements 0..6 (v2[7] untouched = 0xAA).
  // r = 1 writes element 7 into v3[0] (v3[0] becomes 0x17).
  EXPECT_EQ(v2->data_buffer()->Get<uint8_t>()[7], 0x17);
  EXPECT_EQ(v3->data_buffer()->Get<uint8_t>()[0], 0xBB);
  EXPECT_EQ(vec_state.vstart(), 0);
}

TEST_F(RiscVZvtInstructionsTest, VtmvTvRespectsVectorRegisterBoundary) {
  // Use a localized state with 8-byte vector registers (VLEN = 64 bits).
  // With SEW = 8 bits, elements_per_vector = 8 / 1 = 8.
  FlatDemandMemory memory;
  RiscVState state("boundary_test", RiscVXlen::RV32, &memory);
  RiscVVectorState vec_state(&state, 8);
  state.set_rv_vector(&vec_state);
  vec_state.SetVectorType(EncodeVtype(kVsew8, kVlmul1));
  vec_state.set_vector_length(10);
  vec_state.set_vstart(2);

  ASSERT_OK_AND_ASSIGN(auto matrix_state,
                       RiscVZvtMatrixState::Create(&state, 16));
  matrix_state->set_mtype(EncodeMtype(kMtwiden1));
  matrix_state->set_tile_m(10);

  // Turn on MS in mstatus so instructions don't trap.
  ASSERT_OK_AND_ASSIGN(auto* mstatus_csr, state.csr_set()->GetCsr("mstatus"));
  mstatus_csr->Set(static_cast<uint64_t>(
      mstatus_csr->GetUint64() |
      (static_cast<uint64_t>(kMstatusStateInitial) << kMstatusMsShift)));

  auto* v4 = state.GetRegister<RVVectorRegister>("v4").first;
  auto* v5 = state.GetRegister<RVVectorRegister>("v5").first;
  // Initialize v4[7] with 0x77, v5[0] with 0x88, and v5[1] with 0x99.
  v4->data_buffer()->Get<uint8_t>()[7] = 0x77;
  v5->data_buffer()->Get<uint8_t>()[0] = 0x88;
  v5->data_buffer()->Get<uint8_t>()[1] = 0x99;

  std::vector<RegisterBase*> vs_group = {v4, v5};
  auto* vs =
      new RV32VectorSourceOperand(absl::Span<RegisterBase*>(vs_group), "v4");

  auto* inst = new Instruction(0x1000, &state);
  inst->set_size(4);
  inst->AppendSource(vs);
  // TSS: row 0 of tile 0 -> pattern = row, tile = 0, index = 0.
  inst->AppendSource(new ImmediateOperand<uint32_t>(
      EncodeTss(/*tile=*/0, kTssPatternRow, /*index=*/0)));
  inst->set_semantic_function(&RiscVZvtVtmvTv);

  inst->Execute(nullptr);
  inst->DecRef();

  // For elem = 7:
  // Clean code: reg = 7 / 8 = 0, e = 7 % 8 = 7 -> reads v4[7] = 0x77.
  // Mutated code (- 1): reg = 7 / 7 = 1, e = 7 % 7 = 0 -> reads v5[0] = 0x88.
  ASSERT_OK_AND_ASSIGN(auto elem_7, matrix_state->GetElem<uint8_t>(
                                        /*tile=*/0, /*row=*/0, /*col=*/7,
                                        /*element_width_in_bits=*/8));
  EXPECT_EQ(elem_7, 0x77);

  // For elem = 9:
  // Clean code: reg = 9 / 8 = 1, e = 9 % 8 = 1 -> reads v5[1] = 0x99.
  // Mutated code (+ 1): reg = 9 / 9 = 1, e = 9 % 9 = 0 -> reads v5[0] = 0x88.
  ASSERT_OK_AND_ASSIGN(auto elem_9, matrix_state->GetElem<uint8_t>(
                                        /*tile=*/0, /*row=*/0, /*col=*/9,
                                        /*element_width_in_bits=*/8));
  EXPECT_EQ(elem_9, 0x99);
  EXPECT_EQ(vec_state.vstart(), 0);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmTvvFp8E4M3Matmul) {
  // Zvt spec v0.4: vtfmm.tvv OCP FP8 E4M3 x E4M3 -> FP32.
  ConfigureMatrixUnit(kMtwiden4, kVsew8, kVlmul1, /*vl=*/2, /*tk=*/2,
                      /*tm=*/2);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{1.25f, -0.25f}, {0.0f, 10.0f}}));

  // Populate matrix A (FP8 E4M3, SEW=8 => KMAX=4, stride=2):
  WriteVectorRegister<uint8_t>("v8", {0x40, 0x38});   // 2.0f, 1.0f
  WriteVectorRegister<uint8_t>("v10", {0xB0, 0x48});  // -0.5f, 4.0f

  // Populate matrix B (FP8 E4M3):
  WriteVectorRegister<uint8_t>("v16", {0x30, 0x44});  // 0.5f, 3.0f
  WriteVectorRegister<uint8_t>("v18", {0x40, 0xB8});  // 2.0f, -1.0f

  ExecuteMatmul(&RiscVZvtVtfmm);

  ASSERT_OK_AND_ASSIGN(
      auto result, GetTileRegion<float>(/*tile=*/0, /*rows=*/2, /*cols=*/2));
  EXPECT_THAT(result, testing::ElementsAre(1.25f, 6.25f, 8.5f, 9.0f));

  EXPECT_EQ(state_->mstatus()->ms(), kMstatusStateDirty);
  EXPECT_EQ(vsstatus()->ms(),
            state_->v() ? kMstatusStateDirty : kMstatusStateInitial);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmTvvFp8E4M3x5M2Matmul) {
  // Zvt spec v0.4: vtfmm.tvv OCP FP8 E4M3 x E5M2 -> FP32.
  ConfigureMatrixUnit(kMtwiden4, kVsew8, kVlmul1, /*vl=*/2, /*tk=*/2,
                      /*tm=*/2, /*vstart=*/0, /*vta=*/false, /*vma=*/false,
                      /*altfmt=*/true);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{1.25f, -0.25f}, {0.0f, 10.0f}}));

  WriteVectorRegister<uint8_t>("v8", {0x40, 0x38});   // 2.0f, 1.0f (E4M3)
  WriteVectorRegister<uint8_t>("v10", {0xB0, 0x48});  // -0.5f, 4.0f (E4M3)
  WriteVectorRegister<uint8_t>("v16", {0x38, 0x42});  // 0.5f, 3.0f (E5M2)
  WriteVectorRegister<uint8_t>("v18", {0x40, 0xBC});  // 2.0f, -1.0f (E5M2)

  ExecuteMatmul(&RiscVZvtVtfmm);

  ASSERT_OK_AND_ASSIGN(
      auto result, GetTileRegion<float>(/*tile=*/0, /*rows=*/2, /*cols=*/2));
  EXPECT_THAT(result, testing::ElementsAre(1.25f, 6.25f, 8.5f, 9.0f));
}

TEST_F(RiscVZvtInstructionsTest, VtfmmAltTvvFp8E5M2x4M3Matmul) {
  // Zvt spec v0.4: vtfmm.alt.tvv OCP FP8 E5M2 x E4M3 -> FP32.
  ConfigureMatrixUnit(kMtwiden4, kVsew8, kVlmul1, /*vl=*/2, /*tk=*/2,
                      /*tm=*/2);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{1.25f, -0.25f}, {0.0f, 10.0f}}));

  WriteVectorRegister<uint8_t>("v8", {0x40, 0x3C});   // 2.0f, 1.0f (E5M2)
  WriteVectorRegister<uint8_t>("v10", {0xB8, 0x44});  // -0.5f, 4.0f (E5M2)
  WriteVectorRegister<uint8_t>("v16", {0x30, 0x44});  // 0.5f, 3.0f (E4M3)
  WriteVectorRegister<uint8_t>("v18", {0x40, 0xB8});  // 2.0f, -1.0f (E4M3)

  ExecuteMatmul(&RiscVZvtVtfmmAlt);

  ASSERT_OK_AND_ASSIGN(
      auto result, GetTileRegion<float>(/*tile=*/0, /*rows=*/2, /*cols=*/2));
  EXPECT_THAT(result, testing::ElementsAre(1.25f, 6.25f, 8.5f, 9.0f));
}

TEST_F(RiscVZvtInstructionsTest, VtfmmAltTvvFp8E5M2Matmul) {
  // Zvt spec v0.4: vtfmm.alt.tvv OCP FP8 E5M2 x E5M2 -> FP32.
  ConfigureMatrixUnit(kMtwiden4, kVsew8, kVlmul1, /*vl=*/2, /*tk=*/2,
                      /*tm=*/2, /*vstart=*/0, /*vta=*/false, /*vma=*/false,
                      /*altfmt=*/true);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{1.25f, -0.25f}, {0.0f, 10.0f}}));

  WriteVectorRegister<uint8_t>("v8", {0x40, 0x3C});   // 2.0f, 1.0f (E5M2)
  WriteVectorRegister<uint8_t>("v10", {0xB8, 0x44});  // -0.5f, 4.0f (E5M2)
  WriteVectorRegister<uint8_t>("v16", {0x38, 0x42});  // 0.5f, 3.0f (E5M2)
  WriteVectorRegister<uint8_t>("v18", {0x40, 0xBC});  // 2.0f, -1.0f (E5M2)

  ExecuteMatmul(&RiscVZvtVtfmmAlt);

  ASSERT_OK_AND_ASSIGN(
      auto result, GetTileRegion<float>(/*tile=*/0, /*rows=*/2, /*cols=*/2));
  EXPECT_THAT(result, testing::ElementsAre(1.25f, 6.25f, 8.5f, 9.0f));
}

TEST_F(RiscVZvtInstructionsTest, VtmmuSewTrap) {
  // SEW != 8 is invalid for vtmmu.
  ConfigureMatrixUnit(kMtwiden4, kVsew16, kVlmul1, /*vl=*/2, /*tk=*/1,
                      /*tm=*/2);
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteMatmul(&RiscVZvtVtmmu);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtmmsSewTrap) {
  // SEW != 8 is invalid for vtmms.
  ConfigureMatrixUnit(kMtwiden4, kVsew32, kVlmul1, /*vl=*/2, /*tk=*/1,
                      /*tm=*/2);
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteMatmul(&RiscVZvtVtmms);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtmmuTwidenTrap) {
  // TWIDEN != 4 is invalid for vtmmu.
  ConfigureMatrixUnit(kMtwiden2, kVsew8, kVlmul1, /*vl=*/2, /*tk=*/1,
                      /*tm=*/2);
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteMatmul(&RiscVZvtVtmmu);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtmmsTwidenTrap) {
  // TWIDEN != 4 is invalid for vtmms.
  ConfigureMatrixUnit(kMtwiden2, kVsew8, kVlmul1, /*vl=*/2, /*tk=*/1,
                      /*tm=*/2);
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteMatmul(&RiscVZvtVtmms);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtmmuUnconfiguredTrap) {
  ConfigureMatrixUnit(kMtwidenUnconfigured, kVsew8, kVlmul1, /*vl=*/2,
                      /*tk=*/1, /*tm=*/2);
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteMatmul(&RiscVZvtVtmmu);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtmmsUnconfiguredTrap) {
  ConfigureMatrixUnit(kMtwidenUnconfigured, kVsew8, kVlmul1, /*vl=*/0);
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteMatmul(&RiscVZvtVtmms);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmUnconfiguredTrap) {
  ConfigureMatrixUnit(kMtwidenUnconfigured, kVsew8, kVlmul1, /*vl=*/0);
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteMatmul(&RiscVZvtVtfmm);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmAltUnconfiguredTrap) {
  ConfigureMatrixUnit(kMtwidenUnconfigured, kVsew8, kVlmul1, /*vl=*/0);
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteMatmul(&RiscVZvtVtfmmAlt);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, MatmulFcsrFlagsException) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/2, /*tk=*/1,
                      /*tm=*/2);
  state_->set_v(/*v=*/false);
  vsstatus()->set_vs(kMstatusStateInitial);

  // Preserve bits: DZ, UF, NX. Set them to 1.
  const uint32_t preset_fcsr = *FPExceptions::kDivByZero |
                               *FPExceptions::kUnderflow |
                               *FPExceptions::kInexact;
  fp_state_->fcsr()->Write(preset_fcsr);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{0.0f}}));
  WriteVectorRegister<uint32_t>(
      "v8",
      {absl::bit_cast<uint32_t>(std::numeric_limits<float>::infinity()), 0});
  WriteVectorRegister<uint32_t>("v16", {absl::bit_cast<uint32_t>(0.0f), 0});

  ScopedTrapCapture trap_capture(state_.get());
  ExecuteMatmul(&RiscVZvtVtfmm);
  EXPECT_FALSE(trap_capture.trap_taken());

  uint32_t fcsr_after = fp_state_->fcsr()->AsUint32();
  EXPECT_NE(fcsr_after & *FPExceptions::kInvalidOp, 0U);
  EXPECT_EQ(fcsr_after & preset_fcsr, preset_fcsr);

  EXPECT_EQ(state_->mstatus()->fs(), kMstatusStateDirty);
  EXPECT_EQ(vsstatus()->fs(), kMstatusStateDirty);
}

TEST_F(RiscVZvtInstructionsTest, MatmulFcsrFlagsOverflowException) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/2, /*tk=*/1,
                      /*tm=*/2);
  state_->set_v(/*v=*/false);
  vsstatus()->set_vs(kMstatusStateInitial);

  // Preset FCSR to 0 to verify isolation of NX/UF/DZ.
  fp_state_->fcsr()->Write(0U);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{0.0f}}));
  const uint32_t max_bits =
      absl::bit_cast<uint32_t>(std::numeric_limits<float>::max());
  WriteVectorRegister<uint32_t>("v8", {max_bits, 0});
  WriteVectorRegister<uint32_t>("v16", {max_bits, 0});

  ScopedTrapCapture trap_capture(state_.get());
  ExecuteMatmul(&RiscVZvtVtfmm);
  EXPECT_FALSE(trap_capture.trap_taken());

  uint32_t fcsr_after = fp_state_->fcsr()->AsUint32();
  EXPECT_NE(fcsr_after & *FPExceptions::kOverflow, 0U);
  const uint32_t preserved_mask = *FPExceptions::kDivByZero |
                                  *FPExceptions::kUnderflow |
                                  *FPExceptions::kInexact;
  EXPECT_EQ(fcsr_after & preserved_mask, 0U);

  EXPECT_EQ(state_->mstatus()->fs(), kMstatusStateDirty);
  EXPECT_EQ(vsstatus()->fs(), kMstatusStateDirty);
}

TEST_F(RiscVZvtInstructionsTest, VtleTailElementEvaluationPolicy) {
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/2, /*tk=*/0, /*tm=*/0,
                      /*vstart=*/0, /*vta=*/true);

  constexpr uint64_t kBaseAddr = 0x2000;
  WriteMemory<uint8_t>(kBaseAddr, {0x11, 0x22});

  constexpr uint32_t kTileId = 1;
  constexpr uint32_t kRowIndex = 2;
  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/8, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternRow, kRowIndex));

  // ETE for eew=8 is 32; cols 0..1 are loaded and tail 2..31 are filled with
  // 0xFF when vta=1.
  ASSERT_OK_AND_ASSIGN(auto row_vals,
                       GetTileRow<uint8_t>(kTileId, kRowIndex, /*count=*/32));
  std::vector<uint8_t> expected(32, 0xFF);
  expected[0] = 0x11;
  expected[1] = 0x22;
  EXPECT_THAT(row_vals, testing::ElementsAreArray(expected));
}

TEST_F(RiscVZvtInstructionsTest, VtseTailElementEvaluationPolicy) {
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/2, /*tk=*/0, /*tm=*/0,
                      /*vstart=*/0, /*vta=*/true);

  constexpr uint64_t kBaseAddr = 0x2000;
  WriteMemoryConstant<uint8_t>(kBaseAddr, /*count=*/32, 0x55);

  constexpr uint32_t kTileId = 1;
  constexpr uint32_t kRowIndex = 2;
  ASSERT_OK(SetTileRowPattern<uint8_t>(kTileId, kRowIndex, /*count=*/4,
                                       [](int i) { return i + 1; }));

  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/8, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternRow, kRowIndex));

  // Only elements 0 and 1 are stored; elements 2..31 remain undisturbed (0x55)
  // regardless of vta=1.
  std::vector<uint8_t> expected(32, 0x55);
  expected[0] = 1;
  expected[1] = 2;
  EXPECT_THAT(ReadMemory<uint8_t>(kBaseAddr, 32),
              testing::ElementsAreArray(expected));
}

// --- Vtle TSS Exception Tests ---
TEST_F(RiscVZvtInstructionsTest, VtleTssPatternExceedsBoundsTrap) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/32, /*base_addr=*/0x1000,
                      EncodeTss(/*tile=*/1, kTssPatternInvalid, /*index=*/0));

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtleTssRowIndexOverflowTrap) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/32, /*base_addr=*/0x1000,
                      EncodeTss(/*tile=*/1, kTssPatternRow, /*index=*/32));

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtleTssColIndexOverflowTrap) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/32, /*base_addr=*/0x1000,
                      EncodeTss(/*tile=*/1, kTssPatternCol, /*index=*/32));

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

// --- Vtse TSS Exception Tests ---
TEST_F(RiscVZvtInstructionsTest, VtseTssPatternExceedsBoundsTrap) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/32, /*base_addr=*/0x1000,
                      EncodeTss(/*tile=*/1, kTssPatternInvalid, /*index=*/0));

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtseTssRowIndexOverflowTrap) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/32, /*base_addr=*/0x1000,
                      EncodeTss(/*tile=*/1, kTssPatternRow, /*index=*/32));

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtseTssColIndexOverflowTrap) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/32, /*base_addr=*/0x1000,
                      EncodeTss(/*tile=*/1, kTssPatternCol, /*index=*/32));

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, Vtle8AddressMisalignedExceptions) {
  ConfigureMatrixUnit(kMtwiden1, kVsew64, kVlmul2, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  constexpr uint64_t kUnalignedBase = 0x2000 + 1;
  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/8, kUnalignedBase,
                      EncodeTss(/*tile=*/1, kTssPatternRow, /*index=*/0));

  EXPECT_FALSE(trap_capture.trap_taken());
}

TEST_F(RiscVZvtInstructionsTest, Vtle16AddressMisalignedExceptions) {
  ConfigureMatrixUnit(kMtwiden1, kVsew64, kVlmul2, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  constexpr uint64_t kUnalignedBase = 0x2000 + 1;
  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/16, kUnalignedBase,
                      EncodeTss(/*tile=*/1, kTssPatternRow, /*index=*/0));

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kLoadAddressMisaligned);
}

TEST_F(RiscVZvtInstructionsTest, Vtle32AddressMisalignedExceptions) {
  ConfigureMatrixUnit(kMtwiden1, kVsew64, kVlmul2, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  constexpr uint64_t kUnalignedBase = 0x2000 + 1;
  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/32, kUnalignedBase,
                      EncodeTss(/*tile=*/1, kTssPatternRow, /*index=*/0));

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kLoadAddressMisaligned);
}

TEST_F(RiscVZvtInstructionsTest, Vtle64AddressMisalignedExceptions) {
  ConfigureMatrixUnit(kMtwiden1, kVsew64, kVlmul2, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  constexpr uint64_t kUnalignedBase = 0x2000 + 1;
  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/64, kUnalignedBase,
                      EncodeTss(/*tile=*/1, kTssPatternRow, /*index=*/0));

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kLoadAddressMisaligned);
}

TEST_F(RiscVZvtInstructionsTest, Vtse8AddressMisalignedExceptions) {
  ConfigureMatrixUnit(kMtwiden1, kVsew64, kVlmul2, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  constexpr uint64_t kUnalignedBase = 0x2000 + 1;
  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/8, kUnalignedBase,
                      EncodeTss(/*tile=*/1, kTssPatternRow, /*index=*/0));

  EXPECT_FALSE(trap_capture.trap_taken());
}

TEST_F(RiscVZvtInstructionsTest, Vtse16AddressMisalignedExceptions) {
  ConfigureMatrixUnit(kMtwiden1, kVsew64, kVlmul2, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  constexpr uint64_t kUnalignedBase = 0x2000 + 1;
  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/16, kUnalignedBase,
                      EncodeTss(/*tile=*/1, kTssPatternRow, /*index=*/0));

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kStoreAddressMisaligned);
}

TEST_F(RiscVZvtInstructionsTest, Vtse32AddressMisalignedExceptions) {
  ConfigureMatrixUnit(kMtwiden1, kVsew64, kVlmul2, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  constexpr uint64_t kUnalignedBase = 0x2000 + 1;
  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/32, kUnalignedBase,
                      EncodeTss(/*tile=*/1, kTssPatternRow, /*index=*/0));

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kStoreAddressMisaligned);
}

TEST_F(RiscVZvtInstructionsTest, Vtse64AddressMisalignedExceptions) {
  ConfigureMatrixUnit(kMtwiden1, kVsew64, kVlmul2, /*vl=*/4);
  ScopedTrapCapture trap_capture(state_.get());

  constexpr uint64_t kUnalignedBase = 0x2000 + 1;
  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/64, kUnalignedBase,
                      EncodeTss(/*tile=*/1, kTssPatternRow, /*index=*/0));

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kStoreAddressMisaligned);
}

TEST_F(RiscVZvtInstructionsTest, VtseVstartLowerBoundLimitation) {
  ConfigureMatrixUnit(kMtwiden1, kVsew16, kVlmul2, /*vl=*/4, /*tk=*/0, /*tm=*/0,
                      /*vstart=*/2);

  constexpr uint64_t kBaseAddr = 0x2000;
  // Pre-fill memory to verify elements < vstart are unchanged.
  WriteMemoryConstant<uint16_t>(kBaseAddr, /*count=*/4, 0xBBBB);

  constexpr uint32_t kTileId = 0;
  constexpr uint32_t kRowIndex = 2;
  ASSERT_OK(SetTileRowPattern<uint16_t>(
      kTileId, kRowIndex, /*count=*/4,
      [](int i) { return static_cast<uint16_t>(0xAA00 + i); }));

  ScopedTrapCapture trap_capture(state_.get());
  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/16, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternRow, kRowIndex));
  EXPECT_FALSE(trap_capture.trap_taken());

  EXPECT_THAT(ReadMemory<uint16_t>(kBaseAddr, 4),
              testing::ElementsAre(0xBBBB, 0xBBBB, 0xAA02, 0xAA03));
  EXPECT_EQ(vector_state_->vstart(), 0);
}

TEST_F(RiscVZvtInstructionsTest, VtleAscendingElementOrderTraversalLimitation) {
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/4);

  constexpr uint64_t kBaseAddr = 0x2000;
  WriteMemory<uint8_t>(kBaseAddr, {0xAA, 0xBB, 0xCC, 0xDD});

  constexpr uint32_t kTileId = 1;
  constexpr uint32_t kRowIndex = 2;
  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/8, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternRow, kRowIndex));

  ASSERT_OK_AND_ASSIGN(auto row_vals,
                       GetTileRow<uint8_t>(kTileId, kRowIndex, /*count=*/4));
  EXPECT_THAT(row_vals, testing::ElementsAre(0xAA, 0xBB, 0xCC, 0xDD));
}

TEST_F(RiscVZvtInstructionsTest, VtseAscendingElementOrderTraversalLimitation) {
  ConfigureMatrixUnit(kMtwiden1, kVsew8, kVlmul1, /*vl=*/4);

  constexpr uint32_t kTileId = 1;
  constexpr uint32_t kRowIndex = 2;
  ASSERT_OK(SetTileRow<uint8_t>(kTileId, kRowIndex, {0x11, 0x22, 0x33, 0x44}));

  constexpr uint64_t kBaseAddr = 0x2000;
  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/8, kBaseAddr,
                      EncodeTss(kTileId, kTssPatternRow, kRowIndex));

  EXPECT_THAT(ReadMemory<uint8_t>(kBaseAddr, 4),
              testing::ElementsAre(0x11, 0x22, 0x33, 0x44));
}

TEST_F(RiscVZvtInstructionsTest, DoubleToFloatRODPrecision) {
  // Normal value
  EXPECT_EQ(DoubleToFloatROD(1.0, fp_state_.get()), 1.0f);

  // Tie breaking for ROD (Round to Odd). E.g. Exact halfway between two floats.
  // We can just test generic correctness for Inf and NaN for now.
  EXPECT_TRUE(std::isinf(DoubleToFloatROD(
      std::numeric_limits<double>::infinity(), fp_state_.get())));
  EXPECT_TRUE(std::isnan(DoubleToFloatROD(
      std::numeric_limits<double>::quiet_NaN(), fp_state_.get())));
  EXPECT_EQ(DoubleToFloatROD(0.0, fp_state_.get()), 0.0f);
  EXPECT_EQ(DoubleToFloatROD(-0.0, fp_state_.get()), -0.0f);
}

TEST_F(RiscVZvtInstructionsTest, MatmulFcsrNxExceptionInvarianceTrace) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/2, /*tk=*/1,
                      /*tm=*/2);
  fp_state_->fcsr()->Write(0U);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{0.0f}}));
  WriteVectorRegister<uint32_t>("v8",
                                {absl::bit_cast<uint32_t>(1.0000001f), 0});
  WriteVectorRegister<uint32_t>("v16",
                                {absl::bit_cast<uint32_t>(1.0000001f), 0});

  ScopedTrapCapture trap_capture(state_.get());
  ExecuteMatmul(&RiscVZvtVtfmm);
  EXPECT_FALSE(trap_capture.trap_taken());

  ASSERT_OK_AND_ASSIGN(uint32_t res_val,
                       GetTileElem<uint32_t>(/*tile=*/0, /*row=*/0, /*col=*/0));
  EXPECT_NE(res_val, 0U);
  EXPECT_EQ(fp_state_->fcsr()->AsUint32() & *FPExceptions::kInexact, 0U);
}

TEST_F(RiscVZvtInstructionsTest, MatmulFcsrNxExceptionInvarianceTraceFp8) {
  ConfigureMatrixUnit(kMtwiden4, kVsew8, kVlmul1, /*vl=*/2, /*tk=*/1,
                      /*tm=*/2);
  fp_state_->fcsr()->Write(0U);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{0.0f}}));
  WriteVectorRegister<uint8_t>("v8", {0x40, 0});
  WriteVectorRegister<uint8_t>("v16", {0x40, 0});

  ScopedTrapCapture trap_capture(state_.get());
  ExecuteMatmul(&RiscVZvtVtfmm);
  EXPECT_FALSE(trap_capture.trap_taken());

  ASSERT_OK_AND_ASSIGN(uint32_t res_val,
                       GetTileElem<uint32_t>(/*tile=*/0, /*row=*/0, /*col=*/0));
  EXPECT_NE(res_val, 0U);
  EXPECT_EQ(fp_state_->fcsr()->AsUint32() & *FPExceptions::kInexact, 0U);
}

TEST_F(RiscVZvtInstructionsTest, MatmulFcsrNxExceptionInvarianceTraceFp16) {
  ConfigureMatrixUnit(kMtwiden2, kVsew16, kVlmul1, /*vl=*/2, /*tk=*/1,
                      /*tm=*/2);
  fp_state_->fcsr()->Write(0U);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{0.0f}}));
  WriteVectorRegister<uint16_t>("v8", {0x3C00, 0});
  WriteVectorRegister<uint16_t>("v16", {0x3C00, 0});

  ScopedTrapCapture trap_capture(state_.get());
  ExecuteMatmul(&RiscVZvtVtfmm);
  EXPECT_FALSE(trap_capture.trap_taken());

  ASSERT_OK_AND_ASSIGN(uint32_t res_val,
                       GetTileElem<uint32_t>(/*tile=*/0, /*row=*/0, /*col=*/0));
  EXPECT_NE(res_val, 0U);
  EXPECT_EQ(fp_state_->fcsr()->AsUint32() & *FPExceptions::kInexact, 0U);
}

TEST_F(RiscVZvtInstructionsTest, MatmulFcsrNxExceptionInvarianceTraceBf16) {
  ConfigureMatrixUnit(kMtwiden2, kVsew16, kVlmul1, /*vl=*/2, /*tk=*/1,
                      /*tm=*/2, /*vstart=*/0, /*vta=*/false, /*vma=*/false,
                      /*altfmt=*/true);
  fp_state_->fcsr()->Write(0U);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{0.0f}}));
  WriteVectorRegister<uint16_t>("v8", {0x3C00, 0});
  WriteVectorRegister<uint16_t>("v16", {0x3C00, 0});

  ScopedTrapCapture trap_capture(state_.get());
  ExecuteMatmul(&RiscVZvtVtfmmAlt);
  EXPECT_FALSE(trap_capture.trap_taken());

  ASSERT_OK_AND_ASSIGN(uint32_t res_val,
                       GetTileElem<uint32_t>(/*tile=*/0, /*row=*/0, /*col=*/0));
  EXPECT_NE(res_val, 0U);
  EXPECT_EQ(fp_state_->fcsr()->AsUint32() & *FPExceptions::kInexact, 0U);
}

TEST_F(RiscVZvtInstructionsTest, MatmulFcsrUfExceptionInvarianceTrace) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/2, /*tk=*/1,
                      /*tm=*/2);
  fp_state_->fcsr()->Write(0U);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{0.0f}}));
  WriteVectorRegister<uint32_t>(
      "v8", {absl::bit_cast<uint32_t>(std::numeric_limits<float>::min()), 0});
  WriteVectorRegister<uint32_t>("v16", {absl::bit_cast<uint32_t>(0.5f), 0});

  ScopedTrapCapture trap_capture(state_.get());
  ExecuteMatmul(&RiscVZvtVtfmm);
  EXPECT_FALSE(trap_capture.trap_taken());

  ASSERT_OK_AND_ASSIGN(uint32_t res_val,
                       GetTileElem<uint32_t>(/*tile=*/0, /*row=*/0, /*col=*/0));
  EXPECT_NE(res_val, 0U);
  EXPECT_EQ(fp_state_->fcsr()->AsUint32() & *FPExceptions::kUnderflow, 0U);
}

TEST_F(RiscVZvtInstructionsTest, MatmulFcsrUfExceptionInvarianceTraceFp8) {
  ConfigureMatrixUnit(kMtwiden4, kVsew8, kVlmul1, /*vl=*/2, /*tk=*/1,
                      /*tm=*/2);
  fp_state_->fcsr()->Write(0U);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{0.0f}}));
  WriteVectorRegister<uint8_t>("v8", {0x40, 0});
  WriteVectorRegister<uint8_t>("v16", {0x40, 0});

  ScopedTrapCapture trap_capture(state_.get());
  ExecuteMatmul(&RiscVZvtVtfmm);
  EXPECT_FALSE(trap_capture.trap_taken());

  ASSERT_OK_AND_ASSIGN(uint32_t res_val,
                       GetTileElem<uint32_t>(/*tile=*/0, /*row=*/0, /*col=*/0));
  EXPECT_NE(res_val, 0U);
  EXPECT_EQ(fp_state_->fcsr()->AsUint32() & *FPExceptions::kUnderflow, 0U);
}

TEST_F(RiscVZvtInstructionsTest, MatmulFcsrUfExceptionInvarianceTraceFp16) {
  ConfigureMatrixUnit(kMtwiden2, kVsew16, kVlmul1, /*vl=*/2, /*tk=*/1,
                      /*tm=*/2);
  fp_state_->fcsr()->Write(0U);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{0.0f}}));
  WriteVectorRegister<uint16_t>("v8", {0x3C00, 0});
  WriteVectorRegister<uint16_t>("v16", {0x3C00, 0});

  ScopedTrapCapture trap_capture(state_.get());
  ExecuteMatmul(&RiscVZvtVtfmm);
  EXPECT_FALSE(trap_capture.trap_taken());

  ASSERT_OK_AND_ASSIGN(uint32_t res_val,
                       GetTileElem<uint32_t>(/*tile=*/0, /*row=*/0, /*col=*/0));
  EXPECT_NE(res_val, 0U);
  EXPECT_EQ(fp_state_->fcsr()->AsUint32() & *FPExceptions::kUnderflow, 0U);
}

TEST_F(RiscVZvtInstructionsTest, MatmulFcsrUfExceptionInvarianceTraceBf16) {
  ConfigureMatrixUnit(kMtwiden2, kVsew16, kVlmul1, /*vl=*/2, /*tk=*/1,
                      /*tm=*/2, /*vstart=*/0, /*vta=*/false, /*vma=*/false,
                      /*altfmt=*/true);
  fp_state_->fcsr()->Write(0U);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{0.0f}}));
  WriteVectorRegister<uint16_t>("v8", {0x3C00, 0});
  WriteVectorRegister<uint16_t>("v16", {0x3C00, 0});

  ScopedTrapCapture trap_capture(state_.get());
  ExecuteMatmul(&RiscVZvtVtfmmAlt);
  EXPECT_FALSE(trap_capture.trap_taken());

  ASSERT_OK_AND_ASSIGN(uint32_t res_val,
                       GetTileElem<uint32_t>(/*tile=*/0, /*row=*/0, /*col=*/0));
  EXPECT_NE(res_val, 0U);
  EXPECT_EQ(fp_state_->fcsr()->AsUint32() & *FPExceptions::kUnderflow, 0U);
}

TEST_F(RiscVZvtInstructionsTest, MatmulFcsrDzExceptionInvarianceTrace) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/2, /*tk=*/1,
                      /*tm=*/2);
  fp_state_->fcsr()->Write(0U);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{0.0f}}));
  WriteVectorRegister<uint32_t>("v8", {absl::bit_cast<uint32_t>(1.0f), 0});
  WriteVectorRegister<uint32_t>("v16", {absl::bit_cast<uint32_t>(1.0f), 0});

  ScopedTrapCapture trap_capture(state_.get());
  ExecuteMatmul(&RiscVZvtVtfmm);
  EXPECT_FALSE(trap_capture.trap_taken());

  ASSERT_OK_AND_ASSIGN(uint32_t res_val,
                       GetTileElem<uint32_t>(/*tile=*/0, /*row=*/0, /*col=*/0));
  EXPECT_NE(res_val, 0U);
  EXPECT_EQ(fp_state_->fcsr()->AsUint32() & *FPExceptions::kDivByZero, 0U);
}

TEST_F(RiscVZvtInstructionsTest, MatmulFcsrDzExceptionInvarianceTraceFp8) {
  ConfigureMatrixUnit(kMtwiden4, kVsew8, kVlmul1, /*vl=*/2, /*tk=*/1,
                      /*tm=*/2);
  fp_state_->fcsr()->Write(0U);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{0.0f}}));
  WriteVectorRegister<uint8_t>("v8", {0x40, 0});
  WriteVectorRegister<uint8_t>("v16", {0x40, 0});

  ScopedTrapCapture trap_capture(state_.get());
  ExecuteMatmul(&RiscVZvtVtfmm);
  EXPECT_FALSE(trap_capture.trap_taken());

  ASSERT_OK_AND_ASSIGN(uint32_t res_val,
                       GetTileElem<uint32_t>(/*tile=*/0, /*row=*/0, /*col=*/0));
  EXPECT_NE(res_val, 0U);
  EXPECT_EQ(fp_state_->fcsr()->AsUint32() & *FPExceptions::kDivByZero, 0U);
}

TEST_F(RiscVZvtInstructionsTest, MatmulFcsrDzExceptionInvarianceTraceFp16) {
  ConfigureMatrixUnit(kMtwiden2, kVsew16, kVlmul1, /*vl=*/2, /*tk=*/1,
                      /*tm=*/2);
  fp_state_->fcsr()->Write(0U);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{0.0f}}));
  WriteVectorRegister<uint16_t>("v8", {0x3C00, 0});
  WriteVectorRegister<uint16_t>("v16", {0x3C00, 0});

  ScopedTrapCapture trap_capture(state_.get());
  ExecuteMatmul(&RiscVZvtVtfmm);
  EXPECT_FALSE(trap_capture.trap_taken());

  ASSERT_OK_AND_ASSIGN(uint32_t res_val,
                       GetTileElem<uint32_t>(/*tile=*/0, /*row=*/0, /*col=*/0));
  EXPECT_NE(res_val, 0U);
  EXPECT_EQ(fp_state_->fcsr()->AsUint32() & *FPExceptions::kDivByZero, 0U);
}

TEST_F(RiscVZvtInstructionsTest, MatmulFcsrDzExceptionInvarianceTraceBf16) {
  ConfigureMatrixUnit(kMtwiden2, kVsew16, kVlmul1, /*vl=*/2, /*tk=*/1,
                      /*tm=*/2, /*vstart=*/0, /*vta=*/false, /*vma=*/false,
                      /*altfmt=*/true);
  fp_state_->fcsr()->Write(0U);

  ASSERT_OK(SetTileMatrix<float>(/*tile=*/0, {{0.0f}}));
  WriteVectorRegister<uint16_t>("v8", {0x3C00, 0});
  WriteVectorRegister<uint16_t>("v16", {0x3C00, 0});

  ScopedTrapCapture trap_capture(state_.get());
  ExecuteMatmul(&RiscVZvtVtfmmAlt);
  EXPECT_FALSE(trap_capture.trap_taken());

  ASSERT_OK_AND_ASSIGN(uint32_t res_val,
                       GetTileElem<uint32_t>(/*tile=*/0, /*row=*/0, /*col=*/0));
  EXPECT_NE(res_val, 0U);
  EXPECT_EQ(fp_state_->fcsr()->AsUint32() & *FPExceptions::kDivByZero, 0U);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmDynamicDispatchFp8E4M3) {
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();

  vsstatus()->set_ms(kMstatusStateInitial);

  matrix_state_->set_mtype(EncodeMtype(kMtwiden4));
  matrix_state_->set_tile_m(2);
  matrix_state_->set_tile_k(2);
  vector_state_->set_vector_length(2);
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1, /*vta=*/false,
                                           /*vma=*/false,
                                           /*altfmt=*/false));

  std::vector<RegisterBase*> reg_vec;
  for (int i = 0; i < 8; i++) {
    reg_vec.push_back(
        state_->GetRegister<RVVectorRegister>(absl::StrCat("v", 8 + i)).first);
  }
  for (int i = 0; i < 8; ++i) {
    auto dest = reg_vec[i]->data_buffer()->Get<uint32_t>();
    for (size_t j = 0; j < dest.size(); ++j) {
      dest[j] = 0x40404040;
    }
  }
  instruction_->AppendSource(
      new RV32VectorSourceOperand(absl::MakeSpan(reg_vec), "v8"));
  instruction_->AppendSource(
      new RV32VectorSourceOperand(absl::MakeSpan(reg_vec), "v8"));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*tile_d=*/0));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs1=*/8));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs2=*/8));

  // Should not crash, basically testing it reaches the proper matmul function
  ScopedTrapCapture trap_capture(state_.get());
  instruction_->set_semantic_function(&RiscVZvtVtfmm);
  instruction_->Execute(nullptr);
  EXPECT_FALSE(trap_capture.trap_taken());

  // Assert output is physically calculated and present
  ASSERT_OK_AND_ASSIGN(uint32_t res_val, matrix_state_->GetElem<uint32_t>(
                                             /*tile=*/0, /*row=*/0, /*col=*/0,
                                             /*element_width_in_bits=*/32));
  EXPECT_NE(res_val, 0U);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmDynamicDispatchFp8E4M3xE5M2) {
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();
  matrix_state_->set_mtype(EncodeMtype(kMtwiden4));
  matrix_state_->set_tile_m(2);
  matrix_state_->set_tile_k(1);
  vector_state_->set_vector_length(2);
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1,
                                           /*vta=*/false, /*vma=*/false,
                                           /*altfmt=*/true));

  std::vector<RegisterBase*> reg_vec;
  for (int i = 0; i < 8; i++) {
    reg_vec.push_back(
        state_->GetRegister<RVVectorRegister>(absl::StrCat("v", 8 + i)).first);
  }
  for (int i = 0; i < 8; ++i) {
    auto dest = reg_vec[i]->data_buffer()->Get<uint32_t>();
    for (size_t j = 0; j < dest.size(); ++j) {
      dest[j] = 0x40404040;
    }
  }
  instruction_->AppendSource(
      new RV32VectorSourceOperand(absl::MakeSpan(reg_vec), "v8"));
  instruction_->AppendSource(
      new RV32VectorSourceOperand(absl::MakeSpan(reg_vec), "v8"));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*tile_d=*/0));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs1=*/8));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs2=*/8));

  ScopedTrapCapture trap_capture(state_.get());
  instruction_->set_semantic_function(&RiscVZvtVtfmm);
  instruction_->Execute(nullptr);
  EXPECT_FALSE(trap_capture.trap_taken());

  // Assert output is physically calculated and present
  ASSERT_OK_AND_ASSIGN(uint32_t res_val, matrix_state_->GetElem<uint32_t>(
                                             /*tile=*/0, /*row=*/0, /*col=*/0,
                                             /*element_width_in_bits=*/32));
  EXPECT_NE(res_val, 0U);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmDynamicDispatchFp16) {
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();
  matrix_state_->set_mtype(EncodeMtype(kMtwiden2));
  matrix_state_->set_tile_m(2);
  matrix_state_->set_tile_k(1);
  vector_state_->set_vector_length(2);
  vector_state_->SetVectorType(EncodeVtype(kVsew16, kVlmul1));

  std::vector<RegisterBase*> reg_vec;
  for (int i = 0; i < 8; i++) {
    reg_vec.push_back(
        state_->GetRegister<RVVectorRegister>(absl::StrCat("v", 8 + i)).first);
  }
  for (int i = 0; i < 8; ++i) {
    auto dest = reg_vec[i]->data_buffer()->Get<uint32_t>();
    for (size_t j = 0; j < dest.size(); ++j) {
      dest[j] = 0x40404040;
    }
  }
  instruction_->AppendSource(
      new RV32VectorSourceOperand(absl::MakeSpan(reg_vec), "v8"));
  instruction_->AppendSource(
      new RV32VectorSourceOperand(absl::MakeSpan(reg_vec), "v8"));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*tile_d=*/0));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs1=*/8));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs2=*/8));

  ScopedTrapCapture trap_capture(state_.get());
  instruction_->set_semantic_function(&RiscVZvtVtfmm);
  instruction_->Execute(nullptr);
  EXPECT_FALSE(trap_capture.trap_taken());

  // Assert output is physically calculated and present
  ASSERT_OK_AND_ASSIGN(uint32_t res_val, matrix_state_->GetElem<uint32_t>(
                                             /*tile=*/0, /*row=*/0, /*col=*/0,
                                             /*element_width_in_bits=*/32));
  EXPECT_NE(res_val, 0U);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmDynamicDispatchFp32) {
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();
  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  matrix_state_->set_tile_m(2);
  matrix_state_->set_tile_k(1);
  vector_state_->set_vector_length(2);
  vector_state_->SetVectorType(EncodeVtype(kVsew32, kVlmul1));

  std::vector<RegisterBase*> reg_vec;
  for (int i = 0; i < 8; i++) {
    reg_vec.push_back(
        state_->GetRegister<RVVectorRegister>(absl::StrCat("v", 8 + i)).first);
  }
  for (int i = 0; i < 8; ++i) {
    auto dest = reg_vec[i]->data_buffer()->Get<uint32_t>();
    for (size_t j = 0; j < dest.size(); ++j) {
      dest[j] = 0x40404040;
    }
  }
  instruction_->AppendSource(
      new RV32VectorSourceOperand(absl::MakeSpan(reg_vec), "v8"));
  instruction_->AppendSource(
      new RV32VectorSourceOperand(absl::MakeSpan(reg_vec), "v8"));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*tile_d=*/0));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs1=*/8));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs2=*/8));

  ScopedTrapCapture trap_capture(state_.get());
  instruction_->set_semantic_function(&RiscVZvtVtfmm);
  instruction_->Execute(nullptr);
  EXPECT_FALSE(trap_capture.trap_taken());

  // Assert output is physically calculated and present
  ASSERT_OK_AND_ASSIGN(uint32_t res_val, matrix_state_->GetElem<uint32_t>(
                                             /*tile=*/0, /*row=*/0, /*col=*/0,
                                             /*element_width_in_bits=*/32));
  EXPECT_NE(res_val, 0U);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmDynamicDispatchFp64) {
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();
  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  matrix_state_->set_tile_m(2);
  matrix_state_->set_tile_k(1);
  vector_state_->set_vector_length(2);
  vector_state_->SetVectorType(EncodeVtype(kVsew64, kVlmul1));

  std::vector<RegisterBase*> reg_vec;
  for (int i = 0; i < 8; i++) {
    reg_vec.push_back(
        state_->GetRegister<RVVectorRegister>(absl::StrCat("v", 8 + i)).first);
  }
  for (int i = 0; i < 8; ++i) {
    auto dest = reg_vec[i]->data_buffer()->Get<uint32_t>();
    for (size_t j = 0; j < dest.size(); ++j) {
      dest[j] = 0x40404040;
    }
  }
  instruction_->AppendSource(
      new RV32VectorSourceOperand(absl::MakeSpan(reg_vec), "v8"));
  instruction_->AppendSource(
      new RV32VectorSourceOperand(absl::MakeSpan(reg_vec), "v8"));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*tile_d=*/0));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs1=*/8));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs2=*/8));

  ScopedTrapCapture trap_capture(state_.get());
  instruction_->set_semantic_function(&RiscVZvtVtfmm);
  instruction_->Execute(nullptr);
  EXPECT_FALSE(trap_capture.trap_taken());

  // Assert output is physically calculated and present
  ASSERT_OK_AND_ASSIGN(uint64_t res_val, matrix_state_->GetElem<uint64_t>(
                                             /*tile=*/0, /*row=*/0, /*col=*/0,
                                             /*element_width_in_bits=*/64));
  EXPECT_NE(res_val, 0U);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmAltDynamicDispatchFp8E5M2xE4M3) {
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();
  matrix_state_->set_mtype(EncodeMtype(kMtwiden4));
  matrix_state_->set_tile_m(2);
  matrix_state_->set_tile_k(1);
  vector_state_->set_vector_length(2);
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));

  std::vector<RegisterBase*> reg_vec;
  for (int i = 0; i < 8; i++) {
    reg_vec.push_back(
        state_->GetRegister<RVVectorRegister>(absl::StrCat("v", 8 + i)).first);
  }
  for (int i = 0; i < 8; ++i) {
    auto dest = reg_vec[i]->data_buffer()->Get<uint32_t>();
    for (size_t j = 0; j < dest.size(); ++j) {
      dest[j] = 0x40404040;
    }
  }
  instruction_->AppendSource(
      new RV32VectorSourceOperand(absl::MakeSpan(reg_vec), "v8"));
  instruction_->AppendSource(
      new RV32VectorSourceOperand(absl::MakeSpan(reg_vec), "v8"));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*tile_d=*/0));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs1=*/8));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs2=*/8));

  ScopedTrapCapture trap_capture(state_.get());
  instruction_->set_semantic_function(&RiscVZvtVtfmmAlt);
  instruction_->Execute(nullptr);
  EXPECT_FALSE(trap_capture.trap_taken());

  // Assert output is physically calculated and present
  ASSERT_OK_AND_ASSIGN(uint32_t res_val, matrix_state_->GetElem<uint32_t>(
                                             /*tile=*/0, /*row=*/0, /*col=*/0,
                                             /*element_width_in_bits=*/32));
  EXPECT_NE(res_val, 0U);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmAltDynamicDispatchFp8E5M2) {
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();
  matrix_state_->set_mtype(EncodeMtype(kMtwiden4));
  matrix_state_->set_tile_m(2);
  matrix_state_->set_tile_k(1);
  vector_state_->set_vector_length(2);
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1,
                                           /*vta=*/false, /*vma=*/false,
                                           /*altfmt=*/true));

  std::vector<RegisterBase*> reg_vec;
  for (int i = 0; i < 8; i++) {
    reg_vec.push_back(
        state_->GetRegister<RVVectorRegister>(absl::StrCat("v", 8 + i)).first);
  }
  for (int i = 0; i < 8; ++i) {
    auto dest = reg_vec[i]->data_buffer()->Get<uint32_t>();
    for (size_t j = 0; j < dest.size(); ++j) {
      dest[j] = 0x40404040;
    }
  }
  instruction_->AppendSource(
      new RV32VectorSourceOperand(absl::MakeSpan(reg_vec), "v8"));
  instruction_->AppendSource(
      new RV32VectorSourceOperand(absl::MakeSpan(reg_vec), "v8"));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*tile_d=*/0));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs1=*/8));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs2=*/8));

  ScopedTrapCapture trap_capture(state_.get());
  instruction_->set_semantic_function(&RiscVZvtVtfmmAlt);
  instruction_->Execute(nullptr);
  EXPECT_FALSE(trap_capture.trap_taken());

  // Assert output is physically calculated and present
  ASSERT_OK_AND_ASSIGN(uint32_t res_val, matrix_state_->GetElem<uint32_t>(
                                             /*tile=*/0, /*row=*/0, /*col=*/0,
                                             /*element_width_in_bits=*/32));
  EXPECT_NE(res_val, 0U);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmAltDynamicDispatchBf16) {
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();
  matrix_state_->set_mtype(EncodeMtype(kMtwiden2));
  matrix_state_->set_tile_m(2);
  matrix_state_->set_tile_k(1);
  vector_state_->set_vector_length(2);
  vector_state_->SetVectorType(EncodeVtype(kVsew16, kVlmul1,
                                           /*vta=*/false, /*vma=*/false,
                                           /*altfmt=*/true));

  std::vector<RegisterBase*> reg_vec;
  for (int i = 0; i < 8; i++) {
    reg_vec.push_back(
        state_->GetRegister<RVVectorRegister>(absl::StrCat("v", 8 + i)).first);
  }
  for (int i = 0; i < 8; ++i) {
    auto dest = reg_vec[i]->data_buffer()->Get<uint32_t>();
    for (size_t j = 0; j < dest.size(); ++j) {
      dest[j] = 0x40404040;
    }
  }
  instruction_->AppendSource(
      new RV32VectorSourceOperand(absl::MakeSpan(reg_vec), "v8"));
  instruction_->AppendSource(
      new RV32VectorSourceOperand(absl::MakeSpan(reg_vec), "v8"));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*tile_d=*/0));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs1=*/8));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs2=*/8));

  ScopedTrapCapture trap_capture(state_.get());
  instruction_->set_semantic_function(&RiscVZvtVtfmmAlt);
  instruction_->Execute(nullptr);
  EXPECT_FALSE(trap_capture.trap_taken());

  // Assert output is physically calculated and present
  ASSERT_OK_AND_ASSIGN(uint32_t res_val, matrix_state_->GetElem<uint32_t>(
                                             /*tile=*/0, /*row=*/0, /*col=*/0,
                                             /*element_width_in_bits=*/32));
  EXPECT_NE(res_val, 0U);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmTrapUnsupportedSewTwidenAltfmt) {
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();

  // Test SEW=16 altfmt=1 for vtfmm.tvv which is unsupported
  matrix_state_->set_mtype(EncodeMtype(kMtwiden2));
  vector_state_->set_vector_length(2);
  vector_state_->SetVectorType(EncodeVtype(kVsew16, kVlmul1, /*vta=*/false,
                                           /*vma=*/false, /*altfmt=*/true));

  ScopedTrapCapture trap_capture(state_.get());

  std::vector<RegisterBase*> reg_vec;
  for (int i = 0; i < 8; i++) {
    reg_vec.push_back(
        state_->GetRegister<RVVectorRegister>(absl::StrCat("v", 8 + i)).first);
  }
  instruction_->AppendSource(
      new RV32VectorSourceOperand(absl::MakeSpan(reg_vec), "v8"));
  instruction_->AppendSource(
      new RV32VectorSourceOperand(absl::MakeSpan(reg_vec), "v8"));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*tile_d=*/0));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs1=*/8));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs2=*/8));

  instruction_->set_semantic_function(&RiscVZvtVtfmm);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmAltTrapUnsupportedSewTwidenAltfmt) {
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();

  // Test SEW=32 for vtfmm.alt.tvv which is unsupported
  matrix_state_->set_mtype(EncodeMtype(kMtwiden2));
  vector_state_->set_vector_length(2);
  vector_state_->SetVectorType(EncodeVtype(kVsew32, kVlmul1));

  ScopedTrapCapture trap_capture(state_.get());

  std::vector<RegisterBase*> reg_vec;
  for (int i = 0; i < 8; i++) {
    reg_vec.push_back(
        state_->GetRegister<RVVectorRegister>(absl::StrCat("v", 8 + i)).first);
  }
  instruction_->AppendSource(
      new RV32VectorSourceOperand(absl::MakeSpan(reg_vec), "v8"));
  instruction_->AppendSource(
      new RV32VectorSourceOperand(absl::MakeSpan(reg_vec), "v8"));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*tile_d=*/0));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs1=*/8));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs2=*/8));

  instruction_->set_semantic_function(&RiscVZvtVtfmmAlt);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, MsetmtypeiConfiguresAltfmtBit) {
  // mtwiden (zimm10) = 1, vsew = 2, altfmt = 1
  instruction_->AppendSource(
      new ImmediateOperand<uint32_t>(/*mtwiden=*/kMtwiden1));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vsew=*/kVsew16));
  instruction_->AppendSource(
      new ImmediateOperand<uint32_t>(/*altfmt=*/kAltfmtAlternate));

  instruction_->set_semantic_function(&RiscVZvtMsetMtypei);
  instruction_->Execute(nullptr);

  // vsew=2 (1<<4), altfmt=1 (1<<8) => 0x110
  EXPECT_EQ(GetVtypeAltfmt(vector_state_.get()), 1);
}

TEST_F(RiscVZvtInstructionsTest, VectorConfigInvalidationZerosAltfmt) {
  // First set altfmt=1
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1,
                                           /*vta=*/false, /*vma=*/false,
                                           /*altfmt=*/true));
  EXPECT_EQ(GetVtypeAltfmt(vector_state_.get()), 1);

  // Now set vill (invalid) by setting MSB
  vector_state_->SetVectorType(kVillMask);
  EXPECT_EQ(GetVtypeAltfmt(vector_state_.get()), 0);
}

TEST_F(RiscVZvtInstructionsTest, MatmulVmaqaExecutionLoops) {
  // Zvt spec v0.4: vtmmu.tvv (altfmt=0) uint8 x uint8 -> int32.
  // Use tile_k = 4 to exercise multiple k-accumulation iterations.
  ConfigureMatrixUnit(kMtwiden4, kVsew8, kVlmul1, /*vl=*/2, /*tk=*/4,
                      /*tm=*/2);

  ASSERT_OK(SetTileMatrix<int32_t>(/*tile=*/0, {{10, 10}, {10, 10}}));

  // KMAX(8) = 4, kStride = 8 / 4 = 2.
  for (int i = 0; i < 8; ++i) {
    FillVectorRegister<uint8_t>(absl::StrCat("v", 8 + i), 2);
    FillVectorRegister<uint8_t>(absl::StrCat("v", 16 + i), 3);
  }

  ExecuteMatmul(&RiscVZvtVtmmu);

  // acc = 10 + (4 * (2 * 3)) = 34
  ASSERT_OK_AND_ASSIGN(
      auto result, GetTileRegion<int32_t>(/*tile=*/0, /*rows=*/2, /*cols=*/2));
  EXPECT_THAT(result, testing::Each(34));
}

TEST_F(RiscVZvtInstructionsTest, MatmulLmulCapacityAlignmentTrap) {
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();

  vsstatus()->set_ms(kMstatusStateInitial);
  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  vector_state_->set_vector_length(4);
  vector_state_->SetVectorType(EncodeVtype(kVsew64, kVlmul4));

  ScopedTrapCapture trap_capture(state_.get());

  std::vector<RegisterBase*> reg_vec_a;
  for (int i = 0; i < 8; i++) {
    reg_vec_a.push_back(
        state_->GetRegister<RVVectorRegister>(absl::StrCat("v", 1 + i)).first);
  }
  // reg_a = 1, lmul = 4. reg_a % lmul = 1 != 0
  auto* src_op_a = new RV32VectorSourceOperand(absl::MakeSpan(reg_vec_a), "v1");
  instruction_->AppendSource(src_op_a);

  std::vector<RegisterBase*> reg_vec_b;
  for (int i = 0; i < 8; i++) {
    reg_vec_b.push_back(
        state_->GetRegister<RVVectorRegister>(absl::StrCat("v", 8 + i)).first);
  }
  auto* src_op_b = new RV32VectorSourceOperand(absl::MakeSpan(reg_vec_b), "v8");
  instruction_->AppendSource(src_op_b);

  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*tile_d=*/0));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs1=*/1));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs2=*/8));

  instruction_->set_semantic_function(&RiscVZvtVtfmm);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, MatmulExecutionStrideTrap) {
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();

  vsstatus()->set_ms(kMstatusStateInitial);
  matrix_state_->set_mtype(EncodeMtype(kMtwiden2));
  vector_state_->set_vector_length(4);
  vector_state_->SetVectorType(EncodeVtype(kVsew16, kVlmul1));

  ScopedTrapCapture trap_capture(state_.get());

  std::vector<RegisterBase*> reg_vec_a;
  for (int i = 0; i < 8; i++) {
    reg_vec_a.push_back(
        state_->GetRegister<RVVectorRegister>(absl::StrCat("v", 5 + i)).first);
  }
  // SEW=16 -> stride = 4. reg_a = 5. reg_a % 8 = 5 >= 4
  auto* src_op_a = new RV32VectorSourceOperand(absl::MakeSpan(reg_vec_a), "v5");
  instruction_->AppendSource(src_op_a);

  std::vector<RegisterBase*> reg_vec_b;
  for (int i = 0; i < 8; i++) {
    reg_vec_b.push_back(
        state_->GetRegister<RVVectorRegister>(absl::StrCat("v", 8 + i)).first);
  }
  auto* src_op_b = new RV32VectorSourceOperand(absl::MakeSpan(reg_vec_b), "v8");
  instruction_->AppendSource(src_op_b);

  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*tile_d=*/0));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs1=*/5));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs2=*/8));

  instruction_->set_semantic_function(&RiscVZvtVtfmm);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, Fp8E4m3ToF32NaNPermutation) {
  // E4M3 NaN is when exp = 1111 and mantissa != 0
  // 0x7F = 0 1111 111
  float res = ConvertFp8E4m3ToSingle(Fp8E4m3{0x7F});
  EXPECT_TRUE(std::isnan(res));
}

TEST_F(RiscVZvtInstructionsTest, Fp8E5m2ToF32NaNPermutation) {
  // E5M2 NaN is when exp = 11111 and mantissa != 0
  // 0x7F = 0 11111 11
  float res = ConvertFp8E5m2ToSingle(Fp8E5m2{0x7F});
  EXPECT_TRUE(std::isnan(res));
}

TEST_F(RiscVZvtInstructionsTest, Bf16ToF32NaNPermutation) {
  // Bf16 NaN is when exp = 11111111 and mantissa != 0
  // 0x7FC0
  float res = ConvertBf16ToSingle(Bfloat16{0x7FC0});
  EXPECT_TRUE(std::isnan(res));
}

TEST_F(RiscVZvtInstructionsTest, Fp8E4m3ToF32SubnormalPermutation) {
  // E4M3 subnormal is when exp = 0 and mantissa != 0
  // 0x01 = 0 0000 001
  float res = ConvertFp8E4m3ToSingle(Fp8E4m3{0x01});
  EXPECT_TRUE(std::fpclassify(res) == FP_NORMAL);
}

TEST_F(RiscVZvtInstructionsTest, Fp8E5m2ToF32SubnormalPermutation) {
  // E5M2 subnormal is when exp = 0 and mantissa != 0
  // 0x01 = 0 00000 01
  float res = ConvertFp8E5m2ToSingle(Fp8E5m2{0x01});
  EXPECT_TRUE(std::fpclassify(res) == FP_NORMAL);
}

TEST_F(RiscVZvtInstructionsTest, Bf16ToF32SubnormalPermutation) {
  // Bf16 subnormal is when exp = 0 and mantissa != 0
  // 0x0001
  float res = ConvertBf16ToSingle(Bfloat16{0x0001});
  EXPECT_TRUE(std::fpclassify(res) == FP_SUBNORMAL);
}

TEST_F(RiscVZvtInstructionsTest, VtleMisalignedTrapCalculatedAddress) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4, /*tk=*/0, /*tm=*/0,
                      /*vstart=*/2);

  constexpr uint64_t kBaseAddr = 0x2001;  // misaligned!
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/32, kBaseAddr,
                      EncodeTss(/*tile=*/1, kTssPatternRow, /*index=*/2));

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kLoadAddressMisaligned);
  EXPECT_EQ(trap_capture.trap_value(), 0x2001 + 2 * 4);
}

TEST_F(RiscVZvtInstructionsTest, VtseMisalignedTrapCalculatedAddress) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4, /*tk=*/0, /*tm=*/0,
                      /*vstart=*/2);

  constexpr uint64_t kBaseAddr = 0x2001;  // misaligned!
  ScopedTrapCapture trap_capture(state_.get());

  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/32, kBaseAddr,
                      EncodeTss(/*tile=*/1, kTssPatternRow, /*index=*/2));

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kStoreAddressMisaligned);
  EXPECT_EQ(trap_capture.trap_value(), 0x2001 + 2 * 4);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmUnmappedTopologyTrapsIllegalInstruction) {
  // mtwiden = 2
  matrix_state_->set_mtype(EncodeMtype(kMtwiden2));
  vector_state_->SetVectorType(EncodeVtype(kVsew16, kVlmul1,
                                           /*vta=*/false, /*vma=*/false,
                                           /*altfmt=*/true));
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->Submit();

  ScopedTrapCapture trap_capture(state_.get());

  AppendMatmulSources(/*reg_a=*/8, /*reg_b=*/16, /*dest_tile=*/0);
  instruction_->set_semantic_function(&RiscVZvtVtfmm);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest,
       VtfmmAltUnmappedTopologyTrapsIllegalInstruction) {
  // SEW=8 TWIDEN=2 is unmapped for vtfmm.alt.tvv (which requires TWIDEN=4 for
  // SEW=8)
  matrix_state_->set_mtype(EncodeMtype(kMtwiden2));
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->Submit();

  ScopedTrapCapture trap_capture(state_.get());

  AppendMatmulSources(/*reg_a=*/8, /*reg_b=*/16, /*dest_tile=*/0);
  instruction_->set_semantic_function(&RiscVZvtVtfmmAlt);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmSew32IntermediateRoundingTdd) {
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_fs(kMstatusStateInitial);
  state_->mstatus()->Submit();

  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  matrix_state_->set_tile_m(2);
  matrix_state_->set_tile_k(1);
  vector_state_->set_vector_length(2);
  vector_state_->SetVectorType(EncodeVtype(kVsew32, kVlmul1));

  // A = 1 + 2^-12, B = 1 + 2^-12. Product = 1 + 2^-11 + 2^-24.
  // TEW=32 rounding should truncate the 2^-24 before accumulation.

  std::vector<RegisterBase*> reg_vec_a;
  for (int i = 0; i < 8; i++) {
    reg_vec_a.push_back(
        state_->GetRegister<RVVectorRegister>(absl::StrCat("v", 8 + i)).first);
  }
  auto* src_op_a = new RV32VectorSourceOperand(absl::MakeSpan(reg_vec_a), "v8");
  instruction_->AppendSource(src_op_a);

  std::vector<RegisterBase*> reg_vec_b;
  for (int i = 0; i < 8; i++) {
    reg_vec_b.push_back(
        state_->GetRegister<RVVectorRegister>(absl::StrCat("v", 16 + i)).first);
  }
  auto* src_op_b =
      new RV32VectorSourceOperand(absl::MakeSpan(reg_vec_b), "v16");
  instruction_->AppendSource(src_op_b);

  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*tile_d=*/0));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs1=*/8));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs2=*/16));

  instruction_->set_semantic_function(&RiscVZvtVtfmm);
  instruction_->Execute(nullptr);
}

TEST_F(RiscVZvtInstructionsTest, VtDiscardDowngradesToInitialState) {
  state_->mstatus()->set_ms(kMstatusStateDirty);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();

  vector_state_->SetVectorType(EncodeVtype(kVsew32, kVlmul1));

  instruction_->set_semantic_function(&RiscVZvtVtDiscard);
  instruction_->Execute(nullptr);

  EXPECT_EQ(state_->mstatus()->ms(), kMstatusStateInitial);
  EXPECT_EQ(state_->mstatus()->vs(),
            kMstatusStateDirty);  // Clearing vstart dirties VS.
}

TEST_F(RiscVZvtInstructionsTest, VtfmmDirtiesMstatusFsOnMatmul) {
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->set_fs(kMstatusStateInitial);
  state_->mstatus()->Submit();

  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  matrix_state_->set_tile_m(2);
  matrix_state_->set_tile_k(1);
  vector_state_->set_vector_length(2);
  vector_state_->SetVectorType(EncodeVtype(kVsew32, kVlmul1));

  AppendMatmulSources(/*reg_a=*/8, /*reg_b=*/16, /*dest_tile=*/0);

  instruction_->set_semantic_function(&RiscVZvtVtfmm);
  instruction_->Execute(nullptr);

  EXPECT_EQ(state_->mstatus()->fs(), kMstatusStateDirty);
}

TEST_F(RiscVZvtInstructionsTest, VtzeroTrapReservedTile) {
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->Submit();

  // Configure TEW=32 (SEW=8, TWIDEN=4 -> mtwiden=2)
  matrix_state_->set_mtype(EncodeMtype(kMtwiden2));
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));

  // For TEW=32, valid canonical tiles are multiples of 4 (0, 4, 8, 12).
  // Pass tile=1 (reserved / non-canonical) to trigger illegal instruction trap.
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*tile_d=*/1));

  ScopedTrapCapture trap_capture(state_.get());

  instruction_->set_semantic_function(&RiscVZvtVtZero);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtleReservedTssTrapOverMisalignedTrap) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4, /*tk=*/0, /*tm=*/0,
                      /*vstart=*/2);

  ScopedTrapCapture trap_capture(state_.get());
  constexpr uint64_t kBaseAddr = 0x2001;  // Misaligned address.

  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/32, kBaseAddr,
                      EncodeTss(/*tile=*/1, kTssPatternInvalid, /*index=*/2));

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
  EXPECT_EQ(trap_capture.trap_value(), 0);
}

TEST_F(RiscVZvtInstructionsTest, VtseReservedTssTrapOverMisalignedTrap) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4, /*tk=*/0, /*tm=*/0,
                      /*vstart=*/2);

  ScopedTrapCapture trap_capture(state_.get());
  constexpr uint64_t kBaseAddr = 0x2001;  // Misaligned address.

  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/32, kBaseAddr,
                      EncodeTss(/*tile=*/1, kTssPatternInvalid, /*index=*/2));

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
  EXPECT_EQ(trap_capture.trap_value(), 0);
}

TEST_F(RiscVZvtInstructionsTest, VtleVstartGreaterOrEqualCountSkipLimit) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4, /*tk=*/0, /*tm=*/0,
                      /*vstart=*/200);

  ScopedTrapCapture trap_capture(state_.get());
  constexpr uint64_t kBaseAddr = 0x2001;  // Misaligned address.

  ExecuteTileMemoryOp(&RiscVZvtVtle, /*eew=*/32, kBaseAddr,
                      EncodeTss(/*tile=*/0, kTssPatternRow, /*index=*/2));

  EXPECT_FALSE(trap_capture.trap_taken());
  EXPECT_EQ(vector_state_->vstart(), 0);
  EXPECT_EQ(state_->mstatus()->vs(), kMstatusStateDirty);
}

TEST_F(RiscVZvtInstructionsTest, VtseVstartGreaterOrEqualCountSkipLimit) {
  ConfigureMatrixUnit(kMtwiden1, kVsew32, kVlmul1, /*vl=*/4, /*tk=*/0, /*tm=*/0,
                      /*vstart=*/200);

  ScopedTrapCapture trap_capture(state_.get());
  constexpr uint64_t kBaseAddr = 0x2001;  // Misaligned address.

  ExecuteTileMemoryOp(&RiscVZvtVtse, /*eew=*/32, kBaseAddr,
                      EncodeTss(/*tile=*/0, kTssPatternRow, /*index=*/2));

  EXPECT_FALSE(trap_capture.trap_taken());
  EXPECT_EQ(vector_state_->vstart(), 0);
  EXPECT_EQ(state_->mstatus()->vs(), kMstatusStateDirty);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmTrapWhenMtwidenNotZeroAndUnaligned) {
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->set_fs(kMstatusStateInitial);
  state_->mstatus()->Submit();

  // mtwiden = 2 -> TWIDEN = 2, tew = 32, align = 4
  matrix_state_->set_mtype(EncodeMtype(kMtwiden2));
  vector_state_->set_vector_length(2);
  vector_state_->SetVectorType(EncodeVtype(kVsew16, kVlmul1));

  ScopedTrapCapture trap_capture(state_.get());

  AppendVectorGroupSource(/*base_reg=*/8);
  AppendVectorGroupSource(/*base_reg=*/8);
  // Unaligned tile_d.
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*tile_d=*/1));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs1=*/8));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs2=*/8));

  instruction_->set_semantic_function(&RiscVZvtVtfmm);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmAltTrapWhenMtwidenNotZeroAndUnaligned) {
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->set_fs(kMstatusStateInitial);
  state_->mstatus()->Submit();

  matrix_state_->set_mtype(EncodeMtype(kMtwiden2));
  vector_state_->set_vector_length(2);
  vector_state_->SetVectorType(EncodeVtype(kVsew16, kVlmul1, /*vta=*/false,
                                           /*vma=*/false, /*altfmt=*/true));

  ScopedTrapCapture trap_capture(state_.get());

  AppendVectorGroupSource(/*base_reg=*/8);
  AppendVectorGroupSource(/*base_reg=*/8);
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*tile_d=*/1));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs1=*/8));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs2=*/8));

  instruction_->set_semantic_function(&RiscVZvtVtfmmAlt);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtmmuTrapWhenMtwidenNotZeroAndUnaligned) {
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();

  // mtwiden = 3 -> TWIDEN = 4, tew = 32, align = 4
  matrix_state_->set_mtype(EncodeMtype(kMtwiden4));
  vector_state_->set_vector_length(2);
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));

  ScopedTrapCapture trap_capture(state_.get());

  AppendVectorGroupSource(/*base_reg=*/8);
  AppendVectorGroupSource(/*base_reg=*/8);
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*tile_d=*/1));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs1=*/8));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs2=*/8));

  instruction_->set_semantic_function(&RiscVZvtVtmmu);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtmmsTrapWhenMtwidenNotZeroAndUnaligned) {
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();

  matrix_state_->set_mtype(EncodeMtype(kMtwiden4));
  vector_state_->set_vector_length(2);
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));

  ScopedTrapCapture trap_capture(state_.get());

  AppendVectorGroupSource(/*base_reg=*/8);
  AppendVectorGroupSource(/*base_reg=*/8);
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*tile_d=*/1));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs1=*/8));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs2=*/8));

  instruction_->set_semantic_function(&RiscVZvtVtmms);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtmmuUnmappedTopologyTrapsIllegalInstruction) {
  matrix_state_->set_mtype(EncodeMtype(kMtwiden2));
  vector_state_->SetVectorType(EncodeVtype(kVsew16, kVlmul1));
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->Submit();

  ScopedTrapCapture trap_capture(state_.get());

  AppendVectorGroupSource(/*base_reg=*/8);
  AppendVectorGroupSource(/*base_reg=*/8);
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*tile_d=*/1));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs1=*/8));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs2=*/8));

  instruction_->set_semantic_function(&RiscVZvtVtmmu);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtmmsUnmappedTopologyTrapsIllegalInstruction) {
  matrix_state_->set_mtype(EncodeMtype(kMtwiden2));
  vector_state_->SetVectorType(EncodeVtype(kVsew16, kVlmul1));
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->Submit();

  ScopedTrapCapture trap_capture(state_.get());

  AppendVectorGroupSource(/*base_reg=*/8);
  AppendVectorGroupSource(/*base_reg=*/8);
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*tile_d=*/1));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs1=*/8));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(/*vs2=*/8));

  instruction_->set_semantic_function(&RiscVZvtVtmms);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtmmuMappedTopologyComputesCorrectly) {
  matrix_state_->set_mtype(EncodeMtype(kMtwiden4));
  matrix_state_->set_tile_m(1);
  matrix_state_->set_tile_k(2);
  vector_state_->set_vector_length(2);
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();

  ASSERT_OK(matrix_state_->SetElem<int32_t>(
      /*tile=*/0, /*row=*/0, /*col=*/0, /*element_width_in_bits=*/32,
      /*value=*/0));

  ScopedTrapCapture trap_capture(state_.get());

  auto* reg_v8 = state_->GetRegister<RVVectorRegister>("v8").first;
  auto span_v8 = reg_v8->data_buffer()->Get<uint8_t>();
  span_v8[0] = 2;
  span_v8[1] = 3;

  auto* reg_v16 = state_->GetRegister<RVVectorRegister>("v16").first;
  auto span_v16 = reg_v16->data_buffer()->Get<uint8_t>();
  span_v16[0] = 4;
  span_v16[1] = 5;

  AppendMatmulSources(/*reg_a=*/8, /*reg_b=*/16, /*dest_tile=*/0);

  instruction_->set_semantic_function(&RiscVZvtVtmmu);
  instruction_->Execute(nullptr);

  EXPECT_FALSE(trap_capture.trap_taken());
  ASSERT_OK_AND_ASSIGN(auto elem, matrix_state_->GetElem<int32_t>(
                                      /*tile=*/0, /*row=*/0, /*col=*/0,
                                      /*element_width_in_bits=*/32));
  EXPECT_EQ(elem, 8);
}

TEST_F(RiscVZvtInstructionsTest, VtmmsMappedTopologyComputesCorrectly) {
  matrix_state_->set_mtype(EncodeMtype(kMtwiden4));
  matrix_state_->set_tile_m(1);
  matrix_state_->set_tile_k(2);
  vector_state_->set_vector_length(2);
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->Submit();

  ASSERT_OK(matrix_state_->SetElem<int32_t>(
      /*tile=*/0, /*row=*/0, /*col=*/0, /*element_width_in_bits=*/32,
      /*value=*/0));

  ScopedTrapCapture trap_capture(state_.get());

  auto* reg_v8 = state_->GetRegister<RVVectorRegister>("v8").first;
  auto span_v8 = reg_v8->data_buffer()->Get<int8_t>();
  span_v8[0] = -2;
  span_v8[1] = 3;

  auto* reg_v16 = state_->GetRegister<RVVectorRegister>("v16").first;
  auto span_v16 = reg_v16->data_buffer()->Get<int8_t>();
  span_v16[0] = 4;
  span_v16[1] = -5;

  AppendMatmulSources(/*reg_a=*/8, /*reg_b=*/16, /*dest_tile=*/0);

  instruction_->set_semantic_function(&RiscVZvtVtmms);
  instruction_->Execute(nullptr);

  EXPECT_FALSE(trap_capture.trap_taken());
  ASSERT_OK_AND_ASSIGN(auto elem, matrix_state_->GetElem<int32_t>(
                                      /*tile=*/0, /*row=*/0, /*col=*/0,
                                      /*element_width_in_bits=*/32));
  EXPECT_EQ(elem, -8);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmTrapWhenFpStateOff) {
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->set_fs(kMstatusStateOff);  // FS Off -> must trap!
  state_->mstatus()->Submit();

  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  matrix_state_->set_tile_m(2);
  matrix_state_->set_tile_k(1);
  vector_state_->set_vector_length(2);
  vector_state_->SetVectorType(EncodeVtype(kVsew32, kVlmul1));

  ScopedTrapCapture trap_capture(state_.get());

  AppendMatmulSources(/*reg_a=*/8, /*reg_b=*/16, /*dest_tile=*/0);

  instruction_->set_semantic_function(&RiscVZvtVtfmm);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

TEST_F(RiscVZvtInstructionsTest, VtfmmAltTrapWhenFpStateOff) {
  state_->mstatus()->set_ms(kMstatusStateInitial);
  state_->mstatus()->set_vs(kMstatusStateInitial);
  state_->mstatus()->set_fs(kMstatusStateOff);  // FS Off -> must trap!
  state_->mstatus()->Submit();

  matrix_state_->set_mtype(EncodeMtype(kMtwiden4));
  matrix_state_->set_tile_m(2);
  matrix_state_->set_tile_k(1);
  vector_state_->set_vector_length(2);
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));

  ScopedTrapCapture trap_capture(state_.get());

  AppendMatmulSources(/*reg_a=*/8, /*reg_b=*/16, /*dest_tile=*/0);

  instruction_->set_semantic_function(&RiscVZvtVtfmmAlt);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(), *ExceptionCode::kIllegalInstruction);
}

}  // namespace
