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

// Functional and regression unit tests for RISC-V Vector Matrix Extension
// (Zvt v0.4) instruction semantics, configuration, state transitions, and CSR
// dirty status updates.

#include "riscv/riscv_zvt_instructions.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/operand_interface.h"
#include "mpact/sim/generic/register.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_instruction_helpers.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_state.h"
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

using ::mpact::sim::generic::DestinationOperandInterface;
using ::mpact::sim::generic::ImmediateOperand;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::generic::RegisterBase;
using ::mpact::sim::generic::SourceOperandInterface;
using ::mpact::sim::riscv::DoubleToFloatROD;
using ::mpact::sim::riscv::ExceptionCode;
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
using ::mpact::sim::riscv::RiscVFPState;
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
using ::mpact::sim::riscv::RiscVZvtVtmvTv;
using ::mpact::sim::riscv::RiscVZvtVtmvVt;
using ::mpact::sim::riscv::RiscVZvtVtZero;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RV32VectorDestinationOperand;
using ::mpact::sim::riscv::RV32VectorSourceOperand;
using ::mpact::sim::riscv::RV64Register;
using ::mpact::sim::riscv::RVVectorRegister;
using ::mpact::sim::util::FlatDemandMemory;

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
inline constexpr uint32_t kTssPatternShift = 24;
inline constexpr uint32_t kTssTileShift = 27;

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
    state_->set_on_trap(
        [this](bool, uint64_t, uint64_t code, uint64_t, const Instruction*) {
          trap_taken_ = true;
          trap_code_ = code;
          return true;
        });
  }

  bool trap_taken() const { return trap_taken_; }
  uint64_t trap_code() const { return trap_code_; }

 private:
  RiscVState* state_;
  bool trap_taken_ = false;
  uint64_t trap_code_ = 0;
};

// Encapsulates a localized 64-bit RISC-V state, vector state, and matrix
// state along with mstatus status accessors for RV64-specific instruction
// tests.
struct Rv64TestContext {
  FlatDemandMemory memory;
  RiscVState state;
  RiscVVectorState vector_state;
  std::unique_ptr<RiscVZvtMatrixState> matrix_state;

  explicit Rv64TestContext(const std::string& name)
      : state(name, RiscVXlen::RV64, &memory), vector_state(&state, 128) {
    state.set_rv_vector(&vector_state);
    vector_state.SetVectorType(EncodeVtype(kVsew8, kVlmul1));
  }

  static absl::StatusOr<std::unique_ptr<Rv64TestContext>> Create(
      const std::string& name, int tile_elements = 32) {
    auto ctx = std::make_unique<Rv64TestContext>(name);
    auto ms_or = RiscVZvtMatrixState::Create(&ctx->state, tile_elements);
    if (!ms_or.ok()) return ms_or.status();
    ctx->matrix_state = std::move(ms_or.value());
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
  RiscVZvtInstructionsTest() {
    state_ = new RiscVState("test", RiscVXlen::RV32, &memory_);
    vector_state_ = new RiscVVectorState(state_, 128);  // 128 bytes
    state_->set_rv_vector(vector_state_);

    instruction_ = new Instruction(0x1000, state_);
    instruction_->set_size(4);
  }

  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(auto matrix_state,
                         RiscVZvtMatrixState::Create(state_, 32));
    matrix_state_ = matrix_state.release();
  }

  ~RiscVZvtInstructionsTest() override {
    instruction_->DecRef();
    delete matrix_state_;
    delete vector_state_;
    delete state_;
  }

  void ResetInstruction() {
    if (instruction_ != nullptr) {
      instruction_->DecRef();
    }
    instruction_ = new Instruction(0x1000, state_);
    instruction_->set_size(4);
  }

  // Sets the value of scalar register 'name' to 'value'.
  void SetRegisterValue(const std::string& name, uint32_t value) {
    auto* reg = state_->GetRegister<RV32Register>(name).first;
    reg->data_buffer()->Set<uint32_t>(0, value);
  }

  // Returns the value of scalar register 'name'.
  uint32_t GetRegisterValue(const std::string& name) const {
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

  // Creates a scalar source operand for register 'name'.
  SourceOperandInterface* CreateScalarSource(const std::string& name) {
    return state_->GetRegister<RV32Register>(name).first->CreateSourceOperand();
  }

  // Creates a scalar destination operand for register 'name'.
  DestinationOperandInterface* CreateScalarDest(const std::string& name) {
    return state_->GetRegister<RV32Register>(name)
        .first->CreateDestinationOperand(0);
  }

  // Creates a vector destination operand for vector register group 'reg_name'.
  RV32VectorDestinationOperand* CreateVectorDestOperand(
      const std::string& reg_name) {
    auto* reg = state_->GetRegister<RVVectorRegister>(reg_name).first;
    std::vector<RegisterBase*> group = {reg};
    return new RV32VectorDestinationOperand(absl::Span<RegisterBase*>(group), 0,
                                            reg_name);
  }

  // Creates a vector source operand for vector register group 'reg_name'.
  RV32VectorSourceOperand* CreateVectorSourceOperand(
      const std::string& reg_name) {
    auto* reg = state_->GetRegister<RVVectorRegister>(reg_name).first;
    std::vector<RegisterBase*> group = {reg};
    return new RV32VectorSourceOperand(absl::Span<RegisterBase*>(group),
                                       reg_name);
  }

  // Verifies that 'semantic_function' triggers an IllegalInstruction trap
  // when called on a state lacking matrix extension support.
  void ExpectTrapWithoutMatrixState(
      void (*semantic_function)(const Instruction*)) {
    FlatDemandMemory memory;
    RiscVState rv_state("test_no_matrix", RiscVXlen::RV32, &memory);
    ScopedTrapCapture trap_capture(&rv_state);

    auto* inst = new Instruction(0x1000, &rv_state);
    inst->set_size(4);
    inst->set_semantic_function(semantic_function);
    inst->Execute(nullptr);
    inst->DecRef();

    EXPECT_TRUE(trap_capture.trap_taken());
    EXPECT_EQ(trap_capture.trap_code(),
              static_cast<uint64_t>(ExceptionCode::kIllegalInstruction));
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
  RiscVState* state_;
  RiscVVectorState* vector_state_;
  RiscVZvtMatrixState* matrix_state_;
  Instruction* instruction_;
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
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(kMtwiden1));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(kVsew8));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(kAltfmtStandard));

  instruction_->set_semantic_function(&RiscVZvtMsetMtypei);
  instruction_->Execute(nullptr);

  // Semantics state vl = 0; mtype.mtwiden = 1
  EXPECT_EQ(matrix_state_->mtwiden(), kMtwiden1);
  EXPECT_EQ(state_->rv_vector()->vector_length(), 0);
  EXPECT_EQ(state_->rv_vector()->selected_element_width(), 1);
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
  EXPECT_EQ(state_->rv_vector()->vtype() & 0x7, kVlmul1);
  EXPECT_EQ(state_->rv_vector()->max_vector_length(), 32);
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

  // mtwiden = 0, but retain active TM in mtype to ensure msettm resets it.
  matrix_state_->set_mtype(
      EncodeMtype(kMtwidenUnconfigured, /*tk=*/0, /*tm=*/32));
  ASSERT_EQ(matrix_state_->tile_m(), 32);
  instruction_->Execute(nullptr);
  EXPECT_EQ(matrix_state_->tile_m(), 0);
  EXPECT_EQ(GetRegisterValue("x2"), 0);
  EXPECT_EQ(GetMstatusVs(), kMstatusStateDirty);
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

  // mtwiden = 0, but retain active TK in mtype to ensure msettk resets it.
  matrix_state_->set_mtype(
      EncodeMtype(kMtwidenUnconfigured, /*tk=*/4, /*tm=*/0));
  ASSERT_EQ(matrix_state_->tile_k(), 4);
  instruction_->Execute(nullptr);
  EXPECT_EQ(matrix_state_->tile_k(), 0);
  EXPECT_EQ(GetRegisterValue("x2"), 0);
  EXPECT_EQ(GetMstatusVs(), kMstatusStateDirty);
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
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(kMtwiden1));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(kVsew8));
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(kAltfmtStandard));
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
  auto rv64 = Rv64TestContext::Create("rv64_test");
  ASSERT_OK(rv64);

  // Clear VS and SD (bit 63 on RV64).
  (*rv64)->SetMstatusVs(kMstatusStateInitial);
  (*rv64)->state.mstatus()->Set((*rv64)->state.mstatus()->GetUint64() &
                                ~kMstatusSdMaskRv64);

  EXPECT_EQ((*rv64)->GetMstatusVs(), kMstatusStateInitial);
  EXPECT_EQ((*rv64)->GetMstatusSd(), kMstatusSdClean);

  auto* inst = new Instruction(0x1000, &(*rv64)->state);
  inst->set_size(4);
  inst->AppendSource(new ImmediateOperand<uint32_t>(kMtwiden1));
  inst->AppendSource(new ImmediateOperand<uint32_t>(kVsew8));
  inst->AppendSource(new ImmediateOperand<uint32_t>(kAltfmtStandard));
  inst->set_semantic_function(&RiscVZvtMsetMtypei);

  inst->Execute(nullptr);
  inst->DecRef();

  // Must set VS to Dirty (3) and SD (bit 63 on RV64) to 1.
  // If sd_bit was mutated to 31 on RV64, bit 63 is not set.
  EXPECT_EQ((*rv64)->GetMstatusVs(), kMstatusStateDirty);
  EXPECT_EQ((*rv64)->GetMstatusSd(), kMstatusSdSet);
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
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      ASSERT_OK(matrix_state_->SetElem<uint8_t>(0, r, c, 8, 0xAB));
      ASSERT_OK_AND_ASSIGN(auto elem,
                           matrix_state_->GetElem<uint8_t>(0, r, c, 8));
      EXPECT_EQ(elem, 0xAB);
    }
  }

  // Clear mstatus MS and VS.
  SetMstatusMs(kMstatusStateInitial);
  SetMstatusVs(kMstatusStateInitial);

  ResetInstruction();
  instruction_->AppendSource(new ImmediateOperand<uint32_t>(0));  // tile 0
  instruction_->set_semantic_function(&RiscVZvtVtZero);

  instruction_->Execute(nullptr);

  // All active elements in tile 0 must be 0.
  for (int r = 0; r < 4; ++r) {
    ASSERT_OK_AND_ASSIGN(auto row_elements,
                         GetTileRow<uint8_t>(/*tile=*/0, r, /*count=*/4));
    EXPECT_THAT(row_elements, testing::Each(testing::Eq(0)));
  }

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

  ScopedTrapCapture trap_capture(state_);

  ResetInstruction();
  instruction_->set_semantic_function(&RiscVZvtVtDiscard);
  instruction_->Execute(nullptr);

  EXPECT_TRUE(trap_capture.trap_taken());
  EXPECT_EQ(trap_capture.trap_code(),
            static_cast<uint64_t>(ExceptionCode::kIllegalInstruction));
}

TEST_F(RiscVZvtInstructionsTest, VtmvVtTransfersTileSliceToVectorRegister) {
  matrix_state_->set_mtype(EncodeMtype(kMtwiden1));
  vector_state_->SetVectorType(EncodeVtype(kVsew8, kVlmul1));
  matrix_state_->set_tile_m(4);
  vector_state_->set_vector_length(4);

  // Ensure MS is on (initial).
  SetMstatusMs(kMstatusStateInitial);

  // Populate row 1 of tile 0: elements at (0, 1, c) = c + 0x10.
  for (int c = 0; c < 4; ++c) {
    ASSERT_OK(matrix_state_->SetElem<uint8_t>(0, 1, c, 8,
                                              static_cast<uint8_t>(c + 0x10)));
  }

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
  for (int c = 0; c < 4; ++c) {
    ASSERT_OK(matrix_state_->SetElem<uint8_t>(0, 1, c, 8,
                                              static_cast<uint8_t>(c + 0x10)));
  }

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
  ASSERT_OK(matrix_state_->SetElem<uint8_t>(0, 2, 0, 8, 0xEE));

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
  // Under the mutant (if (false) instead of if (db->size == 8)),
  // db->Set<uint32_t>(0, value) is executed, leaving the upper 32 bits
  // untouched.
  auto rv64 = Rv64TestContext::Create("rv64_scalar_test");
  ASSERT_OK(rv64);

  // Turn on MS in mstatus so instructions don't trap.
  (*rv64)->SetMstatusMs(kMstatusStateInitial);

  // Initialize x1 register with high bits set.
  auto* reg1 = (*rv64)->state.GetRegister<RV64Register>("x1").first;
  reg1->data_buffer()->Set<uint64_t>(0, 0x1234'5678'9ABC'DEF0ULL);

  // Pre-seed a recycled 8-byte DataBuffer with non-zero upper bits to
  // guarantee that WriteScalar writes the full 64 bits and does not leave
  // upper bits intact.
  // Kills mutant: if (false) in WriteScalar.
  auto* recycled_db = (*rv64)->state.db_factory()->Allocate<uint64_t>(1);
  recycled_db->Set<uint64_t>(0, 0xDEAD'BEEF'CAFE'BABEULL);
  recycled_db->DecRef();

  auto* inst = new Instruction(0x1000, &(*rv64)->state);
  inst->set_size(4);
  inst->AppendSource(new ImmediateOperand<uint32_t>(4));  // requested col = 4
  inst->AppendDestination(reg1->CreateDestinationOperand(0));
  inst->set_semantic_function(&RiscVZvtMsetTn);

  inst->Execute(nullptr);
  inst->DecRef();

  // Entire 64 bits must be written (upper 32 bits zeroed).
  EXPECT_EQ(reg1->data_buffer()->Get<uint64_t>(0), 4ULL);
  EXPECT_EQ(reg1->data_buffer()->Get<uint64_t>(0) >> 32, 0ULL);
}

TEST_F(RiscVZvtInstructionsTest, SetMatrixStateDirtyRv64UpdatesMstatusMsAndSd) {
  auto rv64 = Rv64TestContext::Create("rv64_matrix_dirty_test");
  ASSERT_OK(rv64);

  // Set MS = Initial (1) so instructions don't trap, and clear SD (bit 63).
  (*rv64)->SetMstatusMs(kMstatusStateInitial);
  (*rv64)->state.mstatus()->Set((*rv64)->state.mstatus()->GetUint64() &
                                ~kMstatusSdMaskRv64);

  EXPECT_EQ((*rv64)->GetMstatusMs(), kMstatusStateInitial);
  EXPECT_EQ((*rv64)->GetMstatusSd(), kMstatusSdClean);

  auto* inst = new Instruction(0x1000, &(*rv64)->state);
  inst->set_size(4);
  inst->AppendSource(new ImmediateOperand<uint32_t>(0));  // tile 0
  inst->set_semantic_function(&RiscVZvtVtZero);

  inst->Execute(nullptr);
  inst->DecRef();

  // Must set MS to Dirty (3) and SD (bit 63 on RV64) to 1.
  // Kills mutant: sd_bit = (state->xlen() != RiscVXlen::RV32) ? 31 : 63.
  EXPECT_EQ((*rv64)->GetMstatusMs(), kMstatusStateDirty);
  EXPECT_EQ((*rv64)->GetMstatusSd(), kMstatusSdSet);
  uint64_t mstatus_val = (*rv64)->state.mstatus()->GetUint64();
  EXPECT_EQ((mstatus_val >> kMstatusSdShiftRv32) & 1ULL, kMstatusSdClean);
}

TEST_F(RiscVZvtInstructionsTest, DoubleToFloatRODInexactJamsLsb) {
  RiscVFPState fp_state(state_->csr_set(), state_);
  state_->set_rv_fp(&fp_state);

  // 1.0f in IEEE 754 is 0x3f800000 (even mantissa, LSB is 0).
  // 1.0 + 1e-12 in double precision rounds towards zero to 1.0f.
  // Inexact check triggers, and bit 0 is jammed to 1, producing 0x3f800001.
  // Kills mutant: f = absl::bit_cast<float>(f_bits); in DoubleToFloatROD.
  double inexact_d = 1.0 + 1e-12;
  float res = DoubleToFloatROD(inexact_d, &fp_state);
  EXPECT_EQ(absl::bit_cast<uint32_t>(res), 0x3f800001U);

  // Exact representation: 1.0 is exact in float, so LSB is not jammed.
  float exact_res = DoubleToFloatROD(1.0, &fp_state);
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
  // Kills mutant: SetVectorStateDirty(matrix_state); in RiscVZvtMsetMtypei.
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
  // Kills mutant: SetVectorStateDirty(matrix_state); in RiscVZvtMsetMtype.
  EXPECT_EQ(GetMstatusVs(), kMstatusStateDirty);
  EXPECT_EQ(GetMstatusSd(/*is_rv64=*/false), kMstatusSdSet);
  // UpdateVectorMatrixGeometry must NOT be called: no exception, vill=0.
  // Kills mutant: Changing line 443 `if (matrix_state->mtwiden() == 0)` to
  // `if (false)`.
  EXPECT_FALSE(state_->rv_vector()->vector_exception());
  EXPECT_EQ(state_->rv_vector()->vtype() & kVillMask, 0U);
  EXPECT_EQ(state_->rv_vector()->selected_element_width(), 2);
}

TEST_F(RiscVZvtInstructionsTest, MsetTnTrapsWhenMatrixStateMissing) {
  ExpectTrapWithoutMatrixState(&RiscVZvtMsetTn);
}

TEST_F(RiscVZvtInstructionsTest, MsetTmTrapsWhenMatrixStateMissing) {
  ExpectTrapWithoutMatrixState(&RiscVZvtMsetTm);
}

TEST_F(RiscVZvtInstructionsTest, MsetTkTrapsWhenMatrixStateMissing) {
  ExpectTrapWithoutMatrixState(&RiscVZvtMsetTk);
}

TEST_F(RiscVZvtInstructionsTest, MsetMtypeTrapsWhenMatrixStateMissing) {
  ExpectTrapWithoutMatrixState(&RiscVZvtMsetMtype);
}

TEST_F(RiscVZvtInstructionsTest, MsetMtypeiTrapsWhenMatrixStateMissing) {
  ExpectTrapWithoutMatrixState(&RiscVZvtMsetMtypei);
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
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      ASSERT_OK(matrix_state_->SetElem<uint8_t>(
          0, r, c, 8, static_cast<uint8_t>((r << 4) | c)));
    }
  }

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
  ASSERT_OK_AND_ASSIGN(auto elem_1_0,
                       matrix_state_->GetElem<uint8_t>(0, 1, 0, 8));
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

  // Kills mutants at lines 185, 187, 188.
  EXPECT_TRUE(state_->rv_vector()->vector_exception());
  EXPECT_EQ(state_->rv_vector()->vector_length(), 0);
  EXPECT_EQ(matrix_state_->mtype(), 0);
}

TEST_F(RiscVZvtInstructionsTest,
       MsetTmZeroResetsTileMAndSetsVsDirtyWhenMtwidenIsZero) {
  SetRegisterValue("x1", 500);
  instruction_->AppendSource(CreateScalarSource("x1"));
  instruction_->AppendDestination(CreateScalarDest("x2"));
  instruction_->set_semantic_function(&RiscVZvtMsetTm);

  // mtwiden = 0 with TM = 24 in mtype.
  matrix_state_->set_mtype(
      EncodeMtype(kMtwidenUnconfigured, /*tk=*/0, /*tm=*/24));
  ASSERT_EQ(matrix_state_->tile_m(), 24);

  // Clear VS in mstatus to Off (0) to verify line 484 sets it dirty.
  SetMstatusVs(kMstatusStateOff);
  ASSERT_EQ(GetMstatusVs(), kMstatusStateOff);

  instruction_->Execute(nullptr);

  // Kills mutant removing matrix_state->set_tile_m(0);
  EXPECT_EQ(matrix_state_->tile_m(), 0);
  EXPECT_EQ(GetRegisterValue("x2"), 0);
  // Kills mutant removing SetVectorStateDirty(matrix_state);
  EXPECT_EQ(GetMstatusVs(), kMstatusStateDirty);
}

TEST_F(RiscVZvtInstructionsTest,
       MsetTkZeroResetsTileKAndSetsVsDirtyWhenMtwidenIsZero) {
  SetRegisterValue("x1", 500);
  instruction_->AppendSource(CreateScalarSource("x1"));
  instruction_->AppendDestination(CreateScalarDest("x2"));
  instruction_->set_semantic_function(&RiscVZvtMsetTk);

  // mtwiden = 0 with TK = 4 in mtype.
  matrix_state_->set_mtype(
      EncodeMtype(kMtwidenUnconfigured, /*tk=*/4, /*tm=*/0));
  ASSERT_EQ(matrix_state_->tile_k(), 4);

  // Clear VS in mstatus to Off (0) to verify line 515 sets it dirty.
  SetMstatusVs(kMstatusStateOff);
  ASSERT_EQ(GetMstatusVs(), kMstatusStateOff);

  instruction_->Execute(nullptr);

  // Kills mutant removing matrix_state->set_tile_k(0);
  EXPECT_EQ(matrix_state_->tile_k(), 0);
  EXPECT_EQ(GetRegisterValue("x2"), 0);
  // Kills mutant removing SetVectorStateDirty(matrix_state);
  EXPECT_EQ(GetMstatusVs(), kMstatusStateDirty);
}

TEST_F(RiscVZvtInstructionsTest, VtmvVtRespectsVectorRegisterBoundary) {
  // Use a localized state with 8-byte vector registers (VLEN = 64 bits).
  // With SEW = 8 bits, elements_per_vector = 8 / 1 = 8.
  // Under the mutant (elements_per_vector - 1), it becomes 7.
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
  for (int c = 0; c < 8; ++c) {
    ASSERT_OK(matrix_state->SetElem<uint8_t>(0, 0, c, 8, 0x10 + c));
  }

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
  // Under mutant (- 1), elements_per_vector = 7.
  // Under mutant (+ 1), elements_per_vector = 9.
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
  ASSERT_OK_AND_ASSIGN(auto elem_7, matrix_state->GetElem<uint8_t>(0, 0, 7, 8));
  EXPECT_EQ(elem_7, 0x77);

  // For elem = 9:
  // Clean code: reg = 9 / 8 = 1, e = 9 % 8 = 1 -> reads v5[1] = 0x99.
  // Mutated code (+ 1): reg = 9 / 9 = 1, e = 9 % 9 = 0 -> reads v5[0] = 0x88.
  ASSERT_OK_AND_ASSIGN(auto elem_9, matrix_state->GetElem<uint8_t>(0, 0, 9, 8));
  EXPECT_EQ(elem_9, 0x99);
  EXPECT_EQ(vec_state.vstart(), 0);
}

}  // namespace
