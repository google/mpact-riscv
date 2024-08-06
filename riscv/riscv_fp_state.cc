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

#include "riscv/riscv_fp_state.h"

#include <cstdint>

#include "absl/log/log.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_fp_host.h"
#include "riscv/riscv_fp_info.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::operator*;  // NOLINT: is used below (clang error).

constexpr char kFcsrName[] = "fcsr";
constexpr uint32_t kFcsrInitial = 0b000'0'0'0'0'0;
constexpr uint32_t kFcsrReadMask = 0b111'1'1111;
constexpr uint32_t kFcsrWriteMask = 0b111'1'1111;

constexpr char kFflagsName[] = "fflags";
constexpr uint32_t kFflagsInitial = 0b0'0000;
constexpr uint32_t kFflagsReadMask = 0b1'1111;
constexpr uint32_t kFflagsWriteMask = 0b1'1111;

constexpr char kFrmName[] = "frm";
constexpr uint32_t kFrmInitial = 0b000;
constexpr uint32_t kFrmReadMask = 0b111;
constexpr uint32_t kFrmWriteMask = 0b111;

// Helper function to avoid some extra code below.
static inline void LogIfError(absl::Status status) {
  if (status.ok()) return;
  LOG(ERROR) << status.message();
}

RiscVFPState::RiscVFPState(RiscVCsrSet *csr_set, ArchState *rv_state)
    : rv_state_(rv_state) {
  fcsr_ = new RiscVFcsr(this);
  frm_ = new RiscVFrm(this);
  fflags_ = new RiscVFflags(this);
  host_fp_interface_ = GetHostFloatingPointInterface();

  LogIfError(csr_set->AddCsr(fcsr_));
  LogIfError(csr_set->AddCsr(frm_));
  LogIfError(csr_set->AddCsr(fflags_));
}

RiscVFPState::~RiscVFPState() {
  delete fcsr_;
  delete frm_;
  delete fflags_;
  delete host_fp_interface_;
}

FPRoundingMode RiscVFPState::GetRoundingMode() const { return rounding_mode_; }

void RiscVFPState::SetRoundingMode(FPRoundingMode mode) {
  if (rounding_mode_ != mode) {
    switch (mode) {
      case FPRoundingMode::kRoundToNearestTiesToMax:
      case FPRoundingMode::kRoundToNearest:
        break;
      case FPRoundingMode::kRoundTowardsZero:
        break;
      case FPRoundingMode::kRoundDown:
        break;
      case FPRoundingMode::kRoundUp:
        break;
      default:
        rounding_mode_valid_ = false;
        LOG(ERROR) << "Illegal rounding mode: " << static_cast<int>(mode);
        return;
    }
    rounding_mode_ = mode;
    frm_->Write(*mode);
    rounding_mode_valid_ = true;
  }
}

// The RiscV fp csr.
RiscVFcsr::RiscVFcsr(RiscVFPState *fp_state)
    : RiscVSimpleCsr<uint32_t>(kFcsrName, RiscVCsrEnum::kFCsr, kFcsrInitial,
                               kFcsrReadMask, kFcsrWriteMask,
                               fp_state->rv_state()),
      fp_state_(fp_state) {}

// The status value is computed from the most recent x86 status value.
uint32_t RiscVFcsr::AsUint32() {
  uint32_t status_value = fp_state_->host_fp_interface()->GetRiscVFcsr();
  auto csr_value = GetUint32();
  auto value = ((csr_value & ~0x1f) | (status_value & 0x1f)) & read_mask();
  return value;
}

uint64_t RiscVFcsr::AsUint64() { return AsUint32(); }

// The status value is converted to x86 and stored for use in next fp
// instruction.
void RiscVFcsr::Write(uint32_t value) {
  auto wr_value = value & write_mask();
  Set(wr_value);
  fp_state_->host_fp_interface()->SetRiscVFcsr(wr_value);
  fp_state_->SetRoundingMode(
      static_cast<FPRoundingMode>((wr_value >> 5) & 0b111));
}

void RiscVFcsr::Write(uint64_t value) { Write(static_cast<uint32_t>(value)); }

// RiscVFflags translates reads and writes into reads and writes of fcsr.
RiscVFflags::RiscVFflags(RiscVFPState *fp_state)
    : RiscVSimpleCsr<uint32_t>(kFflagsName, RiscVCsrEnum::kFFlags,
                               kFflagsInitial, kFflagsReadMask,
                               kFflagsWriteMask, fp_state->rv_state()),
      fp_state_(fp_state) {}

uint32_t RiscVFflags::AsUint32() {
  uint32_t value = fp_state_->fcsr()->AsUint32();
  value &= 0b1'1111;
  return value;
}

void RiscVFflags::Write(uint32_t value) {
  uint32_t current = fp_state_->fcsr()->AsUint32();
  uint32_t new_value = (current & ~write_mask()) | (value & write_mask());
  if (new_value == current) return;
  fp_state_->fcsr()->Write(new_value);
}

uint32_t RiscVFflags::GetUint32() { return AsUint32(); }

void RiscVFflags::Set(uint32_t value) { Write(value); }

// RiscV rm (rounding mode) csr translates reads and writes into reads
// and writes of fcsr.
RiscVFrm::RiscVFrm(RiscVFPState *fp_state)
    : RiscVSimpleCsr<uint32_t>(kFrmName, RiscVCsrEnum::kFrm, kFrmInitial,
                               kFrmReadMask, kFrmWriteMask,
                               fp_state->rv_state()),
      fp_state_(fp_state) {}

uint32_t RiscVFrm::AsUint32() {
  uint32_t value = fp_state_->fcsr()->AsUint32();
  uint32_t rm = (value >> 5) & read_mask();
  return rm;
}

void RiscVFrm::Write(uint32_t value) {
  uint32_t wr_value = value & write_mask();
  uint32_t fcsr = fp_state_->fcsr()->AsUint32();
  uint32_t new_fcsr = ((fcsr & 0b1'1111) | (wr_value << 5));
  fp_state_->fcsr()->Write(new_fcsr);
}

uint32_t RiscVFrm::GetUint32() { return AsUint32(); }

void RiscVFrm::Set(uint32_t value) { Write(value); }

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
