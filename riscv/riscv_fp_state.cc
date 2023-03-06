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
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

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

namespace internal {

// x86 flags.
constexpr uint8_t kPE = 32;  // Inexact result
constexpr uint8_t kUE = 16;  // Underflow
constexpr uint8_t kOE = 8;   // Overflow
constexpr uint8_t kZE = 4;   // Divide by zero
constexpr uint8_t kIE = 1;   // Illegal operation

// RiscV flags.
constexpr uint8_t kNV = 16;  // Invalid operation
constexpr uint8_t kDZ = 8;   // Divide by zero
constexpr uint8_t kOF = 4;   // Overflow
constexpr uint8_t kUF = 2;   // Underflow
constexpr uint8_t kNX = 1;   // Inexact result

// This map converts the x86 fp flags to RiscV fp flags. The x86 layout is:
// PE, UE, OE, ZE, DE, IE, whereas the RiscV layout is
//     IE, ZE, OE, UE, PE  (RiscV terminology: NV, DZ, OF, UF, NX)
// The x86 DE flag bit is not used.
constexpr uint8_t kX86FlagsToRiscvMap[64] = {0,
                                             kNV,
                                             0,
                                             kNV,
                                             kDZ,
                                             kDZ | kNV,
                                             kDZ,
                                             kNV | kDZ,
                                             kOF,
                                             kOF | kNV,
                                             kOF,
                                             kOF | kNV,
                                             kOF | kDZ,
                                             kOF | kDZ | kNV,
                                             kOF | kDZ,
                                             kOF | kDZ | kNV,
                                             kUF,
                                             kUF | kNV,
                                             kUF,
                                             kUF | kNV,
                                             kUF | kDZ,
                                             kUF | kDZ | kNV,
                                             kUF | kDZ,
                                             kUF | kDZ | kNV,
                                             kUF | kOF,
                                             kUF | kOF | kNV,
                                             kUF | kOF,
                                             kUF | kOF | kNV,
                                             kUF | kOF | kDZ,
                                             kUF | kOF | kDZ | kNV,
                                             kUF | kOF | kDZ,
                                             kUF | kOF | kDZ | kNV,
                                             kNX,
                                             kNX | kNV,
                                             kNX,
                                             kNX | kNV,
                                             kNX | kDZ,
                                             kNX | kDZ | kNV,
                                             kNX | kDZ,
                                             kNX | kNV | kDZ,
                                             kNX | kOF,
                                             kNX | kOF | kNV,
                                             kNX | kOF,
                                             kNX | kOF | kNV,
                                             kNX | kOF | kDZ,
                                             kNX | kOF | kDZ | kNV,
                                             kNX | kOF | kDZ,
                                             kNX | kOF | kDZ | kNV,
                                             kNX | kUF,
                                             kNX | kUF | kNV,
                                             kNX | kUF,
                                             kNX | kUF | kNV,
                                             kNX | kUF | kDZ,
                                             kNX | kUF | kDZ | kNV,
                                             kNX | kUF | kDZ,
                                             kNX | kUF | kDZ | kNV,
                                             kNX | kUF | kOF,
                                             kNX | kUF | kOF | kNV,
                                             kNX | kUF | kOF,
                                             kNX | kUF | kOF | kNV,
                                             kNX | kUF | kOF | kDZ,
                                             kNX | kUF | kOF | kDZ | kNV,
                                             kNX | kUF | kOF | kDZ,
                                             kNX | kUF | kOF | kDZ | kNV};

// This map converts RiscV flags to x86 flags.
// The RiscV layout is PE
constexpr uint8_t kRiscVFlagsToX86[32]{
    0,
    kPE,
    kUE,
    kUE | kPE,
    kOE,
    kOE | kPE,
    kOE | kUE,
    kOE | kUE | kPE,
    kZE,
    kZE | kPE,
    kZE | kUE,
    kZE | kUE | kPE,
    kZE | kOE,
    kZE | kOE | kPE,
    kZE | kOE | kUE,
    kZE | kOE | kUE | kPE,
    kIE,
    kIE | kPE,
    kIE | kUE,
    kIE | kUE | kPE,
    kIE | kOE,
    kIE | kOE | kPE,
    kIE | kOE | kUE,
    kIE | kOE | kUE | kPE,
    kIE | kZE | 0,
    kIE | kZE | kPE,
    kIE | kZE | kUE,
    kIE | kZE | kUE | kPE,
    kIE | kZE | kOE,
    kIE | kZE | kOE | kPE,
    kIE | kZE | kOE | kUE,
    kIE | kZE | kOE | kUE | kPE,
};

constexpr uint32_t kX86ToNearest = 0x0000;
constexpr uint32_t kX86ToNegInf = 0x2000;
constexpr uint32_t kX86ToPosInf = 0x4000;
constexpr uint32_t kX86ToZero = 0x6000;

constexpr uint32_t kRiscVToX86RoundingModeMap[8] = {
    kX86ToNearest, kX86ToZero,    kX86ToNegInf,  kX86ToPosInf,
    kX86ToNearest, kX86ToNearest, kX86ToNearest, kX86ToNearest,
};

// Map from x86 rounding mode to RiscV rounding mode.
static constexpr FPRoundingMode kX86ToRiscVRoundingModeMap[] = {
    FPRoundingMode::kRoundToNearest, FPRoundingMode::kRoundDown,
    FPRoundingMode::kRoundUp, FPRoundingMode::kRoundTowardsZero};

// Helper function to convert from x86 host fp status to RiscV fp status.
static uint32_t ConvertX86StatusToRiscVFcsr(uint32_t x86_status) {
  uint32_t x86_rm = (x86_status >> 13) & 0x3;
  uint32_t rm = *kX86ToRiscVRoundingModeMap[x86_rm];
  uint32_t flags = kX86FlagsToRiscvMap[x86_status & 0b11'1111];
  uint32_t value = (rm << 5) | (flags & 0x1f);
  return value;
}

// Helper function to convert from RiscV fp status to x86 host fp status.
static uint32_t ConvertRiscVFcsrToX86Status(uint32_t riscv_fcsr) {
  uint32_t rm = (riscv_fcsr >> 5) & 0x7;
  uint32_t x86_rm = kRiscVToX86RoundingModeMap[rm];
  uint32_t flags = kRiscVFlagsToX86[riscv_fcsr & 0b1'1111];
  uint32_t value = x86_rm | flags;
  return value;
}

}  // namespace internal

#pragma STDC FENV_ACCESS ON

ScopedFPStatus::ScopedFPStatus(RiscVFPState *fp_state) : fp_state_(fp_state) {
  uint32_t sim_status;
  asm volatile("stmxcsr %0\n" : "=m"(cpu_fp_status_), "=X"(sim_status));
  sim_status = fp_state_->sim_fp_status();
  // Save current "dynamic" rounding mode.
  x86_rm_ = sim_status & 0x6000;
  asm volatile("ldmxcsr %0\n" : : "m"(sim_status));
}

ScopedFPStatus::ScopedFPStatus(RiscVFPState *fp_state, uint32_t rm)
    : fp_state_(fp_state) {
  uint32_t sim_status;
  asm volatile("stmxcsr %0\n" : "=m"(cpu_fp_status_), "=X"(sim_status));
  sim_status = fp_state_->sim_fp_status();
  // Save current "dynamic" rounding mode.
  x86_rm_ = sim_status & 0x6000;
  if (rm != 0b111) {
    // Override rounding mode with that specified in the instruction.
    uint32_t new_x86_rm = internal::kRiscVToX86RoundingModeMap[rm];
    sim_status = new_x86_rm | (sim_status & 0x1fff);
  }
  asm volatile("ldmxcsr %0\n" : : "m"(sim_status));
}

ScopedFPStatus::ScopedFPStatus(RiscVFPState *fp_state, FPRoundingMode rm)
    : fp_state_(fp_state) {
  uint32_t sim_status;
  asm volatile("stmxcsr %0\n" : "=m"(cpu_fp_status_), "=X"(sim_status));
  sim_status = fp_state_->sim_fp_status();
  // Save current "dynamic" rounding mode.
  x86_rm_ = sim_status & 0x6000;
  auto rm_value = static_cast<uint32_t>(rm);
  if (rm_value != 0b111) {
    // Override rounding mode with that specified in the instruction.
    uint32_t new_x86_rm = internal::kRiscVToX86RoundingModeMap[rm_value];
    sim_status = new_x86_rm | (sim_status & 0x1fff);
  }
  asm volatile("ldmxcsr %0\n" : : "m"(sim_status));
}

ScopedFPStatus::~ScopedFPStatus() {
  uint32_t x86_sim_status;
  asm volatile("stmxcsr %0\n" : "=m"(x86_sim_status));
  // Restore "dynamic" rounding mode.
  uint32_t rv_sim_status = fp_state_->sim_fp_status();
  uint32_t sim_status = ((rv_sim_status | x86_sim_status) & 0x1fff) | x86_rm_;
  fp_state_->set_sim_fp_status(sim_status);

  asm volatile("ldmxcsr %0\n" : : "m"(cpu_fp_status_), "X"(sim_status));
}

ScopedFPRoundingMode::ScopedFPRoundingMode(RiscVFPState *fp_state) {
  uint32_t sim_status;
  asm volatile("stmxcsr %0\n" : "=m"(cpu_fp_status_), "=X"(sim_status));
  sim_status = fp_state->sim_fp_status();
  // Save current "dynamic" rounding mode.
  x86_rm_ = sim_status & 0x6000;
  asm volatile("ldmxcsr %0\n" : : "m"(sim_status));
}

ScopedFPRoundingMode::ScopedFPRoundingMode(RiscVFPState *fp_state,
                                           uint32_t rm) {
  uint32_t sim_status;
  asm volatile("stmxcsr %0\n" : "=m"(cpu_fp_status_), "=X"(sim_status));
  sim_status = fp_state->sim_fp_status();
  // Save current "dynamic" rounding mode.
  x86_rm_ = sim_status & 0x6000;
  if (rm != 0b111) {
    // Override rounding mode with that specified in the instruction.
    uint32_t new_x86_rm = internal::kRiscVToX86RoundingModeMap[rm];
    sim_status = new_x86_rm | (sim_status & 0x1fff);
  }
  asm volatile("ldmxcsr %0\n" : : "m"(sim_status));
}

ScopedFPRoundingMode::ScopedFPRoundingMode(RiscVFPState *fp_state,
                                           FPRoundingMode rm) {
  uint32_t sim_status;
  asm volatile("stmxcsr %0\n" : "=m"(cpu_fp_status_), "=X"(sim_status));
  sim_status = fp_state->sim_fp_status();
  // Save current "dynamic" rounding mode.
  x86_rm_ = sim_status & 0x6000;
  auto rm_value = static_cast<uint32_t>(rm);
  if (rm_value != 0b111) {
    // Override rounding mode with that specified in the instruction.
    uint32_t new_x86_rm = internal::kRiscVToX86RoundingModeMap[rm_value];
    sim_status = new_x86_rm | (sim_status & 0x1fff);
  }
  asm volatile("ldmxcsr %0\n" : : "m"(sim_status));
}

ScopedFPRoundingMode::~ScopedFPRoundingMode() {
  uint32_t x86_sim_status;
  asm volatile("stmxcsr %0\n" : "=m"(x86_sim_status));
  // Restore "dynamic" rounding mode.
  asm volatile("ldmxcsr %0\n" : : "m"(cpu_fp_status_), "X"(x86_sim_status));
}

#pragma STDC FENV_ACCESS OFF

// Helper function to avoid some extra code below.
static inline void LogIfError(absl::Status status) {
  if (status.ok()) return;
  LOG(ERROR) << status.message();
}

RiscVFPState::RiscVFPState(RiscVState *rv_state) : rv_state_(rv_state) {
  fcsr_ = new RiscVFcsr(this);
  frm_ = new RiscVFrm(this);
  fflags_ = new RiscVFflags(this);

  LogIfError(rv_state_->csr_set()->AddCsr(fcsr_));
  LogIfError(rv_state_->csr_set()->AddCsr(frm_));
  LogIfError(rv_state_->csr_set()->AddCsr(fflags_));
}

RiscVFPState::~RiscVFPState() {
  delete fcsr_;
  delete frm_;
  delete fflags_;
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
  uint32_t cpu_status = fp_state_->sim_fp_status();
  uint32_t status_value = internal::ConvertX86StatusToRiscVFcsr(cpu_status);
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
  uint32_t x86_value = internal::ConvertRiscVFcsrToX86Status(wr_value);
  fp_state_->set_sim_fp_status(x86_value);
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
