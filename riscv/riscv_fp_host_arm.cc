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

#include "absl/log/log.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_fp_host.h"
#include "riscv/riscv_fp_info.h"
#include "riscv/riscv_fp_state.h"

// This file implements the arm versions of the methods/classes that interact
// with the host floating point hw.

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::operator*;  // NOLINT: is used below (clang error).

// arm flags.
constexpr uint8_t kArmIX = 16;  // Inexact result
constexpr uint8_t kArmUF = 8;   // Underflow
constexpr uint8_t kArmOF = 4;   // Overflow
constexpr uint8_t kArmDZ = 2;   // Divide by zero
constexpr uint8_t kArmIO = 1;   // Illegal operation

// RiscV flags.
constexpr uint8_t kNV = static_cast<uint8_t>(FPExceptions::kInvalidOp);
constexpr uint8_t kDZ = static_cast<uint8_t>(FPExceptions::kDivByZero);
constexpr uint8_t kOF = static_cast<uint8_t>(FPExceptions::kOverflow);
constexpr uint8_t kUF = static_cast<uint8_t>(FPExceptions::kUnderflow);
constexpr uint8_t kNX = static_cast<uint8_t>(FPExceptions::kInexact);

// This map converts the arm fp flags to RiscV fp flags. The arm layout is:
//     IX, UF, OF, DZ, IO (RiscV terminology:
//     NX, UF, OF, DZ, NV)
// The RiscV layout is:
//     IO, DZ, OF, UF, IX  (RiscV terminology:
//     NV, DZ, OF, UF, NX)
// The arm DE flag bit is not used.
constexpr uint8_t kArmFlagsToRiscvMap[32] = {
    0,
    kNV,
    kDZ,
    kDZ | kNV,
    kOF,
    kOF | kNV,
    kOF | kDZ,
    kOF | kDZ | kNV,
    kUF,
    kUF | kNV,
    kUF | kDZ,
    kUF | kDZ | kNV,
    kUF | kOF,
    kUF | kOF | kNV,
    kUF | kOF | kDZ,
    kUF | kOF | kDZ | kNV,
    kNX,
    kNX | kNV,
    kNX | kDZ,
    kNX | kDZ | kNV,
    kNX | kOF,
    kNX | kOF | kNV,
    kNX | kOF | kDZ,
    kNX | kOF | kDZ | kNV,
    kNX | kUF,
    kNX | kUF | kNV,
    kNX | kUF | kDZ,
    kNX | kUF | kDZ | kNV,
    kNX | kUF | kOF,
    kNX | kUF | kOF | kNV,
    kNX | kUF | kOF | kDZ,
    kNX | kUF | kOF | kDZ | kNV,
};

// This map converts RiscV flags to arm flags.
// The RiscV layout is IO, DZ, OF UF, IX
constexpr uint8_t kRiscVFlagsToArm[32]{
    0,
    kArmIX,
    kArmUF,
    kArmUF | kArmIX,
    kArmOF,
    kArmOF | kArmIX,
    kArmOF | kArmUF,
    kArmOF | kArmUF | kArmIX,
    kArmDZ,
    kArmDZ | kArmIX,
    kArmDZ | kArmUF,
    kArmDZ | kArmUF | kArmIX,
    kArmDZ | kArmOF,
    kArmDZ | kArmOF | kArmIX,
    kArmDZ | kArmOF | kArmUF,
    kArmDZ | kArmOF | kArmUF | kArmIX,
    kArmIO,
    kArmIO | kArmIX,
    kArmIO | kArmUF,
    kArmIO | kArmUF | kArmIX,
    kArmIO | kArmOF,
    kArmIO | kArmOF | kArmIX,
    kArmIO | kArmOF | kArmUF,
    kArmIO | kArmOF | kArmUF | kArmIX,
    kArmIO | kArmDZ,
    kArmIO | kArmDZ | kArmIX,
    kArmIO | kArmDZ | kArmUF,
    kArmIO | kArmDZ | kArmUF | kArmIX,
    kArmIO | kArmDZ | kArmOF,
    kArmIO | kArmDZ | kArmOF | kArmIX,
    kArmIO | kArmDZ | kArmOF | kArmUF,
    kArmIO | kArmDZ | kArmOF | kArmUF | kArmIX};

constexpr uint32_t kArmToNearest = 0b0000'0000'00'00'0000'0000'0000'0000'0000;
constexpr uint32_t kArmToNegInf = 0b0000'0000'10'00'0000'0000'0000'0000'0000;
constexpr uint32_t kArmToPosInf = 0b0000'0000'01'00'0000'0000'0000'0000'0000;
constexpr uint32_t kArmToZero = 0b0000'0000'11'00'0000'0000'0000'0000'0000;

constexpr uint32_t kRiscVToArmRoundingModeMap[8] = {
    kArmToNearest, kArmToZero,    kArmToNegInf,  kArmToPosInf,
    kArmToNearest, kArmToNearest, kArmToNearest, kArmToNearest,
};

// Map from arm rounding mode to RiscV rounding mode.
static constexpr FPRoundingMode kArmToRiscVRoundingModeMap[] = {
    FPRoundingMode::kRoundToNearest, FPRoundingMode::kRoundUp,
    FPRoundingMode::kRoundDown, FPRoundingMode::kRoundTowardsZero};

struct ArmFPRegs {
  uint64_t fpcr;
  uint64_t fpsr;
};

// This class provides the translations between RiscV and ARM floating point
// state. It also stores the ARM version of the simulated RISCV state
// between uses in simulated floating point instructions.
class ArmFloatingPointInterface : public HostFloatingPointInterface {
 public:
  ArmFloatingPointInterface() = default;
  ~ArmFloatingPointInterface() override = default;
  uint32_t GetRiscVFcsr() override {
    auto arm_rm = (arm_fp_regs_.fpcr >> 22) & 0x3;
    uint32_t rm = *kArmToRiscVRoundingModeMap[arm_rm];
    uint32_t flags = kArmFlagsToRiscvMap[arm_fp_regs_.fpsr & 0b1'1111];
    uint32_t value = (rm << 5) | (flags & 0x1f);
    return value;
  }

  void SetRiscVFcsr(uint32_t riscv_fcsr) override {
    uint32_t rm = (riscv_fcsr >> 5) & 0x7;
    uint64_t arm_rm = kRiscVToArmRoundingModeMap[rm] << 22;
    uint64_t flags = kRiscVFlagsToArm[riscv_fcsr & 0b1'1111];
    arm_fp_regs_.fpcr = arm_rm;
    arm_fp_regs_.fpsr = flags;
  }

  ArmFPRegs arm_fp_regs() const { return arm_fp_regs_; }
  void set_arm_fp_regs(ArmFPRegs arm_fp_regs) { arm_fp_regs_ = arm_fp_regs; }

 private:
  ArmFPRegs arm_fp_regs_ = {0, 0};
};

// Factory function.
HostFloatingPointInterface *GetHostFloatingPointInterface() {
  return new ArmFloatingPointInterface();
}

#pragma STDC FENV_ACCESS ON

ScopedFPStatus::ScopedFPStatus(HostFloatingPointInterface *fp_interface)
    : fp_interface_(fp_interface) {
  uint64_t fpsr, fpcr;
  asm volatile(
      "MRS %x0, FPSR\n"
      "MRS %x1, FPCR\n"
      : "=r"(fpsr), "=r"(fpcr));  // NOLINT(google3-runtime-inline-assembly

  // Pack fpsr and fpcr into cpu_fp_status
  cpu_fp_status_ = (fpsr << 32) | (fpcr & 0xffff'ffff);
  auto *host_fp_interface =
      static_cast<ArmFloatingPointInterface *>(fp_interface_);
  // Get the translated version of the simulated RiscV status.
  auto arm_fp_regs = host_fp_interface->arm_fp_regs();
  // Save current "dynamic" rounding mode.
  host_rm_ = arm_fp_regs.fpcr & kArmToZero;

  asm volatile(
      "MSR FPSR, %x0\n"
      "MSR FPCR, %x1\n"
      :
      : "r"(arm_fp_regs.fpsr),
        "r"(arm_fp_regs.fpcr));  // NOLINT(google3-runtime-inline-assembly)
}

ScopedFPStatus::ScopedFPStatus(HostFloatingPointInterface *fp_interface,
                               uint32_t riscv_rm)
    : fp_interface_(fp_interface) {
  uint64_t fpsr, fpcr;
  asm volatile(
      "MRS %x0, FPSR\n"
      "MRS %x1, FPCR\n"
      : "=r"(fpsr), "=r"(fpcr));  // NOLINT(google3-runtime-inline-assembly

  // Pack fpsr and fpcr into cpu_fp_status
  cpu_fp_status_ = (fpsr << 32) | (fpcr & 0xffff'ffff);
  auto *host_fp_interface =
      static_cast<ArmFloatingPointInterface *>(fp_interface_);
  // Get the translated version of the simulated RiscV status.
  auto arm_fp_regs = host_fp_interface->arm_fp_regs();
  // Save current "dynamic" rounding mode.
  host_rm_ = arm_fp_regs.fpcr & kArmToZero;
  if (riscv_rm != 0b111) {
    // Override rounding mode with that specified in the instruction.
    uint32_t new_arm_rm = kRiscVToArmRoundingModeMap[riscv_rm];
    arm_fp_regs.fpcr = new_arm_rm;
  }

  asm volatile(
      "MSR FPSR, %x0\n"
      "MSR FPCR, %x1\n"
      :
      : "r"(arm_fp_regs.fpsr),
        "r"(arm_fp_regs.fpcr));  // NOLINT(google3-runtime-inline-assembly)
}

ScopedFPStatus::ScopedFPStatus(HostFloatingPointInterface *fp_interface,
                               FPRoundingMode riscv_rm)
    : fp_interface_(fp_interface) {
  uint64_t fpsr, fpcr;
  asm volatile(
      "MRS %x0, FPSR\n"
      "MRS %x1, FPCR\n"
      : "=r"(fpsr), "=r"(fpcr));  // NOLINT(google3-runtime-inline-assembly

  // Pack fpsr and fpcr into cpu_fp_status
  cpu_fp_status_ = (fpsr << 32) | (fpcr & 0xffff'ffff);
  auto *host_fp_interface =
      static_cast<ArmFloatingPointInterface *>(fp_interface_);
  // Get the translated version of the simulated RiscV status.
  auto arm_fp_regs = host_fp_interface->arm_fp_regs();
  // Save current "dynamic" rounding mode.
  host_rm_ = arm_fp_regs.fpcr & kArmToZero;
  auto riscv_rm_value = static_cast<uint32_t>(riscv_rm);
  if (riscv_rm_value != 0b111) {
    // Override rounding mode with that specified in the instruction.
    uint32_t new_arm_rm = kRiscVToArmRoundingModeMap[riscv_rm_value];
    arm_fp_regs.fpcr = new_arm_rm;
  }

  asm volatile(
      "MSR FPSR, %x0\n"
      "MSR FPCR, %x1\n"
      :
      : "r"(arm_fp_regs.fpsr),
        "r"(arm_fp_regs.fpcr));  // NOLINT(google3-runtime-inline-assembly)
}

ScopedFPStatus::~ScopedFPStatus() {
  uint64_t fpsr, fpcr;
  asm volatile("MRS %x0, FPSR\n"
               : "=r"(fpsr));  // NOLINT(google3-runtime-inline-assembly

  // Save the simulated status.
  auto *host_fp_interface =
      static_cast<ArmFloatingPointInterface *>(fp_interface_);
  host_fp_interface->set_arm_fp_regs({host_rm_, fpsr});
  // Restore the saved host status.
  fpcr = cpu_fp_status_ & 0xffff'ffff;
  fpsr = cpu_fp_status_ >> 32;
  asm volatile(
      "MSR FPSR, %x0\n"
      "MSR FPCR, %x1\n"
      :
      : "r"(fpsr), "r"(fpcr));  // NOLINT(google3-runtime-inline-assembly)
}

ScopedFPRoundingMode::ScopedFPRoundingMode(
    HostFloatingPointInterface *fp_interface, uint32_t riscv_rm_value) {
  uint64_t fpsr, fpcr;
  asm volatile(
      "MRS %x0, FPSR\n"
      "MRS %x1, FPCR\n"
      : "=r"(fpsr), "=r"(fpcr));  // NOLINT(google3-runtime-inline-assembly

  // Pack fpsr and fpcr into cpu_fp_status
  cpu_fp_status_ = (fpsr << 32) | (fpcr & 0xffff'ffff);
  auto *host_fp_interface =
      static_cast<ArmFloatingPointInterface *>(fp_interface);
  // Get the translated version of the simulated RiscV status.
  auto arm_fp_regs = host_fp_interface->arm_fp_regs();
  if (riscv_rm_value != 0b111) {
    // Override rounding mode with that specified in the instruction.
    uint32_t new_arm_rm = kRiscVToArmRoundingModeMap[riscv_rm_value];
    arm_fp_regs.fpcr = new_arm_rm;
  }

  asm volatile(
      "MSR FPSR, %x0\n"
      "MSR FPCR, %x1\n"
      :
      : "r"(arm_fp_regs.fpsr),
        "r"(arm_fp_regs.fpcr));  // NOLINT(google3-runtime-inline-assembly)
}

ScopedFPRoundingMode::ScopedFPRoundingMode(
    HostFloatingPointInterface *fp_interface, FPRoundingMode rm) {
  uint64_t fpsr, fpcr;
  asm volatile(
      "MRS %x0, FPSR\n"
      "MRS %x1, FPCR\n"
      : "=r"(fpsr), "=r"(fpcr));  // NOLINT(google3-runtime-inline-assembly

  // Pack fpsr and fpcr into cpu_fp_status
  cpu_fp_status_ = (fpsr << 32) | (fpcr & 0xffff'ffff);
  auto *host_fp_interface =
      static_cast<ArmFloatingPointInterface *>(fp_interface);
  // Get the translated version of the simulated RiscV status.
  auto arm_fp_regs = host_fp_interface->arm_fp_regs();
  auto rm_value = static_cast<uint32_t>(rm);
  if (rm_value != 0b111) {
    // Override rounding mode with that specified in the instruction.
    uint32_t new_arm_rm = kRiscVToArmRoundingModeMap[rm_value];
    arm_fp_regs.fpcr = new_arm_rm;
  }

  asm volatile(
      "MSR FPSR, %x0\n"
      "MSR FPCR, %x1\n"
      :
      : "r"(arm_fp_regs.fpsr),
        "r"(arm_fp_regs.fpcr));  // NOLINT(google3-runtime-inline-assembly)
}

ScopedFPRoundingMode::~ScopedFPRoundingMode() {
  // No need to save the state, for RoundingMode we only needed the rounding
  // mode be set. Just restore the previous x86 fp status.
  uint64_t fpcr = cpu_fp_status_ & 0xffff'ffff;
  uint64_t fpsr = cpu_fp_status_ >> 32;
  asm volatile(
      "MSR FPSR, %x0\n"
      "MSR FPCR, %x1\n"
      :
      : "r"(fpsr), "r"(fpcr));  // NOLINT(google3-runtime-inline-assembly)
}

#pragma STDC FENV_ACCESS OFF

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
