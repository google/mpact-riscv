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

#include <cstdint>

#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_fp_host.h"
#include "riscv/riscv_fp_info.h"
#include "riscv/riscv_fp_state.h"

// This file implements the x86 versions of the methods/classes that interact
// with the host floating point hw.

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::operator*;  // NOLINT: is used below (clang error).

// x86 flags.
constexpr uint8_t kPE = 32;  // Inexact result
constexpr uint8_t kUE = 16;  // Underflow
constexpr uint8_t kOE = 8;   // Overflow
constexpr uint8_t kZE = 4;   // Divide by zero
constexpr uint8_t kIE = 1;   // Illegal operation

// RiscV flags.
constexpr uint8_t kNV = static_cast<uint8_t>(FPExceptions::kInvalidOp);
constexpr uint8_t kDZ = static_cast<uint8_t>(FPExceptions::kDivByZero);
constexpr uint8_t kOF = static_cast<uint8_t>(FPExceptions::kOverflow);
constexpr uint8_t kUF = static_cast<uint8_t>(FPExceptions::kUnderflow);
constexpr uint8_t kNX = static_cast<uint8_t>(FPExceptions::kInexact);

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
    kIE | kZE,
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

static constexpr uint32_t kX86ExceptionFlags = 0x1f80;

// This class provides the translations between RiscV and x86 floating point
// state. It also stores the x86 version of the simulated RISCV state
// between uses in simulated floating point instructions.
class X86FloatingPointInterface : public HostFloatingPointInterface {
 public:
  X86FloatingPointInterface() = default;
  ~X86FloatingPointInterface() override = default;

  uint32_t GetRiscVFcsr() override {
    auto x86_rm = (x86_status_ >> 13) & 0x3;
    uint32_t rm = *kX86ToRiscVRoundingModeMap[x86_rm];
    uint32_t flags = kX86FlagsToRiscvMap[x86_status_ & 0b11'1111];
    uint32_t value = (rm << 5) | (flags & 0x1f);
    return value;
  }

  void SetRiscVFcsr(uint32_t riscv_fcsr) override {
    uint32_t rm = (riscv_fcsr >> 5) & 0x7;
    uint32_t x86_rm = kRiscVToX86RoundingModeMap[rm];
    uint32_t flags = kRiscVFlagsToX86[riscv_fcsr & 0b1'1111];
    x86_status_ = kX86ExceptionFlags | x86_rm | flags;
  }

  uint32_t x86_status() const { return x86_status_; }
  void set_x86_status(uint32_t x86_status) {
    x86_status_ = kX86ExceptionFlags | x86_status;
  }

 private:
  uint32_t x86_status_ = kX86ExceptionFlags;
};

// Factory function.
HostFloatingPointInterface *GetHostFloatingPointInterface() {
  return new X86FloatingPointInterface();
}

#pragma STDC FENV_ACCESS ON

ScopedFPStatus::ScopedFPStatus(HostFloatingPointInterface *fp_interface)
    : fp_interface_(fp_interface) {
  // The host processor status is saved in cpu_fp_status_.
  uint32_t sim_status = 0;
  asm volatile("stmxcsr %0\n"  // NOLINT(google3-runtime-inline-assembly)
               : "=m"(cpu_fp_status_), "+X"(sim_status));
  auto *host_fp_interface =
      static_cast<X86FloatingPointInterface *>(fp_interface_);
  // Get the translated version of the simulated RiscV status.
  sim_status = host_fp_interface->x86_status();
  // Save current "dynamic" rounding mode.
  host_rm_ = sim_status & 0x6000;
  // Set the host status to the translated RiscV status.
  asm volatile("ldmxcsr %0\n"  // NOLINT(google3-runtime-inline-assembly)
               :
               : "m"(sim_status));
}

ScopedFPStatus::ScopedFPStatus(HostFloatingPointInterface *fp_interface,
                               uint32_t rm)
    : fp_interface_(fp_interface) {
  // The host processor status is saved in cpu_fp_status_.
  uint32_t sim_status = 0;
  asm volatile("stmxcsr %0\n"  // NOLINT(google3-runtime-inline-assembly)
               : "=m"(cpu_fp_status_), "=X"(sim_status));
  auto *host_fp_interface =
      static_cast<X86FloatingPointInterface *>(fp_interface_);
  sim_status = host_fp_interface->x86_status();
  // Save current "dynamic" rounding mode.
  host_rm_ = sim_status & 0x6000;
  if (rm != 0b111) {
    // Override rounding mode with that specified in the instruction.
    uint32_t new_x86_rm = kRiscVToX86RoundingModeMap[rm];
    sim_status = new_x86_rm | (sim_status & 0x1fff);
  }
  // Set the host status to the translated RiscV status.
  asm volatile("ldmxcsr %0\n"  // NOLINT(google3-runtime-inline-assembly)
               :
               : "m"(sim_status));
}

ScopedFPStatus::ScopedFPStatus(HostFloatingPointInterface *fp_interface,
                               FPRoundingMode rm)
    : fp_interface_(fp_interface) {
  // The host processor status is saved in cpu_fp_status_.
  uint32_t sim_status = 0;
  asm volatile("stmxcsr %0\n"  // NOLINT(google3-runtime-inline-assembly)
               : "=m"(cpu_fp_status_), "=X"(sim_status));
  auto *host_fp_interface =
      static_cast<X86FloatingPointInterface *>(fp_interface);
  sim_status = host_fp_interface->x86_status();
  // Save current "dynamic" rounding mode.
  host_rm_ = sim_status & 0x6000;
  auto rm_value = static_cast<uint32_t>(rm);
  if (rm_value != 0b111) {
    // Override rounding mode with that specified in the instruction.
    uint32_t new_x86_rm = kRiscVToX86RoundingModeMap[rm_value];
    sim_status = new_x86_rm | (sim_status & 0x1fff);
  }
  // Set the host status to the translated RiscV status.
  asm volatile("ldmxcsr %0\n"  // NOLINT(google3-runtime-inline-assembly)
               :
               : "m"(sim_status));
}

ScopedFPStatus::~ScopedFPStatus() {
  uint32_t x86_sim_status = 0;
  // Get the current x86 status.
  asm volatile(  // NOLINT(google3-runtime-inline-assembly)
      "stmxcsr %0\n"
      : "=m"(x86_sim_status));
  auto *host_fp_interface =
      static_cast<X86FloatingPointInterface *>(fp_interface_);
  // Save the fp status of the simulated instructions.
  uint32_t sim_status = ((x86_sim_status) & 0x1fff) | host_rm_;
  // Save the new status as the x86 version of the RiscV status.
  host_fp_interface->set_x86_status(x86_sim_status);
  // Restore host cpu status.
  asm volatile("ldmxcsr %0\n"  // NOLINT(google3-runtime-inline-assembly)
               :
               : "m"(cpu_fp_status_), "X"(sim_status));
}

ScopedFPRoundingMode::ScopedFPRoundingMode(
    HostFloatingPointInterface *fp_interface) {
  uint32_t sim_status = 0;
  // Get current x86 status and save it in cpu_fp_status_.
  asm volatile("stmxcsr %0\n"  // NOLINT(google3-runtime-inline-assembly)
               : "=m"(cpu_fp_status_), "=X"(sim_status));
  auto *host_fp_interface =
      static_cast<X86FloatingPointInterface *>(fp_interface);
  sim_status = host_fp_interface->x86_status();
  asm volatile("ldmxcsr %0\n"  // NOLINT(google3-runtime-inline-assembly)
               :
               : "m"(sim_status));
}

ScopedFPRoundingMode::ScopedFPRoundingMode(
    HostFloatingPointInterface *fp_interface, uint32_t rm) {
  uint32_t sim_status = 0;
  // Get current x86 status and save it in cpu_fp_status_.
  asm volatile("stmxcsr %0\n"  // NOLINT(google3-runtime-inline-assembly)
               : "=m"(cpu_fp_status_), "=X"(sim_status));
  auto *host_fp_interface =
      static_cast<X86FloatingPointInterface *>(fp_interface);
  sim_status = host_fp_interface->x86_status();
  if (rm != 0b111) {
    // Override rounding mode with that specified in the instruction.
    uint32_t new_x86_rm = kRiscVToX86RoundingModeMap[rm];
    sim_status = new_x86_rm | (sim_status & 0x1fff);
  }
  asm volatile("ldmxcsr %0\n"  // NOLINT(google3-runtime-inline-assembly)
               :
               : "m"(sim_status));
}

ScopedFPRoundingMode::ScopedFPRoundingMode(
    HostFloatingPointInterface *fp_interface, FPRoundingMode rm) {
  uint32_t sim_status = 0;
  // Get current x86 status and save it in cpu_fp_status_.
  asm volatile("stmxcsr %0\n"  // NOLINT(google3-runtime-inline-assembly)
               : "=m"(cpu_fp_status_), "=X"(sim_status));
  auto *host_fp_interface =
      static_cast<X86FloatingPointInterface *>(fp_interface);
  sim_status = host_fp_interface->x86_status();
  auto rm_value = static_cast<uint32_t>(rm);
  if (rm_value != 0b111) {
    // Override rounding mode with that specified in the instruction.
    uint32_t new_x86_rm = kRiscVToX86RoundingModeMap[rm_value];
    sim_status = new_x86_rm | (sim_status & 0x1fff);
  }
  asm volatile("ldmxcsr %0\n"  // NOLINT(google3-runtime-inline-assembly)
               :
               : "m"(sim_status));
}

ScopedFPRoundingMode::~ScopedFPRoundingMode() {
  // No need to save the state, for RoundingMode we only needed the rounding
  // mode be set. Just restore the previous x86 fp status.
  asm volatile(  // NOLINT(google3-runtime-inline-assembly)
      "ldmxcsr %0\n"
      :
      : "m"(cpu_fp_status_));
}

#pragma STDC FENV_ACCESS OFF

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
