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

#ifndef MPACT_RISCV_RISCV_RISCV_FP_HOST_H_
#define MPACT_RISCV_RISCV_RISCV_FP_HOST_H_

#include <cstdint>

#include "riscv/riscv_fp_info.h"

namespace mpact {
namespace sim {
namespace riscv {

class RiscVFPState;

class HostFloatingPointInterface {
 public:
  virtual ~HostFloatingPointInterface() = default;
  virtual uint32_t GetRiscVFcsr() = 0;
  virtual void SetRiscVFcsr(uint32_t riscv_fcsr) = 0;
};

HostFloatingPointInterface *GetHostFloatingPointInterface();

// This class is used to set the native fp status and rounding mode to the
// current simulated status and rounding mode, and then copying the native
// status back to the simulated status using RAII.
class ScopedFPStatus {
 public:
  ScopedFPStatus() = delete;
  explicit ScopedFPStatus(HostFloatingPointInterface *fp_interface);
  ScopedFPStatus(HostFloatingPointInterface *fp_interface, uint32_t rm);
  ScopedFPStatus(HostFloatingPointInterface *fp_interface, FPRoundingMode rm);
  ~ScopedFPStatus();

 private:
  HostFloatingPointInterface *fp_interface_ = nullptr;
  uint32_t host_rm_ = 0;
  uint64_t cpu_fp_status_ = 0;
};

// This class is used to set the native fp rounding mode without affecting the
// current simulated status (fp flags).
class ScopedFPRoundingMode {
 public:
  ScopedFPRoundingMode();
  explicit ScopedFPRoundingMode(HostFloatingPointInterface *fp_interface);
  ScopedFPRoundingMode(HostFloatingPointInterface *fp_interface, uint32_t rm);
  ScopedFPRoundingMode(HostFloatingPointInterface *fp_interface,
                       FPRoundingMode rm);
  ~ScopedFPRoundingMode();

 private:
  uint64_t cpu_fp_status_ = 0;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_FP_HOST_H_
