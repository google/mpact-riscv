// Copyright 2024 Google LLC
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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_PMP_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_PMP_H_

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "mpact/sim/generic/arch_state.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_csr.h"

// This file defines the RiscV Physical Memory Protection management class. For
// now, this class only creates the CSRs and does not model the actual PMP
// functionality. In the future, these classes may be extended to model the
// actual PMP functionality.
//
// The CSRs are created internal to the class, and therefore are owned, and
// subsequently destructed by the destructor. This is done so that in the future
// it is easier to replace the CSRs with new PMP specific CSR classes that will
// be able to (re)configure PMP checking upon CSR writes.

namespace mpact::sim::riscv {

using ::mpact::sim::generic::operator*;  // NOLINT: used below (clang error).
using ::mpact::sim::generic::ArchState;

enum class PmpCfgBits {
  kRead = 1 << 0,
  kWrite = 1 << 1,
  kExecute = 1 << 2,
  kAtomic = 3 << 3,
  kNone = 3 << 5,
  kLock = 1 << 8,
};

class RiscVPmp {
 public:
  explicit RiscVPmp(ArchState* state) : state_(state) {}
  RiscVPmp(const RiscVPmp&) = delete;
  RiscVPmp& operator=(const RiscVPmp&) = delete;

  template <typename T, typename E>
  void CreatePmpCsrs(RiscVCsrSet* csr_set);

  virtual ~RiscVPmp() {
    for (auto* pmp_cfg : pmp_cfg_) delete pmp_cfg;
    for (auto* pmp_addr : pmp_addr_) delete pmp_addr;
  }

 private:
  ArchState* state_;
  RiscVCsrInterface* pmp_addr_[64] = {};
  RiscVCsrInterface* pmp_cfg_[16] = {};
};

template <typename T, typename E>
void RiscVPmp::CreatePmpCsrs(RiscVCsrSet* csr_set) {
  // Create the PMP configuration registers. Each configuration register
  // contains XLEN/8 configuration entries for a total of 64 configuration
  // entries. In the case of XLEN=64, only even-numbered configuration registers
  // are created.
  for (int i = 0; i < 16; ++i) {
    if (sizeof(T) == 8 && (i % 2 != 0)) {
      pmp_cfg_[i] = nullptr;
      continue;
    }
    pmp_cfg_[i] = new RiscVSimpleCsr<T>(absl::StrCat("pmpcfg", i),
                                        *E::kPmpCfg0 + i, state_);
    auto status = csr_set->AddCsr(pmp_cfg_[i]);
    if (!status.ok()) {
      LOG(ERROR) << absl::StrCat("Failed to add PMP configuration register: '",
                                 pmp_cfg_[i]->name(), "': ", status.message());
    }
  }
  // Create the 64 PMP address registers.
  for (int i = 0; i < 64; ++i) {
    pmp_addr_[i] = new RiscVSimpleCsr<T>(absl::StrCat("pmpaddr", i),
                                         *E::kPmpAddr0 + i, state_);
    auto status = csr_set->AddCsr(pmp_addr_[i]);
    if (!status.ok()) {
      LOG(ERROR) << absl::StrCat("Failed to add PMP address register: '",
                                 pmp_addr_[i]->name(), "': ", status.message());
    }
  }
}

}  // namespace mpact::sim::riscv

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_PMP_H_
