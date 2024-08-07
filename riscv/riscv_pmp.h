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
  RiscVCsrInterface* pmp_addr_[16];
  RiscVCsrInterface* pmp_cfg_[4];
};

template <typename T, typename E>
void RiscVPmp::CreatePmpCsrs(RiscVCsrSet* csr_set) {
  // Create the PMP configuration registers. Each configuration register
  // contains XLEN/8 configuration entries for a total of 16 configuration
  // entries. In the case of XLEN=64, there are only two configuration registers
  // pmp_cfg_[0] and pmp_cfg_[2].
  pmp_cfg_[0] = new RiscVSimpleCsr<T>("pmpcfg0", *E::kPmpCfg0, state_);
  pmp_cfg_[1] = sizeof(T) == 8
                    ? nullptr
                    : new RiscVSimpleCsr<T>("pmpcfg1", *E::kPmpCfg1, state_);
  pmp_cfg_[2] = new RiscVSimpleCsr<T>("pmpcfg2", *E::kPmpCfg2, state_);
  pmp_cfg_[3] = sizeof(T) == 8
                    ? nullptr
                    : new RiscVSimpleCsr<T>("pmpcfg3", *E::kPmpCfg3, state_);
  for (auto* pmp_cfg : pmp_cfg_) {
    if (pmp_cfg == nullptr) continue;
    auto status = csr_set->AddCsr(pmp_cfg);
    if (!status.ok()) {
      LOG(ERROR) << absl::StrCat("Failed to add PMP configuration register: '",
                                 pmp_cfg->name(), "': ", status.message());
    }
  }
  // Create the 16 PMP address registers.
  pmp_addr_[0] = new RiscVSimpleCsr<T>("pmpaddr0", *E::kPmpAddr0, state_);
  pmp_addr_[1] = new RiscVSimpleCsr<T>("pmpaddr1", *E::kPmpAddr1, state_);
  pmp_addr_[2] = new RiscVSimpleCsr<T>("pmpaddr2", *E::kPmpAddr2, state_);
  pmp_addr_[3] = new RiscVSimpleCsr<T>("pmpaddr3", *E::kPmpAddr3, state_);
  pmp_addr_[4] = new RiscVSimpleCsr<T>("pmpaddr4", *E::kPmpAddr4, state_);
  pmp_addr_[5] = new RiscVSimpleCsr<T>("pmpaddr5", *E::kPmpAddr5, state_);
  pmp_addr_[6] = new RiscVSimpleCsr<T>("pmpaddr6", *E::kPmpAddr6, state_);
  pmp_addr_[7] = new RiscVSimpleCsr<T>("pmpaddr7", *E::kPmpAddr7, state_);
  pmp_addr_[8] = new RiscVSimpleCsr<T>("pmpaddr8", *E::kPmpAddr8, state_);
  pmp_addr_[9] = new RiscVSimpleCsr<T>("pmpaddr9", *E::kPmpAddr9, state_);
  pmp_addr_[10] = new RiscVSimpleCsr<T>("pmpaddr10", *E::kPmpAddr10, state_);
  pmp_addr_[11] = new RiscVSimpleCsr<T>("pmpaddr11", *E::kPmpAddr11, state_);
  pmp_addr_[12] = new RiscVSimpleCsr<T>("pmpaddr12", *E::kPmpAddr12, state_);
  pmp_addr_[13] = new RiscVSimpleCsr<T>("pmpaddr13", *E::kPmpAddr13, state_);
  pmp_addr_[14] = new RiscVSimpleCsr<T>("pmpaddr14", *E::kPmpAddr14, state_);
  pmp_addr_[15] = new RiscVSimpleCsr<T>("pmpaddr15", *E::kPmpAddr15, state_);
  for (auto* pmp_addr : pmp_addr_) {
    auto status = csr_set->AddCsr(pmp_addr);
    if (!status.ok()) {
      LOG(ERROR) << absl::StrCat("Failed to add PMP address register: '",
                                 pmp_addr->name(), "': ", status.message());
    }
  }
}

}  // namespace mpact::sim::riscv

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_PMP_H_
