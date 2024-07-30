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

#include "riscv/riscv_pmp.h"

#include <cstdint>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_state.h"

namespace mpact::sim::riscv {

// Helper function used in constructors to create the PMP CSRs.
template <typename T>
void CreatePmpCsrs(RiscVState *state, RiscVPmp *pmp) {
  // Create the PMP configuration registers. Each configuration register
  // contains XLEN/8 configuration entries for a total of 16 configuration
  // entries. In the case of XLEN=64, there are only two configuration registers
  // pmp_cfg_[0] and pmp_cfg_[2].
  pmp->pmp_cfg_[0] =
      new RiscVSimpleCsr<T>("pmpcfg0", RiscVCsrEnum::kPmpCfg0, state);
  pmp->pmp_cfg_[1] =
      sizeof(T) == 8
          ? nullptr
          : new RiscVSimpleCsr<T>("pmpcfg1", RiscVCsrEnum::kPmpCfg1, state);
  pmp->pmp_cfg_[2] =
      new RiscVSimpleCsr<T>("pmpcfg2", RiscVCsrEnum::kPmpCfg2, state);
  pmp->pmp_cfg_[3] =
      sizeof(T) == 8
          ? nullptr
          : new RiscVSimpleCsr<T>("pmpcfg3", RiscVCsrEnum::kPmpCfg3, state);
  for (auto *pmp_cfg : pmp->pmp_cfg_) {
    if (pmp_cfg == nullptr) continue;
    auto status = state->csr_set()->AddCsr(pmp_cfg);
    if (!status.ok()) {
      LOG(ERROR) << absl::StrCat("Failed to add PMP configuration register: '",
                                 pmp_cfg->name(), "': ", status.message());
    }
  }
  // Create the 16 PMP address registers.
  pmp->pmp_addr_[0] =
      new RiscVSimpleCsr<T>("pmpaddr0", RiscVCsrEnum::kPmpAddr0, state);
  pmp->pmp_addr_[1] =
      new RiscVSimpleCsr<T>("pmpaddr1", RiscVCsrEnum::kPmpAddr1, state);
  pmp->pmp_addr_[2] =
      new RiscVSimpleCsr<T>("pmpaddr2", RiscVCsrEnum::kPmpAddr2, state);
  pmp->pmp_addr_[3] =
      new RiscVSimpleCsr<T>("pmpaddr3", RiscVCsrEnum::kPmpAddr3, state);
  pmp->pmp_addr_[4] =
      new RiscVSimpleCsr<T>("pmpaddr4", RiscVCsrEnum::kPmpAddr4, state);
  pmp->pmp_addr_[5] =
      new RiscVSimpleCsr<T>("pmpaddr5", RiscVCsrEnum::kPmpAddr5, state);
  pmp->pmp_addr_[6] =
      new RiscVSimpleCsr<T>("pmpaddr6", RiscVCsrEnum::kPmpAddr6, state);
  pmp->pmp_addr_[7] =
      new RiscVSimpleCsr<T>("pmpaddr7", RiscVCsrEnum::kPmpAddr7, state);
  pmp->pmp_addr_[8] =
      new RiscVSimpleCsr<T>("pmpaddr8", RiscVCsrEnum::kPmpAddr8, state);
  pmp->pmp_addr_[9] =
      new RiscVSimpleCsr<T>("pmpaddr9", RiscVCsrEnum::kPmpAddr9, state);
  pmp->pmp_addr_[10] =
      new RiscVSimpleCsr<T>("pmpaddr10", RiscVCsrEnum::kPmpAddr10, state);
  pmp->pmp_addr_[11] =
      new RiscVSimpleCsr<T>("pmpaddr11", RiscVCsrEnum::kPmpAddr11, state);
  pmp->pmp_addr_[12] =
      new RiscVSimpleCsr<T>("pmpaddr12", RiscVCsrEnum::kPmpAddr12, state);
  pmp->pmp_addr_[13] =
      new RiscVSimpleCsr<T>("pmpaddr13", RiscVCsrEnum::kPmpAddr13, state);
  pmp->pmp_addr_[14] =
      new RiscVSimpleCsr<T>("pmpaddr14", RiscVCsrEnum::kPmpAddr14, state);
  pmp->pmp_addr_[15] =
      new RiscVSimpleCsr<T>("pmpaddr15", RiscVCsrEnum::kPmpAddr15, state);
  for (auto *pmp_addr : pmp->pmp_addr_) {
    auto status = state->csr_set()->AddCsr(pmp_addr);
    if (!status.ok()) {
      LOG(ERROR) << absl::StrCat("Failed to add PMP address register: '",
                                 pmp_addr->name(), "': ", status.message());
    }
  }
}

RiscVPmp::RiscVPmp(RiscVState *state) : state_(state) {
  if (state_->xlen() == RiscVXlen::RV32) {
    CreatePmpCsrs<uint32_t>(state, this);
  } else if (state_->xlen() == RiscVXlen::RV64) {
    CreatePmpCsrs<uint64_t>(state, this);
  } else {
    LOG(FATAL) << "Illegal value for xlen";
  }
}

RiscVPmp::~RiscVPmp() {
  // Destruct the CSRs.
  for (auto *pmp_cfg : pmp_cfg_) {
    delete pmp_cfg;
  }
  for (auto *pmp_addr : pmp_addr_) {
    delete pmp_addr;
  }
}

}  // namespace mpact::sim::riscv
