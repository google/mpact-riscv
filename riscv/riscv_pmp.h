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

#include "riscv/riscv_csr.h"
#include "riscv/riscv_state.h"

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

class RiscVPmp;

enum class PmpCfgBits {
  kRead = 1 << 0,
  kWrite = 1 << 1,
  kExecute = 1 << 2,
  kAtomic = 3 << 3,
  kNone = 3 << 5,
  kLock = 1 << 8,
};

template <typename T>
void CreatePmpCsrs(RiscVState* state, RiscVPmp* pmp);

class RiscVPmp {
  friend void CreatePmpCsrs<uint32_t>(RiscVState* state, RiscVPmp* pmp);
  friend void CreatePmpCsrs<uint64_t>(RiscVState* state, RiscVPmp* pmp);

 public:
  RiscVPmp(RiscVState* state);
  RiscVPmp(const RiscVPmp&) = delete;
  RiscVPmp& operator=(const RiscVPmp&) = delete;
  virtual ~RiscVPmp();

 private:
  RiscVState* state_;
  RiscVCsrInterface* pmp_addr_[16];
  RiscVCsrInterface* pmp_cfg_[4];
};

}  // namespace mpact::sim::riscv

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_PMP_H_
