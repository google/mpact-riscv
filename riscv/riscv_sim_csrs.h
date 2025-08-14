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

#ifndef MPACT_RISCV_RISCV_RISCV_SIM_CSRS_H_
#define MPACT_RISCV_RISCV_RISCV_SIM_CSRS_H_

#include <cstdint>
#include <string>

#include "riscv/riscv_csr.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

class RiscVSimModeCsr : public RiscVSimpleCsr<uint32_t> {
 public:
  RiscVSimModeCsr(std::string name, RiscVCsrEnum index, RiscVState* state)
      : RiscVSimpleCsr<uint32_t>(name, index, 0x0, 0x3, 0x3, state),
        state_(state) {}

  uint32_t GetUint32() override;
  uint64_t GetUint64() override;
  void Set(uint32_t value) override;
  void Set(uint64_t value) override;

 private:
  RiscVState* state_;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_SIM_CSRS_H_
