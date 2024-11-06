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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_JVT_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_JVT_H_

#include <cstdint>
#include <string>

#include "riscv/riscv_csr.h"
#include "riscv/riscv_state.h"

namespace mpact::sim::riscv {

template <typename T>
class RiscVJvtCsr : public RiscVSimpleCsr<T> {
 public:
  RiscVJvtCsr(std::string name, RiscVCsrEnum index, T initial_value,
              RiscVState *state)
      : RiscVSimpleCsr<T>(name, index, initial_value, state) {}
  // Clear the low 6 bits of the value (sets mode to 0b00'0000). This will be
  // modified as any new modes are added.
  void Write(uint32_t value) override { this->Set(value & ~0x3f); }
  void Write(uint64_t value) override {
    this->Set(static_cast<uint64_t>(value & ~0x3fULL));
  }
};

}  // namespace mpact::sim::riscv

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_JVT_H_
