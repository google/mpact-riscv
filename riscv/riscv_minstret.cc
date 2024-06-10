// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "riscv/riscv_minstret.h"

#include <cstdint>
#include <string>

#include "riscv/riscv_csr.h"
#include "riscv/riscv_state.h"

namespace mpact::sim::riscv {

using ::mpact::sim::riscv::RiscVCsrEnum;

RiscVMInstret::RiscVMInstret(std::string name, RiscVState *state)
    : RiscVSimpleCsr<uint32_t>(name, RiscVCsrEnum::kMInstret, state) {}

// Read the current value of the counter and apply the offset.
uint32_t RiscVMInstret::GetUint32() {
  if (counter_ == nullptr) return offset_;
  uint32_t value = GetCounterValue() + offset_;
  return value;
}

uint64_t RiscVMInstret::GetUint64() {
  return static_cast<uint64_t>(GetUint32());
}

void RiscVMInstret::Set(uint32_t value) { offset_ = value - GetCounterValue(); }

void RiscVMInstret::Set(uint64_t value) { Set(static_cast<uint32_t>(value)); }

RiscVMInstreth::RiscVMInstreth(std::string name, RiscVState *state)
    : RiscVSimpleCsr<uint32_t>(name, RiscVCsrEnum::kMInstretH, state) {}

// Read the current value of the counter and apply the offset.
uint32_t RiscVMInstreth::GetUint32() {
  if (counter_ == nullptr) return offset_;
  uint32_t value = GetCounterValue() + offset_;
  return value;
}

uint64_t RiscVMInstreth::GetUint64() {
  return static_cast<uint64_t>(GetUint32());
}

void RiscVMInstreth::Set(uint32_t value) {
  offset_ = value - GetCounterValue();
}

void RiscVMInstreth::Set(uint64_t value) { Set(static_cast<uint32_t>(value)); }

}  // namespace mpact::sim::riscv
