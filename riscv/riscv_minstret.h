/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_MINSTRET_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_MINSTRET_H_

#include <cstdint>
#include <string>

#include "mpact/sim/generic/counters.h"
#include "riscv/riscv_csr.h"

// This file provides the declarations for the CherIoT minstret/minstreth
// CSRs. They are tied to the instruction counter of the top level of the
// simulator. That binding is done when the simulator is instantiated. Until
// that is done, the CSR just works like a scratch CSR.

// Since this CSR is both readable and writable, but the counter value cannot
// be changed, every time the register is written, a relative offset is computed
// from the counter, so that the values read are relative to the most recent
// write of the CSR.

namespace mpact::sim::riscv {

using ::mpact::sim::generic::SimpleCounter;
using ::mpact::sim::riscv::RiscVSimpleCsr;

class RiscVState;

class RiscVMInstret : public RiscVSimpleCsr<uint32_t> {
 public:
  RiscVMInstret(std::string name, RiscVState* state);
  RiscVMInstret(const RiscVMInstret&) = delete;
  RiscVMInstret& operator=(const RiscVMInstret&) = delete;
  ~RiscVMInstret() override = default;

  // RiscVSimpleCsr method overrides.
  uint32_t GetUint32() override;
  uint64_t GetUint64() override;

  void Set(uint32_t) override;
  void Set(uint64_t) override;

  void set_counter(SimpleCounter<uint64_t>* counter) { counter_ = counter; }

 private:
  inline uint32_t GetCounterValue() const {
    if (counter_ == nullptr) return 0;
    return static_cast<uint32_t>(counter_->GetValue() & 0xffff'ffffULL);
  };

  SimpleCounter<uint64_t>* counter_ = nullptr;
  uint32_t offset_ = 0;
};

class RiscVMInstreth : public RiscVSimpleCsr<uint32_t> {
 public:
  RiscVMInstreth(std::string name, RiscVState* state);
  RiscVMInstreth(const RiscVMInstret&) = delete;
  RiscVMInstreth& operator=(const RiscVMInstret&) = delete;
  ~RiscVMInstreth() override = default;

  // RiscVSimpleCsr method overrides.
  uint32_t GetUint32() override;
  uint64_t GetUint64() override;

  void Set(uint32_t) override;
  void Set(uint64_t) override;

  void set_counter(SimpleCounter<uint64_t>* counter) { counter_ = counter; }

 private:
  inline uint32_t GetCounterValue() const {
    if (counter_ == nullptr) return 0;
    return static_cast<uint32_t>(counter_->GetValue() >> 32);
  };

  SimpleCounter<uint64_t>* counter_ = nullptr;
  uint32_t offset_ = 0;
};

}  // namespace mpact::sim::riscv

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_MINSTRET_H_
