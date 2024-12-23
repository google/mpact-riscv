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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_CLINT_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_CLINT_H_

#include <cstdint>

#include "mpact/sim/generic/counters.h"
#include "mpact/sim/generic/counters_base.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/ref_count.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "riscv/riscv_xip_xie.h"

namespace mpact {
namespace sim {
namespace riscv {

// This file contains the class definition of a simple implementation of the
// RiscV Core Level Interrupt controller.
//
// The interrupt controller occupies 0x1'0000 bytes in the memory space. Only
// the lower 16 bits of addresses passed to the clint are considered. It is
// the responsibility of the callers to ensure that the memory requests are
// properly routed according to the address.
//
// The interrupt controller models three memory mapped registers:
//     msip:     0xXXXX_0000 4B machine mode software interrupt.
//     mtimecmp: 0xXXXX_4000 8B machine mode timer compare register.
//     mtime:    0xXXXX_BFF8 8B machine mode timer register.
//
// The controller binds to a counter, for instance the instructions executed
// counter, using the CounterValueSetInterface<> interface. The update_counter_
// keeps track of how many times this is updated, and then increments the mtime
// register once for each 'period' updates. That is, the frequency of the mtime
// clock is 1/'period' of the associated counter. The controller implements the
// MemoryInterface to allow for memory-mapped loads and stores. Only the
// non-vector Load/Store methods are implemented.
//
// The controller only uses the low 16 bits of the address. It is assumed that
// any memory requests routed to the controller are done so correctly.

using ::mpact::sim::generic::CounterValueSetInterface;
using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::generic::ReferenceCount;
using ::mpact::sim::generic::SimpleCounter;
using ::mpact::sim::riscv::MipExternalWriteInterface;
using ::mpact::sim::util::MemoryInterface;

class RiscVClint : public CounterValueSetInterface<uint64_t>,
                   public MemoryInterface {
 public:
  RiscVClint(int period, MipExternalWriteInterface *mip_interface);
  RiscVClint() = delete;
  RiscVClint(const RiscVClint &) = delete;
  RiscVClint &operator=(const RiscVClint &) = delete;
  ~RiscVClint() override = default;
  // Resets the interrupt controller.
  void Reset();
  // CounterValueSetInterface override. This is called when the value of the
  // bound counter is modified.
  void SetValue(const uint64_t &val) override;

  // MemoryInterface overrides.
  // Non-vector load method.
  void Load(uint64_t address, DataBuffer *db, Instruction *inst,
            ReferenceCount *context) override;
  // Vector load method - this is stubbed out.
  void Load(DataBuffer *address_db, DataBuffer *mask_db, int el_size,
            DataBuffer *db, Instruction *inst,
            ReferenceCount *context) override;
  // Non-vector store method.
  void Store(uint64_t address, DataBuffer *db) override;
  // Vector store method - this is stubbed out.
  void Store(DataBuffer *address, DataBuffer *mask, int el_size,
             DataBuffer *db) override;

 private:
  // Helpers.
  uint32_t Read(uint32_t offset);
  void Write(uint32_t offset, uint32_t value);
  // Private methods to access the 32 bit portions of the registers.
  uint32_t ReadMTimeLow();
  uint32_t ReadMTimeHigh();
  void WriteMSip(uint32_t value);
  void WriteMTimeCmpLow(uint32_t value);
  void WriteMTimeCmpHigh(uint32_t value);
  void WriteMTimeLow(uint32_t value);
  void WriteMTimeHigh(uint32_t value);

  // The simulated register values.
  uint32_t msip_ = 0;
  uint64_t mtime_ = 0;
  uint64_t mtimecmp_ = 0;
  int mtip_ = 0;
  // mip write interface.
  MipExternalWriteInterface *mip_interface_;
  // Counter for how many updates there have been in current period.
  int update_counter_ = 0;
  int period_ = 0;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_CLINT_H_
