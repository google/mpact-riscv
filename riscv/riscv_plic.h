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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_PLIC_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_PLIC_H_

#include <cstdint>

#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/ref_count.h"
#include "mpact/sim/util/memory/memory_interface.h"

// This file implements the RiscV PLIC (Platform Level Interrupt Controller).
// It has a memory mapped register interface that is used to control the
// interrupts to one or more contexts across one or more RiscV cores.
//
// The interrupt targets (contexts) are notified using the
// RiscVPlicIrqInterface interface. Therefore each target much register its
// own instance of this interface with the plic using the SetContext method.
//
// Interrupt sources communicate with the PLIC using the SetInterrupt method.
//
// The PLIC supports both level and edge triggered interrupts.

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::generic::ReferenceCount;

using ::mpact::sim::util::MemoryInterface;

class RiscVPlicIrqInterface {
 public:
  virtual ~RiscVPlicIrqInterface() = default;
  virtual void SetIrq(bool irq_value) = 0;
};

class RiscVPlic : public MemoryInterface {
 public:
  // The constructor takes the number of interrupt sources and the number of
  // contexts that the PLIC can send interrupts to. A core usually has multiple
  // contexts, one for each privilege level (machine, supervisor, etc.) that is
  // capable of receiving and handling interrupts.
  RiscVPlic(int num_sources, int num_contexts);
  RiscVPlic() = delete;
  RiscVPlic(const RiscVPlic&) = delete;
  RiscVPlic& operator=(const RiscVPlic&) = delete;
  ~RiscVPlic() override;

  // Configure the PLIC state according to the source and context configuration
  // strings. The source configuration string is a semicolon separated list of
  // <source>=<priority> items, where <source> is the interrupt source number
  // and <priority> is an non-negative integer priority value. Higher values
  // have higher priorities. A value of zero disables the source. Any source
  // that is not configured is disabled by default. However, MMR writes can
  // change the configured values.
  //
  // The context configuration string is a semicolon separated list of
  // <context>=<threshold>,(<source>,<enable>)+ items, where <context> is the
  // context number, <source> is the source number, <threshold> is a non
  // negative integer that is the priority threshold for that context, and
  // <enable> is a 0 or 1 value that indicates whether the interrupt source is
  // enabled for that context. Multiple lines for the same context with
  // different sources is allowed. Any context not configured is assumed to
  // disable all sources with a zero priority threshold.
  absl::Status Configure(absl::string_view source_cfg,
                         absl::string_view context_cfg);
  // Interrupt request from the given interrupt source.
  void SetInterrupt(int source, bool value, bool is_level);

  // MemoryInterface overrides.
  // Non-vector load method.
  void Load(uint64_t address, DataBuffer* db, Instruction* inst,
            ReferenceCount* context) override;
  // Vector load method - this is stubbed out.
  void Load(DataBuffer* address_db, DataBuffer* mask_db, int el_size,
            DataBuffer* db, Instruction* inst,
            ReferenceCount* context) override;
  // Non-vector store method.
  void Store(uint64_t address, DataBuffer* db) override;
  // Vector store method - this is stubbed out.
  void Store(DataBuffer* address, DataBuffer* mask, int el_size,
             DataBuffer* db) override;

  void SetContext(int context_no, RiscVPlicIrqInterface* context_if);

 private:
  struct GatewayInfo {
    // The gateway is able to accept the interrupt and send it to the plic core.
    bool ready;
    // Pending bit in gateway. Only gets set if the interrupt is level based.
    bool pending;
    GatewayInfo() : ready(true), pending(false) {}
  };

  // MMR read/write methods.
  uint32_t Read(uint32_t offset);
  void Write(uint32_t offset, uint32_t value);
  // Interrupt claim.
  uint32_t ClaimInterrupt(int context);
  // Signal interrupt completion for the given context and interrupt id.
  void CompleteInterrupt(int context, uint32_t id);
  // Set plic core pending interrupt bit and trigger interrupt to context as
  // needed.
  void SetPlicPendingInterrupt(int source);
  // Handling pending bits.
  void SetPending(int source, bool value);
  bool IsPending(int source);

  int num_sources_;
  int num_contexts_;
  // Interface to call to write the IRQ line for a context.
  RiscVPlicIrqInterface** context_if_;
  // Last value written to the IRQ line for a context.
  bool* context_irq_ = nullptr;
  // Source gateway info.
  GatewayInfo* gateway_info_ = nullptr;
  // Interrupt priorities by source.
  uint32_t* interrupt_priority_ = nullptr;
  // Pending interrupts by source - 32 bits per word.
  uint32_t* interrupt_pending_ = nullptr;
  // Enable bits per context per source - 32 bits per word.
  // Array is organized as interrupt_enabled_[context][source / 32].
  uint32_t** interrupt_enabled_ = nullptr;
  // Priority threshold by context.
  uint32_t* priority_threshold_ = nullptr;
  // Interrupt claim/complete register by context.
  uint32_t* interrupt_claim_complete_ = nullptr;
  // Map from source to context that has the source enabled. This must be
  // updated whenever an an enable bit is changed for a context.
  absl::btree_multimap<int, int> source_to_context_;
  // Map from context to source for which the context has the source enabled.
  absl::btree_multimap<int, int> context_to_source_;
};

class RiscVPlicSourceInterface : public RiscVPlicIrqInterface {
 public:
  RiscVPlicSourceInterface(RiscVPlic* plic, int source, bool is_level);
  RiscVPlicSourceInterface() = delete;
  ~RiscVPlicSourceInterface() override = default;
  void SetIrq(bool irq_value) override {
    if (plic_ != nullptr) plic_->SetInterrupt(source_, irq_value, is_level_);
  };

 private:
  RiscVPlic* plic_ = nullptr;
  int source_ = 0;
  bool is_level_ = false;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_PLIC_H_
