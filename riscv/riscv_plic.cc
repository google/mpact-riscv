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

#include "riscv/riscv_plic.h"

#include <cstdint>
#include <cstring>

#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "re2/re2.h"

namespace mpact {
namespace sim {
namespace riscv {

RiscVPlic::RiscVPlic(int num_sources, int num_contexts)
    : num_sources_(num_sources), num_contexts_(num_contexts) {
  // Initialize the gateway info.
  gateway_info_ = new GatewayInfo[num_sources_];
  // Initialize the context interface.
  context_if_ = new RiscVPlicIrqInterface*[num_contexts_];
  context_irq_ = new bool[num_contexts_];
  for (int i = 0; i < num_contexts_; ++i) {
    context_if_[i] = nullptr;
    context_irq_[i] = false;
  }
  // Initialize the interrupt priority.
  interrupt_priority_ = new uint32_t[num_sources_];
  std::memset(interrupt_priority_, 0, sizeof(uint32_t) * num_sources_);
  // Initialize the interrupt pending bits.
  interrupt_pending_ = new uint32_t[num_sources_ / 32 + 1];
  std::memset(interrupt_pending_, 0,
              sizeof(uint32_t) * (num_sources_ / 32 + 1));
  // Initialize the interrupt enabled bits.
  interrupt_enabled_ = new uint32_t*[num_contexts_];
  for (int i = 0; i < num_contexts_; ++i) {
    interrupt_enabled_[i] = new uint32_t[num_sources_ / 32 + 1];
    std::memset(interrupt_enabled_[i], 0,
                sizeof(uint32_t) * (num_sources_ / 32 + 1));
  }
  // Initialize the priority threshold.
  priority_threshold_ = new uint32_t[num_contexts_];
  std::memset(priority_threshold_, 0, sizeof(uint32_t) * num_contexts_);
  // Initialize the interrupt claim/complete bits.
  interrupt_claim_complete_ = new uint32_t[num_contexts_];
  std::memset(interrupt_claim_complete_, 0, sizeof(uint32_t) * num_contexts_);
}

RiscVPlic::~RiscVPlic() {
  // Clean up all the allocated memory.
  delete[] gateway_info_;
  gateway_info_ = nullptr;
  delete[] context_if_;
  context_if_ = nullptr;
  delete[] context_irq_;
  context_irq_ = nullptr;
  delete[] interrupt_priority_;
  interrupt_priority_ = nullptr;
  delete[] interrupt_pending_;
  interrupt_pending_ = nullptr;
  for (int i = 0; i < num_contexts_; ++i) {
    delete[] interrupt_enabled_[i];
  }
  delete[] interrupt_enabled_;
  interrupt_enabled_ = nullptr;
  delete[] priority_threshold_;
  priority_threshold_ = nullptr;
  delete[] interrupt_claim_complete_;
  interrupt_claim_complete_ = nullptr;
}

absl::Status RiscVPlic::Configure(absl::string_view source_cfg,
                                  absl::string_view context_cfg) {
  // List of "<source>=<priority>;" items.
  RE2 re_source("^(\\d+)\\s*=\\s*(\\d+)\\s*(;|$)");
  int source;
  int priority;
  absl::string_view cfg = source_cfg;
  while (RE2::Consume(&cfg, re_source, &source, &priority)) {
    if (source >= num_sources_) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid source number: ", source));
    }
    interrupt_priority_[source] = priority;
  }
  if (!cfg.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid source configuration: ", cfg));
  }
  // List of "<context>=<source>,<threshold>,<enable>;" items.
  RE2 re_context("^(\\d+)\\s*=\\s*(\\d+)\\s*,\\s*");
  RE2 re_context_source("(\\d+)\\s*,\\s*(\\d)\\s*(,|;)\\s*");
  int context;
  int threshold;
  int enable;
  char terminator;
  cfg = context_cfg;
  while (RE2::Consume(&cfg, re_context, &context, &threshold)) {
    if (context >= num_contexts_) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid context number: ", context));
    }
    priority_threshold_[context] = threshold;
    while (
        RE2::Consume(&cfg, re_context_source, &source, &enable, &terminator)) {
      if (source >= num_sources_) {
        return absl::InvalidArgumentError(
            absl::StrCat("Invalid source number: ", source));
      }
      source_to_context_.insert({source, context});
      context_to_source_.insert({context, source});
      int bit = source & 0x1f;
      int word = source / 32;
      uint32_t mask = ~(1 << bit);
      uint32_t value = interrupt_enabled_[context][word];
      value = (value & mask) | (enable << bit);
      interrupt_enabled_[context][word] = value;
      if (terminator == ';') break;
    }
  }
  if (!cfg.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid context configuration: ", cfg));
  }
  return absl::OkStatus();
}

void RiscVPlic::SetInterrupt(int source, bool value, bool is_level) {
  // Make sure the source is in range.
  if ((source < 0) || (source >= num_sources_)) {
    LOG(WARNING) << "Invalid interrupt source: " << source;
    return;
  }
  // No action for clearing a non-level based interrupt.
  if (!value && !is_level) return;

  auto& info = gateway_info_[source];
  if (!info.ready || !value) {
    if (is_level) info.pending = value;
    return;
  }

  // If it is level based, latch the pending bit in the gateway.
  if (is_level) info.pending = true;

  // If the priority is 0, the interrupt is disabled.
  if (interrupt_priority_[source] == 0) return;

  // Accept the request, set ready to false to prevent any other requests
  // until this has been processed.
  info.ready = false;
  // Set the plic pending bit.
  SetPlicPendingInterrupt(source);
}

void RiscVPlic::SetPlicPendingInterrupt(int source) {
  if ((source <= 0) || (source >= num_sources_)) {
    LOG(ERROR) << "Invalid interrupt source: " << source;
    return;
  }
  // Return if source is already pending.
  if (IsPending(source)) return;
  // Get interrupt priority.
  auto priority = interrupt_priority_[source];
  // Return if source has priority 0.
  if (priority == 0) return;
  // Set source to pending.
  SetPending(source, true);
  // Iterate over all contexts that have this source enabled.
  auto [begin, end] = source_to_context_.equal_range(source);
  for (auto it = begin; it != end; ++it) {
    auto context = it->second;
    // If the priority is less or equal to the threshold, do not trigger the
    // interrupt.
    if (priority <= priority_threshold_[context]) continue;
    if (context_if_[context] == nullptr) continue;
    // Trigger the interrupt.
    context_if_[context]->SetIrq(true);
    context_irq_[context] = true;
  }
}

// Implementation of the memory load interface for reading memory mapped
// registers.
void RiscVPlic::Load(uint64_t address, DataBuffer* db, Instruction* inst,
                     ReferenceCount* context) {
  uint32_t offset = address & 0xff'ffff;
  switch (db->size<uint8_t>()) {
    case 1:
      db->Set<uint8_t>(0, static_cast<uint8_t>(Read(offset)));
      break;
    case 2:
      db->Set<uint16_t>(0, static_cast<uint16_t>(Read(offset)));
      break;
    case 4:
      db->Set<uint32_t>(0, static_cast<uint32_t>(Read(offset)));
      break;
    case 8:
      db->Set<uint32_t>(0, static_cast<uint32_t>(Read(offset)));
      db->Set<uint32_t>(1, static_cast<uint32_t>(Read(offset + 4)));
      break;
    default:
      ::memset(db->raw_ptr(), 0, sizeof(db->size<uint8_t>()));
      break;
  }
  // Execute the instruction to process and write back the load data.
  if (nullptr != inst) {
    if (db->latency() > 0) {
      inst->IncRef();
      if (context != nullptr) context->IncRef();
      inst->state()->function_delay_line()->Add(db->latency(),
                                                [inst, context]() {
                                                  inst->Execute(context);
                                                  if (context != nullptr)
                                                    context->DecRef();
                                                  inst->DecRef();
                                                });
    } else {
      inst->Execute(context);
    }
  }
}

// No support for vector loads.
void RiscVPlic::Load(DataBuffer* address_db, DataBuffer* mask_db, int el_size,
                     DataBuffer* db, Instruction* inst,
                     ReferenceCount* context) {
  LOG(FATAL) << "RiscVPlic does not support vector loads";
}

// Implementation of memory store interface to support writes to memory mapped
// registers.
void RiscVPlic::Store(uint64_t address, DataBuffer* db) {
  uint32_t offset = address & 0xff'ffff;
  switch (db->size<uint8_t>()) {
    case 1:
      return Write(offset, static_cast<uint32_t>(db->Get<uint8_t>(0)));
    case 2:
      return Write(offset, static_cast<uint32_t>(db->Get<uint16_t>(0)));
    case 4:
      return Write(offset, static_cast<uint32_t>(db->Get<uint32_t>(0)));
    case 8:
      return Write(offset, static_cast<uint32_t>(db->Get<uint32_t>(0)));
      return Write(offset + 4, static_cast<uint32_t>(db->Get<uint32_t>(1)));
    default:
      return;
  }
}

void RiscVPlic::SetContext(int context_no, RiscVPlicIrqInterface* context_if) {
  context_if_[context_no] = context_if;
}

// No support for vector stores.
void RiscVPlic::Store(DataBuffer* address, DataBuffer* mask, int el_size,
                      DataBuffer* db) {
  LOG(FATAL) << "RiscVPlic does not support vector stores";
}

uint32_t RiscVPlic::Read(uint32_t offset) {
  uint32_t value = 0;
  // Interrupt priority bit by source.
  if (offset < 0x00'1000) {
    value = interrupt_priority_[offset >> 2];
    return value;
  }

  // Interrupt pending bits by source/
  if (offset < 0x00'2000) {
    offset -= 0x00'1000;
    if (offset > (num_sources_ / 32)) {
      LOG(WARNING) << "Invalid offset: " << offset;
      return 0;
    }
    value = interrupt_pending_[offset];
    return value;
  }

  // Interrupt enable bits for sources by context.
  if (offset < 0x20'0000) {
    offset -= 0x00'2000;
    int context = offset >> 7;
    // Get the context id.
    if (context >= num_contexts_) {
      LOG(ERROR) << "Invalid context number: " << context;
      return 0;
    }
    // Get the word index in the source dimension.
    int word = offset & 0x7f;
    if (word * 32 >= num_sources_ + 32) {
      LOG(ERROR) << "Invalid source number";
      return 0;
    }
    value = interrupt_enabled_[context][word];
    return value;
  }

  // Interrupt priority threshold and claim/complete.
  // Get context id.
  offset -= 0x20'0000;
  int context = offset >> 12;
  if (context >= num_contexts_) {
    LOG(ERROR) << "Invalid context number: " << context;
    return 0;
  }
  auto reg_id = offset & 0xfff;
  switch (reg_id) {
    case 0x0:
      // Priority threshold.
      value = priority_threshold_[context];
      return value;
    case 0x4:
      // Claim/complete interrupt.
      value = ClaimInterrupt(context);
      return value;
    default:
      LOG(ERROR) << "Invalid offset: " << offset;
      break;
  }
  return value;
}

void RiscVPlic::Write(uint32_t offset, uint32_t value) {
  // Interrupt priority bit by source.
  if (offset < 0x00'1000) {
    int source = offset >> 3;
    if (source >= num_sources_) {
      LOG(ERROR) << "Invalid source number: " << source;
      return;
    }
    uint32_t prev = interrupt_priority_[source];
    interrupt_priority_[source] = value;
    // If the priority is being changed from 0 to non-zero, see if there is a
    // pending level based interrupt, and if so, set the plic pending bit.
    if (prev == 0 && value != 0) {
      auto& info = gateway_info_[source];
      if (info.ready && info.pending) {
        SetPlicPendingInterrupt(source);
      }
    }
    return;
  }

  // Interrupt pending bits.
  if (offset < 0x00'2000) {
    offset -= 0x00'1000;
    if (offset > (num_sources_ / 32)) {
      LOG(WARNING) << "Invalid offset: " << offset;
      return;
    }
    uint32_t prev = interrupt_pending_[offset];
    interrupt_pending_[offset] = value;
    // Determine which bits are being set.
    uint32_t bits_set = (value ^ prev) & value;
    // Trigger interrupts for any of the newly set pending bits.
    while (bits_set != 0) {
      int bit = absl::countr_zero(bits_set);
      int source = offset * 32 + bit;
      bits_set &= ~(1 << bit);
      auto [begin, end] = source_to_context_.equal_range(source);
      for (auto it = begin; it != end; ++it) {
        auto context = it->second;
        // If the context IRQ line is not already set, see if the priority is
        // above the threshold, and if so, set the IRQ line.
        if (!context_irq_[context]) {
          // If the priority is less or equal to the threshold, do not trigger
          // the interrupt.
          if (interrupt_priority_[source] <= priority_threshold_[context])
            continue;
          if (context_if_[context] == nullptr) {
            LOG(ERROR) << "No context interface for context " << context;
            continue;
          }
          // Trigger the interrupt.
          context_if_[context]->SetIrq(true);
          context_irq_[context] = true;
        }
      }
    }
    return;
  }

  // Interrupt enable bits for sources by context.
  if (offset < 0x20'0000) {
    offset -= 0x00'2000;
    // Convert from word address to index.
    offset >>= 2;
    // Get the word index in the source dimension.
    int word = (offset >> 2) & 0x1f;
    // Get the context id.
    int context = offset >> 7;
    uint32_t prev = interrupt_enabled_[context][word];
    // Determine which bits are being set.
    uint32_t bits_set = (value ^ prev) & value;
    uint32_t bits_cleared = (value ^ prev) & ~value;
    // Update the enable bits.
    interrupt_enabled_[context][word] = value;
    // Iterate over the set bits and add them to the source/context maps.
    while (bits_set != 0) {
      int bit = absl::countr_zero(bits_set);
      int source = word * 32 + bit;
      source_to_context_.insert({source, context});
      context_to_source_.insert({context, source});
      bits_set &= ~(1 << bit);
      if (context_if_[context] == nullptr) {
        LOG(ERROR) << "No context interface for context " << context;
        continue;
      }
      // If there is no pending IRQ for this context, and the priority is
      // above the threshold, set the IRQ.
      if (!context_irq_[context]) {
        if (interrupt_priority_[source] > priority_threshold_[context]) {
          context_irq_[context] = true;
          context_if_[context]->SetIrq(true);
        }
      }
    }
    // Iterate over the cleared bits and erase them from the source/context
    // maps.
    while (bits_cleared != 0) {
      int bit = absl::countr_zero(bits_cleared);
      int source = word * 32 + bit;
      auto [s2c_begin, s2c_end] = source_to_context_.equal_range(source);
      for (auto s2c_it = s2c_begin; s2c_it != s2c_end; ++s2c_it) {
        if (s2c_it->second == context) {
          source_to_context_.erase(s2c_it);
          break;
        }
      }
      auto [c2s_begin, c2s_end] = context_to_source_.equal_range(context);
      for (auto c2s_it = c2s_begin; c2s_it != c2s_end; ++c2s_it) {
        if (c2s_it->second == source) {
          context_to_source_.erase(c2s_it);
          break;
        }
      }
      bits_cleared &= ~(1 << bit);
    }
    return;
  }

  offset -= 0x20'0000;
  // Interrupt priority threshold and claim/complete.
  // Get context id.
  int context = offset >> 12;
  if (context >= num_contexts_) {
    LOG(ERROR) << "Invalid context number: " << context;
    return;
  }
  auto reg_id = offset & 0xfff;
  switch (reg_id) {
    case 0x0: {
      // Priority threshold.
      uint32_t prev = priority_threshold_[context];
      priority_threshold_[context] = value;
      if (value < prev) {
        // If the priority threshold is being lowered and there is no active
        // interrupt, see if there is a pending interrupt for this context,
        // and if so, set the IRQ.
        if (!context_irq_[context]) {
          if (context_if_[context] == nullptr) {
            LOG(ERROR) << "No context interface for context " << context;
            return;
          }
          auto [begin, end] = context_to_source_.equal_range(context);
          for (auto it = begin; it != end; ++it) {
            if (interrupt_priority_[it->second] > value) {
              context_irq_[context] = true;
              context_if_[context]->SetIrq(true);
              break;
            }
          }
        }
      }
      return;
    }
    case 0x4:
      // Claim/complete interrupt.
      CompleteInterrupt(context, value);
      return;
    default:
      LOG(ERROR) << "Invalid offset: " << offset;
      break;
  }
}

uint32_t RiscVPlic::ClaimInterrupt(int context) {
  if (context < 0 || context >= num_contexts_) return 0;
  uint32_t id = 0;
  int priority = 0;
  // Find the id of the highest-priority pending interrupt for this context.
  auto [begin, end] = context_to_source_.equal_range(context);
  int count = 0;
  int source = 0;
  for (auto it = begin; it != end; ++it) {
    source = it->second;
    // If the source is not pending, skip it.
    if (!IsPending(source)) continue;
    // If the priority is less or equal to the current priority, skip it.
    if (priority > interrupt_priority_[source]) continue;
    // If the priority is the same, the lower id is chosen, so if the new
    // source is greater than the current, go to the next.
    if ((priority == interrupt_priority_[source]) && (id < source)) continue;
    id = source;
    priority = interrupt_priority_[source];
    count++;
  }
  if (id != 0) {
    SetPending(id, false);
    interrupt_claim_complete_[context] = id;
    count--;
  }
  if (count == 0) {
    // If there are zero remaining pending interrupts, clear the IRQ line.
    context_if_[context]->SetIrq(false);
    context_irq_[context] = false;
  }
  return id;
}

void RiscVPlic::CompleteInterrupt(int context, uint32_t id) {
  if (context < 0 || context >= num_contexts_) return;
  // Check if id is in the set of enabled sources for context.
  auto [begin, end] = context_to_source_.equal_range(context);
  bool found = false;
  for (auto it = begin; it != end; ++it) {
    if (it->second == id) {
      found = true;
      break;
    }
  }
  if (!found) return;
  // The PLIC spec only requires that the id be valid for the set of
  // interrupts enabled for the context, not that it matches the
  // interrupt_claim_complete_ value.
  auto source = interrupt_claim_complete_[context];
  interrupt_claim_complete_[context] = 0;
  auto& info = gateway_info_[source];
  // Check to see if there's a pending level based interrupt w priority > 0.
  if (info.pending && (interrupt_priority_[source] > 0)) {
    // Set the plic pending bit but no need to set the ready bit as this will
    // be forwarded to the plic core right away.
    SetPlicPendingInterrupt(source);
    return;
  }
  // Set the gateway ready bit to true.
  info.ready = true;
}

void RiscVPlic::SetPending(int source, bool value) {
  int word = source >> 5;
  int bit = source & 0x1f;
  if (value) {
    interrupt_pending_[word] |= 1 << bit;
  } else {
    interrupt_pending_[word] &= ~(1 << bit);
  }
}

bool RiscVPlic::IsPending(int source) {
  int word = source >> 5;
  int bit = source & 0x1f;
  return (interrupt_pending_[word] & (1 << bit)) != 0;
}

RiscVPlicSourceInterface::RiscVPlicSourceInterface(RiscVPlic* plic, int source,
                                                   bool is_level)
    : plic_(plic), source_(source), is_level_(is_level) {}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
