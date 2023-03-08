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

#ifndef MPACT_RISCV_RISCV_RISCV_BREAKPOINT_H_
#define MPACT_RISCV_RISCV_RISCV_BREAKPOINT_H_

#include "absl/container/btree_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "mpact/sim/generic/component.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/util/memory/memory_interface.h"

// This file defines a class for handling breakpoints for a RiscV32 core. It
// doesn't interact with the core itself, but only with memory. Upon updating
// memory with an ebreak instruction or writing the original instruction back,
// the breakpoint handler calls the invalidate function. The intention is that
// the invalidate function should result in an invalidation of any cached
// decode of the modified instruction address.
namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::generic::DataBufferFactory;
using ::mpact::sim::util::MemoryInterface;

class RiscVBreakpointManager {
 public:
  // 32 bit and 16 bit software breakpoint instructions.
  static constexpr uint32_t kEBreak32 = 0b000000000001'00000'000'00000'1110011;
  static constexpr uint16_t kEBreak16 = 0b100'1'00000'00000'10;

  // Callback type for function to invalidate an instruction.
  using InvalidateFcn = absl::AnyInvocable<void(uint64_t)>;

  RiscVBreakpointManager(MemoryInterface *memory, InvalidateFcn invalidate_fcn);
  ~RiscVBreakpointManager();

  bool HasBreakpoint(uint64_t address);
  // Set/Clear breakpoint at address.
  absl::Status SetBreakpoint(uint64_t address);
  absl::Status ClearBreakpoint(uint64_t address);
  // Disable/Enable breakpoint at address. These act as Set/Clear, but the
  // breakpoint information isn't deleted.
  absl::Status DisableBreakpoint(uint64_t address);
  absl::Status EnableBreakpoint(uint64_t address);
  // Clear all breakpoints.
  void ClearAllBreakpoints();
  // Is the address an active breakpoint?
  bool IsBreakpoint(uint64_t address);

 private:
  // Structure keeping track of breakpoint information.
  struct BreakpointInfo {
    bool is_active;
    // Address.
    uint64_t address;
    // Byte size of the instruction word/breakpoint instruction.
    int size;
    // The original instruction at the breakpoint location.
    uint32_t instruction_word;

    BreakpointInfo(bool is_active_, uint64_t address_, int size_,
                   uint32_t instruction_word_)
        : is_active(is_active_),
          address(address_),
          size(size_),
          instruction_word(instruction_word_) {}
  };

  MemoryInterface *memory_;
  InvalidateFcn invalidate_fcn_;
  DataBuffer *db4_ = nullptr;
  DataBuffer *db2_ = nullptr;
  DataBufferFactory db_factory_;
  absl::btree_map<uint64_t, BreakpointInfo *> breakpoint_map_;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_BREAKPOINT_H_
