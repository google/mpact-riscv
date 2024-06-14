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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_ACTION_POINT_MEMORY_INTERFACE_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_ACTION_POINT_MEMORY_INTERFACE_H_

#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "mpact/sim/generic/action_point_manager_base.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace mpact::sim::riscv {

using ::mpact::sim::generic::ActionPointMemoryInterface;
using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::generic::DataBufferFactory;
using ::mpact::sim::util::MemoryInterface;

// This file defines the RiscVActionPointMemoryInterface class which provides
// the low level memory functionality required to implement breakpoint and other
// 'actions' that need to be performed when an instruction executes. It uses
// the RiscV ebreak instruction to stop execution.

class RiscVActionPointMemoryInterface : public ActionPointMemoryInterface {
 public:
  // 32 bit and 16 bit software breakpoint instructions.
  static constexpr uint32_t kEBreak32 = 0b000000000001'00000'000'00000'1110011;
  static constexpr uint16_t kEBreak16 = 0b100'1'00000'00000'10;
  // Callback type for function to invalidate the decoding of an instruction.
  using InvalidateFcn = absl::AnyInvocable<void(uint64_t address)>;
  // The constructor takes a pointer to the memory interface through which
  // instructions can be read and written,and a function to invalidate the
  // decoding of an instruction.
  RiscVActionPointMemoryInterface(MemoryInterface *memory,
                                  InvalidateFcn invalidate_fcn);
  ~RiscVActionPointMemoryInterface() override;

  // This restores the original instruction in memory, and allows it to be
  // decoded and executed, provided the address is an action point. If not,
  // no action is taken.
  absl::Status WriteOriginalInstruction(uint64_t address) override;
  // Store breakpoint instruction, provided the address is an action point.
  // Otherwise no action is taken.
  absl::Status WriteBreakpointInstruction(uint64_t address) override;

 private:
  struct InstructionInfo {
    int size;
    uint32_t og_instruction_word;
  };

  // Returns the size of the instruction at the given address, or 0 if unknown.
  int GetInstructionSize(uint32_t instruction_word) const;

  // Data buffer factory and two data buffer pointers to use for reading and
  // writing instructions.
  DataBufferFactory db_factory_;
  DataBuffer *db4_ = nullptr;
  DataBuffer *db2_ = nullptr;
  // Maps from address to information about the instruction.
  absl::flat_hash_map<uint64_t, InstructionInfo *> instruction_map_;
  // Interface to program memory.
  MemoryInterface *memory_;
  // Function to be called to invalidate any stored decoding of an instruction.
  InvalidateFcn invalidate_fcn_;
};

}  // namespace mpact::sim::riscv

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_ACTION_POINT_MEMORY_INTERFACE_H_
