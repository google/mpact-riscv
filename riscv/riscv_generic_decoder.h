// Copyright 2025 Google LLC
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

#include <cstdint>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/decoder_interface.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_state.h"

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_GENERIC_DECODER_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_GENERIC_DECODER_H_

namespace mpact::sim::riscv {

using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.

using ::mpact::sim::generic::Instruction;

template <typename OpcodeEnum, typename Encoding, typename Isa>
class RiscVGenericDecoder {
 public:
  RiscVGenericDecoder(RiscVState* state, util::MemoryInterface* memory,
                      Encoding* encoding, Isa* isa)
      : state_(state), memory_(memory), encoding_(encoding), isa_(isa) {
    auto res =
        state_->csr_set()->GetCsr(static_cast<uint64_t>(RiscVCsrEnum::kMIsa));
    if (res.ok()) {
      isa_csr_ = res.value();
    } else {
      isa_csr_ = nullptr;
      LOG(FATAL) << "Failed to get misa CSR: " << res.status();
    }

    // Need a data buffer to load instructions from memory. Allocate a single
    // buffer that can be reused for each instruction word.
    inst_db_ = state_->db_factory()->Allocate<uint32_t>(1);
  }

  ~RiscVGenericDecoder() { inst_db_->DecRef(); }

  Instruction* DecodeInstruction(uint64_t address) {
    Instruction* instruction = nullptr;
    auto status = CheckAddress(address, instruction);
    if (!status.ok()) {
      LOG(FATAL) << "Decoder error check failed.";
      return nullptr;
    }
    if (instruction != nullptr) return instruction;

    // Read the instruction word from memory and parse it in the encoding
    // parser.
    memory_->Load(address, inst_db_, nullptr, nullptr);
    uint32_t iword = inst_db_->Get<uint32_t>(0);
    encoding_->ParseInstruction(iword);

    // Call the isa decoder to obtain a new instruction object for the
    // instruction word that was parsed above.
    instruction = isa_->Decode(address, encoding_);
    return instruction;
  }

 private:
  // Check that the address is valid. If not, return an instruction object that
  // will raise an exception.
  absl::Status CheckAddress(uint64_t address, Instruction*& inst) {
    if (isa_csr_ == nullptr) {
      return absl::InvalidArgumentError("MISA CSR is null");
    }
    if (inst != nullptr) {
      return absl::InvalidArgumentError("Instruction is not null");
    }
    // First check that the address is aligned properly. If not, create and
    // return an instruction object that will raise an exception.
    if (address & 0x1) {
      inst = new generic::Instruction(0, state_);
      inst->set_size(1);
      inst->SetDisassemblyString("Misaligned instruction address");
      inst->set_opcode(*OpcodeEnum::kNone);
      inst->set_address(address);
      inst->set_semantic_function([this](generic::Instruction* inst) {
        state_->Trap(/*is_interrupt*/ false, inst->address(),
                     *ExceptionCode::kInstructionAddressMisaligned,
                     inst->address() ^ 0x1, inst);
      });
      return absl::OkStatus();
    }

    // If the C extension is not supported and the address is not 4 byte
    // aligned, return an instruction that will raise an exception.
    if (((isa_csr_->GetUint64() &
          static_cast<uint64_t>(IsaExtensions::kCompressed)) == 0) &&
        (address & 0x3)) {
      inst = new generic::Instruction(0, state_);
      inst->set_size(1);
      inst->SetDisassemblyString("Misaligned instruction address");
      inst->set_opcode(*OpcodeEnum::kNone);
      inst->set_address(address);
      inst->set_semantic_function([this](generic::Instruction* inst) {
        state_->Trap(/*is_interrupt*/ false, inst->address(),
                     *ExceptionCode::kInstructionAddressMisaligned,
                     inst->address() & ~0x3, inst);
      });
      return absl::OkStatus();
    }

    // If the address is greater than the max address, return an instruction
    // that will raise an exception.
    if (address > state_->max_physical_address()) {
      inst = new generic::Instruction(0, state_);
      inst->set_size(0);
      inst->SetDisassemblyString("Instruction access fault");
      inst->set_opcode(*OpcodeEnum::kNone);
      inst->set_address(address);
      inst->set_semantic_function([this](generic::Instruction* inst) {
        state_->Trap(/*is_interrupt*/ false, inst->address(),
                     *ExceptionCode::kInstructionAccessFault, inst->address(),
                     nullptr);
      });
      return absl::OkStatus();
    }

    return absl::OkStatus();
  }

  RiscVState* state_;
  util::MemoryInterface* memory_;
  Encoding* encoding_;
  Isa* isa_;
  generic::DataBuffer* inst_db_;
  RiscVCsrInterface* isa_csr_;
};

}  // namespace mpact::sim::riscv

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_GENERIC_DECODER_H_
