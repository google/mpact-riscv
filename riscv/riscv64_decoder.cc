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

#include "riscv/riscv64_decoder.h"

#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"
#include "riscv/riscv64g_decoder.h"
#include "riscv/riscv64g_encoding.h"
#include "riscv/riscv64g_enums.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/program_error.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::operator*;  // NOLINT: is used below (clang error).

RiscV64Decoder::RiscV64Decoder(RiscVState *state, util::MemoryInterface *memory)
    : RiscV64Decoder(state, memory, /*use_abi_names*/ true) {}

RiscV64Decoder::RiscV64Decoder(RiscVState *state, util::MemoryInterface *memory,
                               bool use_abi_names)
    : state_(state), memory_(memory) {
  // Get a handle to the internal error in the program error controller.
  decode_error_ = state->program_error_controller()->GetProgramError(
      generic::ProgramErrorController::kInternalErrorName);

  // Need a data buffer to load instructions from memory. Allocate a single
  // buffer that can be reused for each instruction word.
  inst_db_ = state_->db_factory()->Allocate<uint32_t>(1);
  // Allocate the isa factory class, the top level isa decoder instance, and
  // the encoding parser.
  riscv_isa_factory_ = new RV64IsaFactory();
  riscv_isa_ = new isa64::RiscV64GInstructionSet(state, riscv_isa_factory_);
  riscv_encoding_ = new isa64::RiscV64GEncoding(state, use_abi_names);
  decode_error_ = state->program_error_controller()->GetProgramError(
      generic::ProgramErrorController::kInternalErrorName);
}

RiscV64Decoder::~RiscV64Decoder() {
  inst_db_->DecRef();
  delete riscv_isa_;
  delete riscv_isa_factory_;
  delete riscv_encoding_;
}

generic::Instruction *RiscV64Decoder::DecodeInstruction(uint64_t address) {
  // First check that the address is aligned properly. If not, create and return
  // an instruction object that will raise an exception.
  if (address & 0x1) {
    auto *inst = new generic::Instruction(address, state_);
    inst->set_size(1);
    inst->SetDisassemblyString("Misaligned instruction address");
    inst->set_opcode(*isa64::OpcodeEnum::kNone);
    inst->set_address(address);
    inst->set_semantic_function([this, address](generic::Instruction *inst) {
      state_->Trap(/*is_interrupt*/ false, address,
                   *ExceptionCode::kInstructionAddressMisaligned, address ^ 0x1,
                   inst);
    });
    return inst;
  }

  // If the address is greater than the max address, return an instruction
  // object that will raise an exception.
  if (address > state_->max_physical_address()) {
    auto *inst = new generic::Instruction(address, state_);
    inst->set_size(0);
    inst->SetDisassemblyString("Instruction access fault");
    inst->set_opcode(*isa64::OpcodeEnum::kNone);
    inst->set_address(address);
    inst->set_semantic_function([this, address](generic::Instruction *inst) {
      state_->Trap(/*is_interrupt*/ false, address,
                   *ExceptionCode::kInstructionAccessFault, address, nullptr);
    });
    return inst;
  }

  // Read the instruction word from memory and parse it in the encoding parser.
  memory_->Load(address, inst_db_, nullptr, nullptr);
  uint32_t iword = inst_db_->Get<uint32_t>(0);
  riscv_encoding_->ParseInstruction(iword);

  // Call the isa decoder to obtain a new instruction object for the instruction
  // word that was parsed above.
  auto *instruction = riscv_isa_->Decode(address, riscv_encoding_);
  return instruction;
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
