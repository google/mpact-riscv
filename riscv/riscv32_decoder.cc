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

#include "riscv/riscv32_decoder.h"

#include <string>

#include "mpact/sim/generic/program_error.h"

namespace mpact {
namespace sim {
namespace riscv {

RiscV32Decoder::RiscV32Decoder(RiscVState *state, util::MemoryInterface *memory)
    : state_(state), memory_(memory) {
  // Get a handle to the internal error in the program error controller.
  decode_error_ = state->program_error_controller()->GetProgramError(
      generic::ProgramErrorController::kInternalErrorName);

  // Need a data buffer to load instructions from memory. Allocate a single
  // buffer that can be reused for each instruction word.
  inst_db_ = state_->db_factory()->Allocate<uint32_t>(1);
  // Allocate the isa factory class, the top level isa decoder instance, and
  // the encoding parser.
  riscv_isa_factory_ = new RV32IsaFactory();
  riscv_isa_ = new isa32::RiscV32GInstructionSet(state, riscv_isa_factory_);
  riscv_encoding_ = new isa32::RiscV32GEncoding(state);
  decode_error_ = state->program_error_controller()->GetProgramError(
      generic::ProgramErrorController::kInternalErrorName);
}

RiscV32Decoder::~RiscV32Decoder() {
  delete riscv_isa_;
  delete riscv_isa_factory_;
  delete riscv_encoding_;
  inst_db_->DecRef();
}

generic::Instruction *RiscV32Decoder::DecodeInstruction(uint64_t address) {
  // First check that the address is aligned properly. If not, create and return
  // an instruction object that will raise an exception.
  if (address & 0x1) {
    auto *inst = new generic::Instruction(0, state_);
    inst->set_size(1);
    inst->SetDisassemblyString("Misaligned instruction address");
    inst->set_opcode(*isa32::OpcodeEnum::kNone);
    inst->set_address(address);
    inst->set_semantic_function([this](generic::Instruction *inst) {
      state_->Trap(/*is_interrupt*/ false, inst->address(),
                   *ExceptionCode::kInstructionAddressMisaligned,
                   inst->address() ^ 0x1, inst);
    });
    return inst;
  }

  // If the address is greater than the max address, return an instruction
  // that will raise an exception.
  if (address > state_->max_physical_address()) {
    auto *inst = new generic::Instruction(0, state_);
    inst->set_size(0);
    inst->SetDisassemblyString("Instruction access fault");
    inst->set_opcode(*isa32::OpcodeEnum::kNone);
    inst->set_address(address);
    inst->set_semantic_function([this](generic::Instruction *inst) {
      state_->Trap(/*is_interrupt*/ false, inst->address(),
                   *ExceptionCode::kInstructionAccessFault, inst->address(),
                   nullptr);
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
