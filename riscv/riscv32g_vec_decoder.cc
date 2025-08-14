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

#include "riscv/riscv32g_vec_decoder.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "mpact/sim/generic/program_error.h"

namespace mpact {
namespace sim {
namespace riscv {

RiscV32GVecDecoder::RiscV32GVecDecoder(RiscVState* state,
                                       util::MemoryInterface* memory)
    : state_(state), memory_(memory) {
  // Get a handle to the internal error in the program error controller.
  decode_error_ = state->program_error_controller()->GetProgramError(
      generic::ProgramErrorController::kInternalErrorName);

  // Need a data buffer to load instructions from memory. Allocate a single
  // buffer that can be reused for each instruction word.
  inst_db_ = state_->db_factory()->Allocate<uint32_t>(1);
  // Allocate the isa factory class, the top level isa decoder instance, and
  // the encoding parser.
  riscv_isa_factory_ = new RV32GVIsaFactory();
  riscv_isa_ = new isa32v::RiscV32GVInstructionSet(state, riscv_isa_factory_);
  riscv_encoding_ = new isa32v::RiscV32GVecEncoding(state);
  decode_error_ = state->program_error_controller()->GetProgramError(
      generic::ProgramErrorController::kInternalErrorName);
}

RiscV32GVecDecoder::~RiscV32GVecDecoder() {
  inst_db_->DecRef();
  delete riscv_isa_;
  delete riscv_isa_factory_;
  delete riscv_encoding_;
}

generic::Instruction* RiscV32GVecDecoder::DecodeInstruction(uint64_t address) {
  // First check that the address is aligned properly. If not, create and return
  // an instruction object that generates an internal simulator error when
  // executed.
  if (address & 0x1) {
    auto* inst = new generic::Instruction(address, state_);
    std::string error =
        absl::StrCat(absl::Hex(address), ": unaligned instruction address");
    inst->set_semantic_function([error, this](generic::Instruction* inst) {
      decode_error_->Raise(error);
    });
    inst->set_size(1);
    return inst;
  }

  // Read the instruction word from memory and parse it in the encoding parser.
  memory_->Load(address, inst_db_, nullptr, nullptr);
  uint32_t iword = inst_db_->Get<uint32_t>(0);
  riscv_encoding_->ParseInstruction(iword);

  // Call the isa decoder to obtain a new instruction object for the instruction
  // word that was parsed above.
  auto* instruction = riscv_isa_->Decode(address, riscv_encoding_);
  return instruction;
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
