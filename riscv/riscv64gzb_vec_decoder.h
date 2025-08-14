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

#ifndef MPACT_RISCV_RISCV_RISCV64GZB_VEC_DECODER_H_
#define MPACT_RISCV_RISCV_RISCV64GZB_VEC_DECODER_H_

#include <cstdint>
#include <memory>

#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/decoder_interface.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/program_error.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "riscv/riscv64gvzb_decoder.h"
#include "riscv/riscv64gvzb_enums.h"
#include "riscv/riscv64gzb_vec_encoding.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.

// This is the factory class needed by the generated decoder. It is responsible
// for creating the decoder for each slot instance. Since the riscv architecture
// only has a single slot, it's a pretty simple class.
class RV64GVZBIsaFactory : public isa64gvzb::RiscV64GVZBInstructionSetFactory {
 public:
  std::unique_ptr<isa64gvzb::Riscv64gvzbSlot> CreateRiscv64gvzbSlot(
      ArchState* state) override {
    return std::make_unique<isa64gvzb::Riscv64gvzbSlot>(state);
  }
};

// This class implements the generic DecoderInterface and provides a bridge
// to the (isa specific) generated decoder classes. It implements a decoder that
// includes the RV64GZB + vector extensions.
class RiscV64GZBVecDecoder : public generic::DecoderInterface {
 public:
  using SlotEnum = isa64gvzb::SlotEnum;
  using OpcodeEnum = isa64gvzb::OpcodeEnum;

  RiscV64GZBVecDecoder(RiscVState* state, util::MemoryInterface* memory);
  RiscV64GZBVecDecoder() = delete;
  ~RiscV64GZBVecDecoder() override;

  // This will always return a valid instruction that can be executed. In the
  // case of a decode error, the semantic function in the instruction object
  // instance will raise an internal simulator error when executed.
  generic::Instruction* DecodeInstruction(uint64_t address) override;
  // Return the number of opcodes supported by this decoder.
  int GetNumOpcodes() const override { return *OpcodeEnum::kPastMaxValue; }
  // Return the name of the opcode at the given index.
  const char* GetOpcodeName(int index) const override {
    return isa64gvzb::kOpcodeNames[index];
  }

 private:
  RiscVState* const state_;
  util::MemoryInterface* const memory_;

  // Buffer used to load instructions from memory. Re-used for each instruction
  // word.
  generic::DataBuffer* const inst_db_;

  std::unique_ptr<generic::ProgramError> decode_error_;
  std::unique_ptr<isa64gvzb::RiscV64GZBVecEncoding> riscv_encoding_;
  std::unique_ptr<RV64GVZBIsaFactory> riscv_isa_factory_;
  std::unique_ptr<isa64gvzb::RiscV64GVZBInstructionSet> riscv_isa_;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV64GZB_VEC_DECODER_H_
