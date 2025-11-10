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

#ifndef MPACT_RISCV_RISCV_RISCV64G_VEC_DECODER_H_
#define MPACT_RISCV_RISCV_RISCV64G_VEC_DECODER_H_

#include <cstdint>
#include <memory>

#include "mpact/sim/generic/decoder_interface.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "riscv/riscv64g_vec_encoding.h"
#include "riscv/riscv64gv_decoder.h"
#include "riscv/riscv64gv_enums.h"
#include "riscv/riscv_generic_decoder.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

// This is the factory class needed by the generated decoder. It is responsible
// for creating the decoder for each slot instance. Since the riscv architecture
// only has a single slot, it's a pretty simple class.
class RV64GVIsaFactory : public isa64v::RiscV64GVInstructionSetFactory {
 public:
  std::unique_ptr<isa64v::Riscv64gvSlot> CreateRiscv64gvSlot(
      ArchState* state) override {
    return std::make_unique<isa64v::Riscv64gvSlot>(state);
  }
};

// This class implements the generic DecoderInterface and provides a bridge
// to the (isa specific) generated decoder classes. It implements a decoder that
// includes the RV64G + vector extensions.
class RiscV64GVecDecoder : public generic::DecoderInterface {
 public:
  using SlotEnum = isa64v::SlotEnum;
  using OpcodeEnum = isa64v::OpcodeEnum;

  RiscV64GVecDecoder(RiscVState* state, util::MemoryInterface* memory);
  RiscV64GVecDecoder() = delete;

  // This will always return a valid instruction that can be executed. In the
  // case of a decode error, the semantic function in the instruction object
  // instance will raise an internal simulator error when executed.
  generic::Instruction* DecodeInstruction(uint64_t address) override;
  // Return the number of opcodes supported by this decoder.
  int GetNumOpcodes() const override { return *OpcodeEnum::kPastMaxValue; }
  // Return the name of the opcode at the given index.
  const char* GetOpcodeName(int index) const override {
    return isa64v::kOpcodeNames[index];
  }

  // Getter.
  isa64v::RiscV64GVecEncoding* riscv_encoding() const {
    return riscv_encoding_.get();
  }

 private:
  RiscVState* const state_;
  util::MemoryInterface* const memory_;
  std::unique_ptr<RiscVGenericDecoder<RiscVState, isa64v::OpcodeEnum,
                                      isa64v::RiscV64GVecEncoding,
                                      isa64v::RiscV64GVInstructionSet>>
      decoder_;
  std::unique_ptr<isa64v::RiscV64GVecEncoding> riscv_encoding_;
  std::unique_ptr<RV64GVIsaFactory> riscv_isa_factory_;
  std::unique_ptr<isa64v::RiscV64GVInstructionSet> riscv_isa_;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV64G_VEC_DECODER_H_
