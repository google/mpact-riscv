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

#ifndef MPACT_RISCV_RISCV_RISCV64_DECODER_H_
#define MPACT_RISCV_RISCV_RISCV64_DECODER_H_

#include <cstdint>
#include <memory>

#include "mpact/sim/generic/decoder_interface.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "riscv/riscv64g_decoder.h"
#include "riscv/riscv64g_encoding.h"
#include "riscv/riscv64g_enums.h"
#include "riscv/riscv_generic_decoder.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.

// This is the factory class needed by the generated decoder. It is responsible
// for creating the decoder for each slot instance. Since the riscv architecture
// only has a single slot, it's a pretty simple class.
class RV64IsaFactory : public isa64::RiscV64GInstructionSetFactory {
 public:
  std::unique_ptr<isa64::Riscv64gSlot> CreateRiscv64gSlot(
      ArchState* state) override {
    return std::make_unique<isa64::Riscv64gSlot>(state);
  }
};

// This class implements the generic DecoderInterface and provides a bridge
// to the (isa specific) generated decoder classes.
class RiscV64Decoder : public generic::DecoderInterface {
 public:
  using SlotEnum = isa64::SlotEnum;
  using OpcodeEnum = isa64::OpcodeEnum;

  RiscV64Decoder(RiscVState* state, util::MemoryInterface* memory);
  RiscV64Decoder(RiscVState* state, util::MemoryInterface* memory,
                 bool use_abi_names);
  RiscV64Decoder() = delete;

  // This will always return a valid instruction that can be executed. In the
  // case of a decode error, the semantic function in the instruction object
  // instance will raise an internal simulator error when executed.
  generic::Instruction* DecodeInstruction(uint64_t address) override;
  // Return the number of opcodes supported by this decoder.
  int GetNumOpcodes() const override { return *OpcodeEnum::kPastMaxValue; }
  // Return the name of the opcode at the given index.
  const char* GetOpcodeName(int index) const override {
    return isa64::kOpcodeNames[index];
  }

  // Getter.
  isa64::RiscV64GEncoding* riscv_encoding() const {
    return riscv_encoding_.get();
  }

 private:
  RiscVState* state_;
  util::MemoryInterface* memory_;
  std::unique_ptr<RiscVGenericDecoder<RiscVState, isa64::OpcodeEnum,
                                      isa64::RiscV64GEncoding,
                                      isa64::RiscV64GInstructionSet>>
      decoder_;
  std::unique_ptr<isa64::RiscV64GEncoding> riscv_encoding_;
  std::unique_ptr<RV64IsaFactory> riscv_isa_factory_;
  std::unique_ptr<isa64::RiscV64GInstructionSet> riscv_isa_;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV64_DECODER_H_
