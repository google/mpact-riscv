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

#ifndef RISCV_RISCV32_DECODER_H_
#define RISCV_RISCV32_DECODER_H_

#include <memory>

#include "absl/memory/memory.h"
#include "mpact/sim/generic/arch_state.h"
#include "mpact/sim/generic/decoder_interface.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/program_error.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "riscv/riscv32g_decoder.h"
#include "riscv/riscv32g_encoding.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

// This is the factory class needed by the generated decoder. It is responsible
// for creating the decoder for each slot instance. Since the riscv architecture
// only has a single slot, it's a pretty simple class.
class RV32IsaFactory : public isa32::RiscV32GInstructionSetFactory {
 public:
  std::unique_ptr<isa32::Riscv32gSlot> CreateRiscv32gSlot(
      ArchState *state) override {
    return std::make_unique<isa32::Riscv32gSlot>(state);
  }
};

// This class implements the generic DecoderInterface and provides a bridge
// to the (isa specific) generated decoder classes.
class RiscV32Decoder : public generic::DecoderInterface {
 public:
  using SlotEnum = isa32::SlotEnum;
  using OpcodeEnum = isa32::OpcodeEnum;

  RiscV32Decoder(RiscVState *state, util::MemoryInterface *memory);
  RiscV32Decoder() = delete;
  ~RiscV32Decoder() override;

  // This will always return a valid instruction that can be executed. In the
  // case of a decode error, the semantic function in the instruction object
  // instance will raise an internal simulator error when executed.
  generic::Instruction *DecodeInstruction(uint64_t address) override;

  // Getter.
  isa32::RiscV32GEncoding *riscv_encoding() const { return riscv_encoding_; }

 private:
  RiscVState *state_;
  util::MemoryInterface *memory_;
  std::unique_ptr<generic::ProgramError> decode_error_;
  generic::DataBuffer *inst_db_;
  isa32::RiscV32GEncoding *riscv_encoding_;
  RV32IsaFactory *riscv_isa_factory_;
  isa32::RiscV32GInstructionSet *riscv_isa_;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // RISCV_RISCV32_DECODER_H_
