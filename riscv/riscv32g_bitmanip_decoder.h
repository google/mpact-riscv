// Copyright 2024 Google LLC
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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV32G_BITMANIP_DECODER_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV32G_BITMANIP_DECODER_H_

#include <cstdint>
#include <memory>

#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/decoder_interface.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "riscv/riscv32gzb_decoder.h"
#include "riscv/riscv32gzb_encoding.h"
#include "riscv/riscv32gzb_enums.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::Instruction;
using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.

// This is the factory class needed by the generated decoder. It is responsible
// for creating the decoder for each slot instance. Since the riscv architecture
// only has a single slot, it's a pretty simple class.
class RV32GZBIsaFactory : public isa32gzb::RiscV32GZBInstructionSetFactory {
 public:
  std::unique_ptr<isa32gzb::Riscv32gzbSlot> CreateRiscv32gzbSlot(
      ArchState* state) override {
    return std::make_unique<isa32gzb::Riscv32gzbSlot>(state);
  }
};

// This class implements the generic DecoderInterface and provides a bridge
// to the (isa specific) generated decoder classes.
class RiscV32GBitmanipDecoder : public generic::DecoderInterface {
 public:
  using SlotEnum = isa32gzb::SlotEnum;
  using OpcodeEnum = isa32gzb::OpcodeEnum;

  RiscV32GBitmanipDecoder(RiscVState* state, util::MemoryInterface* memory);
  RiscV32GBitmanipDecoder() = delete;
  ~RiscV32GBitmanipDecoder() override;

  // This will always return a valid instruction that can be executed. In the
  // case of a decode error, the semantic function in the instruction object
  // instance will raise an internal simulator error when executed.
  generic::Instruction* DecodeInstruction(uint64_t address) override;
  // Return the number of opcodes supported by this decoder.
  int GetNumOpcodes() const override { return *OpcodeEnum::kPastMaxValue; }
  // Return the name of the opcode at the given index.
  const char* GetOpcodeName(int index) const override {
    return isa32gzb::kOpcodeNames[index];
  }

  // Getter.
  isa32gzb::RiscV32GZBEncoding* riscv_encoding() const {
    return riscv_encoding_;
  }

 private:
  RiscVState* state_;
  util::MemoryInterface* memory_;
  generic::DataBuffer* inst_db_;
  isa32gzb::RiscV32GZBEncoding* riscv_encoding_;
  RV32GZBIsaFactory* riscv_isa_factory_;
  isa32gzb::RiscV32GZBInstructionSet* riscv_isa_;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV32G_BITMANIP_DECODER_H_
