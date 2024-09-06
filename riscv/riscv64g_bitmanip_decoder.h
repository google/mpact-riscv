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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV64G_BITMANIP_DECODER_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV64G_BITMANIP_DECODER_H_

#include <cstdint>
#include <memory>

#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/decoder_interface.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "riscv/riscv64gzb_decoder.h"
#include "riscv/riscv64gzb_encoding.h"
#include "riscv/riscv64gzb_enums.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::Instruction;
using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.

// This is the factory class needed by the generated decoder. It is responsible
// for creating the decoder for each slot instance. Since the riscv architecture
// only has a single slot, it's a pretty simple class.
class RV64GZBIsaFactory : public isa64gzb::RiscV64GZBInstructionSetFactory {
 public:
  std::unique_ptr<isa64gzb::Riscv64gzbSlot> CreateRiscv64gzbSlot(
      ArchState *state) override {
    return std::make_unique<isa64gzb::Riscv64gzbSlot>(state);
  }
};

// This class implements the generic DecoderInterface and provides a bridge
// to the (isa specific) generated decoder classes.
class RiscV64GBitmanipDecoder : public generic::DecoderInterface {
 public:
  using SlotEnum = isa64gzb::SlotEnum;
  using OpcodeEnum = isa64gzb::OpcodeEnum;

  RiscV64GBitmanipDecoder(RiscVState *state, util::MemoryInterface *memory);
  RiscV64GBitmanipDecoder() = delete;
  ~RiscV64GBitmanipDecoder() override;

  // This will always return a valid instruction that can be executed. In the
  // case of a decode error, the semantic function in the instruction object
  // instance will raise an internal simulator error when executed.
  generic::Instruction *DecodeInstruction(uint64_t address) override;
  // Return the number of opcodes supported by this decoder.
  int GetNumOpcodes() const override { return *OpcodeEnum::kPastMaxValue; }
  // Return the name of the opcode at the given index.
  const char *GetOpcodeName(int index) const override {
    return isa64gzb::kOpcodeNames[index];
  }

  // Getter.
  isa64gzb::RiscV64GZBEncoding *riscv_encoding() const {
    return riscv_encoding_;
  }

 private:
  RiscVState *state_;
  util::MemoryInterface *memory_;
  generic::DataBuffer *inst_db_;
  isa64gzb::RiscV64GZBEncoding *riscv_encoding_;
  RV64GZBIsaFactory *riscv_isa_factory_;
  isa64gzb::RiscV64GZBInstructionSet *riscv_isa_;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV64G_BITMANIP_DECODER_H_
