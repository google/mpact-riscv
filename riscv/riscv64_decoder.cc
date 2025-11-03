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
#include <memory>

#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "riscv/riscv64g_decoder.h"
#include "riscv/riscv64g_encoding.h"
#include "riscv/riscv64g_enums.h"
#include "riscv/riscv_generic_decoder.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

RiscV64Decoder::RiscV64Decoder(RiscVState* state, util::MemoryInterface* memory)
    : RiscV64Decoder(state, memory, /*use_abi_names*/ true) {}

RiscV64Decoder::RiscV64Decoder(RiscVState* state, util::MemoryInterface* memory,
                               bool use_abi_names)
    : state_(state), memory_(memory) {
  // Allocate the isa factory class, the top level isa decoder instance, and
  // the encoding parser.
  riscv_isa_factory_ = std::make_unique<RV64IsaFactory>();
  riscv_isa_ = std::make_unique<isa64::RiscV64GInstructionSet>(
      state, riscv_isa_factory_.get());
  riscv_encoding_ =
      std::make_unique<isa64::RiscV64GEncoding>(state, use_abi_names);
  decoder_ = std::make_unique<
      RiscVGenericDecoder<isa64::OpcodeEnum, isa64::RiscV64GEncoding,
                          isa64::RiscV64GInstructionSet>>(
      state, memory, riscv_encoding_.get(), riscv_isa_.get());
}

generic::Instruction* RiscV64Decoder::DecodeInstruction(uint64_t address) {
  return decoder_->DecodeInstruction(address);
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
