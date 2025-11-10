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

#include "riscv/riscv64g_bitmanip_decoder.h"

#include <cstdint>
#include <memory>

#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "riscv/riscv64gzb_decoder.h"
#include "riscv/riscv64gzb_encoding.h"
#include "riscv/riscv64gzb_enums.h"
#include "riscv/riscv_generic_decoder.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

RiscV64GBitmanipDecoder::RiscV64GBitmanipDecoder(RiscVState* state,
                                                 util::MemoryInterface* memory)
    : state_(state), memory_(memory) {
  // Allocate the isa factory class, the top level isa decoder instance, and
  // the encoding parser.
  riscv_isa_factory_ = std::make_unique<RV64GZBIsaFactory>();
  riscv_isa_ = std::make_unique<isa64gzb::RiscV64GZBInstructionSet>(
      state, riscv_isa_factory_.get());
  riscv_encoding_ = std::make_unique<isa64gzb::RiscV64GZBEncoding>(state);
  decoder_ =
      std::make_unique<RiscVGenericDecoder<RiscVState, isa64gzb::OpcodeEnum,
                                           isa64gzb::RiscV64GZBEncoding,
                                           isa64gzb::RiscV64GZBInstructionSet>>(
          state, memory, riscv_encoding_.get(), riscv_isa_.get());
}

generic::Instruction* RiscV64GBitmanipDecoder::DecodeInstruction(
    uint64_t address) {
  return decoder_->DecodeInstruction(address);
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
