// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_MPACT_RISCV_ZVBB_ENCODING_H_
#define THIRD_PARTY_MPACT_RISCV_ZVBB_ENCODING_H_

#include <cstdint>
#include <string>

#include "mpact/sim/generic/simple_resource.h"
#include "mpact/sim/generic/simple_resource_operand.h"
#include "riscv/riscv_encoding_common.h"
#include "riscv/riscv_getter_helpers.h"
#include "riscv/riscv_state.h"
#include "riscv/zvbb_bin_decoder.h"
#include "riscv/zvbb_decoder.h"
#include "riscv/zvbb_enums.h"

namespace mpact::sim::riscv::zvbb {

// This class provides the interface between the generated instruction decoder
// framework (which is agnostic of the actual bit representation of
// instructions) and the instruction representation. This class provides methods
// to return the opcode, source operands, and destination operands for
// instructions according to the operand fields in the encoding.
class ZVBBEncoding : public ZVBBEncodingBase, public RiscVEncodingCommon {
 public:
  explicit ZVBBEncoding(RiscVState *state);
  ~ZVBBEncoding() override;

  void ParseInstruction(uint32_t inst_word);
  OpcodeEnum GetOpcode(SlotEnum, int) override { return opcode_; }
  FormatEnum GetFormat(SlotEnum, int) { return format_; }

  PredicateOperandInterface *GetPredicate(SlotEnum, int, OpcodeEnum,
                                          PredOpEnum) override {
    return nullptr;
  }

  ResourceOperandInterface *GetSimpleResourceOperand(
      SlotEnum, int, OpcodeEnum, SimpleResourceVector &resource_vec,
      int end) override;

  ResourceOperandInterface *GetComplexResourceOperand(
      SlotEnum, int, OpcodeEnum, ComplexResourceEnum resource, int begin,
      int end) override;

  SourceOperandInterface *GetSource(SlotEnum, int, OpcodeEnum, SourceOpEnum op,
                                    int source_no) override;

  DestinationOperandInterface *GetDestination(SlotEnum, int, OpcodeEnum,
                                              DestOpEnum op, int dest_no,
                                              int latency) override;

  int GetLatency(SlotEnum, int, OpcodeEnum, DestOpEnum, int) override {
    return 0;
  }

  // Methods inherited from RiscVEncodingCommon.
  RiscVState *state() const override { return state_; }
  generic::SimpleResourcePool *resource_pool() override {
    return resource_pool_;
  }
  uint32_t inst_word() const override { return inst_word_; }

  const SourceOpGetterMap &source_op_getters() { return source_op_getters_; }
  const DestOpGetterMap &dest_op_getters() { return dest_op_getters_; }
  const SimpleResourceGetterMap &simple_resource_getters() {
    return simple_resource_getters_;
  }
  const ComplexResourceGetterMap &complex_resource_getters() {
    return complex_resource_getters_;
  }

 private:
  std::string GetSimpleResourceName(SimpleResourceEnum resource_enum);

  RiscVState *state_;
  uint32_t inst_word_;
  OpcodeEnum opcode_;
  FormatEnum format_;
  SourceOpGetterMap source_op_getters_;
  DestOpGetterMap dest_op_getters_;
  SimpleResourceGetterMap simple_resource_getters_;
  ComplexResourceGetterMap complex_resource_getters_;
  generic::SimpleResourceDelayLine *resource_delay_line_ = nullptr;
  generic::SimpleResourcePool *resource_pool_ = nullptr;
};

}  // namespace mpact::sim::riscv::zvbb

#endif  // THIRD_PARTY_MPACT_RISCV_ZVBB_ENCODING_H_
