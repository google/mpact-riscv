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

#ifndef MPACT_RISCV_RISCV_RISCV64G_VEC_ENCODING_H_
#define MPACT_RISCV_RISCV_RISCV64G_VEC_ENCODING_H_
#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "mpact/sim/generic/simple_resource.h"
#include "mpact/sim/generic/simple_resource_operand.h"
#include "riscv/riscv64gv_decoder.h"
#include "riscv/riscv64gv_enums.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {
namespace isa64v {

// This class provides the interface between the generated instruction decoder
// framework (which is agnostic of the actual bit representation of
// instructions) and the instruction representation. This class provides methods
// to return the opcode, source operands, and destination operands for
// instructions according to the operand fields in the encoding.
class RiscV64GVecEncoding : public RiscV64GVEncodingBase {
 public:
  using SourceOpGetterMap =
      absl::flat_hash_map<int, absl::AnyInvocable<SourceOperandInterface*()>>;
  using DestOpGetterMap = absl::flat_hash_map<
      int, absl::AnyInvocable<DestinationOperandInterface*(int)>>;
  using SimpleResourceGetterMap =
      absl::flat_hash_map<int, absl::AnyInvocable<generic::SimpleResource*()>>;
  using ComplexResourceGetterMap = absl::flat_hash_map<
      int, absl::AnyInvocable<ResourceOperandInterface*(int, int)>>;

  static constexpr int kParseGroup32Size = 32;
  static constexpr int kParseGroup16Size = 32;

  explicit RiscV64GVecEncoding(RiscVState* state);
  RiscV64GVecEncoding(RiscVState* state, bool use_abi_names);
  ~RiscV64GVecEncoding() override;

  // Parses an instruction and determines the opcode.
  void ParseInstruction(uint32_t inst_word);

  // RiscV64 has a single slot type and single entry, so the following methods
  // ignore the parameters.

  // Returns the opcode in the current instruction representation.
  OpcodeEnum GetOpcode(SlotEnum, int) override { return opcode_; }

  // There is no predicate, so return nullptr.
  PredicateOperandInterface* GetPredicate(SlotEnum, int, OpcodeEnum,
                                          PredOpEnum) override {
    return nullptr;
  }

  // Return the resource operand corresponding to the resource enum. If argument
  // is not kNone, it means that the resource enum is a pool of resources and
  // the resource element from the pool is specified by the
  // ResourceArgumentEnum. This is used for instance for register resources,
  // where the resource itself is a register bank, and the argument specifies
  // which register (or more precisely) which encoding "field" specifies the
  // register number.
  ResourceOperandInterface* GetSimpleResourceOperand(
      SlotEnum, int, OpcodeEnum, SimpleResourceVector& resource_vec,
      int end) override;

  ResourceOperandInterface* GetComplexResourceOperand(
      SlotEnum, int, OpcodeEnum, ComplexResourceEnum resource, int begin,
      int end) override;

  // The following method returns a source operand that corresponds to the
  // particular operand field.
  SourceOperandInterface* GetSource(SlotEnum, int, OpcodeEnum, SourceOpEnum op,
                                    int source_no) override;

  // The following method returns a destination operand that corresponds to the
  // particular operand field.
  DestinationOperandInterface* GetDestination(SlotEnum, int, OpcodeEnum,
                                              DestOpEnum op, int dest_no,
                                              int latency) override;
  // This method returns latency for any destination operand for which the
  // latency specifier in the .isa file is '*'. Since there are none, just
  // return 0.
  int GetLatency(SlotEnum, int, OpcodeEnum, DestOpEnum, int) override {
    return 0;
  }

  // Getter.

  const SourceOpGetterMap& source_op_getters() { return source_op_getters_; }
  const DestOpGetterMap& dest_op_getters() { return dest_op_getters_; }
  const SimpleResourceGetterMap& simple_resource_getters() {
    return simple_resource_getters_;
  }
  const ComplexResourceGetterMap& complex_resource_getters() {
    return complex_resource_getters_;
  }

 protected:
  // Architectural names for the integer registers.
  const std::string xreg_names_[32] = {
      "x0",  "x1",  "x2",  "x3",  "x4",  "x5",  "x6",  "x7",
      "x8",  "x9",  "x10", "x11", "x12", "x13", "x14", "x15",
      "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23",
      "x24", "x25", "x26", "x27", "x28", "x29", "x30", "x31"};
  // ABI names for the integer registers.
  const std::string xreg_abi_names_[32] = {
      "zero", "ra", "sp", "gp", "tp",  "t0",  "t1", "t2", "s0", "s1", "a0",
      "a1",   "a2", "a3", "a4", "a5",  "a6",  "a7", "s2", "s3", "s4", "s5",
      "s6",   "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"};
  // Architectural names for the floating point registers.
  const std::string freg_names_[32] = {
      "f0",  "f1",  "f2",  "f3",  "f4",  "f5",  "f6",  "f7",
      "f8",  "f9",  "f10", "f11", "f12", "f13", "f14", "f15",
      "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23",
      "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31"};
  // ABI names for the floating point registers.
  const std::string freg_abi_names_[32] = {
      "ft0", "ft1", "ft2",  "ft3",  "ft4", "ft5", "ft6",  "ft7",
      "fs0", "fs1", "fa0",  "fa1",  "fa2", "fa3", "fa4",  "fa5",
      "fa6", "fa7", "fs2",  "fs3",  "fs4", "fs5", "fs6",  "fs7",
      "fs8", "fs9", "fs10", "fs11", "ft8", "ft9", "ft10", "ft11"};

  RiscVState* state() const { return state_; }
  OpcodeEnum opcode() const { return opcode_; }
  uint32_t inst_word() const { return inst_word_; }
  generic::SimpleResourcePool* resource_pool() const { return resource_pool_; }

 private:
  std::string GetSimpleResourceName(SimpleResourceEnum resource_enum);
  // These three methods initialize the source and destination operand getter
  // arrays, and the complex resource getter array.
  void InitializeSourceOperandGetters();
  void InitializeDestinationOperandGetters();
  void InitializeVectorSourceOperandGetters();
  void InitializeVectorDestinationOperandGetters();
  void InitializeComplexResourceOperandGetters();
  void InitializeSimpleResourceGetters();

  // Pointer to the register names to be used. This will either point to
  // xreg_names or xreg_abi_names.
  const std::string* xreg_alias_;

  SourceOpGetterMap source_op_getters_;
  DestOpGetterMap dest_op_getters_;
  SimpleResourceGetterMap simple_resource_getters_;
  ComplexResourceGetterMap complex_resource_getters_;
  RiscVState* state_;
  uint32_t inst_word_;
  OpcodeEnum opcode_;
  generic::SimpleResourceDelayLine* resource_delay_line_ = nullptr;
  generic::SimpleResourcePool* resource_pool_ = nullptr;
};

}  // namespace isa64v
}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV64G_VEC_ENCODING_H_
