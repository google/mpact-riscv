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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_GETTERS_RV64_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_GETTERS_RV64_H_

#include <cstdint>

#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/literal_operand.h"
#include "mpact/sim/generic/operand_interface.h"
#include "mpact/sim/generic/resource_operand_interface.h"
#include "riscv/riscv_encoding_common.h"
#include "riscv/riscv_getter_helpers.h"

// This file contains helper functions that are used to initialize "getter maps"
// with lambdas that create source, destination, and resource operands for
// operands only in RiscV64G.

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::DestinationOperandInterface;
using ::mpact::sim::generic::ImmediateOperand;
using ::mpact::sim::generic::IntLiteralOperand;
using ::mpact::sim::generic::ResourceOperandInterface;
using ::mpact::sim::generic::SourceOperandInterface;

// The following function adds source operand getters to the given getter map.
// The function uses the template parameters to get the correct enum type
// for the instruction set being decoded. The Extractors parameter is used to
// get the correct instruction format extractor for the instruction set. The
// IntRegister and FpRegister parameters are used to get the correct register
// types for the instruction set.
template <typename Enum, typename Extractors, typename IntRegister,
          typename FpRegister>
void AddRiscV64SourceGetters(SourceOpGetterMap &getter_map,
                             RiscVEncodingCommon *common) {
  // Source operand getters.
  Insert(getter_map, *Enum::kIUimm6, [common]() {
    return new ImmediateOperand<uint32_t>(
        Extractors::Inst32Format::ExtractRUimm6(common->inst_word()));
  });
}

// This function is used to add the destination operand getters to the given
// "getter map". The function is templated on the enum type that defines the
// destination operand types, the Extractors type that defines the bit
// extraction functions, and the IntRegister and FpRegister types that are used
// to construct the register operand.
template <typename Enum, typename Extractors, typename IntRegister,
          typename FpRegister>
void AddRiscV64DestGetters(DestOpGetterMap &getter_map,
                           RiscVEncodingCommon *common) {
  // Empty.
}

// This function is used to add the simple resource getters to the given
// "getter map". The function is templated on the enum type that defines the
// simple resource types, the Extractors type that defines the bit
// extraction functions, and the IntRegister and FpRegister types that are used
// to construct the register operand.
template <typename Enum, typename Extractors>
void AddRiscV64SimpleResourceGetters(SimpleResourceGetterMap &getter_map,
                                     RiscVEncodingCommon *common) {
  // Empty.
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_GETTERS_RV64_H_
