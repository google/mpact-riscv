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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_GETTERS_RV32_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_GETTERS_RV32_H_

#include "absl/strings/str_cat.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/literal_operand.h"
#include "mpact/sim/generic/operand_interface.h"
#include "mpact/sim/generic/resource_operand_interface.h"
#include "mpact/sim/generic/simple_resource.h"
#include "riscv/riscv_encoding_common.h"
#include "riscv/riscv_getter_helpers.h"
#include "riscv/riscv_register_aliases.h"
#include "riscv/riscv_state.h"

// This file contains helper functions that are used to initialize "getter maps"
// with lambdas that create source, destination, and resource operands for the
// RiscV32G and RiscV32GZB instruction set decoders. The functions are templated
// so they can be used with different decoders with different sets of
// instruction enumerations and register types.

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
void AddRiscV32SourceGetters(SourceOpGetterMap& getter_map,
                             RiscVEncodingCommon* common) {
  // Source operand getters.
  Insert(getter_map, *Enum::kC3frs2, [common]() {
    auto num = Extractors::CS::ExtractCsRs2(common->inst_word());
    return GetRegisterSourceOp<FpRegister>(
        common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
        kFRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kCfrs2, [common]() {
    auto num = Extractors::CR::ExtractRs2(common->inst_word());
    return GetRegisterSourceOp<FpRegister>(
        common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
        kFRegisterAliases[num]);
  });
}

// This function is used to add the destination operand getters to the given
// "getter map". The function is templated on the enum type that defines the
// destination operand types, the Extractors type that defines the bit
// extraction functions, and the IntRegister and FpRegister types that are used
// to construct the register operand.
template <typename Enum, typename Extractors, typename IntRegister,
          typename FpRegister>
void AddRiscV32DestGetters(DestOpGetterMap& getter_map,
                           RiscVEncodingCommon* common) {
  // Destination operand getters.
  Insert(getter_map, *Enum::kC3frd, [common](int latency) {
    int num = Extractors::CL::ExtractClRd(common->inst_word());
    return GetRegisterDestinationOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kFregPrefix, num), latency);
  });
}

// This function is used to add the simple resource getters to the given
// "getter map". The function is templated on the enum type that defines the
// simple resource types, the Extractors type that defines the bit
// extraction functions, and the IntRegister and FpRegister types that are used
// to construct the register operand.
template <typename Enum, typename Extractors>
void AddRiscV32SimpleResourceGetters(SimpleResourceGetterMap& getter_map,
                                     RiscVEncodingCommon* common) {
  Insert(getter_map, *Enum::kC3frd, [common]() -> generic::SimpleResource* {
    int num = Extractors::CL::ExtractClRd(common->inst_word());
    return common->resource_pool()->GetOrAddResource(absl::StrCat("d", num));
  });
  Insert(getter_map, *Enum::kC3frs2, [common]() -> generic::SimpleResource* {
    int num = Extractors::CS::ExtractCsRs2(common->inst_word());
    return common->resource_pool()->GetOrAddResource(absl::StrCat("d", num));
  });
  Insert(getter_map, *Enum::kCfrs2, [common]() -> generic::SimpleResource* {
    int num = Extractors::CR::ExtractRs2(common->inst_word());
    return common->resource_pool()->GetOrAddResource(absl::StrCat("d", num));
  });
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_GETTERS_RV32_H_
