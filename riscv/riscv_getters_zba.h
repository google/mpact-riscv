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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_GETTERS_ZBA_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_GETTERS_ZBA_H_

#include "mpact/sim/generic/literal_operand.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_encoding_common.h"
#include "riscv/riscv_getter_helpers.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::operator*;  // NOLINT: is used below (clang error).

// The following function adds source operand getters to the given getter map.
// The function uses the template parameters to get the correct enum type
// for the instruction set being decoded. The Extractors parameter is used to
// get the correct instruction format extractor for the instruction set. The
// IntRegister and FpRegister parameters are used to get the correct register
// types for the instruction set.
template <typename Enum, typename Extractors, typename IntRegister,
          typename FpRegister>
void AddRiscVZbaSourceGetters(SourceOpGetterMap& getter_map,
                              RiscVEncodingCommon* common) {
  // Source operand getters.
  Insert(getter_map, *Enum::kConst1,
         []() { return new generic::IntLiteralOperand<1>(); });
  Insert(getter_map, *Enum::kConst2,
         []() { return new generic::IntLiteralOperand<2>(); });
  Insert(getter_map, *Enum::kConst3,
         []() { return new generic::IntLiteralOperand<3>(); });
}

template <typename Enum, typename Extractors, typename IntRegister,
          typename FpRegister>
void AddRiscVZbaDestGetters(DestOpGetterMap& getter_map,
                            RiscVEncodingCommon* common) {
  // Destination operand getters.
  // Empty.
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_GETTERS_ZBA_H_
