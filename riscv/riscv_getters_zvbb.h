// Copyright 2025 Google LLC
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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_GETTERS_ZVBB_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_GETTERS_ZVBB_H_

#include <cstdint>
#include <new>

#include "absl/strings/str_cat.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/literal_operand.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_encoding_common.h"
#include "riscv/riscv_getter_helpers.h"
#include "riscv/riscv_getters_vector.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_register_aliases.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::operator*;  // NOLINT: is used below (clang error).

// The following function adds source operand getters to the given getter map.
// The function uses the template parameters to get the correct enum type
// for the instruction set being decoded. The Extractors parameter is used to
// get the correct instruction format extractor for the instruction set.
template <typename Enum, typename Extractors, typename VectorRegister>
void AddRiscVZvbbSourceVectorGetters(SourceOpGetterMap &getter_map,
                                     RiscVEncodingCommon *common) {
  // Source operand getters.
  Insert(getter_map, *Enum::kVmask, [common]() -> SourceOperandInterface * {
    auto vm = Extractors::VArith::ExtractVm(common->inst_word());
    if (vm == 1) {
      // Unmasked, return the True mask.
      return new RV32VectorTrueOperand(common->state());
    }
    // Masked. Return the mask register.
    return mpact::sim::riscv::GetVectorMaskRegisterSourceOp<VectorRegister>(
        common->state(), 0);
  });
  Insert(getter_map, *Enum::kVs1, [common]() -> SourceOperandInterface * {
    auto num = Extractors::VArith::ExtractVs1(common->inst_word());
    return mpact::sim::riscv::GetVectorMaskRegisterSourceOp<VectorRegister>(
        common->state(), num);
  });
  Insert(getter_map, *Enum::kVs2, [common]() -> SourceOperandInterface * {
    auto num = Extractors::VArith::ExtractVs2(common->inst_word());
    return mpact::sim::riscv::GetVectorMaskRegisterSourceOp<VectorRegister>(
        common->state(), num);
  });
}

template <typename Enum, typename Extractors, typename IntegerRegister>
void AddRiscvZvbbSourceScalarGetters(SourceOpGetterMap &getter_map,
                                     RiscVEncodingCommon *common) {
  // Source operand getters.
  Insert(getter_map, *Enum::kRs1, [common]() -> SourceOperandInterface * {
    int num = Extractors::VArith::ExtractRs1(common->inst_word());
    if (num == 0) return new generic::IntLiteralOperand<0>({1});
    return GetRegisterSourceOp<IntegerRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, num),
        kXRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kUimm5, [common]() -> SourceOperandInterface * {
    const auto num = Extractors::VArith::ExtractUimm5(common->inst_word());
    return new generic::ImmediateOperand<int32_t>(num);
  });
  Insert(getter_map, *Enum::kUimm6, [common]() -> SourceOperandInterface * {
    const auto num = Extractors::VArith::ExtractUimm6(common->inst_word());
    return new generic::ImmediateOperand<int32_t>(num);
  });
}

template <typename Enum, typename Extractors, typename VectorRegister>
void AddRiscVZvbbDestGetters(DestOpGetterMap &getter_map,
                             RiscVEncodingCommon *common) {
  // Destination operand getters.
  Insert(getter_map, *Enum::kVd,
         [common](int latency) -> DestinationOperandInterface * {
           auto num = Extractors::VArith::ExtractVd(common->inst_word());
           return mpact::sim::riscv::GetVectorRegisterDestinationOp<
               VectorRegister>(common->state(), latency, num);
         });
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_GETTERS_ZVBB_H_
