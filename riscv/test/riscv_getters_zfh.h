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

#ifndef THIRD_PARTY_MPACT_RISCV_TEST_RISCV_GETTERS_ZFH_H_
#define THIRD_PARTY_MPACT_RISCV_TEST_RISCV_GETTERS_ZFH_H_

#include <cstdint>
#include <string>

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

template <typename Enum, typename Extractors, typename IntegerRegister>
void AddRiscVZfhSourceScalarGetters(SourceOpGetterMap& getter_map,
                                    RiscVEncodingCommon* common) {
  // Source operand getters.
  Insert(getter_map, *Enum::kRs1, [common]() -> SourceOperandInterface* {
    int num = Extractors::Inst32Format::ExtractRs1(common->inst_word());
    if (num == 0) return new generic::IntLiteralOperand<0>({1});
    return GetRegisterSourceOp<IntegerRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, num),
        kXRegisterAliases[num]);
  });
  // I in IImm12 stands for IType instructions.
  Insert(getter_map, *Enum::kIImm12, [common]() -> SourceOperandInterface* {
    const auto num = Extractors::IType::ExtractImm12(common->inst_word());
    return new generic::ImmediateOperand<int32_t>(num);
  });
  // S in SImm12 stands for SType instructions.
  Insert(getter_map, *Enum::kSImm12, [common]() -> SourceOperandInterface* {
    const auto num = Extractors::SType::ExtractSImm(common->inst_word());
    return new generic::ImmediateOperand<int32_t>(num);
  });
  Insert(getter_map, *Enum::kRm, [common]() -> SourceOperandInterface* {
    const auto num =
        Extractors::Inst32Format::ExtractFunc3(common->inst_word());
    return new generic::ImmediateOperand<uint8_t>(num);
  });
}

template <typename Enum, typename Extractors, typename FloatRegister>
void AddRiscVZfhSourceFloatGetters(SourceOpGetterMap& getter_map,
                                   RiscVEncodingCommon* common) {
  // Source operand getters.
  Insert(getter_map, *Enum::kFrs1, [common]() -> SourceOperandInterface* {
    int num = Extractors::Inst32Format::ExtractRs1(common->inst_word());
    return GetRegisterSourceOp<FloatRegister>(
        common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
        kFRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kFrs2, [common]() -> SourceOperandInterface* {
    int num = Extractors::Inst32Format::ExtractRs2(common->inst_word());
    return GetRegisterSourceOp<FloatRegister>(
        common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
        kFRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kFrs3, [common]() -> SourceOperandInterface* {
    int num = Extractors::Inst32Format::ExtractRs3(common->inst_word());
    return GetRegisterSourceOp<FloatRegister>(
        common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
        kFRegisterAliases[num]);
  });
}

template <typename Enum, typename Extractors, typename IntegerRegister>
void AddRiscVZfhDestScalarGetters(DestOpGetterMap& getter_map,
                                  RiscVEncodingCommon* common) {
  // Destination operand getters.
  Insert(getter_map, *Enum::kRd,
         [common](int latency) -> DestinationOperandInterface* {
           auto num = Extractors::Inst32Format::ExtractRd(common->inst_word());
           std::string name = absl::StrCat(RiscVState::kXregPrefix, num);
           return mpact::sim::riscv::GetRegisterDestinationOp<IntegerRegister>(
               common->state(), name, latency);
         });
  Insert(getter_map, *Enum::kFflags,
         [common](int latency) -> DestinationOperandInterface* {
           return GetCSRSetBitsDestinationOp<uint32_t>(common->state(),
                                                       "fflags", latency, "");
         });
}

template <typename Enum, typename Extractors, typename FloatRegister>
void AddRiscVZfhDestFloatGetters(DestOpGetterMap& getter_map,
                                 RiscVEncodingCommon* common) {
  // Destination operand getters.
  Insert(getter_map, *Enum::kFrd,
         [common](int latency) -> DestinationOperandInterface* {
           auto num = Extractors::Inst32Format::ExtractRd(common->inst_word());
           std::string name = absl::StrCat(RiscVState::kFregPrefix, num);
           return mpact::sim::riscv::GetRegisterDestinationOp<FloatRegister>(
               common->state(), name, latency);
         });
}

// This function is used to add the simple resource getters to the given
// "getter map". The function is templated on the enum type that defines the
// simple resource types, the Extractors type that defines the bit
// extraction functions, and the IntRegister and FpRegister types that are used
// to construct the register operand.
template <typename Enum, typename Extractors>
void AddRiscVZfhSimpleResourceGetters(SimpleResourceGetterMap& getter_map,
                                      RiscVEncodingCommon* common) {
  // TODO(julianmb): Add resource getters when appropriate.
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // THIRD_PARTY_MPACT_RISCV_TEST_RISCV_GETTERS_ZFH_H_
