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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_GETTERS_VECTOR_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_GETTERS_VECTOR_H_

#include <cstdint>
#include <new>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/literal_operand.h"
#include "mpact/sim/generic/operand_interface.h"
#include "mpact/sim/generic/register.h"
#include "mpact/sim/generic/resource_operand_interface.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_encoding_common.h"
#include "riscv/riscv_getter_helpers.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::DestinationOperandInterface;
using ::mpact::sim::generic::ImmediateOperand;
using ::mpact::sim::generic::IntLiteralOperand;
using ::mpact::sim::generic::ResourceOperandInterface;
using ::mpact::sim::generic::SourceOperandInterface;
using ::mpact::sim::generic::operator*;  // NOLINT: is used below.

constexpr int kNumRegTable[8] = {8, 1, 2, 1, 4, 1, 2, 1};

template <typename RegType>
inline void GetVRegGroup(RiscVState *state, int reg_num,
                         std::vector<generic::RegisterBase *> *vreg_group) {
  // The number of registers in a vector register group depends on the register
  // index: 0, 8, 16, 24 each have 8 registers, 4, 12, 20, 28 each have 4,
  // 2, 6, 10, 14, 18, 22, 26, 30 each have two, and all odd numbered register
  // groups have only 1.
  int num_regs = kNumRegTable[reg_num % 8];
  for (int i = 0; i < num_regs; i++) {
    auto vreg_name = absl::StrCat(RiscVState::kVregPrefix, reg_num + i);
    vreg_group->push_back(state->GetRegister<RegType>(vreg_name).first);
  }
}
template <typename RegType>
inline SourceOperandInterface *GetVectorRegisterSourceOp(RiscVState *state,
                                                         int reg_num) {
  std::vector<generic::RegisterBase *> vreg_group;
  GetVRegGroup<RegType>(state, reg_num, &vreg_group);
  auto *v_src_op = new RV32VectorSourceOperand(
      absl::Span<generic::RegisterBase *>(vreg_group),
      absl::StrCat(RiscVState::kVregPrefix, reg_num));
  return v_src_op;
}

template <typename RegType>
inline DestinationOperandInterface *GetVectorRegisterDestinationOp(
    RiscVState *state, int latency, int reg_num) {
  std::vector<generic::RegisterBase *> vreg_group;
  GetVRegGroup<RegType>(state, reg_num, &vreg_group);
  auto *v_dst_op = new RV32VectorDestinationOperand(
      absl::Span<generic::RegisterBase *>(vreg_group), latency,
      absl::StrCat(RiscVState::kVregPrefix, reg_num));
  return v_dst_op;
}

template <typename RegType>
inline SourceOperandInterface *GetVectorMaskRegisterSourceOp(RiscVState *state,
                                                             int reg_num) {
  // Mask register groups only have a single register.
  std::vector<generic::RegisterBase *> vreg_group;
  vreg_group.push_back(
      state
          ->GetRegister<RegType>(absl::StrCat(RiscVState::kVregPrefix, reg_num))
          .first);
  auto *v_src_op = new RV32VectorSourceOperand(
      absl::Span<generic::RegisterBase *>(vreg_group),
      absl::StrCat(RiscVState::kVregPrefix, reg_num));
  return v_src_op;
}

template <typename RegType>
inline DestinationOperandInterface *GetVectorMaskRegisterDestinationOp(
    RiscVState *state, int latency, int reg_num) {
  // Mask register groups only have a single register.
  std::vector<generic::RegisterBase *> vreg_group;
  vreg_group.push_back(
      state
          ->GetRegister<RegType>(absl::StrCat(RiscVState::kVregPrefix, reg_num))
          .first);
  auto *v_dst_op = new RV32VectorDestinationOperand(
      absl::Span<generic::RegisterBase *>(vreg_group), latency,
      absl::StrCat(RiscVState::kVregPrefix, reg_num));
  return v_dst_op;
}

// The following function adds source operand getters to the given getter map.
// The function uses the template parameters to get the correct enum type
// for the instruction set being decoded. The Extractors parameter is used to
// get the correct instruction format extractor for the instruction set.
template <typename Enum, typename Extractors, typename VectorRegister>
void AddRiscVVectorSourceGetters(SourceOpGetterMap &getter_map,
                                 RiscVEncodingCommon *common) {
  // Source operand getters.
  Insert(getter_map, *Enum::kVd, [common]() -> SourceOperandInterface * {
    auto num = Extractors::VArith::ExtractVd(common->inst_word());
    return GetVectorRegisterSourceOp<VectorRegister>(common->state(), num);
  });
  Insert(getter_map, *Enum::kVmask, [common]() -> SourceOperandInterface * {
    auto vm = Extractors::VArith::ExtractVm(common->inst_word());
    if (vm == 1) {
      // Unmasked, return the True mask.
      return new RV32VectorTrueOperand(common->state());
    }
    // Masked. Return the mask register.
    return GetVectorMaskRegisterSourceOp<VectorRegister>(common->state(), 0);
  });
  Insert(getter_map, *Enum::kVmaskTrue, [common]() -> SourceOperandInterface * {
    return new RV32VectorTrueOperand(common->state());
  });
  Insert(getter_map, *Enum::kVm, [common]() -> SourceOperandInterface * {
    auto vm = Extractors::VArith::ExtractVm(common->inst_word());
    return new generic::ImmediateOperand<bool>(
        vm, absl::StrCat("vm.", vm ? "t" : "f"));
  });
  Insert(getter_map, *Enum::kVs1, [common]() -> SourceOperandInterface * {
    auto num = Extractors::VArith::ExtractVs1(common->inst_word());
    return GetVectorRegisterSourceOp<VectorRegister>(common->state(), num);
  });
  Insert(getter_map, *Enum::kVs2, [common]() -> SourceOperandInterface * {
    auto num = Extractors::VArith::ExtractVs2(common->inst_word());
    return GetVectorRegisterSourceOp<VectorRegister>(common->state(), num);
  });
  Insert(getter_map, *Enum::kVs3, [common]() -> SourceOperandInterface * {
    auto num = Extractors::VMem::ExtractVs3(common->inst_word());
    return GetVectorRegisterSourceOp<VectorRegister>(common->state(), num);
  });

  Insert(getter_map, *Enum::kSimm5, [common]() -> SourceOperandInterface * {
    const auto num =
        Extractors::Inst32Format::ExtractSimm5(common->inst_word());
    return new generic::ImmediateOperand<int32_t>(num);
  });

  Insert(getter_map, *Enum::kUimm5, [common]() -> SourceOperandInterface * {
    const auto num =
        Extractors::Inst32Format::ExtractUimm5(common->inst_word());
    return new generic::ImmediateOperand<int32_t>(num);
  });

  Insert(getter_map, *Enum::kZimm10, [common]() -> SourceOperandInterface * {
    const auto num =
        Extractors::Inst32Format::ExtractZimm10(common->inst_word());
    return new generic::ImmediateOperand<int32_t>(num);
  });

  Insert(getter_map, *Enum::kZimm11, [common]() -> SourceOperandInterface * {
    const auto num =
        Extractors::Inst32Format::ExtractZimm11(common->inst_word());
    return new generic::ImmediateOperand<int32_t>(num);
  });

  Insert(getter_map, *Enum::kConst1, []() -> SourceOperandInterface * {
    return new generic::ImmediateOperand<int32_t>(1);
  });

  Insert(getter_map, *Enum::kConst2, []() -> SourceOperandInterface * {
    return new generic::ImmediateOperand<int32_t>(2);
  });

  Insert(getter_map, *Enum::kConst4, []() -> SourceOperandInterface * {
    return new generic::ImmediateOperand<int32_t>(4);
  });

  Insert(getter_map, *Enum::kConst8, []() -> SourceOperandInterface * {
    return new generic::ImmediateOperand<int32_t>(8);
  });

  Insert(getter_map, *Enum::kNf, [common]() -> SourceOperandInterface * {
    auto num_fields = Extractors::VMem::ExtractNf(common->inst_word());
    return new generic::ImmediateOperand<uint8_t>(num_fields,
                                                  absl::StrCat(num_fields + 1));
  });
  Insert(getter_map, *Enum::kFs1, [common]() -> SourceOperandInterface * {
    const int num = Extractors::VArith::ExtractRs1(common->inst_word());
    return GetRegisterSourceOp<RV64Register>(common->state(),
                                             std::string(kFregNames[num]),
                                             std::string(kFregAbiNames[num]));
  });
}

// This function is used to add the destination operand getters to the given
// "getter map". The function is templated on the enum type that defines the
// destination operand types, the Extractors type that defines the bit
// extraction functions, and the IntRegister and FpRegister types that are used
// to construct the register operand.
template <typename Enum, typename Extractors, typename VectorRegister>
void AddRiscVVectorDestGetters(DestOpGetterMap &getter_map,
                               RiscVEncodingCommon *common) {
  // Destination operand getters.
  Insert(getter_map, *Enum::kVd,
         [common](int latency) -> DestinationOperandInterface * {
           auto num = Extractors::VArith::ExtractVd(common->inst_word());
           return GetVectorRegisterDestinationOp<VectorRegister>(
               common->state(), latency, num);
         });
  Insert(getter_map, *Enum::kFd,
         [common](int latency) -> DestinationOperandInterface * {
           const int num = Extractors::VArith::ExtractRd(common->inst_word());
           return GetRegisterDestinationOp<RV64Register>(
               common->state(), std::string(kFregNames[num]), latency,
               std::string(kFregAbiNames[num]));
         });
}

// This function is used to add the simple resource getters to the given
// "getter map". The function is templated on the enum type that defines the
// simple resource types, the Extractors type that defines the bit
// extraction functions, and the IntRegister and FpRegister types that are used
// to construct the register operand.
template <typename Enum, typename Extractors>
void AddRiscVVectorSimpleResourceGetters(SimpleResourceGetterMap &getter_map,
                                         RiscVEncodingCommon *common) {
  // TODO(torerik): Add resource getters when appropriate.
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_GETTERS_VECTOR_H_
