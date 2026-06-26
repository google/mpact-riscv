// Copyright 2026 Google LLC
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

// Operand getters specific to the Zvt (matrix) instructions. These supplement
// the RISC-V V getters: the Zvt encoding constructor calls both. The new
// operands are the A/B matrix register groups, the tile destination specifier,
// and a vector group destination for tile<->vector moves.

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_ZVT_GETTERS_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_ZVT_GETTERS_H_

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/operand_interface.h"
#include "mpact/sim/generic/register.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_encoding_common.h"
#include "riscv/riscv_getter_helpers.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"

namespace mpact::sim::riscv {

// Builds a source operand spanning `num_regs` consecutive vector registers
// starting at `reg_num` (clamped to the register file). Unlike the stock
// vs1/vs2 getters, the group size is explicit rather than inferred from the
// register-number alignment, which the strided matrix row layout requires.
template <typename RegType>
::mpact::sim::generic::SourceOperandInterface* GetVectorGroupSourceOp(
    ::mpact::sim::riscv::RiscVState* state, int reg_num, int num_regs) {
  std::vector<::mpact::sim::generic::RegisterBase*> group;
  group.reserve(std::min(num_regs, 32 - reg_num));
  for (int i = 0; i < num_regs && (reg_num + i) < 32; ++i) {
    auto [reg, unused] = state->GetRegister<RegType>(absl::StrCat(
        ::mpact::sim::riscv::RiscVState::kVregPrefix, reg_num + i));
    group.push_back(reg);
  }
  return new ::mpact::sim::riscv::RV32VectorSourceOperand(
      absl::Span<::mpact::sim::generic::RegisterBase*>(group),
      absl::StrCat(::mpact::sim::riscv::RiscVState::kVregPrefix, reg_num));
}

// Builds a destination operand spanning `num_regs` consecutive vector
// registers starting at `reg_num` (clamped to the register file).
template <typename RegType>
::mpact::sim::generic::DestinationOperandInterface* GetVectorGroupDestinationOp(
    ::mpact::sim::riscv::RiscVState* state, int latency, int reg_num,
    int num_regs) {
  std::vector<::mpact::sim::generic::RegisterBase*> group;
  group.reserve(std::min(num_regs, 32 - reg_num));
  for (int i = 0; i < num_regs && (reg_num + i) < 32; ++i) {
    auto [reg, unused] = state->GetRegister<RegType>(absl::StrCat(
        ::mpact::sim::riscv::RiscVState::kVregPrefix, reg_num + i));
    group.push_back(reg);
  }
  return new ::mpact::sim::riscv::RV32VectorDestinationOperand(
      absl::Span<::mpact::sim::generic::RegisterBase*>(group), latency,
      absl::StrCat(::mpact::sim::riscv::RiscVState::kVregPrefix, reg_num));
}

// The maximum vector-register window a matrix operand row group can span (one
// eight-register group, per spec 15.1.1.3).
inline constexpr int kRiscVZvtMatrixGroupRegs = 8;

template <typename SourceOpEnum, typename Extractors>
void AddRiscVZvtSourceGetters(
    ::mpact::sim::riscv::SourceOpGetterMap& getter_map,
    ::mpact::sim::riscv::RiscVEncodingCommon* common) {
  using ::mpact::sim::generic::ImmediateOperand;
  using ::mpact::sim::generic::SourceOperandInterface;
  using ::mpact::sim::generic::operator*;  // NOLINT
  using ::mpact::sim::riscv::Insert;
  using ::mpact::sim::riscv::RVVectorRegister;

  // A matrix operand: vector register group based at vs2.
  Insert(getter_map, *SourceOpEnum::kMatA,
         [common]() -> SourceOperandInterface* {
           int num = Extractors::VMatrix::ExtractVs2(common->inst_word());
           return GetVectorGroupSourceOp<RVVectorRegister>(
               common->state(), num, kRiscVZvtMatrixGroupRegs);
         });
  // B matrix operand: vector register group based at vs1.
  Insert(getter_map, *SourceOpEnum::kMatB,
         [common]() -> SourceOperandInterface* {
           int num = Extractors::VMatrix::ExtractVs1(common->inst_word());
           return GetVectorGroupSourceOp<RVVectorRegister>(
               common->state(), num, kRiscVZvtMatrixGroupRegs);
         });
  // Destination tile specifier (named tile index, 0..15) from the rd MSBs.
  Insert(getter_map, *SourceOpEnum::kMtd,
         [common]() -> SourceOperandInterface* {
           int tile = Extractors::VMatrix::ExtractMtile(common->inst_word());
           return new ImmediateOperand<uint32_t>(tile);
         });
}

template <typename DestOpEnum, typename Extractors>
void AddRiscVZvtDestGetters(::mpact::sim::riscv::DestOpGetterMap& getter_map,
                            ::mpact::sim::riscv::RiscVEncodingCommon* common) {
  using ::mpact::sim::generic::DestinationOperandInterface;
  using ::mpact::sim::generic::operator*;  // NOLINT
  using ::mpact::sim::riscv::Insert;
  using ::mpact::sim::riscv::RVVectorRegister;

  // Vector group destination based at vd, used by vtmv.v.t (tile -> vector).
  // vtmv.v.t uses the VArith field layout; vd is at the same bits in both.
  Insert(getter_map, *DestOpEnum::kMatVd,
         [common](int latency) -> DestinationOperandInterface* {
           int num = Extractors::VArith::ExtractVd(common->inst_word());
           return GetVectorGroupDestinationOp<RVVectorRegister>(
               common->state(), latency, num, kRiscVZvtMatrixGroupRegs);
         });
}

}  // namespace mpact::sim::riscv

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_ZVT_GETTERS_H_
