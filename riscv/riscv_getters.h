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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_GETTERS_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_GETTERS_H_

#include <cstdint>
#include <string>

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
void AddRiscVSourceGetters(SourceOpGetterMap &getter_map,
                           RiscVEncodingCommon *common) {
  // Source operand getters.
  Insert(getter_map, *Enum::kAAq, [common]() -> SourceOperandInterface * {
    if (Extractors::Inst32Format::ExtractAq(common->inst_word())) {
      return new IntLiteralOperand<1>();
    }
    return new IntLiteralOperand<0>();
  });
  Insert(getter_map, *Enum::kARl, [common]() -> SourceOperandInterface * {
    if (Extractors::Inst32Format::ExtractRl(common->inst_word())) {
      return new generic::IntLiteralOperand<1>();
    }
    return new generic::IntLiteralOperand<0>();
  });
  Insert(getter_map, *Enum::kBImm12, [common]() {
    return new ImmediateOperand<int32_t>(
        Extractors::Inst32Format::ExtractBImm(common->inst_word()));
  });
  Insert(getter_map, *Enum::kC3drs2, [common]() {
    auto num = Extractors::CS::ExtractCsRs2(common->inst_word());
    return GetRegisterSourceOp<FpRegister>(
        common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
        kXRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kC3rs1, [common]() {
    auto num = Extractors::CS::ExtractCsRs1(common->inst_word());
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, num),
        kXRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kC3rs2, [common]() {
    auto num = Extractors::CS::ExtractCsRs2(common->inst_word());
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, num),
        kXRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kCSRUimm5, [common]() {
    return new ImmediateOperand<uint32_t>(
        Extractors::Inst32Format::ExtractIUimm5(common->inst_word()));
  });
  Insert(getter_map, *Enum::kCdrs2, [common]() {
    auto num = Extractors::CR::ExtractRs2(common->inst_word());
    return GetRegisterSourceOp<FpRegister>(
        common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
        kXRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kCrs1, [common]() {
    auto num = Extractors::CR::ExtractRs1(common->inst_word());
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, num),
        kXRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kCrs2, [common]() {
    auto num = Extractors::CR::ExtractRs2(common->inst_word());
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, num),
        kXRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kCsr, [common]() {
    auto csr_indx = Extractors::IType::ExtractUImm12(common->inst_word());
    auto res = common->state()->csr_set()->GetCsr(csr_indx);
    if (!res.ok()) {
      return new ImmediateOperand<uint32_t>(csr_indx);
    }
    auto *csr = res.value();
    return new ImmediateOperand<uint32_t>(csr_indx, csr->name());
  });
  Insert(getter_map, *Enum::kDrs1, [common]() -> SourceOperandInterface * {
    int num = Extractors::RType::ExtractRs1(common->inst_word());
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
        kXRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kDrs2, [common]() -> SourceOperandInterface * {
    int num = Extractors::RType::ExtractRs2(common->inst_word());
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
        kXRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kDrs3, [common]() -> SourceOperandInterface * {
    int num = Extractors::R4Type::ExtractRs3(common->inst_word());
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
        kXRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kFrs1, [common]() -> SourceOperandInterface * {
    int num = Extractors::RType::ExtractRs1(common->inst_word());
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
        kXRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kFrs2, [common]() -> SourceOperandInterface * {
    int num = Extractors::RType::ExtractRs2(common->inst_word());
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
        kXRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kFrs3, [common]() -> SourceOperandInterface * {
    int num = Extractors::R4Type::ExtractRs3(common->inst_word());
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
        kXRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kICbImm8, [common]() {
    return new ImmediateOperand<int32_t>(
        Extractors::Inst16Format::ExtractBimm(common->inst_word()));
  });
  Insert(getter_map, *Enum::kICiImm6, [common]() {
    return new ImmediateOperand<int32_t>(
        Extractors::CI::ExtractImm6(common->inst_word()));
  });
  Insert(getter_map, *Enum::kICiImm612, [common]() {
    return new ImmediateOperand<int32_t>(
        Extractors::Inst16Format::ExtractImm18(common->inst_word()));
  });
  Insert(getter_map, *Enum::kICiImm6x16, [common]() {
    return new generic::ImmediateOperand<int32_t>(
        Extractors::Inst16Format::ExtractCiImm10(common->inst_word()));
  });
  Insert(getter_map, *Enum::kICiUimm6, [common]() {
    return new ImmediateOperand<uint32_t>(
        Extractors::Inst16Format::ExtractUimm6(common->inst_word()));
  });
  Insert(getter_map, *Enum::kICiUimm6x4, [common]() {
    return new ImmediateOperand<uint32_t>(
        Extractors::Inst16Format::ExtractCiImmW(common->inst_word()));
  });
  Insert(getter_map, *Enum::kICiUimm6x8, [common]() {
    return new ImmediateOperand<uint32_t>(
        Extractors::Inst16Format::ExtractCiImmD(common->inst_word()));
  });
  Insert(getter_map, *Enum::kICiwUimm8x4, [common]() {
    return new ImmediateOperand<uint32_t>(
        Extractors::Inst16Format::ExtractCiwImm10(common->inst_word()));
  });
  Insert(getter_map, *Enum::kICjImm11, [common]() {
    return new ImmediateOperand<int32_t>(
        Extractors::Inst16Format::ExtractJimm(common->inst_word()));
  });
  Insert(getter_map, *Enum::kIClUimm5x4, [common]() {
    return new ImmediateOperand<uint32_t>(
        Extractors::Inst16Format::ExtractClImmW(common->inst_word()));
  });
  Insert(getter_map, *Enum::kIClUimm5x8, [common]() {
    return new ImmediateOperand<uint32_t>(
        Extractors::Inst16Format::ExtractClImmD(common->inst_word()));
  });
  Insert(getter_map, *Enum::kICssUimm6x4, [common]() {
    return new ImmediateOperand<uint32_t>(
        Extractors::Inst16Format::ExtractCssImmW(common->inst_word()));
  });
  Insert(getter_map, *Enum::kICssUimm6x8, [common]() {
    return new ImmediateOperand<uint32_t>(
        Extractors::Inst16Format::ExtractCssImmD(common->inst_word()));
  });
  Insert(getter_map, *Enum::kIImm12, [common]() {
    return new ImmediateOperand<int32_t>(
        Extractors::Inst32Format::ExtractImm12(common->inst_word()));
  });
  Insert(getter_map, *Enum::kIUimm5, [common]() {
    return new ImmediateOperand<uint32_t>(
        Extractors::Inst32Format::ExtractRUimm5(common->inst_word()));
  });
  Insert(getter_map, *Enum::kJImm12, [common]() {
    return new ImmediateOperand<int32_t>(
        Extractors::Inst32Format::ExtractImm12(common->inst_word()));
  });
  Insert(getter_map, *Enum::kJImm20, [common]() {
    return new ImmediateOperand<int32_t>(
        Extractors::Inst32Format::ExtractJImm(common->inst_word()));
  });
  Insert(getter_map, *Enum::kPred, [common]() {
    return new ImmediateOperand<uint32_t>(
        Extractors::Fence::ExtractPred(common->inst_word()));
  });
  Insert(getter_map, *Enum::kRd, [common]() -> SourceOperandInterface * {
    int num = Extractors::RType::ExtractRd(common->inst_word());
    if (num == 0) return new generic::IntLiteralOperand<0>({1});
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, num),
        kXRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kRm, [common]() -> SourceOperandInterface * {
    int rm = Extractors::RType::ExtractFunc3(common->inst_word());
    switch (rm) {
      case 0:
        return new generic::IntLiteralOperand<0>();
      case 1:
        return new generic::IntLiteralOperand<1>();
      case 2:
        return new generic::IntLiteralOperand<2>();
      case 3:
        return new generic::IntLiteralOperand<3>();
      case 4:
        return new generic::IntLiteralOperand<4>();
      case 5:
        return new generic::IntLiteralOperand<5>();
      case 6:
        return new generic::IntLiteralOperand<6>();
      case 7:
        return new generic::IntLiteralOperand<7>();
      default:
        return nullptr;
    }
  });
  Insert(getter_map, *Enum::kRs1, [common]() -> SourceOperandInterface * {
    int num = Extractors::RType::ExtractRs1(common->inst_word());
    if (num == 0) return new generic::IntLiteralOperand<0>({1});
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, num),
        kXRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kRs2, [common]() -> SourceOperandInterface * {
    int num = Extractors::RType::ExtractRs2(common->inst_word());
    if (num == 0) return new generic::IntLiteralOperand<0>({1});
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, num),
        kXRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kSImm12, [common]() {
    return new ImmediateOperand<int32_t>(
        Extractors::SType::ExtractSImm(common->inst_word()));
  });
  Insert(getter_map, *Enum::kSucc, [common]() {
    return new ImmediateOperand<uint32_t>(
        Extractors::Fence::ExtractSucc(common->inst_word()));
  });
  Insert(getter_map, *Enum::kUImm20, [common]() {
    return new ImmediateOperand<int32_t>(
        Extractors::Inst32Format::ExtractUImm(common->inst_word()));
  });
  Insert(getter_map, *Enum::kX0,
         []() { return new generic::IntLiteralOperand<0>({1}); });
  Insert(getter_map, *Enum::kX2, [common]() {
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, 2),
        kXRegisterAliases[2]);
  });
}

// This function is used to add the destination operand getters to the given
// "getter map". The function is templated on the enum type that defines the
// destination operand types, the Extractors type that defines the bit
// extraction functions, and the IntRegister and FpRegister types that are used
// to construct the register operand.
template <typename Enum, typename Extractors, typename IntRegister,
          typename FpRegister>
void AddRiscVDestGetters(DestOpGetterMap &getter_map,
                         RiscVEncodingCommon *common) {
  // Destination operand getters.
  Insert(getter_map, *Enum::kC3drd, [common](int latency) {
    int num = Extractors::CL::ExtractClRd(common->inst_word());
    return GetRegisterDestinationOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kFregPrefix, num), latency);
  });
  Insert(getter_map, *Enum::kC3rd, [common](int latency) {
    int num = Extractors::CL::ExtractClRd(common->inst_word());
    return GetRegisterDestinationOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, num), latency,
        kXRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kC3rs1, [common](int latency) {
    int num = Extractors::CL::ExtractClRs1(common->inst_word());
    return GetRegisterDestinationOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, num), latency,
        kXRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kCsr, [common](int latency) {
    return GetRegisterDestinationOp<IntRegister>(common->state(),
                                                 RiscVState::kCsrName, latency);
  });
  Insert(getter_map, *Enum::kDrd,
         [common](int latency) -> DestinationOperandInterface * {
           int num = Extractors::RType::ExtractRd(common->inst_word());
           return GetRegisterDestinationOp<IntRegister>(
               common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
               latency);
         });
  Insert(getter_map, *Enum::kFrd,
         [common](int latency) -> DestinationOperandInterface * {
           int num = Extractors::RType::ExtractRd(common->inst_word());
           return GetRegisterDestinationOp<IntRegister>(
               common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
               latency);
         });
  Insert(getter_map, *Enum::kNextPc,
         [common](int latency) -> DestinationOperandInterface * {
           return GetRegisterDestinationOp<IntRegister>(
               common->state(), RiscVState::kPcName, latency);
         });
  Insert(getter_map, *Enum::kRd,
         [common](int latency) -> DestinationOperandInterface * {
           int num = Extractors::RType::ExtractRd(common->inst_word());
           if (num == 0) {
             return GetRegisterDestinationOp<IntRegister>(common->state(),
                                                          "X0Dest", 0);
           } else {
             return GetRegisterDestinationOp<IntRegister>(
                 common->state(), absl::StrCat(RiscVState::kXregPrefix, num),
                 latency, kXRegisterAliases[num]);
           }
         });
  Insert(getter_map, *Enum::kX0,
         [common](int latency) -> DestinationOperandInterface * {
           return GetRegisterDestinationOp<IntRegister>(
               common->state(), "X0Dest", 0, kXRegisterAliases[0]);
         });
  Insert(getter_map, *Enum::kX1, [common](int latency) {
    return GetRegisterDestinationOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, 1), latency,
        kXRegisterAliases[1]);
  });
  Insert(getter_map, *Enum::kX2, [common](int latency) {
    return GetRegisterDestinationOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, 2), latency,
        kXRegisterAliases[2]);
  });
  Insert(getter_map, *Enum::kFflags, [common](int latency) {
    return GetCSRSetBitsDestinationOp<uint32_t>(common->state(), "fflags",
                                                latency, "");
  });
}

// This function is used to add the simple resource getters to the given
// "getter map". The function is templated on the enum type that defines the
// simple resource types, the Extractors type that defines the bit
// extraction functions, and the IntRegister and FpRegister types that are used
// to construct the register operand.
template <typename Enum, typename Extractors>
void AddRiscVSimpleResourceGetters(SimpleResourceGetterMap &getter_map,
                                   RiscVEncodingCommon *common) {
  Insert(getter_map, *Enum::kC3drd, [common]() -> generic::SimpleResource * {
    int num = Extractors::CL::ExtractClRd(common->inst_word());
    return common->resource_pool()->GetOrAddResource(absl::StrCat("d", num));
  });
  Insert(getter_map, *Enum::kC3drs2, [common]() -> generic::SimpleResource * {
    int num = Extractors::CS::ExtractCsRs2(common->inst_word());
    return common->resource_pool()->GetOrAddResource(absl::StrCat("d", num));
  });
  Insert(getter_map, *Enum::kC3rd, [common]() -> generic::SimpleResource * {
    int num = Extractors::CL::ExtractClRd(common->inst_word());
    return common->resource_pool()->GetOrAddResource(absl::StrCat("x", num));
  });
  Insert(getter_map, *Enum::kC3rs1, [common]() -> generic::SimpleResource * {
    int num = Extractors::CS::ExtractCsRs1(common->inst_word());
    return common->resource_pool()->GetOrAddResource(absl::StrCat("x", num));
  });
  Insert(getter_map, *Enum::kC3rs2, [common]() -> generic::SimpleResource * {
    int num = Extractors::CS::ExtractCsRs2(common->inst_word());
    return common->resource_pool()->GetOrAddResource(absl::StrCat("x", num));
  });
  Insert(getter_map, *Enum::kCdrs2, [common]() -> generic::SimpleResource * {
    auto num = Extractors::CR::ExtractRs2(common->inst_word());
    return common->resource_pool()->GetOrAddResource(absl::StrCat("d", num));
  });
  Insert(getter_map, *Enum::kCrs1, [common]() -> generic::SimpleResource * {
    auto num = Extractors::CR::ExtractRs1(common->inst_word());
    return common->resource_pool()->GetOrAddResource(absl::StrCat("x", num));
  });
  Insert(getter_map, *Enum::kCrs2, [common]() -> generic::SimpleResource * {
    auto num = Extractors::CR::ExtractRs2(common->inst_word());
    return common->resource_pool()->GetOrAddResource(absl::StrCat("x", num));
  });
  Insert(getter_map, *Enum::kCsr, [common]() -> generic::SimpleResource * {
    return common->resource_pool()->GetOrAddResource("csr");
  });
  Insert(getter_map, *Enum::kDrd, [common]() -> generic::SimpleResource * {
    int num = Extractors::RType::ExtractRd(common->inst_word());
    return common->resource_pool()->GetOrAddResource(absl::StrCat("d", num));
  });
  Insert(getter_map, *Enum::kDrs1, [common]() -> generic::SimpleResource * {
    int num = Extractors::RType::ExtractRs1(common->inst_word());
    return common->resource_pool()->GetOrAddResource(absl::StrCat("d", num));
  });
  Insert(getter_map, *Enum::kDrs2, [common]() -> generic::SimpleResource * {
    int num = Extractors::RType::ExtractRs2(common->inst_word());
    return common->resource_pool()->GetOrAddResource(absl::StrCat("d", num));
  });
  Insert(getter_map, *Enum::kDrs3, [common]() -> generic::SimpleResource * {
    int num = Extractors::R4Type::ExtractRs3(common->inst_word());
    return common->resource_pool()->GetOrAddResource(absl::StrCat("d", num));
  });
  Insert(getter_map, *Enum::kFrd, [common]() -> generic::SimpleResource * {
    int num = Extractors::RType::ExtractRd(common->inst_word());
    return common->resource_pool()->GetOrAddResource(absl::StrCat("d", num));
  });
  Insert(getter_map, *Enum::kFrs1, [common]() -> generic::SimpleResource * {
    int num = Extractors::RType::ExtractRs1(common->inst_word());
    return common->resource_pool()->GetOrAddResource(absl::StrCat("d", num));
  });
  Insert(getter_map, *Enum::kFrs2, [common]() -> generic::SimpleResource * {
    int num = Extractors::RType::ExtractRs2(common->inst_word());
    return common->resource_pool()->GetOrAddResource(absl::StrCat("d", num));
  });
  Insert(getter_map, *Enum::kFrs3, [common]() -> generic::SimpleResource * {
    int num = Extractors::R4Type::ExtractRs3(common->inst_word());
    return common->resource_pool()->GetOrAddResource(absl::StrCat("d", num));
  });
  Insert(getter_map, *Enum::kNextPc, [common]() -> generic::SimpleResource * {
    return common->resource_pool()->GetOrAddResource("next_pc");
  });
  Insert(getter_map, *Enum::kRd, [common]() -> generic::SimpleResource * {
    auto num = Extractors::RType::ExtractRd(common->inst_word());
    if (num == 0) return nullptr;
    return common->resource_pool()->GetOrAddResource(absl::StrCat("x", num));
  });
  Insert(getter_map, *Enum::kRs1, [common]() -> generic::SimpleResource * {
    auto num = Extractors::RType::ExtractRs1(common->inst_word());
    if (num == 0) return nullptr;
    return common->resource_pool()->GetOrAddResource(absl::StrCat("x", num));
  });
  Insert(getter_map, *Enum::kRs2, [common]() -> generic::SimpleResource * {
    auto num = Extractors::RType::ExtractRs2(common->inst_word());
    if (num == 0) return nullptr;
    return common->resource_pool()->GetOrAddResource(absl::StrCat("x", num));
  });
  Insert(getter_map, *Enum::kX0,
         []() -> generic::SimpleResource * { return nullptr; });
  Insert(getter_map, *Enum::kX1, [common]() -> generic::SimpleResource * {
    return common->resource_pool()->GetOrAddResource("x1");
  });
  Insert(getter_map, *Enum::kX2, [common]() -> generic::SimpleResource * {
    return common->resource_pool()->GetOrAddResource("x2");
  });
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_GETTERS_H_
