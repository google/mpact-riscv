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

// This file adds operand getters for the Zc* extensions.

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_ZC_GETTERS_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_ZC_GETTERS_H_

#include <cstdint>
#include <vector>

#include "absl/strings/str_cat.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/literal_operand.h"
#include "mpact/sim/generic/operand_interface.h"
#include "mpact/sim/generic/resource_operand_interface.h"
#include "riscv/riscv_encoding_common.h"
#include "riscv/riscv_getter_helpers.h"
#include "riscv/riscv_register_aliases.h"
#include "riscv/riscv_state.h"

namespace mpact::sim::riscv {
using ::mpact::sim::generic::DestinationOperandInterface;
using ::mpact::sim::generic::ImmediateOperand;
using ::mpact::sim::generic::IntLiteralOperand;
using ::mpact::sim::generic::ResourceOperandInterface;
using ::mpact::sim::generic::SourceOperandInterface;

using SourceOpIf = SourceOperandInterface;

// The following function adds source operand getters to the given getter map.
// The function uses the template parameters to get the correct enum type
// for the instruction set being decoded. The Extractors parameter is used to
// get the correct instruction format extractor for the instruction set. The
// IntRegister and FpRegister parameters are used to get the correct register
// types for the instruction set.

// Getters for Zca source operands.

template <typename Enum, typename Extractors, typename IntRegister,
          typename FpRegister>
void AddRiscVZcaSourceGetters(SourceOpGetterMap &getter_map,
                              RiscVEncodingCommon *common) {
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
  Insert(getter_map, *Enum::kCrd, [common]() {
    auto num = Extractors::CR::ExtractRd(common->inst_word());
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, num),
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
  Insert(getter_map, *Enum::kICbImm8, [common]() {
    return new ImmediateOperand<int32_t>(
        Extractors::Inst16Format::ExtractBimm(common->inst_word()));
  });
  Insert(getter_map, *Enum::kICiImm6, [common]() {
    return new ImmediateOperand<int32_t>(
        Extractors::CI::ExtractImm6(common->inst_word()));
  });
  Insert(getter_map, *Enum::kICiImm6x16, [common]() {
    return new generic::ImmediateOperand<int32_t>(
        Extractors::Inst16Format::ExtractCiImm10(common->inst_word()));
  });
  Insert(getter_map, *Enum::kICiUimm6, [common]() {
    return new ImmediateOperand<uint32_t>(
        Extractors::Inst16Format::ExtractUimm6(common->inst_word()));
  });
  Insert(getter_map, *Enum::kICiUimm6x4,
         [common]() -> SourceOperandInterface * {
           return new ImmediateOperand<uint32_t>(
               Extractors::Inst16Format::ExtractCiImmW(common->inst_word()));
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
  Insert(getter_map, *Enum::kICssUimm6x4,
         [common]() -> SourceOperandInterface * {
           return new ImmediateOperand<uint32_t>(
               Extractors::Inst16Format::ExtractCssImmW(common->inst_word()));
         });
  Insert(getter_map, *Enum::kX0,
         []() { return new generic::IntLiteralOperand<0>({1}); });
  Insert(getter_map, *Enum::kX2, [common]() {
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, 2),
        kXRegisterAliases[2]);
  });
}

// Getters for Zca destination operands.

template <typename Enum, typename Extractors, typename IntRegister,
          typename FpRegister>
void AddRiscVZcaDestGetters(SourceOpGetterMap &getter_map,
                            RiscVEncodingCommon *common) {
  Insert(getter_map, *Enum::kC3rd, [common](int latency) {
    int num = Extractors::CL::ExtractClRd(common->inst_word());
    return GetRegisterDestinationOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, num), latency);
  });
}

// Getters for Zcb source operands (not covered in Zca).

template <typename Enum, typename Extractors, typename IntRegister,
          typename FpRegister>
void AddRiscVZcbSourceGetters(SourceOpGetterMap &getter_map,
                              RiscVEncodingCommon *common) {
  Insert(getter_map, *Enum::kUimm2b, [common]() {
    auto num = Extractors::CLB::ExtractUimm2(common->inst_word());
    return new ImmediateOperand<uint32_t>(num);
  });
  Insert(getter_map, *Enum::kUimm2h, [common](int latency) {
    auto num = Extractors::CLH::ExtractUimm2(common->inst_word());
    return new ImmediateOperand<uint32_t>(num);
  });
}

// Getters for Zcb destination operands (not covered in Zca).

template <typename Enum, typename Extractors, typename IntRegister,
          typename FpRegister>
void AddRiscVZcbDestGetters(DestOpGetterMap &getter_map,
                            RiscVEncodingCommon *common) {
  Insert(getter_map, *Enum::kC3rs2, [common](int latency) {
    int num = Extractors::CLB::ExtractClRd(common->inst_word());
    return GetRegisterDestinationOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, num), latency);
  });
}

// Getters for Zcf source operands.

template <typename Enum, typename Extractors, typename IntRegister,
          typename FpRegister>
void AddRiscVZcfSourceGetters(SourceOpGetterMap &getter_map,
                              RiscVEncodingCommon *common) {
  Insert(getter_map, *Enum::kC3frs2, [common]() {
    auto num = Extractors::CS::ExtractCsRs2(common->inst_word());
    return GetRegisterSourceOp<FpRegister>(
        common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
        kFRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kC3rs1, [common]() {
    auto num = Extractors::CS::ExtractCsRs1(common->inst_word());
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, num),
        kXRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kCfrs2, [common]() {
    auto num = Extractors::CR::ExtractRs2(common->inst_word());
    return GetRegisterSourceOp<FpRegister>(
        common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
        kFRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kICiUimm6x4,
         [common]() -> SourceOperandInterface * {
           return new ImmediateOperand<uint32_t>(
               Extractors::Inst16Format::ExtractCiImmW(common->inst_word()));
         });
  Insert(getter_map, *Enum::kIClUimm5x4, [common]() {
    return new ImmediateOperand<uint32_t>(
        Extractors::Inst16Format::ExtractClImmW(common->inst_word()));
  });
  Insert(getter_map, *Enum::kICssUimm6x4, [common]() {
    return new ImmediateOperand<uint32_t>(
        Extractors::Inst16Format::ExtractCssImmW(common->inst_word()));
  });
  Insert(getter_map, *Enum::kX2, [common]() {
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, 2),
        kXRegisterAliases[2]);
  });
}

// Getters for Zcf destination operands.

template <typename Enum, typename Extractors, typename IntRegister,
          typename FpRegister>
void AddRiscVZcfDestGetters(DestOpGetterMap &getter_map,
                            RiscVEncodingCommon *common) {
  Insert(getter_map, *Enum::kC3frd, [common](int latency) {
    int num = Extractors::CL::ExtractClRd(common->inst_word());
    return GetRegisterDestinationOp<FpRegister>(
        common->state(), absl::StrCat(RiscVState::kFregPrefix, num), latency);
  });
}

// Getters for Zcd source operands.

template <typename Enum, typename Extractors, typename IntRegister,
          typename FpRegister>
void AddRiscVZcdSourceGetters(SourceOpGetterMap &getter_map,
                              RiscVEncodingCommon *common) {
  Insert(getter_map, *Enum::kC3drs2, [common]() {
    auto num = Extractors::CS::ExtractCsRs2(common->inst_word());
    return GetRegisterSourceOp<FpRegister>(
        common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
        kFRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kC3rs1, [common]() {
    auto num = Extractors::CS::ExtractCsRs1(common->inst_word());
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, num),
        kXRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kCdrs2, [common]() {
    auto num = Extractors::CR::ExtractRs2(common->inst_word());
    return GetRegisterSourceOp<FpRegister>(
        common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
        kFRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kICiUimm6x8,
         [common]() -> SourceOperandInterface * {
           return new ImmediateOperand<uint32_t>(
               Extractors::Inst16Format::ExtractCiImmW(common->inst_word()));
         });
  Insert(getter_map, *Enum::kIClUimm5x8, [common]() {
    return new ImmediateOperand<uint32_t>(
        Extractors::Inst16Format::ExtractClImmW(common->inst_word()));
  });
  Insert(getter_map, *Enum::kICssUimm6x8, [common]() {
    return new ImmediateOperand<uint32_t>(
        Extractors::Inst16Format::ExtractCssImmW(common->inst_word()));
  });
  Insert(getter_map, *Enum::kX2, [common]() {
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, 2),
        kXRegisterAliases[2]);
  });
}

// Getters for Zcd destination operands.

template <typename Enum, typename Extractors, typename IntRegister,
          typename FpRegister>
void AddRiscVZcdDestGetters(DestOpGetterMap &getter_map,
                            RiscVEncodingCommon *common) {
  Insert(getter_map, *Enum::kC3drd, [common](int latency) {
    int num = Extractors::CL::ExtractClRd(common->inst_word());
    return GetRegisterDestinationOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kFregPrefix, num), latency);
  });
}

// The following table maps Zcmp Sreg alias register numbers to Xreg numbers.
constexpr int kSRegToXRegMap[8] = {8, 9, 18, 19, 20, 21, 22, 23};

// Getters for Zcmp source operands.

template <typename Enum, typename Extractors, typename IntRegister,
          typename FpRegister>
void AddRiscVZcmpSourceGetters(SourceOpGetterMap &getter_map,
                               RiscVEncodingCommon *common) {
  Insert(getter_map, *Enum::kRlist, [common]() {
    return new ImmediateOperand<uint32_t>(
        Extractors::CMMP::ExtractRlist(common->inst_word()));
  });
  Insert(getter_map, *Enum::kSpimm, [common]() {
    return new ImmediateOperand<uint32_t>(
        Extractors::CMMP::ExtractSpimm(common->inst_word()));
  });
  Insert(getter_map, *Enum::kSreg1, [common]() {
    int num =
        kSRegToXRegMap[Extractors::CMMV::ExtractRs1p(common->inst_word())];
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, num),
        kXRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kSreg2, [common]() {
    int num =
        kSRegToXRegMap[Extractors::CMMV::ExtractRs2p(common->inst_word())];
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, num),
        kXRegisterAliases[num]);
  });
  Insert(getter_map, *Enum::kX2, [common]() {
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, 2),
        kXRegisterAliases[2]);
  });
  Insert(getter_map, *Enum::kX10, [common]() {
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, 10),
        kXRegisterAliases[10]);
  });
  Insert(getter_map, *Enum::kX11, [common]() {
    return GetRegisterSourceOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, 11),
        kXRegisterAliases[11]);
  });
}

// Getters for Zcmp destination operands.

template <typename Enum, typename Extractors, typename IntRegister,
          typename FpRegister>
void AddRiscVZcmpDestGetters(DestOpGetterMap &getter_map,
                             RiscVEncodingCommon *common) {
  Insert(getter_map, *Enum::kSreg1, [common](int latency) {
    int num =
        kSRegToXRegMap[Extractors::CMMV::ExtractRs1p(common->inst_word())];
    return GetRegisterDestinationOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, num), latency);
  });
  Insert(getter_map, *Enum::kSreg2, [common](int latency) {
    int num =
        kSRegToXRegMap[Extractors::CMMV::ExtractRs2p(common->inst_word())];
    return GetRegisterDestinationOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, num), latency);
  });
  Insert(getter_map, *Enum::kX2, [common](int latency) {
    return GetRegisterDestinationOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, 2), latency);
  });
  Insert(getter_map, *Enum::kX10, [common](int latency) {
    return GetRegisterDestinationOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, 10), latency);
  });
  Insert(getter_map, *Enum::kX11, [common](int latency) {
    return GetRegisterDestinationOp<IntRegister>(
        common->state(), absl::StrCat(RiscVState::kXregPrefix, 11), latency);
  });
}

// Getters for Zcmp list source operands.

template <typename Enum, typename Extractors, typename IntRegister,
          typename FpRegister>
void AddRiscVZcmpListSourceGetters(ListSourceOpGetterMap &getter_map,
                                   RiscVEncodingCommon *common) {
  Insert(getter_map, *Enum::kRlist, [common]() {
    std::vector<SourceOperandInterface *> result;
    int rlist = Extractors::CMMP::ExtractRlist(common->inst_word());
    // Get the value of 'rlist', and add source operands accordingly.
    if (rlist < 4) return result;
    result.push_back(GetRegisterSourceOp<IntRegister>(common->state(), "x1"));
    if (rlist == 4) return result;
    result.push_back(GetRegisterSourceOp<IntRegister>(common->state(), "x8"));
    if (rlist == 5) return result;
    result.push_back(GetRegisterSourceOp<IntRegister>(common->state(), "x9"));
    if (rlist == 6) return result;
    result.push_back(GetRegisterSourceOp<IntRegister>(common->state(), "x18"));
    if (rlist == 7) return result;
    result.push_back(GetRegisterSourceOp<IntRegister>(common->state(), "x19"));
    if (rlist == 8) return result;
    result.push_back(GetRegisterSourceOp<IntRegister>(common->state(), "x20"));
    if (rlist == 9) return result;
    result.push_back(GetRegisterSourceOp<IntRegister>(common->state(), "x21"));
    if (rlist == 10) return result;
    result.push_back(GetRegisterSourceOp<IntRegister>(common->state(), "x22"));
    if (rlist == 11) return result;
    result.push_back(GetRegisterSourceOp<IntRegister>(common->state(), "x23"));
    if (rlist == 12) return result;
    result.push_back(GetRegisterSourceOp<IntRegister>(common->state(), "x24"));
    if (rlist == 13) return result;
    result.push_back(GetRegisterSourceOp<IntRegister>(common->state(), "x25"));
    if (rlist == 14) return result;
    result.push_back(GetRegisterSourceOp<IntRegister>(common->state(), "x26"));
    result.push_back(GetRegisterSourceOp<IntRegister>(common->state(), "x27"));
    return result;
  });
}

// Getters for Zcmp list destination operands.

template <typename Enum, typename Extractors, typename IntRegister,
          typename FpRegister>
void AddRiscVZcmpListDestGetters(ListDestOpGetterMap &getter_map,
                                 RiscVEncodingCommon *common) {
  Insert(getter_map, *Enum::kRlist, [common](std::vector<int> latency) {
    std::vector<DestinationOperandInterface *> result;
    int rlist = Extractors::CMMP::ExtractRlist(common->inst_word());
    // Get the value of 'rlist', and add destination operands accordingly.
    int size = latency.size();
    if (rlist < 4) return result;
    result.push_back(GetRegisterDestinationOp<IntRegister>(
        common->state(), "x1", latency[result.size() % size]));
    if (rlist == 4) return result;
    result.push_back(GetRegisterDestinationOp<IntRegister>(
        common->state(), "x8", latency[result.size() % size]));
    if (rlist == 5) return result;
    result.push_back(GetRegisterDestinationOp<IntRegister>(
        common->state(), "x9", latency[result.size() % size]));
    if (rlist == 6) return result;
    result.push_back(GetRegisterDestinationOp<IntRegister>(
        common->state(), "x18", latency[result.size() % size]));
    if (rlist == 7) return result;
    result.push_back(GetRegisterDestinationOp<IntRegister>(
        common->state(), "x19", latency[result.size() % size]));
    if (rlist == 8) return result;
    result.push_back(GetRegisterDestinationOp<IntRegister>(
        common->state(), "x20", latency[result.size() % size]));
    if (rlist == 9) return result;
    result.push_back(GetRegisterDestinationOp<IntRegister>(
        common->state(), "x21", latency[result.size() % size]));
    if (rlist == 10) return result;
    result.push_back(GetRegisterDestinationOp<IntRegister>(
        common->state(), "x22", latency[result.size() % size]));
    if (rlist == 11) return result;
    result.push_back(GetRegisterDestinationOp<IntRegister>(
        common->state(), "x23", latency[result.size() % size]));
    if (rlist == 12) return result;
    result.push_back(GetRegisterDestinationOp<IntRegister>(
        common->state(), "x24", latency[result.size() % size]));
    if (rlist == 13) return result;
    result.push_back(GetRegisterDestinationOp<IntRegister>(
        common->state(), "x25", latency[result.size() % size]));
    if (rlist == 14) return result;
    result.push_back(GetRegisterDestinationOp<IntRegister>(
        common->state(), "x26", latency[result.size() % size]));
    result.push_back(GetRegisterDestinationOp<IntRegister>(
        common->state(), "x27", latency[result.size() % size]));
    return result;
  });
}

template <typename Enum, typename Extractors, typename IntRegister,
          typename FpRegister>
void AddRiscVZcmtSourceGetters(SourceOpGetterMap &getter_map,
                               RiscVEncodingCommon *common) {
  Insert(getter_map, *Enum::kJtIndex, [common](int latency) {
    int num = Extractors::CMJT::ExtractIndex(common->inst_word());
    return new ImmediateOperand<uint32_t>(num);
  });
}

}  // namespace mpact::sim::riscv

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_ZC_GETTERS_H_
