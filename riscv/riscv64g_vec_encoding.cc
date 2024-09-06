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

#include "riscv/riscv64g_vec_encoding.h"

#include <cstdint>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/literal_operand.h"
#include "mpact/sim/generic/register.h"
#include "mpact/sim/generic/simple_resource.h"
#include "mpact/sim/generic/simple_resource_operand.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv64gv_bin_decoder.h"
#include "riscv/riscv64gv_decoder.h"
#include "riscv/riscv64gv_enums.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {
namespace isa64v {
namespace {

using generic::SimpleResourceOperand;
using ::mpact::sim::generic::operator*;  // NOLINT: is used below.

constexpr int kNumRegTable[8] = {8, 1, 2, 1, 4, 1, 2, 1};

template <typename M, typename E, typename G>
inline void Insert(M &map, E entry, G getter) {
  map.insert(std::make_pair(static_cast<int>(entry), getter));
}

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

// Generic helper functions to create register operands.
template <typename RegType>
inline DestinationOperandInterface *GetRegisterDestinationOp(RiscVState *state,
                                                             std::string name,
                                                             int latency) {
  auto *reg = state->GetRegister<RegType>(name).first;
  return reg->CreateDestinationOperand(latency);
}

template <typename RegType>
inline DestinationOperandInterface *GetRegisterDestinationOp(
    RiscVState *state, std::string name, int latency, std::string op_name) {
  auto *reg = state->GetRegister<RegType>(name).first;
  return reg->CreateDestinationOperand(latency, op_name);
}

template <typename T>
inline DestinationOperandInterface *GetCSRSetBitsDestinationOp(
    RiscVState *state, std::string name, int latency, std::string op_name) {
  auto result = state->csr_set()->GetCsr(name);
  if (!result.ok()) {
    LOG(ERROR) << "No such CSR '" << name << "'";
    return nullptr;
  }
  auto *csr = result.value();
  auto *op = csr->CreateSetDestinationOperand(latency, op_name);
  return op;
}

template <typename RegType>
inline SourceOperandInterface *GetRegisterSourceOp(RiscVState *state,
                                                   std::string name) {
  auto *reg = state->GetRegister<RegType>(name).first;
  auto *op = reg->CreateSourceOperand();
  return op;
}

template <typename RegType>
inline SourceOperandInterface *GetRegisterSourceOp(RiscVState *state,
                                                   std::string name,
                                                   std::string op_name) {
  auto *reg = state->GetRegister<RegType>(name).first;
  auto *op = reg->CreateSourceOperand(op_name);
  return op;
}

}  // namespace

RiscV64GVecEncoding::RiscV64GVecEncoding(RiscVState *state)
    : RiscV64GVecEncoding(state, true) {}

RiscV64GVecEncoding::RiscV64GVecEncoding(RiscVState *state, bool use_abi_names)
    : state_(state) {
  if (use_abi_names) {
    xreg_alias_ = xreg_abi_names_;
  } else {
    xreg_alias_ = xreg_names_;
  }
  InitializeSourceOperandGetters();
  InitializeDestinationOperandGetters();
  InitializeVectorSourceOperandGetters();
  InitializeVectorDestinationOperandGetters();
  InitializeSimpleResourceGetters();
  resource_pool_ = new generic::SimpleResourcePool("RiscV64GV", 128);
  resource_delay_line_ =
      state_->CreateAndAddDelayLine<generic::SimpleResourceDelayLine>(8);
  for (int i = *SourceOpEnum::kNone; i < *SourceOpEnum::kPastMaxValue; ++i) {
    if (source_op_getters_.find(i) == source_op_getters_.end()) {
      LOG(ERROR) << "No getter for source op enum value " << i;
    }
  }
  for (int i = *DestOpEnum::kNone; i < *DestOpEnum::kPastMaxValue; ++i) {
    if (dest_op_getters_.find(i) == dest_op_getters_.end()) {
      LOG(ERROR) << "No getter for destination op enum value " << i;
    }
  }
}

RiscV64GVecEncoding::~RiscV64GVecEncoding() { delete resource_pool_; }

void RiscV64GVecEncoding::InitializeSourceOperandGetters() {
  // Source operand getters.
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kAAq),
                     [this]() -> SourceOperandInterface * {
                       if (encoding64::inst32_format::ExtractAq(inst_word_)) {
                         return new generic::IntLiteralOperand<1>();
                       }
                       return new generic::IntLiteralOperand<0>();
                     }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kARl),
                     [this]() -> SourceOperandInterface * {
                       if (encoding64::inst32_format::ExtractRl(inst_word_)) {
                         return new generic::IntLiteralOperand<1>();
                       }
                       return new generic::IntLiteralOperand<0>();
                     }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kIUimm6), [this]() {
        return new generic::ImmediateOperand<uint32_t>(
            encoding64::r_s_type::ExtractRUimm6(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kBImm12), [this]() {
        return new generic::ImmediateOperand<int32_t>(
            encoding64::inst32_format::ExtractBImm(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kC3drs2), [this]() {
        auto num = encoding64::inst16_format::ExtractCsRs2(inst_word_);
        return GetRegisterSourceOp<RVFpRegister>(
            state_, absl::StrCat(RiscVState::kFregPrefix, num));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kC3rs1), [this]() {
        auto num = encoding64::inst16_format::ExtractCsRs1(inst_word_);
        if (state_->xlen() == RiscVXlen::RV32) {
          return GetRegisterSourceOp<RV64Register>(
              state_, absl::StrCat(RiscVState::kXregPrefix, num),
              xreg_alias_[num]);
        }
        return GetRegisterSourceOp<RV64Register>(
            state_, absl::StrCat(RiscVState::kXregPrefix, num),
            xreg_alias_[num]);
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kC3rs2), [this]() {
        auto num = encoding64::inst16_format::ExtractCsRs2(inst_word_);
        if (state_->xlen() == RiscVXlen::RV32) {
          return GetRegisterSourceOp<RV64Register>(
              state_, absl::StrCat(RiscVState::kXregPrefix, num),
              xreg_alias_[num]);
        }
        return GetRegisterSourceOp<RV64Register>(
            state_, absl::StrCat(RiscVState::kXregPrefix, num),
            xreg_alias_[num]);
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kCSRUimm5), [this]() {
        return new generic::ImmediateOperand<uint32_t>(
            encoding64::inst32_format::ExtractIUimm5(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kCdrs2), [this]() {
        auto num = encoding64::c_r::ExtractRs2(inst_word_);
        return GetRegisterSourceOp<RVFpRegister>(
            state_, absl::StrCat(RiscVState::kFregPrefix, num));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kCrs1), [this]() {
        auto num = encoding64::c_r::ExtractRs1(inst_word_);
        if (state_->xlen() == RiscVXlen::RV32) {
          return GetRegisterSourceOp<RV64Register>(
              state_, absl::StrCat(RiscVState::kXregPrefix, num),
              xreg_alias_[num]);
        }
        return GetRegisterSourceOp<RV64Register>(
            state_, absl::StrCat(RiscVState::kXregPrefix, num),
            xreg_alias_[num]);
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kCrs2), [this]() {
        auto num = encoding64::c_r::ExtractRs2(inst_word_);
        if (state_->xlen() == RiscVXlen::RV32) {
          return GetRegisterSourceOp<RV64Register>(
              state_, absl::StrCat(RiscVState::kXregPrefix, num),
              xreg_alias_[num]);
        }
        return GetRegisterSourceOp<RV64Register>(
            state_, absl::StrCat(RiscVState::kXregPrefix, num),
            xreg_alias_[num]);
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kCsr), [this]() {
        auto csr_indx = encoding64::i_type::ExtractUImm12(inst_word_);
        auto res = state_->csr_set()->GetCsr(csr_indx);
        if (!res.ok()) {
          return new generic::ImmediateOperand<uint32_t>(csr_indx);
        }
        auto *csr = res.value();
        return new generic::ImmediateOperand<uint32_t>(csr_indx, csr->name());
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kDrs1), [this]() {
        int num = encoding64::r_type::ExtractRs1(inst_word_);
        return GetRegisterSourceOp<RVFpRegister>(
            state_, absl::StrCat(RiscVState::kFregPrefix, num));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kDrs2), [this]() {
        int num = encoding64::r_type::ExtractRs2(inst_word_);
        return GetRegisterSourceOp<RVFpRegister>(
            state_, absl::StrCat(RiscVState::kFregPrefix, num));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kDrs3), [this]() {
        int num = encoding64::r4_type::ExtractRs3(inst_word_);
        return GetRegisterSourceOp<RVFpRegister>(
            state_, absl::StrCat(RiscVState::kFregPrefix, num));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kFrs1), [this]() {
        int num = encoding64::r_type::ExtractRs1(inst_word_);
        return GetRegisterSourceOp<RVFpRegister>(
            state_, absl::StrCat(RiscVState::kFregPrefix, num));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kFrs2), [this]() {
        int num = encoding64::r_type::ExtractRs2(inst_word_);
        return GetRegisterSourceOp<RVFpRegister>(
            state_, absl::StrCat(RiscVState::kFregPrefix, num));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kFrs3), [this]() {
        int num = encoding64::r4_type::ExtractRs3(inst_word_);
        return GetRegisterSourceOp<RVFpRegister>(
            state_, absl::StrCat(RiscVState::kFregPrefix, num));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kICbImm8), [this]() {
        return new generic::ImmediateOperand<int32_t>(
            encoding64::inst16_format::ExtractBimm(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kICiImm6), [this]() {
        return new generic::ImmediateOperand<int32_t>(
            encoding64::c_i::ExtractImm6(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kICiImm612), [this]() {
        return new generic::ImmediateOperand<int32_t>(
            encoding64::inst16_format::ExtractImm18(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kICiUimm6), [this]() {
        return new generic::ImmediateOperand<uint32_t>(
            encoding64::inst16_format::ExtractUimm6(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kICiUimm6x4), [this]() {
        return new generic::ImmediateOperand<uint32_t>(
            encoding64::inst16_format::ExtractCiImmW(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kICiUimm6x8), [this]() {
        return new generic::ImmediateOperand<uint32_t>(
            encoding64::inst16_format::ExtractCiImmD(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kICiImm6x16), [this]() {
        return new generic::ImmediateOperand<int32_t>(
            encoding64::inst16_format::ExtractCiImm10(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kICiwUimm8x4), [this]() {
        return new generic::ImmediateOperand<uint32_t>(
            encoding64::inst16_format::ExtractCiwImm10(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kICjImm11), [this]() {
        return new generic::ImmediateOperand<int32_t>(
            encoding64::inst16_format::ExtractJimm(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kIClUimm5x4), [this]() {
        return new generic::ImmediateOperand<uint32_t>(
            encoding64::inst16_format::ExtractClImmW(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kIClUimm5x8), [this]() {
        return new generic::ImmediateOperand<uint32_t>(
            encoding64::inst16_format::ExtractClImmD(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kICssUimm6x4), [this]() {
        return new generic::ImmediateOperand<uint32_t>(
            encoding64::inst16_format::ExtractCssImmW(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kICssUimm6x8), [this]() {
        return new generic::ImmediateOperand<uint32_t>(
            encoding64::inst16_format::ExtractCssImmD(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kIImm12), [this]() {
        return new generic::ImmediateOperand<int32_t>(
            encoding64::inst32_format::ExtractImm12(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kIUimm5), [this]() {
        return new generic::ImmediateOperand<uint32_t>(
            encoding64::inst32_format::ExtractRUimm5(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kJImm12), [this]() {
        return new generic::ImmediateOperand<int32_t>(
            encoding64::inst32_format::ExtractImm12(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kJImm20), [this]() {
        return new generic::ImmediateOperand<int32_t>(
            encoding64::inst32_format::ExtractJImm(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kRm),
                     [this]() -> SourceOperandInterface * {
                       uint32_t rm = (inst_word_ >> 12) & 0x7;
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
                     }));
  source_op_getters_.insert(std::make_pair(
      static_cast<int>(SourceOpEnum::kRd),
      [this]() -> SourceOperandInterface * {
        int num = encoding64::r_type::ExtractRd(inst_word_);
        if (num == 0)
          return new generic::IntLiteralOperand<0>({1}, xreg_alias_[0]);
        if (state_->xlen() == RiscVXlen::RV32) {
          return GetRegisterSourceOp<RV64Register>(
              state_, absl::StrCat(RiscVState::kXregPrefix, num),
              xreg_alias_[num]);
        }
        return GetRegisterSourceOp<RV64Register>(
            state_, absl::StrCat(RiscVState::kXregPrefix, num),
            xreg_alias_[num]);
      }));
  source_op_getters_.insert(std::make_pair(
      static_cast<int>(SourceOpEnum::kRs1),
      [this]() -> SourceOperandInterface * {
        int num = encoding64::r_type::ExtractRs1(inst_word_);
        if (num == 0)
          return new generic::IntLiteralOperand<0>({1}, xreg_alias_[0]);
        if (state_->xlen() == RiscVXlen::RV32) {
          return GetRegisterSourceOp<RV64Register>(
              state_, absl::StrCat(RiscVState::kXregPrefix, num),
              xreg_alias_[num]);
        }
        return GetRegisterSourceOp<RV64Register>(
            state_, absl::StrCat(RiscVState::kXregPrefix, num),
            xreg_alias_[num]);
      }));
  source_op_getters_.insert(std::make_pair(
      static_cast<int>(SourceOpEnum::kRs2),
      [this]() -> SourceOperandInterface * {
        int num = encoding64::r_type::ExtractRs2(inst_word_);
        if (num == 0)
          return new generic::IntLiteralOperand<0>({1}, xreg_alias_[0]);
        if (state_->xlen() == RiscVXlen::RV32) {
          return GetRegisterSourceOp<RV64Register>(
              state_, absl::StrCat(RiscVState::kXregPrefix, num),
              xreg_alias_[num]);
        }
        return GetRegisterSourceOp<RV64Register>(
            state_, absl::StrCat(RiscVState::kXregPrefix, num),
            xreg_alias_[num]);
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kSImm12), [this]() {
        return new generic::ImmediateOperand<int32_t>(
            encoding64::s_type::ExtractSImm(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kUImm20), [this]() {
        uint32_t uimm = encoding64::u_type::ExtractUImm(inst_word_);
        return new generic::ImmediateOperand<int32_t>(
            absl::bit_cast<int32_t>(uimm));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kX0), [this]() {
        return new generic::IntLiteralOperand<0>({1}, xreg_alias_[0]);
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kX2), [this]() {
        if (state_->xlen() == RiscVXlen::RV32) {
          return GetRegisterSourceOp<RV64Register>(
              state_, absl::StrCat(RiscVState::kXregPrefix, 2), xreg_alias_[2]);
        }
        return GetRegisterSourceOp<RV64Register>(
            state_, absl::StrCat(RiscVState::kXregPrefix, 2), xreg_alias_[2]);
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kFs1), [this]() {
        const int num = encoding64::v_arith::ExtractRs1(inst_word_);
        return GetRegisterSourceOp<RV64Register>(state_, freg_names_[num],
                                                 freg_abi_names_[num]);
      }));
  source_op_getters_.insert(std::make_pair(
      static_cast<int>(SourceOpEnum::kNone), []() { return nullptr; }));
}

void RiscV64GVecEncoding::InitializeDestinationOperandGetters() {
  // Destination operand getters.
  dest_op_getters_.insert(
      std::make_pair(static_cast<int>(DestOpEnum::kC3drd), [this](int latency) {
        int num = encoding64::inst16_format::ExtractClRd(inst_word_);
        if (state_->xlen() == RiscVXlen::RV32) {
          return GetRegisterDestinationOp<RV64Register>(
              state_, absl::StrCat(RiscVState::kXregPrefix, num), latency,
              xreg_alias_[num]);
        }
        return GetRegisterDestinationOp<RV64Register>(
            state_, absl::StrCat(RiscVState::kXregPrefix, num), latency,
            xreg_alias_[num]);
      }));
  dest_op_getters_.insert(
      std::make_pair(static_cast<int>(DestOpEnum::kC3rd), [this](int latency) {
        int num = encoding64::inst16_format::ExtractClRd(inst_word_);
        return GetRegisterDestinationOp<RV64Register>(
            state_, absl::StrCat(RiscVState::kXregPrefix, num), latency,
            xreg_alias_[num]);
      }));
  dest_op_getters_.insert(
      std::make_pair(static_cast<int>(DestOpEnum::kC3rs1), [this](int latency) {
        int num = encoding64::inst16_format::ExtractClRs1(inst_word_);
        if (state_->xlen() == RiscVXlen::RV32) {
          return GetRegisterDestinationOp<RV64Register>(
              state_, absl::StrCat(RiscVState::kXregPrefix, num), latency,
              xreg_alias_[num]);
        }
        return GetRegisterDestinationOp<RV64Register>(
            state_, absl::StrCat(RiscVState::kXregPrefix, num), latency,
            xreg_alias_[num]);
      }));
  dest_op_getters_.insert(std::make_pair(static_cast<int>(DestOpEnum::kCsr),
                                         [](int latency) { return nullptr; }));
  dest_op_getters_.insert(
      std::make_pair(static_cast<int>(DestOpEnum::kDrd), [this](int latency) {
        int num = encoding64::r_type::ExtractRd(inst_word_);
        return GetRegisterDestinationOp<RVFpRegister>(
            state_, absl::StrCat(RiscVState::kFregPrefix, num), latency);
      }));
  dest_op_getters_.insert(
      std::make_pair(static_cast<int>(DestOpEnum::kFrd), [this](int latency) {
        int num = encoding64::r_type::ExtractRd(inst_word_);
        return GetRegisterDestinationOp<RVFpRegister>(
            state_, absl::StrCat(RiscVState::kFregPrefix, num), latency);
      }));
  dest_op_getters_.insert(std::make_pair(
      static_cast<int>(DestOpEnum::kNextPc), [this](int latency) {
        if (state_->xlen() == RiscVXlen::RV32) {
          return GetRegisterDestinationOp<RV64Register>(
              state_, RiscVState::kPcName, latency);
        }
        return GetRegisterDestinationOp<RV64Register>(
            state_, RiscVState::kPcName, latency);
      }));
  dest_op_getters_.insert(
      std::make_pair(static_cast<int>(DestOpEnum::kRd),
                     [this](int latency) -> DestinationOperandInterface * {
                       int num = encoding64::r_type::ExtractRd(inst_word_);
                       if (num == 0) {
                         return GetRegisterDestinationOp<RV64Register>(
                             state_, "X0Dest", 0, xreg_alias_[0]);
                       } else {
                         return GetRegisterDestinationOp<RVFpRegister>(
                             state_, absl::StrCat(RiscVState::kXregPrefix, num),
                             latency, xreg_alias_[num]);
                       }
                     }));
  dest_op_getters_.insert(
      std::make_pair(static_cast<int>(DestOpEnum::kX0), [this](int) {
        if (state_->xlen() == RiscVXlen::RV32) {
          return GetRegisterDestinationOp<RV64Register>(state_, "X0Dest", 0,
                                                        xreg_alias_[0]);
        }
        return GetRegisterDestinationOp<RV64Register>(state_, "X0Dest", 0,
                                                      xreg_alias_[0]);
      }));
  dest_op_getters_.insert(
      std::make_pair(static_cast<int>(DestOpEnum::kX1), [this](int latency) {
        if (state_->xlen() == RiscVXlen::RV32) {
          return GetRegisterDestinationOp<RV64Register>(
              state_, absl::StrCat(RiscVState::kXregPrefix, 1), latency,
              xreg_alias_[1]);
        }
        return GetRegisterDestinationOp<RV64Register>(
            state_, absl::StrCat(RiscVState::kXregPrefix, 1), latency,
            xreg_alias_[1]);
      }));
  dest_op_getters_.insert(
      std::make_pair(static_cast<int>(DestOpEnum::kX2), [this](int latency) {
        if (state_->xlen() == RiscVXlen::RV32) {
          return GetRegisterDestinationOp<RV64Register>(
              state_, absl::StrCat(RiscVState::kXregPrefix, 2), latency,
              xreg_alias_[2]);
        }
        return GetRegisterDestinationOp<RV64Register>(
            state_, absl::StrCat(RiscVState::kXregPrefix, 2), latency,
            xreg_alias_[2]);
      }));
  dest_op_getters_.insert(std::make_pair(
      static_cast<int>(DestOpEnum::kFflags), [this](int latency) {
        return GetCSRSetBitsDestinationOp<uint32_t>(state_, "fflags", latency,
                                                    "");
      }));
  dest_op_getters_.insert(
      std::make_pair(static_cast<int>(DestOpEnum::kFd), [this](int latency) {
        const int num = encoding64::v_arith::ExtractRd(inst_word_);
        return GetRegisterDestinationOp<RV64Register>(
            state_, freg_names_[num], latency, freg_abi_names_[num]);
      }));
  dest_op_getters_.insert(std::make_pair(static_cast<int>(DestOpEnum::kNone),
                                         [](int latency) { return nullptr; }));
}

void RiscV64GVecEncoding::InitializeVectorSourceOperandGetters() {
  Insert(source_op_getters_, SourceOpEnum::kVd,
         [this]() -> SourceOperandInterface * {
           auto num = encoding64::v_arith::ExtractVd(inst_word_);
           return GetVectorRegisterSourceOp<RVVectorRegister>(state_, num);
         });
  Insert(source_op_getters_, SourceOpEnum::kVmask,
         [this]() -> SourceOperandInterface * {
           auto vm = encoding64::v_arith::ExtractVm(inst_word_);
           if (vm == 1) {
             // Unmasked, return the True mask.
             return new RV32VectorTrueOperand(state_);
           }
           // Masked. Return the mask register.
           return GetVectorMaskRegisterSourceOp<RVVectorRegister>(state_, 0);
         });
  Insert(source_op_getters_, SourceOpEnum::kVmaskTrue,
         [this]() -> SourceOperandInterface * {
           return new RV32VectorTrueOperand(state_);
         });
  Insert(source_op_getters_, SourceOpEnum::kVm,
         [this]() -> SourceOperandInterface * {
           auto vm = encoding64::v_arith::ExtractVm(inst_word_);
           return new generic::ImmediateOperand<bool>(
               vm, absl::StrCat("vm.", vm ? "t" : "f"));
         });
  Insert(source_op_getters_, SourceOpEnum::kVs1,
         [this]() -> SourceOperandInterface * {
           auto num = encoding64::v_arith::ExtractVs1(inst_word_);
           return GetVectorRegisterSourceOp<RVVectorRegister>(state_, num);
         });
  Insert(source_op_getters_, SourceOpEnum::kVs2,
         [this]() -> SourceOperandInterface * {
           auto num = encoding64::v_arith::ExtractVs2(inst_word_);
           return GetVectorRegisterSourceOp<RVVectorRegister>(state_, num);
         });
  Insert(source_op_getters_, SourceOpEnum::kVs3,
         [this]() -> SourceOperandInterface * {
           auto num = encoding64::v_mem::ExtractVs3(inst_word_);
           return GetVectorRegisterSourceOp<RVVectorRegister>(state_, num);
         });

  Insert(source_op_getters_, SourceOpEnum::kSimm5,
         [this]() -> SourceOperandInterface * {
           const auto num = encoding64::inst32_format::ExtractSimm5(inst_word_);
           return new generic::ImmediateOperand<int32_t>(num);
         });

  Insert(source_op_getters_, SourceOpEnum::kUimm5,
         [this]() -> SourceOperandInterface * {
           const auto num = encoding64::inst32_format::ExtractUimm5(inst_word_);
           return new generic::ImmediateOperand<int32_t>(num);
         });

  Insert(source_op_getters_, SourceOpEnum::kZimm10,
         [this]() -> SourceOperandInterface * {
           const auto num =
               encoding64::inst32_format::ExtractZimm10(inst_word_);
           return new generic::ImmediateOperand<int32_t>(num);
         });

  Insert(source_op_getters_, SourceOpEnum::kZimm11,
         [this]() -> SourceOperandInterface * {
           const auto num =
               encoding64::inst32_format::ExtractZimm11(inst_word_);
           return new generic::ImmediateOperand<int32_t>(num);
         });

  Insert(source_op_getters_, SourceOpEnum::kConst1,
         []() -> SourceOperandInterface * {
           return new generic::ImmediateOperand<int32_t>(1);
         });

  Insert(source_op_getters_, SourceOpEnum::kConst2,
         []() -> SourceOperandInterface * {
           return new generic::ImmediateOperand<int32_t>(2);
         });

  Insert(source_op_getters_, SourceOpEnum::kConst4,
         []() -> SourceOperandInterface * {
           return new generic::ImmediateOperand<int32_t>(4);
         });

  Insert(source_op_getters_, SourceOpEnum::kConst8,
         []() -> SourceOperandInterface * {
           return new generic::ImmediateOperand<int32_t>(8);
         });

  Insert(source_op_getters_, SourceOpEnum::kNf,
         [this]() -> SourceOperandInterface * {
           auto num_fields = encoding64::v_mem::ExtractNf(inst_word_);
           return new generic::ImmediateOperand<uint8_t>(
               num_fields, absl::StrCat(num_fields + 1));
         });
}

void RiscV64GVecEncoding::InitializeVectorDestinationOperandGetters() {
  Insert(dest_op_getters_, DestOpEnum::kVd,
         [this](int latency) -> DestinationOperandInterface * {
           auto num = encoding64::v_arith::ExtractVd(inst_word_);
           return GetVectorRegisterDestinationOp<RVVectorRegister>(
               state_, latency, num);
         });
}

// Parse the instruction word to determine the opcode.
void RiscV64GVecEncoding::ParseInstruction(uint32_t inst_word) {
  inst_word_ = inst_word;
  if ((inst_word_ & 0x3) == 3) {
    opcode_ = mpact::sim::riscv::encoding64::DecodeRiscV64GVInst32(inst_word_);
    return;
  }

  opcode_ = mpact::sim::riscv::encoding64::DecodeRiscVCInst16(
      static_cast<uint16_t>(inst_word_ & 0xffff));
}

void RiscV64GVecEncoding::InitializeComplexResourceOperandGetters() {
  complex_resource_getters_.insert(
      std::make_pair(static_cast<int>(ComplexResourceEnum::kNone),
                     [](int begin, int end) { return nullptr; }));
}

ResourceOperandInterface *RiscV64GVecEncoding::GetComplexResourceOperand(
    SlotEnum, int, OpcodeEnum, ComplexResourceEnum resource, int begin,
    int end) {
  int index = static_cast<int>(resource);
  auto iter = complex_resource_getters_.find(index);
  if (iter == complex_resource_getters_.end()) {
    LOG(WARNING) << "No complex resource getter found for resource: " << index;
    return nullptr;
  }
  return (iter->second)(begin, end);
}

void RiscV64GVecEncoding::InitializeSimpleResourceGetters() {
  simple_resource_getters_.insert(std::make_pair(
      static_cast<int>(SimpleResourceEnum::kNone), []() { return nullptr; }));
  simple_resource_getters_.insert(std::make_pair(
      static_cast<int>(SimpleResourceEnum::kC3drd),
      [this]() -> generic::SimpleResource * {
        int num = encoding64::inst16_format::ExtractClRd(inst_word_);
        return resource_pool_->GetOrAddResource(absl::StrCat("d", num));
      }));
  simple_resource_getters_.insert(std::make_pair(
      static_cast<int>(SimpleResourceEnum::kC3drs2),
      [this]() -> generic::SimpleResource * {
        int num = encoding64::inst16_format::ExtractCsRs2(inst_word_);
        return resource_pool_->GetOrAddResource(absl::StrCat("d", num));
      }));
  simple_resource_getters_.insert(std::make_pair(
      static_cast<int>(SimpleResourceEnum::kC3rd),
      [this]() -> generic::SimpleResource * {
        int num = encoding64::inst16_format::ExtractClRd(inst_word_);
        // If num is 0 it refers to the zero register. No resource.
        if (num == 0) return nullptr;
        return resource_pool_->GetOrAddResource(absl::StrCat("x", num));
      }));
  simple_resource_getters_.insert(std::make_pair(
      static_cast<int>(SimpleResourceEnum::kC3rs1),
      [this]() -> generic::SimpleResource * {
        int num = encoding64::inst16_format::ExtractClRs1(inst_word_);
        // If num is 0 it refers to the zero register. No resource.
        if (num == 0) return nullptr;
        return resource_pool_->GetOrAddResource(absl::StrCat("x", num));
      }));
  simple_resource_getters_.insert(std::make_pair(
      static_cast<int>(SimpleResourceEnum::kC3rs2),
      [this]() -> generic::SimpleResource * {
        int num = encoding64::inst16_format::ExtractCsRs2(inst_word_);
        // If num is 0 it refers to the zero register. No resource.
        if (num == 0) return nullptr;
        return resource_pool_->GetOrAddResource(absl::StrCat("x", num));
      }));
  simple_resource_getters_.insert(std::make_pair(
      static_cast<int>(SimpleResourceEnum::kCdrs2),
      [this]() -> generic::SimpleResource * {
        auto num = encoding64::c_r::ExtractRs2(inst_word_);
        return resource_pool_->GetOrAddResource(absl::StrCat("d", num));
      }));
  simple_resource_getters_.insert(std::make_pair(
      static_cast<int>(SimpleResourceEnum::kCrs1),
      [this]() -> generic::SimpleResource * {
        auto num = encoding64::c_r::ExtractRs1(inst_word_);
        // If num is 0 it refers to the zero register. No resource.
        if (num == 0) return nullptr;
        return resource_pool_->GetOrAddResource(absl::StrCat("x", num));
      }));
  simple_resource_getters_.insert(std::make_pair(
      static_cast<int>(SimpleResourceEnum::kCrs2),
      [this]() -> generic::SimpleResource * {
        auto num = encoding64::c_r::ExtractRs2(inst_word_);
        // If num is 0 it refers to the zero register. No resource.
        if (num == 0) return nullptr;
        return resource_pool_->GetOrAddResource(absl::StrCat("x", num));
      }));
  simple_resource_getters_.insert(
      std::make_pair(static_cast<int>(SimpleResourceEnum::kCsr),
                     [this]() -> generic::SimpleResource * {
                       return resource_pool_->GetOrAddResource("csr");
                     }));
  simple_resource_getters_.insert(std::make_pair(
      static_cast<int>(SimpleResourceEnum::kDrd),
      [this]() -> generic::SimpleResource * {
        auto num = encoding64::r4_type::ExtractRd(inst_word_);
        return resource_pool_->GetOrAddResource(absl::StrCat("d", num));
      }));
  simple_resource_getters_.insert(std::make_pair(
      static_cast<int>(SimpleResourceEnum::kDrs1),
      [this]() -> generic::SimpleResource * {
        auto num = encoding64::a_type::ExtractRs1(inst_word_);
        return resource_pool_->GetOrAddResource(absl::StrCat("d", num));
      }));
  simple_resource_getters_.insert(std::make_pair(
      static_cast<int>(SimpleResourceEnum::kDrs2),
      [this]() -> generic::SimpleResource * {
        auto num = encoding64::a_type::ExtractRs2(inst_word_);
        return resource_pool_->GetOrAddResource(absl::StrCat("d", num));
      }));
  simple_resource_getters_.insert(std::make_pair(
      static_cast<int>(SimpleResourceEnum::kDrs3),
      [this]() -> generic::SimpleResource * {
        auto num = encoding64::r4_type::ExtractRs3(inst_word_);
        return resource_pool_->GetOrAddResource(absl::StrCat("d", num));
      }));
  simple_resource_getters_.insert(std::make_pair(
      static_cast<int>(SimpleResourceEnum::kFrd),
      [this]() -> generic::SimpleResource * {
        auto num = encoding64::r4_type::ExtractRd(inst_word_);
        return resource_pool_->GetOrAddResource(absl::StrCat("d", num));
      }));
  simple_resource_getters_.insert(std::make_pair(
      static_cast<int>(SimpleResourceEnum::kFrs1),
      [this]() -> generic::SimpleResource * {
        auto num = encoding64::r4_type::ExtractRs1(inst_word_);
        return resource_pool_->GetOrAddResource(absl::StrCat("d", num));
      }));
  simple_resource_getters_.insert(std::make_pair(
      static_cast<int>(SimpleResourceEnum::kFrs2),
      [this]() -> generic::SimpleResource * {
        auto num = encoding64::r4_type::ExtractRs2(inst_word_);
        return resource_pool_->GetOrAddResource(absl::StrCat("d", num));
      }));
  simple_resource_getters_.insert(std::make_pair(
      static_cast<int>(SimpleResourceEnum::kFrs3),
      [this]() -> generic::SimpleResource * {
        auto num = encoding64::r4_type::ExtractRs3(inst_word_);
        return resource_pool_->GetOrAddResource(absl::StrCat("d", num));
      }));
  simple_resource_getters_.insert(
      std::make_pair(static_cast<int>(SimpleResourceEnum::kNextPc),
                     [this]() -> generic::SimpleResource * {
                       return resource_pool_->GetOrAddResource("next_pc");
                     }));
  simple_resource_getters_.insert(std::make_pair(
      static_cast<int>(SimpleResourceEnum::kRd),
      [this]() -> generic::SimpleResource * {
        auto num = encoding64::a_type::ExtractRd(inst_word_);
        // If num is 0 it refers to the zero register. No resource.
        if (num == 0) return nullptr;
        return resource_pool_->GetOrAddResource(absl::StrCat("x", num));
      }));
  simple_resource_getters_.insert(std::make_pair(
      static_cast<int>(SimpleResourceEnum::kRs1),
      [this]() -> generic::SimpleResource * {
        auto num = encoding64::a_type::ExtractRs1(inst_word_);
        // If num is 0 it refers to the zero register. No resource.
        if (num == 0) return nullptr;
        return resource_pool_->GetOrAddResource(absl::StrCat("x", num));
      }));
  simple_resource_getters_.insert(std::make_pair(
      static_cast<int>(SimpleResourceEnum::kRs2),
      [this]() -> generic::SimpleResource * {
        auto num = encoding64::a_type::ExtractRs2(inst_word_);
        // If num is 0 it refers to the zero register. No resource.
        if (num == 0) return nullptr;
        return resource_pool_->GetOrAddResource(absl::StrCat("x", num));
      }));
  // X0 is constant 0, so no resource issue.
  simple_resource_getters_.insert(
      std::make_pair(static_cast<int>(SimpleResourceEnum::kX0),
                     []() -> generic::SimpleResource * { return nullptr; }));
  simple_resource_getters_.insert(
      std::make_pair(static_cast<int>(SimpleResourceEnum::kX1),
                     [this]() -> generic::SimpleResource * {
                       return resource_pool_->GetOrAddResource("x1");
                     }));
  simple_resource_getters_.insert(
      std::make_pair(static_cast<int>(SimpleResourceEnum::kX2),
                     [this]() -> generic::SimpleResource * {
                       return resource_pool_->GetOrAddResource("x2");
                     }));
}

ResourceOperandInterface *RiscV64GVecEncoding::GetSimpleResourceOperand(
    SlotEnum, int, OpcodeEnum, SimpleResourceVector &resource_vec, int end) {
  if (resource_vec.empty()) return nullptr;
  auto *resource_set = resource_pool_->CreateResourceSet();
  for (auto resource_enum : resource_vec) {
    int index = static_cast<int>(resource_enum);
    auto iter = simple_resource_getters_.find(index);
    if (iter == simple_resource_getters_.end()) {
      LOG(WARNING) << "No getter for simple resource " << index;
      continue;
    }
    auto *resource = (iter->second)();
    auto status = resource_set->AddResource(resource);
    if (!status.ok()) {
      LOG(ERROR) << "Unable to add resource to resource set ("
                 << static_cast<int>(resource_enum) << ")";
    }
  }
  auto *op = new SimpleResourceOperand(resource_set, end, resource_delay_line_);
  return op;
}

DestinationOperandInterface *RiscV64GVecEncoding::GetDestination(
    SlotEnum, int, OpcodeEnum, DestOpEnum dest_op, int dest_no, int latency) {
  int index = static_cast<int>(dest_op);
  auto iter = dest_op_getters_.find(index);
  if (iter == dest_op_getters_.end()) {
    LOG(ERROR) << "No getter for destination op " << index;
    return nullptr;
  }
  return (iter->second)(latency);
}

SourceOperandInterface *RiscV64GVecEncoding::GetSource(SlotEnum, int,
                                                       OpcodeEnum,
                                                       SourceOpEnum source_op,
                                                       int source_no) {
  int index = static_cast<int>(source_op);
  auto iter = source_op_getters_.find(index);
  if (iter == source_op_getters_.end()) {
    LOG(ERROR) << "No getter for source op " << index;
    return nullptr;
  }
  return (iter->second)();
}

}  // namespace isa64v
}  // namespace riscv
}  // namespace sim
}  // namespace mpact
