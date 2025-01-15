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

#include "riscv/riscv64g_bin_encoder_interface.h"

#include <cstdint>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/asm/resolver_interface.h"
#include "riscv/riscv64g_bin_encoder.h"
#include "riscv/riscv64g_encoder.h"
#include "riscv/riscv64g_enums.h"
#include "riscv/riscv_bin_setters.h"

namespace mpact {
namespace sim {
namespace riscv {
namespace isa64 {

using ::mpact::sim::generic::operator*;  // NOLINT(misc-unused-using-decls)
using ::mpact::sim::util::assembler::ResolverInterface;

RiscV64GBinEncoderInterface::RiscV64GBinEncoderInterface() {
  AddRiscvSourceOpBinSetters<SourceOpEnum, OpMap, encoding64::Encoder>(
      source_op_map_);
  AddRiscvDestOpBinSetters<DestOpEnum, OpMap, encoding64::Encoder>(
      dest_op_map_);
  AddRiscvSourceOpRelocationSetters<OpcodeEnum, SourceOpEnum, RelocationMap>(
      relocation_source_op_map_);
}

absl::StatusOr<std::tuple<uint64_t, int>>
RiscV64GBinEncoderInterface::GetOpcodeEncoding(SlotEnum slot, int entry,
                                               OpcodeEnum opcode,
                                               ResolverInterface *resolver) {
  return encoding64::kOpcodeEncodings->at(opcode);
}

absl::StatusOr<uint64_t> RiscV64GBinEncoderInterface::GetSrcOpEncoding(
    uint64_t address, absl::string_view text, SlotEnum slot, int entry,
    OpcodeEnum opcode, SourceOpEnum source_op, int source_num,
    ResolverInterface *resolver) {
  auto iter = source_op_map_.find(*source_op);
  if (iter == source_op_map_.end()) {
    return absl::NotFoundError(absl::StrCat(
        "Source operand not found for op enum value ", *source_op));
  }
  return iter->second(address, text, resolver);
}

absl::Status RiscV64GBinEncoderInterface::AppendSrcOpRelocation(
    uint64_t address, absl::string_view text, SlotEnum slot, int entry,
    OpcodeEnum opcode, SourceOpEnum source_op, int source_num,
    ResolverInterface *resolver, std::vector<RelocationInfo> &relocations) {
  auto iter = relocation_source_op_map_.find(std::tie(opcode, source_op));
  if (iter == relocation_source_op_map_.end()) return absl::OkStatus();
  return iter->second(address, text, resolver, relocations);
}

absl::StatusOr<uint64_t> RiscV64GBinEncoderInterface::GetDestOpEncoding(
    uint64_t address, absl::string_view text, SlotEnum slot, int entry,
    OpcodeEnum opcode, DestOpEnum dest_op, int dest_num,
    ResolverInterface *resolver) {
  auto iter = dest_op_map_.find(*dest_op);
  if (iter == dest_op_map_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Dest operand not found for op enum value ", *dest_op));
  }
  return iter->second(address, text, resolver);
}

absl::StatusOr<uint64_t> RiscV64GBinEncoderInterface::GetListDestOpEncoding(
    uint64_t address, absl::string_view text, SlotEnum slot, int entry,
    OpcodeEnum opcode, ListDestOpEnum dest_op, int dest_num,
    ResolverInterface *resolver) {
  auto iter = list_dest_op_map_.find(*dest_op);
  if (iter == list_dest_op_map_.end()) {
    return absl::NotFoundError(absl::StrCat(
        "List dest operand not found for op enum value ", *dest_op));
  }
  return iter->second(address, text, resolver);
}

absl::Status RiscV64GBinEncoderInterface::AppendDestOpRelocation(
    uint64_t address, absl::string_view text, SlotEnum slot, int entry,
    OpcodeEnum opcode, DestOpEnum dest_op, int dest_num,
    ResolverInterface *resolver, std::vector<RelocationInfo> &relocations) {
  // There are no destination operands that require relocation.
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> RiscV64GBinEncoderInterface::GetListSrcOpEncoding(
    uint64_t address, absl::string_view text, SlotEnum slot, int entry,
    OpcodeEnum opcode, ListSourceOpEnum source_op, int source_num,
    ResolverInterface *resolver) {
  auto iter = list_source_op_map_.find(*source_op);
  if (iter == list_source_op_map_.end()) {
    return absl::NotFoundError(absl::StrCat(
        "List source operand not found for op enum value ", *source_op));
  }
  return iter->second(address, text, resolver);
}

absl::StatusOr<uint64_t> RiscV64GBinEncoderInterface::GetPredOpEncoding(
    uint64_t address, absl::string_view text, SlotEnum slot, int entry,
    OpcodeEnum opcode, PredOpEnum pred_op, ResolverInterface *resolver) {
  auto iter = pred_op_map_.find(*pred_op);
  if (iter == pred_op_map_.end()) {
    return absl::NotFoundError(absl::StrCat(
        "Predicate operand not found for op enum value ", *pred_op));
  }
  return iter->second(address, text, resolver);
}

}  // namespace isa64
}  // namespace riscv
}  // namespace sim
}  // namespace mpact
