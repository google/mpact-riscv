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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV64G_BIN_ENCODER_INTERFACE_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV64G_BIN_ENCODER_INTERFACE_H_

#include <cstdint>
#include <functional>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mpact/sim/util/asm/opcode_assembler_interface.h"
#include "mpact/sim/util/asm/resolver_interface.h"
#include "riscv/riscv64g_encoder.h"
#include "riscv/riscv64g_enums.h"

// This file defines the class that implements the RiscV64GEncoderInterfaceBase.
// This class is used to convert from a text representation to the binary
// encoding of opcodes, source and destination operands.

namespace mpact {
namespace sim {
namespace riscv {
namespace isa64 {

using ::mpact::sim::util::assembler::RelocationInfo;
using ::mpact::sim::util::assembler::ResolverInterface;

class RiscV64GBinEncoderInterface : public RiscV64GEncoderInterfaceBase {
 public:
  RiscV64GBinEncoderInterface();
  RiscV64GBinEncoderInterface(const RiscV64GBinEncoderInterface &) = delete;
  RiscV64GBinEncoderInterface &operator=(const RiscV64GBinEncoderInterface &) =
      delete;
  ~RiscV64GBinEncoderInterface() override = default;

  absl::StatusOr<std::tuple<uint64_t, int>> GetOpcodeEncoding(
      SlotEnum slot, int entry, OpcodeEnum opcode,
      ResolverInterface *resolver) override;
  absl::StatusOr<uint64_t> GetSrcOpEncoding(
      uint64_t address, absl::string_view text, SlotEnum slot, int entry,
      OpcodeEnum opcode, SourceOpEnum source_op, int source_num,
      ResolverInterface *resolver) override;
  absl::Status AppendSrcOpRelocation(
      uint64_t address, absl::string_view text, SlotEnum slot, int entry,
      OpcodeEnum opcode, SourceOpEnum source_op, int source_num,
      ResolverInterface *resolver,
      std::vector<RelocationInfo> &relocations) override;
  absl::StatusOr<uint64_t> GetDestOpEncoding(
      uint64_t address, absl::string_view text, SlotEnum slot, int entry,
      OpcodeEnum opcode, DestOpEnum dest_op, int dest_num,
      ResolverInterface *resolver) override;
  absl::Status AppendDestOpRelocation(
      uint64_t address, absl::string_view text, SlotEnum slot, int entry,
      OpcodeEnum opcode, DestOpEnum dest_op, int dest_num,
      ResolverInterface *resolver,
      std::vector<RelocationInfo> &relocations) override;
  absl::StatusOr<uint64_t> GetListSrcOpEncoding(
      uint64_t address, absl::string_view text, SlotEnum slot, int entry,
      OpcodeEnum opcode, ListSourceOpEnum source_op, int source_num,
      ResolverInterface *resolver) override;
  absl::StatusOr<uint64_t> GetListDestOpEncoding(
      uint64_t address, absl::string_view text, SlotEnum slot, int entry,
      OpcodeEnum opcode, ListDestOpEnum dest_op, int dest_num,
      ResolverInterface *resolver) override;
  absl::StatusOr<uint64_t> GetPredOpEncoding(
      uint64_t address, absl::string_view text, SlotEnum slot, int entry,
      OpcodeEnum opcode, PredOpEnum pred_op,
      ResolverInterface *resolver) override;

 private:
  using OpMap = absl::flat_hash_map<
      int, std::function<absl::StatusOr<uint64_t>(uint64_t, absl::string_view,
                                                  ResolverInterface *)>>;

  using RelocationMap =
      absl::flat_hash_map<std::tuple<OpcodeEnum, SourceOpEnum>,
                          std::function<absl::Status(
                              uint64_t, absl::string_view, ResolverInterface *,
                              std::vector<RelocationInfo> &)>>;

  OpMap source_op_map_;
  RelocationMap relocation_source_op_map_;
  OpMap dest_op_map_;
  OpMap list_dest_op_map_;
  OpMap list_source_op_map_;
  OpMap pred_op_map_;
};

}  // namespace isa64
}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV64G_BIN_ENCODER_INTERFACE_H_
