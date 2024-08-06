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

#ifndef MPACT_RISCV_RISCV_RISCV_REGISTER_H_
#define MPACT_RISCV_RISCV_RISCV_REGISTER_H_

#include <any>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/operand_interface.h"
#include "mpact/sim/generic/register.h"
#include "mpact/sim/generic/state_item.h"

// File contains shorthand type definitions for RiscV32G registers.

namespace mpact {
namespace sim {
namespace riscv {

class RiscVState;

// The value type of the register must be an unsigned integer type.
using RV32Register = generic::Register<uint32_t>;
using RV64Register = generic::Register<uint64_t>;

using RVXRegister = RV32Register;
using RVFpRegister = RV64Register;

// The RiscV vector registers are treated like a long bit string that can
// be divided into byte, half word, word or double word quantities. RiscV
// allows for up to 8 registers to be combined by setting the LMUL
// appropriately. This operand therefore needs to be able to refer to up to
// 8 registers. Only vector register 0, 8, 16, and 24 can be treated as 8
// register groups. Vector registers 0, 4, 8, 12, 16, 20, 24, and 28, can
// be treated as 4 register groups. Even vector registers can be grouped into
// pairs, while vector registers with odd numbers can not be the start of a
// group. While this operand doesn't check for this, it has to be aware that
// some registers in the register array may be nullptr.

class RV32VectorSourceOperand : public generic::SourceOperandInterface {
 public:
  RV32VectorSourceOperand(absl::Span<generic::RegisterBase *> reg_span,
                          std::string op_name);
  explicit RV32VectorSourceOperand(
      absl::Span<generic::RegisterBase *> reg_span);
  explicit RV32VectorSourceOperand(generic::RegisterBase *reg);
  RV32VectorSourceOperand(generic::RegisterBase *reg, std::string op_name);

  RV32VectorSourceOperand() = delete;
  bool AsBool(int i) override;
  int8_t AsInt8(int i) override;
  uint8_t AsUint8(int i) override;
  int16_t AsInt16(int i) override;
  uint16_t AsUint16(int i) override;
  int32_t AsInt32(int i) override;
  uint32_t AsUint32(int i) override;
  int64_t AsInt64(int i) override;
  uint64_t AsUint64(int i) override;
  // Returns the RegisterBase object wrapped in absl::any.
  std::any GetObject() const override { return std::any(registers_[0]); }
  // Non-inherited method to get the register object.
  generic::RegisterBase *GetRegister(int i) const { return registers_[i]; }
  // Returns the shape of the register.
  std::vector<int> shape() const override { return registers_[0]->shape(); }
  std::string AsString() const override { return registers_[0]->name(); }
  // New methods.
  std::any GetObject(int i) const;
  int size() const { return registers_.size(); }

 private:
  int group_size_ = 0;
  int vector_size_ = 0;
  int vector_byte_size_ = 0;
  std::vector<generic::RegisterBase *> registers_;
  std::string op_name_;
};

class RV32VectorTrueOperand : public RV32VectorSourceOperand {
 public:
  explicit RV32VectorTrueOperand(RiscVState *state);

  RV32VectorTrueOperand() = delete;
  bool AsBool(int) final { return true; }
  int8_t AsInt8(int) final { return 0xff; }
  uint8_t AsUint8(int) final { return 0xff; }
  int16_t AsInt16(int) final { return 0xffff; }
  uint16_t AsUint16(int) final { return 0xffff; }
  int32_t AsInt32(int) final { return 0xffff'ffff; }
  uint32_t AsUint32(int) final { return 0xffff'ffff; }
  int64_t AsInt64(int) final { return 0xffff'ffff'ffff'ffffULL; }
  uint64_t AsUint64(int) final { return 0xffff'ffff'ffff'ffffLL; }
  std::string AsString() const override { return ""; }

 private:
  static constexpr char kName[] = "__VectorTrue__";
};

class RV32VectorDestinationOperand
    : public generic::DestinationOperandInterface {
 public:
  RV32VectorDestinationOperand(absl::Span<generic::RegisterBase *> reg_span,
                               int latency, std::string op_name);
  RV32VectorDestinationOperand(absl::Span<generic::RegisterBase *> reg_span,
                               int latency);
  RV32VectorDestinationOperand(generic::RegisterBase *reg, int latency,
                               std::string op_name);
  RV32VectorDestinationOperand(generic::RegisterBase *reg, int latency);
  generic::DataBuffer *AllocateDataBuffer() override;
  void InitializeDataBuffer(generic::DataBuffer *db) override;
  generic::DataBuffer *CopyDataBuffer() override;
  int latency() const override;
  std::any GetObject() const override;
  std::vector<int> shape() const override;
  std::string AsString() const override;
  // New method.
  generic::DataBuffer *AllocateDataBuffer(int i);
  void InitializeDataBuffer(int i, generic::DataBuffer *db);
  generic::DataBuffer *CopyDataBuffer(int i);
  std::any GetObject(int i) const;
  int size() const { return registers_.size(); }

 private:
  generic::DataBufferFactory *db_factory_;
  generic::DataBufferDelayLine *delay_line_;
  int latency_ = 0;
  int group_size_ = 0;
  int vector_size_ = 0;
  int vector_byte_size_ = 0;
  std::vector<generic::RegisterBase *> registers_;
  std::string op_name_;
};

using RVVectorRegister =
    generic::StateItem<generic::RegisterBase, uint8_t *,
                       RV32VectorSourceOperand, RV32VectorDestinationOperand>;

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_REGISTER_H_
