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

#include <cstdint>

#include "googlemock/include/gmock/gmock.h"  // IWYU pragma: keep
#include "googletest/include/gtest/gtest.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_state.h"

namespace {

using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVVectorState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::RV32VectorSourceOperand;
using ::mpact::sim::riscv::RV32VectorTrueOperand;
using ::mpact::sim::riscv::RVVectorRegister;
using ::mpact::sim::util::FlatDemandMemory;

constexpr int kVLengthInBytes = 64;
// Test fixture.
class RV32VectorTrueTest : public testing::Test {
 protected:
  RV32VectorTrueTest() {
    state_ = new RiscVState("test", RiscVXlen::RV64, &memory_);
    vstate_ = new RiscVVectorState(state_, kVLengthInBytes);
  }
  ~RV32VectorTrueTest() override {
    delete state_;
    delete vstate_;
  }

  FlatDemandMemory memory_;
  RiscVState* state_;
  RiscVVectorState* vstate_;
};

TEST_F(RV32VectorTrueTest, Initial) {
  auto* op = new RV32VectorTrueOperand(state_);
  for (int i = 0; i < op->shape()[0]; ++i) {
    EXPECT_EQ(op->AsUint8(i), 0xff) << "element: " << i;
  }
  delete op;
}

TEST_F(RV32VectorTrueTest, Register) {
  auto* op = new RV32VectorTrueOperand(state_);
  auto* reg = op->GetRegister(0);
  auto span = reg->data_buffer()->Get<uint8_t>();
  for (int i = 0; i < op->shape()[0]; ++i) {
    EXPECT_EQ(span[i], 0xff) << "element: " << i;
  }
  delete op;
}

TEST_F(RV32VectorTrueTest, SourceOperandOutOfBoundsAccessClampsToLastGroup) {
  auto* v2 = state_->GetRegister<RVVectorRegister>("v2").first;
  auto* v3 = state_->GetRegister<RVVectorRegister>("v3").first;
  auto v2_span = v2->data_buffer()->Get<uint8_t>();
  auto v3_span = v3->data_buffer()->Get<uint8_t>();
  for (int i = 0; i < kVLengthInBytes; ++i) {
    v2_span[i] = 0x00;
    v3_span[i] = 0xff;
  }
  auto* v2_base = RV32VectorSourceOperand(v2, "v2").GetRegister(0);
  auto* v3_base = RV32VectorSourceOperand(v3, "v3").GetRegister(0);
  decltype(v2_base) regs[] = {v2_base, v3_base};
  RV32VectorSourceOperand op(regs, "v2");

  // Group 0 reads from v2 (all zeros).
  EXPECT_FALSE(op.AsBool(0));
  EXPECT_EQ(op.AsInt8(0), 0);
  EXPECT_EQ(op.AsUint8(0), 0U);
  EXPECT_EQ(op.AsInt16(0), 0);
  EXPECT_EQ(op.AsUint16(0), 0U);
  EXPECT_EQ(op.AsInt32(0), 0);
  EXPECT_EQ(op.AsUint32(0), 0U);
  EXPECT_EQ(op.AsInt64(0), 0);
  EXPECT_EQ(op.AsUint64(0), 0ULL);
  EXPECT_EQ(op.GetRegister(0), v2);

  // Group 1 reads from v3 (all ones).
  EXPECT_TRUE(op.AsBool(kVLengthInBytes));
  EXPECT_EQ(op.AsInt8(kVLengthInBytes), -1);
  EXPECT_EQ(op.AsUint8(kVLengthInBytes), 0xffU);
  EXPECT_EQ(op.AsInt16(kVLengthInBytes / 2), -1);
  EXPECT_EQ(op.AsUint16(kVLengthInBytes / 2), 0xffffU);
  EXPECT_EQ(op.AsInt32(kVLengthInBytes / 4), -1);
  EXPECT_EQ(op.AsUint32(kVLengthInBytes / 4), 0xffff'ffffU);
  EXPECT_EQ(op.AsInt64(kVLengthInBytes / 8), -1);
  EXPECT_EQ(op.AsUint64(kVLengthInBytes / 8), 0xffff'ffff'ffff'ffffULL);
  EXPECT_EQ(op.GetRegister(1), v3);

  // Out-of-bounds accesses (group >= 2) clamp to the last register in the
  // group (v3, index 1) rather than reading v2 or past the end of registers_.
  EXPECT_TRUE(op.AsBool(2 * kVLengthInBytes));
  EXPECT_EQ(op.AsInt8(2 * kVLengthInBytes), -1);
  EXPECT_EQ(op.AsUint8(2 * kVLengthInBytes), 0xffU);
  EXPECT_EQ(op.AsInt16(2 * (kVLengthInBytes / 2)), -1);
  EXPECT_EQ(op.AsUint16(2 * (kVLengthInBytes / 2)), 0xffffU);
  EXPECT_EQ(op.AsInt32(2 * (kVLengthInBytes / 4)), -1);
  EXPECT_EQ(op.AsUint32(2 * (kVLengthInBytes / 4)), 0xffff'ffffU);
  EXPECT_EQ(op.AsInt64(2 * (kVLengthInBytes / 8)), -1);
  EXPECT_EQ(op.AsUint64(2 * (kVLengthInBytes / 8)), 0xffff'ffff'ffff'ffffULL);
  EXPECT_EQ(op.GetRegister(2), v3);
  EXPECT_EQ(op.GetRegister(7), v3);
}

}  // namespace
