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

#include "googlemock/include/gmock/gmock.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_state.h"

namespace {

using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVVectorState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::RV32VectorTrueOperand;

constexpr int kVLengthInBytes = 64;
// Test fixture.
class RV32VectorTrueTest : public testing::Test {
 protected:
  RV32VectorTrueTest() {
    state_ = new RiscVState("test", RiscVXlen::RV64);
    vstate_ = new RiscVVectorState(state_, kVLengthInBytes);
  }
  ~RV32VectorTrueTest() override {
    delete state_;
    delete vstate_;
  }

  RiscVState *state_;
  RiscVVectorState *vstate_;
};

TEST_F(RV32VectorTrueTest, Initial) {
  auto *op = new RV32VectorTrueOperand(state_);
  for (int i = 0; i < op->shape()[0]; ++i) {
    EXPECT_EQ(op->AsUint8(i), 0xff) << "element: " << i;
  }
  delete op;
}

TEST_F(RV32VectorTrueTest, Register) {
  auto *op = new RV32VectorTrueOperand(state_);
  auto *reg = op->GetRegister(0);
  auto span = reg->data_buffer()->Get<uint8_t>();
  for (int i = 0; i < op->shape()[0]; ++i) {
    EXPECT_EQ(span[i], 0xff) << "element: " << i;
  }
  delete op;
}

}  // namespace
