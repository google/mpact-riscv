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

#include "riscv/riscv_vector_state.h"

#include <cstdint>
#include <memory>

#include "googletest/include/gtest/gtest.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "riscv/riscv_state.h"

namespace {

using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVVectorState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::util::FlatDemandMemory;

constexpr int kVectorLengthInBytes = 32;

class RiscVVectorStateTest : public ::testing::Test {
 protected:
  RiscVVectorStateTest() {
    state_ = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory_);
    rv_vector_ =
        std::make_unique<RiscVVectorState>(state_.get(), kVectorLengthInBytes);
  }

  FlatDemandMemory memory_;
  std::unique_ptr<RiscVState> state_;
  std::unique_ptr<RiscVVectorState> rv_vector_;
};

TEST_F(RiscVVectorStateTest, SetVectorTypeVill) {
  // Set vill bit (bit 31) and some other bits.
  constexpr uint32_t kVtype = RiscVVectorState::kIllegalVectorType | 0b111;
  rv_vector_->SetVectorType(kVtype);

  // When vill is set, all other fields should be cleared.
  EXPECT_EQ(rv_vector_->vector_length_multiplier(), 0);
  EXPECT_EQ(rv_vector_->selected_element_width(), 0);
  EXPECT_FALSE(rv_vector_->vector_tail_agnostic());
  EXPECT_FALSE(rv_vector_->vector_mask_agnostic());
  EXPECT_EQ(rv_vector_->max_vector_length(), 0);
  // vtype CSR should reflect the vill bit, and other bits should be zero.
  EXPECT_EQ(rv_vector_->vtype(), RiscVVectorState::kIllegalVectorType);
}

}  // namespace
