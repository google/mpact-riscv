// Copyright 2023-2024 Google LLC
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

#include "absl/functional/bind_front.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/action_point_manager_base.h"
#include "mpact/sim/generic/breakpoint_manager.h"
#include "mpact/sim/generic/component.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "riscv/riscv_action_point_memory_interface.h"

namespace {

#ifndef EXPECT_OK
#define EXPECT_OK(x) EXPECT_TRUE(x.ok())
#endif

using ::mpact::sim::generic::ActionPointManagerBase;
using ::mpact::sim::generic::BreakpointManager;
using ::mpact::sim::generic::Component;
using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::generic::DataBufferFactory;
using ::mpact::sim::riscv::RiscVActionPointMemoryInterface;
using ::mpact::sim::util::FlatDemandMemory;

constexpr uint64_t kBreak32Address = 0x1000;
constexpr uint64_t kBreak16Address = 0x1004;
constexpr uint64_t kBreakBadAddress = 0x1008;

constexpr uint32_t kAdd32Instruction = 0b0000000'00010'00001'000'00011'0110011;
constexpr uint32_t kAdd16Instruction = 0b100'1'00011'00010'10;
constexpr uint32_t kBadInstruction = 0xffff'ffff;

class RiscVBreakpointTest : public testing::Test, public Component {
 protected:
  RiscVBreakpointTest() : Component("test") {
    db2_ = db_factory_.Allocate<uint16_t>(1);
    db4_ = db_factory_.Allocate<uint32_t>(2);
    memory_ = new FlatDemandMemory(0);

    // Initialize the memory with instruction words.
    db4_->Set<uint32_t>(0, kAdd32Instruction);
    memory_->Store(kBreak32Address, db4_);
    db4_->Set<uint32_t>(0, kBadInstruction);
    memory_->Store(kBreakBadAddress, db4_);
    db2_->Set<uint16_t>(0, kAdd16Instruction);
    memory_->Store(kBreak16Address, db2_);

    rv_ap_memory_interface_ = new RiscVActionPointMemoryInterface(
        memory_, absl::bind_front(&RiscVBreakpointTest::Invalidate, this));
    ap_manager_ = new ActionPointManagerBase(rv_ap_memory_interface_);
    bp_manager_ = new BreakpointManager(ap_manager_, nullptr);
  }

  ~RiscVBreakpointTest() override {
    db2_->DecRef();
    db4_->DecRef();
    delete bp_manager_;
    delete ap_manager_;
    delete rv_ap_memory_interface_;
    delete memory_;
  }

  void Invalidate(uint64_t address) { latch_address_ = address; }

  uint64_t latch_address_ = 0;
  int latch_size_ = -1;
  RiscVActionPointMemoryInterface *rv_ap_memory_interface_;
  ActionPointManagerBase *ap_manager_;
  BreakpointManager *bp_manager_;
  DataBufferFactory db_factory_;
  FlatDemandMemory *memory_;
  DataBuffer *db2_;
  DataBuffer *db4_;
};

TEST_F(RiscVBreakpointTest, Breakpoint16) {
  // There shouldn't be a breakpoint set, and the enable/disable/clear should
  // fail.
  EXPECT_FALSE(bp_manager_->HasBreakpoint(kBreak16Address));
  EXPECT_FALSE(bp_manager_->EnableBreakpoint(kBreak16Address).ok());
  EXPECT_FALSE(bp_manager_->DisableBreakpoint(kBreak16Address).ok());
  EXPECT_FALSE(bp_manager_->ClearBreakpoint(kBreak16Address).ok());

  // Set the breakpoint.
  EXPECT_OK(bp_manager_->SetBreakpoint(kBreak16Address));
  EXPECT_TRUE(bp_manager_->HasBreakpoint(kBreak16Address));
  EXPECT_TRUE(bp_manager_->IsBreakpoint(kBreak16Address));
  memory_->Load(kBreak16Address, db2_, nullptr, nullptr);
  EXPECT_EQ(db2_->Get<uint16_t>(0), RiscVActionPointMemoryInterface::kEBreak16);
  EXPECT_EQ(latch_address_, kBreak16Address);
  latch_address_ = 0;
  latch_size_ = -1;

  // Disable the breakpoint.
  EXPECT_OK(bp_manager_->DisableBreakpoint(kBreak16Address));
  EXPECT_TRUE(bp_manager_->HasBreakpoint(kBreak16Address));
  EXPECT_FALSE(bp_manager_->IsBreakpoint(kBreak16Address));

  // Enable the breakpoint.
  EXPECT_OK(bp_manager_->EnableBreakpoint(kBreak16Address));
  EXPECT_TRUE(bp_manager_->HasBreakpoint(kBreak16Address));
  EXPECT_TRUE(bp_manager_->IsBreakpoint(kBreak16Address));

  // Clear the breakpoint.
  EXPECT_OK(bp_manager_->ClearBreakpoint(kBreak16Address));
  EXPECT_FALSE(bp_manager_->HasBreakpoint(kBreak16Address));
  memory_->Load(kBreak16Address, db2_, nullptr, nullptr);
  EXPECT_EQ(db2_->Get<uint16_t>(0), kAdd16Instruction);
  EXPECT_EQ(latch_address_, kBreak16Address);
  latch_address_ = 0;
  latch_size_ = -1;
}

TEST_F(RiscVBreakpointTest, Breakpoint32) {
  // There shouldn't be a breakpoint set, and the enable/disable/clear should
  // fail.
  EXPECT_FALSE(bp_manager_->HasBreakpoint(kBreak32Address));
  EXPECT_FALSE(bp_manager_->EnableBreakpoint(kBreak32Address).ok());
  EXPECT_FALSE(bp_manager_->DisableBreakpoint(kBreak32Address).ok());
  EXPECT_FALSE(bp_manager_->ClearBreakpoint(kBreak32Address).ok());
  EXPECT_EQ(latch_address_, 0);
  EXPECT_EQ(latch_size_, -1);

  // Set the breakpoint.
  EXPECT_OK(bp_manager_->SetBreakpoint(kBreak32Address));
  EXPECT_TRUE(bp_manager_->HasBreakpoint(kBreak32Address));
  EXPECT_TRUE(bp_manager_->IsBreakpoint(kBreak32Address));
  memory_->Load(kBreak32Address, db4_, nullptr, nullptr);
  EXPECT_EQ(db4_->Get<uint32_t>(0), RiscVActionPointMemoryInterface::kEBreak32);
  EXPECT_EQ(latch_address_, kBreak32Address);
  latch_address_ = 0;
  latch_size_ = -1;

  // Disable the breakpoint.
  EXPECT_OK(bp_manager_->DisableBreakpoint(kBreak32Address));
  EXPECT_TRUE(bp_manager_->HasBreakpoint(kBreak32Address));
  EXPECT_FALSE(bp_manager_->IsBreakpoint(kBreak32Address));
  latch_address_ = 0;
  latch_size_ = -1;

  // Enable the breakpoint.
  EXPECT_OK(bp_manager_->EnableBreakpoint(kBreak32Address));
  EXPECT_TRUE(bp_manager_->HasBreakpoint(kBreak32Address));
  EXPECT_TRUE(bp_manager_->IsBreakpoint(kBreak32Address));
  memory_->Load(kBreak32Address, db4_, nullptr, nullptr);
  latch_address_ = 0;
  latch_size_ = -1;

  // Clear the breakpoint.
  EXPECT_OK(bp_manager_->ClearBreakpoint(kBreak32Address));
  EXPECT_FALSE(bp_manager_->HasBreakpoint(kBreak32Address));
  memory_->Load(kBreak32Address, db4_, nullptr, nullptr);
  EXPECT_EQ(db4_->Get<uint32_t>(0), kAdd32Instruction);
  EXPECT_EQ(latch_address_, kBreak32Address);
  latch_address_ = 0;
  latch_size_ = -1;
}

}  // namespace
