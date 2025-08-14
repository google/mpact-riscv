
// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "riscv/riscv_plic.h"

#include <cstdint>

#include "absl/log/check.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/data_buffer.h"

// This file contains unit tests for the RiscV PLIC model.

namespace {

using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::generic::DataBufferFactory;
using ::mpact::sim::riscv::RiscVPlic;
using ::mpact::sim::riscv::RiscVPlicIrqInterface;

constexpr int kNumSources = 32;
constexpr int kNumContexts = 3;

// Mock interrupt target for a context.
class MockRiscVInterruptTarget : public RiscVPlicIrqInterface {
 public:
  void SetIrq(bool irq_value) override { irq_value_ = irq_value; }

  bool irq_value() const { return irq_value_; }
  void set_irq_value(bool value) { irq_value_ = value; }

 private:
  bool irq_value_ = false;
};

// Test fixture.
class RiscVPlicTest : public ::testing::Test {
 protected:
  RiscVPlicTest() {
    plic_ = new RiscVPlic(kNumSources, kNumContexts);
    for (int i = 0; i < kNumContexts; ++i) {
      target_[i] = new MockRiscVInterruptTarget();
      plic_->SetContext(i, target_[i]);
    }
    db_ = db_factory_.Allocate<uint32_t>(1);
    db_->set_latency(0);
  }

  ~RiscVPlicTest() override {
    delete plic_;
    for (int i = 0; i < kNumContexts; ++i) {
      delete target_[i];
    }
    db_->DecRef();
  }

  // Convenience methods to read/write the PLIC state using its memory
  // interface.
  bool GetEnable(int source, int context) {
    uint32_t word = 0x2000 + (context * 0x80) + (source >> 5);
    int bit = source & 0x1f;
    plic()->Load(word, db_, nullptr, nullptr);
    return (db_->Get<uint32_t>(0) & (1 << bit)) != 0;
  }

  void SetEnable(int source, int context, bool value) {
    uint32_t word = 0x2000 + (context * 0x80) + (source >> 5);
    int bit = source & 0x1f;
    plic()->Load(word, db_, nullptr, nullptr);
    auto span = db_->Get<uint32_t>();
    uint32_t mask = ~(1 << bit);
    uint32_t u_val = static_cast<uint32_t>(value);
    span[0] = (span[0] & mask) | (u_val << bit);
    plic()->Store(word, db_);
  }

  bool GetPending(int source) {
    uint32_t word = 0x1000 + ((source >> 5) << 2);
    int bit = source & 0x1f;
    plic()->Load(word, db_, nullptr, nullptr);
    return (db_->Get<uint32_t>(0) & (1 << bit)) != 0;
  }

  void SetPending(int source, bool value) {
    int word = 0x1000 + ((source >> 5) << 2);
    int bit = source & 0x1f;
    plic()->Load(word, db_, nullptr, nullptr);
    auto span = db_->Get<uint32_t>();
    uint32_t mask = ~(1 << bit);
    span[0] = (span[0] & mask) | (static_cast<uint32_t>(value) << bit);
    plic()->Store(word, db_);
  }

  uint32_t GetPriority(int source) {
    plic()->Load(source << 2, db_, nullptr, nullptr);
    return db_->Get<uint32_t>(0);
  }

  void SetPriority(int source, uint32_t priority) {
    db_->Set<uint32_t>(0, priority);
    plic()->Store(source << 2, db_);
  }

  int GetPriorityThreshold(int context) {
    plic()->Load(0x20'0000 + context * 0x1000, db_, nullptr, nullptr);
    return db_->Get<uint32_t>(0);
  }

  void SetPriorityThreshold(int context, uint32_t threshold) {
    db_->Set<uint32_t>(0, threshold);
    plic()->Store(0x20'0000 + context * 0x1000, db_);
  }

  uint32_t GetInterruptClaim(int context) {
    plic()->Load(0x20'0000 + context * 0x1000 + 4, db_, nullptr, nullptr);
    return db_->Get<uint32_t>(0);
  }

  void SetInterruptClaim(int context, uint32_t claim) {
    db_->Set<uint32_t>(0, claim);
    plic()->Store(0x20'0000 + context * 0x1000 + 4, db_);
  }

  void SetDefaultConfig() {
    // Set priorities to increasing values for all sources.
    // Thresholds are set 8, 16 and 24 for the three contexts respectively.
    // Source 30 is enabled for all contexts.
    auto status = plic()->Configure(
        "0=0;1=1;2=2;3=3;4=4;5=5;6=6;7=7;8=8;9=9;10=10;11=11;"
        "12=12;13=13;14=14;15=15;16=16;17=17;18=18;19=19;20=20;"
        "21=21;22=22;23=23;24=24;25=25;26=26;27=27;28=28;29=29;"
        "30=30;31=31;",
        "0=8,1,1,2,1,3,1,4,1,5,1,6,1,7,1,8,1,9,1,10,1,30,1;"
        "1=16,11,1,12,1,13,1,14,1,15,1,16,1,17,1,18,1,19,1,20,1,30,1;"
        "2=24,19,1,20,1,21,1,22,1,23,1,24,1,25,1,26,1,27,1,28,1,29,1,30,1,31,"
        "1;");
    CHECK_OK(status);
  }

  // Accessors.
  RiscVPlic* plic() { return plic_; }
  DataBufferFactory& db_factory() { return db_factory_; }
  MockRiscVInterruptTarget* target(int i) { return target_[i]; }

 private:
  DataBuffer* db_;
  DataBufferFactory db_factory_;
  RiscVPlic* plic_ = nullptr;
  MockRiscVInterruptTarget* target_[kNumContexts];
};

// Test that the initial state of the PLIC is as expected. Nothing enabled,
// priorities at zero, priority thresholds at zero, no pending interrupts,
// no claimed interrupts, no IRQ lines set.
TEST_F(RiscVPlicTest, InitialState) {
  for (int s = 0; s < kNumSources; ++s) {
    for (int c = 0; c < kNumContexts; ++c) {
      EXPECT_FALSE(GetEnable(s, c));
      EXPECT_FALSE(GetPending(s));
      EXPECT_EQ(GetPriority(s), 0) << "Source " << s;
      EXPECT_EQ(GetPriorityThreshold(c), 0);
      EXPECT_EQ(GetInterruptClaim(c), 0);
      EXPECT_FALSE(target(c)->irq_value());
    }
  }
}

// Verify that the PLIC is configured as expected with the default configuration
// strings.
TEST_F(RiscVPlicTest, DefaultConfiguration) {
  SetDefaultConfig();
  for (int s = 0; s < kNumSources; ++s) {
    for (int c = 0; c < kNumContexts; ++c) {
      switch (c) {
        case 0:
          EXPECT_EQ(GetEnable(s, c), ((s >= 1) && (s <= 10)) || (s == 30))
              << "Source " << s << " context " << c;
          break;
        case 1:
          EXPECT_EQ(GetEnable(s, c), ((s >= 11) && (s <= 20)) || (s == 30))
              << "Source " << s << " context " << c;
          break;
        case 2:
          EXPECT_EQ(GetEnable(s, c), (s >= 19) && (s <= 31))
              << "Source " << s << " context " << c;
          break;
      }
      EXPECT_FALSE(GetPending(s));
      EXPECT_EQ(GetPriority(s), s) << "Source " << s;
      EXPECT_EQ(GetPriorityThreshold(c), (c + 1) * 8) << "Context " << c;
      EXPECT_EQ(GetInterruptClaim(c), 0) << "Context " << c;
      EXPECT_FALSE(target(c)->irq_value()) << "Context " << c;
    }
  }
}

// Test operation of level triggered interrupts.
TEST_F(RiscVPlicTest, LevelTriggeredInterrupt) {
  SetDefaultConfig();
  // Set irq for source 10 level sensitive.
  plic()->SetInterrupt(10, /*value=*/true, /*is_level=*/true);
  // Expect pending to be set.
  EXPECT_TRUE(GetPending(10));
  // Clear the irq.
  plic()->SetInterrupt(10, /*value=*/false, /*is_level=*/true);
  // Expect pending to still be set.
  EXPECT_TRUE(GetPending(10));
  // Expect irq to be set for context 0;
  EXPECT_TRUE(target(0)->irq_value());
  // Clear the irq.
  target(0)->set_irq_value(false);
  // Now raise the irq again.
  plic()->SetInterrupt(10, /*value=*/true, /*is_level=*/true);
  // Expect pending to be set.
  EXPECT_TRUE(GetPending(10));
  // Claim the interrupt.
  uint32_t id = GetInterruptClaim(0);
  EXPECT_EQ(id, 10);
  // A second claim should return 0.
  EXPECT_EQ(GetInterruptClaim(0), 0);
  // Interrupt is no longer pending.
  EXPECT_FALSE(GetPending(10));
  // Complete the interrupt.
  SetInterruptClaim(0, id);
  // Expect pending to be set, as the level is still high.
  EXPECT_TRUE(GetPending(10));
  // Lower the irq.
  plic()->SetInterrupt(10, /*value=*/false, /*is_level=*/true);
  // Expect pending to still be set.
  EXPECT_TRUE(GetPending(10));
}

// Test operation of edge triggered interrupts.
TEST_F(RiscVPlicTest, EdgeTriggeredInterrupt) {
  SetDefaultConfig();
  // Set irq for source 10 level sensitive.
  plic()->SetInterrupt(10, /*value=*/true, /*is_level=*/false);
  // Expect pending to be set.
  EXPECT_TRUE(GetPending(10));
  // Expect irq to be set for context 0;
  EXPECT_TRUE(target(0)->irq_value());
  // Claim the interrupt.
  uint32_t id = GetInterruptClaim(0);
  EXPECT_EQ(id, 10);
  // A second claim should return 0.
  EXPECT_EQ(GetInterruptClaim(0), 0);
  // Interrupt is no longer pending.
  EXPECT_FALSE(GetPending(10));
  // Complete the interrupt.
  SetInterruptClaim(0, id);
  // Expect pending to be cleared.
  EXPECT_FALSE(GetPending(10));
}

TEST_F(RiscVPlicTest, PriorityThreshold) {
  SetDefaultConfig();
  // Signal interrupts for sources 5-10.
  for (int i = 5; i <= 10; ++i) {
    plic()->SetInterrupt(i, /*value=*/true, /*is_level=*/false);
    // Only sources 9 and 10 should trigger interrupts.
    EXPECT_EQ(target(0)->irq_value(), i > 8);
  }
  // Now claim the interrupts, all should be claimed in order of priority,'
  // even those below the threshold. The IRQ line should remain high until
  // there are no more pending interrupts.
  for (int i = 10; i >= 5; --i) {
    EXPECT_TRUE(target(0)->irq_value());
    EXPECT_TRUE(GetPending(i));
    uint32_t id = GetInterruptClaim(0);
    EXPECT_EQ(id, i);
    SetInterruptClaim(0, id);
    EXPECT_FALSE(GetPending(i));
    // IRQ remains high until the last interrupt is completed.
    EXPECT_EQ(target(0)->irq_value(), i != 5);
  }
}

TEST_F(RiscVPlicTest, MultipleTargets) {
  SetDefaultConfig();
  // Signal interrupt for source 30.
  plic()->SetInterrupt(30, /*value=*/true, /*is_level=*/false);
  EXPECT_TRUE(GetPending(30));
  // Expect irq to be set for all contexts.
  for (int c = 0; c < kNumContexts; ++c) {
    EXPECT_TRUE(target(c)->irq_value());
  }
  // Try claiming the interrupt for each context, only the first will succeed.
  for (int c = 0; c < kNumContexts; ++c) {
    uint32_t id = GetInterruptClaim(c);
    // Expect pending to be cleared.
    EXPECT_FALSE(GetPending(30));
    if (c == 0) {
      EXPECT_EQ(id, 30);
    } else {
      EXPECT_EQ(id, 0);
    }
    // The target should be cleared.
    EXPECT_EQ(target(c)->irq_value(), 0);
  }
  // Complete the interrupt for context 0.
  SetInterruptClaim(0, 30);
}
}  // namespace
