#include "riscv/riscv_counter_csr.h"

#include <cstdint>
#include <limits>

#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/counters.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_state.h"

// This file contains unit tests for counter_csr classes used to implement
// mcycle/mcycleh and minstret/minstreth.

namespace {

using ::mpact::sim::generic::SimpleCounter;
using ::mpact::sim::riscv::RiscVCounterCsr;
using ::mpact::sim::riscv::RiscVCounterCsrHigh;
using ::mpact::sim::riscv::RiscVCsrEnum;
using ::mpact::sim::riscv::RiscVPerformanceCounterCsr;
using ::mpact::sim::riscv::RiscVPerformanceCounterCsrHigh;
using ::mpact::sim::riscv::RiscVState;

class RiscVMCycleTest : public ::testing::Test {
 protected:
  RiscVMCycleTest() : counter_("cycles", 0) {};
  ~RiscVMCycleTest() override = default;

  SimpleCounter<uint64_t> counter_;
};

// Verify the operation of 64 bit mcycle with counter increments.
TEST_F(RiscVMCycleTest, GetTest64) {
  RiscVPerformanceCounterCsr<uint64_t, RiscVState> mcycle(
      "mcycle", RiscVCsrEnum::kMCycle, nullptr);
  mcycle.set_counter(&counter_);
  // Initial value should be zero.
  EXPECT_EQ(mcycle.GetUint32(), 0);
  EXPECT_EQ(mcycle.GetUint64(), 0);
  counter_.Increment(1);
  EXPECT_EQ(mcycle.GetUint32(), 1);
  EXPECT_EQ(mcycle.GetUint64(), 1);
  mcycle.Write(100u);
  counter_.Increment(1);
  EXPECT_EQ(mcycle.GetUint32(), 100);
  EXPECT_EQ(mcycle.GetUint64(), 100);
  mcycle.Set(static_cast<uint64_t>(1000));
  counter_.Increment(1);
  EXPECT_EQ(mcycle.GetUint32(), 1000);
  EXPECT_EQ(mcycle.GetUint64(), 1000);
}

// Verify the operation of 32 bit mcycle and mcycleh with counter increments.
TEST_F(RiscVMCycleTest, GetTest32) {
  RiscVPerformanceCounterCsr<uint32_t, RiscVState> mcycle(
      "mcycle", RiscVCsrEnum::kMCycle, nullptr);
  RiscVPerformanceCounterCsrHigh<RiscVState> mcycleh(
      "mcycleh", RiscVCsrEnum::kMCycleH, nullptr, &mcycle);
  mcycle.set_counter(&counter_);
  mcycleh.set_counter(&counter_);
  // Initial value should be zero.
  EXPECT_EQ(mcycle.GetUint32(), 0);
  EXPECT_EQ(mcycle.GetUint64(), 0);
  EXPECT_EQ(mcycleh.GetUint32(), 0);
  EXPECT_EQ(mcycleh.GetUint64(), 0);
  counter_.Increment(1);
  // Only low 32 bits should be set.
  EXPECT_EQ(mcycle.GetUint32(), 1);
  EXPECT_EQ(mcycle.GetUint64(), 1);
  EXPECT_EQ(mcycleh.GetUint32(), 0);
  EXPECT_EQ(mcycleh.GetUint64(), 0);
  // Increment counter by uint32_t max.
  counter_.Increment(std::numeric_limits<uint32_t>::max());
  // Counters should have wrapped around.
  EXPECT_EQ(mcycle.GetUint32(), 0);
  EXPECT_EQ(mcycle.GetUint64(), 0);
  EXPECT_EQ(mcycleh.GetUint32(), 1);
  EXPECT_EQ(mcycleh.GetUint64(), 1);
  // Increment counter by 1 again.
  counter_.Increment(1);
  EXPECT_EQ(mcycle.GetUint32(), 1);
  EXPECT_EQ(mcycle.GetUint64(), 1);
  EXPECT_EQ(mcycleh.GetUint32(), 1);
  EXPECT_EQ(mcycleh.GetUint64(), 1);
}

// Test that write to mcycle is reflected in the value of mcycle.
TEST_F(RiscVMCycleTest, SetTest64) {
  RiscVPerformanceCounterCsr<uint64_t, RiscVState> mcycle(
      "mcycle", RiscVCsrEnum::kMCycle, nullptr);
  mcycle.set_counter(&counter_);
  EXPECT_EQ(mcycle.GetUint32(), 0);
  EXPECT_EQ(mcycle.GetUint64(), 0);
  mcycle.Write(100u);
  counter_.Increment(1);
  EXPECT_EQ(mcycle.GetUint32(), 100);
  EXPECT_EQ(mcycle.GetUint64(), 100);
  counter_.Increment(10);
  EXPECT_EQ(mcycle.GetUint32(), 110);
  EXPECT_EQ(mcycle.GetUint64(), 110);
}

// Test that write to mcycle and mcycleh is reflected in the value of mcycle and
// mcycleh.
TEST_F(RiscVMCycleTest, SetTest32) {
  RiscVPerformanceCounterCsr<uint32_t, RiscVState> mcycle(
      "mcycle", RiscVCsrEnum::kMCycle, nullptr);
  RiscVPerformanceCounterCsrHigh<RiscVState> mcycleh(
      "mcycleh", RiscVCsrEnum::kMCycleH, nullptr, &mcycle);
  mcycle.set_counter(&counter_);
  mcycleh.set_counter(&counter_);
  EXPECT_EQ(mcycle.GetUint32(), 0);
  EXPECT_EQ(mcycle.GetUint64(), 0);
  EXPECT_EQ(mcycleh.GetUint32(), 0);
  EXPECT_EQ(mcycleh.GetUint64(), 0);
  mcycle.Write(100u);
  counter_.Increment(1);
  mcycleh.Write(200u);
  counter_.Increment(1);
  EXPECT_EQ(mcycle.GetUint32(), 100);
  EXPECT_EQ(mcycle.GetUint64(), 100);
  EXPECT_EQ(mcycleh.GetUint32(), 200);
  EXPECT_EQ(mcycleh.GetUint64(), 200);
  counter_.Increment(10);
  EXPECT_EQ(mcycle.GetUint32(), 110);
  EXPECT_EQ(mcycle.GetUint64(), 110);
  EXPECT_EQ(mcycleh.GetUint32(), 200);
  EXPECT_EQ(mcycleh.GetUint64(), 200);
  counter_.Increment(std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(mcycle.GetUint32(), 109);
  EXPECT_EQ(mcycle.GetUint64(), 109);
  EXPECT_EQ(mcycleh.GetUint32(), 201);
  EXPECT_EQ(mcycleh.GetUint64(), 201);
}

// Verify the operation of minstret with counter increments and writes.
// Specifically, ensure that writing to the CSR masks the increment of the
// retiring instruction.
class RiscVMInstretTest : public ::testing::Test {
 protected:
  RiscVMInstretTest() : counter_("instructions", 0) {};
  ~RiscVMInstretTest() override = default;

  SimpleCounter<uint64_t> counter_;
};

// Test that write to 64-bit minstret accounts for the instruction increment.
TEST_F(RiscVMInstretTest, SetTest64) {
  RiscVPerformanceCounterCsr<uint64_t, RiscVState> minstret(
      "minstret", RiscVCsrEnum::kMInstret, nullptr);
  minstret.set_counter(&counter_);

  EXPECT_EQ(minstret.GetUint64(), 0);

  // Write a value to the CSR.
  minstret.Set(1000u);
  // Simulate the instruction retirement (the instruction that did the write).
  counter_.Increment(1);

  // The value read should be exactly what was written, because writes to
  // minstret don't increment the counter.
  EXPECT_EQ(minstret.GetUint64(), 1000);

  // Subsequent increments should work normally.
  counter_.Increment(1);
  EXPECT_EQ(minstret.GetUint64(), 1001);
}

// Test that writes to 32-bit minstret/minstreth account for the instruction
// increment.
TEST_F(RiscVMInstretTest, SetTest32) {
  RiscVPerformanceCounterCsr<uint32_t, RiscVState> minstret(
      "minstret", RiscVCsrEnum::kMInstret, nullptr);
  RiscVPerformanceCounterCsrHigh<RiscVState> minstreth(
      "minstreth", RiscVCsrEnum::kMInstretH, nullptr, &minstret);
  minstret.set_counter(&counter_);
  minstreth.set_counter(&counter_);

  EXPECT_EQ(minstret.GetUint32(), 0);
  EXPECT_EQ(minstreth.GetUint32(), 0);

  // 1. Write Low CSR.
  minstret.Set(100u);
  counter_.Increment(1);
  // Should equal 100 (increment compensated).
  EXPECT_EQ(minstret.GetUint32(), 100);
  EXPECT_EQ(minstreth.GetUint32(), 0);

  // 2. Write High CSR.
  minstreth.Set(5u);
  counter_.Increment(1);
  // Should equal 5 (increment compensated).
  EXPECT_EQ(minstreth.GetUint32(), 5);
  // Side effect: The increment compensation on the high write effectively
  // stalls the low counter for this cycle. This is consistent with preserving
  // the exact 64-bit value desired.
  EXPECT_EQ(minstret.GetUint32(), 100);

  // 3. Normal increment.
  counter_.Increment(1);
  EXPECT_EQ(minstret.GetUint32(), 101);
  EXPECT_EQ(minstreth.GetUint32(), 5);
}

}  // namespace
