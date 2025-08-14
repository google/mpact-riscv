#include "riscv/riscv_clint.h"

#include <cstdint>

#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/counters.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "riscv/riscv_state.h"

// This file contains unit tests for RiscVClint.

namespace {

// MMR offsets.
constexpr uint64_t kMSip = 0x0000;
constexpr uint64_t kMTimeCmp = 0x4000;
constexpr uint64_t kMTime = 0xbff8;

using ::mpact::sim::generic::Instruction;
using ::mpact::sim::generic::SimpleCounter;
using ::mpact::sim::riscv::RiscVClint;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::util::FlatDemandMemory;

// Test fixture for CherioClint tests.
class RiscVClintTest : public ::testing::Test {
 public:
  RiscVClintTest() {
    state_ = new RiscVState("test", RiscVXlen::RV32, &memory_);
    state_->set_on_trap([this](bool is_interrupt, uint64_t trap_value,
                               uint64_t exception_code, uint64_t epc,
                               const Instruction* inst) {
      return TrapHandler(is_interrupt, trap_value, exception_code, epc, inst);
    });
    cycle_counter_.Initialize("cycle_counter", 0);
  }

  ~RiscVClintTest() override { delete state_; }

  // Gets called on a trap.
  bool TrapHandler(bool is_interrupt, uint64_t trap_value,
                   uint64_t exception_code, uint64_t epc,
                   const Instruction* inst);

  FlatDemandMemory memory_;
  // Counter.
  SimpleCounter<uint64_t> cycle_counter_;
  // CherIoT state.
  RiscVState* state_;
  // Latched trap info.
  bool trap_taken_ = false;
  bool trap_is_interrupt_ = false;
  uint64_t trap_value_ = 0;
  uint64_t trap_exception_code_ = 0;
  uint64_t trap_epc_ = 0;
  const Instruction* trap_inst_ = nullptr;
};

// Called when there is a trap in the CherIoT state.
bool RiscVClintTest::TrapHandler(bool is_interrupt, uint64_t trap_value,
                                 uint64_t exception_code, uint64_t epc,
                                 const Instruction* inst) {
  trap_taken_ = true;
  trap_is_interrupt_ = is_interrupt;
  trap_value_ = trap_value;
  trap_exception_code_ = exception_code;
  trap_epc_ = epc;
  trap_inst_ = inst;
  return true;
}

// Test loads and stores for msip memory mapped register.
TEST_F(RiscVClintTest, MSip) {
  auto* db = state_->db_factory()->Allocate<uint32_t>(1);
  db->Set<uint32_t>(0, 0xdeadbeef);
  auto* clint = new RiscVClint(/*period=*/1, state_->mip());
  cycle_counter_.AddListener(clint);
  // Initial value should be zero.
  clint->Load(kMSip, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 0);
  // mip reg bit should be zero.
  EXPECT_EQ(state_->mip()->msip(), 0);

  // Write a 1 - this should cause the msip bit in mip to be set.
  db->Set<uint32_t>(0, 1);
  clint->Store(kMSip, db);

  // Now the value should be 1.
  db->Set<uint32_t>(0, 0xdeadbeef);
  clint->Load(kMSip, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 1);
  // mip reg bit should be set too.
  EXPECT_EQ(state_->mip()->msip(), 1);

  // Write a value with the last bit cleared.
  db->Set<uint32_t>(0, 0xdeadbeee);
  clint->Store(kMSip, db);
  // Now the value should be 0.
  db->Set<uint32_t>(0, 0xdeadbeef);
  clint->Load(kMSip, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 0);
  // mip reg bit should be cleared too.
  EXPECT_EQ(state_->mip()->msip(), 0);
  delete clint;
  db->DecRef();
}

// Test reads/writes to mtimecmp.
TEST_F(RiscVClintTest, MTimeCmp) {
  auto* db = state_->db_factory()->Allocate<uint32_t>(1);
  auto* clint = new RiscVClint(/*period=*/100, state_->mip());
  cycle_counter_.AddListener(clint);

  // Verify value of mtimecmp.
  clint->Load(kMTimeCmp, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 0);
  clint->Load(kMTimeCmp + 4, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 0);

  // Initially, the mip mtip bit should be set, as the mtime is 0 and the
  // mtimecmp is 0.
  EXPECT_EQ(state_->mip()->mtip(), 1);

  // Set mtimecmp to 10, then mtip should be 0.
  db->Set<uint32_t>(0, 10);
  clint->Store(kMTimeCmp, db);
  // Verify value of mtimecmp.
  clint->Load(kMTimeCmp, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 10);
  clint->Load(kMTimeCmp + 4, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 0);
  // mtip should be cleared.
  EXPECT_EQ(state_->mip()->mtip(), 0);
  // Increment the counter 998 times - during which mtip stays 0.
  for (int i = 0; i < 998; ++i) {
    cycle_counter_.Increment(1);
    EXPECT_EQ(state_->mip()->mtip(), 0);
  }
  // Increment once more - mtip should be set.
  cycle_counter_.Increment(1);
  EXPECT_EQ(state_->mip()->mtip(), 1);

  // Set mtimecmp to 5 - mtip should still be set.
  db->Set<uint32_t>(0, 5);
  clint->Store(kMTimeCmp, db);
  EXPECT_EQ(state_->mip()->mtip(), 1);
  // Verify value of mtimecmp.
  clint->Load(kMTimeCmp, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 5);
  clint->Load(kMTimeCmp + 4, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 0);

  // Set mtimecmp to 0x1005 - mtip should be cleared.
  db->Set<uint32_t>(0, 1);
  clint->Store(kMTimeCmp + 4, db);
  EXPECT_EQ(state_->mip()->mtip(), 0);
  // Verify value of mtimecmp.
  clint->Load(kMTimeCmp, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 5);
  clint->Load(kMTimeCmp + 4, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 1);

  // Update mtime.
  db->Set<uint32_t>(0, 0x0000'0004);
  clint->Store(kMTime, db);
  db->Set<uint32_t>(0, 0x0000'00001);
  clint->Store(kMTime + 4, db);
  for (int i = 0; i < 100; i++) {
    cycle_counter_.Increment(1);
  }
  EXPECT_EQ(state_->mip()->mtip(), 1);

  // Clean up.
  delete clint;
  db->DecRef();
}

// Test the reads/writes to mtime.
TEST_F(RiscVClintTest, MTime) {
  auto* db = state_->db_factory()->Allocate<uint32_t>(1);
  auto clint = new RiscVClint(/*period=*/100, state_->mip());
  cycle_counter_.AddListener(clint);
  // Initial value should be zero.
  clint->Load(kMTime, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 0);
  clint->Load(kMTime + 4, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 0);

  // Increment time.
  for (int i = 0; i < 100; ++i) cycle_counter_.Increment(1);
  clint->Load(kMTime, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 1);
  clint->Load(kMTime + 4, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 0);

  // Update mtime to a larger number.
  db->Set<uint32_t>(0, 0xffff'ffff);
  clint->Store(kMTime, db);
  // Read back the values.
  clint->Load(kMTime, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 0xffff'ffff);
  clint->Load(kMTime + 4, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 0);

  // Increment twice.
  for (int i = 0; i < 100; ++i) cycle_counter_.Increment(1);
  clint->Load(kMTime, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 0);
  clint->Load(kMTime + 4, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 1);
  for (int i = 0; i < 100; ++i) cycle_counter_.Increment(1);

  clint->Load(kMTime, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 1);
  clint->Load(kMTime + 4, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 1);

  // Call reset - which writes 0 to mtimer.
  clint->Reset();
  // Verify that the values are zero.
  clint->Load(kMTime, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 0);
  clint->Load(kMTime + 4, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 0);

  // Update mtime to another large number.
  db->Set<uint32_t>(0, 0xffff'ffff);
  clint->Store(kMTime, db);
  db->Set<uint32_t>(0, 0x0003'0004);
  clint->Store(kMTime + 4, db);
  // Read back the values.
  clint->Load(kMTime, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 0xffff'ffff);
  clint->Load(kMTime + 4, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 0x0003'0004);

  // Increment time and verify the value increased accordingly.
  for (int i = 0; i < 100; ++i) cycle_counter_.Increment(1);
  clint->Load(kMTime, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 0x0000'0000U);
  clint->Load(kMTime + 4, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 0x0003'0005);

  delete clint;
  db->DecRef();
}

}  // namespace
