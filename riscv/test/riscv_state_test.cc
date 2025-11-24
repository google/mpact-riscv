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

#include "riscv/riscv_state.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_register.h"

namespace {

using ::mpact::sim::riscv::RiscVCsrEnum;
using ::mpact::sim::riscv::RiscVCsrInterface;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::util::FlatDemandMemory;

constexpr int kPcValue = 0x1000;
constexpr int kMemAddr = 0x1200;
constexpr uint32_t kMemValue = 0xdeadbeef;

// Only limited testing of the RiscVState class for now as it has limited
// additional functionality over the ArchState class.

TEST(RiscVStateTest, Basic) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);
  // Make sure pc has been created.
  auto iter = state->registers()->find("pc");
  auto* ptr = (iter != state->registers()->end()) ? iter->second : nullptr;
  CHECK_NE(ptr, nullptr);
  // Set pc to 0x1000, then read value back through pc operand.
  auto* pc = static_cast<RV32Register*>(ptr);
  pc->data_buffer()->Set<uint32_t>(0, kPcValue);
  auto* pc_op = state->pc_operand();
  EXPECT_EQ(pc_op->AsUint32(0), kPcValue);
}

TEST(RiscVStateTest, Memory) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);
  auto* db = state->db_factory()->Allocate<uint32_t>(1);
  state->LoadMemory(nullptr, kMemAddr, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 0);
  db->Set<uint32_t>(0, kMemValue);
  state->StoreMemory(nullptr, kMemAddr, db);
  db->Set<uint32_t>(0, 0);
  state->LoadMemory(nullptr, kMemAddr, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), kMemValue);
  db->DecRef();
}

TEST(RiscVStateTest, OutOfBoundLoad) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);
  state->set_max_physical_address(kMemAddr - 4);
  state->set_on_trap([](bool is_interrupt, uint64_t trap_value,
                        uint64_t exception_code, uint64_t epc,
                        const mpact::sim::riscv::Instruction* inst) -> bool {
    if (exception_code ==
        static_cast<uint64_t>(
            mpact::sim::riscv::ExceptionCode::kLoadAccessFault)) {
      std::cerr << "Load Access Fault" << std::endl;
      return true;
    }
    return false;
  });
  auto* db = state->db_factory()->Allocate<uint32_t>(1);
  // Create a dummy instruction so trap can dereference the address.
  auto* dummy_inst = new mpact::sim::riscv::Instruction(0x0, nullptr);
  dummy_inst->set_size(4);
  testing::internal::CaptureStderr();
  state->LoadMemory(dummy_inst, kMemAddr, db, nullptr, nullptr);
  const std::string stderr = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr, testing::HasSubstr("Load Access Fault"));
  db->DecRef();
  dummy_inst->DecRef();
}

TEST(RiscVStateTest, PerfCounterCsrNameAndIndexMatch_hpm) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);

  uint32_t hpmcounter_base = static_cast<uint32_t>(RiscVCsrEnum::kCycle);
  for (int i = 3; i < 32; i++) {
    absl::StatusOr<RiscVCsrInterface*> csr_by_name =
        state->csr_set()->GetCsr(absl::StrCat("hpmcounter", i));
    absl::StatusOr<RiscVCsrInterface*> csr_by_index =
        state->csr_set()->GetCsr(hpmcounter_base + i);
    ASSERT_TRUE(csr_by_name.ok());
    ASSERT_TRUE(csr_by_index.ok());
    EXPECT_EQ(*csr_by_name, *csr_by_index);
  }
}

TEST(RiscVStateTest, PerfCounterCsrNameAndIndexMatch_hpm_high) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);

  uint32_t hpmcounter_base_high = static_cast<uint32_t>(RiscVCsrEnum::kCycleH);
  for (int i = 3; i < 32; i++) {
    absl::StatusOr<RiscVCsrInterface*> csr_by_name =
        state->csr_set()->GetCsr(absl::StrCat("hpmcounter", i, "h"));
    absl::StatusOr<RiscVCsrInterface*> csr_by_index =
        state->csr_set()->GetCsr(hpmcounter_base_high + i);
    ASSERT_TRUE(csr_by_name.ok());
    ASSERT_TRUE(csr_by_index.ok());
    EXPECT_EQ(*csr_by_name, *csr_by_index);
  }
}

TEST(RiscVStateTest, PerfCounterCsrNameAndIndexMatch_mhpm) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);

  uint32_t mhpmcounter_base = static_cast<uint32_t>(RiscVCsrEnum::kMCycle);
  for (int i = 3; i < 32; i++) {
    absl::StatusOr<RiscVCsrInterface*> csr_by_name =
        state->csr_set()->GetCsr(absl::StrCat("mhpmcounter", i));
    absl::StatusOr<RiscVCsrInterface*> csr_by_index =
        state->csr_set()->GetCsr(mhpmcounter_base + i);
    ASSERT_TRUE(csr_by_name.ok());
    ASSERT_TRUE(csr_by_index.ok());
    EXPECT_EQ(*csr_by_name, *csr_by_index);
  }
}

TEST(RiscVStateTest, PerfCounterCsrNameAndIndexMatch_mhpm_high) {
  FlatDemandMemory memory;
  auto state = std::make_unique<RiscVState>("test", RiscVXlen::RV32, &memory);

  uint32_t mhpmcounter_base_high =
      static_cast<uint32_t>(RiscVCsrEnum::kMCycleH);
  for (int i = 3; i < 32; i++) {
    absl::StatusOr<RiscVCsrInterface*> csr_by_name =
        state->csr_set()->GetCsr(absl::StrCat("mhpmcounter", i, "h"));
    absl::StatusOr<RiscVCsrInterface*> csr_by_index =
        state->csr_set()->GetCsr(mhpmcounter_base_high + i);
    ASSERT_TRUE(csr_by_name.ok());
    ASSERT_TRUE(csr_by_index.ok());
    EXPECT_EQ(*csr_by_name, *csr_by_index);
  }
}

}  // namespace
