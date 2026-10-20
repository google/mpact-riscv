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

#include "riscv/riscv_csr.h"

#include <cstdint>
#include <ios>

#include "absl/status/status.h"
#include "googlemock/include/gmock/gmock.h"  // IWYU pragma: keep
#include "googletest/include/gtest/gtest.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_xstatus.h"

namespace {

#ifndef EXPECT_OK
#define EXPECT_OK(x) EXPECT_TRUE(x.ok())
#endif

using ::mpact::sim::riscv::RiscV32SimpleCsr;
using ::mpact::sim::riscv::RiscVCsrEnum;
using ::mpact::sim::riscv::RiscVMStatus;
using ::mpact::sim::riscv::RiscVShadowCsr;
using ::mpact::sim::riscv::RiscVSStatus;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVUStatus;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::util::FlatDemandMemory;

constexpr char kCsrName0[] = "csr0";
constexpr char kCsrName1[] = "csr1";
constexpr uint32_t kAllOnes = 0xffff'ffff;
constexpr uint32_t kDeadBeef = 0xdead'beef;
constexpr uint32_t kA5 = 0xaaaa'5555;
constexpr uint32_t kReadMask = 0x00ff'ff00;
constexpr uint32_t kWriteMask = 0x000f'f000;

// Test fixture class.
class RiscV32CsrTest : public testing::Test {
 protected:
  RiscV32CsrTest() {
    state_ = new RiscVState("test", RiscVXlen::RV32, &memory_);
  }
  ~RiscV32CsrTest() override { delete state_; }

  FlatDemandMemory memory_;
  RiscVState* state_;
};

// Test that the simple csr constructs properly and with the expected values.
TEST_F(RiscV32CsrTest, SimpleCsrConstruction) {
  auto* csr0 = new RiscV32SimpleCsr(kCsrName0, RiscVCsrEnum::kUScratch,
                                    kDeadBeef, state_);
  EXPECT_EQ(csr0->name(), kCsrName0);
  EXPECT_EQ(csr0->index(), static_cast<int>(RiscVCsrEnum::kUScratch));
  EXPECT_EQ(csr0->read_mask(), kAllOnes);
  EXPECT_EQ(csr0->write_mask(), kAllOnes);
  csr0->set_read_mask(kReadMask);
  csr0->set_write_mask(kWriteMask);
  EXPECT_EQ(csr0->read_mask(), kReadMask);
  EXPECT_EQ(csr0->write_mask(), kWriteMask);

  auto* csr1 = new RiscV32SimpleCsr(kCsrName1, RiscVCsrEnum::kMScratch, kA5,
                                    kReadMask, kWriteMask, state_);
  EXPECT_EQ(csr1->name(), kCsrName1);
  EXPECT_EQ(csr1->index(), static_cast<int>(RiscVCsrEnum::kMScratch));
  EXPECT_EQ(csr1->read_mask(), kReadMask);
  EXPECT_EQ(csr1->write_mask(), kWriteMask);

  delete csr0;
  delete csr1;
}

// Read and write values from/to the csr.
TEST_F(RiscV32CsrTest, SimpleCsrReadWrite) {
  auto* csr = new RiscV32SimpleCsr(kCsrName1, RiscVCsrEnum::kMScratch, kA5,
                                   kReadMask, kWriteMask, state_);
  EXPECT_EQ(csr->AsUint32(), kA5 & kReadMask);
  csr->Write(kAllOnes);
  EXPECT_EQ(csr->AsUint32(), (kA5 & kReadMask) | (kAllOnes & kWriteMask))
      << std::hex << csr->AsUint32()
      << " != " << ((kA5 & kReadMask) | (kAllOnes & kWriteMask));
  delete csr;
}

// Raw read/writes.
TEST_F(RiscV32CsrTest, SimpleCsrSetGet) {
  auto* csr = new RiscV32SimpleCsr(kCsrName1, RiscVCsrEnum::kMScratch, kA5,
                                   kReadMask, kWriteMask, state_);
  EXPECT_EQ(csr->GetUint32(), kA5);
  csr->Set(kAllOnes);
  EXPECT_EQ(csr->AsUint32(), (kAllOnes & kReadMask));
  EXPECT_EQ(csr->GetUint32(), kAllOnes);
  delete csr;
}

// Test the csr-set class.
TEST_F(RiscV32CsrTest, CsrSet) {
  // Allocate new csr.
  auto* csr = new RiscV32SimpleCsr(kCsrName1, RiscVCsrEnum::kUScratch, kA5,
                                   kReadMask, kWriteMask, state_);
  // Add it to the set, then try to add it again. The second attempt should
  // fail.
  EXPECT_OK(state_->csr_set()->AddCsr(csr));
  EXPECT_EQ(state_->csr_set()->AddCsr(csr).code(),
            absl::StatusCode::kAlreadyExists);
  // Read the csr and validate the value.
  auto result =
      state_->csr_set()->GetCsr(static_cast<int>(RiscVCsrEnum::kUScratch));
  EXPECT_OK(result.status());
  auto* stored_csr = result.value();
  EXPECT_EQ(stored_csr, csr);
  EXPECT_EQ(stored_csr->AsUint32(), (kA5 & kReadMask));
  // Getting a different csr should fail.
  EXPECT_EQ(state_->csr_set()->GetCsr(0xffff).status().code(),
            absl::StatusCode::kNotFound);
  delete csr;
}

// Test that the shadow csr constructs properly and with the expected values.
TEST_F(RiscV32CsrTest, ShadowCsrConstruction) {
  auto* csr0 = new RiscV32SimpleCsr(kCsrName0, RiscVCsrEnum::kMScratch,
                                    kDeadBeef, state_);
  EXPECT_EQ(csr0->name(), kCsrName0);
  EXPECT_EQ(csr0->index(), static_cast<int>(RiscVCsrEnum::kMScratch));

  auto* csr1 = new RiscVShadowCsr<uint32_t>(
      kCsrName1, RiscVCsrEnum::kUScratch, kReadMask, kWriteMask, state_, csr0);
  EXPECT_EQ(csr1->name(), kCsrName1);
  EXPECT_EQ(csr1->index(), static_cast<int>(RiscVCsrEnum::kUScratch));
  EXPECT_EQ(csr1->read_mask(), kReadMask);
  EXPECT_EQ(csr1->write_mask(), kWriteMask);

  EXPECT_EQ(csr1->AsUint32(), csr0->AsUint32() & kReadMask);
  csr1->Write(kAllOnes);
  EXPECT_EQ(csr0->AsUint32(), kDeadBeef | (kAllOnes & kWriteMask));
  delete csr0;
  delete csr1;
}

TEST_F(RiscV32CsrTest, XStatusCsrs) {
  constexpr uint64_t kCustomMask =
      RiscVMStatus::kDefaultReadMask | 0x6000'0600ULL;
  RiscVMStatus mstatus32_default(uint32_t{0}, state_, state_->misa());
  EXPECT_EQ(mstatus32_default.kReadMask, RiscVMStatus::kDefaultReadMask);
  EXPECT_EQ(mstatus32_default.kWriteMask, RiscVMStatus::kDefaultWriteMask);

  RiscVMStatus mstatus32_custom(uint32_t{0}, state_, state_->misa(),
                                kCustomMask, kCustomMask);
  EXPECT_EQ(mstatus32_custom.kReadMask, kCustomMask);
  EXPECT_EQ(mstatus32_custom.kWriteMask, kCustomMask);
  mstatus32_custom.Write(uint32_t{0x6000'0600});
  EXPECT_EQ(mstatus32_custom.AsUint32() & 0x6000'0600U, 0x6000'0600U);

  RiscVMStatus mstatus64_custom(uint64_t{0}, state_, state_->misa(),
                                kCustomMask, kCustomMask);
  EXPECT_EQ(mstatus64_custom.kReadMask, kCustomMask);

  RiscVSStatus sstatus(&mstatus32_custom, state_);
  sstatus.Write(uint32_t{0x1});
  EXPECT_EQ(sstatus.AsUint32() & 0x1U, 0x1U);

  RiscVUStatus ustatus(&mstatus32_custom, state_);
  ustatus.Write(uint32_t{0x11});
  EXPECT_EQ(ustatus.AsUint32() & 0x11U, 0x11U);
  EXPECT_EQ(ustatus.AsUint64() & 0x11ULL, 0x11ULL);
  EXPECT_TRUE(ustatus.uie());
  ustatus.ClearBits(uint32_t{0x1});
  EXPECT_FALSE(ustatus.uie());
  ustatus.SetBits(uint32_t{0x1});
  EXPECT_TRUE(ustatus.uie());
  ustatus.Set(uint32_t{0});
  EXPECT_EQ(ustatus.GetUint32() & 0x11U, 0U);
  ustatus.Set(uint64_t{0x11});
  EXPECT_EQ(ustatus.GetUint64() & 0x11ULL, 0x11ULL);
  ustatus.set_uie(0);
  mstatus32_custom.Submit();
  EXPECT_FALSE(ustatus.uie());
}

}  // namespace
