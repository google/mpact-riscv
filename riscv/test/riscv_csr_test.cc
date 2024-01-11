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

#include <ios>

#include "googlemock/include/gmock/gmock.h"
#include "absl/status/status.h"
#include "riscv/riscv_state.h"

namespace {

#ifndef EXPECT_OK
#define EXPECT_OK(x) EXPECT_TRUE(x.ok())
#endif

using ::mpact::sim::riscv::RiscV32SimpleCsr;
using ::mpact::sim::riscv::RiscVCsrEnum;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVXlen;

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
  RiscV32CsrTest() { state_ = new RiscVState("test", RiscVXlen::RV32); }
  ~RiscV32CsrTest() override { delete state_; }

  RiscVState *state_;
};

// Test that the simple csr constructs properly and with the expected values.
TEST_F(RiscV32CsrTest, SimpleCsrConstruction) {
  auto *csr0 = new RiscV32SimpleCsr(kCsrName0, RiscVCsrEnum::kUScratch,
                                    kDeadBeef, state_);
  EXPECT_EQ(csr0->name(), kCsrName0);
  EXPECT_EQ(csr0->index(), static_cast<int>(RiscVCsrEnum::kUScratch));
  EXPECT_EQ(csr0->read_mask(), kAllOnes);
  EXPECT_EQ(csr0->write_mask(), kAllOnes);
  csr0->set_read_mask(kReadMask);
  csr0->set_write_mask(kWriteMask);
  EXPECT_EQ(csr0->read_mask(), kReadMask);
  EXPECT_EQ(csr0->write_mask(), kWriteMask);

  auto *csr1 = new RiscV32SimpleCsr(kCsrName1, RiscVCsrEnum::kMScratch, kA5,
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
  auto *csr = new RiscV32SimpleCsr(kCsrName1, RiscVCsrEnum::kMScratch, kA5,
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
  auto *csr = new RiscV32SimpleCsr(kCsrName1, RiscVCsrEnum::kMScratch, kA5,
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
  auto *csr = new RiscV32SimpleCsr(kCsrName1, RiscVCsrEnum::kUScratch, kA5,
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
  auto *stored_csr = result.value();
  EXPECT_EQ(stored_csr, csr);
  EXPECT_EQ(stored_csr->AsUint32(), (kA5 & kReadMask));
  // Getting a different csr should fail.
  EXPECT_EQ(state_->csr_set()->GetCsr(0xffff).status().code(),
            absl::StatusCode::kNotFound);
  delete csr;
}

}  // namespace
