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

#include "riscv/riscv_misa.h"

#include <cstdint>

#include "googlemock/include/gmock/gmock.h"

namespace {

using ::mpact::sim::riscv::RiscVMIsa;

// This tests that the Misa register can be set using the Set method, but not
// using the Write method.

TEST(RiscVMisaTest, WriteTest) {
  RiscVMIsa riscv_misa(0U, nullptr);
  // Initial value is 0.
  EXPECT_EQ(riscv_misa.GetUint32(), 0);
  EXPECT_EQ(riscv_misa.AsUint32(), 0);
  // 32 bit write shouldn't change anything.
  riscv_misa.Write(0xdeadbeef);
  EXPECT_EQ(riscv_misa.GetUint32(), 0);
  EXPECT_EQ(riscv_misa.AsUint32(), 0);
  // 64 bit write shouldn't change anything.
  riscv_misa.Write(static_cast<uint64_t>(0xdeadbeef));
  EXPECT_EQ(riscv_misa.GetUint32(), 0);
  EXPECT_EQ(riscv_misa.AsUint32(), 0);
  // 32 bit Set should work.
  riscv_misa.Set(0xdeadbeef);
  EXPECT_EQ(riscv_misa.GetUint64(), 0xdeadbeef);
  EXPECT_EQ(riscv_misa.AsUint32(), 0xdeadbeef & riscv_misa.read_mask());
  // Reset the value.
  riscv_misa.Set(0U);
  EXPECT_EQ(riscv_misa.GetUint32(), 0);
  EXPECT_EQ(riscv_misa.AsUint32(), 0);
  // 64 bit Set shouldn't change anything.
  riscv_misa.Set(static_cast<uint64_t>(0xdeadbeefULL));
  EXPECT_EQ(riscv_misa.GetUint64(), 0xdeadbeef);
  EXPECT_EQ(riscv_misa.AsUint32(), 0xdeadbeef & riscv_misa.read_mask());
}

}  // namespace
