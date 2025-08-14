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

#include "riscv/riscv32_htif_semihost.h"

#include <unistd.h>

#include <cstring>
#include <string>

#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"

namespace {

using ::mpact::sim::generic::DataBufferFactory;
using ::mpact::sim::riscv::RiscV32HtifSemiHost;
using SemiHostAddresses = RiscV32HtifSemiHost::SemiHostAddresses;

constexpr char kHelloWorld[] = "Hello World\n";

class RiscV32HtifSemiHostTest : public testing::Test {
 protected:
  RiscV32HtifSemiHostTest() {
    memory_ = new mpact::sim::util::FlatDemandMemory(0);
    watcher_ = new mpact::sim::util::MemoryWatcher(memory_);
  }

  ~RiscV32HtifSemiHostTest() override {
    delete watcher_;
    delete memory_;
  }

  DataBufferFactory db_factory_;
  mpact::sim::util::MemoryWatcher* watcher_;
  mpact::sim::util::FlatDemandMemory* memory_;
  SemiHostAddresses semi_host_addresses_ = {0x1000, 0x1008, 0x2000, 0x2008};
};

// Verify that the class is constructed without mishap.
TEST_F(RiscV32HtifSemiHostTest, Constructors) {
  bool halted = false;
  std::string error;
  auto* one = new RiscV32HtifSemiHost(watcher_, memory_, semi_host_addresses_);
  one->SetHaltCallback([&halted]() { halted = true; });
  one->SetErrorCallback([&error](std::string err) { error = err; });
  auto* two = new RiscV32HtifSemiHost(
      watcher_, memory_, semi_host_addresses_, [&halted]() { halted = true; },
      [&error](std::string err) { error = err; });
  delete one;
  delete two;
}

// Ensure that the class detects the halting condition properly.
TEST_F(RiscV32HtifSemiHostTest, Halting) {
  bool halted = false;
  bool error_call = false;
  std::string error;
  auto* semi_host =
      new RiscV32HtifSemiHost(watcher_, memory_, semi_host_addresses_);
  semi_host->SetHaltCallback([&halted]() { halted = true; });
  semi_host->SetErrorCallback([&error, &error_call](std::string msg) {
    error_call = true;
    error = msg;
  });
  auto* db1 = db_factory_.Allocate<uint8_t>(1);
  auto* db8 = db_factory_.Allocate<uint64_t>(1);
  db1->Set<uint8_t>(0, 1);
  db8->Set<uint64_t>(0, 1);
  watcher_->Store(semi_host_addresses_.tohost, db8);
  watcher_->Store(semi_host_addresses_.tohost_ready, db1);

  EXPECT_TRUE(halted);
  EXPECT_FALSE(error_call);
  watcher_->Load(semi_host_addresses_.fromhost_ready, db1, nullptr, nullptr);
  EXPECT_EQ(db1->Get<uint8_t>(0), 0);
  db1->DecRef();
  db8->DecRef();
  delete semi_host;
}

// Verify that errors are signaled appropriately.
TEST_F(RiscV32HtifSemiHostTest, Error) {
  bool halted = false;
  bool error_call = false;
  std::string error;
  auto* semi_host =
      new RiscV32HtifSemiHost(watcher_, memory_, semi_host_addresses_);
  semi_host->SetHaltCallback([&halted]() { halted = true; });
  semi_host->SetErrorCallback([&error, &error_call](std::string msg) {
    error_call = true;
    error = msg;
  });
  auto* db1 = db_factory_.Allocate<uint8_t>(1);
  auto* db8 = db_factory_.Allocate<uint64_t>(1);
  auto* db64 = db_factory_.Allocate<uint64_t>(8);

  db1->Set<uint8_t>(0, 1);
  db8->Set<uint64_t>(0, 0x4000);
  for (auto& el : db64->Get<uint64_t>()) el = 0;
  db64->Set<uint64_t>(0, 123);
  watcher_->Store(0x4000, db64);
  watcher_->Store(semi_host_addresses_.tohost, db8);
  watcher_->Store(semi_host_addresses_.tohost_ready, db1);

  EXPECT_FALSE(halted);
  EXPECT_TRUE(error_call);
  watcher_->Load(semi_host_addresses_.fromhost_ready, db1, nullptr, nullptr);
  EXPECT_EQ(db1->Get<uint8_t>(0), 0);
  db1->DecRef();
  db8->DecRef();
  db64->DecRef();
  delete semi_host;
}

// Test the syswrite system call (printf to stderr).
TEST_F(RiscV32HtifSemiHostTest, Syswrite) {
  testing::internal::CaptureStderr();

  bool halted = false;
  bool error_call = false;
  std::string error;
  auto* semi_host =
      new RiscV32HtifSemiHost(watcher_, memory_, semi_host_addresses_);
  semi_host->SetHaltCallback([&halted]() { halted = true; });
  semi_host->SetErrorCallback([&error, &error_call](std::string msg) {
    error_call = true;
    error = msg;
  });
  auto* db1 = db_factory_.Allocate<uint8_t>(1);
  auto* db8 = db_factory_.Allocate<uint64_t>(1);
  auto* db16 = db_factory_.Allocate<uint8_t>(16);
  auto* db64 = db_factory_.Allocate<uint64_t>(8);

  db1->Set<uint8_t>(0, 1);
  db8->Set<uint64_t>(0, 0x4000);
  std::memcpy(db16->raw_ptr(), kHelloWorld, strlen(kHelloWorld));
  for (auto& el : db64->Get<uint64_t>()) el = 0;
  db64->Set<uint64_t>(0, 64);
  db64->Set<uint64_t>(1, 2);
  db64->Set<uint64_t>(2, 0x4100);
  db64->Set<uint64_t>(3, strlen("Hello World\n"));
  watcher_->Store(0x4100, db16);
  watcher_->Store(0x4000, db64);
  watcher_->Store(semi_host_addresses_.tohost, db8);
  watcher_->Store(semi_host_addresses_.tohost_ready, db1);

  std::string capture_stderr = testing::internal::GetCapturedStderr();

  EXPECT_STREQ("Hello World\n", capture_stderr.c_str());
  EXPECT_FALSE(halted);
  EXPECT_FALSE(error_call);
  watcher_->Load(semi_host_addresses_.fromhost_ready, db1, nullptr, nullptr);
  EXPECT_EQ(db1->Get<uint8_t>(0), 1);
  db1->DecRef();
  db8->DecRef();
  db16->DecRef();
  db64->DecRef();
  delete semi_host;
}

}  // namespace
