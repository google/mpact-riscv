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

#include "riscv/riscv_top.h"

#include <cstdint>
#include <string>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/core_debug_interface.h"
#include "mpact/sim/generic/decoder_interface.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "mpact/sim/util/memory/memory_watcher.h"
#include "mpact/sim/util/program_loader/elf_program_loader.h"
#include "riscv/riscv32_decoder.h"
#include "riscv/riscv32_htif_semihost.h"
#include "riscv/riscv64_decoder.h"
#include "riscv/riscv_arm_semihost.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_register_aliases.h"
#include "riscv/riscv_state.h"

namespace {

using ::mpact::sim::generic::operator*;  // NOLINT: is used below (clang error).

#ifndef EXPECT_OK
#define EXPECT_OK(x) EXPECT_TRUE(x.ok())
#endif

using ::mpact::sim::generic::DecoderInterface;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::RiscV32Decoder;
using ::mpact::sim::riscv::RiscV64Decoder;
using ::mpact::sim::riscv::RiscVArmSemihost;
using ::mpact::sim::riscv::RiscVFPState;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVTop;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RV64Register;
using ::mpact::sim::riscv::RVFpRegister;
using ::mpact::sim::util::FlatDemandMemory;

using HaltReason = ::mpact::sim::generic::CoreDebugInterface::HaltReason;
constexpr char kHtifFileName[] = "hello_world.elf";
constexpr char kArmFileName[] = "hello_world_arm.elf";
constexpr char kArm64FileName[] = "hello_world_64.elf";

// The depot path to the test directory.
constexpr char kDepotPath[] = "riscv/test/";

// Helper function to get symbol addresses from the loader.
static bool GetMagicAddresses(
    mpact::sim::util::ElfProgramLoader* loader,
    mpact::sim::riscv::RiscV32HtifSemiHost::SemiHostAddresses* magic) {
  auto result = loader->GetSymbol("tohost_ready");
  if (!result.ok()) return false;
  magic->tohost_ready = result.value().first;

  result = loader->GetSymbol("tohost");
  if (!result.ok()) return false;
  magic->tohost = result.value().first;

  result = loader->GetSymbol("fromhost_ready");
  if (!result.ok()) return false;
  magic->fromhost_ready = result.value().first;

  result = loader->GetSymbol("fromhost");
  if (!result.ok()) return false;
  magic->fromhost = result.value().first;

  return true;
}

// Helper RAII style class for setting up HTIF semihosting.
class HtifSemihostSetup {
 public:
  HtifSemihostSetup(RiscVTop* top, mpact::sim::util::ElfProgramLoader* loader,
                    mpact::sim::util::MemoryInterface* memory)
      : memory_(memory), state_(top->state()) {
    mpact::sim::riscv::RiscV32HtifSemiHost::SemiHostAddresses magic;
    if (GetMagicAddresses(loader, &magic)) {
      watcher_ = new mpact::sim::util::MemoryWatcher(memory);
      semihost_ = new mpact::sim::riscv::RiscV32HtifSemiHost(
          watcher_, memory, magic,
          [top]() {
            top->RequestHalt(RiscVTop::HaltReason::kSemihostHaltRequest,
                             nullptr);
          },
          [top](std::string) {
            top->RequestHalt(RiscVTop::HaltReason::kSemihostHaltRequest,
                             nullptr);
          });
      top->state()->set_memory(watcher_);
    }
  }

  ~HtifSemihostSetup() {
    state_->set_memory(memory_);
    delete semihost_;
    delete watcher_;
  }

 private:
  mpact::sim::util::MemoryInterface* memory_ = nullptr;
  mpact::sim::riscv::RiscVState* state_ = nullptr;
  mpact::sim::util::MemoryWatcher* watcher_ = nullptr;
  mpact::sim::riscv::RiscV32HtifSemiHost* semihost_ = nullptr;
};

// Helper RAII style class for setting up ARM semihosting.
class ArmSemihostSetup {
 public:
  explicit ArmSemihostSetup(RiscVTop* top,
                            mpact::sim::util::MemoryInterface* memory) {
    auto xlen = top->state()->xlen();
    if (xlen == RiscVXlen::RV64) {
      semihost_ = new RiscVArmSemihost(RiscVArmSemihost::BitWidth::kWord64,
                                       memory, memory);
    } else {
      semihost_ = new RiscVArmSemihost(RiscVArmSemihost::BitWidth::kWord32,
                                       memory, memory);
    }
    auto semihost = semihost_;
    top->state()->AddEbreakHandler([semihost,
                                    top](const Instruction* inst) -> bool {
      if (semihost->IsSemihostingCall(inst)) {
        semihost->OnEBreak(inst);
      } else {
        top->RequestHalt(RiscVTop::HaltReason::kSoftwareBreakpoint, nullptr);
      }
      return true;
    });
    semihost_->set_exit_callback([top]() {
      top->RequestHalt(RiscVTop::HaltReason::kSemihostHaltRequest, nullptr);
    });
  }

  ~ArmSemihostSetup() { delete semihost_; }

 private:
  mpact::sim::riscv::RiscVArmSemihost* semihost_ = nullptr;
};

class RiscVTopTest : public testing::Test {
 protected:
  RiscVTopTest() {
    memory_ = new FlatDemandMemory();
    state_ = new RiscVState("RV32", RiscVXlen::RV32, memory_);
    fp_state_ = new RiscVFPState(state_->csr_set(), state_);
    state_->set_rv_fp(fp_state_);
    decoder_ = new RiscV32Decoder(state_, memory_);

    // Make sure the architectural and abi register aliases are added.
    std::string reg_name;
    for (int i = 0; i < 32; i++) {
      reg_name = absl::StrCat(RiscVState::kXregPrefix, i);
      (void)state_->AddRegister<RV32Register>(reg_name);
      (void)state_->AddRegisterAlias<RV32Register>(
          reg_name, ::mpact::sim::riscv::kXRegisterAliases[i]);
    }
    for (int i = 0; i < 32; i++) {
      reg_name = absl::StrCat(RiscVState::kFregPrefix, i);
      (void)state_->AddRegister<RVFpRegister>(reg_name);
      (void)state_->AddRegisterAlias<RVFpRegister>(
          reg_name, ::mpact::sim::riscv::kFRegisterAliases[i]);
    }

    riscv_top_ = new RiscVTop("RV32", state_, decoder_);
    // Set up the elf loader.
    loader_ = new mpact::sim::util::ElfProgramLoader(memory_);
  }

  ~RiscVTopTest() override {
    delete loader_;
    delete decoder_;
    delete fp_state_;
    delete riscv_top_;
    delete state_;
    delete memory_;
  }

  void LoadFile(const std::string file_name) {
    const std::string input_file_name =
        absl::StrCat(kDepotPath, "testfiles/", file_name);
    auto result = loader_->LoadProgram(input_file_name);
    CHECK_OK(result);
    entry_point_ = result.value();
  }

  uint64_t entry_point_;
  RiscVTop* riscv_top_ = nullptr;
  mpact::sim::util::ElfProgramLoader* loader_ = nullptr;
  FlatDemandMemory* memory_ = nullptr;
  RiscVState* state_ = nullptr;
  RiscVFPState* fp_state_ = nullptr;
  DecoderInterface* decoder_ = nullptr;
};

// Runs the program from beginning to end using HTIF semihosting.
TEST_F(RiscVTopTest, RunProgramHtif) {
  LoadFile(kHtifFileName);
  HtifSemihostSetup htif_semihost(riscv_top_, loader_, memory_);
  testing::internal::CaptureStdout();
  EXPECT_OK(riscv_top_->WriteRegister("pc", entry_point_));
  // Initialize stack pointer.
  EXPECT_OK(riscv_top_->WriteRegister("sp", 0x200000));
  EXPECT_OK(riscv_top_->Run());
  EXPECT_OK(riscv_top_->Wait());
  auto halt_result = riscv_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSemihostHaltRequest));
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Arch: RV32\nHello World!\n", stdout_str);
}

// Runs the program from beginning to end using ARM semihosting.
TEST_F(RiscVTopTest, RunProgramArm) {
  LoadFile(kArmFileName);
  ArmSemihostSetup arm_semihost(riscv_top_, memory_);
  testing::internal::CaptureStdout();
  EXPECT_OK(riscv_top_->WriteRegister("pc", entry_point_));
  // Initialize stack pointer.
  EXPECT_OK(riscv_top_->WriteRegister("sp", 0x200000));
  EXPECT_OK(riscv_top_->Run());
  EXPECT_OK(riscv_top_->Wait());
  auto halt_result = riscv_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSemihostHaltRequest));
  const std::string stdout_str = testing::internal::GetCapturedStdout();
  EXPECT_EQ("Hello World! 5\n", stdout_str);
}

// Steps through the program from beginning to end.
TEST_F(RiscVTopTest, StepProgramHtif) {
  LoadFile(kHtifFileName);
  HtifSemihostSetup htif_semihost(riscv_top_, loader_, memory_);
  testing::internal::CaptureStdout();
  EXPECT_OK(riscv_top_->WriteRegister("pc", entry_point_));
  // Initialize stack pointer.
  EXPECT_OK(riscv_top_->WriteRegister("sp", 0x200000));

  auto res = riscv_top_->Step(10000);
  EXPECT_OK(res.status());
  auto halt_result = riscv_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSemihostHaltRequest));

  EXPECT_EQ("Arch: RV32\nHello World!\n",
            testing::internal::GetCapturedStdout());
}

// Steps through the program from beginning to end.
TEST_F(RiscVTopTest, StepProgramArm) {
  LoadFile(kArmFileName);
  ArmSemihostSetup arm_semihost(riscv_top_, memory_);
  testing::internal::CaptureStdout();
  EXPECT_OK(riscv_top_->WriteRegister("pc", entry_point_));
  // Initialize stack pointer.
  EXPECT_OK(riscv_top_->WriteRegister("sp", 0x200000));

  auto res = riscv_top_->Step(10000);
  EXPECT_OK(res.status());
  auto halt_result = riscv_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSemihostHaltRequest));

  EXPECT_EQ("Hello World! 5\n", testing::internal::GetCapturedStdout());
}

// Sets/Clears breakpoints without executing the program.
TEST_F(RiscVTopTest, SetAndClearBreakpoint) {
  LoadFile(kHtifFileName);
  auto result = loader_->GetSymbol("printf");
  EXPECT_OK(result);
  auto address = result.value().first;
  EXPECT_EQ(riscv_top_->ClearSwBreakpoint(address).code(),
            absl::StatusCode::kNotFound);
  EXPECT_OK(riscv_top_->SetSwBreakpoint(address));
  EXPECT_EQ(riscv_top_->SetSwBreakpoint(address).code(),
            absl::StatusCode::kAlreadyExists);
  EXPECT_OK(riscv_top_->ClearSwBreakpoint(address));
  EXPECT_EQ(riscv_top_->ClearSwBreakpoint(address).code(),
            absl::StatusCode::kNotFound);
  EXPECT_OK(riscv_top_->SetSwBreakpoint(address));
  EXPECT_OK(riscv_top_->ClearAllSwBreakpoints());
  EXPECT_EQ(riscv_top_->ClearSwBreakpoint(address).code(),
            absl::StatusCode::kNotFound);
}

// Runs program with breakpint at printf with htif semihosting.
TEST_F(RiscVTopTest, RunWithBreakpointHtif) {
  LoadFile(kHtifFileName);
  HtifSemihostSetup htif_semihost(riscv_top_, loader_, memory_);
  // Set breakpoint at printf.
  auto result = loader_->GetSymbol("printf");
  EXPECT_OK(result);
  auto address = result.value().first;
  EXPECT_OK(riscv_top_->SetSwBreakpoint(address));
  EXPECT_OK(riscv_top_->WriteRegister("pc", entry_point_));
  // Initialize stack pointer.
  EXPECT_OK(riscv_top_->WriteRegister("sp", 0x200000));

  // Run to printf. Capture stdout
  testing::internal::CaptureStdout();
  EXPECT_OK(riscv_top_->Run());
  EXPECT_OK(riscv_top_->Wait());
  // Should be stopped at breakpoint, but nothing printed.
  auto halt_result = riscv_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSoftwareBreakpoint));
  EXPECT_EQ(testing::internal::GetCapturedStdout().size(), 0);

  // Run to printf. Capture stdout.
  testing::internal::CaptureStdout();
  EXPECT_OK(riscv_top_->Run());
  EXPECT_OK(riscv_top_->Wait());
  // Should be stopped at breakpoint. Captured 'Arch: RV32\n'.
  halt_result = riscv_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSoftwareBreakpoint));
  EXPECT_STREQ("Arch: RV32\n", testing::internal::GetCapturedStdout().c_str());

  // Run to the end of the program, and capture stdout.
  testing::internal::CaptureStdout();
  EXPECT_OK(riscv_top_->Run());
  EXPECT_OK(riscv_top_->Wait());
  // Should be stopped due to semihost halt request. Captured 'Hello World!
  // 5\n'.
  halt_result = riscv_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSemihostHaltRequest));
  EXPECT_EQ("Hello World!\n", testing::internal::GetCapturedStdout());
}

// Runs program with breakpint at printf with arm semihosting.
TEST_F(RiscVTopTest, RunWithBreakpointArm) {
  LoadFile(kArmFileName);
  ArmSemihostSetup arm_semihost(riscv_top_, memory_);
  // Set breakpoint at printf.
  auto result = loader_->GetSymbol("printf");
  EXPECT_OK(result);
  auto address = result.value().first;
  EXPECT_OK(riscv_top_->SetSwBreakpoint(address));
  EXPECT_OK(riscv_top_->WriteRegister("pc", entry_point_));
  // Initialize stack pointer.
  EXPECT_OK(riscv_top_->WriteRegister("sp", 0x200000));

  // Run to printf. Capture stdout.
  testing::internal::CaptureStdout();
  EXPECT_OK(riscv_top_->Run());
  EXPECT_OK(riscv_top_->Wait());

  // Should be stopped at breakpoint, but nothing printed.
  auto halt_result = riscv_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSoftwareBreakpoint));
  EXPECT_EQ(testing::internal::GetCapturedStdout().size(), 0);

  // Run to the end of the program. Capture stdout.
  testing::internal::CaptureStdout();
  EXPECT_OK(riscv_top_->Run());
  EXPECT_OK(riscv_top_->Wait());

  // Should be stopped due to semihost halt request. Captured 'Hello World!
  // 5\n'.
  halt_result = riscv_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSemihostHaltRequest));
  EXPECT_EQ("Hello World! 5\n", testing::internal::GetCapturedStdout());
}

// Memory read/write test.
TEST_F(RiscVTopTest, Memory) {
  uint8_t byte_data = 0xab;
  uint16_t half_data = 0xabcd;
  uint32_t word_data = 0xba5eba11;
  uint64_t dword_data = 0x5ca1ab1e'0ddball;
  EXPECT_OK(riscv_top_->WriteMemory(0x1000, &byte_data, sizeof(byte_data)));
  EXPECT_OK(riscv_top_->WriteMemory(0x1004, &half_data, sizeof(half_data)));
  EXPECT_OK(riscv_top_->WriteMemory(0x1008, &word_data, sizeof(word_data)));
  EXPECT_OK(riscv_top_->WriteMemory(0x1010, &dword_data, sizeof(dword_data)));

  uint8_t byte_value;
  uint16_t half_value;
  uint32_t word_value;
  uint64_t dword_value;

  EXPECT_OK(riscv_top_->ReadMemory(0x1000, &byte_value, sizeof(byte_value)));
  EXPECT_OK(riscv_top_->ReadMemory(0x1004, &half_value, sizeof(half_value)));
  EXPECT_OK(riscv_top_->ReadMemory(0x1008, &word_value, sizeof(word_value)));
  EXPECT_OK(riscv_top_->ReadMemory(0x1010, &dword_value, sizeof(dword_value)));

  EXPECT_EQ(byte_data, byte_value);
  EXPECT_EQ(half_data, half_value);
  EXPECT_EQ(word_data, word_value);
  EXPECT_EQ(dword_data, dword_value);

  EXPECT_OK(riscv_top_->ReadMemory(0x1000, &byte_value, sizeof(byte_value)));
  EXPECT_OK(riscv_top_->ReadMemory(0x1000, &half_value, sizeof(half_value)));
  EXPECT_OK(riscv_top_->ReadMemory(0x1000, &word_value, sizeof(word_value)));
  EXPECT_OK(riscv_top_->ReadMemory(0x1000, &dword_value, sizeof(dword_value)));

  EXPECT_EQ(byte_data, byte_value);
  EXPECT_EQ(byte_data, half_value);
  EXPECT_EQ(byte_data, word_value);
  EXPECT_EQ(0x0000'abcd'0000'00ab, dword_value);
}

// Register name test.
TEST_F(RiscVTopTest, RegisterNames) {
  // Test x-names and numbers.
  uint32_t word_value;
  for (int i = 0; i < 32; i++) {
    std::string name = absl::StrCat("x", i);
    auto result = riscv_top_->ReadRegister(name);
    EXPECT_OK(result.status());
    word_value = result.value();
    EXPECT_OK(riscv_top_->WriteRegister(name, word_value));
  }
  // Test d-names and numbers.
  uint64_t dword_value;
  for (int i = 0; i < 32; i++) {
    std::string name = absl::StrCat("f", i);
    auto result = riscv_top_->ReadRegister(name);
    EXPECT_OK(result.status());
    dword_value = result.value();
    EXPECT_OK(riscv_top_->WriteRegister(name, dword_value));
  }
  // Not found.
  EXPECT_EQ(riscv_top_->ReadRegister("x32").status().code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ(riscv_top_->WriteRegister("x32", word_value).code(),
            absl::StatusCode::kNotFound);
  // Aliases.
  for (auto& [name, alias] : {std::tuple<std::string, std::string>{"x1", "ra"},
                              {"x4", "tp"},
                              {"x8", "s0"}}) {
    uint32_t write_value = 0xba5eba11;
    EXPECT_OK(riscv_top_->WriteRegister(name, write_value));
    uint32_t read_value;
    auto result = riscv_top_->ReadRegister(alias);
    EXPECT_OK(result.status());
    read_value = result.value();
    EXPECT_EQ(read_value, write_value);
  }
}

// Directly read/write the memory addresses that are out-of-bound.
TEST_F(RiscVTopTest, ReadWriteOutOfBoundMemory) {
  // Set the machine to have 16-byte memory
  constexpr uint64_t kTestMemerySize = 0x10;
  constexpr uint64_t kMaxPhysicalAddress = kTestMemerySize - 1;
  constexpr uint64_t kBinaryAddress = 0;
  riscv_top_->state()->set_max_physical_address(kMaxPhysicalAddress);
  uint8_t mem_bytes[kTestMemerySize + 4] = {0};
  // Read the memory with the length greater than the physical memory size. The
  // read operation is successful within the physical memory size range.
  auto result =
      riscv_top_->ReadMemory(kBinaryAddress, mem_bytes, sizeof(mem_bytes));
  EXPECT_OK(result);
  EXPECT_EQ(result.value(), kTestMemerySize);
  // Read at the maximum physical address, so only one byte can be read.
  result =
      riscv_top_->ReadMemory(kMaxPhysicalAddress, mem_bytes, sizeof(mem_bytes));
  EXPECT_OK(result);
  EXPECT_EQ(result.value(), 1);
  // Read the memory with the staring address out of the physical memory range.
  // The read operation returns error.
  result =
      riscv_top_->ReadMemory(kTestMemerySize + 4, mem_bytes, sizeof(mem_bytes));
  EXPECT_FALSE(result.ok());

  // Write the memory with the length greater than the physical memory size. The
  // write operation is successful within the physical memory size range.
  result =
      riscv_top_->WriteMemory(kBinaryAddress, mem_bytes, sizeof(mem_bytes));
  EXPECT_OK(result);
  EXPECT_EQ(result.value(), kTestMemerySize);
  // Write at the maximum physical address, so only one byte can be written.
  result = riscv_top_->WriteMemory(kMaxPhysicalAddress, mem_bytes,
                                   sizeof(mem_bytes));
  EXPECT_OK(result);
  EXPECT_EQ(result.value(), 1);

  // Write the memory with the staring address out of the physical memory range.
  // The write operation returns error.
  result = riscv_top_->WriteMemory(kTestMemerySize + 4, mem_bytes,
                                   sizeof(mem_bytes));
  EXPECT_FALSE(result.ok());
}

// This test will verify that the 64 bit version executes a program properly.
// No need to test other aspects of the top.
TEST_F(RiscVTopTest, RiscV64) {
  delete riscv_top_;
  delete loader_;
  delete decoder_;
  delete state_;

  // New top with 64 bit registers.
  state_ = new RiscVState("RV64", RiscVXlen::RV64, memory_);
  state_->set_rv_fp(fp_state_);
  decoder_ = new RiscV64Decoder(state_, memory_);

  // Make sure the architectural and abi register aliases are added.
  std::string reg_name;
  for (int i = 0; i < 32; i++) {
    reg_name = absl::StrCat(RiscVState::kXregPrefix, i);
    (void)state_->AddRegister<RV64Register>(reg_name);
    (void)state_->AddRegisterAlias<RV64Register>(
        reg_name, ::mpact::sim::riscv::kXRegisterAliases[i]);
  }
  for (int i = 0; i < 32; i++) {
    reg_name = absl::StrCat(RiscVState::kFregPrefix, i);
    (void)state_->AddRegister<RVFpRegister>(reg_name);
    (void)state_->AddRegisterAlias<RVFpRegister>(
        reg_name, ::mpact::sim::riscv::kFRegisterAliases[i]);
  }

  riscv_top_ = new RiscVTop("RV64", state_, decoder_);
  // Set up the elf loader.
  loader_ = new mpact::sim::util::ElfProgramLoader(memory_);

  LoadFile(kArm64FileName);
  ArmSemihostSetup arm_semihost(riscv_top_, memory_);
  // Set breakpoint at printf.
  auto result = loader_->GetSymbol("printf");
  EXPECT_OK(result);
  auto address = result.value().first;
  EXPECT_OK(riscv_top_->SetSwBreakpoint(address));
  EXPECT_OK(riscv_top_->WriteRegister("pc", entry_point_));
  // Initialize stack pointer.
  EXPECT_OK(riscv_top_->WriteRegister("sp", 0x200000));

  // Run to printf.
  testing::internal::CaptureStdout();
  EXPECT_OK(riscv_top_->Run());
  EXPECT_OK(riscv_top_->Wait());

  // Should be stopped at breakpoint, but nothing printed.
  auto halt_result = riscv_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSoftwareBreakpoint));
  EXPECT_EQ(testing::internal::GetCapturedStdout().size(), 0);

  // Run to the end of the program.
  testing::internal::CaptureStdout();
  EXPECT_OK(riscv_top_->Run());
  EXPECT_OK(riscv_top_->Wait());

  // Should be stopped due to semihost halt request. Captured 'Hello World!
  // 5\n'.
  halt_result = riscv_top_->GetLastHaltReason();
  CHECK_OK(halt_result);
  EXPECT_EQ(static_cast<int>(halt_result.value()),
            static_cast<int>(HaltReason::kSemihostHaltRequest));
  EXPECT_EQ("Hello world! 5\n", testing::internal::GetCapturedStdout());
}

}  // namespace
