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

#include "riscv/riscv_arm_semihost.h"

#include <time.h>
#include <unistd.h>

#include <cstring>
#include <string>

#include "googlemock/include/gmock/gmock.h"
#include "absl/log/check.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/memory/memory_interface.h"

// This file implements the tests for the class that implements the ARM style
// semihosting for the RiscV architecture. The test is done for both 32 bit and
// 64 bit versions of the architecture.

namespace {

// Test input file to use for some of the file operations.
constexpr char kFileName[] = "test_file.txt";

// The depot path to the test directory.
constexpr char kDepotPath[] = "riscv/test/";

using ::mpact::sim::generic::DataBufferFactory;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::RiscVArmSemihost;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RV64Register;

constexpr char kHelloWorld[] = "Hello World\n";

constexpr uint64_t kSemihostCallAddress = 0x1004;
constexpr uint64_t kSemihostNonCallAddress = 0x2004;
constexpr uint64_t kParameterAddress = 0x3000;
constexpr uint64_t kStringAddress = 0x4000;
constexpr uint64_t kBufferAddress = 0x5000;

constexpr int kTmpNamLength = L_tmpnam * 3 / 2;

constexpr int kBufferSize = 128;

// Opcode sequence that idenfifies the semihosting call. This consists
// of a shift immediate on the x0 (constant 0 register), followed by
// an ebreak instruction, followed by a different shift immediate on x0.
constexpr uint32_t kSemihostCallSequence[] = {0x01f01013, 0x00100073,
                                              0x40705013};

// A sequence that does not match the semihosting call.
constexpr uint32_t kSemihostNonCallSequence[] = {0x0e01013, 0x00100073,
                                                 0x40605013};

// Exception code;
constexpr int kAdpStoppedApplicationExit = 0x20026;
// Semihosting function call codes.
constexpr int kSysException = 0x18;
constexpr int kSysTime = 0x11;
constexpr int kSysWrite = 0x05;
constexpr int kSysTickFreq = 0x31;
constexpr int kSysHeapInfo = 0x16;
constexpr int kSysClose = 0x02;
constexpr int kSysFlen = 0x0c;
constexpr int kSysOpen = 0x01;
constexpr int kSysSeek = 0x0a;
constexpr int kSysRead = 0x06;
constexpr int kSysTmpnam = 0x0d;

// Helper template to get register type from integer type.
template <typename T>
struct RegisterType {};

template <>
struct RegisterType<uint32_t> {
  using type = RV32Register;
};

template <>
struct RegisterType<uint64_t> {
  using type = RV64Register;
};

// Test fixture for RiscV Arm Semihosting.
class RiscVArmSemihostTest : public testing::Test {
 public:
  void SetTrue(bool *value) { *value = true; }
  RiscVArmSemihost *semi32() const { return semi32_; }
  RiscVArmSemihost *semi64() const { return semi64_; }
  RV32Register *a0_32() const { return a0_32_; }
  RV32Register *a1_32() const { return a1_32_; }
  RV64Register *a0_64() const { return a0_64_; }
  RV64Register *a1_64() const { return a1_64_; }
  Instruction *semihost_inst_32() const { return semihost_inst_32_; }
  Instruction *semihost_inst_64() const { return semihost_inst_64_; }
  Instruction *non_semihost_inst_32() const { return non_semihost_inst_32_; }
  Instruction *non_semihost_inst_64() const { return non_semihost_inst_64_; }
  DataBufferFactory *db_factory() { return &db_factory_; }
  mpact::sim::util::MemoryInterface *memory() const { return memory_; }

 protected:
  RiscVArmSemihostTest() {
    memory_ = new mpact::sim::util::FlatDemandMemory(0);
    state32_ = new RiscVState("test", RiscVXlen::RV32, memory_);
    state64_ = new RiscVState("test", RiscVXlen::RV64, memory_);

    // Store the semihost call sequence to memory.
    auto db = db_factory_.Allocate<uint32_t>(3);
    for (int i = 0; i < 3; i++) {
      db->Set<uint32_t>(i, kSemihostCallSequence[i]);
    }
    memory_->Store(kSemihostCallAddress - 4, db);

    // Store the non-semihost call sequence to memory.
    for (int i = 0; i < 3; i++) {
      db->Set<uint32_t>(i, kSemihostNonCallSequence[i]);
    }
    memory_->Store(kSemihostNonCallAddress - 4, db);
    db->DecRef();

    // Create instruction objects for semihost and non-semihost for 32 and 64
    // bit architectures.
    semihost_inst_32_ = new Instruction(kSemihostCallAddress, state32_);
    semihost_inst_64_ = new Instruction(kSemihostCallAddress, state64_);
    non_semihost_inst_32_ = new Instruction(kSemihostNonCallAddress, state32_);
    non_semihost_inst_64_ = new Instruction(kSemihostNonCallAddress, state64_);

    // Create registers used in the calls. Just use the alias names as opposed
    // to the x register names.
    a0_32_ = state32_->GetRegister<RV32Register>("x10").first;
    a1_32_ = state32_->GetRegister<RV32Register>("x11").first;
    a0_64_ = state64_->GetRegister<RV64Register>("x10").first;
    a1_64_ = state64_->GetRegister<RV64Register>("x11").first;

    // Create the semihosting class instances.
    semi32_ = new RiscVArmSemihost(RiscVArmSemihost::BitWidth::kWord32, memory_,
                                   memory_);
    semi64_ = new RiscVArmSemihost(RiscVArmSemihost::BitWidth::kWord64, memory_,
                                   memory_);
  }

  ~RiscVArmSemihostTest() override {
    delete memory_;
    delete state32_;
    delete state64_;
    semihost_inst_32_->DecRef();
    semihost_inst_64_->DecRef();
    non_semihost_inst_32_->DecRef();
    non_semihost_inst_64_->DecRef();
    delete semi32_;
    delete semi64_;
  }

  DataBufferFactory db_factory_;
  mpact::sim::util::FlatDemandMemory *memory_;
  RiscVState *state32_;
  RiscVState *state64_;
  Instruction *semihost_inst_32_;
  Instruction *semihost_inst_64_;
  Instruction *non_semihost_inst_32_;
  Instruction *non_semihost_inst_64_;
  RiscVArmSemihost *semi32_;
  RiscVArmSemihost *semi64_;
  RV32Register *a0_32_, *a1_32_;
  RV64Register *a0_64_, *a1_64_;
};

// Templated helper functions to select members from the test fixture by type.

// Semihosting instance.
template <typename T>
RiscVArmSemihost *SemiHost(RiscVArmSemihostTest *test) {}

template <>
RiscVArmSemihost *SemiHost<uint32_t>(RiscVArmSemihostTest *test) {
  return test->semi32();
}

template <>
RiscVArmSemihost *SemiHost<uint64_t>(RiscVArmSemihostTest *test) {
  return test->semi64();
}

// Register.
template <typename T>
typename RegisterType<T>::type *AReg(RiscVArmSemihostTest *test, T num) {}

template <>
typename RegisterType<uint32_t>::type *AReg(RiscVArmSemihostTest *test,
                                            uint32_t num) {
  if (num == 0) return test->a0_32();
  if (num == 1) return test->a1_32();
  return nullptr;
}

template <>
typename RegisterType<uint64_t>::type *AReg(RiscVArmSemihostTest *test,
                                            uint64_t num) {
  if (num == 0) return test->a0_64();
  if (num == 1) return test->a1_64();
  return nullptr;
}

// Semihosting instruction instance.
template <typename T>
Instruction *SemihostInst(RiscVArmSemihostTest *test, T num = 0) {}

template <>
Instruction *SemihostInst<uint32_t>(RiscVArmSemihostTest *test, uint32_t) {
  return test->semihost_inst_32();
}

template <>
Instruction *SemihostInst<uint64_t>(RiscVArmSemihostTest *test, uint64_t) {
  return test->semihost_inst_64();
}

// Non-semihosting instruction instance.
template <typename T>
Instruction *NonSemihostInst(RiscVArmSemihostTest *test, T num = 0) {}

template <>
Instruction *NonSemihostInst<uint32_t>(RiscVArmSemihostTest *test, uint32_t) {
  return test->non_semihost_inst_32();
}
template <>
Instruction *NonSemihostInst<uint64_t>(RiscVArmSemihostTest *test, uint64_t) {
  return test->non_semihost_inst_64();
}

// Verify that the semihosting call is recognized properly by using the time
// call.

template <typename T>
void CallRecognitionTest(RiscVArmSemihostTest *test) {
  AReg<T>(test, 0)->data_buffer()->template Set<T>(0, kSysTime);
  AReg<T>(test, 1)->data_buffer()->template Set<T>(0, 0);
  SemiHost<T>(test)->OnEBreak(SemihostInst<T>(test));
  EXPECT_NEAR(AReg<T>(test, 0)->data_buffer()->template Get<T>(0),
              time(nullptr), 2);
  EXPECT_TRUE(SemiHost<T>(test)->IsSemihostingCall(SemihostInst<T>(test)));
}

TEST_F(RiscVArmSemihostTest, CallRecognition32) {
  CallRecognitionTest<uint32_t>(this);
}

TEST_F(RiscVArmSemihostTest, CallRecognition64) {
  CallRecognitionTest<uint64_t>(this);
}

// Verify that a sequence that doesn't match doesn't trigger semihosting call.

template <typename T>
void CallNonRecognitionTest(RiscVArmSemihostTest *test) {
  AReg<T>(test, 0)->data_buffer()->template Set<T>(0, kSysTime);
  AReg<T>(test, 1)->data_buffer()->template Set<T>(0, 0);
  SemiHost<T>(test)->OnEBreak(NonSemihostInst<T>(test));
  EXPECT_EQ(AReg<T>(test, 0)->data_buffer()->template Get<uint32_t>(0),
            kSysTime);
  EXPECT_FALSE(SemiHost<T>(test)->IsSemihostingCall(NonSemihostInst<T>(test)));
}

TEST_F(RiscVArmSemihostTest, CallNonRecognition32) {
  CallNonRecognitionTest<uint32_t>(this);
}

TEST_F(RiscVArmSemihostTest, CallNonRecognition64) {
  CallNonRecognitionTest<uint64_t>(this);
}

// RiscV32 puts appliation exit code into A1.
TEST_F(RiscVArmSemihostTest, ApplicationExit32) {
  bool detected = false;
  SemiHost<uint32_t>(this)->set_exit_callback(
      [&detected]() { detected = true; });

  // Set up register values.
  AReg<uint32_t>(this, 0)->data_buffer()->template Set<uint32_t>(0,
                                                                 kSysException);
  AReg<uint32_t>(this, 1)->data_buffer()->template Set<uint32_t>(
      0, kAdpStoppedApplicationExit);

  // Make the callback.
  SemiHost<uint32_t>(this)->OnEBreak(SemihostInst<uint32_t>(this));

  EXPECT_TRUE(detected);
}

// RiscV64 puts application exit code into memory, with the pointer in A1.
TEST_F(RiscVArmSemihostTest, ApplicationExit64) {
  bool detected = false;
  SemiHost<uint64_t>(this)->set_exit_callback(
      [&detected]() { detected = true; });

  // Write exception code to memory.
  auto *db = db_factory()->Allocate<uint64_t>(1);
  db->template Set<uint64_t>(0, kAdpStoppedApplicationExit);
  memory_->Store(kParameterAddress, db);
  db->DecRef();
  // Set up parameter block.
  AReg<uint64_t>(this, 0)->data_buffer()->template Set<uint64_t>(0,
                                                                 kSysException);
  AReg<uint64_t>(this, 1)->data_buffer()->template Set<uint64_t>(
      0, kParameterAddress);

  // Make the callback.
  SemiHost<uint64_t>(this)->OnEBreak(SemihostInst<uint64_t>(this));

  EXPECT_TRUE(detected);
}

// Test the Write system call (printf to stderr).
template <typename T>
void SysWriteTest(RiscVArmSemihostTest *test) {
  testing::internal::CaptureStderr();

  // Write string to memory.
  auto *str_db = test->db_factory()->Allocate<uint8_t>(strlen(kHelloWorld));
  std::memcpy(str_db->raw_ptr(), kHelloWorld, strlen(kHelloWorld));
  test->memory()->Store(kStringAddress, str_db);
  str_db->DecRef();

  // Set up parameter block.
  auto *db = test->db_factory()->Allocate<T>(3);
  db->template Set<T>(0, 2);
  db->template Set<T>(1, kStringAddress);
  db->template Set<T>(2, strlen(kHelloWorld));
  test->memory()->Store(kParameterAddress, db);
  db->DecRef();

  // Set up register values.
  AReg<T>(test, 0)->data_buffer()->template Set<T>(0, kSysWrite);
  AReg<T>(test, 1)->data_buffer()->template Set<T>(0, kParameterAddress);

  // Make the callback.
  SemiHost<T>(test)->OnEBreak(SemihostInst<T>(test));

  std::string capture_stderr = testing::internal::GetCapturedStderr();

  EXPECT_STREQ(kHelloWorld, capture_stderr.c_str());
}

TEST_F(RiscVArmSemihostTest, SysWrite32) { SysWriteTest<uint32_t>(this); }

TEST_F(RiscVArmSemihostTest, SysWrite64) { SysWriteTest<uint64_t>(this); }

// Systick should just return -1 for now.

template <typename T>
void SysTickTest(RiscVArmSemihostTest *test) {
  AReg<T>(test, 0)->data_buffer()->template Set<T>(0, kSysTickFreq);
  AReg<T>(test, 1)->data_buffer()->template Set<T>(0, 0);
  SemiHost<T>(test)->OnEBreak(SemihostInst<T>(test));
  EXPECT_EQ(AReg<T>(test, 0)
                ->data_buffer()
                ->template Get<typename std::make_signed<T>::type>(0),
            -1);
}

TEST_F(RiscVArmSemihostTest, SysTick32) { SysTickTest<uint32_t>(this); }

TEST_F(RiscVArmSemihostTest, SysTick64) { SysTickTest<uint64_t>(this); }

// Test SysOpen

template <typename T>
void SysOpenTest(RiscVArmSemihostTest *test) {
  std::string input_file = absl::StrCat(kDepotPath, "testfiles/", kFileName);
  auto *str_db = test->db_factory()->Allocate<uint8_t>(input_file.length() + 1);
  std::memcpy(str_db->raw_ptr(), input_file.c_str(), input_file.length() + 1);
  test->memory()->Store(kStringAddress, str_db);
  str_db->DecRef();

  // Set up parameter block.
  auto *db = test->db_factory()->Allocate<T>(3);
  db->template Set<T>(0, kStringAddress);
  db->template Set<T>(1, 0);
  db->template Set<T>(2, input_file.length());
  test->memory()->Store(kParameterAddress, db);
  db->DecRef();

  // Set up register values.
  AReg<T>(test, 0)->data_buffer()->template Set<T>(0, kSysOpen);
  AReg<T>(test, 1)->data_buffer()->template Set<T>(0, kParameterAddress);

  // Make the callback.
  SemiHost<T>(test)->OnEBreak(SemihostInst<T>(test));

  EXPECT_GT(AReg<T>(test, 0)
                ->data_buffer()
                ->template Get<typename std::make_signed<T>::type>(0),
            0);

  // Remove a character from the end of the file name and try again. This time
  // it should fail.
  input_file = input_file.substr(0, input_file.length() - 1);
  str_db = test->db_factory()->Allocate<uint8_t>(input_file.length() + 1);
  std::memcpy(str_db->raw_ptr(), input_file.c_str(), input_file.length() + 1);
  test->memory()->Store(kStringAddress, str_db);
  str_db->DecRef();

  // Set up parameter block.
  db = test->db_factory()->Allocate<T>(3);
  db->template Set<T>(0, kStringAddress);
  db->template Set<T>(1, 0);
  db->template Set<T>(2, input_file.length());
  test->memory()->Store(kParameterAddress, db);
  db->DecRef();

  // Set up register values.
  AReg<T>(test, 0)->data_buffer()->template Set<T>(0, kSysOpen);
  AReg<T>(test, 1)->data_buffer()->template Set<T>(0, kParameterAddress);

  // Make the callback.
  SemiHost<T>(test)->OnEBreak(SemihostInst<T>(test));

  EXPECT_LT(AReg<T>(test, 0)
                ->data_buffer()
                ->template Get<typename std::make_signed<T>::type>(0),
            0);
}

TEST_F(RiscVArmSemihostTest, SysOpen32) { SysOpenTest<uint32_t>(this); }

TEST_F(RiscVArmSemihostTest, SysOpen64) { SysOpenTest<uint64_t>(this); }

// Test sys open and close.
template <typename T>
void SysOpenCloseTest(RiscVArmSemihostTest *test) {
  std::string input_file = absl::StrCat(kDepotPath, "testfiles/", kFileName);
  auto *str_db = test->db_factory()->Allocate<uint8_t>(input_file.length() + 1);
  std::memcpy(str_db->raw_ptr(), input_file.c_str(), input_file.length() + 1);
  test->memory()->Store(kStringAddress, str_db);
  str_db->DecRef();

  // Set up parameter block.
  auto *db = test->db_factory()->Allocate<T>(3);
  db->template Set<T>(0, kStringAddress);
  db->template Set<T>(1, 0);
  db->template Set<T>(2, input_file.length());
  test->memory()->Store(kParameterAddress, db);
  db->DecRef();

  // Set up register values.
  AReg<T>(test, 0)->data_buffer()->template Set<T>(0, kSysOpen);
  AReg<T>(test, 1)->data_buffer()->template Set<T>(0, kParameterAddress);

  // Make the callback.
  SemiHost<T>(test)->OnEBreak(SemihostInst<T>(test));

  int fd = AReg<T>(test, 0)->data_buffer()->template Get<int32_t>(0);
  CHECK_GT(fd, 0);

  // Now close that file descriptor.
  db = test->db_factory()->Allocate<T>(1);
  db->template Set<T>(0, fd);
  test->memory()->Store(kParameterAddress, db);
  db->DecRef();

  // Set up register values.
  AReg<T>(test, 0)->data_buffer()->template Set<T>(0, kSysClose);
  AReg<T>(test, 1)->data_buffer()->template Set<T>(0, kParameterAddress);

  SemiHost<T>(test)->OnEBreak(SemihostInst<T>(test));

  EXPECT_EQ(AReg<T>(test, 0)
                ->data_buffer()
                ->template Get<typename std::make_signed<T>::type>(0),
            0);

  // Try closing again this should be an error.
  AReg<T>(test, 0)->data_buffer()->template Set<T>(0, kSysClose);
  AReg<T>(test, 1)->data_buffer()->template Set<T>(0, kParameterAddress);

  SemiHost<T>(test)->OnEBreak(SemihostInst<T>(test));

  EXPECT_EQ(AReg<T>(test, 0)
                ->data_buffer()
                ->template Get<typename std::make_signed<T>::type>(0),
            -1);
}

// Test SysOpen and SysClose
TEST_F(RiscVArmSemihostTest, SysOpenClose32) {
  SysOpenCloseTest<uint32_t>(this);
}

TEST_F(RiscVArmSemihostTest, SysOpenClose64) {
  SysOpenCloseTest<uint64_t>(this);
}

// Test open-read-close.

template <typename T>
void SysOpenReadClose(RiscVArmSemihostTest *test) {
  // Open the file.
  std::string input_file = absl::StrCat(kDepotPath, "testfiles/", kFileName);
  auto *str_db = test->db_factory()->Allocate<uint8_t>(input_file.length() + 1);
  std::memcpy(str_db->raw_ptr(), input_file.c_str(), input_file.length() + 1);
  test->memory()->Store(kStringAddress, str_db);
  str_db->DecRef();

  // Set up parameter block.
  auto *db = test->db_factory()->Allocate<T>(3);
  db->template Set<T>(0, kStringAddress);
  db->template Set<T>(1, 0);
  db->template Set<T>(2, input_file.length());
  test->memory()->Store(kParameterAddress, db);
  db->DecRef();

  // Set up register values.
  AReg<T>(test, 0)->data_buffer()->template Set<T>(0, kSysOpen);
  AReg<T>(test, 1)->data_buffer()->template Set<T>(0, kParameterAddress);

  // Make the callback.
  SemiHost<T>(test)->OnEBreak(SemihostInst<T>(test));

  int fd = AReg<T>(test, 0)->data_buffer()->template Get<int32_t>(0);
  CHECK_GT(fd, 0) << "Invalid file descriptor: " << fd;

  // Read from the file.
  db = test->db_factory()->Allocate<T>(3);
  db->template Set<T>(0, fd);
  db->template Set<T>(1, kBufferAddress);
  db->template Set<T>(2, kBufferSize);
  test->memory()->Store(kParameterAddress, db);
  db->DecRef();

  AReg<T>(test, 0)->data_buffer()->template Set<T>(0, kSysRead);
  AReg<T>(test, 1)->data_buffer()->template Set<T>(0, kParameterAddress);

  // Make the callback.
  SemiHost<T>(test)->OnEBreak(SemihostInst<T>(test));

  int read_ret = AReg<T>(test, 0)->data_buffer()->template Get<int32_t>(0);

  db = test->db_factory()->Allocate<uint8_t>(6);
  test->memory()->Load(kBufferAddress, db, nullptr, nullptr);
  std::string file_value(reinterpret_cast<char *>(db->raw_ptr()), 6);
  db->DecRef();

  // Verify the number of characters read and that the characters are correct.
  EXPECT_STREQ("abcdef", file_value.c_str());
  EXPECT_EQ(read_ret, 7);

  // Now close that file descriptor.
  db = test->db_factory()->Allocate<T>(1);
  db->template Set<T>(0, fd);
  test->memory()->Store(kParameterAddress, db);
  db->DecRef();

  // Set up register values.
  AReg<T>(test, 0)->data_buffer()->template Set<T>(0, kSysClose);
  AReg<T>(test, 1)->data_buffer()->template Set<T>(0, kParameterAddress);

  SemiHost<T>(test)->OnEBreak(SemihostInst<T>(test));

  CHECK_EQ(AReg<T>(test, 0)->data_buffer()->template Get<int32_t>(0), 0)
      << "Failed to close file";
}

TEST_F(RiscVArmSemihostTest, SysOpenReadClose32) {
  SysOpenReadClose<uint32_t>(this);
}

TEST_F(RiscVArmSemihostTest, SysOpenReadClose64) {
  SysOpenReadClose<uint64_t>(this);
}

// Test flen (file length).

template <typename T>
void SysFlenTest(RiscVArmSemihostTest *test) {  // Open the file.
  std::string input_file = absl::StrCat(kDepotPath, "testfiles/", kFileName);
  auto *str_db = test->db_factory()->Allocate<uint8_t>(input_file.length() + 1);
  std::memcpy(str_db->raw_ptr(), input_file.c_str(), input_file.length() + 1);
  test->memory()->Store(kStringAddress, str_db);
  str_db->DecRef();

  // Set up parameter block.
  auto *db = test->db_factory()->Allocate<T>(3);
  db->template Set<T>(0, kStringAddress);
  db->template Set<T>(1, 0);
  db->template Set<T>(2, input_file.length());
  test->memory()->Store(kParameterAddress, db);
  db->DecRef();

  // Set up register values.
  AReg<T>(test, 0)->data_buffer()->template Set<T>(0, kSysOpen);
  AReg<T>(test, 1)->data_buffer()->template Set<T>(0, kParameterAddress);

  // Make the callback.
  SemiHost<T>(test)->OnEBreak(SemihostInst<T>(test));

  int fd = AReg<T>(test, 0)->data_buffer()->template Get<int32_t>(0);
  CHECK_GT(fd, 0) << "Invalid file descriptor: " << fd;

  // Get the length of the file (7).
  db = test->db_factory()->Allocate<T>(1);
  db->template Set<T>(0, fd);
  test->memory()->Store(kParameterAddress, db);
  db->DecRef();

  AReg<T>(test, 0)->data_buffer()->template Set<T>(0, kSysFlen);
  AReg<T>(test, 1)->data_buffer()->template Set<T>(0, kParameterAddress);

  // Make the callback.
  SemiHost<T>(test)->OnEBreak(SemihostInst<T>(test));

  int flen_ret = AReg<T>(test, 0)
                     ->data_buffer()
                     ->template Get<typename std::make_signed<T>::type>(0);

  EXPECT_EQ(flen_ret, 7);
}

TEST_F(RiscVArmSemihostTest, SysFlen32) { SysFlenTest<uint32_t>(this); }

TEST_F(RiscVArmSemihostTest, SysFlen64) { SysFlenTest<uint64_t>(this); }

// Test tmpnam call.

template <typename T>
void SysTmpnamTest(RiscVArmSemihostTest *test) {
  // Set up parameter block.
  auto *db = test->db_factory()->Allocate<T>(3);
  db->template Set<T>(0, kBufferAddress);
  db->template Set<T>(1, 1);
  db->template Set<T>(2, kTmpNamLength);
  test->memory()->Store(kParameterAddress, db);
  db->DecRef();

  // Set up register values.
  AReg<T>(test, 0)->data_buffer()->template Set<T>(0, kSysTmpnam);
  AReg<T>(test, 1)->data_buffer()->template Set<T>(0, kParameterAddress);

  SemiHost<T>(test)->OnEBreak(SemihostInst<T>(test));

  // Check return value.
  EXPECT_EQ(AReg<T>(test, 0)->data_buffer()->template Get<T>(0), 0);

  // Fetch the string and check the length.
  auto tmpnam_db = test->db_factory()->Allocate<char>(kTmpNamLength);
  test->memory()->Load(kBufferAddress, tmpnam_db, nullptr, nullptr);

  char *tmp_name = static_cast<char *>(tmpnam_db->raw_ptr());
  auto length = strlen(tmp_name);
  EXPECT_LE(length, kTmpNamLength);
  EXPECT_GT(length, 0);

  tmpnam_db->DecRef();
}

TEST_F(RiscVArmSemihostTest, SysTmpnam32) { SysTmpnamTest<uint32_t>(this); }

TEST_F(RiscVArmSemihostTest, SysTmpnam64) { SysTmpnamTest<uint64_t>(this); }

// Test that HeapInfo returns a struct with all zeros.

template <typename T>
void SysHeapInfoTest(RiscVArmSemihostTest *test) {
  // Set up register values.
  AReg<T>(test, 0)->data_buffer()->template Set<T>(0, kSysHeapInfo);
  AReg<T>(test, 1)->data_buffer()->template Set<T>(0, kBufferAddress);

  SemiHost<T>(test)->OnEBreak(SemihostInst<T>(test));

  // Fetch the returned values.
  auto *db = test->db_factory()->Allocate<T>(4);
  test->memory()->Load(kBufferAddress, db, nullptr, nullptr);
  EXPECT_EQ(db->template Get<T>(0), 0);
  EXPECT_EQ(db->template Get<T>(1), 0);
  EXPECT_EQ(db->template Get<T>(2), 0);
  EXPECT_EQ(db->template Get<T>(3), 0);
  db->DecRef();
}

TEST_F(RiscVArmSemihostTest, SysHeapInfo32) { SysHeapInfoTest<uint32_t>(this); }

TEST_F(RiscVArmSemihostTest, SysHeapInfo64) { SysHeapInfoTest<uint64_t>(this); }

// Test that seek works by opening a file, seeking, then reading from the file.

template <typename T>
void SysSeekTest(RiscVArmSemihostTest *test) {
  std::string input_file = absl::StrCat(kDepotPath, "testfiles/", kFileName);
  auto *str_db = test->db_factory()->Allocate<uint8_t>(input_file.length() + 1);
  std::memcpy(str_db->raw_ptr(), input_file.c_str(), input_file.length() + 1);
  test->memory()->Store(kStringAddress, str_db);
  str_db->DecRef();

  // Set up parameter block.
  auto *db = test->db_factory()->Allocate<T>(3);
  db->template Set<T>(0, kStringAddress);
  db->template Set<T>(1, 0);
  db->template Set<T>(2, input_file.length());
  test->memory()->Store(kParameterAddress, db);
  db->DecRef();

  // Set up register values.
  AReg<T>(test, 0)->data_buffer()->template Set<T>(0, kSysOpen);
  AReg<T>(test, 1)->data_buffer()->template Set<T>(0, kParameterAddress);

  // Make the callback.
  SemiHost<T>(test)->OnEBreak(SemihostInst<T>(test));

  int fd = AReg<T>(test, 0)
               ->data_buffer()
               ->template Get<typename std::make_signed<T>::type>(0);
  CHECK_GT(fd, 0) << "Invalid file descriptor: " << fd;

  // Seek to position 2 (file length 7).
  db = test->db_factory()->Allocate<T>(2);
  db->template Set<T>(0, fd);
  db->template Set<T>(1, 2);
  test->memory()->Store(kParameterAddress, db);
  db->DecRef();

  AReg<T>(test, 0)->data_buffer()->template Set<T>(0, kSysSeek);
  AReg<T>(test, 1)->data_buffer()->template Set<T>(0, kParameterAddress);

  // Make the callback.
  SemiHost<T>(test)->OnEBreak(SemihostInst<T>(test));

  int seek_ret = AReg<T>(test, 0)
                     ->data_buffer()
                     ->template Get<typename std::make_signed<T>::type>(0);

  EXPECT_EQ(seek_ret, 0);

  // Now read from that file descriptor to verify that the file is in the
  // right position.

  // Read from the file.
  db = test->db_factory()->Allocate<T>(3);
  db->template Set<T>(0, fd);
  db->template Set<T>(1, kBufferAddress);
  db->template Set<T>(2, kBufferSize);
  test->memory()->Store(kParameterAddress, db);
  db->DecRef();

  AReg<T>(test, 0)->data_buffer()->template Set<T>(0, kSysRead);
  AReg<T>(test, 1)->data_buffer()->template Set<T>(0, kParameterAddress);

  // Make the callback.
  SemiHost<T>(test)->OnEBreak(SemihostInst<T>(test));

  int read_ret = AReg<T>(test, 0)
                     ->data_buffer()
                     ->template Get<typename std::make_signed<T>::type>(0);

  db = test->db_factory()->Allocate<uint8_t>(6);
  test->memory()->Load(kBufferAddress, db, nullptr, nullptr);
  std::string file_value(reinterpret_cast<char *>(db->raw_ptr()), 4);
  db->DecRef();

  // Verify the number of characters read and that the characters are correct.
  EXPECT_STREQ("cdef", file_value.c_str());
  EXPECT_EQ(read_ret, 5);
}

TEST_F(RiscVArmSemihostTest, SysSeek32) { SysSeekTest<uint32_t>(this); }

TEST_F(RiscVArmSemihostTest, SysSeek64) { SysSeekTest<uint64_t>(this); }

}  // namespace
