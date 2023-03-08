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

#ifndef MPACT_RISCV_RISCV_RISCV_ARM_SEMIHOST_H_
#define MPACT_RISCV_RISCV_RISCV_ARM_SEMIHOST_H_

#include <functional>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/memory_interface.h"

// This file defines a class that is used to implement ARM style RiscV
// semihosting as described at https://github.com/riscv/riscv-semihosting-spec.
// This class can be used for both 32 and 64 bit versions of the RiscV
// architecture.

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::Instruction;

class RiscVArmSemihost {
 public:
  // Enum for specifying bit width of the architecture.
  enum class BitWidth {
    kWord32,
    kWord64,
  };
  // These opcodes occur before and after an ebreak instruction to indicate
  // that it is a semihosting call.
  // The sequence is:
  // n - 4:   Slli x0, x0, 0x1f
  // n    :   Ebreak
  // n + 4:   Srai x0, x0, 0x7
  static constexpr uint32_t kSlliNop1f = 0x01f01013;
  static constexpr uint32_t kEBreak = 0x00100073;
  static constexpr uint32_t kSraiNop7 = 0x40705013;

  // Register names for the argument registers.
  static constexpr char kA0Name[] = "x10";  // Also known as "a0".
  static constexpr char kA1Name[] = "x11";  // Also known as "a1".

  // Constructor/destructor.
  RiscVArmSemihost(BitWidth bit_width, util::MemoryInterface *i_memory_if,
                   util::MemoryInterface *d_memory_if);
  RiscVArmSemihost() = delete;
  RiscVArmSemihost(const RiscVArmSemihost &) = delete;
  ~RiscVArmSemihost();

  // If the instruction is a semihosting call, execute the requested function.
  void OnEBreak(const Instruction *inst);
  // Return true if the instruction is a semihosting call.
  bool IsSemihostingCall(const Instruction *inst);

  void set_exit_callback(std::function<void()> cb) { exit_callback_ = cb; }
  void set_exception_callback(std::function<void(uint64_t)> cb) {
    exception_callback_ = cb;
  }

 private:
  using SemihostOperation = std::function<absl::Status(uint64_t, uint64_t *)>;

  std::function<void()> exit_callback_;
  std::function<void(uint64_t)> exception_callback_;

  // Exception code for application exit.
  static constexpr int kAdpStoppedApplicationExit = 0x20026;

  // The following op codes are taken from ARM semihosting documentation.
  static constexpr int kSysClose = 0x02;
  static constexpr int kSysClock = 0x10;
  static constexpr int kSysElapsed = 0x30;
  static constexpr int kSysErrno = 0x13;
  static constexpr int kSysException = 0x18;
  static constexpr int kSysFlen = 0x0c;
  static constexpr int kSysGetCmdline = 0x15;
  static constexpr int kSysHeapInfo = 0x16;
  static constexpr int kSysIsError = 0x08;
  static constexpr int kSysIsTty = 0x09;
  static constexpr int kSysOpen = 0x01;
  static constexpr int kSysRead = 0x06;
  static constexpr int kSysReadc = 0x07;
  static constexpr int kSysRemove = 0x0e;
  static constexpr int kSysRename = 0x0f;
  static constexpr int kSysSeek = 0x0a;
  static constexpr int kSysSystem = 0x12;
  static constexpr int kSysTickFreq = 0x31;
  static constexpr int kSysTime = 0x11;
  static constexpr int kSysTmpnam = 0x0d;
  static constexpr int kSysWrite = 0x05;
  static constexpr int kSysWritec = 0x03;
  static constexpr int kSysWrite0 = 0x04;

  // Functions that implement the semihosting operations.
  absl::Status SysClose(uint64_t parameter, uint64_t *ret_val);
  absl::Status SysClock(uint64_t parameter, uint64_t *ret_val);
  absl::Status SysElapsed(uint64_t parameter, uint64_t *ret_val);
  absl::Status SysErrno(uint64_t parameter, uint64_t *ret_val);
  absl::Status SysException(uint64_t parameter, uint64_t *ret_val);
  absl::Status SysFlen(uint64_t parameter, uint64_t *ret_val);
  absl::Status SysGetCmdline(uint64_t parameter, uint64_t *ret_val);
  absl::Status SysHeapInfo(uint64_t parameter, uint64_t *ret_val);
  absl::Status SysIsError(uint64_t parameter, uint64_t *ret_val);
  absl::Status SysIsTty(uint64_t parameter, uint64_t *ret_val);
  absl::Status SysOpen(uint64_t parameter, uint64_t *ret_val);
  absl::Status SysRead(uint64_t parameter, uint64_t *ret_val);
  absl::Status SysReadc(uint64_t parameter, uint64_t *ret_val);
  absl::Status SysRemove(uint64_t parameter, uint64_t *ret_val);
  absl::Status SysRename(uint64_t parameter, uint64_t *ret_val);
  absl::Status SysSeek(uint64_t parameter, uint64_t *ret_val);
  absl::Status SysSystem(uint64_t parameter, uint64_t *ret_val);
  absl::Status SysTickFreq(uint64_t parameter, uint64_t *ret_val);
  absl::Status SysTime(uint64_t parameter, uint64_t *ret_val);
  absl::Status SysTmpnam(uint64_t parameter, uint64_t *ret_val);
  absl::Status SysWrite(uint64_t parameter, uint64_t *ret_val);
  absl::Status SysWritec(uint64_t parameter, uint64_t *ret_val);
  absl::Status SysWrite0(uint64_t parameter, uint64_t *ret_val);

  bool is_32_bit_;
  int sys_errno_ = 0;
  generic::DataBufferFactory db_factory_;
  // Data buffers used by some of the operations. Giving them lifetime of the
  // semihosting instance saves some on allocating and freeing them for each
  // operation.
  generic::DataBuffer *db_inst_;  // Instruction word(s) data buffer.
  generic::DataBuffer *db1_;      // 1 word data buffer.
  generic::DataBuffer *db2_;      // 2 word data buffer.
  generic::DataBuffer *db3_;      // 3 word data buffer.
  generic::DataBuffer *db4_;      // 4 word data buffer.
  // Memory interfaces to use to access intstruction and data memory.
  util::MemoryInterface *i_memory_if_;
  util::MemoryInterface *d_memory_if_;
  // Map from opcode to semihosting function.
  absl::flat_hash_map<uint64_t, SemihostOperation> semihost_operations_;
  // Map of target file descriptors to host file descriptors.
  absl::flat_hash_map<int, int> fd_map_;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_ARM_SEMIHOST_H_
