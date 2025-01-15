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

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>

#include "absl/functional/bind_front.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mpact/sim/generic/register.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace mpact {
namespace sim {
namespace riscv {

RiscVArmSemihost::RiscVArmSemihost(BitWidth bit_width,
                                   util::MemoryInterface *i_memory_if,
                                   util::MemoryInterface *d_memory_if)
    : i_memory_if_(i_memory_if),
      d_memory_if_(d_memory_if),
      // Put the functions that implement the different semihosting operations
      // into a map according to their function code.
      semihost_operations_(
          {{kSysClose, absl::bind_front(&RiscVArmSemihost::SysClose, this)},
           {kSysClock, absl::bind_front(&RiscVArmSemihost::SysClock, this)},
           {kSysElapsed, absl::bind_front(&RiscVArmSemihost::SysElapsed, this)},
           {kSysErrno, absl::bind_front(&RiscVArmSemihost::SysErrno, this)},
           {kSysException,
            absl::bind_front(&RiscVArmSemihost::SysException, this)},
           {kSysFlen, absl::bind_front(&RiscVArmSemihost::SysFlen, this)},
           {kSysGetCmdline,
            absl::bind_front(&RiscVArmSemihost::SysGetCmdline, this)},
           {kSysHeapInfo,
            absl::bind_front(&RiscVArmSemihost::SysHeapInfo, this)},
           {kSysIsError, absl::bind_front(&RiscVArmSemihost::SysIsError, this)},
           {kSysIsTty, absl::bind_front(&RiscVArmSemihost::SysIsTty, this)},
           {kSysOpen, absl::bind_front(&RiscVArmSemihost::SysOpen, this)},
           {kSysRead, absl::bind_front(&RiscVArmSemihost::SysRead, this)},
           {kSysReadc, absl::bind_front(&RiscVArmSemihost::SysReadc, this)},
           {kSysRemove, absl::bind_front(&RiscVArmSemihost::SysRemove, this)},
           {kSysRename, absl::bind_front(&RiscVArmSemihost::SysRename, this)},
           {kSysSeek, absl::bind_front(&RiscVArmSemihost::SysSeek, this)},
           {kSysSystem, absl::bind_front(&RiscVArmSemihost::SysSystem, this)},
           {kSysTickFreq,
            absl::bind_front(&RiscVArmSemihost::SysTickFreq, this)},
           {kSysTime, absl::bind_front(&RiscVArmSemihost::SysTime, this)},
           {kSysTmpnam, absl::bind_front(&RiscVArmSemihost::SysTmpnam, this)},
           {kSysWrite, absl::bind_front(&RiscVArmSemihost::SysWrite, this)},
           {kSysWritec, absl::bind_front(&RiscVArmSemihost::SysWritec, this)},
           {kSysWrite0,
            absl::bind_front(&RiscVArmSemihost::SysWrite0, this)}}) {
  is_32_bit_ = bit_width == BitWidth::kWord32;
  // Pre-allocate some fixed length data buffers that are used across semihost
  // calls. Set the length according to the register width for those used to
  // fetch parameters from data memory.
  db_inst_ = db_factory_.Allocate<uint32_t>(3);
  db1_ = is_32_bit_ ? db_factory_.Allocate<uint32_t>(1)
                    : db_factory_.Allocate<uint64_t>(1);
  db2_ = is_32_bit_ ? db_factory_.Allocate<uint32_t>(2)
                    : db_factory_.Allocate<uint64_t>(2);
  db3_ = is_32_bit_ ? db_factory_.Allocate<uint32_t>(3)
                    : db_factory_.Allocate<uint64_t>(3);
  db4_ = is_32_bit_ ? db_factory_.Allocate<uint32_t>(4)
                    : db_factory_.Allocate<uint64_t>(4);
  // stdin
  fd_map_.insert(std::make_pair(0, 0));
  // stdout
  fd_map_.insert(std::make_pair(1, 1));
  // stderr
  fd_map_.insert(std::make_pair(2, 2));
}

RiscVArmSemihost::~RiscVArmSemihost() {
  db_inst_->DecRef();
  db1_->DecRef();
  db2_->DecRef();
  db3_->DecRef();
  db4_->DecRef();
}

bool RiscVArmSemihost::IsSemihostingCall(const Instruction *inst) {
  if (inst == nullptr) return false;
  // Load the instruction words on either side of the ebreak instruction.
  uint64_t address = inst->address() - 4;
  i_memory_if_->Load(address, db_inst_, nullptr, nullptr);
  // Check to see if this is a semihosting call, if not, return.
  if ((db_inst_->Get<uint32_t>(0) != kSlliNop1f) ||
      (db_inst_->Get<uint32_t>(1) != kEBreak) ||
      (db_inst_->Get<uint32_t>(2) != kSraiNop7)) {
    return false;
  }
  return true;
}

void RiscVArmSemihost::OnEBreak(const Instruction *inst) {
  if (!IsSemihostingCall(inst)) return;

  // Handle the semihosting call.
  auto registers = inst->state()->registers();
  auto iter = registers->find(kA0Name);
  auto *a0 = iter == registers->end() ? nullptr : iter->second;
  iter = registers->find(kA1Name);
  auto *a1 = iter == registers->end() ? nullptr : iter->second;
  if ((a0 == nullptr) || (a1 == nullptr)) {
    LOG(ERROR) << "Failed to fetch semihost argument registers";
  }

  // Read the op number and the address of the parameter block from the
  // registers.
  uint64_t op_num = 0;
  uint64_t parameter = 0;
  if (is_32_bit_) {
    op_num = a0->data_buffer()->Get<uint32_t>(0);
    parameter = a1->data_buffer()->Get<uint32_t>(0);
  } else {
    op_num = a0->data_buffer()->Get<uint64_t>(0);
    parameter = a1->data_buffer()->Get<uint64_t>(0);
  }

  // Make sure the op number is valid.
  auto op = semihost_operations_.find(op_num);
  if (op == semihost_operations_.end()) {
    LOG(ERROR) << absl::StrCat("Illegal semihost operation (",
                               absl::Hex(op_num), ")");
    return;
  }

  // Call the semihosting op.
  uint64_t ret_val = 0;
  auto status = op->second(parameter, &ret_val);

  // In case of error, set an error code and log the error message.
  if (!status.ok()) {
    LOG(ERROR) << absl::StrCat("Semihost error: ", status.message());
    if (is_32_bit_) {
      a0->data_buffer()->Set<uint32_t>(0, std::numeric_limits<uint32_t>::max());
    } else {
      a0->data_buffer()->Set<uint64_t>(0, std::numeric_limits<uint64_t>::max());
    }
  } else {
    if (is_32_bit_) {
      a0->data_buffer()->Set<uint32_t>(0, static_cast<uint32_t>(ret_val));
    } else {
      a0->data_buffer()->Set<uint64_t>(0, ret_val);
    }
  }
}

absl::Status RiscVArmSemihost::SysClose(uint64_t parameter, uint64_t *ret_val) {
  // Load the file descriptor from the parameter block.
  d_memory_if_->Load(parameter, db1_, nullptr, nullptr);
  int target_fd = is_32_bit_ ? static_cast<int>(db1_->Get<uint32_t>(0))
                             : static_cast<int>(db1_->Get<uint64_t>(0));
  // Check to see that the target file descriptor is in the map, if not,
  // the fd is not for an opened file.
  auto iter = fd_map_.find(target_fd);
  if (iter == fd_map_.end()) {
    *ret_val = -1ULL;
    return absl::OkStatus();
  }
  // Get the host fd and close the file.
  int host_fd = iter->second;
  // If it's stdin, stdout, or stderr, ignore, just return ok.
  if (host_fd <= 2) {
    *ret_val = 0;
    return absl::OkStatus();
  }
  int ret = close(host_fd);
  if (!ret) {
    sys_errno_ = errno;
  } else {
    // Remove the file descriptor from the map.
    fd_map_.erase(iter);
  }
  *ret_val = static_cast<uint64_t>(ret);
  return absl::OkStatus();
}

// Currently not implemented, will implement once there is need for it.
absl::Status RiscVArmSemihost::SysClock(uint64_t parameter, uint64_t *ret_val) {
  return absl::UnimplementedError("SysClock not implemented");
  // TODO: Complete implementation.
}

// Currently not implemented, will implement once there is need for it.
absl::Status RiscVArmSemihost::SysElapsed(uint64_t parameter,
                                          uint64_t *ret_val) {
  return absl::UnimplementedError("SysElapsed not implemented");
  // TODO: Complete implementation.
}

// Return the value of the simulated errno.
absl::Status RiscVArmSemihost::SysErrno(uint64_t parameter, uint64_t *ret_val) {
  *ret_val = sys_errno_;
  return absl::OkStatus();
}

// Exception notification. The program should be terminated.
absl::Status RiscVArmSemihost::SysException(uint64_t parameter, uint64_t *) {
  // In gcc it seems like the parameter value is passed in the parameter
  // register for RV32, but stored in memory, and then a pointer passed in the
  // parameter register for RV64. A bit odd...
  uint32_t param_value;
  if (is_32_bit_) {
    param_value = static_cast<uint32_t>(parameter);
  } else {
    d_memory_if_->Load(parameter, db1_, nullptr, nullptr);
    param_value = static_cast<uint32_t>(db1_->Get<uint64_t>(0));
  }
  if (param_value == kAdpStoppedApplicationExit) {
    if (exit_callback_ != nullptr) {
      exit_callback_();
      return absl::OkStatus();
    }
    LOG(ERROR) << "Program exit not caught in ARM semihosting - no callback";
    return absl::NotFoundError("Exit callback not valid in ARM semihosting");
  }
  LOG(ERROR) << absl::StrCat(
      "Exception ", absl::Hex(param_value), " other than ApplicationExit (",
      absl::Hex(kAdpStoppedApplicationExit), ") are not implemented");
  return absl::UnimplementedError(
      "Exceptions other than ApplicationExit are not implemented");
}

// Return the length of a file given by the file descriptor.
absl::Status RiscVArmSemihost::SysFlen(uint64_t parameter, uint64_t *ret_val) {
  // Load the targete file descriptor.
  d_memory_if_->Load(parameter, db1_, nullptr, nullptr);
  int target_fd = is_32_bit_ ? static_cast<int>(db1_->Get<uint32_t>(0))
                             : static_cast<int>(db1_->Get<uint64_t>(0));
  // Verify that the file descriptor is valid.
  auto iter = fd_map_.find(target_fd);
  if (iter == fd_map_.end()) {
    *ret_val = -1ULL;
    return absl::OkStatus();
  }
  auto host_fd = iter->second;
  // Get the file info, and get the file length.
  struct stat statbuf;
  int res = fstat(host_fd, &statbuf);
  if (res < 0) {
    sys_errno_ = errno;
    *ret_val = -1ULL;
    return absl::OkStatus();
  }
  *ret_val = static_cast<uint64_t>(statbuf.st_size);
  return absl::OkStatus();
}

// Currently unimplemented. Will implement if there is a demand.
absl::Status RiscVArmSemihost::SysGetCmdline(uint64_t parameter,
                                             uint64_t *ret_val) {
  return absl::UnimplementedError("SysGetCmdline not implemented");
  // TODO: Complete implementation.
}

// Returns 0 information indicating that the call doesn't provide this info.
absl::Status RiscVArmSemihost::SysHeapInfo(uint64_t parameter,
                                           uint64_t *ret_val) {
  d_memory_if_->Load(parameter, db1_, nullptr, nullptr);
  uint64_t block_address = is_32_bit_
                               ? static_cast<int>(db1_->Get<uint32_t>(0))
                               : static_cast<int>(db1_->Get<uint64_t>(0));
  // Return all zeros.
  auto *db = db_factory_.Allocate<uint32_t>(4);
  d_memory_if_->Load(block_address, db, nullptr, nullptr);
  db->Set<uint32_t>(0, 0);
  db->Set<uint32_t>(1, 0);
  db->Set<uint32_t>(2, 0);
  db->Set<uint32_t>(3, 0);
  d_memory_if_->Store(block_address, db);
  db->DecRef();
  return absl::OkStatus();
}

// This function is not implemented for now. Will look into it if there is
// demand.
absl::Status RiscVArmSemihost::SysIsError(uint64_t parameter,
                                          uint64_t *ret_val) {
  return absl::UnimplementedError("SysGetCmdline not implemented");
  // TODO: Complete implementation.
}

// Check if the fd is a tty.
absl::Status RiscVArmSemihost::SysIsTty(uint64_t parameter, uint64_t *ret_val) {
  // Load the target file descriptor.
  d_memory_if_->Load(parameter, db1_, nullptr, nullptr);
  int target_fd = is_32_bit_ ? static_cast<int>(db1_->Get<uint32_t>(0))
                             : static_cast<int>(db1_->Get<uint64_t>(0));
  // Check if the fd is valid.
  auto iter = fd_map_.find(target_fd);
  if (iter == fd_map_.end()) {
    *ret_val = -1ULL;
    return absl::OkStatus();
  }
  auto host_fd = iter->second;
  // Return the value of the isatty call.
  int ret = isatty(host_fd);
  if (!ret) {
    sys_errno_ = errno;
  }
  *ret_val = static_cast<uint64_t>(ret);
  return absl::OkStatus();
}

// Open a file, return the file descriptor if successful.
absl::Status RiscVArmSemihost::SysOpen(uint64_t parameter, uint64_t *ret_val) {
  // Load the parameter block consisiting of pointer to a string, the file open
  // mode, and the length of the string.
  d_memory_if_->Load(parameter, db3_, nullptr, nullptr);
  uint64_t string_address = is_32_bit_
                                ? static_cast<uint64_t>(db3_->Get<uint32_t>(0))
                                : static_cast<uint64_t>(db3_->Get<uint64_t>(0));
  int mode = is_32_bit_ ? static_cast<int>(db3_->Get<int32_t>(1))
                        : static_cast<int>(db3_->Get<int64_t>(1));
  uint64_t file_name_len = is_32_bit_
                               ? static_cast<int>(db3_->Get<uint32_t>(2))
                               : static_cast<int>(db3_->Get<uint64_t>(2));
  // Allocate a data buffer for the file name string, load it, and initialize
  // a string variable with it.
  std::string file_name;
  if (file_name_len > 0) {
    auto *db_c = db_factory_.Allocate<uint8_t>(file_name_len);
    d_memory_if_->Load(string_address, db_c, nullptr, nullptr);
    file_name =
        std::string(static_cast<char *>(db_c->raw_ptr()), file_name_len);
    db_c->DecRef();
  }
  // If the name is ":tt" then it's either cin or cout depending on the mode.
  // In this case just dup the corresponding host fd's.
  int host_fd;
  int target_fd;
  if (file_name == ":tt") {
    if (mode == O_RDONLY) {
      host_fd = 0;
      target_fd = dup(0);
    } else {
      host_fd = 1;
      target_fd = dup(1);
    }
  } else {
    // Open the file, create if needed for write/update modes.
    if (mode == O_RDONLY) {
      host_fd = open(file_name.c_str(), 0, mode);
    } else {
      host_fd = open(file_name.c_str(), O_CREAT, mode);
    }
    target_fd = dup(host_fd);
  }
  *ret_val = static_cast<uint64_t>(target_fd);
  if (target_fd < 0) {
    sys_errno_ = errno;
    return absl::OkStatus();
  }

  fd_map_.insert(std::make_pair(target_fd, host_fd));
  return absl::OkStatus();
}

absl::Status RiscVArmSemihost::SysRead(uint64_t parameter, uint64_t *ret_val) {
  // Load the parameter block, consisting of the target file descriptor, the
  // target buffer address, and buffer length.
  d_memory_if_->Load(parameter, db3_, nullptr, nullptr);
  int target_fd = is_32_bit_ ? static_cast<int>(db3_->Get<uint32_t>(0))
                             : static_cast<int>(db3_->Get<uint64_t>(0));
  uint64_t buffer_address = is_32_bit_
                                ? static_cast<uint64_t>(db3_->Get<uint32_t>(1))
                                : static_cast<uint64_t>(db3_->Get<uint64_t>(1));
  int length = is_32_bit_ ? static_cast<int>(db3_->Get<int32_t>(2))
                          : static_cast<int>(db3_->Get<int64_t>(2));
  // Check that the file descriptor is valid.
  auto iter = fd_map_.find(target_fd);
  if (iter == fd_map_.end()) {
    *ret_val = -1ULL;
    return absl::OkStatus();
  }
  auto host_fd = iter->second;
  // Allocate a data buffer sufficient for the target buffer length.
  auto *db = db_factory_.Allocate<uint8_t>(length);
  // Read from the file/
  int res = read(host_fd, db->raw_ptr(), length);
  *ret_val = static_cast<uint64_t>(res);
  if (res < 0) {
    sys_errno_ = errno;
    return absl::OkStatus();
  }
  // Write to the buffer.
  d_memory_if_->Store(buffer_address, db);
  db->DecRef();
  return absl::OkStatus();
}

// Read a byte from the debug console. This is not implemented for now.
absl::Status RiscVArmSemihost::SysReadc(uint64_t parameter, uint64_t *ret_val) {
  return absl::UnimplementedError("SysReadc not implemented");
  // TODO: Complete implementation.
}

// Remove a file from the host file system. This will not be implemented.
absl::Status RiscVArmSemihost::SysRemove(uint64_t parameter,
                                         uint64_t *ret_val) {
  return absl::UnimplementedError("SysRemove not implemented");
}

// Rename a file in the host file system. This will not be implemented.
absl::Status RiscVArmSemihost::SysRename(uint64_t parameter,
                                         uint64_t *ret_val) {
  return absl::UnimplementedError("SysRename not implemented");
}

// Seek in the file specified by the target file descriptor.
absl::Status RiscVArmSemihost::SysSeek(uint64_t parameter, uint64_t *ret_val) {
  // Load the parameters consisting of the target fd and the desired seek
  // offset.
  d_memory_if_->Load(parameter, db2_, nullptr, nullptr);
  int target_fd = is_32_bit_ ? static_cast<int>(db2_->Get<uint32_t>(0))
                             : static_cast<int>(db2_->Get<uint64_t>(0));
  uint64_t offset = is_32_bit_ ? static_cast<int>(db2_->Get<uint32_t>(1))
                               : static_cast<int>(db2_->Get<uint64_t>(1));
  // Verify that the target fd is valid.
  auto iter = fd_map_.find(target_fd);
  if (iter == fd_map_.end()) {
    *ret_val = -1ULL;
    return absl::OkStatus();
  }
  auto host_fd = iter->second;
  // Perform the seek relative to the beginning of the file.
  auto res = lseek(host_fd, offset, SEEK_SET);
  if (res < 0) {
    sys_errno_ = errno;
    res = -1;
  } else {
    res = 0;
  }
  *ret_val = static_cast<uint64_t>(res);
  return absl::OkStatus();
}

// Execute a command in the shell of the host. This will not be implemented.
absl::Status RiscVArmSemihost::SysSystem(uint64_t parameter,
                                         uint64_t *ret_val) {
  return absl::UnimplementedError("SysSystem not implemented");
}

// Return the system tick frequency. For now just return -1 to indicate that
// this call is not fully supported.
absl::Status RiscVArmSemihost::SysTickFreq(uint64_t parameter,
                                           uint64_t *ret_val) {
  *ret_val = -1ULL;
  return absl::OkStatus();
}

// Return unix time in seconds.
absl::Status RiscVArmSemihost::SysTime(uint64_t parameter, uint64_t *ret_val) {
  *ret_val = time(nullptr);
  return absl::OkStatus();
}

// Return a temporary file name.
absl::Status RiscVArmSemihost::SysTmpnam(uint64_t parameter,
                                         uint64_t *ret_val) {
  // Load parameters consisting of a pointer to a buffer, an int (0-255) that
  // is a target identifier for this filename, and the length of the buffer.
  d_memory_if_->Load(parameter, db3_, nullptr, nullptr);
  uint64_t buffer_address = is_32_bit_
                                ? static_cast<int>(db3_->Get<uint32_t>(0))
                                : static_cast<int>(db3_->Get<uint64_t>(0));
  int target_id = is_32_bit_ ? static_cast<uint64_t>(db3_->Get<uint32_t>(1))
                             : static_cast<uint64_t>(db3_->Get<uint64_t>(1));
  int length = is_32_bit_ ? static_cast<int>(db3_->Get<int32_t>(2))
                          : static_cast<int>(db3_->Get<int64_t>(2));
  // Validate the target_id.
  if (target_id < 0 || target_id > 255) {
    *ret_val = -1ULL;
    return absl::OkStatus();
  }
  // The length of the buffer has to be at least L_tmpnam
  if (length < L_tmpnam) {
    *ret_val = -1ULL;
    return absl::OkStatus();
  }
  // Allocate a data buffer and call tmpnam, then write the name to the buffer.
  auto tmpnam_db = db_factory_.Allocate<uint8_t>(length);
  tmpnam(static_cast<char *>(tmpnam_db->raw_ptr()));
  d_memory_if_->Store(buffer_address, tmpnam_db);
  tmpnam_db->DecRef();
  return absl::OkStatus();
}

// Write data to a file.
absl::Status RiscVArmSemihost::SysWrite(uint64_t parameter, uint64_t *ret_val) {
  // Load parameters consisting of target fd, target buffer address, and
  // length.
  d_memory_if_->Load(parameter, db3_, nullptr, nullptr);
  int target_fd = is_32_bit_ ? static_cast<int>(db3_->Get<uint32_t>(0))
                             : static_cast<int>(db3_->Get<uint64_t>(0));
  uint64_t buffer_address = is_32_bit_
                                ? static_cast<uint64_t>(db3_->Get<uint32_t>(1))
                                : static_cast<uint64_t>(db3_->Get<uint64_t>(1));
  int length = is_32_bit_ ? static_cast<int>(db3_->Get<int32_t>(2))
                          : static_cast<int>(db3_->Get<int64_t>(2));
  // Verify that the target fd is valid.
  auto iter = fd_map_.find(target_fd);
  if (iter == fd_map_.end()) {
    *ret_val = -1ULL;
    return absl::OkStatus();
  }
  auto host_fd = iter->second;
  // Allocate the data buffer necessary to read the data to be written to the
  // file.
  auto *db = db_factory_.Allocate<uint8_t>(length);
  d_memory_if_->Load(buffer_address, db, nullptr, nullptr);
  // Write the data to the file.
  int res = write(host_fd, db->raw_ptr(), length);
  *ret_val = static_cast<uint64_t>(res);
  if (res < 0) {
    sys_errno_ = errno;
  }
  *ret_val = static_cast<uint64_t>(res);
  db->DecRef();
  return absl::OkStatus();
}

// Write a byte to the degbug console. This is not implemented for now.
absl::Status RiscVArmSemihost::SysWritec(uint64_t parameter,
                                         uint64_t *ret_val) {
  return absl::UnimplementedError("SysWritec not implemented");
}

// Write a null terminated string to the debug console. This is not implemented
// for now.
absl::Status RiscVArmSemihost::SysWrite0(uint64_t parameter,
                                         uint64_t *ret_val) {
  return absl::UnimplementedError("SysWrite0 not implemented");
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
