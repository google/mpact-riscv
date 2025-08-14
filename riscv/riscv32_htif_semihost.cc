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

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <utility>

#include "absl/functional/bind_front.h"
#include "absl/strings/str_cat.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "mpact/sim/util/memory/memory_watcher.h"

namespace mpact {
namespace sim {
namespace riscv {

constexpr int kRiscvAtFdCwd = -100;

using AddressRange = mpact::sim::util::MemoryWatcher::AddressRange;

RiscV32HtifSemiHost::RiscV32HtifSemiHost(
    util::MemoryWatcher* watcher, util::MemoryInterface* memory,
    const SemiHostAddresses& magic_addresses)
    : RiscV32HtifSemiHost(watcher, memory, magic_addresses, nullptr, nullptr) {}

RiscV32HtifSemiHost::RiscV32HtifSemiHost(
    util::MemoryWatcher* watcher, util::MemoryInterface* memory,
    const SemiHostAddresses& magic_addresses, HaltCallback halt_callback,
    ErrorCallback error_callback)
    : halt_callback_(std::move(halt_callback)),
      error_callback_(std::move(error_callback)),
      watcher_(watcher),
      memory_(memory) {
  magic_addresses_ = magic_addresses;
  (void)watcher_->SetStoreWatchCallback(
      AddressRange(magic_addresses_.tohost_ready),
      absl::bind_front(&RiscV32HtifSemiHost::StoreEvent, this));

  // Initialize fd_map_.
  fd_map_.emplace(0, 0);
  fd_map_.emplace(1, 1);
  fd_map_.emplace(2, 2);
  fd_map_.emplace(kRiscvAtFdCwd, AT_FDCWD);
}

RiscV32HtifSemiHost::~RiscV32HtifSemiHost() {
  // Ignore any error when clearing the previous watchpoint.
  (void)watcher_->ClearStoreWatchCallback(magic_addresses_.tohost_ready);
}

void RiscV32HtifSemiHost::SetMagicAddresses(
    const SemiHostAddresses& magic_addresses) {
  // Clear any previous store watchpoint. Ignore any error.
  (void)watcher_->ClearStoreWatchCallback(magic_addresses_.tohost_ready);
  magic_addresses_ = magic_addresses;
  (void)watcher_->SetStoreWatchCallback(
      AddressRange(magic_addresses_.tohost_ready),
      absl::bind_front(&RiscV32HtifSemiHost::StoreEvent, this));
}

void RiscV32HtifSemiHost::SetHaltCallback(HaltCallback callback) {
  halt_callback_ = std::move(callback);
}

void RiscV32HtifSemiHost::SetErrorCallback(ErrorCallback callback) {
  error_callback_ = std::move(callback);
}

void RiscV32HtifSemiHost::StoreEvent(uint64_t address, int size) {
  // Return if the ready byte isn't part of the store event.
  if ((address > magic_addresses_.tohost_ready) ||
      (address + size - 1 < magic_addresses_.tohost_ready))
    return;

  // Read the value of the ready_db, if not 1, then the tohost location
  // does not contain data, and there's nothing to do.
  auto* ready_db = db_factory_.Allocate<uint8_t>(1);
  memory_->Load(magic_addresses_.tohost_ready, ready_db, nullptr, nullptr);
  if (ready_db->Get<uint8_t>(0) != 1) {
    ready_db->DecRef();
    return;
  }

  // Get the payload double word. Extract the command.
  auto* payload_db = db_factory_.Allocate<uint64_t>(1);
  memory_->Load(magic_addresses_.tohost, payload_db, nullptr, nullptr);
  uint64_t payload = payload_db->Get<uint64_t>(0);
  uint8_t device = payload >> 56;
  uint8_t command = (payload >> 48) & 0xff;
  // If it's not the syscall device and command, just return.
  if ((device != 0) || (command != 0)) return;

  // If the payload lsb is 1, then it's the end of the run. Halt (but don't
  // catch fire!).
  if (payload & 0x1) {
    if (halt_callback_ != nullptr) halt_callback_();
    payload_db->DecRef();
    ready_db->DecRef();
    return;
  }
  // The payload contains a pointer to an 8 double word parameter block. Load
  // that block.
  auto* parameter_db = db_factory_.Allocate<uint64_t>(8);
  memory_->Load(payload, parameter_db, nullptr, nullptr);
  auto parameter_span = parameter_db->Get<uint64_t>();
  int64_t return_value = 0;
  // The first parameter specifies the syscall.
  switch (parameter_span[0]) {
    // TODO: Add other semihosting calls.
    case 56:  // sys_openat
    {
      int dirfd = parameter_span[1];
      auto iter = fd_map_.find(dirfd);
      if (iter == fd_map_.end()) {
        dirfd = -1;
      } else {
        dirfd = iter->second;
      }
      auto* db = db_factory_.Allocate(parameter_span[3]);
      auto address = parameter_span[2];
      memory_->Load(address, db, nullptr, nullptr);
      char* name = static_cast<char*>(db->raw_ptr());
      return_value = openat(dirfd, name, parameter_span[4], parameter_span[5]);
      db->DecRef();
    } break;
    case 57:  // sys_close
    {
      auto fd_iter = fd_map_.find(parameter_span[1]);
      if (fd_iter == fd_map_.end()) {
        return_value = -1;
        break;
      }
      fd_map_.erase(fd_iter);
      return_value = 0;
    } break;
    case 62:  // sys_lseek
    {
      auto iter = fd_map_.find(parameter_span[1]);
      int fd = iter == fd_map_.end() ? -1 : iter->second;
      off_t offset = static_cast<off_t>(parameter_span[2]);
      int whence = static_cast<int>(parameter_span[3]);
      return_value = lseek(fd, offset, whence);
    } break;
    case 63:  // sys_read
    {
      auto iter = fd_map_.find(parameter_span[1]);
      int fd = iter == fd_map_.end() ? -1 : iter->second;
      size_t len = static_cast<size_t>(parameter_span[3]);
      auto* data_db = db_factory_.Allocate<uint8_t>(len);
      int res = read(fd, data_db->raw_ptr(), len);
      if (res > 0) {
        generic::DataBuffer* db = nullptr;
        // Need to see if fewer bytes were read than there was space for. If so
        // need to use a different db for the store.
        if (res == len) {
          db = data_db;
        } else {
          db = db_factory_.Allocate<uint8_t>(res);
          std::memcpy(db->raw_ptr(), data_db->raw_ptr(), res);
          data_db->DecRef();
        }
        memory_->Store(parameter_span[2], db);
        db->DecRef();
      }
      return_value = res;
    } break;
    case 64:  // sys_write
    {
      auto iter = fd_map_.find(parameter_span[1]);
      int fd = iter == fd_map_.end() ? -1 : iter->second;
      auto* data_db = db_factory_.Allocate<uint8_t>(parameter_span[3]);
      memory_->Load(parameter_span[2], data_db, nullptr, nullptr);
      size_t len = static_cast<size_t>(parameter_span[3]);
      return_value = write(fd, data_db->raw_ptr(), len);
      data_db->DecRef();
    } break;
    case 67:  // sys_pread
    {
      auto iter = fd_map_.find(parameter_span[1]);
      int fd = iter == fd_map_.end() ? -1 : iter->second;
      size_t len = static_cast<size_t>(parameter_span[3]);
      off_t offset = static_cast<off_t>(parameter_span[4]);
      auto* data_db = db_factory_.Allocate<uint8_t>(len);
      int res = pread(fd, data_db->raw_ptr(), len, offset);
      if (res > 0) {
        generic::DataBuffer* db = nullptr;
        // Need to see if fewer bytes were read than there was space for. If so
        // need to use a different db for the store.
        if (res == len) {
          db = data_db;
        } else {
          db = db_factory_.Allocate<uint8_t>(res);
          std::memcpy(db->raw_ptr(), data_db->raw_ptr(), res);
          data_db->DecRef();
        }
        memory_->Store(parameter_span[2], db);
        db->DecRef();
      }
      return_value = res;
    } break;
    case 68:  // sys_pwrite
    {
      auto iter = fd_map_.find(parameter_span[1]);
      int fd = iter == fd_map_.end() ? -1 : iter->second;
      auto* data_db = db_factory_.Allocate<uint8_t>(parameter_span[3]);
      memory_->Load(parameter_span[2], data_db, nullptr, nullptr);
      size_t len = static_cast<size_t>(parameter_span[3]);
      off_t offset = static_cast<off_t>(parameter_span[4]);
      return_value = pwrite(fd, data_db->raw_ptr(), len, offset);
      data_db->DecRef();
    } break;
    case 93:  // sys_exit
      if (halt_callback_ != nullptr) halt_callback_();
      break;
    default:
      // For now, ignore the others.
      parameter_db->DecRef();
      payload_db->DecRef();
      ready_db->DecRef();
      error_callback_(absl::StrCat("Unhandled syscall: ", parameter_span[0]));
      return;
  }
  parameter_db->DecRef();
  // Write the response packets.
  payload_db->Set<uint64_t>(0, return_value);
  memory_->Store(payload, payload_db);
  payload_db->DecRef();
  ready_db->Set<uint8_t>(0, 1);
  // Signal that the response is ready.
  memory_->Store(magic_addresses_.fromhost_ready, ready_db);
  ready_db->DecRef();
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
