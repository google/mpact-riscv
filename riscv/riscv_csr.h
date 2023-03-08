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

#ifndef MPACT_RISCV_RISCV_RISCV_CSR_H_
#define MPACT_RISCV_RISCV_RISCV_CSR_H_

#include <any>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/operand_interface.h"

// This file contains definitions for classes used to model the RiscV constrol
// and status registers. For now, these are not modeled as actual register
// state, instead, they're tied into the RiscV machine state a bit more. This
// is so that side-effects from reads/writes can more easily be handled.

namespace mpact {
namespace sim {
namespace riscv {

class RiscVState;

enum class RiscVCsrEnum {
  // User trap setup.
  kUStatus = 0x000,
  kUIe = 0x004,
  kUTvec = 0x005,
  // User trap handling.
  kUScratch = 0x040,
  kUEpc = 0x041,
  kUCause = 0x042,
  kUTval = 0x043,
  kUIp = 0x044,
  // User floating point CSRs.
  kFFlags = 0x001,
  kFrm = 0x002,
  kFCsr = 0x003,
  // User counters/timers.
  kCycle = 0xc00,
  kTime = 0xc01,
  kInstret = 0xc02,

  // Ignoring perf monitoring counters for now.

  kCycleH = 0xc80,
  kTimeH = 0xc81,
  kInstretH = 0x82,

  // Ignoring high bits of perf monitoring counters for now.

  // Supervisor trap setup.
  kSStatus = 0x100,
  kSEDeleg = 0x102,
  kSIDeleg = 0x103,
  kSIe = 0x104,
  kSTvec = 0x105,
  kSCounteren = 0x106,
  // Supervisor trap handling.
  kSScratch = 0x140,  // Scratch register for supervisor trap handlers.
  kSEpc = 0x141,      // Supervisor exception program counter.
  kSCause = 0x142,    // Supervisor trap cause.
  kSTval = 0x143,     // Supervisor bad address or instruction.
  kSIp = 0x144,       // Supervisor interrupt pending.

  // Supervisor protection and translation.
  kSAtp = 0x180,  // Supervisor address translation and protection.

  // Machine information registers.
  kMVendorId = 0xf11,  // Vendor ID.
  kMArchId = 0xf12,    // Architecture ID.
  kMImpId = 0xf13,     // Implementation ID.
  kMHartId = 0xf14,    // Hardware thread ID.
  // Machine trap setup.
  kMStatus = 0x300,     // Machine status register.
  kMIsa = 0x301,        // ISA and extensions.
  kMEDeleg = 0x302,     // Machine exception delegation register.
  kMIDeleg = 0x303,     // Machine interrupt delegation register.
  kMIe = 0x304,         // Machine interrupt-enable register.
  kMTvec = 0x305,       // Machine trap-handler base address.
  kMCounterEn = 0x306,  // Machine counter enable.
  // Machine trap handling.
  kMScratch = 0x340,  // Scratch register for machine trap handlers.
  kMEpc = 0x341,      // Machine exception program counter.
  kMCause = 0x342,    // Machine trap cause.
  kMTval = 0x343,     // Machine bad address or instruction.
  kMIp = 0x344,       // Machine interrupt pending.

  // Ignoring machine memory protection for now.

  kMCycle = 0xb00,    // Machine cycle counter.
  kMInstret = 0xb02,  // Machine instructions-retired counter.

  // Ignoring machine performance counters for now.

  kMCycleH = 0xc80,   // Upper 32 bits of mcycle.
  kMInstretH = 0x82,  // Upper 32 bits of MInstret

  // Ignoring machine counter setup for now.

  // Ignoring debug trace registers.

  // Ignoring debug mode registers.

  // Simulator specific CSRs. These are numbered 0x1800-0x18ff.
  kSimMode = 0x1800,
};

// Base class for CSRs contains the name and index. The index is made wide in
// case memory mapped registers need to be supported with their address as
// the index.
class RiscVCsrBase {
 public:
  RiscVCsrBase(std::string name, uint64_t index, RiscVState *state)
      : name_(name), index_(index), state_(state) {}
  RiscVCsrBase() = delete;
  virtual ~RiscVCsrBase() = default;

  // Source and destination creation interface.
  virtual generic::SourceOperandInterface *CreateSourceOperand() = 0;
  virtual generic::DestinationOperandInterface *CreateSetDestinationOperand(
      int latency, std::string op_name) = 0;
  virtual generic::DestinationOperandInterface *CreateClearDestinationOperand(
      int latency, std::string op_name) = 0;
  virtual generic::DestinationOperandInterface *CreateWriteDestinationOperand(
      int latency, std::string op_name) = 0;

  // Three getters: name, index, and state
  const std::string &name() { return name_; }
  uint64_t index() const { return index_; }
  RiscVState *state() { return state_; }

 private:
  std::string name_;
  uint64_t index_ = 0;
  RiscVState *state_;
};

class RiscVCsrInterface : public RiscVCsrBase {
 public:
  RiscVCsrInterface(std::string name, uint64_t index, RiscVState *state)
      : RiscVCsrBase(name, index, state) {}
  RiscVCsrInterface() = delete;
  ~RiscVCsrInterface() override = default;
  // Return the value, modified as per read mask.
  virtual uint32_t AsUint32() = 0;
  virtual uint64_t AsUint64() = 0;
  // Write the value, modified as per write mask.
  virtual void Write(uint32_t value) = 0;
  virtual void Write(uint64_t value) = 0;
  // Set the bits that are set in value, leave other bits unchanged.
  virtual void SetBits(uint32_t value) = 0;
  virtual void SetBits(uint64_t value) = 0;
  // Clear the bits that are set in value, leave other bits unchanged.
  virtual void ClearBits(uint32_t value) = 0;
  virtual void ClearBits(uint64_t value) = 0;
  // Return the value, ignoring the read mask.
  virtual uint32_t GetUint32() = 0;
  virtual uint64_t GetUint64() = 0;
  // Sets the value, ignoring the write mask.
  virtual void Set(uint32_t value) = 0;
  virtual void Set(uint64_t value) = 0;
  // Size of value.
  virtual size_t size() const = 0;
  // Set to reset value.
  virtual void Reset() = 0;
};

class RiscVCsrWriteDb : public generic::DataBufferDestination {
 public:
  RiscVCsrWriteDb() = delete;
  explicit RiscVCsrWriteDb(RiscVCsrInterface *csr) : csr_(csr) {}

  void SetDataBuffer(generic::DataBuffer *db) override;

 private:
  RiscVCsrInterface *csr_;
};

class RiscVCsrClearBitsDb : public generic::DataBufferDestination {
 public:
  RiscVCsrClearBitsDb() = delete;
  explicit RiscVCsrClearBitsDb(RiscVCsrInterface *csr) : csr_(csr) {}

  void SetDataBuffer(generic::DataBuffer *db) override;

 private:
  RiscVCsrInterface *csr_;
};

class RiscVCsrSetBitsDb : public generic::DataBufferDestination {
 public:
  RiscVCsrSetBitsDb() = delete;
  explicit RiscVCsrSetBitsDb(RiscVCsrInterface *csr) : csr_(csr) {}

  void SetDataBuffer(generic::DataBuffer *db) override;

 private:
  RiscVCsrInterface *csr_;
};

// RiscV CSR class.
template <typename T>
class RiscVSimpleCsr : public RiscVCsrInterface {
 public:
  RiscVSimpleCsr(std::string name, RiscVCsrEnum index, RiscVState *state)
      : RiscVSimpleCsr(
            name, index, 0,
            std::numeric_limits<typename std::make_unsigned<T>::type>::max(),
            std::numeric_limits<typename std::make_unsigned<T>::type>::max(),
            state) {}
  RiscVSimpleCsr(std::string name, RiscVCsrEnum index, T initial_value,
                 RiscVState *state)
      : RiscVSimpleCsr(
            name, index, initial_value,
            std::numeric_limits<typename std::make_unsigned<T>::type>::max(),
            std::numeric_limits<typename std::make_unsigned<T>::type>::max(),
            state) {}
  RiscVSimpleCsr(std::string name, RiscVCsrEnum index, T read_mask,
                 T write_mask, RiscVState *state)
      : RiscVSimpleCsr(name, index, 0, read_mask, write_mask, state) {}
  RiscVSimpleCsr(std::string name, RiscVCsrEnum index, T initial_value,
                 T read_mask, T write_mask, RiscVState *state)
      : RiscVCsrInterface(name, static_cast<uint64_t>(index), state),
        value_(initial_value),
        read_mask_(read_mask),
        write_mask_(write_mask),
        write_target_(new RiscVCsrWriteDb(this)),
        set_bits_target_(new RiscVCsrSetBitsDb(this)),
        clear_bits_target_(new RiscVCsrClearBitsDb(this)) {}

  ~RiscVSimpleCsr() override {
    delete write_target_;
    delete set_bits_target_;
    delete clear_bits_target_;
  }

  // Disable default and copy constructor, as well as assignment.
  RiscVSimpleCsr() = delete;
  RiscVSimpleCsr(const RiscVSimpleCsr &) = delete;
  RiscVSimpleCsr &operator=(const RiscVSimpleCsr &) = delete;

  // Return the value, modified as per read mask.
  uint32_t AsUint32() override {
    return static_cast<uint32_t>(value_ & read_mask_);
  }
  uint64_t AsUint64() override {
    return static_cast<uint64_t>(value_ & read_mask_);
  }
  // Write the value, modified as per write mask.
  void Write(uint32_t value) override {
    T t_value = static_cast<T>(value);
    Set((t_value & write_mask_) | (value_ & ~write_mask_));
  }
  void Write(uint64_t value) override {
    T t_value = static_cast<T>(value);
    Set((t_value & write_mask_) | (value_ & ~write_mask_));
  }
  // Set the bits specified in the value. Don't change the other bits.
  void SetBits(uint32_t value) override { Write(GetUint32() | value); }
  void SetBits(uint64_t value) override { Write(GetUint64() | value); }
  // Clear the bits specified in the value. Don't change the other bits.
  void ClearBits(uint32_t value) override { Write(GetUint32() & ~value); }
  void ClearBits(uint64_t value) override { Write(GetUint64() & ~value); }
  // Return the value, ignoring the read mask and any read related side-effects.
  uint32_t GetUint32() override { return static_cast<uint32_t>(value_); }
  uint64_t GetUint64() override { return static_cast<uint64_t>(value_); }
  // Sets the value, ignoring the write mask and any write related side-effects.
  void Set(uint32_t value) override { value_ = static_cast<T>(value); }
  void Set(uint64_t value) override { value_ = static_cast<T>(value); }

  // Operand creation interface.
  generic::SourceOperandInterface *CreateSourceOperand() override;
  generic::DestinationOperandInterface *CreateSetDestinationOperand(
      int latency, std::string op_name) override;
  generic::DestinationOperandInterface *CreateClearDestinationOperand(
      int latency, std::string op_name) override;
  generic::DestinationOperandInterface *CreateWriteDestinationOperand(
      int latency, std::string op_name) override;

  // Getters and setters.
  T read_mask() const { return read_mask_; }
  void set_read_mask(T value) { read_mask_ = value; }
  T write_mask() const { return write_mask_; }
  void set_write_mask(T value) { write_mask_ = value; }

  RiscVCsrWriteDb *write_target() const { return write_target_; }
  RiscVCsrSetBitsDb *set_bits_target() const { return set_bits_target_; }
  RiscVCsrClearBitsDb *clear_bits_target() const { return clear_bits_target_; }

  size_t size() const override { return sizeof(T); }

  void Reset() override { Set(static_cast<uint64_t>(0ULL)); }

 private:
  std::string name_;
  RiscVCsrEnum index_;
  T value_ = 0;
  T read_mask_ =
      std::numeric_limits<typename std::make_unsigned<T>::type>::max();
  T write_mask_ =
      std::numeric_limits<typename std::make_unsigned<T>::type>::max();
  RiscVCsrWriteDb *write_target_;
  RiscVCsrSetBitsDb *set_bits_target_;
  RiscVCsrClearBitsDb *clear_bits_target_;
};

using RiscV32SimpleCsr = RiscVSimpleCsr<uint32_t>;
using RiscV64SimpleCsr = RiscVSimpleCsr<uint64_t>;

class RiscVCsrSet {
 public:
  // Constructor.
  RiscVCsrSet() = default;
  RiscVCsrSet(const RiscVCsrSet &) = delete;
  RiscVCsrSet &operator=(const RiscVCsrSet &) = delete;

  // Add the CSR to the CsrSet. Return error if it already exists. The storage
  // is owned by the caller, and must remain valid for the duration of
  // the lifetime of the CsrSet instance, or explicitly removed.
  absl::Status AddCsr(RiscVCsrInterface *csr);
  // Methods to get a handle to the CSR.
  absl::StatusOr<RiscVCsrInterface *> GetCsr(absl::string_view name);
  absl::StatusOr<RiscVCsrInterface *> GetCsr(uint64_t index);
  // Remove the CSR from the CsrSet.
  absl::Status RemoveCsr(uint64_t csr_index);

 private:
  absl::flat_hash_map<std::string, RiscVCsrInterface *> csr_name_map_;
  absl::flat_hash_map<uint64_t, RiscVCsrInterface *> csr_index_map_;
};

// Source operand type for CSR.
class RiscVCsrSourceOperand : public generic::SourceOperandInterface {
 public:
  // Constructor. Note, default constructor is deleted.
  RiscVCsrSourceOperand(RiscVCsrInterface *csr, const std::string op_name);
  explicit RiscVCsrSourceOperand(RiscVCsrInterface *csr);
  RiscVCsrSourceOperand() = delete;

  // Methods to read the value of the CSR.
  bool AsBool(int i) final;
  int8_t AsInt8(int i) final;
  uint8_t AsUint8(int i) final;
  int16_t AsInt16(int i) final;
  uint16_t AsUint16(int i) final;
  int32_t AsInt32(int i) final;
  uint32_t AsUint32(int i) final;
  int64_t AsInt64(int i) final;
  uint64_t AsUint64(int i) final;
  // Returns the RiscVCsrInterface<T> object wrapped in absl::any.
  std::any GetObject() const final { return std::any(csr_); }
  // Non-inherited method to get the register object.
  RiscVCsrInterface *GetCsr() const { return csr_; }

  std::vector<int> shape() const final { return {1}; }

  std::string AsString() const final { return op_name_; }

 private:
  RiscVCsrInterface *csr_;
  std::string op_name_;
};

// RiscV Csr destination operand type .
class RiscVCsrDestinationOperand : public generic::DestinationOperandInterface {
 public:
  // Constructor and Destructor
  RiscVCsrDestinationOperand(RiscVCsrInterface *csr,
                             generic::DataBufferDestination *db_dest,
                             int latency);
  RiscVCsrDestinationOperand(RiscVCsrInterface *csr,
                             generic::DataBufferDestination *db_dest,
                             int latency, std::string op_name);
  RiscVCsrDestinationOperand() = delete;

  // Initializes the DataBuffer instance so that when Submit is called, it can
  // be entered into the correct delay line, with the correct latency, targeting
  // the correct csr.
  void InitializeDataBuffer(generic::DataBuffer *db) override;

  // Allocates and returns an initialized DataBuffer instance that contains a
  // copy of the current value of the csr. This is useful when only part
  // of the destination register will be modified.
  generic::DataBuffer *CopyDataBuffer() override;

  // Allocates and returns an initialized DataBuffer instance.
  generic::DataBuffer *AllocateDataBuffer() final;

  // Returns the latency associated with writes to this csr operand.
  int latency() const override { return latency_; }

  // Returns the csr interface object wrapped in absl::any.
  std::any GetObject() const override { return std::any(csr_); }

  std::vector<int> shape() const override { return {1}; }

  // Non-inherited method to get the register object.
  RiscVCsrInterface *GetRiscVCsr() const { return csr_; }

  std::string AsString() const override { return op_name_; }

 private:
  RiscVCsrInterface *csr_;
  generic::DataBufferDestination *db_dest_;
  generic::DataBufferFactory *db_factory_;
  int latency_;
  generic::DataBufferDelayLine *delay_line_;
  std::string op_name_;
};

template <typename T>
generic::DestinationOperandInterface *
RiscVSimpleCsr<T>::CreateSetDestinationOperand(int latency,
                                               std::string op_name) {
  return new RiscVCsrDestinationOperand(this, this->set_bits_target(), latency,
                                        op_name);
}

template <typename T>
generic::DestinationOperandInterface *
RiscVSimpleCsr<T>::CreateClearDestinationOperand(int latency,
                                                 std::string op_name) {
  return new RiscVCsrDestinationOperand(this, this->clear_bits_target(),
                                        latency, op_name);
}

template <typename T>
generic::DestinationOperandInterface *
RiscVSimpleCsr<T>::CreateWriteDestinationOperand(int latency,
                                                 std::string op_name) {
  return new RiscVCsrDestinationOperand(this, this->write_target(), latency,
                                        op_name);
}

template <typename T>
generic::SourceOperandInterface *RiscVSimpleCsr<T>::CreateSourceOperand() {
  return new RiscVCsrSourceOperand(this);
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_CSR_H_
