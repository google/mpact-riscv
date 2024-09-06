#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_COUNTER_CSR_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_COUNTER_CSR_H_

#include <cstdint>
#include <limits>
#include <string>

#include "mpact/sim/generic/counters.h"
#include "riscv/riscv_csr.h"

// This file defines a base class from which to derive CSR classes for
// minstret/minstreth and CounterCsr/mycycleh.Once instantiated, the instances
// of these classes need to be tied to a cycle counter of the top level of the
// simulator. That binding is done when the simulator is instantiated. Until
// that is done, the CSRs just work like scratch CSRs.

// Since these CSRs are both readable and writable, but the counter value cannot
// be changed, every time the registers are written, a relative offset is
// computed from the counter, so that the values read are relative to the most
// recent write of the CSR.

namespace mpact::sim::riscv {

using ::mpact::sim::generic::SimpleCounter;
using ::mpact::sim::riscv::RiscVSimpleCsr;

template <typename S>
class RiscVCounterCsrHigh;

// This class implements the 64 bit CSR, or the low 32 bit CSR depending on the
// RiscV xlen width.
template <typename T, typename S>
class RiscVCounterCsr : public RiscVSimpleCsr<T> {
  friend class RiscVCounterCsrHigh<S>;

 public:
  const T kMax = std::numeric_limits<T>::max();
  RiscVCounterCsr(std::string name, RiscVCsrEnum csr_enum, S* state)
      : RiscVSimpleCsr<T>(name, csr_enum, state) {}
  RiscVCounterCsr(const RiscVCounterCsr&) = delete;
  RiscVCounterCsr& operator=(const RiscVCounterCsr&) = delete;
  ~RiscVCounterCsr() override = default;

  // RiscVSimpleCsr method overrides.
  uint32_t GetUint32() override {
    auto value = (GetCounterValue() + offset_) & kMax;
    return value;
  }
  uint64_t GetUint64() override {
    return (GetCounterValue() + offset_) & kMax;
  };

  // Any value written to the CSR is used to create an offset from the current
  // value of the counter.
  void Set(uint32_t value) override { offset_ = value - GetCounterValue(); }
  void Set(uint64_t value) override {
    offset_ = (value & kMax) - GetCounterValue();
  }
  // This is called to tie a cycle counter to the CSR.
  void set_counter(SimpleCounter<uint64_t>* counter) { counter_ = counter; }

 private:
  inline T GetCounterValue() const {
    if (counter_ == nullptr) return 0;
    return static_cast<T>(counter_->GetValue() & kMax);
  };

  SimpleCounter<uint64_t>* counter_ = nullptr;
  T offset_ = 0;
};

// This class implements the "high" version of the CSR on 32-bit RiscV.
template <typename S>
class RiscVCounterCsrHigh : public RiscVSimpleCsr<uint32_t> {
 public:
  RiscVCounterCsrHigh(std::string name, RiscVCsrEnum csr_enum, S* state,
                      RiscVCounterCsr<uint32_t, S>* low_csr)
      : RiscVSimpleCsr<uint32_t>(name, csr_enum, state), low_csr_(low_csr) {}
  RiscVCounterCsrHigh(const RiscVCounterCsrHigh&) = delete;
  RiscVCounterCsrHigh& operator=(const RiscVCounterCsrHigh&) = delete;
  ~RiscVCounterCsrHigh() override = default;

  // RiscVSimpleCsr method overrides.
  uint32_t GetUint32() override {
    uint64_t offset = (offset_ << 32) | low_csr_->offset_;
    uint64_t value64 = GetCounterValue() + offset;
    uint32_t value = static_cast<uint32_t>(value64 >> 32);
    return value;
  };
  uint64_t GetUint64() override { return static_cast<uint64_t>(GetUint32()); };

  // Any value written to the CSR is used to create an offset from the current
  // value of the counter.
  void Set(uint32_t value) override {
    offset_ = value - (GetCounterValue() >> 32);
  };
  void Set(uint64_t value) override { Set(static_cast<uint32_t>(value)); };

  // This is called to tie a cycle counter to the CSR.
  void set_counter(SimpleCounter<uint64_t>* counter) { counter_ = counter; }

 private:
  inline uint64_t GetCounterValue() const {
    if (counter_ == nullptr) return 0;
    return counter_->GetValue();
  };

  RiscVCounterCsr<uint32_t, S>* low_csr_;
  SimpleCounter<uint64_t>* counter_ = nullptr;
  uint64_t offset_ = 0;
};

}  // namespace mpact::sim::riscv

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_COUNTER_CSR_H_
