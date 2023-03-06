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

#ifndef RISCV_RISCV_XIP_XIE_H_
#define RISCV_RISCV_XIP_XIE_H_

#include "riscv/riscv_csr.h"

// This file defines the classes for the interrupt pending and enable
// registers for machine mode (mip/mie), supervisor mode (sip/sie), and
// TODO(torerik) user mode (uip/uie).

namespace mpact {
namespace sim {
namespace riscv {

class RiscVState;

// xip - x mode interrupt pending registers.

class RiscVMIp : public RiscVSimpleCsr<uint32_t> {
 public:
  // Read and Write masks.
  static constexpr uint32_t kReadMask = 0b1011'1011'1011;
  static constexpr uint32_t kWriteMask = 0b0011'0011'1011;

  // Disable default constructor.
  RiscVMIp() = delete;
  RiscVMIp(uint32_t initial_value, RiscVState* state);
  ~RiscVMIp() override = default;

  // RiscVSimpleCsr method overrides.
  void Set(uint32_t) override;
  void Set(uint64_t) override;

  // X external interrupt pending.
  bool meip() { return GetterHelper<11, 0b1>(); }
  bool seip() { return GetterHelper<9, 0b1>(); }
  bool ueip() { return GetterHelper<8, 0b1>(); }
  void set_meip(uint32_t value) { SetterHelper<11, 0b1>(value); }
  void set_seip(uint32_t value) { SetterHelper<9, 0b1>(value); }
  void set_ueip(uint32_t value) { SetterHelper<8, 0b1>(value); }

  // X timer interrupt pending.
  bool mtip() { return GetterHelper<7, 0b1>(); }
  bool stip() { return GetterHelper<5, 0b1>(); }
  bool utip() { return GetterHelper<4, 0b1>(); }
  void set_mtip(uint32_t value) { SetterHelper<7, 0b1>(value); }
  void set_stip(uint32_t value) { SetterHelper<5, 0b1>(value); }
  void set_utip(uint32_t value) { SetterHelper<4, 0b1>(value); }

  // X software interrupt pending.
  bool msip() { return GetterHelper<3, 0b1>(); }
  bool ssip() { return GetterHelper<1, 0b1>(); }
  bool usip() { return GetterHelper<0, 0b1>(); }
  void set_msip(uint32_t value) { SetterHelper<3, 0b1>(value); }
  void set_ssip(uint32_t value) { SetterHelper<1, 0b1>(value); }
  void set_usip(uint32_t value) { SetterHelper<0, 0b1>(value); }

 private:
  // Template function to help implement the getters.
  template <int Shift, uint32_t BitMask>
  inline int GetterHelper() {
    return (GetUint32() >> Shift) & BitMask;
  }
  // Template function to help implement the setters.
  template <int Shift, uint32_t BitMask>
  inline void SetterHelper(uint32_t value) {
    uint64_t bit_value = value & BitMask;
    uint64_t new_value =
        (GetUint64() & ~(BitMask << Shift)) | (bit_value << Shift);
    Write(new_value);
  }
};

// The supervisor mode sip is an interface to mip. The visibility of mip bits
// depends on the value of mideleg. Delegated interrupts bits are readable and
// writable in sip as they would be in mip.
class RiscVSIp : public RiscVSimpleCsr<uint32_t> {
 public:
  // Read and Write masks.
  static constexpr uint32_t kReadMask = 0b1011'1011'1011;
  static constexpr uint32_t kWriteMask = 0b0011'0011'0011;
  static constexpr uint32_t kMBitMask = 0b1000'1000'1000;
  static constexpr uint32_t kSBitMask = 0b0010'0010'0010;
  static constexpr uint32_t kUBitMask = 0b0001'0001'0001;

  // Disable default constructor.
  RiscVSIp() = delete;
  RiscVSIp(RiscVMIp* mip, RiscVCsrInterface* mideleg, RiscVState* state);
  ~RiscVSIp() override = default;

  // RiscVSimpleCsr method overrides.
  void Set(uint32_t) override;
  void Set(uint64_t) override;
  uint32_t GetUint32() override;
  uint64_t GetUint64() override;

  // X external interrupt pending.
  bool meip() {
    return mip_->meip() && (mideleg_->AsUint32() & 0b1000'0000'0000) != 0;
  }
  bool seip() { return mip_->seip(); }
  bool ueip() { return mip_->ueip(); }
  void set_meip(uint32_t value) {
    if ((mideleg_->AsUint32() & 0b1000'0000'0000) != 0) mip_->set_meip(value);
  }
  void set_seip(uint32_t value) { mip_->set_seip(value); }
  void set_ueip(uint32_t value) { mip_->set_ueip(value); }

  // X timer interrupt pending.
  bool mtip() {
    return mip_->mtip() && (mideleg_->AsUint32() & 0b1000'0000) != 0;
  }
  bool stip() { return mip_->stip(); }
  bool utip() { return mip_->utip(); }
  void set_mtip(uint32_t value) {
    if ((mideleg_->AsUint32() & 0b1000'0000) != 0) mip_->set_mtip(value);
  }
  void set_stip(uint32_t value) { mip_->set_stip(value); }
  void set_utip(uint32_t value) { mip_->set_utip(value); }

  // X software interrupt pending.
  bool msip() { return mip_->msip() && (mideleg_->AsUint32() & 0b1000) != 0; }
  bool ssip() { return mip_->ssip(); }
  bool usip() { return mip_->usip(); }
  void set_msip(uint32_t value) {
    if ((mideleg_->AsUint32() & 0b1000) != 0) mip_->set_msip(value);
  }
  void set_ssip(uint32_t value) { mip_->set_ssip(value); }
  void set_usip(uint32_t value) { mip_->set_usip(value); }

 private:
  RiscVMIp* mip_;
  RiscVCsrInterface* mideleg_;
};

// xie - x mode interrupt enable registers.

class RiscVMIe : public RiscVSimpleCsr<uint32_t> {
 public:
  // Read and Write masks.
  static constexpr uint32_t kReadMask = 0b1011'1011'1011;
  static constexpr uint32_t kWriteMask = 0b1011'1011'1011;

  // Disable default constructor.
  RiscVMIe() = delete;
  RiscVMIe(uint32_t initial_value, RiscVState* state);
  ~RiscVMIe() override = default;

  // RiscVSimpleCsr method overrides.
  void Set(uint32_t) override;
  void Set(uint64_t) override;

  // X external interrupt pending.
  bool meie() { return GetterHelper<11, 0b1>(); }
  bool seie() { return GetterHelper<9, 0b1>(); }
  bool ueie() { return GetterHelper<8, 0b1>(); }
  void set_meie(uint32_t value) { SetterHelper<11, 0b1>(value); }
  void set_seie(uint32_t value) { SetterHelper<9, 0b1>(value); }
  void set_ueie(uint32_t value) { SetterHelper<8, 0b1>(value); }

  // X timer interrupt pending.
  bool mtie() { return GetterHelper<7, 0b1>(); }
  bool stie() { return GetterHelper<5, 0b1>(); }
  bool utie() { return GetterHelper<4, 0b1>(); }
  void set_mtie(uint32_t value) { SetterHelper<7, 0b1>(value); }
  void set_stie(uint32_t value) { SetterHelper<5, 0b1>(value); }
  void set_utie(uint32_t value) { SetterHelper<4, 0b1>(value); }

  // X software interrupt pending.
  bool msie() { return GetterHelper<3, 0b1>(); }
  bool ssie() { return GetterHelper<1, 0b1>(); }
  bool usie() { return GetterHelper<0, 0b1>(); }
  void set_msie(uint32_t value) { SetterHelper<3, 0b1>(value); }
  void set_ssie(uint32_t value) { SetterHelper<1, 0b1>(value); }
  void set_usie(uint32_t value) { SetterHelper<0, 0b1>(value); }

 private:
  // Template function to help implement the getters.
  template <int Shift, uint32_t BitMask>
  inline int GetterHelper() {
    return (GetUint32() >> Shift) & BitMask;
  }
  // Template function to help implement the setters.
  template <int Shift, uint32_t BitMask>
  inline void SetterHelper(uint32_t value) {
    uint64_t bit_value = value & BitMask;
    uint64_t new_value =
        (GetUint64() & ~(BitMask << Shift)) | (bit_value << Shift);
    Write(new_value);
  }
};

// The supervisor sie is an interface to the mie.
class RiscVSIe : public RiscVSimpleCsr<uint32_t> {
 public:
  // Read and Write masks.
  static constexpr uint32_t kReadMask = 0b1011'1011'1011;
  static constexpr uint32_t kWriteMask = 0b1011'1011'1011;

  // Disable default constructor.
  RiscVSIe() = delete;
  RiscVSIe(RiscVMIe* mie, RiscVCsrInterface* mideleg, RiscVState* state);
  ~RiscVSIe() override = default;

  // RiscVSimpleCsr method overrides.
  void Set(uint32_t) override;
  void Set(uint64_t) override;
  uint32_t GetUint32() override;
  uint64_t GetUint64() override;

  // X external interrupt pending.
  bool meie() {
    return mie_->meie() && (mideleg_->AsUint32() & 0b1000'0000'0000) != 0;
  }
  bool seie() { return mie_->seie(); }
  bool ueie() { return mie_->ueie(); }
  void set_meie(uint32_t value) {
    if ((mideleg_->AsUint32() & 0b1000'0000'0000) != 0) mie_->set_meie(value);
  }
  void set_seie(uint32_t value) { mie_->set_seie(value); }
  void set_ueie(uint32_t value) { mie_->set_ueie(value); }

  // X timer interrupt pending.
  bool mtie() {
    return mie_->mtie() && (mideleg_->AsUint32() & 0b1000'0000) != 0;
  }
  bool stie() { return mie_->stie(); }
  bool utie() { return mie_->utie(); }
  void set_mtie(uint32_t value) {
    if ((mideleg_->AsUint32() & 0b1000'0000) != 0) mie_->set_mtie(value);
  }
  void set_stie(uint32_t value) { mie_->set_stie(value); }
  void set_utie(uint32_t value) { mie_->set_utie(value); }

  // X software interrupt pending.
  bool msie() { return mie_->msie() && (mideleg_->AsUint32() & 0b1000) != 0; }
  bool ssie() { return mie_->ssie(); }
  bool usie() { return mie_->usie(); }
  void set_msie(uint32_t value) {
    if ((mideleg_->AsUint32() & 0b1000) != 0) mie_->set_msie(value);
  }
  void set_ssie(uint32_t value) { mie_->set_ssie(value); }
  void set_usie(uint32_t value) { mie_->set_usie(value); }

 private:
  RiscVMIe* mie_;
  RiscVCsrInterface* mideleg_;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // RISCV_RISCV_XIP_XIE_H_
