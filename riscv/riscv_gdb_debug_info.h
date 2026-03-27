// Copyright 2026 Google LLC
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

// This file contains the definition of the RiscVGdbDebugInfo class. The
// RiscVGdbDebugInfo class is used to provide information about the registers
// and target XML to gdbserver.

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_GDB_DEBUG_INFO_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_GDB_DEBUG_INFO_H_

#include <string>
#include <string_view>

#include "absl/strings/string_view.h"
#include "mpact/sim/generic/debug_info.h"

namespace mpact::sim::riscv {

enum class RiscVGdbRegisterEnum : int {
  // Integer registers.
  kGprFirst = 0,
  kGprX0 = kGprFirst,
  kGprX1,
  kGprX2,
  kGprX3,
  kGprX4,
  kGprX5,
  kGprX6,
  kGprX7,
  kGprX8,
  kGprX9,
  kGprX10,
  kGprX11,
  kGprX12,
  kGprX13,
  kGprX14,
  kGprX15,
  kGprX16,
  kGprX17,
  kGprX18,
  kGprX19,
  kGprX20,
  kGprX21,
  kGprX22,
  kGprX23,
  kGprX24,
  kGprX25,
  kGprX26,
  kGprX27,
  kGprX28,
  kGprX29,
  kGprX30,
  kGprX31,
  kGprPc,
  kGprLast = kGprPc,
  // Floating point registers.
  kFprFirst = kGprLast + 1,
  kFprF0 = kFprFirst,
  kFprF1,
  kFprF2,
  kFprF3,
  kFprF4,
  kFprF5,
  kFprF6,
  kFprF7,
  kFprF8,
  kFprF9,
  kFprF10,
  kFprF11,
  kFprF12,
  kFprF13,
  kFprF14,
  kFprF15,
  kFprF16,
  kFprF17,
  kFprF18,
  kFprF19,
  kFprF20,
  kFprF21,
  kFprF22,
  kFprF23,
  kFprF24,
  kFprF25,
  kFprF26,
  kFprF27,
  kFprF28,
  kFprF29,
  kFprF30,
  kFprF31,
  kFprLast = kFprF31,
  kFprFcsr,
  // Vector registers.
  kVprFirst = kFprFcsr + 1,
  kVprV0 = kVprFirst,
  kVprV1,
  kVprV2,
  kVprV3,
  kVprV4,
  kVprV5,
  kVprV6,
  kVprV7,
  kVprV8,
  kVprV9,
  kVprV10,
  kVprV11,
  kVprV12,
  kVprV13,
  kVprV14,
  kVprV15,
  kVprV16,
  kVprV17,
  kVprV18,
  kVprV19,
  kVprV20,
  kVprV21,
  kVprV22,
  kVprV23,
  kVprV24,
  kVprV25,
  kVprV26,
  kVprV27,
  kVprV28,
  kVprV29,
  kVprV30,
  kVprV31,
  kVprLast = kVprV31,
  kVprVstart,
  kVprVxsat,
  kVprVxrm,
  kVprVcsr,
  kVprVl,
  kVprVtype,
  kVprVlenb,
};

class RiscVGdbDebugInfo : public generic::DebugInfo {
 public:
  using DebugRegisterMap = generic::DebugInfo::DebugRegisterMap;

  static RiscVGdbDebugInfo* Instance(int gpr_width, int fp_width,
                                     int vec_width);

  const DebugRegisterMap& debug_register_map() const override {
    return debug_register_map_;
  }
  int GetFirstGpr() const override {
    return static_cast<int>(RiscVGdbRegisterEnum::kGprFirst);
  };
  int GetLastGpr() const override {
    return static_cast<int>(RiscVGdbRegisterEnum::kGprLast);
  }
  int GetGprWidth() const override { return gpr_width_; };
  // For now assume that all registers have the same width as the GPRs.
  int GetRegisterByteWidth(int register_number) const override {
    return gpr_width_ / 8;
  }
  std::string_view GetLLDBHostInfo() const override {
    return absl::string_view(host_info_);
  }
  std::string_view GetGdbTargetXml() const override {
    return absl::string_view(gdb_target_xml_);
  }

 private:
  RiscVGdbDebugInfo(int gpr_width, int fp_width, int vec_width);
  int gpr_width_;
  int fp_width_;
  int vec_width_;
  std::string host_info_;
  DebugRegisterMap debug_register_map_;
  // "Escaped" GDB target XML string.
  std::string gdb_target_xml_;
};

}  // namespace mpact::sim::riscv

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_GDB_DEBUG_INFO_H_
