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

#include "riscv/riscv_gdb_debug_info.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "mpact/sim/generic/type_helpers.h"

namespace mpact::sim::riscv {

using ::mpact::sim::generic::operator*;  // NOLINT

RiscVGdbDebugInfo* RiscVGdbDebugInfo::Instance(int gpr_width) {
  if (gpr_width != 32 && gpr_width != 64) {
    return nullptr;
  }
  static RiscVGdbDebugInfo* instance32 = nullptr;
  static RiscVGdbDebugInfo* instance64 = nullptr;
  if (gpr_width == 32) {
    if (instance32 == nullptr) {
      instance32 = new RiscVGdbDebugInfo(gpr_width);
    }
    return instance32;
  } else {
    if (instance64 == nullptr) {
      instance64 = new RiscVGdbDebugInfo(gpr_width);
    }
    return instance64;
  }
  return nullptr;
}

RiscVGdbDebugInfo::RiscVGdbDebugInfo(int gpr_width) : gpr_width_(gpr_width) {
  host_info_ = absl::StrFormat(
      "triple:riscv%d-unknown-elf;"
      "endian:little;"
      "ptrsize:%d;"
      "vendor:riscv;",
      gpr_width_, gpr_width_ / 8);
  // PC register.
  debug_register_map_.emplace(*RiscVGdbRegisterEnum::kGprPc, "pc");
  // Integer registers.
  for (int i = *RiscVGdbRegisterEnum::kGprX0;
       i <= *RiscVGdbRegisterEnum::kGprX31; ++i) {
    debug_register_map_.emplace(
        i, absl::StrCat("x", i - *RiscVGdbRegisterEnum::kGprX0));
  }
  // Floating point registers.
  for (int i = *RiscVGdbRegisterEnum::kFprFirst;
       i <= *RiscVGdbRegisterEnum::kFprLast; ++i) {
    debug_register_map_.emplace(
        i, absl::StrCat("f", i - *RiscVGdbRegisterEnum::kFprFirst));
  }
  // CSRs.
  debug_register_map_.emplace(*RiscVGdbRegisterEnum::kFprFcsr, "fcsr");
  debug_register_map_.emplace(*RiscVGdbRegisterEnum::kVprVstart, "vstart");
  debug_register_map_.emplace(*RiscVGdbRegisterEnum::kVprVxsat, "vxsat");
  debug_register_map_.emplace(*RiscVGdbRegisterEnum::kVprVxrm, "vxrm");
  debug_register_map_.emplace(*RiscVGdbRegisterEnum::kVprVcsr, "vcsr");
  debug_register_map_.emplace(*RiscVGdbRegisterEnum::kVprVl, "vl");
  debug_register_map_.emplace(*RiscVGdbRegisterEnum::kVprVType, "vtype");
  debug_register_map_.emplace(*RiscVGdbRegisterEnum::kVprVlenb, "vlenb");
  // xml
  if (gpr_width == 32) {
    gdb_target_xml_ =
#include "riscv/riscv_gdb_debug_info_32.inc"
  } else {
    gdb_target_xml_ =
#include "riscv/riscv_gdb_debug_info_64.inc"
  }
}

}  // namespace mpact::sim::riscv
