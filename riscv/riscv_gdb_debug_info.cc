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

#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "mpact/sim/generic/type_helpers.h"

namespace {

std::string EscapeString(std::string_view str) {
  std::string escaped_string;
  escaped_string.reserve(str.size() * 2);
  for (char c : str) {
    if ((c == '$') || (c == '#') || (c == '*') || (c == '}')) {
      escaped_string.push_back('}');
      escaped_string.push_back(c ^ 0x20);
    } else {
      escaped_string.push_back(c);
    }
  }
  return escaped_string;
}

}  // namespace

namespace mpact::sim::riscv {

using ::mpact::sim::generic::operator*;  // NOLINT

RiscVGdbDebugInfo* RiscVGdbDebugInfo::Instance(int gpr_width, int fp_width,
                                               int vec_width) {
  static absl::flat_hash_map<int, RiscVGdbDebugInfo*> instance_map;
  // Check for valid widths.
  if (gpr_width != 32 && gpr_width != 64) return nullptr;
  if (fp_width != 0 && fp_width != 32 && fp_width != 64) return nullptr;
  if (vec_width != 0 && vec_width != 128 && vec_width != 256 &&
      vec_width != 512 && vec_width != 1024) {
    return nullptr;
  }
  int key = gpr_width | (fp_width << 16) | (vec_width << 24);
  auto it = instance_map.find(key);
  if (it != instance_map.end()) {
    return it->second;
  }
  RiscVGdbDebugInfo* instance =
      new RiscVGdbDebugInfo(gpr_width, fp_width, vec_width);
  instance_map[key] = instance;
  return instance;
}

RiscVGdbDebugInfo::RiscVGdbDebugInfo(int gpr_width, int fp_width, int vec_width)
    : gpr_width_(gpr_width), fp_width_(fp_width), vec_width_(vec_width) {
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
  if (fp_width != 0) {
    // Floating point registers.
    for (int i = *RiscVGdbRegisterEnum::kFprFirst;
         i <= *RiscVGdbRegisterEnum::kFprLast; ++i) {
      debug_register_map_.emplace(
          i, absl::StrCat("f", i - *RiscVGdbRegisterEnum::kFprFirst));
    }
    // CSRs.
    debug_register_map_.emplace(*RiscVGdbRegisterEnum::kFprFcsr, "fcsr");
  }
  if (vec_width != 0) {
    // Vector control registers.
    debug_register_map_.emplace(*RiscVGdbRegisterEnum::kVprVstart, "vstart");
    debug_register_map_.emplace(*RiscVGdbRegisterEnum::kVprVxsat, "vxsat");
    debug_register_map_.emplace(*RiscVGdbRegisterEnum::kVprVxrm, "vxrm");
    debug_register_map_.emplace(*RiscVGdbRegisterEnum::kVprVcsr, "vcsr");
    debug_register_map_.emplace(*RiscVGdbRegisterEnum::kVprVl, "vl");
    debug_register_map_.emplace(*RiscVGdbRegisterEnum::kVprVtype, "vtype");
    debug_register_map_.emplace(*RiscVGdbRegisterEnum::kVprVlenb, "vlenb");
  }
  // Construct the gdb target xml string.
  gdb_target_xml_ = R"xml(<?xml version="1.0"?>
<!DOCTYPE target SYSTEM "gdb-target.dtd">
<target version="1.0">
  <architecture>riscv</architecture>)xml";
  absl::StrAppend(&gdb_target_xml_,
                  "\n  <feature name=\"org.gnu.gdb.riscv.cpu\">\n");
  // GPR registers.
  int regnum = 0;
  for (int i = 0; i < 32; ++i) {
    absl::StrAppend(
        &gdb_target_xml_,
        absl::StrFormat("    <reg name=\"x%d\" bitsize=\"%d\" regnum=\"%d\" "
                        "type=\"uint%d\" group=\"general\"/>\n",
                        i, gpr_width_, regnum++, gpr_width_));
  }
  absl::StrAppend(
      &gdb_target_xml_,
      absl::StrFormat("    <reg name=\"pc\" bitsize=\"%d\" regnum=\"%d\" "
                      "type=\"code_ptr\" group=\"general\"/>\n",
                      gpr_width_, regnum++));
  absl::StrAppend(&gdb_target_xml_, "  </feature>\n");

  // FP registers.
  if (fp_width != 0) {
    absl::StrAppend(&gdb_target_xml_,
                    "  <feature name=\"org.gnu.gdb.riscv.fpu\">\n");
    std::string fp_type = fp_width == 64 ? "ieee_double" : "ieee_float";
    for (int i = 0; i < 32; ++i) {
      absl::StrAppend(
          &gdb_target_xml_,
          absl::StrFormat("    <reg name=\"f%d\" bitsize=\"%d\" "
                          "regnum=\"%d\" type=\"%s\" group=\"float\"/>\n",
                          i, fp_width_, regnum++, fp_type));
    }
    absl::StrAppend(
        &gdb_target_xml_,
        absl::StrFormat("    <reg name=\"fcsr\" bitsize=\"%d\" regnum=\"%d\" "
                        "type=\"uint32\" group=\"float\"/>\n",
                        32, regnum++));
    absl::StrAppend(&gdb_target_xml_, "  </feature>\n");
  }
  // Vector registers.
  if (vec_width != 0) {
    absl::StrAppend(&gdb_target_xml_,
                    "  <feature name=\"org.gnu.gdb.riscv.vector\">\n");
    // Add vector data type.
    absl::StrAppend(
        &gdb_target_xml_,
        absl::StrFormat(
            "    <vector id=\"v%d\" type=\"uint8\" count=\"%d\"/>\n",
            vec_width_, vec_width_ / 8));
    for (int i = 0; i < 32; ++i) {
      absl::StrAppend(&gdb_target_xml_,
                      absl::StrFormat("    <reg name=\"v%d\" bitsize=\"%d\" "
                                      "regnum=\"%d\" type=\"v%d\" "
                                      "group=\"vector\"/>\n",
                                      i, vec_width_, regnum++, vec_width_));
    }
    absl::StrAppend(&gdb_target_xml_,
                    "    <reg name=\"vstart\" bitsize=\"32\" type=\"uint32\" "
                    "group=\"vector\"/>\n");
    absl::StrAppend(&gdb_target_xml_,
                    "    <reg name=\"vxsat\" bitsize=\"32\" type=\"uint32\" "
                    "group=\"vector\"/>\n");
    absl::StrAppend(&gdb_target_xml_,
                    "    <reg name=\"vxrm\" bitsize=\"32\" type=\"uint32\" "
                    "group=\"vector\"/>\n");
    absl::StrAppend(&gdb_target_xml_,
                    "    <reg name=\"vcsr\" bitsize=\"32\" type=\"uint32\" "
                    "group=\"vector\"/>\n");
    absl::StrAppend(&gdb_target_xml_,
                    "    <reg name=\"vl\" bitsize=\"32\" type=\"uint32\" "
                    "group=\"vector\"/>\n");
    absl::StrAppend(&gdb_target_xml_,
                    "    <reg name=\"vtype\" bitsize=\"32\" type=\"uint32\" "
                    "group=\"vector\"/>\n");
    absl::StrAppend(&gdb_target_xml_,
                    "    <reg name=\"vlenb\" bitsize=\"32\" type=\"uint32\" "
                    "group=\"vector\"/>\n");
    absl::StrAppend(&gdb_target_xml_, "  </feature>\n");
  }
  absl::StrAppend(&gdb_target_xml_, "</target>\n");
  // Escape the string for GDB
  gdb_target_xml_ = EscapeString(gdb_target_xml_);
}

}  // namespace mpact::sim::riscv
