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

#ifndef RISCV_RISCV_DEBUG_INFO_H_
#define RISCV_RISCV_DEBUG_INFO_H_

#include <string>

#include "absl/container/flat_hash_map.h"

namespace mpact {
namespace sim {
namespace riscv {

// Enumeration of the RiscV debug ids for accessible registers.
enum class DebugRegisterEnum : uint32_t {
  // CSRs.
  // Program counter.
  kPc = 0x07b1,

  // Integer registers.
  kX0 = 0x1000,
  kX1,
  kX2,
  kX3,
  kX4,
  kX5,
  kX6,
  kX7,
  kX8,
  kX9,
  kX10,
  kX11,
  kX12,
  kX13,
  kX14,
  kX15,
  kX16,
  kX17,
  kX18,
  kX19,
  kX20,
  kX21,
  kX22,
  kX23,
  kX24,
  kX25,
  kX26,
  kX27,
  kX28,
  kX29,
  kX30,
  kX31,

  // Floating point registers.
  kF0 = 0x1020,
  kF1,
  kF2,
  kF3,
  kF4,
  kF5,
  kF6,
  kF7,
  kF8,
  kF9,
  kF10,
  kF11,
  kF12,
  kF13,
  kF14,
  kF15,
  kF16,
  kF17,
  kF18,
  kF19,
  kF20,
  kF21,
  kF22,
  kF23,
  kF24,
  kF25,
  kF26,
  kF27,
  kF28,
  kF29,
  kF30,
  kF31,

  // Vector registers.
  kV0 = 0x1020,
  kV1,
  kV2,
  kV3,
  kV4,
  kV5,
  kV6,
  kV7,
  kV8,
  kV9,
  kV10,
  kV11,
  kV12,
  kV13,
  kV14,
  kV15,
  kV16,
  kV17,
  kV18,
  kV19,
  kV20,
  kV21,
  kV22,
  kV23,
  kV24,
  kV25,
  kV26,
  kV27,
  kV28,
  kV29,
  kV30,
  kV31,
};

// Singleton class used to store RiscV debug register ids.
class RiscVDebugInfo {
 public:
  using DebugRegisterMap = absl::flat_hash_map<uint32_t, std::string>;

  static RiscVDebugInfo* Instance();

  const DebugRegisterMap& debug_register_map() const {
    return debug_register_map_;
  }

 private:
  RiscVDebugInfo();
  DebugRegisterMap debug_register_map_;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // RISCV_RISCV_DEBUG_INFO_H_
