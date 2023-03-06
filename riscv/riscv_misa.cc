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

#include "riscv/riscv_misa.h"

#include "absl/log/log.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_state.h"

namespace mpact {
namespace sim {
namespace riscv {

// Helper functions to convert between 32 and 64 bit views of misa.
static inline uint64_t StretchMisa32(uint32_t value) {
  uint64_t value64 = static_cast<uint64_t>(value);
  value64 = ((value64 & 0xc000'0000) << 32) | (value64 & 0x03ff'ffff);
  return value64;
}

static inline uint32_t CompressMisa64(uint64_t value) {
  uint32_t value32 = ((value >> 32) & 0xc000'0000) | (value & 0x3fff'ffff);
  return value32;
}

// Constructors.
RiscVMIsa::RiscVMIsa(uint32_t initial_value, RiscVState *state)
    : RiscVMIsa(StretchMisa32(initial_value), state) {}

RiscVMIsa::RiscVMIsa(uint64_t initial_value, RiscVState *state)
    : RiscVSimpleCsr<uint64_t>("misa", RiscVCsrEnum::kMIsa, initial_value,
                               /*read_mask*/ 0xc000'0000'03ff'ffffULL,
                               /*write_mask*/ 0, state) {
  read_mask_32_ = CompressMisa64(read_mask());
}

// Override implementations.
uint32_t RiscVMIsa::AsUint32() {
  // The value is stored as if xlen is 64, so adjust the layout.
  return GetUint32() & read_mask_32_;
}

uint32_t RiscVMIsa::GetUint32() {
  // The value is stored as if xlen is 64, so adjust the layout.
  return CompressMisa64(GetUint64());
}

void RiscVMIsa::Set(uint32_t value) {
  /* misa is not writable */
  LOG(WARNING) << "misa is not writable - write ignored";
}

void RiscVMIsa::Set(uint64_t value) {
  /* misa is not writable */
  LOG(WARNING) << "misa is not writable - write ignored";
}

bool RiscVMIsa::HasSupervisorMode() {
  return GetUint32() & *IsaExtension::kSupervisorMode;
}

bool RiscVMIsa::HasUserMode() { return true; }

RiscVXlen RiscVMIsa::Xlen() {
  switch ((GetUint64() >> 62) & 0xb11) {
    case 0b01:
      return RiscVXlen::RV32;
    case 0b10:
      return RiscVXlen::RV64;
    default:
      return RiscVXlen::RVUnknown;
  }
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
