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

#ifndef MPACT_RISCV_RISCV_RISCV_MISA_H_
#define MPACT_RISCV_RISCV_RISCV_MISA_H_

#include <cstdint>

#include "mpact/sim/generic/arch_state.h"
#include "riscv/riscv_csr.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::ArchState;

// This file defines the RiscVMisa class which implements the misa csr as
// a subclass or RiscVSimpleCsr<uint64_t>
class RiscVState;
enum class RiscVXlen : uint64_t;

class RiscVMIsa : public RiscVSimpleCsr<uint64_t> {
 public:
  // Disable default constructor.
  RiscVMIsa() = delete;
  RiscVMIsa(uint32_t initial_value, ArchState* state);
  RiscVMIsa(uint64_t initial_value, ArchState* state);
  ~RiscVMIsa() override = default;

  // RiscVSimpleCsr method overrides.
  uint32_t AsUint32() override;
  // Note, GetUInt32() compresses the full 64 bit stored version of Misa into
  // the 32 bit view. Use GetUInt64 to get the full uncompressed view.
  uint32_t GetUint32() override;
  // Overwriting Write methods to disable writing to the register. However,
  // the value can be changed programmatically by using the Set method.
  void Write(uint32_t) override;
  void Write(uint64_t) override;

  bool HasSupervisorMode();
  bool HasUserMode();

  // Returns the xlen.
  RiscVXlen Xlen();

 private:
  void Validate();

  uint32_t read_mask_32_;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_MISA_H_
