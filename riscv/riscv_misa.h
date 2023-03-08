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

#include "riscv/riscv_csr.h"

namespace mpact {
namespace sim {
namespace riscv {

// This file defines the RiscVMisa class which implements the misa csr as
// a subclass or RiscVSimpleCsr<uint64_t>
class RiscVState;
enum class RiscVXlen : uint64_t;

class RiscVMIsa : public RiscVSimpleCsr<uint64_t> {
 public:
  // Disable default constructor.
  RiscVMIsa() = delete;
  RiscVMIsa(uint32_t initial_value, RiscVState *state);
  RiscVMIsa(uint64_t initial_value, RiscVState *state);
  ~RiscVMIsa() override = default;

  // RiscVSimpleCsr method overrides.
  uint32_t AsUint32() override;
  uint32_t GetUint32() override;
  void Set(uint32_t) override;
  void Set(uint64_t) override;

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
