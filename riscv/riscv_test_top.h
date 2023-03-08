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

#ifndef MPACT_RISCV_RISCV_RISCV_TEST_TOP_H_
#define MPACT_RISCV_RISCV_RISCV_TEST_TOP_H_

namespace mpact {
namespace sim {
namespace riscv {

class RiscVTestTop : public RiscVTop {
 public:
  RiscVTestTop(std::string name, RiscVXlen xlen);
  RiscVTestTop(std::string name, util::MemoryInterface *memory,
               RRiscVXlen xlen);
  RiscVTestTop(std::string name, util::MemoryInterface *memory,
               util::AtomicMemoryOpInterface *atomic_memory, RiscVXlen xlen);
  ~RiscVTestTop() override = default;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_TEST_TOP_H_
