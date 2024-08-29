// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_ENCODING_COMMON_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_ENCODING_COMMON_H_

#include <cstdint>

#include "mpact/sim/generic/simple_resource.h"

namespace mpact {
namespace sim {
namespace riscv {

class RiscVState;

// This class provides a common interface for accessing the state and
// instruction word for the RiscVEncoding classes (scalar, vector, vector
// + fp).

class RiscVEncodingCommon {
 public:
  virtual ~RiscVEncodingCommon() = default;

  // Accessors.
  virtual RiscVState *state() const = 0;
  virtual generic::SimpleResourcePool *resource_pool() = 0;
  virtual uint32_t inst_word() const = 0;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_ENCODING_COMMON_H_
