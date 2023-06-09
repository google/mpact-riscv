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

#ifndef MPACT_RISCV_RISCV_STOULL_WRAPPER_H_
#define MPACT_RISCV_RISCV_STOULL_WRAPPER_H_

#include <string>

#include "absl/status/statusor.h"

// This is a wrapper to isolate any exceptions generated by stoull and prevent
// them from leaking out.

namespace mpact {
namespace sim {
namespace riscv {
namespace internal {

absl::StatusOr<uint64_t> stoull(const std::string str, size_t *indx = nullptr,
                                int base = 10);

}  // namespace internal
}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_STOULL_WRAPPER_H_
