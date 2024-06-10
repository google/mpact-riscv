/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_INSTRUMENTATION_CONTROL_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_INSTRUMENTATION_CONTROL_H_

#include <string>

#include "absl/strings/string_view.h"
#include "mpact/sim/util/memory/memory_use_profiler.h"
#include "re2/re2.h"
#include "riscv/debug_command_shell.h"
#include "riscv/riscv_top.h"

namespace mpact::sim::riscv {

class RiscVInstrumentationControl {
 public:
  RiscVInstrumentationControl(DebugCommandShell *shell, RiscVTop *riscv_top,
                              util::MemoryUseProfiler *mem_profiler);

  bool PerformShellCommand(absl::string_view input,
                           const DebugCommandShell::CoreAccess &core_access,
                           std::string &output);

  std::string Usage() const;

 private:
  DebugCommandShell *shell_;
  RiscVTop *top_ = nullptr;
  util::MemoryUseProfiler *mem_profiler_;
  LazyRE2 pattern_re_;
};

}  // namespace mpact::sim::riscv

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_INSTRUMENTATION_CONTROL_H_
