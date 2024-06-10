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

#include "riscv/riscv_instrumentation_control.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mpact/sim/util/memory/memory_use_profiler.h"
#include "re2/re2.h"
#include "riscv/debug_command_shell.h"
#include "riscv/riscv_top.h"
#include "riscv/stoull_wrapper.h"

namespace mpact::sim::riscv {

using ::mpact::sim::util::MemoryUseProfiler;

RiscVInstrumentationControl::RiscVInstrumentationControl(
    DebugCommandShell *shell, RiscVTop *riscv_top,
    MemoryUseProfiler *mem_profiler)
    : shell_(shell),
      top_(riscv_top),
      mem_profiler_(mem_profiler),
      pattern_re_{
          R"(\s*stats\s+(enable|disable)\s+(counters|iprofile|mprofile|all)(?:\s+(\w+))?\s*)"} {
}

bool RiscVInstrumentationControl::PerformShellCommand(
    absl::string_view input, const DebugCommandShell::CoreAccess &core_access,
    std::string &output) {
  std::string cmd;
  std::string what;
  std::string where;
  absl::AnyInvocable<void(uint64_t, int)> function;
  if (!RE2::FullMatch(input, *pattern_re_, &cmd, &what, &where)) return false;
  // See what the pattern matched and set up an appropriate lambda.
  if (cmd == "enable") {
    if (what == "counters") {
      function = [this](uint64_t, int) { top_->EnableStatistics(); };
    } else if (what == "iprofile") {
      function = [this](uint64_t, int) {
        top_->counter_pc()->SetIsEnabled(true);
      };
    } else if (what == "mprofile") {
      function = [this](uint64_t, int) { mem_profiler_->set_is_enabled(true); };
    } else if (what == "all") {
      function = [this](uint64_t, int) {
        top_->EnableStatistics();
        top_->counter_pc()->SetIsEnabled(true);
        mem_profiler_->set_is_enabled(true);
      };
    } else {
      output = absl::StrCat("Error: Unknown instrumentation: '", what, "'");
      return true;
    }
  } else if (cmd == "disable") {
    if (what == "counters") {
      function = [this](uint64_t, int) { top_->DisableStatistics(); };
    } else if (what == "iprofile") {
      function = [this](uint64_t, int) {
        top_->counter_pc()->SetIsEnabled(false);
      };
    } else if (what == "mprofile") {
      function = [this](uint64_t, int) {
        mem_profiler_->set_is_enabled(false);
      };
    } else if (what == "all") {
      function = [this](uint64_t, int) {
        top_->DisableStatistics();
        top_->counter_pc()->SetIsEnabled(false);
        mem_profiler_->set_is_enabled(false);
      };
    } else {
      output = absl::StrCat("Error: Unknown instrumentation: '", what, "'");
      return true;
    }
  }
  // If 'where' is empty, we just execute the callable. It is not attached
  // to any instruction.
  if (where.empty()) {
    function(0, 0);
    return true;
  }
  // At this point we need to set the function to be executed upon execution
  // of an instruction a specified in 'where'.
  size_t index;
  uint64_t address = 0;
  // Attempt to convert to a number.
  auto convert_result = riscv::internal::stoull(where, &index, 0);
  // If successful and the entire string was consumed, the number is good and
  // we use that as the address.
  if (convert_result.ok() && (index >= where.size())) {
    address = convert_result.value();
  } else {
    // If it's out of range, signal an error.
    if (convert_result.status().code() == absl::StatusCode::kOutOfRange) {
      output = absl::StrCat("Error: ", where, " is out of range");
      return true;
    }
    // Let's see if it is a symbol.
    auto *loader = core_access.loader_getter();
    if (loader == nullptr) {
      output = "Error: cannot perform symbol lookup";
      return true;
    }
    auto result = loader->GetSymbol(where);
    if (!result.ok()) {
      output = absl::StrCat("Error: symbol ", where, " not found");
      return true;
    }
    address = result.value().first;
  }
  // Now we set an action for the function.
  std::string action_name = absl::StrCat(cmd, "-", what, "-", where);
  auto status =
      shell_->SetActionPoint(address, action_name, std::move(function));
  if (!status.ok()) {
    output = absl::StrCat("Error: ", status.message());
  }
  return true;
}

std::string RiscVInstrumentationControl::Usage() const {
  return
      R"raw(
  stats enable (counters|iprofile|mprofile|all) [VALUE|SYMBOL]
                                   - enable counters, iprofile, mprofile, or all
                                     when executing instruction at address VALUE
                                     or value of SYMBOL, or immediately if
                                     neither is specified.
  stats disable (counters|iprofile|mprofile|all) [VALUE|SYMBOL]
                                   - disable counters, iprofile, mprofile, or all
                                     when executing instruction at address VALUE
                                     or value of SYMBOL, or immediately if
                                     neither is specified.
    )raw";
}

}  // namespace mpact::sim::riscv
