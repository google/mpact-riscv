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

#include "riscv/debug_command_shell.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <istream>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mpact/sim/generic/core_debug_interface.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/type_helpers.h"
#include "re2/re2.h"
#include "riscv/riscv_debug_interface.h"
#include "riscv/riscv_top.h"
#include "riscv/stoull_wrapper.h"

namespace mpact {
namespace sim {
namespace riscv {

using HaltReason = ::mpact::sim::generic::CoreDebugInterface::HaltReason;
using ::mpact::sim::generic::operator*;  // NOLINT: used below (clang error).

// The constructor initializes all the regular expressions and the help string.
DebugCommandShell::DebugCommandShell()
    : quit_re_{R"(\s*quit\s*)"},
      core_re_{R"(\s*core\s+(\d+)\s*)"},
      run_re_{R"(\s*(?:run|c)\s*)"},
      run_free_re_{R"(\s*run\s+free\s*)"},
      wait_re_{R"(\s*wait\s*)"},
      step_1_re_{R"(\s*step\s*)"},
      step_n_re_{R"(\s*step\s+(\d+)\s*)"},
      halt_re_{R"(\s*halt\s*)"},
      read_reg_re_{R"(\s*reg\s+get\s+(\$?(\w|\.)+)(\s+[foduxX]\d+)?\s*)"},
      read_reg2_re_{R"(\s*reg\s+(\$?(\w|\.)+)(\s+[foduxX]\d+)?\s*)"},
      write_reg_re_{R"(\s*reg\s+set\s+(\$?\w+)\s+(\w+)\s*)"},
      rd_vreg_re_{R"(\s*vreg(?:\s+get)?\s+(\$?\w+)(?:(?:\s*\:(\d+))?)"
                  R"((?:\s+(?:([oduxX])(8|16|32|64))?)?)?\s*)"},
      wr_vreg_re_{R"(\s*vreg\s+set\s+(\$?\w+)\s*\:(\d+))"
                  R"(\s+([oduxX])(8|16|32|64)\s+(\w+)\s*)"},
      read_mem_re_{R"(\s*mem\s+get\s+(\w+)(?:\s+([foduxX]\d+|i)?)?\s*)"},
      read_mem2_re_{R"(\s*mem\s+(\w+)(?:\s+([foduxX]\d+|i)?)?\s*)"},
      write_mem_re_{R"(\s*mem\s+set\s+(\w+)\s+([oduxX]\d+)?\s+(\w+)\s*)"},
      set_break_re_{R"(\s*break\s+set\s+(\$?\w+)\s*)"},
      set_break2_re_{R"(\s*break\s+(\$?\w+)\s*)"},
      set_break_n_re_{R"(\s*break\s+(set\s+)?\#(\d+)\s*)"},
      list_break_re_{R"(\s*break\s*)"},
      clear_break_n_re_{R"(\s*break\s+clear\s+\#(\d+)\s*)"},
      clear_break_re_{R"(\s*break\s+clear\s+(\$?\w+)\s*)"},
      clear_all_break_re_{R"(\s*break\s+clear-all\s*)"},
      set_watch_re_{R"(\s*watch\s+set\s+(\w+)\s+(\w+)(\s+r|\s+w|\s+rw)?\s*)"},
      set_watch2_re_{R"(\s*watch\s+(\w+)\s+(\w+)(\s+r|\s+w|\s+rw)?\s*)"},
      set_watch_n_re_{R"(\s*watch\s+(set\s+)?\#(\d+)\s*)"},
      list_watch_re_{R"(\s*watch\s*)"},
      clear_watch_re_{R"(\s*watch\s+clear\s+(\w+)(\s+r|\s+w|\s+rw)?\s*)"},
      clear_watch_n_re_{R"(\s*watch\s+clear\s+\#(\d+)\s*)"},
      clear_all_watch_re_{R"(\s*watch\s+clear-all\s*)"},
      list_action_re_{R"(\s*action\s*)"},
      enable_action_n_re_{R"(\s*action\s+enable\s+\*(\d+)\s*)"},
      disable_action_n_re_{R"(\s*action\s+disable\s+\*(\d+)\s*)"},
      clear_action_n_re_{R"(\s*action\s+clear\s+\*(\d+)\s*)"},
      clear_all_action_re_{R"(\s*action\s+clear-all\s*)"},
      branch_trace_re_{R"(\s*branch-trace\s*)"},
      exec_re_{R"(\s*exec\s+(.+)\s*)"},
      empty_re_{R"(\s*(?:\#.*)?)"},
      help_re_{R"(\s*help\s*)"} {
  help_message_ = R"raw(  quit                             - exit command shell.
  core [N]                         - direct subsequent commands to core N
                                     (default: 0).
  run                              - run program from current pc until a
                                     breakpoint or exit. Wait until halted.
  run free                         - run program in background from current pc
                                     until breakpoint or exit.
  wait                             - wait for any free run to complete.
  step [N]                         - step [N] instructions (default: 1).
  halt                             - halt a running program.
  reg get NAME [FORMAT]            - get the value or register NAME.
  reg NAME [FORMAT]                - get the value of register NAME.
  reg set NAME VALUE               - set register NAME to VALUE.
  reg set NAME SYMBOL              - set register NAME to value of SYMBOL.
  vreg get NAME[:INDEX] [FORMAT]   - get the value of register NAME as a vector
                                     of elements according to FORMAT. The format
                                     is a letter (o, d u x, or X) followed by
                                     width (8, 16, 32, 64). The default format
                                     is x32. INDEX may be specified as a number
                                     to select a specific vector element.
  vreg set NAME[:INDEX] [FORMAT]   - set the value of vector register NAME to
      VALUE                          the given value. If INDEX is specified,
                                     only the given element is updated. If INDEX
                                     is omitted, the value will be written to
                                     the 0th element in the vector (according)
                                     to FORMAT, and the rest of the vector is
                                     cleared.
  mem get VALUE [FORMAT]           - get memory from location VALUE according to
                                     format. The format is a letter (o, d, u, x,
                                     or X) followed by width (8, 16, 32, 64).
                                     The default format is x32.
  mem get SYMBOL [FORMAT]          - get memory from location SYMBOL and format
                                     according to FORMAT (see above).
  mem SYMBOL [FORMAT]              - get memory from location SYMBOL and format
                                     according to FORMAT (see above).
  mem set VALUE [FORMAT] VALUE     - set memory at location VALUE(1) to VALUE(2)
                                     according to FORMAT. Default format is x32.
  mem set SYMBOL [FORMAT] VALUE    - set memory at location SYMBOL to VALUE
                                     according to FORMAT. Default format is x32.
  break [set] VALUE                - set breakpoint at address VALUE.
  break [set] SYMBOL               - set breakpoint at value of SYMBOL.
  break set #<N>                   - reactivate breakpoint index N.
  break #<N>                       - reactivate breakpoint index N.
  break clear VALUE                - clear breakpoint at address VALUE.
  break clear SYMBOL               - clear breakpoint at value of SYMBOL.
  break clear #<N>                 - clear breakpoint index N.
  break clear-all                  - remove all breakpoints.
  break                            - list breakpoints.
  watch [set] VALUE len [r|w|rw]   - set watchpoint at value (read, write, or
                                     readwrite) - default is write.
  watch [set] SYMBOL len [r|w|rw]  - set watchpoint at value (read, write, or
                                     readwrite) - default is write.
  watch set #<N>                   - reactivate watchpoint index N.
  watch clear VALUE [r|w|rw]       - clear watchpoint at value (read, write, or
                                     readwrite) - default is write.
  watch clear SYMBOL [r|w|rw]      - clear watchpoint at symbol (read, write or
                                     readwrite) - default is write.
  watch clear #<N>                 - clear watchpoint index N.
  watch clear-all                  - remove all watchpoints.
  watch                            - list watchpoints.
  action enable #<N>               - enable action point with index N.
  action disable #<N>              - disable action point with index N.
  action clear #<N>                - clear action point with index N.
  action clear-all                 - clear all action points.
  action                           - list action points.
  branch-trace                     - list the control flow change (includes
                                     interrupts) w/out repetitions due to loops.
  exec    NAME                     - load commands from file 'NAME' and execute
                                     each line as a command. Lines starting with
                                     a '#' are treated as comments.
  help                             - display this message.

  Special register names:
  $all                             - core set of registers (e.g., reg $all).
)raw";
}

void DebugCommandShell::AddCore(const CoreAccess& core_access) {
  core_access_.push_back(core_access);
  core_action_point_id_.push_back(0);
  core_action_point_info_.emplace_back();
}

void DebugCommandShell::AddCores(const std::vector<CoreAccess>& core_access) {
  for (auto& core : core_access) {
    AddCore(core);
  }
}
// NOLINTBEGIN(readability/fn_size)
void DebugCommandShell::Run(std::istream& is, std::ostream& os) {
  // Assumes the max linesize is 512.
  command_streams_.push_back(&is);
  constexpr int kLineSize = 512;
  char line[kLineSize];
  std::string previous_line;
  current_core_ = 0;
  absl::string_view line_view;
  bool halt_reason = false;
  while (true) {
    // Prompt and read in the next command.
    auto pc_result =
        core_access_[current_core_].debug_interface->ReadRegister("pc");
    std::string prompt;
    if (halt_reason) {
      halt_reason = false;
      auto result =
          core_access_[current_core_].debug_interface->GetLastHaltReason();
      if (result.ok()) {
        switch (result.value()) {
          case *HaltReason::kSoftwareBreakpoint:
            absl::StrAppend(&prompt, "Stopped at software breakpoint\n");
            break;
          case *HaltReason::kUserRequest:
            absl::StrAppend(&prompt, "Stopped at user request\n");
            break;
          case *HaltReason::kDataWatchPoint:
            absl::StrAppend(&prompt, "Stopped at data watchpoint\n");
            break;
          case *HaltReason::kProgramDone:
            absl::StrAppend(&prompt, "Program done\n");
            break;
          default:
            if ((result.value() >= *HaltReason::kUserSpecifiedMin) &&
                (result.value() <= *HaltReason::kUserSpecifiedMax)) {
              absl::StrAppend(&prompt, "Stopped for custom halt reason\n");
            }
            break;
        }
      }
    }
    if (pc_result.ok()) {
      auto* loader = core_access_[current_core_].loader_getter();
      if (loader != nullptr) {
        auto fcn_result = loader->GetFunctionName(pc_result.value());
        if (fcn_result.ok()) {
          absl::StrAppend(&prompt, "[", fcn_result.value(), "]:\n");
        }
        auto symbol_result = loader->GetFcnSymbolName(pc_result.value());
        if (symbol_result.ok()) {
          absl::StrAppend(&prompt, symbol_result.value(), ":\n");
        }
      }
      absl::StrAppend(&prompt,
                      absl::Hex(pc_result.value(), absl::PadSpec::kZeroPad8));
      auto disasm_result =
          core_access_[current_core_].debug_interface->GetDisassembly(
              pc_result.value());
      if (disasm_result.ok()) {
        absl::StrAppend(&prompt, "   ", disasm_result.value());
      }
      absl::StrAppend(&prompt, "\n");
    }
    absl::StrAppend(&prompt, "[", current_core_, "] > ");
    while (!command_streams_.empty()) {
      auto& current_is = *command_streams_.back();
      // Ignore comments or empty lines.
      bool is_file = command_streams_.size() > 1;
      // Read a command from the input stream. If it's from a file, then ignore
      // empty lines and comments.
      do {
        if (command_streams_.size() == 1) os << prompt;
        current_is.getline(line, kLineSize);
      } while ((is_file && RE2::FullMatch(line, *empty_re_)) &&
               !current_is.bad() && !current_is.eof());

      if (command_streams_.empty()) return;

      // If the current is at eof or gone bad, pop the stream and try the next.
      if (current_is.bad() || current_is.eof()) {
        // If it's not the only stream, delete the stream since it was allocated
        // for an exec command.
        if (is_file) {
          delete command_streams_.back();
        }
        command_streams_.pop_back();
        previous_line = previous_commands_.back();
        previous_commands_.pop_back();
        continue;
      }
      // We have a valid command.
      break;
    }

    if (command_streams_.empty()) {
      os << "Error: input end of file or bad stream state\n" << std::endl;
      os.flush();
      return;
    }

    if (line[0] != '\0') {
      previous_line = line;
    }
    line_view = absl::string_view(previous_line);

    // Start matching commands.

    // First try any added custom commands.
    bool executed = false;
    for (auto& fcn : command_functions_) {
      std::string output;
      executed = fcn(line_view, core_access_[current_core_], output);
      if (executed) {
        os << output << std::endl;
        break;
      }
    }
    if (executed) continue;

    // quit
    if (RE2::FullMatch(line_view, *quit_re_)) return;

    // core N
    if (int new_core;
        RE2::FullMatch(line_view, *core_re_, RE2::CRadix(&new_core))) {
      if (new_core >= core_access_.size()) {
        os << absl::StrCat("Error: core number must be less than ",
                           core_access_.size())
           << std::endl;
        os.flush();
        continue;
      }
      current_core_ = new_core;
      continue;
    }

    // run
    if (RE2::FullMatch(line_view, *run_re_)) {
      auto run_result = core_access_[current_core_].debug_interface->Run();
      if (!run_result.ok()) {
        os << "Error: " << run_result.message() << std::endl;
        os.flush();
      }
      auto wait_result = core_access_[current_core_].debug_interface->Wait();
      if (!wait_result.ok()) {
        os << "Error: " << wait_result.message() << std::endl;
        os.flush();
      }
      continue;
    }

    // run free
    if (RE2::FullMatch(line_view, *run_free_re_)) {
      auto result = core_access_[current_core_].debug_interface->Run();
      if (!result.ok()) {
        os << "Error: " << result.message() << std::endl;
        os.flush();
      }
      continue;
    }

    // wait
    if (RE2::FullMatch(line_view, *wait_re_)) {
      auto result = core_access_[current_core_].debug_interface->Wait();
      if (!result.ok()) {
        os << "Error: " << result.message() << std::endl;
        os.flush();
      }
      continue;
    }

    // step
    if (RE2::FullMatch(line_view, *step_1_re_)) {
      auto result = core_access_[current_core_].debug_interface->Step(1);
      if (!result.status().ok()) {
        os << "Error: " << result.status().message() << std::endl;
        os.flush();
      }
      if (result.value() != 1) {
        os << result.value() << " instructions executed" << std::endl;
        os.flush();
      }
      continue;
    }

    // step N
    if (int count;
        RE2::FullMatch(line_view, *step_n_re_, RE2::CRadix(&count))) {
      auto result = core_access_[current_core_].debug_interface->Step(count);
      if (!result.status().ok()) {
        os << "Error: " << result.status().message() << std::endl;
        os.flush();
      }
      if (result.value() != count) {
        os << result.value() << " instructions executed" << std::endl;
        os.flush();
      }
      continue;
    }

    // halt
    if (RE2::FullMatch(line_view, *halt_re_)) {
      auto result = core_access_[current_core_].debug_interface->Halt();
      if (!result.ok()) {
        os << "Error: " << result.message() << std::endl;
        os.flush();
      }
      continue;
    }

    // reg read NAME
    if (std::string name, format;
        RE2::FullMatch(line_view, *read_reg_re_, &name, &format)) {
      auto result =
          core_access_[current_core_].debug_interface->ReadRegister(name);
      if (result.ok()) {
        os << absl::StrCat(name, " = ", absl::Hex(result.value())) << std::endl;
      } else {
        os << "Error: " << result.status().message() << std::endl;
      }
      os.flush();
      continue;
    }

    // reg write NAME = VALUE
    if (std::string name, value;
        RE2::FullMatch(line_view, *write_reg_re_, &name, &value)) {
      auto result = GetValueFromString(current_core_, value, /*radix=*/0);
      if (!result.ok()) {
        os << absl::StrCat("Error: '", value, "' ", result.status().message())
           << std::endl;
        os.flush();
        continue;
      }
      auto write_result =
          core_access_[current_core_].debug_interface->WriteRegister(
              name, result.value());
      if (!write_result.ok()) {
        os << "Error: " << write_result.message() << std::endl;
        os.flush();
      }
      continue;
    }

    // vreg read NAME:[INDEX] [FORMAT]
    if (std::string name, index, format, width; RE2::FullMatch(
            line_view, *rd_vreg_re_, &name, &index, &format, &width)) {
      if (format.empty()) format = "x";
      if (width.empty()) width = "32";
      int width_value;
      if (!absl::SimpleAtoi(width, &width_value)) {
        os << "Error: converting '" << width << "' to integer\n";
        os.flush();
        continue;
      }
      // Get the data buffer.
      auto result =
          core_access_[current_core_].debug_interface->GetRegisterDataBuffer(
              name);
      // Check for error when accessing register.
      if (!result.ok()) {
        os << "Error: " << result.status().message() << std::endl;
        os.flush();
        continue;
      }
      auto* db = result.value();
      // Check for null data buffer.
      if (db == nullptr) {
        os << "Error: register '" << name << "' has no data buffer\n";
        os.flush();
        continue;
      }
      // Output the value to os.
      if (index.empty()) {
        // Output the entire register.
        os << FormatAllDbValues(db, format, width_value) << std::endl;
        os.flush();
        continue;
      }
      int index_value;
      if (!absl::SimpleAtoi(index, &index_value)) {
        os << "Error: converting '" << index << "' to integer\n";
        os.flush();
        continue;
      }
      os << FormatSingleDbValue(db, format, width_value, index_value)
         << std::endl;
      os.flush();
      continue;
    }

    // vreg set NAME[:INDEX] [FORMAT] VALUE
    if (std::string name, index, format, width, value; RE2::FullMatch(
            line_view, *wr_vreg_re_, &name, &index, &format, &width, &value)) {
      if (format.empty()) format = "x";
      if (width.empty()) width = "32";
      int width_value;
      if (!absl::SimpleAtoi(width, &width_value)) {
        os << "Error: converting '" << width << "' to integer\n";
        os.flush();
        continue;
      }
      // Get the data buffer.
      auto result =
          core_access_[current_core_].debug_interface->GetRegisterDataBuffer(
              name);
      // Check for error when accessing register.
      if (!result.ok()) {
        os << "Error: " << result.status().message() << std::endl;
        continue;
      }
      auto* db = result.value();
      // Check for null data buffer.
      if (db == nullptr) {
        os << "Error: register '" << name << "' has no data buffer\n";
        continue;
      }
      int index_value;
      if (!absl::SimpleAtoi(index, &index_value)) {
        os << "Error: converting '" << index << "' to integer\n";
        os.flush();
        continue;
      }
      auto status =
          WriteSingleValueToDb(value, db, format, width_value, index_value);
      if (!status.ok()) {
        os << "Error: " << status.message() << std::endl;
        os.flush();
      }
      continue;
    }

    // mem get VALUE | SYMBOL [FORMAT]

    if (std::string str_value, format;
        RE2::FullMatch(line_view, *read_mem_re_, &str_value, &format)) {
      os << ReadMemory(current_core_, str_value, format) << std::endl;
      continue;
    }

    if (std::string str_value1, format, str_value2; RE2::FullMatch(
            line_view, *write_mem_re_, &str_value1, &format, &str_value2)) {
      os << WriteMemory(current_core_, str_value1, format, str_value2)
         << std::endl;
      continue;
    }

    // break set VALUE | SYMBOL
    if (std::string str_value;
        RE2::FullMatch(line_view, *set_break_re_, &str_value)) {
      auto result = GetValueFromString(current_core_, str_value, /*radix=*/0);
      if (!result.ok()) {
        os << absl::StrCat("Error: '", str_value, "' ",
                           result.status().message())
           << std::endl;
        os.flush();
        continue;
      }
      auto cmd_result =
          core_access_[current_core_].debug_interface->SetSwBreakpoint(
              result.value());
      if (!cmd_result.ok()) {
        os << "Error:  " << cmd_result.message() << std::endl;
        os.flush();
        continue;
      }
      os << absl::StrCat("Breakpoint set at 0x",
                         absl::Hex(result.value(), absl::PadSpec::kZeroPad8))
         << std::endl;
      continue;
    }

    // break set #<N>
    if (std::string str_value, num_value;
        RE2::FullMatch(line_view, *set_break_n_re_, &str_value, &num_value)) {
      int index;
      if (!absl::SimpleAtoi(num_value, &index)) {
        os << absl::StrCat("Error: cannot parse '", str_value,
                           "' as a breakpoint index\n");
        continue;
      }
      auto iter = core_access_[current_core_].breakpoint_map.find(index);
      if (iter == core_access_[current_core_].breakpoint_map.end()) {
        os << absl::StrCat("Error: no breakpoint with index ", index, "\n");
        continue;
      }
      uint64_t address = iter->second;
      if (!core_access_[current_core_].debug_interface->HasBreakpoint(
              address)) {
        auto status =
            core_access_[current_core_].debug_interface->SetSwBreakpoint(
                address);
        if (!status.ok()) {
          os << absl::StrCat("Error: ", status.message(), "\n");
        }
      } else {
        os << "Breakpoint already active\n";
      }
      continue;
    }

    // break clear #<N>
    if (std::string str_value;
        RE2::FullMatch(line_view, *clear_break_n_re_, &str_value)) {
      int index;
      if (!absl::SimpleAtoi(str_value, &index)) {
        os << absl::StrCat("Error: cannot parse '", str_value,
                           "' as a breakpoint index\n");
        continue;
      }
      auto iter = core_access_[current_core_].breakpoint_map.find(index);
      if (iter == core_access_[current_core_].breakpoint_map.end()) {
        os << absl::StrCat("Error: no breakpoint with index ", index, "\n");
        continue;
      }
      uint64_t address = iter->second;
      if (core_access_[current_core_].debug_interface->HasBreakpoint(address)) {
        auto status =
            core_access_[current_core_].debug_interface->ClearSwBreakpoint(
                address);
        if (!status.ok()) {
          os << absl::StrCat("Error: ", status.message(), "\n");
        }
      }
      continue;
    }

    // break clear-all
    if (RE2::FullMatch(line_view, *clear_all_break_re_)) {
      auto result =
          core_access_[current_core_].debug_interface->ClearAllSwBreakpoints();
      if (!result.ok()) {
        os << absl::StrCat("Error: ", result.message()) << std::endl;
        os.flush();
        continue;
      }
      os << "All breakpoints removed" << std::endl;
      continue;
    }

    // break clear VALUE | SYMBOL
    if (std::string str_value;
        RE2::FullMatch(line_view, *clear_break_re_, &str_value)) {
      auto result = GetValueFromString(current_core_, str_value, /*radix=*/0);
      if (!result.ok()) {
        os << absl::StrCat("Error: '", str_value, "' ",
                           result.status().message())
           << std::endl;
        os.flush();
        continue;
      }
      auto cmd_result =
          core_access_[current_core_].debug_interface->ClearSwBreakpoint(
              result.value());
      if (!cmd_result.ok()) {
        os << "Error:  " << cmd_result.message() << std::endl;
        os.flush();
        continue;
      }
      os << absl::StrCat("Breakpoint removed from 0x",
                         absl::Hex(result.value(), absl::PadSpec::kZeroPad8))
         << std::endl;
      continue;
    }

    // break list
    if (RE2::FullMatch(line_view, *list_break_re_)) {
      std::string bp_list;
      for (auto [index, address] : core_access_[current_core_].breakpoint_map) {
        bool active =
            core_access_[current_core_].debug_interface->HasBreakpoint(address);
        std::string symbol;
        auto* loader = core_access_[current_core_].loader_getter();
        if (loader != nullptr) {
          auto res = loader->GetFcnSymbolName(address);
          if (res.ok()) symbol = std::move(res.value());
        }
        absl::StrAppend(&bp_list,
                        absl::StrFormat("  %3d   %-8s   0x%08x   %s\n", index,
                                        active ? "active" : "inactive", address,
                                        symbol.empty() ? "-" : symbol));
      }
      os << absl::StrCat("Breakpoints:\n", bp_list, "\n");
      continue;
    }

    // help
    if (RE2::FullMatch(line_view, *help_re_)) {
      for (auto const& usage : command_usage_) {
        os << usage << std::endl;
      }
      os << help_message_;
      os.flush();
      continue;
    }

    // reg NAME
    if (std::string name, format;
        RE2::FullMatch(line_view, *read_reg2_re_, &name, &format)) {
      auto result =
          core_access_[current_core_].debug_interface->ReadRegister(name);
      if (result.ok()) {
        os << absl::StrCat(name, " = ", absl::Hex(result.value())) << std::endl;
      } else {
        os << "Error: " << result.status().message() << std::endl;
      }
      os.flush();
      continue;
    }

    // break SYMBOL | VALUE
    if (std::string str_value;
        RE2::FullMatch(line_view, *set_break2_re_, &str_value)) {
      auto result = GetValueFromString(current_core_, str_value, /*radix=*/0);
      if (!result.ok()) {
        os << absl::StrCat("Error: '", str_value, "' ",
                           result.status().message())
           << std::endl;
        os.flush();
        continue;
      }
      auto cmd_result =
          core_access_[current_core_].debug_interface->SetSwBreakpoint(
              result.value());
      if (!cmd_result.ok()) {
        os << "Error:  " << cmd_result.message() << std::endl;
        os.flush();
        continue;
      }
      core_access_[current_core_]
          .breakpoint_map[core_access_[current_core_].breakpoint_index++] =
          result.value();
      os << absl::StrCat("Breakpoint set at 0x",
                         absl::Hex(result.value(), absl::PadSpec::kZeroPad8))
         << std::endl;
      continue;
    }

    // watch set SYMBOL | VALUE  <length> [r|w|rw]
    if (std::string str_value, length_value, rw_value; RE2::FullMatch(
            line_view, *set_watch_re_, &str_value, &length_value, &rw_value)) {
      auto result = GetValueFromString(current_core_, str_value, /*radix=*/0);
      if (!result.ok()) {
        os << absl::StrCat("Error: '", str_value, "' ",
                           result.status().message())
           << std::endl;
        os.flush();
        continue;
      }
      if (!rw_value.empty()) {
        rw_value = rw_value.substr(rw_value.find_first_not_of(' '));
      }
      AccessType access_type = AccessType::kStore;
      if (rw_value == "r") {
        access_type = AccessType::kLoad;
      } else if (rw_value == "rw") {
        access_type = AccessType::kStore;
      }

      uint64_t address = result.value();
      result = GetValueFromString(current_core_, length_value, /*radix=*/0);
      if (!result.ok()) {
        os << absl::StrCat("Error: cannot parse '", length_value,
                           "' as a length\n");
        os.flush();
        continue;
      }
      size_t length = result.value();
      auto* riscv_interface = reinterpret_cast<RiscVDebugInterface*>(
          core_access_[current_core_].debug_interface);
      auto cmd_result =
          riscv_interface->SetDataWatchpoint(address, length, access_type);
      if (!cmd_result.ok()) {
        os << "Error:  " << cmd_result.message() << std::endl;
        os.flush();
        continue;
      }
      core_access_[current_core_]
          .watchpoint_map[core_access_[current_core_].watchpoint_index++] = {
          address, length, access_type, /*active=*/true};
      os << absl::StrCat("Watchpoint set at 0x",
                         absl::Hex(address, absl::PadSpec::kZeroPad8))
         << std::endl;
      continue;
    }

    // watch set #<N>
    if (std::string str_value, num_value;
        RE2::FullMatch(line_view, *set_watch_n_re_, &str_value, &num_value)) {
      int index;
      if (!absl::SimpleAtoi(num_value, &index)) {
        os << absl::StrCat("Error: cannot parse '", str_value,
                           "' as a watchpoint index\n");
        continue;
      }
      auto iter = core_access_[current_core_].watchpoint_map.find(index);
      if (iter == core_access_[current_core_].watchpoint_map.end()) {
        os << absl::StrCat("Error: no watchpoint with index ", index, "\n");
        continue;
      }
      if (iter->second.active) {
        os << "Watchpoint already active\n";
        continue;
      }
      uint64_t address = iter->second.address;
      size_t length = iter->second.length;
      AccessType access_type = iter->second.access_type;
      auto* riscv_interface = reinterpret_cast<RiscVDebugInterface*>(
          core_access_[current_core_].debug_interface);
      auto status =
          riscv_interface->SetDataWatchpoint(address, length, access_type);
      if (!status.ok()) {
        os << absl::StrCat("Error: ", status.message(), "\n");
        continue;
      }
      iter->second.active = true;
      continue;
    }

    // watch clear #<N>
    if (std::string str_value;
        RE2::FullMatch(line_view, *clear_watch_n_re_, &str_value)) {
      int index;
      if (!absl::SimpleAtoi(str_value, &index)) {
        os << absl::StrCat("Error: cannot parse '", str_value,
                           "' as a watchpoint index\n");
        continue;
      }
      auto iter = core_access_[current_core_].watchpoint_map.find(index);
      if (iter == core_access_[current_core_].watchpoint_map.end()) {
        os << absl::StrCat("Error: no watchpoint with index ", index, "\n");
        continue;
      }
      if (!iter->second.active) continue;
      uint64_t address = iter->second.address;
      auto access_type = iter->second.access_type;
      auto* riscv_interface = reinterpret_cast<RiscVDebugInterface*>(
          core_access_[current_core_].debug_interface);
      auto status = riscv_interface->ClearDataWatchpoint(address, access_type);
      if (!status.ok()) {
        os << absl::StrCat("Error: ", status.message(), "\n");
        continue;
      }
      iter->second.active = false;
      continue;
    }

    // watch clear-all
    if (RE2::FullMatch(line_view, *clear_all_watch_re_)) {
      for (auto& [index, info] : core_access_[current_core_].watchpoint_map) {
        if (!info.active) continue;

        uint64_t address = info.address;
        auto access_type = info.access_type;
        auto* riscv_interface = reinterpret_cast<RiscVDebugInterface*>(
            core_access_[current_core_].debug_interface);
        auto status =
            riscv_interface->ClearDataWatchpoint(address, access_type);
        if (!status.ok()) {
          os << absl::StrCat("Error: ", status.message(), "\n");
          continue;
        }
        info.active = false;
      }
      os << "All watchpoints removed" << std::endl;
      continue;
    }

    // watch clear VALUE | SYMBOL [r|w|rw]
    if (std::string str_value, rw_value;
        RE2::FullMatch(line_view, *clear_watch_re_, &str_value, &rw_value)) {
      auto result = GetValueFromString(current_core_, str_value, /*radix=*/0);
      if (!result.ok()) {
        os << absl::StrCat("Error: '", str_value, "' ",
                           result.status().message())
           << std::endl;
        os.flush();
        continue;
      }
      if (!rw_value.empty()) {
        rw_value = rw_value.substr(rw_value.find_first_not_of(' '));
      }
      auto access_type = AccessType::kStore;
      if (rw_value == "r") {
        access_type = AccessType::kLoad;
      } else if (rw_value == "rw") {
        access_type = AccessType::kLoadStore;
      }
      bool done = false;
      for (auto& [index, info] : core_access_[current_core_].watchpoint_map) {
        if ((info.address == result.value()) &&
            (info.access_type == access_type)) {
          auto* riscv_interface = reinterpret_cast<RiscVDebugInterface*>(
              core_access_[current_core_].debug_interface);
          auto cmd_result =
              riscv_interface->ClearDataWatchpoint(result.value(), access_type);
          if (!cmd_result.ok()) {
            os << "Error:  " << cmd_result.message() << std::endl;
            os.flush();
            break;
          }
          info.active = false;
          done = true;
          break;
        }
      }
      if (!done) {
        continue;
      }
      os << absl::StrCat("Watchpoint removed from 0x",
                         absl::Hex(result.value(), absl::PadSpec::kZeroPad8))
         << std::endl;
      continue;
    }

    // watch SYMBOL | VALUE [r|w|rw]
    if (std::string str_value, length_value, rw_value; RE2::FullMatch(
            line_view, *set_watch2_re_, &str_value, &length_value, &rw_value)) {
      auto result = GetValueFromString(current_core_, str_value, /*radix=*/0);
      if (!result.ok()) {
        os << absl::StrCat("Error: '", str_value, "' ",
                           result.status().message())
           << std::endl;
        os.flush();
        continue;
      }
      AccessType access_type = AccessType::kStore;
      if (!rw_value.empty()) {
        rw_value = rw_value.substr(rw_value.find_first_not_of(' '));
      }
      if (rw_value == "r") {
        access_type = AccessType::kLoad;
      } else if (rw_value == "rw") {
        access_type = AccessType::kLoadStore;
      }
      uint64_t address = result.value();
      result = GetValueFromString(current_core_, length_value, /*radix=*/0);
      if (!result.ok()) {
        os << absl::StrCat("Error: cannot parse '", length_value,
                           "' as a length\n");
        os.flush();
        continue;
      }
      size_t length = result.value();
      auto* riscv_interface = reinterpret_cast<RiscVDebugInterface*>(
          core_access_[current_core_].debug_interface);
      auto cmd_result =
          riscv_interface->SetDataWatchpoint(address, length, access_type);
      if (!cmd_result.ok()) {
        os << "Error:  " << cmd_result.message() << std::endl;
        os.flush();
        continue;
      }
      core_access_[current_core_]
          .watchpoint_map[core_access_[current_core_].watchpoint_index++] = {
          address, length, access_type, /*active=*/true};
      os << absl::StrCat("Watchpoint set at 0x",
                         absl::Hex(address, absl::PadSpec::kZeroPad8))
         << std::endl;
      continue;
    }

    // watch list
    if (RE2::FullMatch(line_view, *list_watch_re_)) {
      std::string bp_list;
      for (auto const& [index, info] :
           core_access_[current_core_].watchpoint_map) {
        std::string symbol;
        auto* loader = core_access_[current_core_].loader_getter();
        if (loader != nullptr) {
          auto res = loader->GetFcnSymbolName(info.address);
          if (res.ok()) symbol = std::move(res.value());
        }
        std::string access_type;
        switch (info.access_type) {
          case AccessType::kStore:
            access_type = "w";
            break;
          case AccessType::kLoad:
            access_type = "r";
            break;
          case AccessType::kLoadStore:
            access_type = "rw";
        }
        absl::StrAppend(
            &bp_list,
            absl::StrFormat("  %3d   %-8s   0x%08x   %3d   %2s   %s\n", index,
                            info.active ? "active" : "inactive", info.address,
                            info.length, access_type,
                            symbol.empty() ? "-" : symbol));
      }
      os << absl::StrCat("Watchpoints:\n", bp_list, "\n");
      continue;
    }

    // mem get VALUE | SYMBOL [FORMAT]

    if (std::string str_value, format;
        RE2::FullMatch(line_view, *read_mem2_re_, &str_value, &format)) {
      os << ReadMemory(current_core_, str_value, format) << std::endl;
      continue;
    }
    // Action points.
    if (RE2::FullMatch(line_view, *list_action_re_)) {
      os << ListActionPoints();
      continue;
    }
    if (std::string str_value;
        RE2::FullMatch(line_view, *enable_action_n_re_, &str_value)) {
      os << EnableActionPointN(str_value);
      continue;
    }
    if (std::string str_value;
        RE2::FullMatch(line_view, *disable_action_n_re_, &str_value)) {
      os << DisableActionPointN(str_value);
      continue;
    }
    if (std::string str_value;
        RE2::FullMatch(line_view, *clear_action_n_re_, &str_value)) {
      os << ClearActionPointN(str_value);
      continue;
    }
    if (RE2::FullMatch(line_view, *clear_all_action_re_)) {
      os << ClearAllActionPoints();
      continue;
    }

    // branch-trace.
    // This prints out a list of the last "branch trace size" pairs of <from,
    // to> control flow changes (including interrupts), with no repetitions for
    // loops.
    if (RE2::FullMatch(line_view, *branch_trace_re_)) {
      // Get the index of the head of the queue.
      auto head_result =
          core_access_[current_core_].debug_interface->ReadRegister(
              "$branch_trace_head");
      if (!head_result.ok()) {
        os << "Error: " << head_result.status().message() << "\n";
        continue;
      }
      // Adjust by one, as the head points to the most recent valid entry.
      auto head = head_result.value() + 1;
      // Get the branch trace data buffer.
      auto result =
          core_access_[current_core_].debug_interface->GetRegisterDataBuffer(
              "$branch_trace");
      if (!result.ok()) {
        os << "Error: " << result.status().message() << "\n";
        continue;
      }
      auto* db = result.value();
      // Check for null data buffer.
      if (db == nullptr) {
        os << "Error: register '$branch_trace' has no data buffer\n";
        os.flush();
        continue;
      }
      // Get a span for the branch trace.
      auto trace_span = db->Get<BranchTraceEntry>();
      auto size = trace_span.size();
      os << absl::StrFormat("     %-8s      %-8s     %8s\n", "From", "To",
                            "Count");
      for (int i = 0; i < size; ++i) {
        auto index = (head + i) % size;
        auto [from, to, count] = trace_span[index];
        // Ignore 0 -> 0 entries. Those are the initial values.
        if (count == 0) continue;
        os << absl::StrFormat("   0x%08x -> 0x%08x     %8u\n", from, to, count);
      }
      os.flush();
      continue;
    }

    if (std::string file_name;
        RE2::FullMatch(line_view, *exec_re_, &file_name)) {
      auto* ifile = new std::ifstream(file_name);
      if (!ifile->is_open() || !ifile->good()) {
        os << "Error: unable to open '" << file_name << "'\n";
        os.flush();
        continue;
      }
      previous_commands_.push_back(previous_line);
      command_streams_.push_back(ifile);
      continue;
    }

    // At this point a valid command should have been matched, so assume an
    // error.
    os << absl::StrCat("Error: unrecognized command '", line, "'") << std::endl;
    os.flush();
  }
}
// NOLINTEND(readability/fn_size)

void DebugCommandShell::AddCommand(absl::string_view usage,
                                   CommandFunction command_function) {
  command_usage_.emplace_back(usage);
  command_functions_.emplace_back(std::move(command_function));
}

namespace {

template <typename T>
struct HexFormat {};

template <>
struct HexFormat<uint8_t> {
  static const absl::ParsedFormat<'x'> format;
  static const absl::ParsedFormat<'X'> format_cap;
};
const absl::ParsedFormat<'x'> HexFormat<uint8_t>::format{"%02x"};
const absl::ParsedFormat<'X'> HexFormat<uint8_t>::format_cap{"%02X"};

template <>
struct HexFormat<uint16_t> {
  static const absl::ParsedFormat<'x'> format;
  static const absl::ParsedFormat<'X'> format_cap;
};
const absl::ParsedFormat<'x'> HexFormat<uint16_t>::format{"%04x"};
const absl::ParsedFormat<'X'> HexFormat<uint16_t>::format_cap{"%04X"};

template <>
struct HexFormat<uint32_t> {
  static const absl::ParsedFormat<'x'> format;
  static const absl::ParsedFormat<'X'> format_cap;
};
const absl::ParsedFormat<'x'> HexFormat<uint32_t>::format{"%08x"};
const absl::ParsedFormat<'X'> HexFormat<uint32_t>::format_cap{"%08X"};

template <>
struct HexFormat<uint64_t> {
  static const absl::ParsedFormat<'x'> format;
  static const absl::ParsedFormat<'X'> format_cap;
};
const absl::ParsedFormat<'x'> HexFormat<uint64_t>::format{"%016x"};
const absl::ParsedFormat<'X'> HexFormat<uint64_t>::format_cap{"%016X"};

// Templated helper function to help format integer values of different widths.
template <typename T>
std::string FormatDbValue(generic::DataBuffer* db, const std::string& format,
                          int index) {
  std::string output;
  if (index < 0 || index >= db->size<T>()) return "Error: index out of range";
  T value = db->Get<T>(index);
  switch (format[0]) {
    case 'd':
      output += absl::StrFormat("%d", value);
      break;
    case 'o':
      output += absl::StrFormat("%o", value);
      break;
    case 'u':
      output += absl::StrFormat("%u", value);
      break;
    case 'x':
      output += absl::StrFormat(HexFormat<T>::format, value);
      break;
    case 'X':
      output += absl::StrFormat(HexFormat<T>::format_cap, value);
      break;
    default:
      output = absl::StrCat("Error: invalid '", format, "'");
      break;
  }
  return output;
}

template <typename T>
absl::Status WriteDbValue(const std::string& str_value,
                          const std::string& format, int index,
                          generic::DataBuffer* db) {
  if (index < 0 || index >= db->size<T>())
    return absl::OutOfRangeError("Error: index out of range");
  if (format[0] == 'd') {
    int64_t value;
    if (!absl::SimpleAtoi(str_value, &value)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Error: could not convert '", str_value, "' to number"));
    }
    db->Set<T>(index, static_cast<T>(value));
    return absl::OkStatus();
  }
  if (format[0] == 'u') {
    uint64_t value;
    if (!absl::SimpleAtoi(str_value, &value)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Error: could not convert '", str_value, "' to number"));
    }
    db->Set<T>(index, static_cast<T>(value));
    return absl::OkStatus();
  }
  if (format[0] == 'x' || format[0] == 'X') {
    uint64_t value;
    if (!absl::SimpleHexAtoi(str_value, &value)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Error: could not convert '", str_value, "' to number"));
    }
    db->Set<T>(index, static_cast<T>(value));
    return absl::OkStatus();
  }
  return absl::InternalError(absl::StrCat("Unsupported format '", format, "'"));
}

}  // namespace

std::string DebugCommandShell::FormatSingleDbValue(generic::DataBuffer* db,
                                                   const std::string& format,
                                                   int width, int index) const {
  switch (width) {
    case 8:
      return FormatDbValue<uint8_t>(db, format, index);
    case 16:
      return FormatDbValue<uint16_t>(db, format, index);
    case 32:
      return FormatDbValue<uint32_t>(db, format, index);
    case 64:
      return FormatDbValue<uint64_t>(db, format, index);
    default:
      return absl::StrCat("Error: illegal width '", width, "'");
  }
}

std::string DebugCommandShell::FormatAllDbValues(generic::DataBuffer* db,
                                                 const std::string& format,
                                                 int width) const {
  std::string output;
  std::string sep;
  switch (width) {
    case 8:
      for (int i = 0; i < db->size<uint8_t>(); ++i) {
        output += sep + FormatDbValue<uint8_t>(db, format, i);
        sep = ":";
      }
      break;
    case 16:
      for (int i = 0; i < db->size<uint16_t>(); ++i) {
        output += sep + FormatDbValue<uint16_t>(db, format, i);
        sep = ":";
      }
      break;
    case 32:
      for (int i = 0; i < db->size<uint32_t>(); ++i) {
        output += sep + FormatDbValue<uint32_t>(db, format, i);
        sep = ":";
      }
      break;
    case 64:
      for (int i = 0; i < db->size<uint64_t>(); ++i) {
        output += sep + FormatDbValue<uint64_t>(db, format, i);
        sep = ":";
      }
      break;
    default:
      output = absl::StrCat("Error: illegal width '", width, "'");
      break;
  }
  return output;
}

absl::Status DebugCommandShell::WriteSingleValueToDb(
    const std::string& str_value, generic::DataBuffer* db, std::string format,
    int width, int index) const {
  switch (width) {
    case 8:
      return WriteDbValue<uint8_t>(str_value, format, index, db);
    case 16:
      return WriteDbValue<uint16_t>(str_value, format, index, db);
    case 32:
      return WriteDbValue<uint32_t>(str_value, format, index, db);
    case 64:
      return WriteDbValue<uint64_t>(str_value, format, index, db);
    default:
      return absl::InternalError("Error");
  }
  return absl::OkStatus();
}

std::string DebugCommandShell::ReadMemory(int core,
                                          const std::string& str_value,
                                          const std::string& format) {
  int size = 0;
  char format_char = 'x';
  int bit_width = 32;

  if (!format.empty()) {
    if (format[0] == 'i') {
      format_char = 'i';
    } else {
      // Check the format specification.
      auto pos = format.find_first_not_of(' ');
      format_char = format[pos];
      auto status = internal::stoull(format.substr(pos + 1));
      bit_width = 0;
      if (!status.ok()) {
        return absl::StrCat("Error '", format.substr(pos + 1),
                            "': ", status.status().message());
      }
      bit_width = status.value();
      if ((bit_width != 8) && (bit_width != 16) && (bit_width != 32) &&
          (bit_width != 64)) {
        return absl::StrCat("Illegal bit width specification: ", bit_width);
      }
    }
  }

  // Get the address.
  auto result = GetValueFromString(core, str_value, /*radix=*/0);
  if (!result.ok()) {
    return absl::StrCat("Error: '", str_value, "' ", result.status().message());
  }
  auto address = result.value();

  // If format is 'i', then we are getting a disassembled instruction. Ignore
  // bitwidth.
  if (format_char == 'i') {
    auto disasm_result =
        core_access_[current_core_].debug_interface->GetDisassembly(address);
    if (!disasm_result.ok()) {
      return absl::StrCat("Error: ", disasm_result.status().message());
    }
    return absl::StrCat("    ", disasm_result.value());
  }

  // Perform the memory access.
  size = bit_width / 8;
  if (size > kMemBufferSize) size = kMemBufferSize;
  auto mem_result = core_access_[current_core_].debug_interface->ReadMemory(
      address, mem_buffer_, size);
  if (!mem_result.ok()) {
    return absl::StrCat("Error: ", mem_result.status().message());
  }

  // Get the result and format it.
  void* void_buffer = mem_buffer_;
  std::string output;
  if ((format_char == 'f') && (bit_width >= 32)) {
    switch (bit_width) {
      case 32:
        output = absl::StrCat(*reinterpret_cast<float*>(void_buffer));
        break;
      case 64:
        output = absl::StrCat(*reinterpret_cast<double*>(void_buffer));
        break;
      default:
        break;
    }
  } else if (format_char == 'd') {
    switch (bit_width) {
      case 8:
        output = absl::StrCat(*static_cast<int8_t*>(void_buffer));
        break;
      case 16:
        output = absl::StrCat(*static_cast<int16_t*>(void_buffer));
        break;
      case 32:
        output = absl::StrCat(*static_cast<int32_t*>(void_buffer));
        break;
      case 64:
        output = absl::StrCat(*static_cast<int64_t*>(void_buffer));
        break;
      default:
        break;
    }
  } else {
    uint64_t val = 0;
    auto pad = absl::PadSpec::kNoPad;
    switch (bit_width) {
      case 8:
        val = *static_cast<uint8_t*>(void_buffer);
        pad = absl::PadSpec::kZeroPad2;
        break;
      case 16:
        val = *static_cast<uint16_t*>(void_buffer);
        pad = absl::PadSpec::kZeroPad4;
        break;
      case 32:
        val = *static_cast<uint32_t*>(void_buffer);
        pad = absl::PadSpec::kZeroPad8;
        break;
      case 64:
        val = *static_cast<uint64_t*>(void_buffer);
        pad = absl::PadSpec::kZeroPad16;
        break;
    }
    std::string format_string;
    if ((format_char == 'x') || (format_char == 'X')) {
      output = absl::StrCat(absl::Hex(val, pad));
    } else if (format_char == 'u') {
      output = absl::StrCat(val);
    } else {
      output = absl::StrFormat("%o", val);
    }
  }
  return absl::StrCat("[", absl::Hex(address, absl::PadSpec::kZeroPad8),
                      "] = ", output);
}

std::string DebugCommandShell::WriteMemory(int core,
                                           const std::string& str_value1,
                                           const std::string& format,
                                           const std::string& str_value2) {
  int size = 0;
  char format_char = '\0';
  int radix = 0;
  int bit_width = 32;
  if (!format.empty()) {
    // Check the format specification.
    auto pos = format.find_first_not_of(' ');
    format_char = format[pos];
    if ((format_char == 'x') || (format_char == 'X')) {
      radix = 16;
    } else if ((format_char == 'd') || (format_char == 'u')) {
      radix = 10;
    } else if (format_char == 'o') {
      radix = 8;
    }
    auto status = internal::stoull(format.substr(pos + 1));
    bit_width = 0;
    if (!status.ok()) {
      return absl::StrCat("Error '", format.substr(pos + 1),
                          "': ", status.status().message());
    }
    bit_width = status.value();
  }

  // Determine the zero padding for hex number.
  auto pad = absl::kNoPad;
  if (bit_width == 8) {
    pad = absl::kZeroPad2;
  } else if (bit_width == 16) {
    pad = absl::kZeroPad4;
  } else if (bit_width == 32) {
    pad = absl::kZeroPad8;
  } else if (bit_width == 64) {
    pad = absl::kZeroPad16;
  } else {
    return absl::StrCat("Illegal bit width specification: ", bit_width);
  }

  // Get the address.
  auto result = GetValueFromString(core, str_value1, /*radix=*/0);
  if (!result.ok()) {
    return absl::StrCat("Error: '", str_value1, "' ",
                        result.status().message());
  }
  auto address = result.value();

  // Get the value to be stored.
  auto value_result = GetValueFromString(core, str_value2, radix);
  if (!value_result.ok()) {
    return absl::StrCat("Error: '", str_value2, "' ",
                        result.status().message());
  }
  int64_t mem_value = value_result.value();

  // Perform the memory access.
  size = bit_width / 8;
  if (size > kMemBufferSize) size = kMemBufferSize;
  std::memcpy(mem_buffer_, &mem_value, size);
  auto mem_result = core_access_[current_core_].debug_interface->WriteMemory(
      address, mem_buffer_, size);
  if (!mem_result.ok()) {
    return absl::StrCat("Error: ", mem_result.status().message());
  }
  return absl::StrCat("[", absl::Hex(address, absl::kZeroPad8),
                      "] = ", absl::Hex(mem_value, pad));
}

absl::StatusOr<uint64_t> DebugCommandShell::GetValueFromString(
    int core, const std::string& str_value, int radix) {
  size_t index;
  // Attempt to convert to a number.
  auto convert_result = internal::stoull(str_value, &index, radix);
  // If successful and the entire string was consumed, the number is good.
  if (convert_result.ok() && (index >= str_value.size())) {
    return convert_result.value();
  }
  // If it's out of range, signal an error.
  if (convert_result.status().code() == absl::StatusCode::kOutOfRange) {
    return convert_result.status();
  }
  // If all else fails, let's see if it's a symbol.
  auto* loader = core_access_[current_core_].loader_getter();
  if (loader == nullptr) return absl::NotFoundError("No symbol table");
  auto result = loader->GetSymbol(str_value);
  if (!result.ok()) return result.status();
  return result.value().first;
}

std::string DebugCommandShell::FormatRegister(
    int core, const std::string& reg_name) const {
  std::string output;
  auto result =
      core_access_[current_core_].debug_interface->ReadRegister(reg_name);
  if (result.ok()) {
    absl::StrAppend(&output, reg_name, " = ", absl::Hex(result.value()));
  } else {
    absl::StrAppend(&output, "Error reading '", reg_name,
                    "': ", result.status().message());
  }
  return output;
}

std::string DebugCommandShell::FormatAllRegisters(int core) const {
  std::string output;
  for (auto const& reg_name : reg_vector_) {
    absl::StrAppend(&output, FormatRegister(current_core_, reg_name), "\n");
  }
  return output;
}

// Action point methods.
std::string DebugCommandShell::ListActionPoints() {
  std::string output;
  auto& action_map = core_action_point_info_[current_core_];
  for (auto const& [local_id, info] : action_map) {
    absl::StrAppend(
        &output,
        absl::StrFormat("%02d  [0x%08lx] %8s  %s\n", local_id, info.address,
                        info.is_enabled ? "enabled" : "disabled", info.name));
  }
  return output;
}

std::string DebugCommandShell::EnableActionPointN(
    const std::string& index_str) {
  auto res = riscv::internal::stoull(index_str, nullptr, 10);
  if (!res.ok()) {
    return std::string(res.status().message());
  }
  auto& action_map = core_action_point_info_[current_core_];
  int index = res.value();
  auto it = action_map.find(index);
  if (it == action_map.end()) {
    return absl::StrCat("Action point ", index, " not found");
  }
  auto& info = it->second;
  if (info.is_enabled) {
    return absl::StrCat("Action point ", index, " is already enabled");
  }
  info.is_enabled = true;
  auto* dbg_if = core_access_[current_core_].debug_interface;
  auto* riscv_dbg_if = static_cast<RiscVDebugInterface*>(dbg_if);
  auto status = riscv_dbg_if->EnableAction(info.address, info.id);
  if (!status.ok()) {
    return absl::StrCat("Error: ", status.message());
  }
  return "";
}

std::string DebugCommandShell::DisableActionPointN(
    const std::string& index_str) {
  auto res = riscv::internal::stoull(index_str, nullptr, 10);
  if (!res.ok()) {
    return std::string(res.status().message());
  }
  auto& action_map = core_action_point_info_[current_core_];
  int index = res.value();
  auto it = action_map.find(index);
  if (it == action_map.end()) {
    return absl::StrCat("Action point ", index, " not found");
  }
  auto& info = it->second;
  if (!info.is_enabled) {
    return absl::StrCat("Action point ", index, " is already disabled");
  }
  info.is_enabled = false;
  auto* dbg_if = core_access_[current_core_].debug_interface;
  auto* riscv_dbg_if = static_cast<RiscVDebugInterface*>(dbg_if);
  auto status = riscv_dbg_if->DisableAction(info.address, info.id);
  if (!status.ok()) {
    return absl::StrCat("Error: ", status.message());
  }
  return "";
}

std::string DebugCommandShell::ClearActionPointN(const std::string& index_str) {
  auto res = riscv::internal::stoull(index_str, nullptr, 10);
  if (!res.ok()) {
    return std::string(res.status().message());
  }
  auto& action_map = core_action_point_info_[current_core_];
  int index = res.value();
  auto it = action_map.find(index);
  if (it == action_map.end()) {
    return absl::StrCat("Action point ", index, " not found");
  }
  auto& info = it->second;
  auto* dbg_if = core_access_[current_core_].debug_interface;
  auto* riscv_dbg_if = static_cast<RiscVDebugInterface*>(dbg_if);
  auto status = riscv_dbg_if->ClearActionPoint(info.address, info.id);
  if (!status.ok()) {
    return absl::StrCat("Error: ", status.message());
  }
  action_map.erase(it);
  return "";
}

std::string DebugCommandShell::ClearAllActionPoints() {
  std::string output;
  auto* dbg_if = core_access_[current_core_].debug_interface;
  auto* riscv_dbg_if = static_cast<RiscVDebugInterface*>(dbg_if);
  for (auto& [local_id, info] : core_action_point_info_[current_core_]) {
    auto status = riscv_dbg_if->ClearActionPoint(info.address, info.id);
    if (!status.ok()) {
      absl::StrAppend(&output, "Error: ", status.message());
    }
  }
  return output;
}

absl::Status DebugCommandShell::SetActionPoint(
    uint64_t address, std::string name,
    absl::AnyInvocable<void(uint64_t, int)> function) {
  auto* dbg_if = core_access_[current_core_].debug_interface;
  auto* riscv_dbg_if = static_cast<RiscVDebugInterface*>(dbg_if);
  auto result = riscv_dbg_if->SetActionPoint(address, std::move(function));
  if (!result.ok()) {
    return absl::InternalError(result.status().message());
  }
  int id = result.value();
  int local_id = core_action_point_id_[current_core_]++;
  auto& action_map = core_action_point_info_[current_core_];
  action_map.emplace(local_id, ActionPointInfo{address, id, name, true});
  return absl::OkStatus();
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
