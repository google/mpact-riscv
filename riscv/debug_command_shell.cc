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

#include <cstring>
#include <istream>
#include <ostream>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "riscv/stoull_wrapper.h"

namespace mpact {
namespace sim {
namespace riscv {

// The constructor initializes all the regular expressions and the help string.
DebugCommandShell::DebugCommandShell(std::vector<CoreAccess> core_access)
    : core_access_(core_access),
      quit_re_{R"(\s*quit\s*)"},
      core_re_{R"(\s*core\s+(\d+)\s*)"},
      run_re_{R"(\s*run\s*)"},
      run_free_re_{R"(\s*run\s+free\s*)"},
      wait_re_{R"(\s*wait\s*)"},
      step_1_re_{R"(\s*step\s*)"},
      step_n_re_{R"(\s*step\s+(\d+)\s*)"},
      halt_re_{R"(\s*halt\s*)"},
      read_reg_re_{R"(\s*reg\s+get\s+(\$?\w+)(\s+[foduxX]\d+)?\s*)"},
      read_reg2_re_{R"(\s*reg\s+(\$?\w+)(\s+[foduxX]\d+)?\s*)"},
      write_reg_re_{R"(\s*reg\s+set\s+(\$?\w+)\s+(\w+)\s*)"},
      read_mem_re_{R"(\s*mem\s+get\s+(\w+)\s+([foduxX]\d+)?\s*)"},
      read_mem2_re_{R"(\s*mem\s+(\w+)\s+([foduxX]\d+)?\s*)"},
      write_mem_re_{R"(\s*mem\s+set\s+(\w+)\s+([oduxX]\d+)?\s+(\w+)\s*)"},
      set_break_re_{R"(\s*break\s+set\s+(\w+)\s*)"},
      set_break2_re_{R"(\s*break\s+(\w+)\s*)"},
      clear_break_re_{R"(\s*break\s+clear\s+(\w+)\s*)"},
      clear_all_break_re_{R"(\s*break\s+clear-all\s*)"},
      help_re_{R"(\s*help\s*)"} {
  help_message_ = R"raw(

    quit                           - exit command shell.
    core [N]                       - direct subsequent commands to core N
                                     (default: 0).
    run                            - run program from current pc until a
                                     breakpoint or exit. Wait until halted.
    run free                       - run program in background from current pc
                                     until breakpoint or exit.
    wait                           - wait for any free run to complete.
    step [N]                       - step [N] instructions (default: 1).
    halt                           - halt a running program.
    reg get NAME [FORMAT]          - get the value or register NAME.
    reg NAME [FORMAT]              - get the value of register NAME.
    reg set NAME VALUE             - set register NAME to VALUE.
    reg set NAME SYMBOL            - set register NAME to value of SYMBOL.
    mem get VALUE [FORMAT]         - get memory from location VALUE according to
                                     format. The format is a letter (o, d, u, x,
                                     or X) followed by width (8, 16, 32, 64).
                                     The default format is x32.
    mem get SYMBOL [FORMAT]        - get memory from location SYMBOL and format
                                     according to FORMAT (see above).
    mem SYMBOL [FORMAT]            - get memory from location SYMBOL and format
                                     according to FORMAT (see above).
    mem set VALUE [FORMAT] VALUE   - set memory at location VALUE(1) to VALUE(2)
                                     according to FORMAT. Default format is x32.
    mem set SYMBOL [FORMAT] VALUE  - set memory at location SYMBOL to VALUE
                                     according to FORMAT. Default format is x32.
    break set VALUE                - set breakpoint at address VALUE.
    break set SYMBOL               - set breakpoint at value of SYMBOL.
    break VALUE                    - set breakpoint at address VALUE.
    break SYMBOL                   - set breakpoint at value of SYMBOL.
    break clear VALUE              - clear breakpoint at address VALUE.
    break clear SYMBOL             - clear breakpoint at value of SYMBOL.
    break clear all                - remove all breakpoints.
    help                           - display this message.

)raw";
}

void DebugCommandShell::Run(std::istream &is, std::ostream &os) {
  // Assumes the max linesize is 512.
  constexpr int kLineSize = 512;
  char line[kLineSize];
  std::string previous_line;
  int core = 0;
  absl::string_view line_view;
  while (true) {
    // Prompt and read in the next command.
    auto pc_result = core_access_[core].debug_interface->ReadRegister("pc");
    std::string prompt;
    if (pc_result.ok()) {
      auto symbol_result =
          core_access_[core].loader->GetFcnSymbolName(pc_result.value());
      if (symbol_result.ok()) {
        absl::StrAppend(&prompt, symbol_result.value(), ":\n");
      }
      absl::StrAppend(&prompt,
                      absl::Hex(pc_result.value(), absl::PadSpec::kZeroPad8));
      auto disasm_result =
          core_access_[core].debug_interface->GetDisassembly(pc_result.value());
      if (disasm_result.ok()) {
        absl::StrAppend(&prompt, "   ", disasm_result.value());
      }
      absl::StrAppend(&prompt, "\n");
    }
    absl::StrAppend(&prompt, "[", core, "] > ");
    os << prompt;
    is.getline(line, kLineSize);

    // If the file is at eof or gone bad, return.
    if (is.bad() || is.eof()) {
      os << "Error: input end of file or bad stream state\n" << std::endl;
      os.flush();
      return;
    }

    if (line[0] != '\0') {
      previous_line = line;
    }
    line_view = absl::string_view(previous_line);

    // Start matching commands.

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
      core = new_core;
      continue;
    }

    // run
    if (RE2::FullMatch(line_view, *run_re_)) {
      auto run_result = core_access_[core].debug_interface->Run();
      if (!run_result.ok()) {
        os << "Error: " << run_result.message() << std::endl;
        os.flush();
      }
      auto wait_result = core_access_[core].debug_interface->Wait();
      if (!wait_result.ok()) {
        os << "Error: " << wait_result.message() << std::endl;
        os.flush();
      }
      continue;
    }

    // run free
    if (RE2::FullMatch(line_view, *run_free_re_)) {
      auto result = core_access_[core].debug_interface->Run();
      if (!result.ok()) {
        os << "Error: " << result.message() << std::endl;
        os.flush();
      }
      continue;
    }

    // wait
    if (RE2::FullMatch(line_view, *wait_re_)) {
      auto result = core_access_[core].debug_interface->Wait();
      if (!result.ok()) {
        os << "Error: " << result.message() << std::endl;
        os.flush();
      }
      continue;
    }

    // step
    if (RE2::FullMatch(line_view, *step_1_re_)) {
      auto result = core_access_[core].debug_interface->Step(1);
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
      auto result = core_access_[core].debug_interface->Step(count);
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
      auto result = core_access_[core].debug_interface->Halt();
      if (!result.ok()) {
        os << "Error: " << result.message() << std::endl;
        os.flush();
      }
      continue;
    }

    // reg read NAME
    if (std::string name, format;
        RE2::FullMatch(line_view, *read_reg_re_, &name, &format)) {
      auto result = core_access_[core].debug_interface->ReadRegister(name);
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
      auto result = GetValueFromString(core, value, /*radix=*/0);
      if (!result.ok()) {
        os << absl::StrCat("Error: '", value, "' ", result.status().message())
           << std::endl;
        os.flush();
        continue;
      }
      auto write_result = core_access_[core].debug_interface->WriteRegister(
          name, result.value());
      if (!write_result.ok()) {
        os << "Error: " << write_result.message() << std::endl;
        os.flush();
      }
      continue;
    }

    // mem get VALUE | SYMBOL [FORMAT]

    if (std::string str_value, format;
        RE2::FullMatch(line_view, *read_mem_re_, &str_value, &format)) {
      os << ReadMemory(core, str_value, format) << std::endl;
      continue;
    }

    if (std::string str_value1, format, str_value2; RE2::FullMatch(
            line_view, *write_mem_re_, &str_value1, &format, &str_value2)) {
      os << WriteMemory(core, str_value1, format, str_value2) << std::endl;
      continue;
    }

    // break set VALUE | SYMBOL
    if (std::string str_value;
        RE2::FullMatch(line_view, *set_break_re_, &str_value)) {
      auto result = GetValueFromString(core, str_value, /*radix=*/0);
      if (!result.ok()) {
        os << absl::StrCat("Error: '", str_value, "' ",
                           result.status().message())
           << std::endl;
        os.flush();
        continue;
      }
      auto cmd_result =
          core_access_[core].debug_interface->SetSwBreakpoint(result.value());
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

    // break clear-all
    if (RE2::FullMatch(line_view, *clear_all_break_re_)) {
      auto result = core_access_[core].debug_interface->ClearAllSwBreakpoints();
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
      auto result = GetValueFromString(core, str_value, /*radix=*/0);
      if (!result.ok()) {
        os << absl::StrCat("Error: '", str_value, "' ",
                           result.status().message())
           << std::endl;
        os.flush();
        continue;
      }
      auto cmd_result =
          core_access_[core].debug_interface->ClearSwBreakpoint(result.value());
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

    // help
    if (RE2::FullMatch(line_view, *help_re_)) {
      os << help_message_;
      os.flush();
      continue;
    }

    // reg NAME
    if (std::string name, format;
        RE2::FullMatch(line_view, *read_reg2_re_, &name, &format)) {
      auto result = core_access_[core].debug_interface->ReadRegister(name);
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
      auto result = GetValueFromString(core, str_value, /*radix=*/0);
      if (!result.ok()) {
        os << absl::StrCat("Error: '", str_value, "' ",
                           result.status().message())
           << std::endl;
        os.flush();
        continue;
      }
      auto cmd_result =
          core_access_[core].debug_interface->SetSwBreakpoint(result.value());
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

    // mem get VALUE | SYMBOL [FORMAT]

    if (std::string str_value, format;
        RE2::FullMatch(line_view, *read_mem2_re_, &str_value, &format)) {
      os << ReadMemory(core, str_value, format) << std::endl;
      continue;
    }

    // At this point a valid command should have been matched, so assume an
    // error.
    os << absl::StrCat("Error: unrecognized command '", line, "'") << std::endl;
    os.flush();
  }
}

std::string DebugCommandShell::ReadMemory(int core,
                                          const std::string &str_value,
                                          const std::string &format) {
  int size = 0;
  char format_char = 'x';
  int bit_width = 32;

  if (!format.empty()) {
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

  // Get the address.
  auto result = GetValueFromString(core, str_value, /*radix=*/0);
  if (!result.ok()) {
    return absl::StrCat("Error: '", str_value, "' ", result.status().message());
  }
  auto address = result.value();

  // Perform the memory access.
  size = bit_width / 8;
  if (size > kMemBufferSize) size = kMemBufferSize;
  auto mem_result = core_access_[core].debug_interface->ReadMemory(
      address, mem_buffer_, size);
  if (!mem_result.ok()) {
    return absl::StrCat("Error: ", mem_result.status().message());
  }

  // Get the result and format it.
  void *void_buffer = mem_buffer_;
  std::string output;
  if ((format_char == 'f') && (bit_width >= 32)) {
    switch (bit_width) {
      case 32:
        output = absl::StrCat(*reinterpret_cast<float *>(void_buffer));
        break;
      case 64:
        output = absl::StrCat(*reinterpret_cast<double *>(void_buffer));
        break;
      default:
        break;
    }
  } else if (format_char == 'd') {
    switch (bit_width) {
      case 8:
        output = absl::StrCat(*static_cast<int8_t *>(void_buffer));
        break;
      case 16:
        output = absl::StrCat(*static_cast<int16_t *>(void_buffer));
        break;
      case 32:
        output = absl::StrCat(*static_cast<int32_t *>(void_buffer));
        break;
      case 64:
        output = absl::StrCat(*static_cast<int64_t *>(void_buffer));
        break;
      default:
        break;
    }
  } else {
    uint64_t val = 0;
    auto pad = absl::PadSpec::kNoPad;
    switch (bit_width) {
      case 8:
        val = *static_cast<uint8_t *>(void_buffer);
        pad = absl::PadSpec::kZeroPad2;
        break;
      case 16:
        val = *static_cast<uint16_t *>(void_buffer);
        pad = absl::PadSpec::kZeroPad4;
        break;
      case 32:
        val = *static_cast<uint32_t *>(void_buffer);
        pad = absl::PadSpec::kZeroPad8;
        break;
      case 64:
        val = *static_cast<uint64_t *>(void_buffer);
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
                                           const std::string &str_value1,
                                           const std::string &format,
                                           const std::string &str_value2) {
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
  auto mem_result = core_access_[core].debug_interface->WriteMemory(
      address, mem_buffer_, size);
  if (!mem_result.ok()) {
    return absl::StrCat("Error: ", mem_result.status().message());
  }
  return absl::StrCat("[", absl::Hex(address, absl::kZeroPad8),
                      "] = ", absl::Hex(mem_value, pad));
}

absl::StatusOr<uint64_t> DebugCommandShell::GetValueFromString(
    int core, const std::string &str_value, int radix) {
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
  auto result = core_access_[core].loader->GetSymbol(str_value);
  if (!result.ok()) return result.status();
  return result.value().first;
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
