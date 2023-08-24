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

#ifndef MPACT_RISCV_RISCV_DEBUG_COMMAND_SHELL_H_
#define MPACT_RISCV_RISCV_DEBUG_COMMAND_SHELL_H_

#include <cstdint>
#include <iostream>
#include <istream>
#include <ostream>
#include <string>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mpact/sim/generic/core_debug_interface.h"
#include "mpact/sim/util/program_loader/elf_program_loader.h"
#include "re2/re2.h"

namespace mpact {
namespace sim {
namespace riscv {

namespace mpact::sim::generic {
class DataBuffer;
}  // namespace mpact::sim::generic

// This class implements an interactive command shell for a set of cores
// simulated by the MPact simulator using the CoreDebugInterface.
class DebugCommandShell {
 public:
  static constexpr int kMemBufferSize = 32;

  // Each core must provide the debug interface and the elf loader.
  struct CoreAccess {
    generic::CoreDebugInterface *debug_interface;
    util::ElfProgramLoader *loader;
  };

  // Type of custom command processing invocables. It takes a string_view of
  // the current text input, the current core access structure, and a string
  // to be written to the command shell output. The invocable should return
  // true if the command input string was successfully matched, regardless of
  // any error while executing the command, in which case the output string
  // should be set to an appropriate error message.
  using CommandFunction = absl::AnyInvocable<bool(
      absl::string_view, const CoreAccess &, std::string &)>;

  // Default constructor is deleted.
  DebugCommandShell() = delete;
  // Pass in a vector of CoreAccess structs, one per core in the system.
  explicit DebugCommandShell(std::vector<CoreAccess> core_access);

  // The run method is the command interpreter. It parses the command strings,
  // executes the corresponding commands, displays results and error messages.
  void Run(std::istream &is, std::ostream &os);

  // This adds a custom command to the command interpreter. Usage will be added
  // to the standard command usage. The callable will be called before the
  // standard commands are processed.
  void AddCommand(absl::string_view usage, CommandFunction command_function);

 private:
  // Helper method for formatting single data buffer value.
  std::string FormatSingleDbValue(generic::DataBuffer *db,
                                  const std::string &format, int width,
                                  int index) const;
  // Helper method for formatting multiple data buffer values.
  std::string FormatAllDbValues(generic::DataBuffer *db,
                                const std::string &format, int width) const;
  // Helper method for writing single data buffer value.
  absl::Status WriteSingleValueToDb(const std::string &str_value,
                                    generic::DataBuffer *db, std::string format,
                                    int width, int index) const;

  // Helper method for processing read memory command.
  std::string ReadMemory(int core, const std::string &str_value,
                         const std::string &format);
  // Helper method for processing write memory command.
  std::string WriteMemory(int core, const std::string &str_value1,
                          const std::string &format,
                          const std::string &str_value2);
  // Helper method used to parse a numeric string or use the string as a
  // symbol name for lookup in the loader.
  absl::StatusOr<uint64_t> GetValueFromString(int core,
                                              const std::string &str_value,
                                              int radix);

  std::vector<CoreAccess> core_access_;
  // Help message displayed for command 'help'.
  std::string help_message_;

  // Regular expression variables used to parse commands in the shell.
  LazyRE2 quit_re_;
  LazyRE2 core_re_;
  LazyRE2 run_re_;
  LazyRE2 run_free_re_;
  LazyRE2 wait_re_;
  LazyRE2 step_1_re_;
  LazyRE2 step_n_re_;
  LazyRE2 halt_re_;
  LazyRE2 read_reg_re_;
  LazyRE2 read_reg2_re_;
  LazyRE2 write_reg_re_;
  LazyRE2 rd_vreg_re_;
  LazyRE2 wr_vreg_re_;
  LazyRE2 read_mem_re_;
  LazyRE2 read_mem2_re_;
  LazyRE2 write_mem_re_;
  LazyRE2 set_break_re_;
  LazyRE2 set_break2_re_;
  LazyRE2 clear_break_re_;
  LazyRE2 clear_all_break_re_;
  LazyRE2 help_re_;

  uint8_t mem_buffer_[kMemBufferSize];
  std::vector<CommandFunction> command_functions_;
  std::vector<std::string> command_usage_;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_DEBUG_COMMAND_SHELL_H_
