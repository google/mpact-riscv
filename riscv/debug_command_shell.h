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
#include <deque>
#include <iostream>
#include <istream>
#include <ostream>
#include <string>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mpact/sim/generic/core_debug_interface.h"
#include "mpact/sim/generic/debug_command_shell_interface.h"
#include "re2/re2.h"

namespace mpact {
namespace sim {
namespace riscv {

namespace mpact::sim::generic {
class DataBuffer;
}  // namespace mpact::sim::generic

// This class implements an interactive command shell for a set of cores
// simulated by the MPact simulator using the CoreDebugInterface.
class DebugCommandShell : public sim::generic::DebugCommandShellInterface {
 public:
  // Default constructor is deleted.
  DebugCommandShell();

  // Add core access to the system. All cores must be added before calling Run.
  void AddCore(const CoreAccess& core_access) override;
  void AddCores(const std::vector<CoreAccess>& core_access) override;

  // The run method is the command interpreter. It parses the command strings,
  // executes the corresponding commands, displays results and error messages.
  void Run(std::istream& is, std::ostream& os) override;

  // This adds a custom command to the command interpreter. Usage will be added
  // to the standard command usage. The callable will be called before the
  // standard commands are processed.
  void AddCommand(absl::string_view usage,
                  CommandFunction command_function) override;

  // This adds an action point at the given address with 'function' as the
  // action. The string 'name' is used to give it an identifier to use when
  // listed.
  absl::Status SetActionPoint(uint64_t address, std::string name,
                              absl::AnyInvocable<void(uint64_t, int)> function);

 private:
  // Struct to track action point information.
  struct ActionPointInfo {
    uint64_t address;
    int id;
    std::string name;
    bool is_enabled;
  };

  // Helper method for formatting single data buffer value.
  std::string FormatSingleDbValue(generic::DataBuffer* db,
                                  const std::string& format, int width,
                                  int index) const;
  // Helper method for formatting multiple data buffer values.
  std::string FormatAllDbValues(generic::DataBuffer* db,
                                const std::string& format, int width) const;
  // Helper method for writing single data buffer value.
  absl::Status WriteSingleValueToDb(const std::string& str_value,
                                    generic::DataBuffer* db, std::string format,
                                    int width, int index) const;

  // Helper method for processing read memory command.
  std::string ReadMemory(int core, const std::string& str_value,
                         const std::string& format);
  // Helper method for processing write memory command.
  std::string WriteMemory(int core, const std::string& str_value1,
                          const std::string& format,
                          const std::string& str_value2);
  // Helper method used to parse a numeric string or use the string as a
  // symbol name for lookup in the loader.
  absl::StatusOr<uint64_t> GetValueFromString(int core,
                                              const std::string& str_value,
                                              int radix);
  // Reads and formats a register.
  std::string FormatRegister(int core, const std::string& reg_name) const;
  // Reads and formats $all registers - stored in reg_vec_.
  std::string FormatAllRegisters(int core) const;

  // Action point handling.
  std::string ListActionPoints();
  std::string EnableActionPointN(const std::string& index_str);
  std::string DisableActionPointN(const std::string& index_str);
  std::string ClearActionPointN(const std::string& index_str);
  std::string ClearAllActionPoints();

  std::vector<CoreAccess> core_access_;
  // Help message displayed for command 'help'.
  std::string help_message_;

  // Regular expression variables used to parse commands in the shell.
  LazyRE2 quit_re_;
  LazyRE2 core_re_;
  // Run control commands.
  LazyRE2 run_re_;
  LazyRE2 run_free_re_;
  LazyRE2 wait_re_;
  LazyRE2 step_1_re_;
  LazyRE2 step_n_re_;
  LazyRE2 halt_re_;
  // Read/write registers.
  LazyRE2 read_reg_re_;
  LazyRE2 read_reg2_re_;
  LazyRE2 write_reg_re_;
  LazyRE2 rd_vreg_re_;
  LazyRE2 wr_vreg_re_;
  // Read/write memory commands.
  LazyRE2 read_mem_re_;
  LazyRE2 read_mem2_re_;
  LazyRE2 write_mem_re_;
  // Breakpoint commands.
  LazyRE2 set_break_re_;
  LazyRE2 set_break2_re_;
  LazyRE2 set_break_n_re_;
  LazyRE2 list_break_re_;
  LazyRE2 clear_break_n_re_;
  LazyRE2 clear_break_re_;
  LazyRE2 clear_all_break_re_;
  // Watch point commands.
  LazyRE2 set_watch_re_;
  LazyRE2 set_watch2_re_;
  LazyRE2 set_watch_n_re_;
  LazyRE2 list_watch_re_;
  LazyRE2 clear_watch_re_;
  LazyRE2 clear_watch_n_re_;
  LazyRE2 clear_all_watch_re_;
  // Action point commands.
  LazyRE2 list_action_re_;
  LazyRE2 enable_action_n_re_;
  LazyRE2 disable_action_n_re_;
  LazyRE2 clear_action_n_re_;
  LazyRE2 clear_all_action_re_;
  // Branch trace
  LazyRE2 branch_trace_re_;
  // Execute commands from a file.
  LazyRE2 exec_re_;
  // Empty command.
  LazyRE2 empty_re_;
  LazyRE2 help_re_;

  int current_core_;
  uint8_t mem_buffer_[kMemBufferSize];
  std::vector<CommandFunction> command_functions_;
  std::vector<std::string> command_usage_;
  std::vector<std::string> reg_vector_;
  absl::flat_hash_set<std::string> exec_file_names_;
  std::deque<std::istream*> command_streams_;
  std::deque<std::string> previous_commands_;
  std::vector<absl::btree_map<int, ActionPointInfo>> core_action_point_info_;
  std::vector<int> core_action_point_id_;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_DEBUG_COMMAND_SHELL_H_
