#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_ACTION_POINT_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_ACTION_POINT_H_

#include <cstdint>
#include <utility>

#include "absl/container/btree_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace mpact::sim::riscv {

using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::generic::DataBufferFactory;
using ::mpact::sim::util::MemoryInterface;

// This file defines the RiscVActionPointManager class which provides the low
// level functionality required to implement breakpoint and other 'actions'
// that need to be performed when an instruction executes. It uses the RiscV
// ebreak instruction to stop execution. A handler will check whether an
// executed ebreak is due to an action point, or if it is part of a program.
// If it is an action point, the handler will call into this class to execute
// all enabled action points at that address.
//
// The ebreak is only written to memory if there is at least one enabled action
// and is replaced with the original instruction when all actions are disabled.

class RiscVActionPointManager {
 public:
  // 32 bit and 16 bit software breakpoint instructions.
  static constexpr uint32_t kEBreak32 = 0b000000000001'00000'000'00000'1110011;
  static constexpr uint16_t kEBreak16 = 0b100'1'00000'00000'10;

  // Function type for actions - void function that takes a 64 bit address.
  using ActionFcn = absl::AnyInvocable<void(uint64_t address, int id)>;
  // Callback type for function to invalidate the decoding of an instruction.
  using InvalidateFcn = absl::AnyInvocable<void(uint64_t address)>;
  // Since RiscV can be customized to a great extent, it is possible that the
  // instruction size may need to be determined in different ways for different
  // implementations. Thus, require a function to return the instruction size in
  // bytes. The function is called from SetActionPoint with the 64 bit address
  // and the 32 bit instruction word located at the desired breakpoint address.
  // For now, the function must return either 2 (bytes), 4 (bytes), or 0 (for an
  // unrecognized instruction).
  using InstructionSizeFcn =
      absl::AnyInvocable<int(uint64_t address, uint32_t instruction_word)>;

  RiscVActionPointManager(MemoryInterface *memory, InvalidateFcn invalidate_fcn,
                          InstructionSizeFcn instruction_size_fcn);
  // This constructor will use an internal instruction size function that sets
  // the instruction size based on base RiscV architecture instruction
  // encodings.
  RiscVActionPointManager(MemoryInterface *memory,
                          InvalidateFcn invalidate_fcn);
  ~RiscVActionPointManager();

  // Returns true if the given address has an action point, regardless of
  // whether it is active or not.
  bool HasActionPoint(uint64_t address) const;
  // Set action_fcn to be executed when reaching address. There may be multiple
  // actions on an instruction so an id is returned on successfully setting an
  // action point.
  absl::StatusOr<int> SetAction(uint64_t address, ActionFcn action_fcn);
  // Remove the action point with the given id.
  absl::Status ClearAction(uint64_t address, int id);
  // Enable/Disable the action point with the given id.
  absl::Status EnableAction(uint64_t address, int id);
  absl::Status DisableAction(uint64_t address, int id);
  // Return true if there is at least one enabled 'action' at this address.
  bool IsActionPointActive(uint64_t address) const;
  // Return true if the given 'action' is enabled.
  bool IsActionEnabled(uint64_t address, int id) const;
  // Remove all action points.
  void ClearAllActionPoints();
  // Perform all enabled actions for the given address. If address is not an
  // action point, not action is performed.
  void PerformActions(uint64_t address);
  // This restores the original instruction in memory, and allows it to be
  // decoded and executed, provided the address is an action point. If not,
  // no action is taken.
  void WriteOriginalInstruction(uint64_t address);
  // Store breakpoint instruction, provided the address is an action point.
  // Otherwise no action is taken.
  void WriteBreakpointInstruction(uint64_t address);

 private:
  // Struct to store information about an individual 'action'. It contains a
  // callable and a flag that indicates whether the action is enabled or not.
  struct ActionInfo {
    ActionFcn action_fcn;
    bool is_enabled;
    ActionInfo(ActionFcn action_fcn, bool is_enabled)
        : action_fcn(::std::move(action_fcn)), is_enabled(is_enabled) {}
  };

  struct ActionPointInfo {
    // Address of the action point.
    uint64_t address;
    // Size of the original instruction.
    int size;
    // The original instruction word.
    uint32_t instruction_word;
    // Id to use for the next added 'action'.
    int next_id = 0;
    // Number of 'actions' that are enabled.
    int num_active = 0;
    // Map from id to action function struct.
    absl::btree_map<int, ActionInfo *> action_map;
    ActionPointInfo(uint64_t address, int size, uint32_t instruction_word)
        : address(address), size(size), instruction_word(instruction_word) {}
  };

  void WriteBreakpointInstruction(ActionPointInfo *ap);
  void WriteOriginalInstruction(ActionPointInfo *ap);
  int GetInstructionSize(uint64_t address, uint32_t instruction_word) const;

  // Interface to program memory.
  MemoryInterface *memory_;
  // Function to be called to invalidate any stored decoding of an instruction.
  InvalidateFcn invalidate_fcn_;
  // Function to be called to determine size of an instruction.
  InstructionSizeFcn instruction_size_fcn_;
  // Data buffer pointers used to read/write instructions in memory.
  DataBuffer *db4_ = nullptr;
  DataBuffer *db2_ = nullptr;
  DataBufferFactory db_factory_;
  // Map from address to action info struct.
  absl::btree_map<uint64_t, ActionPointInfo *> action_point_map_;
};

}  // namespace mpact::sim::riscv

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_ACTION_POINT_H_
