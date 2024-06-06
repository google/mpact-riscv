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

#ifndef MPACT_RISCV_RISCV_RISCV_STATE_H_
#define MPACT_RISCV_RISCV_RISCV_STATE_H_

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mpact/sim/generic/arch_state.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/operand_interface.h"
#include "mpact/sim/generic/ref_count.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_misa.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_vector_state.h"
#include "riscv/riscv_xip_xie.h"
#include "riscv/riscv_xstatus.h"

namespace mpact {
namespace sim {
namespace riscv {

using ArchState = ::mpact::sim::generic::ArchState;
using DataBuffer = ::mpact::sim::generic::DataBuffer;
using Instruction = ::mpact::sim::generic::Instruction;
using ReferenceCount = ::mpact::sim::generic::ReferenceCount;

// Exception codes.
enum class ExceptionCode : uint64_t {
  kInstructionAddressMisaligned = 0,
  kInstructionAccessFault = 1,
  kIllegalInstruction = 2,
  kBreakpoint = 3,
  kLoadAddressMisaligned = 4,
  kLoadAccessFault = 5,
  kStoreAddressMisaligned = 6,
  kStoreAccessFault = 7,
  kEnvCallFromUMode = 8,
  kEnvCallFromSMode = 9,
  kEnvCallFromMMode = 11,
  kInstructionPageFault = 12,
  kLoadPageFault = 13,
  kStorePageFault = 15,
};

// Interrupt codes.
enum class InterruptCode : uint64_t {
  kUserSoftwareInterrupt = 0,
  kSupervisorSoftwareInterrupt = 1,
  kMachineSoftwareInterrupt = 3,
  kUserTimerInterrupt = 4,
  kSupervisorTimerInterrupt = 5,
  kMachineTimerInterrupt = 7,
  kUserExternalInterrupt = 8,
  kSupervisorExternalInterrupt = 9,
  kMachineExternalInterrupt = 11,
  kNone = std::numeric_limits<uint64_t>::max(),
};

// Isa extensions.
enum class IsaExtension : uint64_t {
  kAtomic = 1 << 0,
  kBitManipulation = 1 << 1,
  kCompressed = 1 << 2,
  kDoublePrecisionFp = 1 << 3,
  kRV32EBase = 1 << 4,
  kSinglePrecisionFp = 1 << 5,
  kGExtension = 1 << 6,
  kHypervisor = 1 << 7,
  kRVIBaseIsa = 1 << 8,
  kJExtension = 1 << 9,
  kKReserved = 1 << 10,
  kLExtension = 1 << 11,
  kIntegerMulDiv = 1 << 12,
  kUserLevelInterrupts = 1 << 13,
  kOReserved = 1 << 14,
  kPExtension = 1 << 15,
  kQuadPrecisionFp = 1 << 16,
  kRReserved = 1 << 17,
  kSupervisorMode = 1 << 18,
  kTExtension = 1 << 19,
  kUserMode = 1 << 20,
  kVectorExtension = 1 << 21,
  kWReserved = 1 << 22,
  kNonStandardExtension = 1 << 23,
  kYReserved = 1 << 24,
  kZReserved = 1 << 25,
  kNone = std::numeric_limits<uint64_t>::max(),
};

// Privilege modes.
enum class PrivilegeMode : uint64_t {
  kUser = 0b00,
  kSupervisor = 0b01,
  kMachine = 0b11,
};

// A simple load context class for convenience.
struct LoadContext : public generic::ReferenceCount {
  explicit LoadContext(DataBuffer *vdb) : value_db(vdb) {}
  ~LoadContext() override {
    if (value_db != nullptr) value_db->DecRef();
  }

  // Override the base class method so that the data buffer can be DecRef'ed
  // when the context object is recycled.
  void OnRefCountIsZero() override {
    if (value_db != nullptr) value_db->DecRef();
    value_db = nullptr;
    // Call the base class method.
    generic::ReferenceCount::OnRefCountIsZero();
  }
  // Data buffers for the value loaded from memory (byte, half, word, etc.).
  DataBuffer *value_db = nullptr;
};

// Vector load context class.
struct VectorLoadContext : public generic::ReferenceCount {
  VectorLoadContext(DataBuffer *vdb, DataBuffer *mdb, int element_width_,
                    int vstart_, int vlength_)
      : value_db(vdb),
        mask_db(mdb),
        element_width(element_width_),
        vstart(vstart_),
        vlength(vlength_) {}
  ~VectorLoadContext() override {
    if (value_db != nullptr) value_db->DecRef();
    if (mask_db != nullptr) mask_db->DecRef();
  }
  // Override the base class method so that the data buffers can be DecRef'ed
  // when the context object is recycled.
  void OnRefCountIsZero() override {
    if (value_db != nullptr) value_db->DecRef();
    value_db = nullptr;
    if (mask_db != nullptr) mask_db->DecRef();
    mask_db = nullptr;
    // Call the base class method.
    generic::ReferenceCount::OnRefCountIsZero();
  }
  // DataBuffer instances for the value loaded from memory.
  DataBuffer *value_db = nullptr;
  // Mask data buffer.
  DataBuffer *mask_db = nullptr;
  // Vector element width.
  int element_width;
  // Starting element index.
  int vstart;
  // Vector length.
  int vlength;
};

// Supported values of Xlen.
enum class RiscVXlen : uint64_t {
  RV32 = 0b01,
  RV64 = 0b10,
  RVUnknown = 4,
};

// Forward declare a template function defined in the .cc file.
template <typename T>
void CreateCsrs(RiscVState *, std::vector<RiscVCsrInterface *> &);

class RiscVFPState;

// Class that extends ArchState with RiscV specific methods. These methods
// implement RiscV specific memory operations, memory/IO fencing, system
// calls and software breakpoints.
class RiscVState : public ArchState {
 public:
  friend void CreateCsrs<uint32_t>(RiscVState *,
                                   std::vector<RiscVCsrInterface *> &);
  friend void CreateCsrs<uint64_t>(RiscVState *,
                                   std::vector<RiscVCsrInterface *> &);

  static constexpr char kFregPrefix[] = "f";
  static constexpr char kXregPrefix[] = "x";
  static constexpr char kVregPrefix[] = "v";
  static constexpr char kCsrName[] = "csr";
  static constexpr char kNextPcName[] = "next_pc";
  static constexpr char kPcName[] = "pc";

  RiscVState(absl::string_view id, RiscVXlen xlen,
             util::MemoryInterface *memory,
             util::AtomicMemoryOpInterface *atomic_memory);
  RiscVState(absl::string_view id, RiscVXlen xlen,
             util::MemoryInterface *memory)
      : RiscVState(id, xlen, memory, nullptr) {}
  RiscVState(absl::string_view id, RiscVXlen xlen)
      : RiscVState(id, xlen, nullptr, nullptr) {}
  ~RiscVState() override;

  // Deleted Constructors and operators.
  RiscVState(const RiscVState &) = delete;
  RiscVState(RiscVState &&) = delete;
  RiscVState &operator=(const RiscVState &) = delete;
  RiscVState &operator=(RiscVState &&) = delete;

  // Return a pair consisting of pointer to the named register and a bool that
  // is true if the register had to be created, and false if it was found
  // in the register map (or if nullptr is returned).
  template <typename RegisterType>
  std::pair<RegisterType *, bool> GetRegister(absl::string_view name) {
    // If the register already exists, return a pointer to the register.
    auto ptr = registers()->find(std::string(name));
    if (ptr != registers()->end())
      return std::make_pair(static_cast<RegisterType *>(ptr->second), false);
    // Create a new register and return a pointer to the object.
    return std::make_pair(AddRegister<RegisterType>(name), true);
  }

  // Specialization for RiscV vector registers.
  template <>
  std::pair<RVVectorRegister *, bool> GetRegister<RVVectorRegister>(
      absl::string_view name) {
    int vector_byte_width = vector_register_width();
    if (vector_byte_width == 0) return std::make_pair(nullptr, false);
    auto ptr = registers()->find(std::string(name));
    if (ptr != registers()->end())
      return std::make_pair(static_cast<RVVectorRegister *>(ptr->second),
                            false);
    // Create a new register and return a pointer to the object.
    return std::make_pair(
        AddRegister<RVVectorRegister>(name, vector_byte_width), true);
  }

  // Add register alias.
  template <typename RegisterType>
  absl::Status AddRegisterAlias(absl::string_view current_name,
                                absl::string_view new_name) {
    auto ptr = registers()->find(std::string(current_name));
    if (ptr == registers()->end()) {
      return absl::NotFoundError(
          absl::StrCat("Register '", current_name, "' does not exist."));
    }
    AddRegister(new_name, ptr->second);
    return absl::OkStatus();
  }

  // Methods called by instruction semantic functions to load from memory.
  void LoadMemory(const Instruction *inst, uint64_t address, DataBuffer *db,
                  Instruction *child_inst, ReferenceCount *context);
  void LoadMemory(const Instruction *inst, DataBuffer *address_db,
                  DataBuffer *mask_db, int el_size, DataBuffer *db,
                  Instruction *child_inst, ReferenceCount *context);
  // Methods called by instruction semantic functions to store to memory.
  void StoreMemory(const Instruction *inst, uint64_t address, DataBuffer *db);
  void StoreMemory(const Instruction *inst, DataBuffer *address_db,
                   DataBuffer *mask_db, int el_size, DataBuffer *db);
  // Called by the fence instruction semantic function to signal a fence
  // operation.
  void Fence(const Instruction *inst, int fm, int predecessor, int successor);
  // Synchronize instruction and data streams.
  void FenceI(const Instruction *inst);
  // System call.
  void ECall(const Instruction *inst);
  // Breakpoint.
  void EBreak(const Instruction *inst);
  // WFI
  void WFI(const Instruction *inst);
  // Ceases execution on the core. This is a non-standard instruction that
  // quiesces traffic for embedded cores before halting. The core must be reset
  // to come out of this state.
  void Cease(const Instruction *inst);
  // Trap.
  void Trap(bool is_interrupt, uint64_t trap_value, uint64_t exception_code,
            uint64_t epc, const Instruction *inst);
  // Add ebreak handler.
  void AddEbreakHandler(absl::AnyInvocable<bool(const Instruction *)> handler) {
    on_ebreak_.emplace_back(std::move(handler));
  }
  // This function is called after any event that may have caused an interrupt
  // to be registered as pending or enabled. If the interrupt can be taken
  // it registers it as available.
  void CheckForInterrupt() override;
  // This function is called when the return pc for the available interrupt
  // is known. If there is no available interrupt, it just returns.
  void TakeAvailableInterrupt(uint64_t epc);

  // Indicates that the program has returned from handling an interrupt. This
  // decrements the interrupt handler depth and should be called by the
  // implementations of mret, sret, and uret.
  void SignalReturnFromInterrupt() { --interrupt_handler_depth_; }

  // Returns the depth of the interrupt handler currently being executed, or
  // zero if no interrupt handler is being executed.
  int InterruptHandlerDepth() const { return interrupt_handler_depth_; }

  // Accessors.
  void set_memory(util::MemoryInterface *memory) { memory_ = memory; }
  util::MemoryInterface *memory() const { return memory_; }
  util::AtomicMemoryOpInterface *atomic_memory() const {
    return atomic_memory_;
  }

  void set_max_physical_address(const uint64_t max_physical_address);
  uint64_t max_physical_address() const { return max_physical_address_; }

  // Setters for handlers for ecall, and trap. The handler returns true
  // if the instruction/event was handled, and false otherwise.

  void set_on_ecall(absl::AnyInvocable<bool(const Instruction *)> callback) {
    on_ecall_ = std::move(callback);
  }

  void set_on_wfi(absl::AnyInvocable<bool(const Instruction *)> callback) {
    on_wfi_ = std::move(callback);
  }

  void set_on_cease(absl::AnyInvocable<bool(const Instruction *)> callback) {
    on_cease_ = std::move(callback);
  }

  void set_on_trap(
      absl::AnyInvocable<bool(bool /*is_interrupt*/, uint64_t /*trap_value*/,
                              uint64_t /*exception_code*/, uint64_t /*epc*/,
                              const Instruction *)>
          callback) {
    on_trap_ = std::move(callback);
  }

  int flen() const { return flen_; }
  RiscVXlen xlen() const { return xlen_; }
  RiscVVectorState *rv_vector() const { return rv_vector_; }
  void set_rv_vector(RiscVVectorState *value) { rv_vector_ = value; }
  RiscVFPState *rv_fp() const { return rv_fp_; }
  void set_rv_fp(RiscVFPState *value) { rv_fp_ = value; }
  void set_vector_register_width(int value) { vector_register_width_ = value; }
  int vector_register_width() const { return vector_register_width_; }

  RiscVCsrSet *csr_set() const { return csr_set_; }

  PrivilegeMode privilege_mode() const { return privilege_mode_; }
  void set_privilege_mode(PrivilegeMode privilege_mode) {
    privilege_mode_ = privilege_mode;
  }

  // Returns true if an interrupt is available for the core to take or false
  // otherwise.
  inline bool is_interrupt_available() const { return is_interrupt_available_; }
  // Resets the is_interrupt_available flag to false. This should only be called
  // when resetting the RISCV core, as 'is_interrupt_available' is Normally
  // reset during the interrupt handling flow.
  inline void reset_is_interrupt_available() {
    is_interrupt_available_ = false;
  }

  // Getters for select CSRs.
  RiscVMStatus *mstatus() const { return mstatus_; }
  RiscVMIsa *misa() const { return misa_; }
  RiscVMIp *mip() const { return mip_; }
  RiscVMIe *mie() const { return mie_; }
  RiscVCsrInterface *mtvec() const { return mtvec_; }
  RiscVCsrInterface *mepc() const { return mepc_; }
  RiscVCsrInterface *mcause() const { return mcause_; }
  RiscVCsrInterface *medeleg() const { return medeleg_; }
  RiscVCsrInterface *mideleg() const { return mideleg_; }
  RiscVSIp *sip() const { return sip_; }
  RiscVSIe *sie() const { return sie_; }
  RiscVCsrInterface *stvec() const { return stvec_; }
  RiscVCsrInterface *sepc() const { return sepc_; }
  RiscVCsrInterface *scause() const { return scause_; }
  RiscVCsrInterface *sideleg() const { return sideleg_; }

 private:
  InterruptCode PickInterrupt(uint32_t interrupts);
  RiscVXlen xlen_;
  uint64_t max_physical_address_;
  RiscVVectorState *rv_vector_ = nullptr;
  RiscVFPState *rv_fp_ = nullptr;
  // Program counter register.
  generic::RegisterBase *pc_;
  // Operands used to access pc values generically. Note, the pc value may read
  // as the address of the next instruction during execution of an instruction,
  // so the address of the instruction executing should be used instead.
  generic::SourceOperandInterface *pc_src_operand_ = nullptr;
  generic::DestinationOperandInterface *pc_dst_operand_ = nullptr;
  int vector_register_width_ = 0;
  int flen_ = 0;
  util::FlatDemandMemory *owned_memory_ = nullptr;
  util::MemoryInterface *memory_ = nullptr;
  util::AtomicMemoryOpInterface *atomic_memory_ = nullptr;
  RiscVCsrSet *csr_set_ = nullptr;
  std::vector<absl::AnyInvocable<bool(const Instruction *)>> on_ebreak_;
  absl::AnyInvocable<bool(const Instruction *)> on_ecall_;
  absl::AnyInvocable<bool(bool, uint64_t, uint64_t, uint64_t,
                          const Instruction *)>
      on_trap_;
  absl::AnyInvocable<bool(const Instruction *)> on_wfi_;
  absl::AnyInvocable<bool(const Instruction *)> on_cease_;
  std::vector<RiscVCsrInterface *> csr_vec_;
  // For interrupt handling.
  bool is_interrupt_available_ = false;
  int interrupt_handler_depth_ = 0;
  InterruptCode available_interrupt_code_ = InterruptCode::kNone;
  // By default, execute in machine mode.
  PrivilegeMode privilege_mode_ = PrivilegeMode::kMachine;
  // Handles to frequently used CSRs.
  RiscVMStatus *mstatus_ = nullptr;
  RiscVMIsa *misa_ = nullptr;
  RiscVMIp *mip_ = nullptr;
  RiscVMIe *mie_ = nullptr;
  RiscVCsrInterface *mtvec_ = nullptr;
  RiscVCsrInterface *mepc_ = nullptr;
  RiscVCsrInterface *mcause_ = nullptr;
  RiscVCsrInterface *medeleg_ = nullptr;
  RiscVCsrInterface *mideleg_ = nullptr;
  RiscVSIp *sip_ = nullptr;
  RiscVSIe *sie_ = nullptr;
  RiscVCsrInterface *stvec_ = nullptr;
  RiscVCsrInterface *sepc_ = nullptr;
  RiscVCsrInterface *scause_ = nullptr;
  RiscVCsrInterface *sideleg_ = nullptr;
};

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_RISCV_STATE_H_
