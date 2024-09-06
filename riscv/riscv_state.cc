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

#include "riscv/riscv_state.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mpact/sim/generic/arch_state.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "riscv/riscv_counter_csr.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_misa.h"
#include "riscv/riscv_pmp.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_sim_csrs.h"
#include "riscv/riscv_xip_xie.h"
#include "riscv/riscv_xstatus.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::operator*;  // NOLINT: used below (clang error).

// These helper templates are used to store information about the CSR registers
// for 32 and 64 bit versions of RiscV.
template <typename T>
struct CsrInfo {};

template <>
struct CsrInfo<uint32_t> {
  using T = uint32_t;
  static constexpr T kMhartidRMask = std::numeric_limits<T>::max();
  static constexpr T kMhartidWMask = 0;
  static constexpr T kMstatusInitialValue = 2000;
  static constexpr T kUstatusRMask = 0x11;
  static constexpr T kUstatusWMask = 0x11;
  static constexpr T kSstatusRMask = 0x800d'e133;
  static constexpr T kSstatusWMask = 0x800d'e133;
  static constexpr T kMisaInitialValue =
      (*RiscVXlen::RV32 << 30) | *IsaExtension::kIntegerMulDiv |
      *IsaExtension::kRVIBaseIsa | *IsaExtension::kGExtension |
      *IsaExtension::kSinglePrecisionFp | *IsaExtension::kDoublePrecisionFp |
      *IsaExtension::kCompressed | *IsaExtension::kAtomic |
      *IsaExtension::kSupervisorMode;
  static constexpr T kMisaRMask = 0xc3ff'ffff;
  static constexpr T kMisaWMask = 0x0;
  // Can't delegate env call from M-mode.
  static constexpr T kMEdelegRMask = 0x0000'b3ff;
  static constexpr T kMEdelegWMask = 0x0000'b3ff;
  static constexpr T kMIdelegRMask = 0x0bbb;
  static constexpr T kMIdelegWMask = 0x0bbb;
};

template <>
struct CsrInfo<uint64_t> {
  using T = uint64_t;
  static constexpr T kMhartidRMask = std::numeric_limits<T>::max();
  static constexpr T kMhartidWMask = 0;
  static constexpr T kMstatusInitialValue = 0x0000'000a'0000'2000ULL;
  static constexpr T kUstatusRMask = 0x11;
  static constexpr T kUstatusWMask = 0x11;
  static constexpr T kSstatusRMask = 0x8000'0003'000d'e133ULL;
  static constexpr T kSstatusWMask = 0x8000'0003'000d'e133ULL;
  static constexpr T kMisaInitialValue =
      (*RiscVXlen::RV64 << 62) | *IsaExtension::kIntegerMulDiv |
      *IsaExtension::kRVIBaseIsa | *IsaExtension::kGExtension |
      *IsaExtension::kSinglePrecisionFp | *IsaExtension::kDoublePrecisionFp |
      *IsaExtension::kCompressed | *IsaExtension::kAtomic |
      *IsaExtension::kSupervisorMode;

  static constexpr T kMisaRMask = 0xc000'0000'03ff'ffffULL;
  static constexpr T kMisaWMask = 0x0;
  // Can't delegate env call from M-mode.
  static constexpr T kMEdelegRMask = 0x0000'0000'0000'b3ffULL;
  static constexpr T kMEdelegWMask = 0x0000'0000'0000'b3ffULL;
  static constexpr T kMIdelegRMask = 0x0bbb;
  static constexpr T kMIdelegWMask = 0x0bbb;
};

// Three templated helper functions used to create individual CSRs.

// This creates the CSR and assigns it to a pointer in the state object. Type
// can be inferred from the state object pointer.
template <typename T, typename... Ps>
T *CreateCsr(RiscVState *state, T *&ptr,
             std::vector<RiscVCsrInterface *> &csr_vec, Ps... pargs) {
  auto *csr = new T(pargs...);
  auto result = state->csr_set()->AddCsr(csr);
  if (!result.ok()) {
    LOG(ERROR) << absl::StrCat("Failed to add csr '", csr->name(),
                               "': ", result.message());
    delete csr;
    return nullptr;
  }
  csr_vec.push_back(csr);
  ptr = csr;
  return csr;
}

// This creates the CSR and assigns it to a pointer in the state object, however
// that pointer is of abstract type, so the CSR type cannot be inferred, but
// has to be specified in the call.
template <typename T, typename... Ps>
T *CreateCsr(RiscVState *state, RiscVCsrInterface *&ptr,
             std::vector<RiscVCsrInterface *> &csr_vec, Ps... pargs) {
  auto *csr = new T(pargs...);
  auto result = state->csr_set()->AddCsr(csr);
  if (!result.ok()) {
    LOG(ERROR) << absl::StrCat("Failed to add csr '", csr->name(),
                               "': ", result.message());
    delete csr;
    return nullptr;
  }
  csr_vec.push_back(csr);
  ptr = csr;
  return csr;
}

// This creates the CSR, but does not assign it to a pointer in the state
// object. That means the type cannot be inferred, but has to be specified
// in the call.
template <typename T, typename... Ps>
T *CreateCsr(RiscVState *state, std::vector<RiscVCsrInterface *> &csr_vec,
             Ps... pargs) {
  auto *csr = new T(pargs...);
  auto result = state->csr_set()->AddCsr(csr);
  if (!result.ok()) {
    LOG(ERROR) << absl::StrCat("Failed to add csr '", csr->name(),
                               "': ", result.message());
    delete csr;
    return nullptr;
  }
  csr_vec.push_back(csr);
  return csr;
}

// Templated helper function that is used to create the set of CSRs needed
// for simulation.
template <typename T>
void CreateCsrs(RiscVState *state, std::vector<RiscVCsrInterface *> &csr_vec) {
  absl::Status result;
  // Create CSRs.
  // misa
  auto *misa = CreateCsr(state, state->misa_, csr_vec,
                         CsrInfo<T>::kMisaInitialValue, state);
  CHECK_NE(misa, nullptr);
  // mtvec
  CHECK_NE(CreateCsr<RiscVSimpleCsr<T>>(state, state->mtvec_, csr_vec, "mtvec",
                                        RiscVCsrEnum::kMTvec, 0, state),
           nullptr);
  // mcause
  CHECK_NE(
      CreateCsr<RiscVSimpleCsr<T>>(state, state->mcause_, csr_vec, "mcause",
                                   RiscVCsrEnum::kMCause, 0, state),
      nullptr);

  // Mip and Mie are always 32 bit.
  // mip
  auto *mip = CreateCsr(state, state->mip_, csr_vec, 0, state);
  CHECK_NE(mip, nullptr);

  // mie
  auto *mie = CreateCsr(state, state->mie_, csr_vec, 0, state);
  CHECK_NE(mie, nullptr);

  // mhartid
  CHECK_NE(CreateCsr<RiscVSimpleCsr<T>>(
               state, csr_vec, "mhartid", RiscVCsrEnum::kMHartId, 0,
               CsrInfo<T>::kMhartidRMask, CsrInfo<T>::kMhartidWMask, state),
           nullptr);

  // mepc
  CHECK_NE(CreateCsr<RiscVSimpleCsr<T>>(state, state->mepc_, csr_vec, "mepc",
                                        RiscVCsrEnum::kMEpc, 0, state),
           nullptr);

  // mscratch
  CHECK_NE(CreateCsr<RiscVSimpleCsr<T>>(state, csr_vec, "mscratch",
                                        RiscVCsrEnum::kMScratch, 0, state),
           nullptr);

  // medeleg - machine mode exception delegation register.
  CHECK_NE(CreateCsr<RiscVSimpleCsr<T>>(state, state->medeleg_, csr_vec,
                                        "medeleg", RiscVCsrEnum::kMEDeleg, 0,
                                        CsrInfo<T>::kMEdelegRMask,
                                        CsrInfo<T>::kMEdelegWMask, state),
           nullptr);

  // mideleg - machine mode interrupt delegation register.
  auto *mideleg = CreateCsr<RiscVSimpleCsr<T>>(
      state, state->mideleg_, csr_vec, "mideleg", RiscVCsrEnum::kMIDeleg, 0,
      CsrInfo<T>::kMIdelegRMask, CsrInfo<T>::kMIdelegWMask, state);
  CHECK_NE(mideleg, nullptr);

  // mstatus
  auto *mstatus =
      CreateCsr(state, state->mstatus_, csr_vec,
                CsrInfo<uint64_t>::kMstatusInitialValue, state, misa);
  CHECK_NE(mstatus, nullptr);
  // mtval
  CHECK_NE(CreateCsr<RiscVSimpleCsr<T>>(state, csr_vec, "mtval",
                                        RiscVCsrEnum::kMTval, 0, state),
           nullptr);

  // minstret/minstreth
  auto *minstret = CreateCsr<RiscVCounterCsr<T, RiscVState>>(
      state, csr_vec, "minstret", RiscVCsrEnum ::kMInstret, state);
  CHECK_NE(minstret, nullptr);
  if (sizeof(T) == sizeof(uint32_t)) {
    CHECK_NE(CreateCsr<RiscVCounterCsrHigh<RiscVState>>(
                 state, csr_vec, "minstreth", RiscVCsrEnum::kMInstretH, state,
                 reinterpret_cast<RiscVCounterCsr<uint32_t, RiscVState> *>(
                     minstret)),
             nullptr);
  }
  // mcycle/mcycleh
  auto *mcycle = CreateCsr<RiscVCounterCsr<T, RiscVState>>(
      state, csr_vec, "mcycle", RiscVCsrEnum::kMCycle, state);
  CHECK_NE(mcycle, nullptr);
  if (sizeof(T) == sizeof(uint32_t)) {
    CHECK_NE(
        CreateCsr<RiscVCounterCsrHigh<RiscVState>>(
            state, csr_vec, "mcycleh", RiscVCsrEnum::kMCycleH, state,
            reinterpret_cast<RiscVCounterCsr<uint32_t, RiscVState> *>(mcycle)),
        nullptr);
  }

  // Supervisor level CSRs

  // sstatus
  CHECK_NE(CreateCsr<RiscVSStatus>(state, csr_vec, mstatus, state), nullptr);

  // sip and sie are always 32 bit.
  // sip - supervisor interrupt pending register.
  CHECK_NE(CreateCsr(state, state->sip_, csr_vec, mip, mideleg, state),
           nullptr);

  // sie - supervisor interrupt enable register.
  CHECK_NE(CreateCsr(state, state->sie_, csr_vec, mie, mideleg, state),
           nullptr);

  // stvec - supervisor trap vector register.
  CHECK_NE(CreateCsr<RiscVSimpleCsr<T>>(state, state->stvec_, csr_vec, "stvec",
                                        RiscVCsrEnum::kSTvec, 0, state),
           nullptr);

  // scause - supervisor trap cause register.
  CHECK_NE(
      CreateCsr<RiscVSimpleCsr<T>>(state, state->scause_, csr_vec, "scause",
                                   RiscVCsrEnum::kSCause, 0, state),
      nullptr);

  // sepc - supervisor exception pc register.
  CHECK_NE(CreateCsr<RiscVSimpleCsr<T>>(state, state->sepc_, csr_vec, "sepc",
                                        RiscVCsrEnum::kSEpc, 0, state),
           nullptr);
  // stval
  CHECK_NE(CreateCsr<RiscVSimpleCsr<T>>(state, csr_vec, "stval",
                                        RiscVCsrEnum::kSTval, 0, state),
           nullptr);

  // sscratch
  CHECK_NE(CreateCsr<RiscVSimpleCsr<T>>(state, csr_vec, "sscratch",
                                        RiscVCsrEnum::kSScratch, 0, state),
           nullptr);

  // sideleg - machine mode interrupt delegation register.
  CHECK_NE(CreateCsr<RiscVSimpleCsr<T>>(state, state->sideleg_, csr_vec,
                                        "sideleg", RiscVCsrEnum::kSIDeleg, 0,
                                        CsrInfo<T>::kMIdelegRMask,
                                        CsrInfo<T>::kMIdelegWMask, state),
           nullptr);

  // User level CSRs

  // ustatus
  CHECK_NE(CreateCsr<RiscVSimpleCsr<T>>(
               state, csr_vec, "ustatus", RiscVCsrEnum::kUStatus, 0,
               CsrInfo<T>::kUstatusRMask, CsrInfo<T>::kUstatusWMask, state),
           nullptr);

  // PMP CSRs
  state->pmp_ = new RiscVPmp(state);
  state->pmp_->CreatePmpCsrs<T, RiscVCsrEnum>(state->csr_set());

  // Simulator CSRs

  // Access current privilege mode.
  CHECK_NE(CreateCsr<RiscVSimModeCsr>(state, csr_vec, "$mode",
                                      RiscVCsrEnum::kSimMode, state),
           nullptr);
}

// This value is in the RV32ISA manual to support MMU, although in "BARE" mode
// only the bottom 32-bit is valid.
constexpr uint64_t kRiscv32MaxMemorySize = 0x3f'ffff'ffffULL;
constexpr uint64_t kRiscv64MaxMemorySize = 0x00ff'ffff'ffff'ffffULL;

RiscVState::RiscVState(absl::string_view id, RiscVXlen xlen,
                       util::MemoryInterface *memory,
                       util::AtomicMemoryOpInterface *atomic_memory)
    : ArchState(id),
      xlen_(xlen),
      memory_(memory),
      atomic_memory_(atomic_memory),
      csr_set_(new RiscVCsrSet()),
      counter_interrupts_taken_("interrupts_taken", 0),
      counter_interrupt_returns_("interrupt_returns", 0) {
  CHECK_OK(AddCounter(&counter_interrupt_returns_));
  CHECK_OK(AddCounter(&counter_interrupts_taken_));
  DataBuffer *db = nullptr;
  switch (xlen_) {
    case RiscVXlen::RV32: {
      auto *pc32 = GetRegister<RV32Register>(kPcName).first;
      pc_src_operand_ = pc32->CreateSourceOperand();
      pc_dst_operand_ = pc32->CreateDestinationOperand(0);
      pc_ = pc32;
      db = db_factory()->Allocate<RV32Register::ValueType>(1);
      db->Set<uint32_t>(0, 0);
      CreateCsrs<uint32_t>(this, csr_vec_);
      max_physical_address_ = kRiscv32MaxMemorySize;
      break;
    }
    case RiscVXlen::RV64: {
      auto *pc64 = GetRegister<RV64Register>(kPcName).first;
      pc_src_operand_ = pc64->CreateSourceOperand();
      pc_dst_operand_ = pc64->CreateDestinationOperand(0);
      pc_ = pc64;
      db = db_factory()->Allocate<RV64Register::ValueType>(1);
      db->Set<uint64_t>(0, 0);
      CreateCsrs<uint64_t>(this, csr_vec_);
      max_physical_address_ = kRiscv64MaxMemorySize;
      break;
    }
    default:
      LOG(ERROR) << "Unsupported xlen";
      return;
  }

  set_pc_operand(pc_src_operand_);
  pc_->SetDataBuffer(db);
  db->DecRef();

  // Set the flen value based on the ISA features.
  // Note, the FP register class must be set appropriately as well.
  auto result = csr_set()->GetCsr("misa");
  if (!result.ok()) {
    LOG(ERROR) << "Failed to get misa register";
    return;
  }
  auto *misa = result.value();
  auto misa_value = misa->AsUint32();
  if (misa_value & *IsaExtension::kSinglePrecisionFp) {
    flen_ = 32;
  }
  if (misa_value & *IsaExtension::kDoublePrecisionFp) {
    flen_ = 64;
  }
  if (misa_value & *IsaExtension::kQuadPrecisionFp) {
    flen_ = 128;
  }
}

RiscVState::~RiscVState() {
  delete pc_src_operand_;
  delete pc_dst_operand_;
  delete csr_set_;
  delete pmp_;
  for (auto *csr : csr_vec_) {
    delete csr;
  }
  csr_vec_.clear();
}

void RiscVState::set_max_physical_address(uint64_t max_physical_address) {
  switch (xlen_) {
    case RiscVXlen::RV32:
      max_physical_address_ =
          std::min(max_physical_address, kRiscv32MaxMemorySize);
      break;
    case RiscVXlen::RV64:
      max_physical_address_ =
          std::min(max_physical_address, kRiscv64MaxMemorySize);
      break;
    default:
      break;
  }
}

void RiscVState::LoadMemory(const Instruction *inst, uint64_t address,
                            DataBuffer *db, Instruction *child_inst,
                            ReferenceCount *context) {
  if (address > max_physical_address_) {
    Trap(/*is_interrupt*/ false, address, *ExceptionCode::kLoadAccessFault,
         inst->address(), inst);
    return;
  }
  memory_->Load(address, db, child_inst, context);
}

void RiscVState::LoadMemory(const Instruction *inst, DataBuffer *address_db,
                            DataBuffer *mask_db, int el_size, DataBuffer *db,
                            Instruction *child_inst, ReferenceCount *context) {
  for (auto address : address_db->Get<uint64_t>()) {
    if (address > max_physical_address_) {
      Trap(/*is_interrupt*/ false, address, *ExceptionCode::kLoadAccessFault,
           inst->address(), inst);
      return;
    }
  }
  memory_->Load(address_db, mask_db, el_size, db, child_inst, context);
}

void RiscVState::StoreMemory(const Instruction *inst, uint64_t address,
                             DataBuffer *db) {
  if (address > max_physical_address_) {
    Trap(/*is_interrupt*/ false, address, *ExceptionCode::kStoreAccessFault,
         inst->address(), inst);
    return;
  }
  memory_->Store(address, db);
}

void RiscVState::StoreMemory(const Instruction *inst, DataBuffer *address_db,
                             DataBuffer *mask_db, int el_size, DataBuffer *db) {
  for (auto address : address_db->Get<uint64_t>()) {
    if (address > max_physical_address_) {
      Trap(/*is_interrupt*/ false, address, *ExceptionCode::kStoreAccessFault,
           inst->address(), inst);
      return;
    }
  }
  memory_->Store(address_db, mask_db, el_size, db);
}

void RiscVState::Fence(const Instruction *inst, int fm, int predecessor,
                       int successor) {
  // TODO: Add fence operation once operations have non-zero latency.
}

void RiscVState::FenceI(const Instruction *inst) {
  // TODO: Add instruction fence operation when needed.
}

void RiscVState::ECall(const Instruction *inst) {
  if (on_ecall_ != nullptr) {
    auto res = on_ecall_(inst);
    if (res) return;
  }

  std::string where = (inst != nullptr)
                          ? absl::StrCat(absl::Hex(inst->address()))
                          : "unknown location";

  ExceptionCode code;
  switch (privilege_mode()) {
    case PrivilegeMode::kUser:
      code = ExceptionCode::kEnvCallFromUMode;
      break;
    case PrivilegeMode::kSupervisor:
      code = ExceptionCode::kEnvCallFromSMode;
      break;
    case PrivilegeMode::kMachine:
      code = ExceptionCode::kEnvCallFromMMode;
      break;
    default:
      LOG(ERROR) << "Unknown privilege mode";
      return;
  }

  uint64_t epc = inst->address();
  Trap(/*is_interrupt*/ false, 0, *code, epc, inst);
}

void RiscVState::EBreak(const Instruction *inst) {
  for (auto &handler : on_ebreak_) {
    bool res = handler(inst);
    if (res) return;
  }

  // Set the return address to the current instruction.
  auto epc = (inst != nullptr) ? inst->address() : 0;
  Trap(/*is_interrupt=*/false, 0, 3, epc, inst);
}

void RiscVState::WFI(const Instruction *inst) {
  if (on_wfi_ != nullptr) {
    bool res = on_wfi_(inst);
    if (res) return;
  }

  std::string where = (inst != nullptr)
                          ? absl::StrCat(absl::Hex(inst->address()))
                          : "unknown location";

  LOG(INFO) << "No handler for wfi: treating as nop: " << where;
}

void RiscVState::Cease(const Instruction *inst) {
  if (on_cease_ != nullptr) {
    const bool res = on_cease_(inst);
    if (res) return;
  }

  // If no handler is specidied, then CEASE is treated as an infinite loop.
  auto current_xlen = xlen();
  auto *db = pc_dst_operand_->AllocateDataBuffer();
  if (current_xlen == RiscVXlen::RV32) {
    db->SetSubmit<uint32_t>(0, static_cast<uint32_t>(inst->address()));
  } else if (current_xlen == RiscVXlen::RV64) {
    db->SetSubmit<uint64_t>(0, inst->address());
  } else {
    LOG(ERROR) << "Unknown xlen";
  }

  const std::string where = (inst != nullptr)
                                ? absl::StrCat(absl::Hex(inst->address()))
                                : "unknown location";

  LOG(INFO) << "No handler for cease: treating as an infinite loop: " << where;
}

void RiscVState::Trap(bool is_interrupt, uint64_t trap_value,
                      uint64_t exception_code, uint64_t epc,
                      const Instruction *inst) {
  if (on_trap_ != nullptr) {
    bool res = on_trap_(is_interrupt, trap_value, exception_code, epc, inst);
    if (res) return;
  }

  // Default behavior.

  // Determine where the interrupt should be taken. Default is machine mode,
  // but if the interrupt/exception is delegated, it may be taken in supervisor
  // mode (or if supervisor mode delegates it, in user mode). Traps cannot be
  // delegated lower than the current level of execution. E.g., a trap in
  // machine mode, cannot be delegated to supervisor mode. Additionally, an
  // interrupt that is delegated, is masked at the level of the delegator, i.e.,
  // an interrupt delegated to supervisor mode will never be taken while
  // executing machine mode code. This, however, is handled in the
  // CheckForInterrupt method.

  PrivilegeMode destination_mode = PrivilegeMode::kMachine;

  // Determine the destination execution mode.
  if (is_interrupt) {
    if (mideleg_->AsUint32() & (1U << exception_code)) {
      destination_mode = PrivilegeMode::kSupervisor;
      if (sideleg_->AsUint32() & (1U << exception_code)) {
        LOG(ERROR)
            << "Full support for user mode interrupts not implemented yet";
        destination_mode = PrivilegeMode::kUser;
      }
    }
  } else {
    // Exceptions are not delegated to a lower privilege level.
    if ((privilege_mode() != PrivilegeMode::kMachine) &&
        (medeleg_->AsUint64() & (1ULL << exception_code))) {
      destination_mode = PrivilegeMode::kSupervisor;
      // There is no support for user level exceptions.
    }
  }

  // Based on the destination privilege mode, select the CSRs that will be
  // used.
  RiscVCsrInterface *epc_csr = nullptr;
  RiscVCsrInterface *cause_csr = nullptr;
  RiscVCsrInterface *tvec_csr = nullptr;
  if (destination_mode == PrivilegeMode::kMachine) {
    epc_csr = mepc_;
    cause_csr = mcause_;
    tvec_csr = mtvec_;
  } else if (destination_mode == PrivilegeMode::kSupervisor) {
    epc_csr = sepc_;
    cause_csr = scause_;
    tvec_csr = stvec_;
  } else {
    LOG(ERROR) << "Invalid destination execution mode";
    return;
  }

  // Get trap destination.
  int trap_vector_mode = tvec_csr->AsUint64() & 0x3ULL;
  uint64_t trap_target = tvec_csr->AsUint64() & ~0x3ULL;
  if (trap_vector_mode == 1) {
    trap_target += 4 * exception_code;
  }

  // Set xepc.
  epc_csr->Set(epc);
  // Set xcause.
  cause_csr->Set(exception_code);
  auto current_xlen = xlen();
  if (is_interrupt) {
    if (current_xlen == RiscVXlen::RV32) {
      cause_csr->SetBits(static_cast<uint32_t>(0x8000'0000));
    } else if (current_xlen == RiscVXlen::RV64) {
      cause_csr->SetBits(static_cast<uint64_t>(0x8000'0000'0000'0000ULL));
    } else {
      LOG(ERROR) << "Unknown xlen";
    }
  }
  // Set mstatus bits accordingly.

  if (destination_mode == PrivilegeMode::kMachine) {
    // Set the privilege mode to return to after the interrupt.
    mstatus_->set_mpp(*(privilege_mode()));
    // Save the current interrupt enable to mpie.
    mstatus_->set_mpie(mstatus_->mie());
    // Disable further interrupts.
    mstatus_->set_mie(0);
  } else if (destination_mode == PrivilegeMode::kSupervisor) {
    // Set the privilege mode to return to after the interrupt.
    mstatus_->set_spp(*privilege_mode() & 0x1);
    // Save the current interrupt enable to mpie.
    mstatus_->set_spie(mstatus_->sie());
    // Disable further interrupts.
    mstatus_->set_sie(0);
  }

  // Advance data buffer delay line until empty. Flush pending writes to
  // register and possibly pc.
  while (!data_buffer_delay_line()->IsEmpty()) {
    data_buffer_delay_line()->Advance();
  }

  // Update the PC.
  auto *db = pc_dst_operand_->AllocateDataBuffer();
  if (current_xlen == RiscVXlen::RV32) {
    db->SetSubmit<uint32_t>(0, static_cast<uint32_t>(trap_target));
  } else if (current_xlen == RiscVXlen::RV64) {
    db->SetSubmit<uint64_t>(0, trap_target);
  } else {
    LOG(ERROR) << "Unknown xlen";
  }
  set_privilege_mode(destination_mode);
  mstatus_->Submit();
}

// CheckForInterrupt is called whenever any relevant bits in the interrupt
// enable and set bits are changed. It should always be scheduled to execute
// from the function_delay_line, that way it is executed after an instruction
// has completed execution.
void RiscVState::CheckForInterrupt() {
  // Compute interrupts enabled at each level using [ms]ideleg.
  uint32_t interrupts = mip_->AsUint32() & mie_->AsUint32();
  if (interrupts == 0) return;

  uint32_t m_interrupts = interrupts & ~mideleg_->AsUint32();
  interrupts = interrupts & mideleg_->AsUint32();
  uint32_t s_interrupts = interrupts & ~sideleg_->AsUint32();
  uint32_t u_interrupts = interrupts & sideleg_->AsUint32();

  auto priv_mode = privilege_mode();

  // Interrupt at level L is enabled if current priv level is < L, or if
  // mstatus->Lie is set. If priv level > L, then enable is off.
  bool mie = (priv_mode != PrivilegeMode::kMachine) || mstatus_->mie();
  bool sie = ((*priv_mode < *PrivilegeMode::kSupervisor) || mstatus_->sie()) &&
             (priv_mode != PrivilegeMode::kMachine);
  bool uie = mstatus_->uie() && (priv_mode == PrivilegeMode::kUser);

  // No interrupts can be taken.
  if (!(mie || sie || uie)) return;

  InterruptCode code;
  if (mie && (m_interrupts != 0)) {
    // Take an interrupt to machine mode.
    code = PickInterrupt(m_interrupts);
  } else if (sie && (s_interrupts != 0)) {
    // Take an interrupt to supervisor mode.
    code = PickInterrupt(s_interrupts);
  } else if (uie && (u_interrupts != 0)) {
    // Take an interrupt to user mode.
    code = PickInterrupt(u_interrupts);
    LOG(ERROR) << "User mode interrupts not supported yet";
    return;
  } else {
    // No eligible interrupts to take.
    return;
  }

  available_interrupt_code_ = code;
  is_interrupt_available_ = true;
}

// Take the interrupt that is pending.
void RiscVState::TakeAvailableInterrupt(uint64_t epc) {
  // Make sure an interrupt is set as pending by CheckForInterrupt.
  if (!is_interrupt_available_) return;
  // Initiate the interrupt.
  Trap(/*is_interrupt*/ true, 0, *available_interrupt_code_, epc, nullptr);
  // Clear pending interrupt.
  is_interrupt_available_ = false;
  counter_interrupts_taken_.Increment(1);
  available_interrupt_code_ = InterruptCode::kNone;
}

// The priority order of the interrupts are as follows:
// mei, msi, mti, sei, ssi, sti, uei, usi, uti.
InterruptCode RiscVState::PickInterrupt(uint32_t interrupts) {
  if (interrupts & (1 << *InterruptCode::kMachineExternalInterrupt))
    return InterruptCode::kMachineExternalInterrupt;
  if (interrupts & (1 << *InterruptCode::kMachineSoftwareInterrupt))
    return InterruptCode::kMachineSoftwareInterrupt;
  if (interrupts & (1 << *InterruptCode::kMachineTimerInterrupt))
    return InterruptCode::kMachineTimerInterrupt;

  if (interrupts & (1 << *InterruptCode::kSupervisorExternalInterrupt))
    return InterruptCode::kSupervisorExternalInterrupt;
  if (interrupts & (1 << *InterruptCode::kSupervisorSoftwareInterrupt))
    return InterruptCode::kSupervisorSoftwareInterrupt;
  if (interrupts & (1 << *InterruptCode::kSupervisorTimerInterrupt))
    return InterruptCode::kSupervisorTimerInterrupt;

  if (interrupts & (1 << *InterruptCode::kUserExternalInterrupt))
    return InterruptCode::kUserExternalInterrupt;
  if (interrupts & (1 << *InterruptCode::kUserSoftwareInterrupt))
    return InterruptCode::kUserSoftwareInterrupt;
  if (interrupts & (1 << *InterruptCode::kUserTimerInterrupt))
    return InterruptCode::kUserTimerInterrupt;

  return InterruptCode::kNone;
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
