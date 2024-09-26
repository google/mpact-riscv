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

#include "riscv/riscv_renode.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mpact/sim/generic/core_debug_interface.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/proto/component_data.pb.h"
#include "mpact/sim/proto/component_data.proto.h"
#include "mpact/sim/util/memory/atomic_memory.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "mpact/sim/util/memory/memory_watcher.h"
#include "mpact/sim/util/memory/single_initiator_router.h"
#include "mpact/sim/util/other/instruction_profiler.h"
#include "riscv/debug_command_shell.h"
#include "riscv/riscv32_decoder.h"
#include "riscv/riscv64_decoder.h"
#include "riscv/riscv_arm_semihost.h"
#include "riscv/riscv_cli_forwarder.h"
#include "riscv/riscv_clint.h"
#include "riscv/riscv_debug_info.h"
#include "riscv/riscv_debug_interface.h"
#include "riscv/riscv_instrumentation_control.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_register_aliases.h"
#include "riscv/riscv_renode_cli_top.h"
#include "riscv/riscv_renode_register_info.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_top.h"
#include "riscv/stoull_wrapper.h"
#include "src/google/protobuf/text_format.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::proto::ComponentData;
using ::mpact::sim::proto::ComponentValueEntry;
using ::mpact::sim::riscv::RiscVClint;
using ::mpact::sim::util::AtomicMemoryOpInterface;
using ::mpact::sim::util::MemoryWatcher;

using HaltReasonValueType =
    ::mpact::sim::generic::CoreDebugInterface::HaltReasonValueType;
using HaltReason = ::mpact::sim::generic::CoreDebugInterface::HaltReason;
using RunStatus = ::mpact::sim::generic::CoreDebugInterface::RunStatus;

// Configuration names.
constexpr std::string_view kMemoryBase = "memoryBase";
constexpr std::string_view kMemorySize = "memorySize";
constexpr std::string_view kClintMMRBase = "clintMMRBase";
constexpr std::string_view kCLIPort = "cliPort";
constexpr std::string_view kWaitForCLI = "waitForCLI";
constexpr std::string_view kInstProfile = "instProfile";
constexpr std::string_view kMemProfile = "memProfile";
constexpr std::string_view kStackEnd = "stackEnd";
constexpr std::string_view kStackSize = "stackSize";
constexpr std::string_view kICache = "iCache";
constexpr std::string_view kDCache = "dCache";

constexpr char kStackEndSymbolName[] = "__stack_end";
constexpr char kStackSizeSymbolName[] = "__stack_size";

RiscVRenode::RiscVRenode(std::string name, MemoryInterface *renode_sysbus,
                         RiscVXlen xlen)
    : name_(name), renode_sysbus_(renode_sysbus) {
  router_ = new util::SingleInitiatorRouter(name + "_router");
  renode_router_ = new util::SingleInitiatorRouter(name + "_renode_router");
  auto *data_memory = static_cast<MemoryInterface *>(router_);
  // Instantiate memory profiler, but disable it until the config information
  // has been received.
  mem_profiler_ = new MemoryUseProfiler(data_memory);
  mem_profiler_->set_is_enabled(false);
  // Set up state, decoder, and top.
  rv_state_ = new RiscVState("RiscVRenode", xlen, mem_profiler_,
                             static_cast<AtomicMemoryOpInterface *>(router_));
  rv_fp_state_ = new RiscVFPState(rv_state_->csr_set(), rv_state_);
  rv_state_->set_rv_fp(rv_fp_state_);
  std::string reg_name;
  if (xlen == RiscVXlen::RV32) {
    rv_decoder_ = new RiscV32Decoder(rv_state_, data_memory);
    // Make sure the architectural and abi register aliases are added.
    for (int i = 0; i < 32; i++) {
      reg_name = absl::StrCat(RiscVState::kXregPrefix, i);
      (void)rv_state_->AddRegister<RV32Register>(reg_name);
      (void)rv_state_->AddRegisterAlias<RV32Register>(
          reg_name, ::mpact::sim::riscv::kXRegisterAliases[i]);
    }
  } else {
    rv_decoder_ = new RiscV64Decoder(rv_state_, data_memory);
    for (int i = 0; i < 32; i++) {
      reg_name = absl::StrCat(RiscVState::kXregPrefix, i);
      (void)rv_state_->AddRegister<RV64Register>(reg_name);
      (void)rv_state_->AddRegisterAlias<RV64Register>(
          reg_name, ::mpact::sim::riscv::kXRegisterAliases[i]);
    }
  }

  for (int i = 0; i < 32; i++) {
    reg_name = absl::StrCat(RiscVState::kFregPrefix, i);
    (void)rv_state_->AddRegister<RVFpRegister>(reg_name);
    (void)rv_state_->AddRegisterAlias<RVFpRegister>(
        reg_name, ::mpact::sim::riscv::kFRegisterAliases[i]);
  }

  riscv_top_ = new RiscVTop(name, rv_state_, rv_decoder_);

  // Set up the memory router with the system bus. Other devices are added once
  // config info has been received. Add a tagged default memory transactor, so
  // that any tagged loads/stores are forward to the sysbus without tags.
  CHECK_OK(router_->AddDefaultTarget<MemoryInterface>(renode_sysbus));

  // Create memory. These memories will be added to the core router when there
  // is configuration data for the address space that belongs to the core. The
  // memories will be added to the renode router immediately as the default
  // target, since memory references from ReNode are only in the memory range
  // exposed on the sysbus.
  memory_ = new util::FlatDemandMemory();
  atomic_memory_ = new util::AtomicMemory(memory_);

  CHECK_OK(renode_router_->AddDefaultTarget<MemoryInterface>(memory_));

  // Set up semihosting.
  semihost_ = new RiscVArmSemihost(xlen == RiscVXlen::RV32
                                       ? RiscVArmSemihost::BitWidth::kWord32
                                       : RiscVArmSemihost::BitWidth::kWord64,
                                   data_memory, data_memory);
  // Set up special handlers (ebreak, wfi, ecall).
  riscv_top_->state()->AddEbreakHandler([this](const Instruction *inst) {
    if (this->semihost_->IsSemihostingCall(inst)) {
      this->semihost_->OnEBreak(inst);
      return true;
    }
    if (this->riscv_top_->HasBreakpoint(inst->address())) {
      this->riscv_top_->RequestHalt(HaltReason::kSoftwareBreakpoint, nullptr);
      return true;
    }
    return false;
  });
  riscv_top_->state()->set_on_wfi([](const Instruction *) { return true; });
  riscv_top_->state()->set_on_ecall([](const Instruction *) { return false; });
  semihost_->set_exit_callback([this]() {
    LOG(INFO) << "Simulation halting due to semihosting exit";
    this->riscv_top_->RequestHalt(HaltReason::kProgramDone, nullptr);
  });
}

RiscVRenode::~RiscVRenode() {
  // Halt the core just to be safe.
  (void)riscv_top_->Halt();
  // Write out instruction profile.
  if (inst_profiler_ != nullptr) {
    std::string inst_profile_file_name =
        absl::StrCat("./mpact_riscv_", name_, "_inst_profile.csv");
    std::fstream inst_profile_file(inst_profile_file_name.c_str(),
                                   std::ios_base::out);
    if (!inst_profile_file.good()) {
      LOG(ERROR) << "Failed to write profile to file";
    } else {
      inst_profiler_->WriteProfile(inst_profile_file);
    }
    inst_profile_file.close();
  }
  if (mem_profiler_ != nullptr) {
    std::string mem_profile_file_name =
        absl::StrCat("./mpact_riscv_", name_, "_mem_profile.csv");
    std::fstream mem_profile_file(mem_profile_file_name.c_str(),
                                  std::ios_base::out);
    if (!mem_profile_file.good()) {
      LOG(ERROR) << "Failed to write profile to file";
    } else {
      mem_profiler_->WriteProfile(mem_profile_file);
    }
    mem_profile_file.close();
  }
  // Export counters.
  auto component_proto = std::make_unique<ComponentData>();
  CHECK_OK(riscv_top_->Export(component_proto.get()))
      << "Failed to export proto";
  std::string proto_file_name;
  proto_file_name = absl::StrCat("./mpact_riscv_", name_, ".proto");
  std::fstream proto_file(proto_file_name.c_str(), std::ios_base::out);
  std::string serialized;
  if (!proto_file.good()) {
    LOG(ERROR) << "Failed to open proto file for writing";
  } else if (!google::protobuf::TextFormat::PrintToString(
                 *component_proto.get(), &serialized)) {
    LOG(ERROR) << "Failed to serialize protos";
  } else {
    proto_file << serialized;
  }
  proto_file.close();
  // Clean up.
  delete renode_router_;
  delete mem_profiler_;
  delete inst_profiler_;
  delete instrumentation_control_;
  delete program_loader_;
  delete cmd_shell_;
  delete socket_cli_;
  delete riscv_renode_cli_top_;
  delete riscv_cli_forwarder_;
  delete rv_decoder_;
  delete rv_fp_state_;
  delete riscv_top_;
  delete rv_state_;
  delete semihost_;
  delete router_;
  delete atomic_memory_;
  delete memory_;
  delete clint_;
}

absl::StatusOr<uint64_t> RiscVRenode::LoadExecutable(const char *elf_file_name,
                                                     bool for_symbols_only) {
  program_loader_ = new ElfProgramLoader(this);
  uint64_t entry_pt = 0;
  if (for_symbols_only) {
    auto res = program_loader_->LoadSymbols(elf_file_name);
    if (!res.ok()) {
      return res.status();
    }
    entry_pt = res.value();
  } else {
    auto res = program_loader_->LoadProgram(elf_file_name);
    if (!res.ok()) {
      return res.status();
    }
    entry_pt = res.value();
  }
  auto res = program_loader_->GetSymbol("tohost");
  // Add watchpoint for tohost if the symbol exists.
  if (res.ok()) {
    // If there is a 'tohost' symbol, set up a write watchpoint on that address
    // to catch writes that mark program exit.
    uint64_t tohost_addr = res.value().first;
    // Add to_host watchpoint that halts the execution when program exit is
    // signaled.
    auto *db = riscv_top_->state()->db_factory()->Allocate<uint32_t>(2);
    auto status = riscv_top_->memory_watcher()->SetStoreWatchCallback(
        MemoryWatcher::AddressRange{tohost_addr,
                                    tohost_addr + 2 * sizeof(uint32_t) - 1},
        [this, tohost_addr, db](uint64_t addr, int sz) {
          static DataBuffer *load_db = db;
          if (load_db == nullptr) return;
          memory_->Load(tohost_addr, load_db, nullptr, nullptr);
          uint32_t code = load_db->Get<uint32_t>(0);
          if (code & 0x1) {
            // The return code is in the upper 31 bits.
            code >>= 1;
            LOG(INFO) << absl::StrCat(
                "Simulation halting due to tohost write: exit ",
                absl::Hex(code));
            (void)riscv_top_->RequestHalt(HaltReason::kProgramDone, nullptr);
            load_db->DecRef();
          }
        });
  }
  // Add instruction profiler it hasn't already been added.
  if (inst_profiler_ == nullptr) {
    inst_profiler_ = new util::InstructionProfiler(*program_loader_, 2);
    riscv_top_->counter_pc()->AddListener(inst_profiler_);
    riscv_top_->counter_pc()->SetIsEnabled(false);
  } else {
    // If it has been added already, set the elf loader, and make sure the pc
    // counter is enabled.
    inst_profiler_->SetElfLoader(program_loader_);
    riscv_top_->counter_pc()->SetIsEnabled(true);
  }
  return entry_pt;
}

// Each of the following methods checks to see if the command line enabled
// "top" interface is null. If it is not, it uses that to control the simulator,
// as it provides proper prioritization and handling of both ReNode and command
// line commands. Otherwise it uses the riscv_top interface directly.

absl::StatusOr<int> RiscVRenode::Step(int num) {
  if (riscv_renode_cli_top_ != nullptr)
    return riscv_renode_cli_top_->RenodeStep(num);
  return riscv_top_->Step(num);
}

absl::StatusOr<HaltReasonValueType> RiscVRenode::GetLastHaltReason() {
  if (riscv_renode_cli_top_ != nullptr)
    return riscv_renode_cli_top_->RenodeGetLastHaltReason();
  return riscv_top_->GetLastHaltReason();
}

// Perform direct read of the memory through the renode router. The renode
// router avoids routing the request back out to the sysbus.
absl::StatusOr<size_t> RiscVRenode::ReadMemory(uint64_t address, void *buf,
                                               size_t length) {
  auto *db = db_factory_.Allocate<uint8_t>(length);
  renode_router_->Load(address, db, nullptr, nullptr);
  std::memcpy(buf, db->raw_ptr(), length);
  db->DecRef();
  return length;
}

// Perform direct write of the memory through the renode router. The renode
// router avoids routing the request back out to the sysbus.
absl::StatusOr<size_t> RiscVRenode::WriteMemory(uint64_t address,
                                                const void *buf,
                                                size_t length) {
  auto *db = db_factory_.Allocate<uint8_t>(length);
  std::memcpy(db->raw_ptr(), buf, length);
  renode_router_->Store(address, db);
  db->DecRef();
  return length;
}

absl::StatusOr<uint64_t> RiscVRenode::ReadRegister(uint32_t reg_id) {
  auto ptr = RiscVDebugInfo::Instance()->debug_register_map().find(reg_id);
  if (ptr == RiscVDebugInfo::Instance()->debug_register_map().end()) {
    return absl::NotFoundError(
        absl::StrCat("Not found reg id: ", absl::Hex(reg_id)));
  }
  if (riscv_renode_cli_top_ != nullptr)
    return riscv_renode_cli_top_->RenodeReadRegister(ptr->second);
  return riscv_top_->ReadRegister(ptr->second);
}

absl::Status RiscVRenode::WriteRegister(uint32_t reg_id, uint64_t value) {
  auto ptr = RiscVDebugInfo::Instance()->debug_register_map().find(reg_id);
  if (ptr == RiscVDebugInfo::Instance()->debug_register_map().end()) {
    return absl::NotFoundError(
        absl::StrCat("Not found reg id: ", absl::Hex(reg_id)));
  }
  if (riscv_renode_cli_top_ != nullptr)
    return riscv_renode_cli_top_->RenodeWriteRegister(ptr->second, value);
  return riscv_top_->WriteRegister(ptr->second, value);
}

int32_t RiscVRenode::GetRenodeRegisterInfoSize() const {
  return RiscVRenodeRegisterInfo::GetRenodeRegisterInfo().size();
}

absl::Status RiscVRenode::GetRenodeRegisterInfo(int32_t index, int32_t max_len,
                                                char *name,
                                                RenodeCpuRegister &info) {
  auto const &register_info = RiscVRenodeRegisterInfo::GetRenodeRegisterInfo();
  if ((index < 0) || (index >= register_info.size())) {
    return absl::OutOfRangeError(
        absl::StrCat("Register info index (", index, ") out of range"));
  }
  info = register_info[index];
  auto const &reg_map = RiscVDebugInfo::Instance()->debug_register_map();
  auto ptr = reg_map.find(info.index);
  if (ptr == reg_map.end()) {
    name[0] = '\0';
  } else {
    strncpy(name, ptr->second.c_str(), max_len);
  }
  return absl::OkStatus();
}

static absl::StatusOr<uint64_t> ParseNumber(const std::string &number) {
  if (number.empty()) {
    return absl::InvalidArgumentError("Empty number");
  }
  absl::StatusOr<uint64_t> res;
  if ((number.size() > 2) && (number.substr(0, 2) == "0x")) {
    res = riscv::internal::stoull(number.substr(2), nullptr, 16);
  } else if (number[0] == '0') {
    res = riscv::internal::stoull(number.substr(1), nullptr, 8);
  } else {
    res = riscv::internal::stoull(number, nullptr, 10);
  }
  if (!res.ok()) {
    LOG(ERROR) << "Invalid number: " << number;
    return absl::InvalidArgumentError(absl::StrCat("Invalid number: ", number));
  }
  return res.value();
}

absl::Status RiscVRenode::SetConfig(const char *config_names[],
                                    const char *config_values[], int size) {
  std::string icache_cfg;
  std::string dcache_cfg;
  uint64_t memory_base = 0;
  uint64_t memory_size = 0;
  uint64_t clint_mmr_base = 0;
  uint64_t stack_end_value = 0;
  bool stack_end_set = false;
  uint64_t stack_size_value = 0;
  bool stack_size_set = false;
  bool do_inst_profile = false;
  int cli_port = 0;
  int wait_for_cli = 0;
  for (int i = 0; i < size; ++i) {
    std::string name(config_names[i]);
    if (name == kICache) {
      icache_cfg = config_values[i];
    } else if (name == kDCache) {
      dcache_cfg = config_values[i];
    } else {
      auto res = ParseNumber(config_values[i]);
      if (!res.ok()) {
        return res.status();
      }
      auto value = res.value();
      if (name == kMemoryBase) {
        memory_base = value;
      } else if (name == kMemorySize) {
        memory_size = value;
      } else if (name == kClintMMRBase) {
        clint_mmr_base = value;
      } else if (name == kCLIPort) {
        cli_port = value;
      } else if (name == kWaitForCLI) {
        wait_for_cli = value;
      } else if (name == kInstProfile) {
        do_inst_profile = value != 0;
      } else if (name == kMemProfile) {
        mem_profiler_->set_is_enabled(value != 0);
      } else if (name == kStackEnd) {
        stack_end_value = value;
        stack_end_set = true;
      } else if (name == kStackSize) {
        stack_size_value = value;
        stack_size_set = true;
      } else {
        LOG(ERROR) << "Unknown config name: " << name << " "
                   << config_values[i];
      }
    }
  }
  if (memory_size == 0) {
    return absl::InvalidArgumentError("tagged_memory_size is 0");
  }
  // Add the memory targets.
  CHECK_OK(router_->AddTarget<AtomicMemoryOpInterface>(
      atomic_memory_, memory_base, memory_base + memory_size - 1));
  CHECK_OK(router_->AddTarget<MemoryInterface>(memory_, memory_base,
                                               memory_base + memory_size - 1));
  // Memory mapped devices.
  if (clint_mmr_base != 0) {
    clint_ = new RiscVClint(/*period=*/100, riscv_top_->state()->mip());
    riscv_top_->counter_num_cycles()->AddListener(clint_);
    // Core local interrupt controller - clint.
    CHECK_OK(router_->AddTarget<MemoryInterface>(clint_, clint_mmr_base,
                                                 clint_mmr_base + 0xffffULL));
  }
  // Instruction profiler.
  if (do_inst_profile) {
    if (inst_profiler_ == nullptr) {
      if (program_loader_ == nullptr) {
        // If the program loader is null, assume that it will be added later,
        // but don't enable the trace until it is.
        inst_profiler_ = new util::InstructionProfiler(2);
        riscv_top_->counter_pc()->SetIsEnabled(false);
      } else {
        inst_profiler_ = new util::InstructionProfiler(*program_loader_, 2);
        riscv_top_->counter_pc()->SetIsEnabled(true);
      }
      riscv_top_->counter_pc()->AddListener(inst_profiler_);
    }
  }
  // If the cli port has been specified, then instantiate the requisite classes.
  if (cli_port != 0 && (riscv_renode_cli_top_ == nullptr)) {
    riscv_renode_cli_top_ =
        new RiscVRenodeCLITop(riscv_top_, wait_for_cli != 0);
    riscv_cli_forwarder_ = new RiscVCLIForwarder(riscv_renode_cli_top_);
    cmd_shell_ = new DebugCommandShell();
    instrumentation_control_ =
        new RiscVInstrumentationControl(cmd_shell_, riscv_top_, mem_profiler_);
    cmd_shell_->AddCore(
        {static_cast<RiscVDebugInterface *>(riscv_cli_forwarder_),
         [this]() { return program_loader_; }});
    cmd_shell_->AddCommand(
        instrumentation_control_->Usage(),
        absl::bind_front(&RiscVInstrumentationControl::PerformShellCommand,
                         instrumentation_control_));
    socket_cli_ =
        new SocketCLI(cli_port, *cmd_shell_,
                      absl::bind_front(&RiscVRenodeCLITop::SetConnected,
                                       riscv_renode_cli_top_));
    if (!socket_cli_->good()) {
      return absl::InternalError(
          absl::StrCat("Failed to create socket CLI (", errno, ")"));
    }
  }
  // Set up the stack if so specified (either by symbol in executable or by
  // configuration settings).
  bool initialize_stack = false;
  uint64_t stack_end = 0;
  if (program_loader_ != nullptr) {
    auto res = program_loader_->GetSymbol(kStackEndSymbolName);
    if (res.ok()) {
      stack_end = res.value().first;
      initialize_stack = true;
    }
  }
  if (stack_end_set) {
    stack_end = stack_end_value;
    initialize_stack = true;
  }

  if (initialize_stack) {
    uint64_t stack_size = 32 * 1024;
    // Does the executable have a valid GNU_STACK segment? If so, override the
    // default
    if (program_loader_ != nullptr) {
      auto loader_res = program_loader_->GetStackSize();
      if (loader_res.ok()) {
        stack_end = loader_res.value();
      }

      // If the __stack_size symbol is defined then override.
      auto res = program_loader_->GetSymbol(kStackSizeSymbolName);
      if (res.ok()) {
        stack_size = res.value().first;
      }
    }

    // If the flag is set, then override.
    if (stack_size_set) stack_size = stack_size_value;

    auto sp_write = riscv_top_->WriteRegister("sp", stack_end + stack_size);
    if (!sp_write.ok()) return sp_write;
  }
  if (!icache_cfg.empty()) {
    ComponentValueEntry icache_value;
    icache_value.set_name("icache");
    icache_value.set_string_value(icache_cfg);
    auto *cfg = riscv_top_->GetConfig("icache");
    auto status = cfg->Import(&icache_value);
    if (!status.ok()) return status;
  }
  if (!dcache_cfg.empty()) {
    ComponentValueEntry dcache_value;
    dcache_value.set_name("dcache");
    dcache_value.set_string_value(dcache_cfg);
    auto *cfg = riscv_top_->GetConfig("dcache");
    auto status = cfg->Import(&dcache_value);
    if (!status.ok()) return status;
  }
  return absl::OkStatus();
}

absl::Status RiscVRenode::SetIrqValue(int32_t irq_num, bool irq_value) {
  switch (irq_num) {
    case *riscv::InterruptCode::kMachineExternalInterrupt:
      riscv_top_->state()->mip()->set_meip(irq_value);
      return absl::OkStatus();
    case *riscv::InterruptCode::kMachineTimerInterrupt:
      riscv_top_->state()->mip()->set_mtip(irq_value);
      return absl::OkStatus();
    case *riscv::InterruptCode::kMachineSoftwareInterrupt:
      riscv_top_->state()->mip()->set_msip(irq_value);
      return absl::OkStatus();
    default:
      return absl::NotFoundError(
          absl::StrCat("Unsupported irq number: ", irq_num));
  }
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
