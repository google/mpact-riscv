#include "riscv/riscv32_renode.h"

#include <string>

#include "mpact/sim/util/memory/memory_interface.h"
#include "mpact/sim/util/renode/renode_debug_interface.h"
#include "riscv/riscv_renode.h"
#include "riscv/riscv_state.h"

::mpact::sim::util::renode::RenodeDebugInterface *CreateMpactSim(
    std::string name, std::string cpu_type,
    ::mpact::sim::util::MemoryInterface *renode_sysbus) {
  auto *top = new ::mpact::sim::riscv::RiscVRenode(
      name, renode_sysbus, ::mpact::sim::riscv::RiscVXlen ::RV32);
  return top;
}
