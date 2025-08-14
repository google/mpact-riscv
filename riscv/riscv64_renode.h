#ifndef THIRD_PARTY_MPACT_RISCV_RISCV64_RENODE_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV64_RENODE_H_

#include <string>

#include "mpact/sim/util/memory/memory_interface.h"
#include "mpact/sim/util/renode/renode_debug_interface.h"

// This file defines the factory method for creating a RiscV 64 simulator for
// use with ReNode.

extern ::mpact::sim::util::renode::RenodeDebugInterface* CreateMpactSim(
    std::string name, std::string cpu_type,
    ::mpact::sim::util::MemoryInterface* renode_sysbus);

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV64_RENODE_H_
