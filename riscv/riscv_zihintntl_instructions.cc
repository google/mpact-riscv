#include "riscv/riscv_zihintntl_instructions.h"

#include "mpact/sim/generic/instruction.h"

namespace mpact::sim::riscv {

using ::mpact::sim::generic::Instruction;

void RiscVNtlP1(const Instruction *) { /* empty */ }
void RiscVNtlPall(const Instruction *) { /* empty */ }
void RiscVNtlS1(const Instruction *) { /* empty */ }
void RiscVNtlAll(const Instruction *) { /* empty */ }

}  // namespace mpact::sim::riscv
