#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_GETTER_HELPERS_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_GETTER_HELPERS_H_

#include <string>

#include "absl/log/log.h"
#include "mpact/sim/generic/operand_interface.h"
#include "riscv/riscv_state.h"

// This file contains helper functions that are used to create commonly used
// operands for RiscV instructions.

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::DestinationOperandInterface;
using ::mpact::sim::generic::SourceOperandInterface;

// Helper function to insert and entry into a "getter" map. This is used in
// the riscv_*_getter.h files.
template <typename M, typename E, typename G>
inline void Insert(M &map, E entry, G getter) {
  map.insert(std::make_pair(static_cast<int>(entry), getter));
}

// Generic helper functions to create register operands.
template <typename RegType>
inline DestinationOperandInterface *GetRegisterDestinationOp(RiscVState *state,
                                                             std::string name,
                                                             int latency) {
  auto *reg = state->GetRegister<RegType>(name).first;
  return reg->CreateDestinationOperand(latency);
}

template <typename RegType>
inline DestinationOperandInterface *GetRegisterDestinationOp(
    RiscVState *state, std::string name, int latency, std::string op_name) {
  auto *reg = state->GetRegister<RegType>(name).first;
  return reg->CreateDestinationOperand(latency, op_name);
}

template <typename T>
inline DestinationOperandInterface *GetCSRSetBitsDestinationOp(
    RiscVState *state, std::string name, int latency, std::string op_name) {
  auto result = state->csr_set()->GetCsr(name);
  if (!result.ok()) {
    LOG(ERROR) << "No such CSR '" << name << "'";
    return nullptr;
  }
  auto *csr = result.value();
  auto *op = csr->CreateSetDestinationOperand(latency, op_name);
  return op;
}

template <typename RegType>
inline SourceOperandInterface *GetRegisterSourceOp(RiscVState *state,
                                                   std::string name) {
  auto *reg = state->GetRegister<RegType>(name).first;
  auto *op = reg->CreateSourceOperand();
  return op;
}

template <typename RegType>
inline SourceOperandInterface *GetRegisterSourceOp(RiscVState *state,
                                                   std::string name,
                                                   std::string op_name) {
  auto *reg = state->GetRegister<RegType>(name).first;
  auto *op = reg->CreateSourceOperand(op_name);
  return op;
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_GETTER_HELPERS_H_
