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

#include "riscv/riscv_vector_fp_reduction_instructions.h"

#include <functional>

#include "absl/log/log.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_fp_host.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_instruction_helpers.h"
#include "riscv/riscv_vector_state.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::FPTypeInfo;

// These reduction instructions take an accumulator and a value and returns
// the result of the reduction operation. Each partial sum is stored to a
// separate entry in the destination vector.

// Sum reduction.
void Vfredosum(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp);
  switch (sew) {
    case 4:
      return RiscVBinaryReductionVectorOp<float, float, float>(
          rv_vector, inst,
          [](float acc, float vs2) -> float { return acc + vs2; });
      return;
    case 8:
      return RiscVBinaryReductionVectorOp<double, double, double>(
          rv_vector, inst,
          [](double acc, double vs2) -> double { return acc + vs2; });
      return;
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

void Vfwredosum(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp);
  switch (sew) {
    case 4:
      return RiscVBinaryReductionVectorOp<double, float, double>(
          rv_vector, inst, [](double acc, float vs2) -> double {
            return acc + static_cast<double>(vs2);
          });
      return;
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// Templated helper function for vfmin and vfmax instructions.
template <typename T>
inline T MaxMinHelper(T vs2, T vs1, std::function<T(T, T)> operation) {
  // If either operand is a signaling NaN or if both operands are NaNs, then
  // return a canonical (non-signaling) NaN.
  if (FPTypeInfo<T>::IsSNaN(vs1) || FPTypeInfo<T>::IsSNaN(vs2) ||
      (FPTypeInfo<T>::IsNaN(vs2) && FPTypeInfo<T>::IsNaN(vs1))) {
    typename FPTypeInfo<T>::UIntType c_nan = FPTypeInfo<T>::kCanonicalNaN;
    return *reinterpret_cast<T *>(&c_nan);
  }
  // If either operand is a NaN return the other.
  if (FPTypeInfo<T>::IsNaN(vs2)) return vs1;
  if (FPTypeInfo<T>::IsNaN(vs1)) return vs2;
  // Return the min/max of the two operands.
  return operation(vs2, vs1);
}

// FP min reduction.
void Vfredmin(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp);
  switch (sew) {
    case 4:
      return RiscVBinaryReductionVectorOp<float, float, float>(
          rv_vector, inst, [](float acc, float vs2) -> float {
            return MaxMinHelper<float>(acc, vs2,
                                       [](float acc, float vs2) -> float {
                                         return (acc > vs2) ? vs2 : acc;
                                       });
          });
      return;
    case 8:
      return RiscVBinaryReductionVectorOp<double, double, double>(
          rv_vector, inst, [](double acc, double vs2) -> double {
            return MaxMinHelper<double>(acc, vs2,
                                        [](double acc, double vs2) -> double {
                                          return (acc > vs2) ? vs2 : acc;
                                        });
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

// FP max reduction.
void Vfredmax(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp);
  switch (sew) {
    case 4:
      return RiscVBinaryReductionVectorOp<float, float, float>(
          rv_vector, inst, [](float acc, float vs2) -> float {
            return MaxMinHelper<float>(acc, vs2,
                                       [](float acc, float vs2) -> float {
                                         return (acc < vs2) ? vs2 : acc;
                                       });
          });
      return;
    case 8:
      return RiscVBinaryReductionVectorOp<double, double, double>(
          rv_vector, inst, [](double acc, double vs2) -> double {
            return MaxMinHelper<double>(acc, vs2,
                                        [](double acc, double vs2) -> double {
                                          return (acc < vs2) ? vs2 : acc;
                                        });
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Illegal SEW value";
      return;
  }
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
