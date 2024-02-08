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

#include "riscv/riscv_vector_fp_instructions.h"

#include <cstdint>
#include <functional>
#include <tuple>

#include "absl/log/log.h"
#include "riscv/riscv_fp_host.h"
#include "riscv/riscv_fp_info.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_instruction_helpers.h"
#include "riscv/riscv_vector_state.h"
#include "mpact/sim/generic/type_helpers.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::FPTypeInfo;

// Floating point add.
void Vfadd(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVBinaryVectorOp<float, float, float>(
          rv_vector, inst,
          [](float vs2, float vs1) -> float { return vs2 + vs1; });
    case 8:
      return RiscVBinaryVectorOp<double, double, double>(
          rv_vector, inst,
          [](double vs2, double vs1) -> double { return vs2 + vs1; });
    default:
      LOG(ERROR) << "Vfadd: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Floating point subtract.
void Vfsub(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVBinaryVectorOp<float, float, float>(
          rv_vector, inst,
          [](float vs2, float vs1) -> float { return vs2 - vs1; });
    case 8:
      return RiscVBinaryVectorOp<double, double, double>(
          rv_vector, inst,
          [](double vs2, double vs1) -> double { return vs2 - vs1; });
    default:
      LOG(ERROR) << "Vfsub: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Reverse floating point subtract (rs1 - vs2).
void Vfrsub(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVBinaryVectorOp<float, float, float>(
          rv_vector, inst,
          [](float vs2, float vs1) -> float { return vs1 - vs2; });
    case 8:
      return RiscVBinaryVectorOp<double, double, double>(
          rv_vector, inst,
          [](double vs2, double vs1) -> double { return vs1 - vs2; });
    default:
      LOG(ERROR) << "Vfrsub: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Widening floating point add.
void Vfwadd(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVBinaryVectorOp<double, float, float>(
          rv_vector, inst, [](float vs2, float vs1) -> double {
            double vs2_d = static_cast<double>(vs2);
            double vs1_d = static_cast<double>(vs1);
            return (vs2_d + vs1_d);
          });
    default:
      LOG(ERROR) << "Vfwadd: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Widening floating point subtract.
void Vfwsub(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVBinaryVectorOp<double, float, float>(
          rv_vector, inst, [](float vs2, float vs1) -> double {
            double vs2_d = static_cast<double>(vs2);
            double vs1_d = static_cast<double>(vs1);
            return (vs2_d - vs1_d);
          });
    default:
      LOG(ERROR) << "Vfwsub: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Widening floating point add with wide operand (vs2).
void Vfwaddw(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVBinaryVectorOp<double, double, float>(
          rv_vector, inst, [](double vs2_d, float vs1) -> double {
            double vs1_d = static_cast<double>(vs1);
            return (vs2_d + vs1_d);
          });
    default:
      LOG(ERROR) << "Vfwaddw: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Widening floating point subtract with wide operand (vs2).
void Vfwsubw(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVBinaryVectorOp<double, double, float>(
          rv_vector, inst, [](double vs2_d, float vs1) -> double {
            double vs1_d = static_cast<double>(vs1);
            return (vs2_d - vs1_d);
          });
    default:
      LOG(ERROR) << "Vfwsubw: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Floating point multiply.
void Vfmul(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVBinaryVectorOp<float, float, float>(
          rv_vector, inst,
          [](float vs2, float vs1) -> float { return vs2 * vs1; });
    case 8:
      return RiscVBinaryVectorOp<double, double, double>(
          rv_vector, inst,
          [](double vs2, double vs1) -> double { return vs2 * vs1; });
    default:
      LOG(ERROR) << "Vfmul: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Floating point division vs2/vs1;
void Vfdiv(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVBinaryVectorOp<float, float, float>(
          rv_vector, inst,
          [](float vs2, float vs1) -> float { return vs2 / vs1; });
    case 8:
      return RiscVBinaryVectorOp<double, double, double>(
          rv_vector, inst,
          [](double vs2, double vs1) -> double { return vs2 / vs1; });
    default:
      LOG(ERROR) << "Vfdiv: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Floating point reverse division vs1/vs2.
void Vfrdiv(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVBinaryVectorOp<float, float, float>(
          rv_vector, inst,
          [](float vs2, float vs1) -> float { return vs1 / vs2; });
    case 8:
      return RiscVBinaryVectorOp<double, double, double>(
          rv_vector, inst,
          [](double vs2, double vs1) -> double { return vs1 / vs2; });
    default:
      LOG(ERROR) << "Vfrdiv: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Widening floating point multiply.
void Vfwmul(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVBinaryVectorOp<double, float, float>(
          rv_vector, inst, [](float vs2, float vs1) -> double {
            double vs2_d = static_cast<double>(vs2);
            double vs1_d = static_cast<double>(vs1);
            return (vs2_d * vs1_d);
          });
    default:
      LOG(ERROR) << "Vfwadd: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Floating point multiply and add vs2.
void Vfmadd(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVTernaryVectorOp<float, float, float>(
          rv_vector, inst, [](float vs2, float vs1, float vd) -> float {
            return (vs1 * vd) + vs2;
          });
    case 8:
      return RiscVTernaryVectorOp<double, double, double>(
          rv_vector, inst, [](double vs2, double vs1, double vd) -> double {
            return (vs1 * vd) + vs2;
          });
    default:
      LOG(ERROR) << "Vfmadd: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Negated floating point multiply and add vs2.
void Vfnmadd(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVTernaryVectorOp<float, float, float>(
          rv_vector, inst, [](float vs2, float vs1, float vd) -> float {
            return (-(vs1 * vd)) - vs2;
          });
    case 8:
      return RiscVTernaryVectorOp<double, double, double>(
          rv_vector, inst, [](double vs2, double vs1, double vd) -> double {
            return (-(vs1 * vd)) - vs2;
          });
    default:
      LOG(ERROR) << "Vfnmadd: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Floating point multiply and subtract vs2.
void Vfmsub(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVTernaryVectorOp<float, float, float>(
          rv_vector, inst, [](float vs2, float vs1, float vd) -> float {
            return (vs1 * vd) - vs2;
          });
    case 8:
      return RiscVTernaryVectorOp<double, double, double>(
          rv_vector, inst, [](double vs2, double vs1, double vd) -> double {
            return (vs1 * vd) - vs2;
          });
    default:
      LOG(ERROR) << "Vfmsub: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Negated floating point multiply and subtract vs2.
void Vfnmsub(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVTernaryVectorOp<float, float, float>(
          rv_vector, inst, [](float vs2, float vs1, float vd) -> float {
            return (-(vs1 * vd)) + vs2;
          });
    case 8:
      return RiscVTernaryVectorOp<double, double, double>(
          rv_vector, inst, [](double vs2, double vs1, double vd) -> double {
            return (-(vs1 * vd)) + vs2;
          });
    default:
      LOG(ERROR) << "Vfnmsub: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Floating point multiply and accumulate vd.
void Vfmacc(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVTernaryVectorOp<float, float, float>(
          rv_vector, inst, [](float vs2, float vs1, float vd) -> float {
            return ((vs1 * vs2) + vd);
          });
    case 8:
      return RiscVTernaryVectorOp<double, double, double>(
          rv_vector, inst, [](double vs2, double vs1, double vd) -> double {
            return ((vs1 * vs2) + vd);
          });
    default:
      LOG(ERROR) << "Vfmacc: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Negated floating point multiply and accumulate vd.
void Vfnmacc(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVTernaryVectorOp<float, float, float>(
          rv_vector, inst, [](float vs2, float vs1, float vd) -> float {
            return (-(vs1 * vs2)) - vd;
          });
    case 8:
      return RiscVTernaryVectorOp<double, double, double>(
          rv_vector, inst, [](double vs2, double vs1, double vd) -> double {
            return (-(vs1 * vs2)) - vd;
          });
    default:
      LOG(ERROR) << "Vfnmacc: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Floating point multiply and subtract vd.
void Vfmsac(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVTernaryVectorOp<float, float, float>(
          rv_vector, inst, [](float vs2, float vs1, float vd) -> float {
            return ((vs1 * vs2) - vd);
          });
    case 8:
      return RiscVTernaryVectorOp<double, double, double>(
          rv_vector, inst, [](double vs2, double vs1, double vd) -> double {
            return ((vs1 * vs2) - vd);
          });
    default:
      LOG(ERROR) << "Vfmsac: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Negated floating point multiply and subtract vd.
void Vfnmsac(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVTernaryVectorOp<float, float, float>(
          rv_vector, inst, [](float vs2, float vs1, float vd) -> float {
            return (-(vs1 * vs2) + vd);
          });
    case 8:
      return RiscVTernaryVectorOp<double, double, double>(
          rv_vector, inst, [](double vs2, double vs1, double vd) -> double {
            return (-(vs1 * vs2) + vd);
          });
    default:
      LOG(ERROR) << "Vfnmsac: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Widening floating point multiply and accumulate vd.
void Vfwmacc(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVTernaryVectorOp<double, float, float>(
          rv_vector, inst, [](float vs2, float vs1, double vd) -> double {
            double vs1_d = vs1;
            double vs2_d = vs2;
            return ((vs1_d * vs2_d) + vd);
          });
    default:
      LOG(ERROR) << "Vfwmacc: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Widening negated floating point multiply and accumulate vd.
void Vfwnmacc(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVTernaryVectorOp<double, float, float>(
          rv_vector, inst, [](float vs2, float vs1, double vd) -> double {
            double vs1_d = vs1;
            double vs2_d = vs2;
            return (-(vs1_d * vs2_d)) - vd;
          });
    default:
      LOG(ERROR) << "Vfwnmacc: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Widening floating point multiply and subtract vd.
void Vfwmsac(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVTernaryVectorOp<double, float, float>(
          rv_vector, inst, [](float vs2, float vs1, double vd) -> double {
            double vs1_d = vs1;
            double vs2_d = vs2;
            return ((vs1_d * vs2_d) - vd);
          });
    default:
      LOG(ERROR) << "Vfwmsac: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Widening negated floating point multiply and subtract vd.
void Vfwnmsac(const Instruction *inst) {
  auto *rv_fp = static_cast<RiscVState *>(inst->state())->rv_fp();
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVTernaryVectorOp<double, float, float>(
          rv_vector, inst, [](float vs2, float vs1, double vd) -> double {
            double vs1_d = vs1;
            double vs2_d = vs2;
            return (-(vs1_d * vs2_d)) + vd;
          });
    default:
      LOG(ERROR) << "Vfwnmsac: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Change the sign of vs2 to the sign of vs1.
void Vfsgnj(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst, [](uint32_t vs2, uint32_t vs1) -> uint32_t {
            return (vs2 & 0x7fff'ffff) | (vs1 & 0x8000'0000);
          });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst, [](uint64_t vs2, uint64_t vs1) -> uint64_t {
            return (vs2 & 0x7fff'ffff'ffff'ffff) |
                   (vs1 & 0x8000'0000'0000'0000);
          });
    default:
      LOG(ERROR) << "Vfsgnj: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Change the sign of vs2 to the negation of the sign of vs1.
void Vfsgnjn(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst, [](uint32_t vs2, uint32_t vs1) -> uint32_t {
            return (vs2 & 0x7fff'ffff) | (~vs1 & 0x8000'0000);
          });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst, [](uint64_t vs2, uint64_t vs1) -> uint64_t {
            return (vs2 & 0x7fff'ffff'ffff'ffff) |
                   (~vs1 & 0x8000'0000'0000'0000);
          });
    default:
      LOG(ERROR) << "Vfsgnjn: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Change the sign of vs2 to the xor of the sign of the two operands.
void Vfsgnjx(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst, [](uint32_t vs2, uint32_t vs1) -> uint32_t {
            return (vs2 & 0x7fff'ffff) | ((vs1 ^ vs2) & 0x8000'0000);
          });
    case 8:
      return RiscVBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst, [](uint64_t vs2, uint64_t vs1) -> uint64_t {
            return (vs2 & 0x7fff'ffff'ffff'ffff) ^
                   ((vs1 ^ vs2) & 0x8000'0000'0000'0000);
          });
    default:
      LOG(ERROR) << "Vfsgnjx: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Templated helper function for vfmin and vfmax instructions.
template <typename T>
inline std::tuple<T, uint32_t> MaxMinHelper(T vs2, T vs1,
                                            std::function<T(T, T)> operation) {
  // If either operand is a signaling NaN or if both operands are NaNs, then
  // return a canonical (non-signaling) NaN.
  uint32_t flag = 0;
  if (FPTypeInfo<T>::IsSNaN(vs1) || FPTypeInfo<T>::IsSNaN(vs2)) {
    flag = static_cast<uint32_t>(FPExceptions::kInvalidOp);
  }
  if (FPTypeInfo<T>::IsNaN(vs2) && FPTypeInfo<T>::IsNaN(vs1)) {
    auto c_nan = FPTypeInfo<T>::kCanonicalNaN;
    return std::make_tuple(*reinterpret_cast<T *>(&c_nan), flag);
  }
  // If either operand is a NaN return the other.
  if (FPTypeInfo<T>::IsNaN(vs2)) return std::tie(vs1, flag);
  if (FPTypeInfo<T>::IsNaN(vs1)) return std::tie(vs2, flag);
  // Return the min/max of the two operands.
  if ((vs2 == 0.0) && (vs1 == 0.0)) {
    T tmp2 = std::signbit(vs2) ? -1.0 : 1;
    T tmp1 = std::signbit(vs1) ? -1.0 : 1;
    return std::make_tuple(operation(tmp2, tmp1) == tmp2 ? vs2 : vs1, 0);
  }
  return std::make_tuple(operation(vs2, vs1), flag);
}

// Vector floating point min.
void Vfmin(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVBinaryVectorOpWithFflags<float, float, float>(
          rv_vector, inst,
          [](float vs2, float vs1) -> std::tuple<float, uint32_t> {
            using T = float;
            return MaxMinHelper<T>(vs2, vs1, [](T vs2, T vs1) -> T {
              return (vs1 < vs2) ? vs1 : vs2;
            });
          });
    case 8:
      return RiscVBinaryVectorOpWithFflags<double, double, double>(
          rv_vector, inst,
          [](double vs2, double vs1) -> std::tuple<double, uint32_t> {
            using T = double;
            return MaxMinHelper<T>(vs2, vs1, [](T vs2, T vs1) -> T {
              return (vs1 < vs2) ? vs1 : vs2;
            });
          });
    default:
      LOG(ERROR) << "Vfmin: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Vector floating point max.
void Vfmax(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVBinaryVectorOpWithFflags<float, float, float>(
          rv_vector, inst,
          [](float vs2, float vs1) -> std::tuple<float, uint32_t> {
            using T = float;
            return MaxMinHelper<T>(vs2, vs1, [](T vs2, T vs1) -> T {
              return (vs1 > vs2) ? vs1 : vs2;
            });
          });
    case 8:
      return RiscVBinaryVectorOpWithFflags<double, double, double>(
          rv_vector, inst,
          [](double vs2, double vs1) -> std::tuple<double, uint32_t> {
            using T = double;
            return MaxMinHelper<T>(vs2, vs1, [](T vs2, T vs1) -> T {
              return (vs1 > vs2) ? vs1 : vs2;
            });
          });
    default:
      LOG(ERROR) << "Vfmax: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Vector fp merge.
void Vfmerge(const Instruction *inst) {
  auto *rv_vector = static_cast<RiscVState *>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVMaskBinaryVectorOp<uint32_t, uint32_t, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2, uint32_t vs1, bool mask) -> uint32_t {
            return mask ? vs1 : vs2;
          });
    case 8:
      return RiscVMaskBinaryVectorOp<uint64_t, uint64_t, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2, uint64_t vs1, bool mask) -> uint64_t {
            return mask ? vs1 : vs2;
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "Vfmerge: Illegal sew (" << sew << ")";
      return;
  }
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
