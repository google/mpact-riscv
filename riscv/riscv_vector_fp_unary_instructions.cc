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

#include "riscv/riscv_vector_fp_unary_instructions.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

#include "absl/log/log.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_fp_host.h"
#include "riscv/riscv_fp_info.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_instruction_helpers.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_instruction_helpers.h"
#include "riscv/riscv_vector_state.h"

namespace mpact {
namespace sim {
namespace riscv {

// These tables contain the 7 bits of mantissa used by the approximated
// reciprocal square root and reciprocal instructions.
static const int kRecipSqrtMantissaTable[128] = {
    52,  51,  50,  48,  47,  46,  44,  43,  42,  41,  40,  39,  38,  36,  35,
    34,  33,  32,  31,  30,  30,  29,  28,  27,  26,  25,  24,  23,  23,  22,
    21,  20,  19,  19,  18,  17,  16,  16,  15,  14,  14,  13,  12,  12,  11,
    10,  10,  9,   9,   8,   7,   7,   6,   6,   5,   4,   4,   3,   3,   2,
    2,   1,   1,   0,   127, 125, 123, 121, 119, 118, 116, 114, 113, 111, 109,
    108, 106, 105, 103, 102, 100, 99,  97,  96,  95,  93,  92,  91,  20,  88,
    87,  86,  85,  84,  83,  82,  80,  79,  78,  77,  76,  75,  74,  73,  72,
    71,  70,  70,  69,  68,  67,  66,  65,  64,  63,  63,  62,  61,  60,  59,
    59,  58,  57,  56,  56,  55,  54,  53,
};

static const int kRecipMantissaTable[128] = {
    127, 125, 123, 121, 119, 117, 116, 114, 112, 110, 109, 107, 105, 104, 102,
    100, 99,  97,  96,  94,  93,  91,  90,  88,  87,  85,  84,  83,  81,  80,
    79,  77,  76,  75,  74,  72,  71,  70,  69,  68,  66,  65,  64,  63,  62,
    61,  60,  59,  58,  57,  56,  55,  54,  53,  52,  51,  50,  49,  48,  47,
    46,  45,  44,  43,  42,  41,  40,  40,  39,  38,  37,  36,  35,  35,  34,
    33,  32,  31,  31,  30,  29,  28,  28,  27,  26,  25,  25,  24,  23,  23,
    22,  21,  21,  20,  19,  19,  18,  17,  17,  16,  15,  15,  14,  14,  13,
    12,  12,  11,  11,  10,  9,   9,   8,   8,   7,   7,   6,   5,   5,   4,
    4,   3,   3,   2,   2,   1,   1,   0};

// Move float from scalar fp register to vector register(all elements).
void Vfmvvf(const Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  const int vl = rv_vector->vector_length();
  if (rv_vector->vstart() > 0) return;
  if (vl == 0) return;

  const int sew = rv_vector->selected_element_width();
  auto dest_op =
      static_cast<RV32VectorDestinationOperand*>(inst->Destination(0));
  auto dest_db = dest_op->CopyDataBuffer();
  switch (sew) {
    case 4:
      for (int i = 0; i < vl; ++i) {
        dest_db->Set<uint32_t>(
            i, generic::GetInstructionSource<uint32_t>(inst, 0, 0));
      }
      break;
    case 8:
      for (int i = 0; i < vl; ++i) {
        dest_db->Set<uint64_t>(
            i, generic::GetInstructionSource<uint64_t>(inst, 0, 0));
      }
      break;
    default:
      dest_db->DecRef();
      LOG(ERROR) << "Vfmv.s.f: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
  dest_db->Submit();
  rv_vector->clear_vstart();
}

// Move float from vector to scalar fp register(first element).
void Vfmvsf(const Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  if (rv_vector->vstart() > 0) return;
  if (rv_vector->vector_length() == 0) return;
  int sew = rv_vector->selected_element_width();
  auto dest_op =
      static_cast<RV32VectorDestinationOperand*>(inst->Destination(0));
  auto dest_db = dest_op->CopyDataBuffer();
  switch (sew) {
    case 4:
      dest_db->Set<uint32_t>(
          0, generic::GetInstructionSource<uint32_t>(inst, 0, 0));
      break;
    case 8:
      dest_db->Set<uint64_t>(
          0, generic::GetInstructionSource<uint64_t>(inst, 0, 0));
      break;
    default:
      dest_db->DecRef();
      LOG(ERROR) << "Vfmv.s.f: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
  dest_db->Submit();
  rv_vector->clear_vstart();
}

// Move scalar floating point value to element 0 of vector register.
void Vfmvfs(const Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  auto dest_op = inst->Destination(0);
  auto dest_db = dest_op->AllocateDataBuffer();
  int db_size = dest_db->size<uint8_t>();
  switch (sew) {
    case 4: {
      uint64_t value = generic::GetInstructionSource<uint32_t>(inst, 0, 0);
      if (db_size == 4) {
        dest_db->Set<uint32_t>(0, value);
      } else if (db_size == 8) {
        uint64_t val64 = 0xffff'ffff'0000'0000ULL | value;
        dest_db->Set<uint64_t>(0, val64);
      } else {
        LOG(ERROR) << "Unexpected databuffer size in Vfmvfs";
      }
      break;
    }
    case 8:
      dest_db->Set<uint64_t>(
          0, generic::GetInstructionSource<uint64_t>(inst, 0, 0));
      break;
    default:
      dest_db->DecRef();
      LOG(ERROR) << "Vfmv.f.s: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
  dest_db->Submit();
  rv_vector->clear_vstart();
}

// Convert floating point to unsigned integer.
void Vfcvtxufv(const Instruction* inst) {
  auto* rv_fp = static_cast<RiscVState*>(inst->state())->rv_fp();
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVUnaryVectorOpWithFflags<uint32_t, float>(
          rv_vector, inst, [](float vs2) -> std::tuple<uint32_t, uint32_t> {
            return CvtHelper<float, uint32_t>(vs2);
          });
    case 8:
      return RiscVUnaryVectorOpWithFflags<uint64_t, double>(
          rv_vector, inst, [](double vs2) -> std::tuple<uint64_t, uint32_t> {
            return CvtHelper<double, uint64_t>(vs2);
          });
    default:
      LOG(ERROR) << "Vfcvt.xu.fv: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Convert floating point to signed integer.
void Vfcvtxfv(const Instruction* inst) {
  auto* rv_fp = static_cast<RiscVState*>(inst->state())->rv_fp();
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVUnaryVectorOpWithFflags<int32_t, float>(
          rv_vector, inst, [](float vs2) -> std::tuple<int32_t, uint32_t> {
            return CvtHelper<float, int32_t>(vs2);
          });
    case 8:
      return RiscVUnaryVectorOpWithFflags<int64_t, double>(
          rv_vector, inst, [](double vs2) -> std::tuple<int64_t, uint32_t> {
            return CvtHelper<double, int64_t>(vs2);
          });
    default:
      LOG(ERROR) << "Vfcvt.x.fv: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Convert unsigned integer to floating point.
void Vfcvtfxuv(const Instruction* inst) {
  auto* rv_fp = static_cast<RiscVState*>(inst->state())->rv_fp();
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVUnaryVectorOp<float, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2) -> float { return static_cast<float>(vs2); });
    case 8:
      return RiscVUnaryVectorOp<double, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2) -> double { return static_cast<double>(vs2); });
    default:
      LOG(ERROR) << "Vfcvt.f.xuv: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Convert signed integer to floating point.
void Vfcvtfxv(const Instruction* inst) {
  auto* rv_fp = static_cast<RiscVState*>(inst->state())->rv_fp();
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 4:
      return RiscVUnaryVectorOp<float, int32_t>(
          rv_vector, inst,
          [](int32_t vs2) -> float { return static_cast<float>(vs2); });
    case 8:
      return RiscVUnaryVectorOp<double, int64_t>(
          rv_vector, inst,
          [](int64_t vs2) -> double { return static_cast<double>(vs2); });
    default:
      LOG(ERROR) << "Vfcvt.f.xv: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Convert floating point to unsigned integer with truncation.
void Vfcvtrtzxufv(const Instruction* inst) {
  auto* rv_state = static_cast<RiscVState*>(inst->state());
  auto* rv_vector = rv_state->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVUnaryVectorOpWithFflags<uint32_t, float>(
          rv_vector, inst, [](float vs2) -> std::tuple<uint32_t, uint32_t> {
            return CvtHelper<float, uint32_t>(vs2);
          });
    case 8:
      return RiscVUnaryVectorOpWithFflags<uint64_t, double>(
          rv_vector, inst, [](double vs2) -> std::tuple<uint64_t, uint32_t> {
            return CvtHelper<double, uint64_t>(vs2);
          });
    default:
      LOG(ERROR) << "Vfcvt.rtz.xu.fv: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Convert floating point to signed integer with truncation.
void Vfcvtrtzxfv(const Instruction* inst) {
  auto* rv_state = static_cast<RiscVState*>(inst->state());
  auto* rv_vector = rv_state->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVUnaryVectorOpWithFflags<int32_t, float>(
          rv_vector, inst, [](float vs2) -> std::tuple<int32_t, uint32_t> {
            return CvtHelper<float, int32_t>(vs2);
          });
    case 8:
      return RiscVUnaryVectorOpWithFflags<int64_t, double>(
          rv_vector, inst, [](double vs2) -> std::tuple<int64_t, uint32_t> {
            return CvtHelper<double, int64_t>(vs2);
          });
    default:
      LOG(ERROR) << "Vfcvt.rtz.x.fv: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Widening conversion of floating point to unsigned integer.
void Vfwcvtxufv(const Instruction* inst) {
  auto* rv_fp = static_cast<RiscVState*>(inst->state())->rv_fp();
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVUnaryVectorOpWithFflags<uint64_t, float>(
          rv_vector, inst, [](float vs2) -> std::tuple<uint64_t, uint32_t> {
            return CvtHelper<float, uint64_t>(vs2);
          });
    default:
      LOG(ERROR) << "Vfwcvt.xu.fv: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Widening conversion of floating point to signed integer.
void Vfwcvtxfv(const Instruction* inst) {
  auto* rv_fp = static_cast<RiscVState*>(inst->state())->rv_fp();
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVUnaryVectorOpWithFflags<int64_t, float>(
          rv_vector, inst, [](float vs2) -> std::tuple<int64_t, uint32_t> {
            return CvtHelper<float, int64_t>(vs2);
          });
    default:
      LOG(ERROR) << "Vfwcvt.x.fv: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Wideing conversion of floating point to floating point.
void Vfwcvtffv(const Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVUnaryVectorOp<double, float>(
          rv_vector, inst,
          [](float vs2) -> double { return static_cast<double>(vs2); });
    default:
      LOG(ERROR) << "Vfwcvt.f.fv: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Widening conversion of unsigned integer to floating point.
void Vfwcvtfxuv(const Instruction* inst) {
  auto* rv_fp = static_cast<RiscVState*>(inst->state())->rv_fp();
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 2:
      return RiscVUnaryVectorOp<float, uint16_t>(
          rv_vector, inst,
          [](uint16_t vs2) -> float { return static_cast<float>(vs2); });
    case 4:
      return RiscVUnaryVectorOp<double, uint32_t>(
          rv_vector, inst,
          [](uint32_t vs2) -> double { return static_cast<double>(vs2); });
    default:
      LOG(ERROR) << "Vfwcvt.f.xuv: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Widening conversion of signed integer to floating point.
void Vfwcvtfxv(const Instruction* inst) {
  auto* rv_fp = static_cast<RiscVState*>(inst->state())->rv_fp();
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 2:
      return RiscVUnaryVectorOp<float, int16_t>(
          rv_vector, inst,
          [](int16_t vs2) -> float { return static_cast<float>(vs2); });
    case 4:
      return RiscVUnaryVectorOp<double, int32_t>(
          rv_vector, inst,
          [](int32_t vs2) -> double { return static_cast<double>(vs2); });
    default:
      LOG(ERROR) << "Vfwcvt.f.xuv: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Widening conversion of floating point to unsigned integer with truncation.
void Vfwcvtrtzxufv(const Instruction* inst) {
  auto* rv_state = static_cast<RiscVState*>(inst->state());
  auto* rv_vector = rv_state->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVUnaryVectorOpWithFflags<uint64_t, float>(
          rv_vector, inst, [](float vs2) -> std::tuple<uint64_t, uint32_t> {
            return CvtHelper<float, uint64_t>(vs2);
          });
    default:
      LOG(ERROR) << "Vwfcvt.rtz.xu.fv: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Widening conversion of floating point to signed integer with truncation.
void Vfwcvtrtzxfv(const Instruction* inst) {
  auto* rv_state = static_cast<RiscVState*>(inst->state());
  auto* rv_vector = rv_state->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVUnaryVectorOpWithFflags<int64_t, float>(
          rv_vector, inst, [](float vs2) -> std::tuple<int64_t, uint32_t> {
            return CvtHelper<float, int64_t>(vs2);
          });
    default:
      LOG(ERROR) << "Vwfcvt.rtz.x.fv: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Narrowing conversion of floating point to unsigned integer.
void Vfncvtxufw(const Instruction* inst) {
  auto* rv_fp = static_cast<RiscVState*>(inst->state())->rv_fp();
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVUnaryVectorOpWithFflags<uint16_t, float>(
          rv_vector, inst, [](float vs2) -> std::tuple<uint16_t, uint32_t> {
            return CvtHelper<float, uint16_t>(vs2);
          });
    case 8:
      return RiscVUnaryVectorOpWithFflags<uint32_t, double>(
          rv_vector, inst, [](double vs2) -> std::tuple<uint32_t, uint32_t> {
            return CvtHelper<double, uint32_t>(vs2);
          });
    default:
      LOG(ERROR) << "Vfncvt.xu.fw: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Narrowing conversion of floating point to signed integer.
void Vfncvtxfw(const Instruction* inst) {
  auto* rv_fp = static_cast<RiscVState*>(inst->state())->rv_fp();
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVUnaryVectorOpWithFflags<int16_t, float>(
          rv_vector, inst, [](float vs2) -> std::tuple<int16_t, uint32_t> {
            return CvtHelper<float, int16_t>(vs2);
          });
    case 8:
      return RiscVUnaryVectorOpWithFflags<int32_t, double>(
          rv_vector, inst, [](double vs2) -> std::tuple<int32_t, uint32_t> {
            return CvtHelper<double, int32_t>(vs2);
          });
    default:
      LOG(ERROR) << "Vfncvt.x.fw: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Narrowing conversion of floating point to floating point.
void Vfncvtffw(const Instruction* inst) {
  auto* rv_fp = static_cast<RiscVState*>(inst->state())->rv_fp();
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 8:
      return RiscVUnaryVectorOp<float, double>(
          rv_vector, inst,
          [](double vs2) -> float { return static_cast<float>(vs2); });
    default:
      LOG(ERROR) << "Vfwcvt.f.fw: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Narrowing conversion of floating point to floating point rounding to odd.
void Vfncvtrodffw(const Instruction* inst) {
  auto* rv_fp = static_cast<RiscVState*>(inst->state())->rv_fp();
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  // The rounding mode is round to odd, which means that the lsb of the new
  // mantissa is either 1 or it is the logical or of all the bits to the right
  // in the original width mantissa.
  switch (sew) {
    case 8:
      return RiscVUnaryVectorOp<float, double>(
          rv_vector, inst, [](double vs2) -> float {
            if (FPTypeInfo<double>::IsNaN(vs2) ||
                FPTypeInfo<double>::IsInf(vs2)) {
              return static_cast<float>(vs2);
            }
            using UIntD = typename FPTypeInfo<double>::UIntType;
            using UIntF = typename FPTypeInfo<float>::UIntType;
            UIntD uval = *reinterpret_cast<UIntD*>(&vs2);
            int sig_diff =
                FPTypeInfo<double>::kSigSize - FPTypeInfo<float>::kSigSize;
            UIntD mask = (1ULL << sig_diff) - 1;
            UIntF bit = (mask & uval) != 0;
            auto res = static_cast<float>(vs2);
            if (FPTypeInfo<float>::IsInf(res)) return res;
            UIntF ures = *reinterpret_cast<UIntF*>(&res);
            ures |= bit;
            return *reinterpret_cast<float*>(&ures);
          });
    default:
      LOG(ERROR) << "Vfwcvt.rod.f.fw: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Narrowing conversion of unsigned integer to floating point.
void Vfncvtfxuw(const Instruction* inst) {
  auto* rv_fp = static_cast<RiscVState*>(inst->state())->rv_fp();
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 8:
      return RiscVUnaryVectorOp<float, uint64_t>(
          rv_vector, inst,
          [](uint64_t vs2) -> float { return static_cast<float>(vs2); });
    default:
      LOG(ERROR) << "Vfncvt.f.xuw: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Narrowing conversion of signed integeer to floating point.
void Vfncvtfxw(const Instruction* inst) {
  auto* rv_fp = static_cast<RiscVState*>(inst->state())->rv_fp();
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  switch (sew) {
    case 8:
      return RiscVUnaryVectorOp<float, int64_t>(
          rv_vector, inst,
          [](int64_t vs2) -> float { return static_cast<float>(vs2); });
    default:
      LOG(ERROR) << "Vfncvt.f.xw: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Narrowing conversion of floating point to unsigned integer with truncation.
void Vfncvtrtzxufw(const Instruction* inst) {
  auto* rv_state = static_cast<RiscVState*>(inst->state());
  auto* rv_vector = rv_state->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVUnaryVectorOpWithFflags<uint16_t, float>(
          rv_vector, inst, [](float vs2) -> std::tuple<uint16_t, uint32_t> {
            return CvtHelper<float, uint16_t>(vs2);
          });
    case 8:
      return RiscVUnaryVectorOpWithFflags<uint32_t, double>(
          rv_vector, inst, [](double vs2) -> std::tuple<uint32_t, uint32_t> {
            return CvtHelper<double, uint32_t>(vs2);
          });
    default:
      LOG(ERROR) << "Vfcvt.rtz.xu.fw: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Narrowing conversion of floating point to signed integer with truncation.
void Vfncvtrtzxfw(const Instruction* inst) {
  auto* rv_state = static_cast<RiscVState*>(inst->state());
  auto* rv_vector = rv_state->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVUnaryVectorOpWithFflags<int16_t, float>(
          rv_vector, inst, [](float vs2) -> std::tuple<int16_t, uint32_t> {
            return CvtHelper<float, int16_t>(vs2);
          });
    case 8:
      return RiscVUnaryVectorOpWithFflags<int32_t, double>(
          rv_vector, inst, [](double vs2) -> std::tuple<int32_t, uint32_t> {
            return CvtHelper<double, int32_t>(vs2);
          });
    default:
      LOG(ERROR) << "Vfcvt.rtz.xu.fw: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Templated helper function to compute square root.
template <typename T>
inline std::tuple<T, uint32_t> SqrtHelper(T vs2) {
  uint32_t flags = 0;
  T res;
  if (FPTypeInfo<T>::IsNaN(vs2) || vs2 < 0.0) {
    auto value = FPTypeInfo<T>::kCanonicalNaN;
    res = *reinterpret_cast<T*>(&value);
    flags = *FPExceptions::kInvalidOp;
    return std::tie(res, flags);
  }
  if (vs2 == 0.0) return std::tie(vs2, flags);
  res = sqrt(vs2);
  return std::tie(res, flags);
}

// Square root.
void Vfsqrtv(const Instruction* inst) {
  auto* rv_fp = static_cast<RiscVState*>(inst->state())->rv_fp();
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  if (!rv_fp->rounding_mode_valid()) {
    LOG(ERROR) << "Invalid rounding mode";
    rv_vector->set_vector_exception();
    return;
  }
  int sew = rv_vector->selected_element_width();
  uint32_t flags = 0;
  {
    ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
    switch (sew) {
      case 4:
        RiscVUnaryVectorOp<float, float>(rv_vector, inst,
                                         [&flags](float vs2) -> float {
                                           auto [res, f] = SqrtHelper(vs2);
                                           flags |= f;
                                           return res;
                                         });
        break;
      case 8:
        RiscVUnaryVectorOp<double, double>(rv_vector, inst,
                                           [&flags](double vs2) -> double {
                                             auto [res, f] = SqrtHelper(vs2);
                                             flags |= f;
                                             return res;
                                           });
        break;
      default:
        LOG(ERROR) << "Vffcvt.f.xuv: Illegal sew (" << sew << ")";
        rv_vector->set_vector_exception();
        return;
    }
  }
  auto* fflags = rv_fp->fflags();
  fflags->Write(flags | fflags->AsUint32());
}

// Templated helper function to compute the Reciprocal Square Root
// approximation for valid inputs.
template <typename T>
inline T RecipSqrt7(T value) {
  using Uint = typename FPTypeInfo<T>::UIntType;
  Uint uint_value = *reinterpret_cast<Uint*>(&value);
  // The input value is positive. Negative values are already handled.
  int norm_exponent =
      (uint_value & FPTypeInfo<T>::kExpMask) >> FPTypeInfo<T>::kSigSize;
  Uint norm_mantissa = uint_value & FPTypeInfo<T>::kSigMask;
  if (norm_exponent == 0) {  // The value is a denormal.
    Uint mask = static_cast<Uint>(1) << (FPTypeInfo<T>::kSigSize - 1);
    // Normalize the mantissa and exponent by shifting the mantissa left until
    // the most significant bit is one.
    while ((norm_mantissa & mask) == 0) {
      norm_exponent--;
      norm_mantissa <<= 1;
    }
    // Shift it left once more - so it becomes the "implied" bit, and not used
    // in the lookup below.
    norm_mantissa <<= 1;
  }
  int index = (norm_exponent & 0b1) << 6 |
              ((norm_mantissa >> (FPTypeInfo<T>::kSigSize - 6)) & 0b11'1111);
  Uint new_mantissa = static_cast<Uint>(kRecipSqrtMantissaTable[index])
                      << (FPTypeInfo<T>::kSigSize - 7);
  Uint new_exponent = (3 * FPTypeInfo<T>::kExpBias - 1 - norm_exponent) / 2;
  Uint new_value = (new_exponent << FPTypeInfo<T>::kSigSize) | new_mantissa;
  T new_fp_value = *reinterpret_cast<T*>(&new_value);
  return new_fp_value;
}

// Templated helper function to compute the Reciprocal Square Root
// approximation for all values.
template <typename T>
inline std::tuple<T, uint32_t> RecipSqrt7Helper(T value) {
  auto fp_class = std::fpclassify(value);
  T return_value = std::numeric_limits<T>::quiet_NaN();
  uint32_t fflags = 0;
  switch (fp_class) {
    case FP_INFINITE:
      return_value =
          std::signbit(value) ? std::numeric_limits<T>::quiet_NaN() : 0.0;
      fflags = (uint32_t)FPExceptions::kInvalidOp;
      break;
    case FP_NAN:
      // Just propagate the NaN.
      return_value = std::numeric_limits<T>::quiet_NaN();
      fflags = (uint32_t)FPExceptions::kInvalidOp;
      break;
    case FP_ZERO:
      return_value = std::signbit(value) ? -std::numeric_limits<T>::infinity()
                                         : std::numeric_limits<T>::infinity();
      fflags = (uint32_t)FPExceptions::kDivByZero;
      break;
    case FP_SUBNORMAL:
    case FP_NORMAL:
      if (std::signbit(value)) {
        return_value = std::numeric_limits<T>::quiet_NaN();
        fflags = (uint32_t)FPExceptions::kInvalidOp;
      } else {
        return_value = RecipSqrt7(value);
      }
      break;
    default:
      LOG(ERROR) << "RecipSqrt7Helper: Illegal fp_class (" << fp_class << ")";
      break;
  }
  return std::make_tuple(return_value, fflags);
}

// Approximation of reciprocal square root to 7 bits mantissa.
void Vfrsqrt7v(const Instruction* inst) {
  auto* rv_fp = static_cast<RiscVState*>(inst->state())->rv_fp();
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVUnaryVectorOpWithFflags<float, float>(
          rv_vector, inst, [rv_fp](float vs2) -> std::tuple<float, uint32_t> {
            ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
            return RecipSqrt7Helper(vs2);
          });
    case 8:
      return RiscVUnaryVectorOpWithFflags<double, double>(
          rv_vector, inst, [rv_fp](double vs2) -> std::tuple<double, uint32_t> {
            ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
            return RecipSqrt7Helper(vs2);
          });
    default:
      LOG(ERROR) << "vfrsqrt7.v: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Templated helper function to compute the Reciprocal approximation for valid
// normal floating point inputs.
template <typename T>
inline T Recip7(T value, FPRoundingMode rm) {
  using Uint = typename FPTypeInfo<T>::UIntType;
  using Int = typename FPTypeInfo<T>::IntType;
  Uint uint_value = *reinterpret_cast<Uint*>(&value);
  Int norm_exponent =
      (uint_value & FPTypeInfo<T>::kExpMask) >> FPTypeInfo<T>::kSigSize;
  Uint norm_mantissa = uint_value & FPTypeInfo<T>::kSigMask;
  if (norm_exponent == 0) {  // The value is a denormal.
    Uint msb = static_cast<Uint>(1) << (FPTypeInfo<T>::kSigSize - 1);
    // Normalize the mantissa and exponent by shifting the mantissa left until
    // the most significant bit is one.
    while (norm_mantissa && ((norm_mantissa & msb) == 0)) {
      norm_exponent--;
      norm_mantissa <<= 1;
    }
    // Shift it left once more - so it becomes the "implied" bit, and not used
    // in the lookup below.
    norm_mantissa <<= 1;
  }
  Int new_exponent = 2 * FPTypeInfo<T>::kExpBias - 1 - norm_exponent;
  // If the exponent is too high, then return exceptional values.
  if (new_exponent > 2 * FPTypeInfo<T>::kExpBias) {
    switch (rm) {
      case FPRoundingMode::kRoundDown:
        return std::signbit(value) ? -std::numeric_limits<T>::infinity()
                                   : std::numeric_limits<T>::max();
      case FPRoundingMode::kRoundTowardsZero:
        return std::signbit(value) ? std::numeric_limits<T>::lowest()
                                   : std::numeric_limits<T>::max();
      case FPRoundingMode::kRoundToNearestTiesToMax:
      case FPRoundingMode::kRoundToNearest:
        return std::signbit(value) ? -std::numeric_limits<T>::infinity()
                                   : std::numeric_limits<T>::infinity();
      case FPRoundingMode::kRoundUp:
        return std::signbit(value) ? std::numeric_limits<T>::lowest()
                                   : std::numeric_limits<T>::infinity();
      default:
        // kDynamic can't happen.
        return std::numeric_limits<T>::quiet_NaN();
    }
  }
  // Perform table lookup and compute the new value using the new exponent.
  int index = (norm_mantissa >> (FPTypeInfo<T>::kSigSize - 7)) & 0b111'1111;
  Uint new_mantissa = static_cast<Uint>(kRecipMantissaTable[index])
                      << (FPTypeInfo<T>::kSigSize - 7);
  // If the new exponent is negative or 0, the result is denormal. First
  // shift the mantissa right and or in the implied '1'.
  if (new_exponent <= 0) {
    new_mantissa = (new_mantissa >> 1) | 0b100'0000;
    // If the exponent is less than 0, shift the mantissa right.
    if (new_exponent < 0) {
      new_mantissa >>= 1;
      new_exponent = 0;
    }
    new_mantissa &= 0b111'1111;
  }
  Uint new_value = (new_exponent << FPTypeInfo<T>::kSigSize) | new_mantissa;
  T new_fp_value = *reinterpret_cast<T*>(&new_value);
  return value < 0.0 ? -new_fp_value : new_fp_value;
}

// Templated helper function to compute the Reciprocal approximation for all
// values including non-normal floating point values.
template <typename T>
inline T Recip7Helper(T value, FPRoundingMode rm) {
  auto fp_class = std::fpclassify(value);

  switch (fp_class) {
    case FP_INFINITE:
      // TODO: raise exception.
      return std::signbit(value) ? -0.0 : 0;
    case FP_NAN:
      // Just propagate the NaN.
      return std::numeric_limits<T>::quiet_NaN();
    case FP_ZERO:
      return std::signbit(value) ? -std::numeric_limits<T>::infinity()
                                 : std::numeric_limits<T>::infinity();
    case FP_SUBNORMAL:
    case FP_NORMAL:
      return Recip7(value, rm);
  }
  return std::numeric_limits<T>::quiet_NaN();
}

// Approximate reciprocal to 7 bits of mantissa.
void Vfrec7v(const Instruction* inst) {
  auto* rv_fp = static_cast<RiscVState*>(inst->state())->rv_fp();
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  ScopedFPStatus set_fpstatus(rv_fp->host_fp_interface());
  auto rm = rv_fp->GetRoundingMode();
  switch (sew) {
    case 4:
      return RiscVUnaryVectorOp<float, float>(
          rv_vector, inst,
          [rm](float vs2) -> float { return Recip7Helper(vs2, rm); });
    case 8:
      return RiscVUnaryVectorOp<double, double>(
          rv_vector, inst,
          [rm](double vs2) -> double { return Recip7Helper(vs2, rm); });
    default:
      LOG(ERROR) << "vfrec7.v: Illegal sew (" << sew << ")";
      rv_vector->set_vector_exception();
      return;
  }
}

// Classify floating point value.
void Vfclassv(const Instruction* inst) {
  auto* rv_vector = static_cast<RiscVState*>(inst->state())->rv_vector();
  int sew = rv_vector->selected_element_width();
  switch (sew) {
    case 4:
      return RiscVUnaryVectorOp<uint32_t, float>(
          rv_vector, inst, [](float vs2) -> uint32_t {
            return static_cast<uint32_t>(ClassifyFP(vs2));
          });
    case 8:
      return RiscVUnaryVectorOp<uint64_t, double>(
          rv_vector, inst, [](double vs2) -> uint64_t {
            return static_cast<uint64_t>(ClassifyFP(vs2));
          });
    default:
      rv_vector->set_vector_exception();
      LOG(ERROR) << "vfclass.v: Illegal sew (" << sew << ")";
      return;
  }
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
