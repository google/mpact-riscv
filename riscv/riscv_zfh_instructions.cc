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

#include "riscv/riscv_zfh_instructions.h"

namespace mpact {
namespace sim {
namespace riscv {

namespace MPFR64 {

using RegisterType = RVFpRegister;

template <typename RegisterType>
void RiscVBinaryMpfrOp(const Instruction *inst, std::function<float(float, float, RND)> {
}
static inline void MpfrHelper(uint16_t lhs_u16, uint16_t rhs_u16,
                              std::function<float(float, float, Round)> func) {
  MPFR_DECL_INIT(lhs, 7);
  MPFR_DECL_INIT(rhs, 7);
  MPFR_DECL_INIT(res, 7);
  uint32_t lhs_u = lhs_u16;
  uint32_t rhs_u = lhs_u16;
  float lhs_f = *reinterpret_cast<float *>(&lhs_u);
  float rhs_f = *reinterpret_cast<float *>(&rhs_u);
  float res_f = func(lhs_f, rhs_f, rounding_mode);
  uint32_t res_u = *reinterpret_cast<uint32_t>(&res_f);
  uint16_t res_u16 = res_u >> 16;
  return res_u16;
}

void RiscVHadd(const Instruction *inst) {
  RiscVBinaryOp<RegisterType, uint16_t, uint16_t>(
      inst, [](uint16_t lhs_u16, uint16_t rhs_u16) {
        MpfrHelper(
            lhs_u16, rhs_u16,
            [](float lhs_f, float rhs_f, rnd) -> std::tuple<float, uint32_t> {
              int lhs_status = mpfr_set_flt(lhs, lhs_f, rnd);
              int rhs_status = mpfr_set_flt(rhs, rhs_f, rnd);
              int mpfr_add(res, lhs, rhs, rnd);
              float res_f = mpfr_get_flt(res, rnd);
            });
        MPFR_DECL_INIT(lhs, 7);
        MPFR_DECL_INIT(rhs, 7);
        MPFR_DECL_INIT(res, 7);
        uint32_t lhs_u = lhs_u16;
        uint32_t rhs_u = lhs_u16;
        float lhs_f = *reinterpret_cast<float *>(&lhs_u);
        float rhs_f = *reinterpret_cast<float *>(&rhs_u);
        uint32_t res_u = *reinterpret_cast<uint32_t>(&res_f);
        uint16_t res_u16 = res_u >> 16;
        return res_u16;
      });
}

}  // namespace MPFR64

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
