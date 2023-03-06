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

#include "riscv/riscv_vector_opi_instructions.h"

#include <algorithm>
#include <ios>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/functional/bind_front.h"
#include "absl/random/random.h"
#include "absl/strings/string_view.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/instruction.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_vector_state.h"
#include "riscv/test/riscv_vector_instructions_test_base.h"

// This file contains test cases for most of the RiscV OPIVV, IPIVX and OPIVI
// instructions. The only instructions not covered by this file are the vector
// permutation instructions.

namespace {

using ::absl::Span;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::MakeUnsigned;
using ::mpact::sim::riscv::RiscVVectorState;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RVVectorRegister;
using ::mpact::sim::riscv::test::RiscVVectorInstructionsTestBase;

// Semantic functions.
using ::mpact::sim::riscv::Vadc;
using ::mpact::sim::riscv::Vadd;
using ::mpact::sim::riscv::Vand;
using ::mpact::sim::riscv::Vmadc;
using ::mpact::sim::riscv::Vmax;
using ::mpact::sim::riscv::Vmaxu;
using ::mpact::sim::riscv::Vmerge;
using ::mpact::sim::riscv::Vmin;
using ::mpact::sim::riscv::Vminu;
using ::mpact::sim::riscv::Vmsbc;
using ::mpact::sim::riscv::Vmseq;
using ::mpact::sim::riscv::Vmsgt;
using ::mpact::sim::riscv::Vmsgtu;
using ::mpact::sim::riscv::Vmsle;
using ::mpact::sim::riscv::Vmsleu;
using ::mpact::sim::riscv::Vmslt;
using ::mpact::sim::riscv::Vmsltu;
using ::mpact::sim::riscv::Vmsne;
using ::mpact::sim::riscv::Vmvr;
using ::mpact::sim::riscv::Vnclip;
using ::mpact::sim::riscv::Vnclipu;
using ::mpact::sim::riscv::Vnsra;
using ::mpact::sim::riscv::Vnsrl;
using ::mpact::sim::riscv::Vor;
using ::mpact::sim::riscv::Vrsub;
using ::mpact::sim::riscv::Vsadd;
using ::mpact::sim::riscv::Vsaddu;
using ::mpact::sim::riscv::Vsbc;
using ::mpact::sim::riscv::Vsll;
using ::mpact::sim::riscv::Vsmul;
using ::mpact::sim::riscv::Vsra;
using ::mpact::sim::riscv::Vsrl;
using ::mpact::sim::riscv::Vssra;
using ::mpact::sim::riscv::Vssrl;
using ::mpact::sim::riscv::Vssub;
using ::mpact::sim::riscv::Vssubu;
using ::mpact::sim::riscv::Vsub;
using ::mpact::sim::riscv::Vxor;

using ::mpact::sim::riscv::WideType;
using ::mpact::sim::riscv::test::kVd;
using ::mpact::sim::riscv::test::kVectorLengthInBytes;
using ::mpact::sim::riscv::test::kVs2;

class RiscVVectorInstructionsTest : public RiscVVectorInstructionsTestBase {};

// Each instruction is tested for each element width, and for vector-vector
// as well as vector-scalar (as applicable).

// Vector add.
// Vector-vector.
TEST_F(RiscVVectorInstructionsTest, Vadd8VV) {
  SetSemanticFunction(&Vadd);
  BinaryOpTestHelperVV<uint8_t, uint8_t, uint8_t>(
      "Vadd8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t { return val0 + val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vadd16VV) {
  SetSemanticFunction(&Vadd);
  BinaryOpTestHelperVV<uint16_t, uint16_t, uint16_t>(
      "Vadd16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t { return val0 + val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vadd32VV) {
  SetSemanticFunction(&Vadd);
  BinaryOpTestHelperVV<uint32_t, uint32_t, uint32_t>(
      "Vadd32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t { return val0 + val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vadd64VV) {
  SetSemanticFunction(&Vadd);
  BinaryOpTestHelperVV<uint64_t, uint64_t, uint64_t>(
      "Vadd64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t { return val0 + val1; });
}

// Vector-scalar.
TEST_F(RiscVVectorInstructionsTest, Vadd8VX) {
  SetSemanticFunction(&Vadd);

  BinaryOpTestHelperVX<uint8_t, uint8_t, uint8_t>(
      "Vadd8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t {
        return val0 + static_cast<uint8_t>(val1);
      });
}

TEST_F(RiscVVectorInstructionsTest, Vadd16VX) {
  SetSemanticFunction(&Vadd);
  BinaryOpTestHelperVX<uint16_t, uint16_t, uint16_t>(
      "Vadd16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t { return val0 + val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vadd32VX) {
  SetSemanticFunction(&Vadd);
  BinaryOpTestHelperVX<uint32_t, uint32_t, uint32_t>(
      "Vadd32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t { return val0 + val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vadd64VX) {
  SetSemanticFunction(&Vadd);
  BinaryOpTestHelperVX<uint64_t, uint64_t, uint64_t>(
      "Vadd64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t { return val0 + val1; });
}

// Vector subtract.
// Vector-vector.
TEST_F(RiscVVectorInstructionsTest, Vsub8VV) {
  SetSemanticFunction(&Vsub);
  BinaryOpTestHelperVV<uint8_t, uint8_t, uint8_t>(
      "Vsub8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t { return val0 - val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vsub16VV) {
  SetSemanticFunction(&Vsub);
  BinaryOpTestHelperVV<uint16_t, uint16_t, uint16_t>(
      "Vsub16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t { return val0 - val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vsub32VV) {
  SetSemanticFunction(&Vsub);
  BinaryOpTestHelperVV<uint32_t, uint32_t, uint32_t>(
      "Vsub32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t { return val0 - val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vsub64VV) {
  SetSemanticFunction(&Vsub);
  BinaryOpTestHelperVV<uint64_t, uint64_t, uint64_t>(
      "Vsub64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t { return val0 - val1; });
}

// Vector-scalar.
TEST_F(RiscVVectorInstructionsTest, Vsub8VX) {
  SetSemanticFunction(&Vsub);
  BinaryOpTestHelperVX<uint8_t, uint8_t, uint8_t>(
      "Vsub8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t { return val0 - val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vsub16VX) {
  SetSemanticFunction(&Vsub);
  BinaryOpTestHelperVX<uint16_t, uint16_t, uint16_t>(
      "Vsub16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t { return val0 - val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vsub32VX) {
  SetSemanticFunction(&Vsub);
  BinaryOpTestHelperVX<uint32_t, uint32_t, uint32_t>(
      "Vsub32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t { return val0 - val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vsub64VX) {
  SetSemanticFunction(&Vsub);
  BinaryOpTestHelperVX<uint64_t, uint64_t, uint64_t>(
      "Vsub64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t { return val0 - val1; });
}

// Vector reverse subtract.
// Vector-Scalar only.
TEST_F(RiscVVectorInstructionsTest, Vrsub8VX) {
  SetSemanticFunction(&Vrsub);
  BinaryOpTestHelperVX<uint8_t, uint8_t, uint8_t>(
      "Vrsub8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t { return val1 - val0; });
}

TEST_F(RiscVVectorInstructionsTest, Vrsub16VX) {
  SetSemanticFunction(&Vrsub);
  BinaryOpTestHelperVX<uint16_t, uint16_t, uint16_t>(
      "Vrsub16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t { return val1 - val0; });
}

TEST_F(RiscVVectorInstructionsTest, Vrsub32VX) {
  SetSemanticFunction(&Vrsub);
  BinaryOpTestHelperVX<uint32_t, uint32_t, uint32_t>(
      "Vrsub32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t { return val1 - val0; });
}

TEST_F(RiscVVectorInstructionsTest, Vrsub64VX) {
  SetSemanticFunction(&Vrsub);
  BinaryOpTestHelperVX<uint64_t, uint64_t, uint64_t>(
      "Vrsub64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t { return val1 - val0; });
}

// Vector and.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vand8VV) {
  SetSemanticFunction(&Vand);

  BinaryOpTestHelperVV<uint8_t, uint8_t, uint8_t>(
      "Vand8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t { return val0 & val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vand16VV) {
  SetSemanticFunction(&Vand);

  BinaryOpTestHelperVV<uint16_t, uint16_t, uint16_t>(
      "Vand16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t { return val0 & val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vand32VV) {
  SetSemanticFunction(&Vand);

  BinaryOpTestHelperVV<uint32_t, uint32_t, uint32_t>(
      "Vand32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t { return val0 & val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vand64VV) {
  SetSemanticFunction(&Vand);

  BinaryOpTestHelperVV<uint64_t, uint64_t, uint64_t>(
      "Vand64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t { return val0 & val1; });
}

// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vand8VX) {
  SetSemanticFunction(&Vand);

  BinaryOpTestHelperVX<uint8_t, uint8_t, uint8_t>(
      "Vand8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t { return val0 & val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vand16VX) {
  SetSemanticFunction(&Vand);

  BinaryOpTestHelperVX<uint16_t, uint16_t, uint16_t>(
      "Vand16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t { return val0 & val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vand32VX) {
  SetSemanticFunction(&Vand);

  BinaryOpTestHelperVX<uint32_t, uint32_t, uint32_t>(
      "Vand32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t { return val0 & val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vand64VX) {
  SetSemanticFunction(&Vand);

  BinaryOpTestHelperVX<uint64_t, uint64_t, uint64_t>(
      "Vand64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t { return val0 & val1; });
}

// Vector or.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vor8VV) {
  SetSemanticFunction(&Vor);

  BinaryOpTestHelperVV<uint8_t, uint8_t, uint8_t>(
      "Vor8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t { return val0 | val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vor16VV) {
  SetSemanticFunction(&Vor);

  BinaryOpTestHelperVV<uint16_t, uint16_t, uint16_t>(
      "Vor16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t { return val0 | val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vor32VV) {
  SetSemanticFunction(&Vor);

  BinaryOpTestHelperVV<uint32_t, uint32_t, uint32_t>(
      "Vor32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t { return val0 | val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vor64VV) {
  SetSemanticFunction(&Vor);

  BinaryOpTestHelperVV<uint64_t, uint64_t, uint64_t>(
      "Vor64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t { return val0 | val1; });
}

// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vor8VX) {
  SetSemanticFunction(&Vor);

  BinaryOpTestHelperVX<uint8_t, uint8_t, uint8_t>(
      "Vor8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t { return val0 | val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vor16VX) {
  SetSemanticFunction(&Vor);

  BinaryOpTestHelperVX<uint16_t, uint16_t, uint16_t>(
      "Vor16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t { return val0 | val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vor32VX) {
  SetSemanticFunction(&Vor);

  BinaryOpTestHelperVX<uint32_t, uint32_t, uint32_t>(
      "Vor32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t { return val0 | val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vor64VX) {
  SetSemanticFunction(&Vor);

  BinaryOpTestHelperVX<uint64_t, uint64_t, uint64_t>(
      "Vor64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t { return val0 | val1; });
}

// Vector xor.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vxor8VV) {
  SetSemanticFunction(&Vxor);

  BinaryOpTestHelperVV<uint8_t, uint8_t, uint8_t>(
      "Vxor8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t { return val0 ^ val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vxor16VV) {
  SetSemanticFunction(&Vxor);

  BinaryOpTestHelperVV<uint16_t, uint16_t, uint16_t>(
      "Vxor16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t { return val0 ^ val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vxor32VV) {
  SetSemanticFunction(&Vxor);

  BinaryOpTestHelperVV<uint32_t, uint32_t, uint32_t>(
      "Vxor32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t { return val0 ^ val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vxor64VV) {
  SetSemanticFunction(&Vxor);

  BinaryOpTestHelperVV<uint64_t, uint64_t, uint64_t>(
      "Vxor64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t { return val0 ^ val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vxor8VX) {
  SetSemanticFunction(&Vxor);

  BinaryOpTestHelperVX<uint8_t, uint8_t, uint8_t>(
      "Vxor8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t { return val0 ^ val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vxor16VX) {
  SetSemanticFunction(&Vxor);

  BinaryOpTestHelperVX<uint16_t, uint16_t, uint16_t>(
      "Vxor16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t { return val0 ^ val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vxor32VX) {
  SetSemanticFunction(&Vxor);

  BinaryOpTestHelperVX<uint32_t, uint32_t, uint32_t>(
      "Vxor32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t { return val0 ^ val1; });
}

TEST_F(RiscVVectorInstructionsTest, Vxor64VX) {
  SetSemanticFunction(&Vxor);

  BinaryOpTestHelperVX<uint64_t, uint64_t, uint64_t>(
      "Vxor64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t { return val0 ^ val1; });
}

// Vector sll.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vsll8VV) {
  SetSemanticFunction(&Vsll);

  BinaryOpTestHelperVV<uint8_t, uint8_t, uint8_t>(
      "Vsll8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t {
        return val0 << (val1 & 0b111);
      });
}

TEST_F(RiscVVectorInstructionsTest, Vsll16VV) {
  SetSemanticFunction(&Vsll);

  BinaryOpTestHelperVV<uint16_t, uint16_t, uint16_t>(
      "Vsll16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t {
        return val0 << (val1 & 0b1111);
      });
}

TEST_F(RiscVVectorInstructionsTest, Vsll32VV) {
  SetSemanticFunction(&Vsll);

  BinaryOpTestHelperVV<uint32_t, uint32_t, uint32_t>(
      "Vsll32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t {
        return val0 << (val1 & 0b1'1111);
      });
}

TEST_F(RiscVVectorInstructionsTest, Vsll64VV) {
  SetSemanticFunction(&Vsll);

  BinaryOpTestHelperVV<uint64_t, uint64_t, uint64_t>(
      "Vsll64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t {
        return val0 << (val1 & 0b11'1111);
      });
}

// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vsll8VX) {
  SetSemanticFunction(&Vsll);

  BinaryOpTestHelperVX<uint8_t, uint8_t, uint8_t>(
      "Vsll8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t {
        return val0 << (val1 & 0b111);
      });
}

TEST_F(RiscVVectorInstructionsTest, Vsll16VX) {
  SetSemanticFunction(&Vsll);

  BinaryOpTestHelperVX<uint16_t, uint16_t, uint16_t>(
      "Vsll16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t {
        return val0 << (val1 & 0b1111);
      });
}

TEST_F(RiscVVectorInstructionsTest, Vsll32VX) {
  SetSemanticFunction(&Vsll);

  BinaryOpTestHelperVX<uint32_t, uint32_t, uint32_t>(
      "Vsll32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t {
        return val0 << (val1 & 0b1'1111);
      });
}

TEST_F(RiscVVectorInstructionsTest, Vsll64VX) {
  SetSemanticFunction(&Vsll);

  BinaryOpTestHelperVX<uint64_t, uint64_t, uint64_t>(
      "Vsll64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t {
        return val0 << (val1 & 0b11'1111);
      });
}

// Vector srl.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vsrl8VV) {
  SetSemanticFunction(&Vsrl);

  BinaryOpTestHelperVV<uint8_t, uint8_t, uint8_t>(
      "Vsrl8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t {
        return val0 >> (val1 & 0b111);
      });
}

TEST_F(RiscVVectorInstructionsTest, Vsrl16VV) {
  SetSemanticFunction(&Vsrl);

  BinaryOpTestHelperVV<uint16_t, uint16_t, uint16_t>(
      "Vsrl16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t {
        return val0 >> (val1 & 0b1111);
      });
}

TEST_F(RiscVVectorInstructionsTest, Vsrl32VV) {
  SetSemanticFunction(&Vsrl);

  BinaryOpTestHelperVV<uint32_t, uint32_t, uint32_t>(
      "Vsrl32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t {
        return val0 >> (val1 & 0b1'1111);
      });
}

TEST_F(RiscVVectorInstructionsTest, Vsrl64VV) {
  SetSemanticFunction(&Vsrl);

  BinaryOpTestHelperVV<uint64_t, uint64_t, uint64_t>(
      "Vsrl64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t {
        return val0 >> (val1 & 0b11'1111);
      });
}

// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vsrl8VX) {
  SetSemanticFunction(&Vsrl);

  BinaryOpTestHelperVX<uint8_t, uint8_t, uint8_t>(
      "Vsrl8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t {
        return val0 >> (val1 & 0b111);
      });
}

TEST_F(RiscVVectorInstructionsTest, Vsrl16VX) {
  SetSemanticFunction(&Vsrl);

  BinaryOpTestHelperVX<uint16_t, uint16_t, uint16_t>(
      "Vsrl16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t {
        return val0 >> (val1 & 0b1111);
      });
}

TEST_F(RiscVVectorInstructionsTest, Vsrl32VX) {
  SetSemanticFunction(&Vsrl);

  BinaryOpTestHelperVX<uint32_t, uint32_t, uint32_t>(
      "Vsrl32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t {
        return val0 >> (val1 & 0b1'1111);
      });
}

TEST_F(RiscVVectorInstructionsTest, Vsrl64VX) {
  SetSemanticFunction(&Vsrl);

  BinaryOpTestHelperVX<uint64_t, uint64_t, uint64_t>(
      "Vsrl64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t {
        return val0 >> (val1 & 0b11'1111);
      });
}

// Vector sra.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vsra8VV) {
  SetSemanticFunction(&Vsra);

  BinaryOpTestHelperVV<uint8_t, int8_t, uint8_t>(
      "Vsra8", /*sew*/ 8, instruction_,
      [](int8_t val0, uint8_t val1) -> int8_t {
        return val0 >> (val1 & 0b111);
      });
}

TEST_F(RiscVVectorInstructionsTest, Vsra16VV) {
  SetSemanticFunction(&Vsra);

  BinaryOpTestHelperVV<uint16_t, int16_t, uint16_t>(
      "Vsra16", /*sew*/ 16, instruction_,
      [](int16_t val0, uint16_t val1) -> int16_t {
        return val0 >> (val1 & 0b1111);
      });
}

TEST_F(RiscVVectorInstructionsTest, Vsra32VV) {
  SetSemanticFunction(&Vsra);

  BinaryOpTestHelperVV<uint32_t, int32_t, uint32_t>(
      "Vsra32", /*sew*/ 32, instruction_,
      [](int32_t val0, uint32_t val1) -> int32_t {
        return val0 >> (val1 & 0b1'1111);
      });
}

TEST_F(RiscVVectorInstructionsTest, Vsra64VV) {
  SetSemanticFunction(&Vsra);

  BinaryOpTestHelperVV<uint64_t, int64_t, uint64_t>(
      "Vsll64", /*sew*/ 64, instruction_,
      [](int64_t val0, uint64_t val1) -> int64_t {
        return val0 >> (val1 & 0b11'1111);
      });
}

// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vsra8VX) {
  SetSemanticFunction(&Vsra);

  BinaryOpTestHelperVX<uint8_t, int8_t, uint8_t>(
      "Vsra8", /*sew*/ 8, instruction_,
      [](int8_t val0, uint8_t val1) -> int8_t {
        return val0 >> (val1 & 0b111);
      });
}

TEST_F(RiscVVectorInstructionsTest, Vsra16VX) {
  SetSemanticFunction(&Vsra);

  BinaryOpTestHelperVX<uint16_t, int16_t, uint16_t>(
      "Vsra16", /*sew*/ 16, instruction_,
      [](int16_t val0, uint16_t val1) -> int16_t {
        return val0 >> (val1 & 0b1111);
      });
}

TEST_F(RiscVVectorInstructionsTest, Vsra32VX) {
  SetSemanticFunction(&Vsra);

  BinaryOpTestHelperVX<uint32_t, int32_t, uint32_t>(
      "Vsra32", /*sew*/ 32, instruction_,
      [](int32_t val0, uint32_t val1) -> int32_t {
        return val0 >> (val1 & 0b1'1111);
      });
}

TEST_F(RiscVVectorInstructionsTest, Vsra64VX) {
  SetSemanticFunction(&Vsra);

  BinaryOpTestHelperVX<uint64_t, int64_t, uint64_t>(
      "Vsll64", /*sew*/ 64, instruction_,
      [](int64_t val0, uint64_t val1) -> int64_t {
        return val0 >> (val1 & 0b11'1111);
      });
}

// Vector narrowing srl.
// Vector-Vector.VV
TEST_F(RiscVVectorInstructionsTest, Vnsrl8VV) {
  SetSemanticFunction(&Vnsrl);

  BinaryOpTestHelperVV<uint8_t, uint16_t, uint8_t>(
      "Vsra8", /*sew*/ 8, instruction_,
      [](uint16_t val0, uint8_t val1) -> uint8_t {
        return static_cast<uint8_t>(val0 >> (val1 & 0b1111));
      });
}

TEST_F(RiscVVectorInstructionsTest, Vnsrl16VV) {
  SetSemanticFunction(&Vnsrl);

  BinaryOpTestHelperVV<uint16_t, uint32_t, uint16_t>(
      "Vsll16", /*sew*/ 16, instruction_,
      [](uint32_t val0, uint16_t val1) -> uint16_t {
        return static_cast<uint16_t>(val0 >> (val1 & 0b1'1111));
      });
}

TEST_F(RiscVVectorInstructionsTest, Vnsrl32VV) {
  SetSemanticFunction(&Vnsrl);

  BinaryOpTestHelperVV<uint32_t, uint64_t, uint32_t>(
      "Vsll32", /*sew*/ 32, instruction_,
      [](uint64_t val0, uint32_t val1) -> uint32_t {
        return static_cast<uint32_t>(val0 >> (val1 & 0b11'1111));
      });
}

// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vnsrl8VX) {
  SetSemanticFunction(&Vnsrl);

  BinaryOpTestHelperVX<uint8_t, uint16_t, uint8_t>(
      "Vsra8", /*sew*/ 8, instruction_,
      [](uint16_t val0, uint8_t val1) -> uint8_t {
        return static_cast<uint8_t>(val0 >> (val1 & 0b1111));
      });
}

TEST_F(RiscVVectorInstructionsTest, Vnsrl16VX) {
  SetSemanticFunction(&Vnsrl);

  BinaryOpTestHelperVX<uint16_t, uint32_t, uint16_t>(
      "Vsll16", /*sew*/ 16, instruction_,
      [](uint32_t val0, uint16_t val1) -> uint16_t {
        return static_cast<uint16_t>(val0 >> (val1 & 0b1'1111));
      });
}

TEST_F(RiscVVectorInstructionsTest, Vnsrl32VX) {
  SetSemanticFunction(&Vnsrl);

  BinaryOpTestHelperVX<uint32_t, uint64_t, uint32_t>(
      "Vsll32", /*sew*/ 32, instruction_,
      [](uint64_t val0, uint32_t val1) -> uint32_t {
        return static_cast<uint32_t>(val0 >> (val1 & 0b11'1111));
      });
}

// Vector narrowing sra.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vnsra8VV) {
  SetSemanticFunction(&Vnsra);

  BinaryOpTestHelperVV<uint8_t, uint16_t, uint8_t>(
      "Vsra8", /*sew*/ 8, instruction_,
      [](int16_t val0, uint8_t val1) -> uint8_t {
        return static_cast<uint8_t>(val0 >> (val1 & 0b1111));
      });
}

TEST_F(RiscVVectorInstructionsTest, Vnsra16VV) {
  SetSemanticFunction(&Vnsra);

  BinaryOpTestHelperVV<uint16_t, uint32_t, uint16_t>(
      "Vsll16", /*sew*/ 16, instruction_,
      [](int32_t val0, uint16_t val1) -> uint16_t {
        return static_cast<uint16_t>(val0 >> (val1 & 0b1'1111));
      });
}

TEST_F(RiscVVectorInstructionsTest, Vnsra32VV) {
  SetSemanticFunction(&Vnsra);

  BinaryOpTestHelperVV<uint32_t, uint64_t, uint32_t>(
      "Vsll32", /*sew*/ 32, instruction_,
      [](int64_t val0, uint32_t val1) -> uint32_t {
        return static_cast<uint32_t>(val0 >> (val1 & 0b11'1111));
      });
}

// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vnsra8VX) {
  SetSemanticFunction(&Vnsra);

  BinaryOpTestHelperVX<uint8_t, uint16_t, uint8_t>(
      "Vsra8", /*sew*/ 8, instruction_,
      [](int16_t val0, uint8_t val1) -> uint8_t {
        return static_cast<uint8_t>(val0 >> (val1 & 0b1111));
      });
}

TEST_F(RiscVVectorInstructionsTest, Vnsra16VX) {
  SetSemanticFunction(&Vnsra);

  BinaryOpTestHelperVX<uint16_t, uint32_t, uint16_t>(
      "Vsll16", /*sew*/ 16, instruction_,
      [](int32_t val0, uint16_t val1) -> uint16_t {
        return static_cast<uint16_t>(val0 >> (val1 & 0b1'1111));
      });
}

TEST_F(RiscVVectorInstructionsTest, Vnsra32VX) {
  SetSemanticFunction(&Vnsra);

  BinaryOpTestHelperVX<uint32_t, uint64_t, uint32_t>(
      "Vsll32", /*sew*/ 32, instruction_,
      [](int64_t val0, uint32_t val1) -> uint32_t {
        return static_cast<uint32_t>(val0 >> (val1 & 0b11'1111));
      });
}

// Vector unsigned min.
// Vector-Vector
TEST_F(RiscVVectorInstructionsTest, Vminu8VV) {
  SetSemanticFunction(&Vminu);
  BinaryOpTestHelperVV<uint8_t, uint8_t, uint8_t>(
      "Vminu8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t {
        return (val0 < val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vminu16VV) {
  SetSemanticFunction(&Vminu);
  BinaryOpTestHelperVV<uint16_t, uint16_t, uint16_t>(
      "Vminu16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t {
        return (val0 < val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vminu32VV) {
  SetSemanticFunction(&Vminu);
  BinaryOpTestHelperVV<uint32_t, uint32_t, uint32_t>(
      "Vminu32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t {
        return (val0 < val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vminu64VV) {
  SetSemanticFunction(&Vminu);
  BinaryOpTestHelperVV<uint64_t, uint64_t, uint64_t>(
      "Vminu64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t {
        return (val0 < val1) ? val0 : val1;
      });
}
// Vector-Scalar
TEST_F(RiscVVectorInstructionsTest, Vminu8VX) {
  SetSemanticFunction(&Vminu);
  BinaryOpTestHelperVX<uint8_t, uint8_t, uint8_t>(
      "Vminu8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t {
        return (val0 < val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vminu16VX) {
  SetSemanticFunction(&Vminu);
  BinaryOpTestHelperVX<uint16_t, uint16_t, uint16_t>(
      "Vminu16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t {
        return (val0 < val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vminu32VX) {
  SetSemanticFunction(&Vminu);
  BinaryOpTestHelperVX<uint32_t, uint32_t, uint32_t>(
      "Vminu32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t {
        return (val0 < val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vminu64VX) {
  SetSemanticFunction(&Vminu);
  BinaryOpTestHelperVX<uint64_t, uint64_t, uint64_t>(
      "Vminu64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t {
        return (val0 < val1) ? val0 : val1;
      });
}

// Vector signed min.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vmin8VV) {
  SetSemanticFunction(&Vmin);
  BinaryOpTestHelperVV<int8_t, int8_t, int8_t>(
      "Vmin8", /*sew*/ 8, instruction_, [](int8_t val0, int8_t val1) -> int8_t {
        return (val0 < val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmin16VV) {
  SetSemanticFunction(&Vmin);
  BinaryOpTestHelperVV<int16_t, int16_t, int16_t>(
      "Vmin16", /*sew*/ 16, instruction_,
      [](int16_t val0, int16_t val1) -> int16_t {
        return (val0 < val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmin32VV) {
  SetSemanticFunction(&Vmin);
  BinaryOpTestHelperVV<int32_t, int32_t, int32_t>(
      "Vmin32", /*sew*/ 32, instruction_,
      [](int32_t val0, int32_t val1) -> int32_t {
        return (val0 < val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmin64VV) {
  SetSemanticFunction(&Vmin);
  BinaryOpTestHelperVV<int64_t, int64_t, int64_t>(
      "Vmin64", /*sew*/ 64, instruction_,
      [](int64_t val0, int64_t val1) -> int64_t {
        return (val0 < val1) ? val0 : val1;
      });
}
// Vector-Scalar
TEST_F(RiscVVectorInstructionsTest, Vmin8VX) {
  SetSemanticFunction(&Vmin);
  BinaryOpTestHelperVX<int8_t, int8_t, int8_t>(
      "Vmin8", /*sew*/ 8, instruction_, [](int8_t val0, int8_t val1) -> int8_t {
        return (val0 < val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmin16VX) {
  SetSemanticFunction(&Vmin);
  BinaryOpTestHelperVX<int16_t, int16_t, int16_t>(
      "Vmin16", /*sew*/ 16, instruction_,
      [](int16_t val0, int16_t val1) -> int16_t {
        return (val0 < val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmin32VX) {
  SetSemanticFunction(&Vmin);
  BinaryOpTestHelperVX<int32_t, int32_t, int32_t>(
      "Vmin32", /*sew*/ 32, instruction_,
      [](int32_t val0, int32_t val1) -> int32_t {
        return (val0 < val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmin64VX) {
  SetSemanticFunction(&Vmin);
  BinaryOpTestHelperVX<int64_t, int64_t, int64_t>(
      "Vmin64", /*sew*/ 64, instruction_,
      [](int64_t val0, int64_t val1) -> int64_t {
        return (val0 < val1) ? val0 : val1;
      });
}

// Vector unsigned max.
// Vector-Vector
TEST_F(RiscVVectorInstructionsTest, Vmaxu8VV) {
  SetSemanticFunction(&Vmaxu);
  BinaryOpTestHelperVV<uint8_t, uint8_t, uint8_t>(
      "Vmaxu8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t {
        return (val0 > val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmaxu16VV) {
  SetSemanticFunction(&Vmaxu);
  BinaryOpTestHelperVV<uint16_t, uint16_t, uint16_t>(
      "Vmaxu16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t {
        return (val0 > val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmaxu32VV) {
  SetSemanticFunction(&Vmaxu);
  BinaryOpTestHelperVV<uint32_t, uint32_t, uint32_t>(
      "Vmaxu32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t {
        return (val0 > val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmaxu64VV) {
  SetSemanticFunction(&Vmaxu);
  BinaryOpTestHelperVV<uint64_t, uint64_t, uint64_t>(
      "Vmaxu64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t {
        return (val0 > val1) ? val0 : val1;
      });
}
// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vmaxu8VX) {
  SetSemanticFunction(&Vmaxu);
  BinaryOpTestHelperVX<uint8_t, uint8_t, uint8_t>(
      "Vmaxu8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t {
        return (val0 > val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmaxu16VX) {
  SetSemanticFunction(&Vmaxu);
  BinaryOpTestHelperVX<uint16_t, uint16_t, uint16_t>(
      "Vmaxu16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t {
        return (val0 > val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmaxu32VX) {
  SetSemanticFunction(&Vmaxu);
  BinaryOpTestHelperVX<uint32_t, uint32_t, uint32_t>(
      "Vmaxu32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t {
        return (val0 > val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmaxu64VX) {
  SetSemanticFunction(&Vmaxu);
  BinaryOpTestHelperVX<uint64_t, uint64_t, uint64_t>(
      "Vmaxu64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t {
        return (val0 > val1) ? val0 : val1;
      });
}
// Vector signed max.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vmax8VV) {
  SetSemanticFunction(&Vmax);
  BinaryOpTestHelperVV<int8_t, int8_t, int8_t>(
      "Vmin8", /*sew*/ 8, instruction_, [](int8_t val0, int8_t val1) -> int8_t {
        return (val0 > val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmax16VV) {
  SetSemanticFunction(&Vmax);
  BinaryOpTestHelperVV<int16_t, int16_t, int16_t>(
      "Vmin16", /*sew*/ 16, instruction_,
      [](int16_t val0, int16_t val1) -> int16_t {
        return (val0 > val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmax32VV) {
  SetSemanticFunction(&Vmax);
  BinaryOpTestHelperVV<int32_t, int32_t, int32_t>(
      "Vmin32", /*sew*/ 32, instruction_,
      [](int32_t val0, int32_t val1) -> int32_t {
        return (val0 > val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmax64VV) {
  SetSemanticFunction(&Vmax);
  BinaryOpTestHelperVV<int64_t, int64_t, int64_t>(
      "Vmin64", /*sew*/ 64, instruction_,
      [](int64_t val0, int64_t val1) -> int64_t {
        return (val0 > val1) ? val0 : val1;
      });
}
// Vector-Scalar
TEST_F(RiscVVectorInstructionsTest, Vmax8VX) {
  SetSemanticFunction(&Vmax);
  BinaryOpTestHelperVX<int8_t, int8_t, int8_t>(
      "Vmin8", /*sew*/ 8, instruction_, [](int8_t val0, int8_t val1) -> int8_t {
        return (val0 > val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmax16VX) {
  SetSemanticFunction(&Vmax);
  BinaryOpTestHelperVX<int16_t, int16_t, int16_t>(
      "Vmin16", /*sew*/ 16, instruction_,
      [](int16_t val0, int16_t val1) -> int16_t {
        return (val0 > val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmax32VX) {
  SetSemanticFunction(&Vmax);
  BinaryOpTestHelperVX<int32_t, int32_t, int32_t>(
      "Vmin32", /*sew*/ 32, instruction_,
      [](int32_t val0, int32_t val1) -> int32_t {
        return (val0 > val1) ? val0 : val1;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmax64VX) {
  SetSemanticFunction(&Vmax);
  BinaryOpTestHelperVX<int64_t, int64_t, int64_t>(
      "Vmin64", /*sew*/ 64, instruction_,
      [](int64_t val0, int64_t val1) -> int64_t {
        return (val0 > val1) ? val0 : val1;
      });
}

// Integer compare instructions.

// Vector mask set equal.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vmseq8VV) {
  SetSemanticFunction(&Vmseq);
  BinaryMaskOpTestHelperVV<uint8_t, uint8_t>(
      "Vmseq8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t {
        return (val0 == val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmseq16VV) {
  SetSemanticFunction(&Vmseq);
  BinaryMaskOpTestHelperVV<uint16_t, uint16_t>(
      "Vmseq16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t {
        return (val0 == val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmseq32VV) {
  SetSemanticFunction(&Vmseq);
  BinaryMaskOpTestHelperVV<uint32_t, uint32_t>(
      "Vmseq32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t {
        return (val0 == val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmseq64VV) {
  SetSemanticFunction(&Vmseq);
  BinaryMaskOpTestHelperVV<uint64_t, uint64_t>(
      "Vmseq64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t {
        return (val0 == val1) ? 1 : 0;
      });
}
// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vmseq8VX) {
  SetSemanticFunction(&Vmseq);
  BinaryMaskOpTestHelperVX<uint8_t, uint8_t>(
      "Vmseq8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t {
        return (val0 == val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmseq16VX) {
  SetSemanticFunction(&Vmseq);
  BinaryMaskOpTestHelperVX<uint16_t, uint16_t>(
      "Vmseq16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint8_t {
        return (val0 == val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmseq32VX) {
  SetSemanticFunction(&Vmseq);
  BinaryMaskOpTestHelperVX<uint32_t, uint32_t>(
      "Vmseq32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint8_t {
        return (val0 == val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmseq64VX) {
  SetSemanticFunction(&Vmseq);
  BinaryMaskOpTestHelperVX<uint64_t, uint64_t>(
      "Vmseq64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint8_t {
        return (val0 == val1) ? 1 : 0;
      });
}

// Vector mask set not equal.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vmsne8VV) {
  SetSemanticFunction(&Vmsne);
  BinaryMaskOpTestHelperVV<uint8_t, uint8_t>(
      "Vmsne8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t {
        return (val0 != val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsne16VV) {
  SetSemanticFunction(&Vmsne);
  BinaryMaskOpTestHelperVV<uint16_t, uint16_t>(
      "Vmsne16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t {
        return (val0 != val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsne32VV) {
  SetSemanticFunction(&Vmsne);
  BinaryMaskOpTestHelperVV<uint32_t, uint32_t>(
      "Vmsne32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t {
        return (val0 != val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsne64VV) {
  SetSemanticFunction(&Vmsne);
  BinaryMaskOpTestHelperVV<uint64_t, uint64_t>(
      "Vmsne64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t {
        return (val0 != val1) ? 1 : 0;
      });
}
// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vmsne8VX) {
  SetSemanticFunction(&Vmsne);
  BinaryMaskOpTestHelperVX<uint8_t, uint8_t>(
      "Vmsne8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t {
        return (val0 != val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsne16VX) {
  SetSemanticFunction(&Vmsne);
  BinaryMaskOpTestHelperVX<uint16_t, uint16_t>(
      "Vmsne16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint8_t {
        return (val0 != val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsne32VX) {
  SetSemanticFunction(&Vmsne);
  BinaryMaskOpTestHelperVX<uint32_t, uint32_t>(
      "Vmsne32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint8_t {
        return (val0 != val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsne64VX) {
  SetSemanticFunction(&Vmsne);
  BinaryMaskOpTestHelperVX<uint64_t, uint64_t>(
      "Vmsne64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint8_t {
        return (val0 != val1) ? 1 : 0;
      });
}

// Vector mask unsigned set less than.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vmsltu8VV) {
  SetSemanticFunction(&Vmsltu);
  BinaryMaskOpTestHelperVV<uint8_t, uint8_t>(
      "Vmsltu8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t {
        return (val0 < val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsltu16VV) {
  SetSemanticFunction(&Vmsltu);
  BinaryMaskOpTestHelperVV<uint16_t, uint16_t>(
      "Vmsltu16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t {
        return (val0 < val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsltu32VV) {
  SetSemanticFunction(&Vmsltu);
  BinaryMaskOpTestHelperVV<uint32_t, uint32_t>(
      "Vmsltu32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t {
        return (val0 < val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsltu64VV) {
  SetSemanticFunction(&Vmsltu);
  BinaryMaskOpTestHelperVV<uint64_t, uint64_t>(
      "Vmsltu64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t {
        return (val0 < val1) ? 1 : 0;
      });
}
// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vmsltu8VX) {
  SetSemanticFunction(&Vmsltu);
  BinaryMaskOpTestHelperVX<uint8_t, uint8_t>(
      "Vmsltu8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t {
        return (val0 < val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsltu16VX) {
  SetSemanticFunction(&Vmsltu);
  BinaryMaskOpTestHelperVX<uint16_t, uint16_t>(
      "Vmsltu16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint8_t {
        return (val0 < val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsltu32VX) {
  SetSemanticFunction(&Vmsltu);
  BinaryMaskOpTestHelperVX<uint32_t, uint32_t>(
      "Vmsltu32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint8_t {
        return (val0 < val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsltu64VX) {
  SetSemanticFunction(&Vmsltu);
  BinaryMaskOpTestHelperVX<uint64_t, uint64_t>(
      "Vmsltu64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint8_t {
        return (val0 < val1) ? 1 : 0;
      });
}

// Vector mask signed set less than.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vmslt8VV) {
  SetSemanticFunction(&Vmslt);
  BinaryMaskOpTestHelperVV<int8_t, int8_t>(
      "Vmslt8", /*sew*/ 8, instruction_,
      [](int8_t val0, int8_t val1) -> uint8_t {
        return (val0 < val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmslt16VV) {
  SetSemanticFunction(&Vmslt);
  BinaryMaskOpTestHelperVV<int16_t, int16_t>(
      "Vmslt16", /*sew*/ 16, instruction_,
      [](int16_t val0, int16_t val1) -> uint16_t {
        return (val0 < val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmslt32VV) {
  SetSemanticFunction(&Vmslt);
  BinaryMaskOpTestHelperVV<int32_t, int32_t>(
      "Vmslt32", /*sew*/ 32, instruction_,
      [](int32_t val0, int32_t val1) -> uint32_t {
        return (val0 < val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmslt64VV) {
  SetSemanticFunction(&Vmslt);
  BinaryMaskOpTestHelperVV<int64_t, int64_t>(
      "Vmslt64", /*sew*/ 64, instruction_,
      [](int64_t val0, int64_t val1) -> uint64_t {
        return (val0 < val1) ? 1 : 0;
      });
}
// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vmslt8VX) {
  SetSemanticFunction(&Vmslt);
  BinaryMaskOpTestHelperVX<int8_t, int8_t>(
      "Vmslt8", /*sew*/ 8, instruction_,
      [](int8_t val0, int8_t val1) -> uint8_t {
        return (val0 < val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmslt16VX) {
  SetSemanticFunction(&Vmslt);
  BinaryMaskOpTestHelperVX<int16_t, int16_t>(
      "Vmslt16", /*sew*/ 16, instruction_,
      [](int16_t val0, int16_t val1) -> uint8_t {
        return (val0 < val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmslt32VX) {
  SetSemanticFunction(&Vmslt);
  BinaryMaskOpTestHelperVX<int32_t, int32_t>(
      "Vmslt32", /*sew*/ 32, instruction_,
      [](int32_t val0, int32_t val1) -> uint8_t {
        return (val0 < val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmslt64VX) {
  SetSemanticFunction(&Vmslt);
  BinaryMaskOpTestHelperVX<int64_t, int64_t>(
      "Vmslt64", /*sew*/ 64, instruction_,
      [](int64_t val0, int64_t val1) -> uint8_t {
        return (val0 < val1) ? 1 : 0;
      });
}

// Vector mask unsigned set less than or equal.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vmsleu8VV) {
  SetSemanticFunction(&Vmsleu);
  BinaryMaskOpTestHelperVV<uint8_t, uint8_t>(
      "Vmsleu8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t {
        return (val0 <= val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsleu16VV) {
  SetSemanticFunction(&Vmsleu);
  BinaryMaskOpTestHelperVV<uint16_t, uint16_t>(
      "Vmsleu16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint16_t {
        return (val0 <= val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsleu32VV) {
  SetSemanticFunction(&Vmsleu);
  BinaryMaskOpTestHelperVV<uint32_t, uint32_t>(
      "Vmsleu32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint32_t {
        return (val0 <= val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsleu64VV) {
  SetSemanticFunction(&Vmsleu);
  BinaryMaskOpTestHelperVV<uint64_t, uint64_t>(
      "Vmsleu64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint64_t {
        return (val0 <= val1) ? 1 : 0;
      });
}
// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vmsleu8VX) {
  SetSemanticFunction(&Vmsleu);
  BinaryMaskOpTestHelperVX<uint8_t, uint8_t>(
      "Vmsleu8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t {
        return (val0 <= val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsleu16VX) {
  SetSemanticFunction(&Vmsleu);
  BinaryMaskOpTestHelperVX<uint16_t, uint16_t>(
      "Vmsleu16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint8_t {
        return (val0 <= val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsleu32VX) {
  SetSemanticFunction(&Vmsleu);
  BinaryMaskOpTestHelperVX<uint32_t, uint32_t>(
      "Vmsleu32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint8_t {
        return (val0 <= val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsleu64VX) {
  SetSemanticFunction(&Vmsleu);
  BinaryMaskOpTestHelperVX<uint64_t, uint64_t>(
      "Vmsleu64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint8_t {
        return (val0 <= val1) ? 1 : 0;
      });
}

// Vector mask signed set less than or equal.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vmsle8VV) {
  SetSemanticFunction(&Vmsle);
  BinaryMaskOpTestHelperVV<int8_t, int8_t>(
      "Vmsle8", /*sew*/ 8, instruction_,
      [](int8_t val0, int8_t val1) -> uint8_t {
        return (val0 <= val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsle16VV) {
  SetSemanticFunction(&Vmsle);
  BinaryMaskOpTestHelperVV<int16_t, int16_t>(
      "Vmsle16", /*sew*/ 16, instruction_,
      [](int16_t val0, int16_t val1) -> uint16_t {
        return (val0 <= val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsle32VV) {
  SetSemanticFunction(&Vmsle);
  BinaryMaskOpTestHelperVV<int32_t, int32_t>(
      "Vmsle32", /*sew*/ 32, instruction_,
      [](int32_t val0, int32_t val1) -> uint32_t {
        return (val0 <= val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsle64VV) {
  SetSemanticFunction(&Vmsle);
  BinaryMaskOpTestHelperVV<int64_t, int64_t>(
      "Vmsle64", /*sew*/ 64, instruction_,
      [](int64_t val0, int64_t val1) -> uint64_t {
        return (val0 <= val1) ? 1 : 0;
      });
}
// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vmsle8VX) {
  SetSemanticFunction(&Vmsle);
  BinaryMaskOpTestHelperVX<int8_t, int8_t>(
      "Vmsle8", /*sew*/ 8, instruction_,
      [](int8_t val0, int8_t val1) -> uint8_t {
        return (val0 <= val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsle16VX) {
  SetSemanticFunction(&Vmsle);
  BinaryMaskOpTestHelperVX<int16_t, int16_t>(
      "Vmsle16", /*sew*/ 16, instruction_,
      [](int16_t val0, int16_t val1) -> uint8_t {
        return (val0 <= val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsle32VX) {
  SetSemanticFunction(&Vmsle);
  BinaryMaskOpTestHelperVX<int32_t, int32_t>(
      "Vmsle32", /*sew*/ 32, instruction_,
      [](int32_t val0, int32_t val1) -> uint8_t {
        return (val0 <= val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsle64VX) {
  SetSemanticFunction(&Vmsle);
  BinaryMaskOpTestHelperVX<int64_t, int64_t>(
      "Vmsle64", /*sew*/ 64, instruction_,
      [](int64_t val0, int64_t val1) -> uint8_t {
        return (val0 <= val1) ? 1 : 0;
      });
}

// Vector mask unsigned set greater than.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vmsgtu8VX) {
  SetSemanticFunction(&Vmsgtu);
  BinaryMaskOpTestHelperVX<uint8_t, uint8_t>(
      "Vmsgtu8", /*sew*/ 8, instruction_,
      [](uint8_t val0, uint8_t val1) -> uint8_t {
        return (val0 > val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsgtu16VX) {
  SetSemanticFunction(&Vmsgtu);
  BinaryMaskOpTestHelperVX<uint16_t, uint16_t>(
      "Vmsgtu16", /*sew*/ 16, instruction_,
      [](uint16_t val0, uint16_t val1) -> uint8_t {
        return (val0 > val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsgtu32VX) {
  SetSemanticFunction(&Vmsgtu);
  BinaryMaskOpTestHelperVX<uint32_t, uint32_t>(
      "Vmsgtu32", /*sew*/ 32, instruction_,
      [](uint32_t val0, uint32_t val1) -> uint8_t {
        return (val0 > val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsgtu64VX) {
  SetSemanticFunction(&Vmsgtu);
  BinaryMaskOpTestHelperVX<uint64_t, uint64_t>(
      "Vmsgtuk64", /*sew*/ 64, instruction_,
      [](uint64_t val0, uint64_t val1) -> uint8_t {
        return (val0 > val1) ? 1 : 0;
      });
}

// Vector mask signed set greater than.
// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vmsgt8VX) {
  SetSemanticFunction(&Vmsgt);
  BinaryMaskOpTestHelperVX<int8_t, int8_t>(
      "Vmsgt8", /*sew*/ 8, instruction_,
      [](int8_t val0, int8_t val1) -> uint8_t {
        return (val0 > val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsgt16VX) {
  SetSemanticFunction(&Vmsgt);
  BinaryMaskOpTestHelperVX<int16_t, int16_t>(
      "Vmsgt16", /*sew*/ 16, instruction_,
      [](int16_t val0, int16_t val1) -> uint8_t {
        return (val0 > val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsgt32VX) {
  SetSemanticFunction(&Vmsgt);
  BinaryMaskOpTestHelperVX<int32_t, int32_t>(
      "Vmsgt32", /*sew*/ 32, instruction_,
      [](int32_t val0, int32_t val1) -> uint8_t {
        return (val0 > val1) ? 1 : 0;
      });
}
TEST_F(RiscVVectorInstructionsTest, Vmsgt64VX) {
  SetSemanticFunction(&Vmsgt);
  BinaryMaskOpTestHelperVX<int64_t, int64_t>(
      "Vmsgt64", /*sew*/ 64, instruction_,
      [](int64_t val0, int64_t val1) -> uint8_t {
        return (val0 > val1) ? 1 : 0;
      });
}

// Vector unsigned saturated add.
template <typename T>
T VsadduHelper(T val0, T val1) {
  T sum = val0 + val1;
  if (sum < val1) {
    sum = std::numeric_limits<T>::max();
  }
  return sum;
}
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vsaddu8VV) {
  SetSemanticFunction(&Vsaddu);
  BinaryOpTestHelperVV<uint8_t, uint8_t, uint8_t>(
      "Vsaddu8", /*sew*/ 8, instruction_, VsadduHelper<uint8_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vsaddu16VV) {
  SetSemanticFunction(&Vsaddu);
  BinaryOpTestHelperVV<uint16_t, uint16_t, uint16_t>(
      "Vsaddu16", /*sew*/ 16, instruction_, VsadduHelper<uint16_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vsaddu32VV) {
  SetSemanticFunction(&Vsaddu);
  BinaryOpTestHelperVV<uint32_t, uint32_t, uint32_t>(
      "Vsaddu32", /*sew*/ 32, instruction_, VsadduHelper<uint32_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vsaddu64VV) {
  SetSemanticFunction(&Vsaddu);
  BinaryOpTestHelperVV<uint64_t, uint64_t, uint64_t>(
      "Vsaddu64", /*sew*/ 64, instruction_, VsadduHelper<uint64_t>);
}

// Vector-Scalar
TEST_F(RiscVVectorInstructionsTest, Vsaddu8VX) {
  SetSemanticFunction(&Vsaddu);
  BinaryOpTestHelperVX<uint8_t, uint8_t, uint8_t>(
      "Vsaddu8", /*sew*/ 8, instruction_, VsadduHelper<uint8_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vsaddu16VX) {
  SetSemanticFunction(&Vsaddu);
  BinaryOpTestHelperVX<uint16_t, uint16_t, uint16_t>(
      "Vsaddu16", /*sew*/ 16, instruction_, VsadduHelper<uint16_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vsaddu32VX) {
  SetSemanticFunction(&Vsaddu);
  BinaryOpTestHelperVX<uint32_t, uint32_t, uint32_t>(
      "Vsaddu32", /*sew*/ 32, instruction_, VsadduHelper<uint32_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vsaddu64VX) {
  SetSemanticFunction(&Vsaddu);
  BinaryOpTestHelperVX<uint64_t, uint64_t, uint64_t>(
      "Vsaddu64", /*sew*/ 64, instruction_, VsadduHelper<uint64_t>);
}

// Vector signed saturated add.
template <typename T>
T VsaddHelper(T val0, T val1) {
  using WT = typename WideType<T>::type;
  WT wval0 = static_cast<WT>(val0);
  WT wval1 = static_cast<WT>(val1);
  WT wsum = wval0 + wval1;
  if (wsum > std::numeric_limits<T>::max()) {
    return std::numeric_limits<T>::max();
  }
  if (wsum < std::numeric_limits<T>::min()) {
    return std::numeric_limits<T>::min();
  }
  T sum = static_cast<T>(wsum);
  return sum;
}

// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vsadd8VV) {
  SetSemanticFunction(&Vsadd);
  BinaryOpTestHelperVV<int8_t, int8_t, int8_t>(
      "Vsadd8", /*sew*/ 8, instruction_, VsaddHelper<int8_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vsadd16VV) {
  SetSemanticFunction(&Vsadd);
  BinaryOpTestHelperVV<int16_t, int16_t, int16_t>(
      "Vsadd16", /*sew*/ 16, instruction_, VsaddHelper<int16_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vsadd32VV) {
  SetSemanticFunction(&Vsadd);
  BinaryOpTestHelperVV<int32_t, int32_t, int32_t>(
      "Vsadd32", /*sew*/ 32, instruction_, VsaddHelper<int32_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vsadd64VV) {
  SetSemanticFunction(&Vsadd);
  BinaryOpTestHelperVV<int64_t, int64_t, int64_t>(
      "Vsadd64", /*sew*/ 64, instruction_, VsaddHelper<int64_t>);
}

// Vector-Scalar
TEST_F(RiscVVectorInstructionsTest, Vsadd8VX) {
  SetSemanticFunction(&Vsadd);
  BinaryOpTestHelperVX<int8_t, int8_t, int8_t>(
      "Vsadd8", /*sew*/ 8, instruction_, VsaddHelper<int8_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vsadd16VX) {
  SetSemanticFunction(&Vsadd);
  BinaryOpTestHelperVX<int16_t, int16_t, int16_t>(
      "Vsadd16", /*sew*/ 16, instruction_, VsaddHelper<int16_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vsadd32VX) {
  SetSemanticFunction(&Vsadd);
  BinaryOpTestHelperVX<int32_t, int32_t, int32_t>(
      "Vsadd32", /*sew*/ 32, instruction_, VsaddHelper<int32_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vsadd64VX) {
  SetSemanticFunction(&Vsadd);
  BinaryOpTestHelperVX<int64_t, int64_t, int64_t>(
      "Vsadd64", /*sew*/ 64, instruction_, VsaddHelper<int64_t>);
}

// Vector unsigned saturated subtract.
// Vector-Vector.
template <typename T>
T SsubuHelper(T val0, T val1) {
  T diff = val0 - val1;
  if (val0 < val1) {
    diff = 0;
  }
  return diff;
}

TEST_F(RiscVVectorInstructionsTest, Vssubu8VV) {
  SetSemanticFunction(&Vssubu);
  BinaryOpTestHelperVV<uint8_t, uint8_t, uint8_t>(
      "Vssubu8", /*sew*/ 8, instruction_, SsubuHelper<uint8_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vssubu16VV) {
  SetSemanticFunction(&Vssubu);
  BinaryOpTestHelperVV<uint16_t, uint16_t, uint16_t>(
      "Vssubu16", /*sew*/ 16, instruction_, SsubuHelper<uint16_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vssubu32VV) {
  SetSemanticFunction(&Vssubu);
  BinaryOpTestHelperVV<uint32_t, uint32_t, uint32_t>(
      "Vssubu32", /*sew*/ 32, instruction_, SsubuHelper<uint32_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vssubu64VV) {
  SetSemanticFunction(&Vssubu);
  BinaryOpTestHelperVV<uint64_t, uint64_t, uint64_t>(
      "Vssubu64", /*sew*/ 64, instruction_, SsubuHelper<uint64_t>);
}

// Vector-Scalar
TEST_F(RiscVVectorInstructionsTest, Vssubu8VX) {
  SetSemanticFunction(&Vssubu);
  BinaryOpTestHelperVX<uint8_t, uint8_t, uint8_t>(
      "Vssubu8", /*sew*/ 8, instruction_, SsubuHelper<uint8_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vssubu16VX) {
  SetSemanticFunction(&Vssubu);
  BinaryOpTestHelperVX<uint16_t, uint16_t, uint16_t>(
      "Vssubu16", /*sew*/ 16, instruction_, SsubuHelper<uint16_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vssubu32VX) {
  SetSemanticFunction(&Vssubu);
  BinaryOpTestHelperVX<uint32_t, uint32_t, uint32_t>(
      "Vssubu32", /*sew*/ 32, instruction_, SsubuHelper<uint32_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vssubu64VX) {
  SetSemanticFunction(&Vssubu);
  BinaryOpTestHelperVX<uint64_t, uint64_t, uint64_t>(
      "Vssubu64", /*sew*/ 64, instruction_, SsubuHelper<uint64_t>);
}

// Vector signed saturated subtract.
template <typename T>
T VssubHelper(T val0, T val1) {
  using UT = typename MakeUnsigned<T>::type;
  UT uval0 = static_cast<UT>(val0);
  UT uval1 = static_cast<UT>(val1);
  UT udiff = uval0 - uval1;
  T diff = static_cast<T>(udiff);
  if (val0 < 0 && val1 >= 0 && diff >= 0) return std::numeric_limits<T>::min();
  if (val0 >= 0 && val1 < 0 && diff < 0) return std::numeric_limits<T>::max();
  return diff;
}
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vssub8VV) {
  SetSemanticFunction(&Vssub);
  BinaryOpTestHelperVV<int8_t, int8_t, int8_t>(
      "Vssub8", /*sew*/ 8, instruction_, VssubHelper<int8_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vssub16VV) {
  SetSemanticFunction(&Vssub);
  BinaryOpTestHelperVV<int16_t, int16_t, int16_t>(
      "Vssub16", /*sew*/ 16, instruction_, VssubHelper<int16_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vssub32VV) {
  SetSemanticFunction(&Vssub);
  BinaryOpTestHelperVV<int32_t, int32_t, int32_t>(
      "Vssub32", /*sew*/ 32, instruction_, VssubHelper<int32_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vssub64VV) {
  SetSemanticFunction(&Vssub);
  BinaryOpTestHelperVV<int64_t, int64_t, int64_t>(
      "Vssub64", /*sew*/ 64, instruction_, VssubHelper<int64_t>);
}

// Vector-Scalar
TEST_F(RiscVVectorInstructionsTest, Vssub8VX) {
  SetSemanticFunction(&Vssub);
  BinaryOpTestHelperVX<int8_t, int8_t, int8_t>(
      "Vssub8", /*sew*/ 8, instruction_, VssubHelper<int8_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vssub16VX) {
  SetSemanticFunction(&Vssub);
  BinaryOpTestHelperVX<int16_t, int16_t, int16_t>(
      "Vssub16", /*sew*/ 16, instruction_, VssubHelper<int16_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vssub32VX) {
  SetSemanticFunction(&Vssub);
  BinaryOpTestHelperVX<int32_t, int32_t, int32_t>(
      "Vssub32", /*sew*/ 32, instruction_, VssubHelper<int32_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vssub64VX) {
  SetSemanticFunction(&Vssub);
  BinaryOpTestHelperVX<int64_t, int64_t, int64_t>(
      "Vssub64", /*sew*/ 64, instruction_, VssubHelper<int64_t>);
}

template <typename T>
T VadcHelper(T vs2, T vs1, bool mask) {
  return vs2 + vs1 + mask;
}

// Vector add with carry.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vadc8VV) {
  SetSemanticFunction(&Vadc);
  BinaryOpWithMaskTestHelperVV<uint8_t, uint8_t, uint8_t>(
      "Vadc", /*sew*/ 8, instruction_, VadcHelper<uint8_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vadc16VV) {
  SetSemanticFunction(&Vadc);
  BinaryOpWithMaskTestHelperVV<uint16_t, uint16_t, uint16_t>(
      "Vadc", /*sew*/ 16, instruction_, VadcHelper<uint16_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vadc32VV) {
  SetSemanticFunction(&Vadc);
  BinaryOpWithMaskTestHelperVV<uint32_t, uint32_t, uint32_t>(
      "Vadc", /*sew*/ 32, instruction_, VadcHelper<uint32_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vadc64VV) {
  SetSemanticFunction(&Vadc);
  BinaryOpWithMaskTestHelperVV<uint64_t, uint64_t, uint64_t>(
      "Vadc", /*sew*/ 64, instruction_, VadcHelper<uint64_t>);
}
// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vadc8VX) {
  SetSemanticFunction(&Vadc);
  BinaryOpWithMaskTestHelperVX<uint8_t, uint8_t, uint8_t>(
      "Vadc", /*sew*/ 8, instruction_, VadcHelper<uint8_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vadc16VX) {
  SetSemanticFunction(&Vadc);
  BinaryOpWithMaskTestHelperVX<uint16_t, uint16_t, uint16_t>(
      "Vadc", /*sew*/ 16, instruction_, VadcHelper<uint16_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vadc32VX) {
  SetSemanticFunction(&Vadc);
  BinaryOpWithMaskTestHelperVX<uint32_t, uint32_t, uint32_t>(
      "Vadc", /*sew*/ 32, instruction_, VadcHelper<uint32_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vadc64VX) {
  SetSemanticFunction(&Vadc);
  BinaryOpWithMaskTestHelperVX<uint64_t, uint64_t, uint64_t>(
      "Vadc", /*sew*/ 64, instruction_, VadcHelper<uint64_t>);
}

template <typename T>
uint8_t VmadcHelper(T vs2, T vs1, bool mask_value) {
  T cin = ((vs2 & 0b1) + (vs1 & 0b1) + mask_value);
  cin >>= 1;
  vs2 >>= 1;
  vs1 >>= 1;
  T sum = vs2 + vs1 + cin;
  sum >>= sizeof(T) * 8 - 1;
  return sum;
}

// Vector compute carry from add with carry.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vmadc8VV) {
  SetSemanticFunction(&Vmadc);
  BinaryMaskOpWithMaskTestHelperVV<uint8_t, uint8_t>(
      "Vmadc", /*sew*/ 8, instruction_, VmadcHelper<uint8_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vmadc16VV) {
  SetSemanticFunction(&Vmadc);
  BinaryMaskOpWithMaskTestHelperVV<uint16_t, uint16_t>(
      "Vmadc", /*sew*/ 16, instruction_, VmadcHelper<uint16_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vmadc32VV) {
  SetSemanticFunction(&Vmadc);
  BinaryMaskOpWithMaskTestHelperVV<uint32_t, uint32_t>(
      "Vmadc", /*sew*/ 32, instruction_, VmadcHelper<uint32_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vmadc64VV) {
  SetSemanticFunction(&Vmadc);
  BinaryMaskOpWithMaskTestHelperVV<uint64_t, uint64_t>(
      "Vmadc", /*sew*/ 64, instruction_, VmadcHelper<uint64_t>);
}
// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vmadc8VX) {
  SetSemanticFunction(&Vmadc);
  BinaryMaskOpWithMaskTestHelperVX<uint8_t, uint8_t>(
      "Vmadc", /*sew*/ 8, instruction_, VmadcHelper<uint8_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vmadc16VX) {
  SetSemanticFunction(&Vmadc);
  BinaryMaskOpWithMaskTestHelperVX<uint16_t, uint16_t>(
      "Vmadc", /*sew*/ 16, instruction_, VmadcHelper<uint16_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vmadc32VX) {
  SetSemanticFunction(&Vmadc);
  BinaryMaskOpWithMaskTestHelperVX<uint32_t, uint32_t>(
      "Vmadc", /*sew*/ 32, instruction_, VmadcHelper<uint32_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vmadc64VX) {
  SetSemanticFunction(&Vmadc);
  BinaryMaskOpWithMaskTestHelperVX<uint64_t, uint64_t>(
      "Vmadc", /*sew*/ 64, instruction_, VmadcHelper<uint64_t>);
}

template <typename T>
T VsbcHelper(T vs2, T vs1, bool mask) {
  return vs2 - vs1 - mask;
}
// Vector subtract with borrow.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vsbc8VV) {
  SetSemanticFunction(&Vsbc);
  BinaryOpWithMaskTestHelperVV<uint8_t, uint8_t, uint8_t>(
      "Vsbc", /*sew*/ 8, instruction_, VsbcHelper<uint8_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vsbc16VV) {
  SetSemanticFunction(&Vsbc);
  BinaryOpWithMaskTestHelperVV<uint16_t, uint16_t, uint16_t>(
      "Vsbc", /*sew*/ 16, instruction_, VsbcHelper<uint16_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vsbc32VV) {
  SetSemanticFunction(&Vsbc);
  BinaryOpWithMaskTestHelperVV<uint32_t, uint32_t, uint32_t>(
      "Vsbc", /*sew*/ 32, instruction_, VsbcHelper<uint32_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vsbc64VV) {
  SetSemanticFunction(&Vsbc);
  BinaryOpWithMaskTestHelperVV<uint64_t, uint64_t, uint64_t>(
      "Vsbc", /*sew*/ 64, instruction_, VsbcHelper<uint64_t>);
}
// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vsbc8VX) {
  SetSemanticFunction(&Vsbc);
  BinaryOpWithMaskTestHelperVX<uint8_t, uint8_t, uint8_t>(
      "Vsbc", /*sew*/ 8, instruction_, VsbcHelper<uint8_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vsbc16VX) {
  SetSemanticFunction(&Vsbc);
  BinaryOpWithMaskTestHelperVX<uint16_t, uint16_t, uint16_t>(
      "Vsbc", /*sew*/ 16, instruction_, VsbcHelper<uint16_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vsbc32VX) {
  SetSemanticFunction(&Vsbc);
  BinaryOpWithMaskTestHelperVX<uint32_t, uint32_t, uint32_t>(
      "Vsbc", /*sew*/ 32, instruction_, VsbcHelper<uint32_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vsbc64VX) {
  SetSemanticFunction(&Vsbc);
  BinaryOpWithMaskTestHelperVX<uint64_t, uint64_t, uint64_t>(
      "Vsbc", /*sew*/ 64, instruction_, VsbcHelper<uint64_t>);
}

template <typename T>
uint8_t VmsbcHelper(T vs2, T vs1, bool mask_value) {
  if (vs2 == vs1) return mask_value;
  if (vs2 < vs1) return 1;
  return 0;
}

// Vector compute carry from subtract with borrow.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vmsbc8VV) {
  SetSemanticFunction(&Vmsbc);
  BinaryMaskOpWithMaskTestHelperVV<uint8_t, uint8_t>(
      "Vmsbc", /*sew*/ 8, instruction_, VmsbcHelper<uint8_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vmsbc16VV) {
  SetSemanticFunction(&Vmsbc);
  BinaryMaskOpWithMaskTestHelperVV<uint16_t, uint16_t>(
      "Vmsbc", /*sew*/ 16, instruction_, VmsbcHelper<uint16_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vmsbc32VV) {
  SetSemanticFunction(&Vmsbc);
  BinaryMaskOpWithMaskTestHelperVV<uint32_t, uint32_t>(
      "Vmsbc", /*sew*/ 32, instruction_, VmsbcHelper<uint32_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vmsbc64VV) {
  SetSemanticFunction(&Vmsbc);
  BinaryMaskOpWithMaskTestHelperVV<uint64_t, uint64_t>(
      "Vmsbc", /*sew*/ 64, instruction_, VmsbcHelper<uint64_t>);
}
// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vmsbc8VX) {
  SetSemanticFunction(&Vmsbc);
  BinaryMaskOpWithMaskTestHelperVX<uint8_t, uint8_t>(
      "Vmsbc", /*sew*/ 8, instruction_, VmsbcHelper<uint8_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vmsbc16VX) {
  SetSemanticFunction(&Vmsbc);
  BinaryMaskOpWithMaskTestHelperVX<uint16_t, uint16_t>(
      "Vmsbc", /*sew*/ 16, instruction_, VmsbcHelper<uint16_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vmsbc32VX) {
  SetSemanticFunction(&Vmsbc);
  BinaryMaskOpWithMaskTestHelperVX<uint32_t, uint32_t>(
      "Vmsbc", /*sew*/ 32, instruction_, VmsbcHelper<uint32_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vmsbc64VX) {
  SetSemanticFunction(&Vmsbc);
  BinaryMaskOpWithMaskTestHelperVX<uint64_t, uint64_t>(
      "Vmsbc", /*sew*/ 64, instruction_, VmsbcHelper<uint64_t>);
}

// Vector merge.
template <typename T>
T VmergeHelper(T vs2, T vs1, bool mask_value) {
  return mask_value ? vs1 : vs2;
}
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vmerge8VV) {
  SetSemanticFunction(&Vmerge);
  BinaryOpWithMaskTestHelperVV<uint8_t, uint8_t, uint8_t>(
      "Vmerge", /*sew*/ 8, instruction_, VmergeHelper<uint8_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vmerge16VV) {
  SetSemanticFunction(&Vmerge);
  BinaryOpWithMaskTestHelperVV<uint16_t, uint16_t, uint16_t>(
      "Vmerge", /*sew*/ 16, instruction_, VmergeHelper<uint16_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vmerge32VV) {
  SetSemanticFunction(&Vmerge);
  BinaryOpWithMaskTestHelperVV<uint32_t, uint32_t, uint32_t>(
      "mergec", /*sew*/ 32, instruction_, VmergeHelper<uint32_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vmerge64VV) {
  SetSemanticFunction(&Vmerge);
  BinaryOpWithMaskTestHelperVV<uint64_t, uint64_t, uint64_t>(
      "Vmerge", /*sew*/ 64, instruction_, VmergeHelper<uint64_t>);
}
// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vmerge8VX) {
  SetSemanticFunction(&Vmerge);
  BinaryOpWithMaskTestHelperVX<uint8_t, uint8_t, uint8_t>(
      "Vmerge", /*sew*/ 8, instruction_, VmergeHelper<uint8_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vmerge16VX) {
  SetSemanticFunction(&Vmerge);
  BinaryOpWithMaskTestHelperVX<uint16_t, uint16_t, uint16_t>(
      "Vmerge", /*sew*/ 16, instruction_, VmergeHelper<uint16_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vmerge32VX) {
  SetSemanticFunction(&Vmerge);
  BinaryOpWithMaskTestHelperVX<uint32_t, uint32_t, uint32_t>(
      "mergec", /*sew*/ 32, instruction_, VmergeHelper<uint32_t>);
}
TEST_F(RiscVVectorInstructionsTest, Vmerge64VX) {
  SetSemanticFunction(&Vmerge);
  BinaryOpWithMaskTestHelperVX<uint64_t, uint64_t, uint64_t>(
      "Vmerge", /*sew*/ 64, instruction_, VmergeHelper<uint64_t>);
}

// This wrapper function factors out the main body of the Vmvr test.
void VmvrWrapper(int num_reg, RiscVVectorInstructionsTest *tester,
                 Instruction *inst) {
  tester->SetSemanticFunction(absl::bind_front(&Vmvr, num_reg));
  // Number of elements per vector register.
  constexpr int vs2_size = kVectorLengthInBytes / sizeof(uint64_t);
  // Input values for 8 registers.
  uint64_t vs2_value[vs2_size * 8];
  auto vs2_span = Span<uint64_t>(vs2_value);
  tester->AppendVectorRegisterOperands({kVs2}, {kVd});
  // Initialize input values.
  tester->FillArrayWithRandomValues<uint64_t>(vs2_span);
  for (int i = 0; i < 8; i++) {
    auto vs2_name = absl::StrCat("v", kVs2 + i);
    tester->SetVectorRegisterValues<uint64_t>(
        {{vs2_name, vs2_span.subspan(vs2_size * i, vs2_size)}});
  }
  tester->ClearVectorRegisterGroup(kVd, 8);
  inst->Execute();
  EXPECT_FALSE(tester->rv_vector()->vector_exception());
  int count = 0;
  for (int reg = kVd; reg < kVd + 8; reg++) {
    auto dest_span = tester->vreg()[reg]->data_buffer()->Get<uint64_t>();
    for (int i = 0; i < kVectorLengthInBytes / sizeof(uint64_t); i++) {
      if (reg < kVd + num_reg) {
        EXPECT_EQ(vs2_span[count], dest_span[i])
            << "count: " << count << "  i: " << i;
      } else {
        EXPECT_EQ(0, dest_span[i]) << "count: " << count << "  i: " << i;
      }
      count++;
    }
  }
}

// Vector move register.
TEST_F(RiscVVectorInstructionsTest, Vmvr1) {
  VmvrWrapper(1, this, instruction_);
}

TEST_F(RiscVVectorInstructionsTest, Vmvr2) {
  VmvrWrapper(2, this, instruction_);
}
TEST_F(RiscVVectorInstructionsTest, Vmvr4) {
  VmvrWrapper(4, this, instruction_);
}
TEST_F(RiscVVectorInstructionsTest, Vmvr8) {
  VmvrWrapper(8, this, instruction_);
}

// Templated helper functions for Vssr testing.
template <typename T>
T VssrHelper(RiscVVectorInstructionsTest *tester, T vs2, T vs1,
             int rounding_mode) {
  using UT = typename MakeUnsigned<T>::type;
  int max_shift = (sizeof(T) << 3) - 1;
  int shift_amount = static_cast<int>(vs1 & max_shift);
  // Extract the bits that will be lost + 1.
  UT lost_bits = vs2;
  if (shift_amount < max_shift) {
    lost_bits = vs2 & ~(std::numeric_limits<UT>::max() << (shift_amount + 1));
  }
  T result = vs2 >> shift_amount;
  result += static_cast<T>(tester->RoundBits(shift_amount + 1, lost_bits));
  return result;
}

// These wrapper functions simplify the test bodies, and make it a little
// easier to avoid errors due to type and sew specifications.
template <typename T>
void VssrVVWrapper(absl::string_view base_name, Instruction *inst,
                   RiscVVectorInstructionsTest *tester) {
  // Iterate across rounding modes.
  for (int rm = 0; rm < 4; rm++) {
    tester->rv_vector()->set_vxrm(rm);
    tester->BinaryOpTestHelperVV<T, T, T>(
        absl::StrCat("Vssrl_", rm), /*sew*/ sizeof(T) * 8, inst,
        [rm, tester](T vs2, T vs1) -> T {
          return VssrHelper<T>(tester, vs2, vs1, rm);
        });
  }
}
template <typename T>
void VssrVXWrapper(absl::string_view base_name, Instruction *inst,
                   RiscVVectorInstructionsTest *tester) {
  // Iterate across rounding modes.
  for (int rm = 0; rm < 4; rm++) {
    tester->rv_vector()->set_vxrm(rm);
    tester->BinaryOpTestHelperVX<T, T, T>(
        absl::StrCat("Vssrl_", rm), /*sew*/ sizeof(T) * 8, inst,
        [rm, tester](T vs2, T vs1) -> T {
          return VssrHelper<T>(tester, vs2, vs1, rm);
        });
  }
}
// Vector shift right logical with rounding.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vssrl8VV) {
  SetSemanticFunction(&Vssrl);
  VssrVVWrapper<uint8_t>("Vssrl", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vssrl16VV) {
  SetSemanticFunction(&Vssrl);
  VssrVVWrapper<uint16_t>("Vssrl", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vssrl32VV) {
  SetSemanticFunction(&Vssrl);
  VssrVVWrapper<uint32_t>("Vssrl", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vssrl64VV) {
  SetSemanticFunction(&Vssrl);
  VssrVVWrapper<uint64_t>("Vssrl", instruction_, this);
}
// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vssrl8VX) {
  SetSemanticFunction(&Vssrl);
  VssrVXWrapper<uint8_t>("Vssrl", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vssrl16VX) {
  SetSemanticFunction(&Vssrl);
  VssrVXWrapper<uint16_t>("Vssrl", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vssrl32VX) {
  SetSemanticFunction(&Vssrl);
  VssrVXWrapper<uint32_t>("Vssrl", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vssrl64VX) {
  SetSemanticFunction(&Vssrl);
  VssrVXWrapper<uint64_t>("Vssrl", instruction_, this);
}

// Vector shift right arithmetic with rounding.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vssra8VV) {
  SetSemanticFunction(&Vssra);
  VssrVVWrapper<int8_t>("Vssra", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vssra16VV) {
  SetSemanticFunction(&Vssra);
  VssrVVWrapper<int16_t>("Vssal", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vssra32VV) {
  SetSemanticFunction(&Vssra);
  VssrVVWrapper<int32_t>("Vssal", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vssra64VV) {
  SetSemanticFunction(&Vssra);
  VssrVVWrapper<int64_t>("Vssal", instruction_, this);
}
// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vssra8VX) {
  SetSemanticFunction(&Vssra);
  VssrVXWrapper<int8_t>("Vssra", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vssra16VX) {
  SetSemanticFunction(&Vssra);
  VssrVXWrapper<int16_t>("Vssal", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vssra32VX) {
  SetSemanticFunction(&Vssra);
  VssrVXWrapper<int32_t>("Vssal", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vssra64VX) {
  SetSemanticFunction(&Vssra);
  VssrVXWrapper<int64_t>("Vssal", instruction_, this);
}

// Templated helper functions for Vnclip/Vnclipu instructions.
template <typename T, typename WideT>
T VnclipHelper(RiscVVectorInstructionsTest *tester, WideT vs2, T vs1, int rm,
               RiscVVectorState *rv_vector) {
  auto vs1_16 = static_cast<WideT>(vs1);
  auto shifted = VssrHelper<WideT>(tester, vs2, vs1_16, rm);
  if (shifted < std::numeric_limits<T>::min()) {
    rv_vector->set_vxsat(true);
    return std::numeric_limits<T>::min();
  }
  if (shifted > std::numeric_limits<T>::max()) {
    rv_vector->set_vxsat(true);
    return std::numeric_limits<T>::max();
  }
  return static_cast<T>(shifted);
}

template <typename T>
void VnclipVVWrapper(absl::string_view base_name, Instruction *inst,
                     RiscVVectorInstructionsTest *tester) {
  using WT = typename WideType<T>::type;
  for (int rm = 0; rm < 4; rm++) {
    tester->rv_vector()->set_vxrm(rm);
    tester->BinaryOpTestHelperVV<T, WT, T>(
        absl::StrCat(base_name, "_", rm), sizeof(T) * 8, inst,
        [rm, tester](WT vs2, T vs1) -> T {
          return VnclipHelper<T, WT>(tester, vs2, vs1, rm, tester->rv_vector());
        });
  }
}
template <typename T>
void VnclipVXWrapper(absl::string_view base_name, Instruction *inst,
                     RiscVVectorInstructionsTest *tester) {
  using WT = typename WideType<T>::type;
  for (int rm = 0; rm < 4; rm++) {
    tester->rv_vector()->set_vxrm(rm);
    tester->BinaryOpTestHelperVV<T, WT, T>(
        absl::StrCat(base_name, "_", rm), sizeof(T) * 8, inst,
        [rm, tester](WT vs2, T vs1) -> T {
          return VnclipHelper<T, WT>(tester, vs2, vs1, rm, tester->rv_vector());
        });
  }
}
// Vector shift right logical with rounding and saturation.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vnclip8VV) {
  SetSemanticFunction(&Vnclip);
  VnclipVVWrapper<int8_t>("Vnclip", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vnclip16VV) {
  SetSemanticFunction(&Vnclip);
  VnclipVVWrapper<int16_t>("Vnclip", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vnclip32VV) {
  SetSemanticFunction(&Vnclip);
  VnclipVVWrapper<int32_t>("Vnclip", instruction_, this);
}
// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vnclip8VX) {
  SetSemanticFunction(&Vnclip);
  VnclipVXWrapper<int8_t>("Vnclip", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vnclip16VX) {
  SetSemanticFunction(&Vnclip);
  VnclipVXWrapper<int16_t>("Vnclip", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vnclip32VX) {
  SetSemanticFunction(&Vnclip);
  VnclipVXWrapper<int32_t>("Vnclip", instruction_, this);
}

// Vector shift right arithmetic with rounding and saturation.
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vnclipu8VV) {
  SetSemanticFunction(&Vnclipu);
  VnclipVVWrapper<uint8_t>("Vnclipu", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vnclipu16VV) {
  SetSemanticFunction(&Vnclipu);
  VnclipVVWrapper<uint16_t>("Vnclipu", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vnclipu32VV) {
  SetSemanticFunction(&Vnclipu);
  SetSemanticFunction(&Vnclipu);
  VnclipVVWrapper<uint32_t>("Vnclipu", instruction_, this);
}
// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vnclipu8VX) {
  SetSemanticFunction(&Vnclipu);
  VnclipVXWrapper<uint8_t>("Vnclipu", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vnclipu16VX) {
  SetSemanticFunction(&Vnclipu);
  VnclipVXWrapper<uint16_t>("Vnclipu", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vnclipu32VX) {
  SetSemanticFunction(&Vnclipu);
  VnclipVXWrapper<uint32_t>("Vnclipu", instruction_, this);
}

// Vector fractional multiply with rounding and saturation.
template <typename T>
T VsmulHelper(RiscVVectorInstructionsTest *tester, T vs2, T vs1, int rm,
              RiscVVectorState *rv_vector) {
  using WT = typename WideType<T>::type;
  WT vs2_w = static_cast<WT>(vs2);
  WT vs1_w = static_cast<WT>(vs1);
  WT prod = vs2_w * vs1_w;
  WT res = VssrHelper<WT>(tester, prod, sizeof(T) * 8 - 1, rm);
  if (res > std::numeric_limits<T>::max()) {
    return std::numeric_limits<T>::max();
  }
  if (res < std::numeric_limits<T>::min()) {
    return std::numeric_limits<T>::min();
  }
  return static_cast<T>(res);
}

template <typename T>
void VsmulVVWrapper(absl::string_view base_name, Instruction *inst,
                    RiscVVectorInstructionsTest *tester) {
  for (int rm = 0; rm < 4; rm++) {
    tester->rv_vector()->set_vxrm(rm);
    tester->BinaryOpTestHelperVV<T, T, T>(
        absl::StrCat(base_name, "_", rm), sizeof(T) * 8, inst,
        [rm, tester](T vs2, T vs1) -> T {
          return VsmulHelper<T>(tester, vs2, vs1, rm, tester->rv_vector());
        });
  }
}
template <typename T>
void VsmulVXWrapper(absl::string_view base_name, Instruction *inst,
                    RiscVVectorInstructionsTest *tester) {
  for (int rm = 0; rm < 4; rm++) {
    tester->rv_vector()->set_vxrm(rm);
    tester->BinaryOpTestHelperVV<T, T, T>(
        absl::StrCat(base_name, "_", rm), sizeof(T) * 8, inst,
        [rm, tester](T vs2, T vs1) -> T {
          return VsmulHelper<T>(tester, vs2, vs1, rm, tester->rv_vector());
        });
  }
}
// Vector-Vector.
TEST_F(RiscVVectorInstructionsTest, Vsmpy8VV) {
  SetSemanticFunction(&Vsmul);
  VsmulVVWrapper<int8_t>("Vsmul", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vsmpy16VV) {
  SetSemanticFunction(&Vsmul);
  VsmulVVWrapper<int16_t>("Vsuly", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vsmpy32VV) {
  SetSemanticFunction(&Vsmul);
  VsmulVVWrapper<int32_t>("Vsuly", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vsmpy64VV) {
  SetSemanticFunction(&Vsmul);
  VsmulVVWrapper<int64_t>("Vsuly", instruction_, this);
}
// Vector-Scalar.
TEST_F(RiscVVectorInstructionsTest, Vsmpy8VX) {
  SetSemanticFunction(&Vsmul);
  VsmulVXWrapper<int8_t>("Vsmul", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vsmpy16VX) {
  SetSemanticFunction(&Vsmul);
  VsmulVXWrapper<int16_t>("Vsuly", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vsmpy32VX) {
  SetSemanticFunction(&Vsmul);
  VsmulVXWrapper<int32_t>("Vsuly", instruction_, this);
}
TEST_F(RiscVVectorInstructionsTest, Vsmpy64VX) {
  SetSemanticFunction(&Vsmul);
  VsmulVXWrapper<int64_t>("Vsuly", instruction_, this);
}
}  // namespace
