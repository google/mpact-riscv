// Copyright 2025 Google LLC
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

#include <sys/types.h>

#include <cstdint>

#include "absl/strings/str_cat.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_vector_basic_bit_manipulation_instructions.h"
#include "riscv/test/riscv_vector_instructions_test_base.h"

// This file contains tests for the RiscV vector basic bit manipulations.

namespace {

using ::mpact::sim::generic::WideType;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RVVectorRegister;
using ::mpact::sim::riscv::Vandn;
using ::mpact::sim::riscv::Vbrev;
using ::mpact::sim::riscv::Vbrev8;
using ::mpact::sim::riscv::Vclz;
using ::mpact::sim::riscv::Vctz;
using ::mpact::sim::riscv::VectorVcpop;
using ::mpact::sim::riscv::Vrev8;
using ::mpact::sim::riscv::Vrol;
using ::mpact::sim::riscv::Vror;
using ::mpact::sim::riscv::Vwsll;
using ::mpact::sim::riscv::test::RiscVVectorInstructionsTestBase;

class RiscVVectorBasicBitManipulationTest
    : public RiscVVectorInstructionsTestBase {};

// Helper function for testing the vandn_vv instruction. Generate the expected
// result using the bitwise operator.
template <typename T>
inline void VandnVVHelper(RiscVVectorBasicBitManipulationTest *tester) {
  tester->SetSemanticFunction(&Vandn);
  tester->BinaryOpTestHelperVV<T, T, T>(
      absl::StrCat("Vandn", sizeof(T) * 8, "vv"), /*sew*/ sizeof(T) * 8,
      tester->instruction(), [](T vs2, T vs1) -> T { return ~vs1 & vs2; });
}

// Helper function for testing the vandn_vx instruction. Generate the expected
// result using the bitwise operator.
template <typename T>
inline void VandnVXHelper(RiscVVectorBasicBitManipulationTest *tester) {
  tester->SetSemanticFunction(&Vandn);
  tester->BinaryOpTestHelperVX<T, T, T>(
      absl::StrCat("Vandn", sizeof(T) * 8, "vx"), /*sew*/ sizeof(T) * 8,
      tester->instruction(), [](T vs2, T rs1) -> T { return ~rs1 & vs2; });
}

// Helper function for testing the vbrev_v instruction. Generate the expected
// result by reversing the input bits.
template <typename T>
inline void VbrevVHelper(RiscVVectorBasicBitManipulationTest *tester) {
  tester->SetSemanticFunction(&Vbrev);
  tester->UnaryOpTestHelperV<T, T>(absl::StrCat("Vbrev", sizeof(T) * 8, "v"),
                                   /*sew*/ sizeof(T) * 8, tester->instruction(),
                                   [](T vs2) -> T {
                                     T result = 0;
                                     for (int i = 0; i < sizeof(T) * 8; ++i) {
                                       result = (result << 1) | (vs2 & 1);
                                       vs2 >>= 1;
                                     }
                                     return result;
                                   });
}

// Helper function for testing the vbrev8_v instruction. Generate the expected
// result by reversing the bits in each of the input bytes.
template <typename T>
inline void Vbrev8VHelper(RiscVVectorBasicBitManipulationTest *tester) {
  tester->SetSemanticFunction(&Vbrev8);
  tester->UnaryOpTestHelperV<T, T>(
      absl::StrCat("Vbrev8", sizeof(T) * 8, "v"), /*sew*/ sizeof(T) * 8,
      tester->instruction(), [](T vs2) -> T {
        T result = 0;
        for (int offset = 0; offset < sizeof(T) * 8; offset += 8) {
          uint8_t byte = (vs2 >> offset) & 0xFF;
          T reversed_byte = 0;
          for (int j = 0; j < 8; ++j) {
            reversed_byte = (reversed_byte << 1) | (byte & 1);
            byte >>= 1;
          }
          result |= reversed_byte << offset;
        }
        return result;
      });
}

// Helper function for testing the vbrev_v instruction. Generate the expected
// result by reversing the bytes of the input.
template <typename T>
inline void Vrev8VHelper(RiscVVectorBasicBitManipulationTest *tester) {
  tester->SetSemanticFunction(&Vrev8);
  tester->UnaryOpTestHelperV<T, T>(
      absl::StrCat("Vrev8", sizeof(T) * 8, "v"), /*sew*/ sizeof(T) * 8,
      tester->instruction(), [](T vs2) -> T {
        T result = 0;
        for (int offset = 0; offset < sizeof(T) * 8; offset += 8) {
          uint8_t byte = (vs2 >> offset) & 0xff;
          result = (result << 8) | byte;
        }
        return result;
      });
}

// Helper function for testing the vrol_vv instruction. Generate the expected
// result by rotating the input bits left.
template <typename T>
inline void VrolVVHelper(RiscVVectorBasicBitManipulationTest *tester) {
  tester->SetSemanticFunction(&Vrol);
  tester->BinaryOpTestHelperVV<T, T, T>(
      absl::StrCat("Vrol", sizeof(T) * 8, "vv"), /*sew*/ sizeof(T) * 8,
      tester->instruction(), [](T vs2, T vs1) -> T {
        T rotate_mask = sizeof(T) * 8 - 1;
        T rotate_amount = (vs1 & rotate_mask);
        return (vs2 << rotate_amount) |
               (vs2 >> (sizeof(T) * 8 - rotate_amount));
      });
}

// Helper function for testing the vrol_vx instruction. Generate the expected
// result by rotating the input bits left.
template <typename T>
inline void VrolVXHelper(RiscVVectorBasicBitManipulationTest *tester) {
  tester->SetSemanticFunction(&Vrol);
  tester->BinaryOpTestHelperVX<T, T, T>(
      absl::StrCat("Vrol", sizeof(T) * 8, "vx"), /*sew*/ sizeof(T) * 8,
      tester->instruction(), [](T vs2, T rs1) -> T {
        T rotate_mask = sizeof(T) * 8 - 1;
        T rotate_amount = (rs1 & rotate_mask);
        return (vs2 << rotate_amount) |
               (vs2 >> (sizeof(T) * 8 - rotate_amount));
      });
}

// Helper function for testing the vror_vv instruction. Generate the expected
// result by rotating the input bits right.
template <typename T>
inline void VrorVVHelper(RiscVVectorBasicBitManipulationTest *tester) {
  tester->SetSemanticFunction(&Vror);
  tester->BinaryOpTestHelperVV<T, T, T>(
      absl::StrCat("Vror", sizeof(T) * 8, "vv"), /*sew*/ sizeof(T) * 8,
      tester->instruction(), [](T vs2, T vs1) -> T {
        T rotate_mask = sizeof(T) * 8 - 1;
        T rotate_amount = (vs1 & rotate_mask);
        return (vs2 >> rotate_amount) |
               (vs2 << (sizeof(T) * 8 - rotate_amount));
      });
}

// Helper function for testing the vror_vx instruction. Generate the expected
// result by rotating the input bits right.
template <typename T>
inline void VrorVXHelper(RiscVVectorBasicBitManipulationTest *tester) {
  tester->SetSemanticFunction(&Vror);
  tester->BinaryOpTestHelperVV<T, T, T>(
      absl::StrCat("Vror", sizeof(T) * 8, "vx"), /*sew*/ sizeof(T) * 8,
      tester->instruction(), [](T vs2, T rs1) -> T {
        T rotate_mask = sizeof(T) * 8 - 1;
        T rotate_amount = (rs1 & rotate_mask);
        return (vs2 >> rotate_amount) |
               (vs2 << (sizeof(T) * 8 - rotate_amount));
      });
}

// Helper function for testing the vror_vi instruction. Generate the expected
// result by rotating the input bits right.
template <typename T>
inline void VrorVIHelper(RiscVVectorBasicBitManipulationTest *tester) {
  tester->SetSemanticFunction(&Vror);
  tester->BinaryOpTestHelperVV<T, T, T>(
      absl::StrCat("Vror", sizeof(T) * 8, "vi"), /*sew*/ sizeof(T) * 8,
      tester->instruction(), [](T vs2, T imm) -> T {
        T rotate_mask = sizeof(T) * 8 - 1;
        T rotate_amount = (imm & rotate_mask);
        return (vs2 >> rotate_amount) |
               (vs2 << (sizeof(T) * 8 - rotate_amount));
      });
}

// Helper function for testing the vclz_v instruction. Generate the expected
// result by counting the number of leading zeros in the input.
template <typename T>
inline void VclzVHelper(RiscVVectorBasicBitManipulationTest *tester) {
  tester->SetSemanticFunction(&Vclz);
  tester->UnaryOpTestHelperV<T, T>(
      absl::StrCat("vclz", sizeof(T) * 8, "v"), /*sew*/ sizeof(T) * 8,
      tester->instruction(), [](T vs2) -> T {
        T mask = static_cast<T>(1) << (sizeof(T) * 8 - 1);
        for (int i = 0; i < sizeof(T) * 8; ++i) {
          if ((vs2 & mask) != 0) {
            return i;
          }
          mask >>= 1;
        }
        return static_cast<T>(sizeof(T) * 8);
      });
}

// Helper function for testing the vctz_v instruction. Generate the expected
// result by counting the number of trailing zeros in the input.
template <typename T>
inline void VctzVHelper(RiscVVectorBasicBitManipulationTest *tester) {
  tester->SetSemanticFunction(&Vctz);
  tester->UnaryOpTestHelperV<T, T>(absl::StrCat("vctz", sizeof(T) * 8, "v"),
                                   /*sew*/ sizeof(T) * 8, tester->instruction(),
                                   [](T vs2) -> T {
                                     T mask = static_cast<T>(1);
                                     for (int i = 0; i < sizeof(T) * 8; ++i) {
                                       if ((vs2 & mask) != 0) {
                                         return i;
                                       }
                                       mask <<= 1;
                                     }
                                     return static_cast<T>(sizeof(T) * 8);
                                   });
}

// Helper function for testing the vcpop_v instruction. Generate the expected
// result by counting the number of bits set in the input.
template <typename T>
inline void VcpopVHelper(RiscVVectorBasicBitManipulationTest *tester) {
  tester->SetSemanticFunction(&VectorVcpop);
  tester->UnaryOpTestHelperV<T, T>(absl::StrCat("vcpop", sizeof(T) * 8, "v"),
                                   /*sew*/ sizeof(T) * 8, tester->instruction(),
                                   [](T vs2) -> T {
                                     T result = 0;
                                     for (int i = 0; i < sizeof(T) * 8; ++i) {
                                       result += (vs2 & 1) ? 1 : 0;
                                       vs2 >>= 1;
                                     }
                                     return result;
                                   });
}

// Helper function for testing the vwsll_vv instruction. Generate the expected
// result by shifting the widened input left.
template <typename T>
inline void VwsllVVHelper(RiscVVectorBasicBitManipulationTest *tester) {
  using WT = typename WideType<T>::type;
  tester->SetSemanticFunction(&Vwsll);
  tester->BinaryOpTestHelperVV<WT, T, T>(
      absl::StrCat("Vwsll", sizeof(T) * 8, "vv"), /*sew*/ sizeof(T) * 8,
      tester->instruction(), [](T vs2, T vs1) -> WT {
        T shift_mask = 2 * 8 * sizeof(T) - 1;
        T shift_amount = (vs1 & shift_mask);
        return static_cast<WT>(vs2) << shift_amount;
      });
}

// Helper function for testing the vwsll_vx instruction. Generate the expected
// result by shifting the widened input left.
template <typename T>
inline void VwsllVXHelper(RiscVVectorBasicBitManipulationTest *tester) {
  using WT = typename WideType<T>::type;
  tester->SetSemanticFunction(&Vwsll);
  tester->BinaryOpTestHelperVV<WT, T, T>(
      absl::StrCat("Vwsll", sizeof(T) * 8, "vx"), /*sew*/ sizeof(T) * 8,
      tester->instruction(), [](T vs2, T rs1) -> WT {
        T shift_mask = 2 * 8 * sizeof(T) - 1;
        T shift_amount = (rs1 & shift_mask);
        return static_cast<WT>(vs2) << shift_amount;
      });
}

// Helper function for testing the vwsll_vi instruction. Generate the expected
// result by shifting the widened input left.
template <typename T>
inline void VwsllVIHelper(RiscVVectorBasicBitManipulationTest *tester) {
  using WT = typename WideType<T>::type;
  tester->SetSemanticFunction(&Vwsll);
  tester->BinaryOpTestHelperVV<WT, T, T>(
      absl::StrCat("Vwsll", sizeof(T) * 8, "vi"), /*sew*/ sizeof(T) * 8,
      tester->instruction(), [](T vs2, T imm) -> WT {
        T shift_mask = 2 * 8 * sizeof(T) - 1;
        T shift_amount = (imm & shift_mask);
        return static_cast<WT>(vs2) << shift_amount;
      });
}

TEST_F(RiscVVectorBasicBitManipulationTest, vandn) {
  VandnVVHelper<uint8_t>(this);
  ResetInstruction();
  VandnVVHelper<uint16_t>(this);
  ResetInstruction();
  VandnVVHelper<uint32_t>(this);
  ResetInstruction();
  VandnVVHelper<uint64_t>(this);
  ResetInstruction();

  VandnVXHelper<uint8_t>(this);
  ResetInstruction();
  VandnVXHelper<uint16_t>(this);
  ResetInstruction();
  VandnVXHelper<uint32_t>(this);
  ResetInstruction();
  VandnVXHelper<uint64_t>(this);
  ResetInstruction();
}

TEST_F(RiscVVectorBasicBitManipulationTest, vbrev8) {
  Vbrev8VHelper<uint8_t>(this);
  ResetInstruction();
  Vbrev8VHelper<uint16_t>(this);
  ResetInstruction();
  Vbrev8VHelper<uint32_t>(this);
  ResetInstruction();
  Vbrev8VHelper<uint64_t>(this);
  ResetInstruction();
}

TEST_F(RiscVVectorBasicBitManipulationTest, vrev8) {
  Vrev8VHelper<uint8_t>(this);
  ResetInstruction();
  Vrev8VHelper<uint16_t>(this);
  ResetInstruction();
  Vrev8VHelper<uint32_t>(this);
  ResetInstruction();
  Vrev8VHelper<uint64_t>(this);
  ResetInstruction();
}

TEST_F(RiscVVectorBasicBitManipulationTest, vrol) {
  VrolVVHelper<uint8_t>(this);
  ResetInstruction();
  VrolVVHelper<uint16_t>(this);
  ResetInstruction();
  VrolVVHelper<uint32_t>(this);
  ResetInstruction();
  VrolVVHelper<uint64_t>(this);
  ResetInstruction();

  VrolVXHelper<uint8_t>(this);
  ResetInstruction();
  VrolVXHelper<uint16_t>(this);
  ResetInstruction();
  VrolVXHelper<uint32_t>(this);
  ResetInstruction();
  VrolVXHelper<uint64_t>(this);
  ResetInstruction();
}

TEST_F(RiscVVectorBasicBitManipulationTest, vror) {
  VrorVVHelper<uint8_t>(this);
  ResetInstruction();
  VrorVVHelper<uint16_t>(this);
  ResetInstruction();
  VrorVVHelper<uint32_t>(this);
  ResetInstruction();
  VrorVVHelper<uint64_t>(this);
  ResetInstruction();

  VrorVXHelper<uint8_t>(this);
  ResetInstruction();
  VrorVXHelper<uint16_t>(this);
  ResetInstruction();
  VrorVXHelper<uint32_t>(this);
  ResetInstruction();
  VrorVXHelper<uint64_t>(this);
  ResetInstruction();

  VrorVIHelper<uint8_t>(this);
  ResetInstruction();
  VrorVIHelper<uint16_t>(this);
  ResetInstruction();
  VrorVIHelper<uint32_t>(this);
  ResetInstruction();
  VrorVIHelper<uint64_t>(this);
  ResetInstruction();
}

TEST_F(RiscVVectorBasicBitManipulationTest, vbrev) {
  VbrevVHelper<uint8_t>(this);
  ResetInstruction();
  VbrevVHelper<uint16_t>(this);
  ResetInstruction();
  VbrevVHelper<uint32_t>(this);
  ResetInstruction();
  VbrevVHelper<uint64_t>(this);
  ResetInstruction();
}

TEST_F(RiscVVectorBasicBitManipulationTest, vclzv) {
  VclzVHelper<uint8_t>(this);
  ResetInstruction();
  VclzVHelper<uint16_t>(this);
  ResetInstruction();
  VclzVHelper<uint32_t>(this);
  ResetInstruction();
  VclzVHelper<uint64_t>(this);
  ResetInstruction();
}

TEST_F(RiscVVectorBasicBitManipulationTest, vctz) {
  VctzVHelper<uint8_t>(this);
  ResetInstruction();
  VctzVHelper<uint16_t>(this);
  ResetInstruction();
  VctzVHelper<uint32_t>(this);
  ResetInstruction();
  VctzVHelper<uint64_t>(this);
  ResetInstruction();
}

TEST_F(RiscVVectorBasicBitManipulationTest, vcpop) {
  VcpopVHelper<uint8_t>(this);
  ResetInstruction();
  VcpopVHelper<uint16_t>(this);
  ResetInstruction();
  VcpopVHelper<uint32_t>(this);
  ResetInstruction();
  VcpopVHelper<uint64_t>(this);
  ResetInstruction();
}

TEST_F(RiscVVectorBasicBitManipulationTest, vwsll) {
  VwsllVVHelper<uint8_t>(this);
  ResetInstruction();
  VwsllVVHelper<uint16_t>(this);
  ResetInstruction();
  VwsllVVHelper<uint32_t>(this);
  ResetInstruction();

  VwsllVXHelper<uint8_t>(this);
  ResetInstruction();
  VwsllVXHelper<uint16_t>(this);
  ResetInstruction();
  VwsllVXHelper<uint32_t>(this);
  ResetInstruction();

  VwsllVIHelper<uint8_t>(this);
  ResetInstruction();
  VwsllVIHelper<uint16_t>(this);
  ResetInstruction();
  VwsllVIHelper<uint32_t>(this);
  ResetInstruction();
}

}  // namespace
