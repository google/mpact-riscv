// Copyright 2026 Google LLC
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

// Decoder-level tests to verify that the immediate argument to vector shift
// instructions is handled correctly (i.e. treated as unsigned).

#include <array>
#include <cstdint>
#include <cstring>
#include <ios>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/decoder_interface.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "riscv/riscv32g_vec_decoder.h"
#include "riscv/riscv64g_vec_decoder.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_state.h"

namespace mpact::sim::riscv::test {
namespace {

using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::generic::DecoderInterface;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::util::FlatDemandMemory;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::TestWithParam;
using ::testing::Values;

// Func6 encodings for the instructions under test.
enum class Func6 : uint32_t {
  kVsll = 0b100101,
  kVsrl = 0b101000,
  kVsra = 0b101001,
  kVssrl = 0b101010,
  kVssra = 0b101011,
  kVslideup = 0b001110,
  kVslidedown = 0b001111,
};

constexpr uint64_t kInstructionAddress = 0x1000;
constexpr int kVectorByteLength = 64;  // 512 bits.
constexpr uint32_t kVd = 2;
constexpr uint32_t kVs2 = 4;

// Encodes an OPIVI (vector-immediate) instruction word given the 6-bit opcode
// (`func6`), destination vector register (`vd`), source-2 vector register
// (`vs2`), and the 5-bit unsigned immediate (`uimm5`).
constexpr uint32_t EncodeVArithVI(Func6 func6, uint32_t vd, uint32_t vs2,
                                  uint32_t uimm5) {
  return ((static_cast<uint32_t>(func6) & 0x3f) << 26) |
         (1u << 25) |  // vm = 1 (unmasked).
         ((vs2 & 0x1f) << 20) | ((uimm5 & 0x1f) << 15) |
         (0b011u << 12) |  // funct3 = OPIVI.
         ((vd & 0x1f) << 7) | 0b101'0111u;
}

// Parameterized fixture testing both RV32 and RV64 vector decoders.
class RiscVVectorShiftImmTest : public TestWithParam<RiscVXlen> {
 protected:
  void SetUp() override {
    xlen_ = GetParam();
    memory_ = std::make_unique<FlatDemandMemory>(0);
    state_ = std::make_unique<RiscVState>("test_state", xlen_, memory_.get());
    fp_state_ = std::make_unique<RiscVFPState>(state_->csr_set(), state_.get());
    state_->set_rv_fp(fp_state_.get());
    rv_vector_ =
        std::make_unique<RiscVVectorState>(state_.get(), kVectorByteLength);
    state_->set_rv_vector(rv_vector_.get());

    if (xlen_ == RiscVXlen::RV64) {
      decoder_ =
          std::make_unique<RiscV64GVecDecoder>(state_.get(), memory_.get());
    } else {
      decoder_ =
          std::make_unique<RiscV32GVecDecoder>(state_.get(), memory_.get());
    }

    vs2_reg_ =
        state_->GetRegister<RVVectorRegister>(absl::StrCat("v", kVs2)).first;
    vd_reg_ =
        state_->GetRegister<RVVectorRegister>(absl::StrCat("v", kVd)).first;
    ASSERT_NE(vs2_reg_, nullptr);
    ASSERT_NE(vd_reg_, nullptr);
  }

  // Configures the vector unit with the given vector type encoding (`vtype`),
  // number of active elements (`num_elements`), and resets `vstart` to 0.
  void ConfigureVectorUnit(uint32_t vtype, int num_elements) {
    rv_vector_->SetVectorType(vtype);
    rv_vector_->set_vector_length(num_elements);
    rv_vector_->set_vstart(0);
  }

  // Writes `instruction_word` into simulated memory at `kInstructionAddress`
  // and decodes it.
  Instruction* DecodeInstruction(uint32_t instruction_word) {
    DataBuffer* data_buffer = state_->db_factory()->Allocate<uint32_t>(1);
    data_buffer->Set<uint32_t>(0, instruction_word);
    memory_->Store(kInstructionAddress, data_buffer);
    data_buffer->DecRef();
    return decoder_->DecodeInstruction(kInstructionAddress);
  }

  // Decodes and executes the instruction word at `kInstructionAddress`.
  void ExecuteInstruction(uint32_t instruction_word) {
    Instruction* instruction = DecodeInstruction(instruction_word);
    ASSERT_NE(instruction, nullptr);
    instruction->Execute();
    instruction->DecRef();
  }

  // Executes `func6` for every combination of (immediate, test_value) and
  // checks the result against the expected C++ reference value.
  template <typename T>
  void RunShiftSweep(Func6 func6, uint32_t vtype, int num_elements,
                     uint32_t mask, absl::Span<const T> test_values,
                     absl::Span<const uint32_t> immediates) {
    ConfigureVectorUnit(vtype, num_elements);
    for (uint32_t immediate : immediates) {
      uint32_t effective_shift = immediate & mask;
      for (T test_value : test_values) {
        absl::Span<T> vs2_span = vs2_reg_->data_buffer()->Get<T>();
        for (int i = 0; i < num_elements; ++i) {
          vs2_span[i] = test_value;
        }
        std::memset(vd_reg_->data_buffer()->raw_ptr(), 0xcc, kVectorByteLength);

        ExecuteInstruction(EncodeVArithVI(func6, kVd, kVs2, immediate));

        T expected = {};
        switch (func6) {
          case Func6::kVsll:
            expected =
                static_cast<T>(static_cast<std::make_unsigned_t<T>>(test_value)
                               << effective_shift);
            break;
          case Func6::kVsrl:
            expected = static_cast<T>(
                static_cast<std::make_unsigned_t<T>>(test_value) >>
                effective_shift);
            break;
          case Func6::kVsra:
            expected =
                static_cast<T>(static_cast<std::make_signed_t<T>>(test_value) >>
                               effective_shift);
            break;
          default:
            FAIL() << "Unsupported func6 for shift sweep: "
                   << static_cast<uint32_t>(func6);
        }

        absl::Span<T> vd_span = vd_reg_->data_buffer()->Get<T>();
        for (int i = 0; i < num_elements; ++i) {
          EXPECT_EQ(vd_span[i], expected)
              << "func6=0x" << std::hex << static_cast<uint32_t>(func6)
              << " immediate=" << immediate << " (effective=" << effective_shift
              << ")"
              << " test_value=0x" << static_cast<uint64_t>(test_value)
              << " element " << std::dec << i;
        }
      }
    }
  }

  RiscVXlen xlen_;
  std::unique_ptr<FlatDemandMemory> memory_;
  std::unique_ptr<RiscVState> state_;
  std::unique_ptr<RiscVFPState> fp_state_;
  std::unique_ptr<RiscVVectorState> rv_vector_;
  std::unique_ptr<DecoderInterface> decoder_;
  RVVectorRegister* vs2_reg_ = nullptr;
  RVVectorRegister* vd_reg_ = nullptr;
};

// Tests to verify that the immediate arguments to vector shift instructions are
// not sign-extended when the highest-order bit is 1.

// vsll: 1 << 21 = 0x200000.
TEST_P(RiscVVectorShiftImmTest, VsllViImm21Sew64) {
  constexpr int kNumElements = 8;
  ConfigureVectorUnit(/*vtype=*/0x18, kNumElements);  // SEW=64, LMUL=1.

  absl::Span<uint64_t> vs2_span = vs2_reg_->data_buffer()->Get<uint64_t>();
  for (int i = 0; i < kNumElements; ++i) {
    vs2_span[i] = 1ULL;
  }
  std::memset(vd_reg_->data_buffer()->raw_ptr(), 0, kVectorByteLength);
  ExecuteInstruction(EncodeVArithVI(Func6::kVsll, kVd, kVs2, 21));
  absl::Span<uint64_t> vd_span = vd_reg_->data_buffer()->Get<uint64_t>();
  for (int i = 0; i < kNumElements; ++i) {
    EXPECT_EQ(vd_span[i], 1ULL << 21) << "vsll element " << i;
  }
}

// vsrl: (1<<53) >> 21 = 1<<32.
TEST_P(RiscVVectorShiftImmTest, VsrlViImm21Sew64) {
  constexpr int kNumElements = 8;
  ConfigureVectorUnit(/*vtype=*/0x18, kNumElements);  // SEW=64, LMUL=1.

  absl::Span<uint64_t> vs2_span = vs2_reg_->data_buffer()->Get<uint64_t>();
  for (int i = 0; i < kNumElements; ++i) {
    vs2_span[i] = 1ULL << 53;
  }
  std::memset(vd_reg_->data_buffer()->raw_ptr(), 0, kVectorByteLength);
  ExecuteInstruction(EncodeVArithVI(Func6::kVsrl, kVd, kVs2, 21));
  absl::Span<uint64_t> vd_span = vd_reg_->data_buffer()->Get<uint64_t>();
  for (int i = 0; i < kNumElements; ++i) {
    EXPECT_EQ(vd_span[i], (1ULL << 53) >> 21) << "vsrl element " << i;
  }
}

// vsra: Arithmetic right-shift of MSB-set value by 21.
TEST_P(RiscVVectorShiftImmTest, VsraViImm21Sew64) {
  constexpr int kNumElements = 8;
  ConfigureVectorUnit(/*vtype=*/0x18, kNumElements);  // SEW=64, LMUL=1.

  absl::Span<uint64_t> vs2_span = vs2_reg_->data_buffer()->Get<uint64_t>();
  for (int i = 0; i < kNumElements; ++i) {
    vs2_span[i] = 0x8000'0000'0000'0000ULL;
  }
  std::memset(vd_reg_->data_buffer()->raw_ptr(), 0, kVectorByteLength);
  ExecuteInstruction(EncodeVArithVI(Func6::kVsra, kVd, kVs2, 21));
  absl::Span<uint64_t> vd_span = vd_reg_->data_buffer()->Get<uint64_t>();
  constexpr uint64_t kExpected = static_cast<uint64_t>(
      static_cast<int64_t>(0x8000'0000'0000'0000ULL) >> 21);
  for (int i = 0; i < kNumElements; ++i) {
    EXPECT_EQ(vd_span[i], kExpected) << "vsra element " << i;
  }
}

// Exhaustive sweep: all 32 immediates, all element widths, all ops
TEST_P(RiscVVectorShiftImmTest, FullSweepAllImmediatesAllSews) {
  std::array<uint32_t, 32> all_immediates;
  std::iota(all_immediates.begin(), all_immediates.end(), 0);

  // SEW=64 (vtype 0x18, 8 elements, 6-bit mask).
  RunShiftSweep<uint64_t>(Func6::kVsll, 0x18, 8, 0x3f,
                          {1ULL, 0x8000'0000'0000'0000ULL}, all_immediates);
  RunShiftSweep<uint64_t>(Func6::kVsrl, 0x18, 8, 0x3f,
                          {0x8000'0000'0000'0000ULL, ~0ULL}, all_immediates);
  RunShiftSweep<int64_t>(
      Func6::kVsra, 0x18, 8, 0x3f,
      {static_cast<int64_t>(0x8000'0000'0000'0000ULL), 0x7fff'ffff'ffff'ffffLL},
      all_immediates);

  // SEW=32 (vtype 0x10, 16 elements, 5-bit mask).
  RunShiftSweep<uint32_t>(Func6::kVsll, 0x10, 16, 0x1f, {1U, 0x8000'0000U},
                          all_immediates);
  RunShiftSweep<uint32_t>(Func6::kVsrl, 0x10, 16, 0x1f, {0x8000'0000U, ~0U},
                          all_immediates);
  RunShiftSweep<int32_t>(Func6::kVsra, 0x10, 16, 0x1f,
                         {static_cast<int32_t>(0x8000'0000U), 0x7fff'ffff},
                         all_immediates);

  // SEW=16 (vtype 0x08, 32 elements, 4-bit mask).
  RunShiftSweep<uint16_t>(Func6::kVsll, 0x08, 32, 0xf,
                          {uint16_t{1}, uint16_t{0x8000}}, all_immediates);
  RunShiftSweep<uint16_t>(Func6::kVsrl, 0x08, 32, 0xf,
                          {uint16_t{0x8000}, uint16_t{0xffff}}, all_immediates);
  RunShiftSweep<int16_t>(Func6::kVsra, 0x08, 32, 0xf,
                         {static_cast<int16_t>(0x8000), int16_t{0x7fff}},
                         all_immediates);

  // SEW=8 (vtype 0x00, 64 elements, 3-bit mask).
  RunShiftSweep<uint8_t>(Func6::kVsll, 0x00, 64, 0x7,
                         {uint8_t{1}, uint8_t{0x80}}, all_immediates);
  RunShiftSweep<uint8_t>(Func6::kVsrl, 0x00, 64, 0x7,
                         {uint8_t{0x80}, uint8_t{0xff}}, all_immediates);
  RunShiftSweep<int8_t>(Func6::kVsra, 0x00, 64, 0x7,
                        {static_cast<int8_t>(0x80), int8_t{0x7f}},
                        all_immediates);
}

// Tests for large immediate arguments to saturating shift instructions

// vssrl: (1 << 25) >> 21 = 1 << 4 = 16.
TEST_P(RiscVVectorShiftImmTest, VssrlViImm21Sew64) {
  constexpr int kNumElements = 8;
  ConfigureVectorUnit(/*vtype=*/0x18, kNumElements);  // SEW=64, LMUL=1.
  rv_vector_->set_vxrm(0);                            // Round-to-nearest-up.

  absl::Span<uint64_t> vs2_span = vs2_reg_->data_buffer()->Get<uint64_t>();
  for (int i = 0; i < kNumElements; ++i) {
    vs2_span[i] = 1ULL << 25;
  }
  std::memset(vd_reg_->data_buffer()->raw_ptr(), 0, kVectorByteLength);
  ExecuteInstruction(EncodeVArithVI(Func6::kVssrl, kVd, kVs2, 21));
  absl::Span<uint64_t> vd_span = vd_reg_->data_buffer()->Get<uint64_t>();
  for (int i = 0; i < kNumElements; ++i) {
    EXPECT_EQ(vd_span[i], 1ULL << 4) << "vssrl element " << i;
  }
}

// vssra: -(1<<25) >> 21 = -(1<<4) = -16.
TEST_P(RiscVVectorShiftImmTest, VssraViImm21Sew64) {
  constexpr int kNumElements = 8;
  ConfigureVectorUnit(/*vtype=*/0x18, kNumElements);  // SEW=64, LMUL=1.
  rv_vector_->set_vxrm(0);                            // Round-to-nearest-up.

  absl::Span<int64_t> vs2_span = vs2_reg_->data_buffer()->Get<int64_t>();
  for (int i = 0; i < kNumElements; ++i) {
    vs2_span[i] = -static_cast<int64_t>(1ULL << 25);
  }
  std::memset(vd_reg_->data_buffer()->raw_ptr(), 0, kVectorByteLength);
  ExecuteInstruction(EncodeVArithVI(Func6::kVssra, kVd, kVs2, 21));
  absl::Span<int64_t> vd_span = vd_reg_->data_buffer()->Get<int64_t>();
  for (int i = 0; i < kNumElements; ++i) {
    EXPECT_EQ(vd_span[i], -static_cast<int64_t>(1ULL << 4))
        << "vssra element " << i;
  }
}

// Tests for large immediate arguments to `vslideup`/`vslidedown` instructions

// vslideup: elements 0..20 retain vd, elements 21..31 get vs2[0..10].
TEST_P(RiscVVectorShiftImmTest, VslideupViImm21) {
  ConfigureVectorUnit(/*vtype=*/0x00, /*num_elements=*/32);  // SEW=8, LMUL=1.

  absl::Span<uint8_t> vs2_span = vs2_reg_->data_buffer()->Get<uint8_t>();
  absl::Span<uint8_t> vd_init_span = vd_reg_->data_buffer()->Get<uint8_t>();
  for (int i = 0; i < kVectorByteLength; ++i) {
    vs2_span[i] = 100 + i;
    vd_init_span[i] = 10;
  }
  ExecuteInstruction(EncodeVArithVI(Func6::kVslideup, kVd, kVs2, 21));
  absl::Span<uint8_t> vd_span = vd_reg_->data_buffer()->Get<uint8_t>();
  for (int i = 0; i < 21; ++i) {
    EXPECT_EQ(vd_span[i], 10) << "vslideup vd[" << i << "] should be unchanged";
  }
  for (int i = 21; i < 32; ++i) {
    EXPECT_EQ(vd_span[i], 100 + (i - 21))
        << "vslideup vd[" << i << "] should be vs2[" << (i - 21) << "]";
  }
}

// vslidedown: vd[i] = vs2[i + 21].
TEST_P(RiscVVectorShiftImmTest, VslidedownViImm21) {
  ConfigureVectorUnit(/*vtype=*/0x00, /*num_elements=*/32);  // SEW=8, LMUL=1.

  absl::Span<uint8_t> vs2_span = vs2_reg_->data_buffer()->Get<uint8_t>();
  for (int i = 0; i < kVectorByteLength; ++i) {
    vs2_span[i] = 50 + i;
  }
  std::memset(vd_reg_->data_buffer()->raw_ptr(), 0, kVectorByteLength);
  ExecuteInstruction(EncodeVArithVI(Func6::kVslidedown, kVd, kVs2, 21));
  absl::Span<uint8_t> vd_span = vd_reg_->data_buffer()->Get<uint8_t>();
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(vd_span[i], 50 + i + 21)
        << "vslidedown vd[" << i << "] should be vs2[" << (i + 21) << "]";
  }
}

// In dissasembly, immediate should print as unsigned (21, not -11) ---
TEST_P(RiscVVectorShiftImmTest, DisassemblyShowsUnsignedImmediate) {
  for (Func6 func6 : {Func6::kVsll, Func6::kVsrl, Func6::kVsra}) {
    Instruction* instruction =
        DecodeInstruction(EncodeVArithVI(func6, kVd, kVs2, 21));
    ASSERT_NE(instruction, nullptr);
    std::string disassembly = instruction->AsString();
    instruction->DecRef();
    EXPECT_THAT(disassembly, HasSubstr("21"));
    EXPECT_THAT(disassembly, Not(HasSubstr("-11")));
  }
}

INSTANTIATE_TEST_SUITE_P(XlenVariants, RiscVVectorShiftImmTest,
                         Values(RiscVXlen::RV64, RiscVXlen::RV32));

}  // namespace
}  // namespace mpact::sim::riscv::test
