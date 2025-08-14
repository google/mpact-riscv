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

#include "riscv/riscv_vector_permute_instructions.h"

#include <cstdint>

#include "absl/random/random.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/instruction.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_vector_state.h"
#include "riscv/test/riscv_vector_instructions_test_base.h"

// This file contains tests for the RiscV vector permute instructions.

namespace {

using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RVVectorRegister;
using ::mpact::sim::riscv::test::RiscVVectorInstructionsTestBase;

using ::mpact::sim::riscv::Vcompress;
using ::mpact::sim::riscv::Vrgather;
using ::mpact::sim::riscv::Vrgatherei16;
using ::mpact::sim::riscv::Vslide1down;
using ::mpact::sim::riscv::Vslide1up;
using ::mpact::sim::riscv::Vslidedown;
using ::mpact::sim::riscv::Vslideup;

using ::mpact::sim::riscv::test::kA5Mask;
using ::mpact::sim::riscv::test::kLmulSettingByLogSize;
using ::mpact::sim::riscv::test::kRs1Name;
using ::mpact::sim::riscv::test::kSewSettingsByByteSize;
using ::mpact::sim::riscv::test::kVd;
using ::mpact::sim::riscv::test::kVectorLengthInBytes;
using ::mpact::sim::riscv::test::kVmask;
using ::mpact::sim::riscv::test::kVmaskName;
using ::mpact::sim::riscv::test::kVs1;
using ::mpact::sim::riscv::test::kVs2;

class RiscVVectorPermuteInstructionsTest
    : public RiscVVectorInstructionsTestBase {};

// Helper function for vector-vector vrgather instructions.
template <typename T, typename I>
void VrgatherVVHelper(RiscVVectorPermuteInstructionsTest* tester,
                      Instruction* inst) {
  auto* rv_vector = tester->rv_vector();
  // Configure vector unit for sew and maximum lmul.
  uint32_t vtype = 0;
  int max_regs = 8;
  if (sizeof(I) > sizeof(T)) {
    // This happens for vrgatherei16 when sew is 8.
    vtype = (kSewSettingsByByteSize[sizeof(T)] << 3) | kLmulSettingByLogSize[6];
    max_regs = 4;
  } else {
    vtype = (kSewSettingsByByteSize[sizeof(T)] << 3) | kLmulSettingByLogSize[7];
  }
  tester->ConfigureVectorUnit(vtype, 2048);

  int vlen = rv_vector->vector_length();
  int num_values_per_reg = kVectorLengthInBytes / sizeof(T);
  int max_index = num_values_per_reg * max_regs;
  int num_indices_per_reg = kVectorLengthInBytes / sizeof(I);
  // Initialize vs2 and vs1 to random values.
  for (int reg = kVs2; reg < kVs2 + max_regs; reg++) {
    auto span = tester->vreg()[reg]->data_buffer()->Get<T>();
    for (int i = 0; i < num_values_per_reg; i++) {
      span[i] = tester->RandomValue<T>();
    }
  }
  for (int reg = kVs1; reg < kVs1 + max_regs; reg++) {
    auto span = tester->vreg()[reg]->data_buffer()->Get<I>();
    for (int i = 0; i < num_indices_per_reg; i++) {
      span[i] =
          absl::Uniform(absl::IntervalClosed, tester->bitgen(), 0, 2 * vlen);
    }
  }
  tester->SetVectorRegisterValues<uint8_t>({{kVmaskName, kA5Mask}});

  inst->Execute();
  for (int i = 0; i < vlen; i++) {
    int value_reg_offset = i / num_values_per_reg;
    int value_elem_index = i % num_values_per_reg;
    int index_reg_offset = i / num_indices_per_reg;
    int index_elem_index = i % num_indices_per_reg;
    int mask_index = i >> 8;
    int mask_offset = i & 0b111;
    bool mask_value = (kA5Mask[mask_index] >> mask_offset) & 0b1;
    // Get the destination value.
    T dst = tester->vreg()[kVd + value_reg_offset]->data_buffer()->Get<T>(
        value_elem_index);
    if (mask_value) {
      // If it's an active element, get the index value.
      I indx = tester->vreg()[kVs1 + index_reg_offset]->data_buffer()->Get<I>(
          index_elem_index);
      if (indx >= max_index) {  // If the index is out of range, compare to 0.
        EXPECT_EQ(0, dst);
      } else {  // Else, get the src value at that index and compare.
        int reg = kVs2 + indx / num_values_per_reg;
        int element = indx % num_values_per_reg;
        T src = tester->vreg()[reg]->data_buffer()->Get<T>(element);
        EXPECT_EQ(src, dst);
      }
    } else {  // Inactive values are unchanged.
      T src = tester->vreg()[kVd + value_reg_offset]->data_buffer()->Get<T>(
          value_elem_index);
      EXPECT_EQ(src, dst) << "index: " << i << " offset: " << value_reg_offset
                          << "  elem: " << value_elem_index;
    }
  }
}

// Helper function for vector-scalar vrgather instructions.
template <typename T>
void VrgatherVSHelper(RiscVVectorPermuteInstructionsTest* tester,
                      Instruction* inst) {
  auto* rv_vector = tester->rv_vector();
  // Configure vector unit.
  uint32_t vtype =
      (kSewSettingsByByteSize[sizeof(T)] << 3) | kLmulSettingByLogSize[7];
  tester->ConfigureVectorUnit(vtype, 2048);
  int vlen = rv_vector->vector_length();
  int num_values_per_reg = kVectorLengthInBytes / sizeof(T);
  int max_index = num_values_per_reg * 8;
  // Initialize vs2 to random values.
  for (int reg = kVs2; reg < kVs2 + 8; reg++) {
    auto span = tester->vreg()[reg]->data_buffer()->Get<T>();
    for (int i = 0; i < num_values_per_reg; i++) {
      span[i] = tester->RandomValue<T>();
    }
  }
  tester->SetVectorRegisterValues<uint8_t>({{kVmaskName, kA5Mask}});
  // Try 20 different index values randomly.
  for (int num = 0; num < 20; num++) {
    // Set a random index value.
    RV32Register::ValueType index_value =
        absl::Uniform(absl::IntervalClosed, tester->bitgen(), 0, 2 * vlen);
    tester->SetRegisterValues<RV32Register::ValueType>(
        {{kRs1Name, index_value}});
    // Execute the instruction.
    inst->Execute();
    for (int i = 0; i < vlen; i++) {
      int value_reg_offset = i / num_values_per_reg;
      int value_elem_index = i % num_values_per_reg;
      int mask_index = i >> 8;
      int mask_offset = i & 0b111;
      bool mask_value = (kA5Mask[mask_index] >> mask_offset) & 0b1;
      // Get the destination value.
      T dst = tester->vreg()[kVd + value_reg_offset]->data_buffer()->Get<T>(
          value_elem_index);
      if (mask_value) {  // If it's an active value.
        // If the index is out of range, it's 0.
        if (index_value >= max_index) {
          EXPECT_EQ(0, dst) << "max: " << max_index << " indx: " << index_value;
        } else {  // Otherwise, get the src value from vs2 and compare.
          int reg = index_value / num_values_per_reg;
          int element = index_value % num_values_per_reg;
          T src = tester->vreg()[kVs2 + reg]->data_buffer()->Get<T>(element);
          EXPECT_EQ(src, dst);
        }
      } else {  // Inactive values are unchanged.
        T src = tester->vreg()[kVd + value_reg_offset]->data_buffer()->Get<T>(
            value_elem_index);
        EXPECT_EQ(src, dst);
      }
    }
  }
}

// Test vrgather instruction for 1, 2, 4, and 8 byte SEWs - vector index.
TEST_F(RiscVVectorPermuteInstructionsTest, VrgatherVV8) {
  SetSemanticFunction(&Vrgather);
  AppendVectorRegisterOperands({kVs2, kVs1, kVmask}, {kVd});
  VrgatherVVHelper<uint8_t, uint8_t>(this, instruction_);
}

TEST_F(RiscVVectorPermuteInstructionsTest, VrgatherVV16) {
  SetSemanticFunction(&Vrgather);
  AppendVectorRegisterOperands({kVs2, kVs1, kVmask}, {kVd});
  VrgatherVVHelper<uint16_t, uint16_t>(this, instruction_);
}

TEST_F(RiscVVectorPermuteInstructionsTest, VrgatherVV32) {
  SetSemanticFunction(&Vrgather);
  AppendVectorRegisterOperands({kVs2, kVs1, kVmask}, {kVd});
  VrgatherVVHelper<uint32_t, uint32_t>(this, instruction_);
}

TEST_F(RiscVVectorPermuteInstructionsTest, VrgatherVV64) {
  SetSemanticFunction(&Vrgather);
  AppendVectorRegisterOperands({kVs2, kVs1, kVmask}, {kVd});
  VrgatherVVHelper<uint64_t, uint64_t>(this, instruction_);
}

// Test vrgather instruction for 1, 2, 4, and 8 byte SEWs - scalar index.
TEST_F(RiscVVectorPermuteInstructionsTest, VrgatherVS8) {
  SetSemanticFunction(&Vrgather);
  AppendVectorRegisterOperands({kVs2}, {});
  AppendRegisterOperands<RV32Register>({kRs1Name}, {});
  AppendVectorRegisterOperands({kVmask}, {kVd});
  VrgatherVSHelper<uint8_t>(this, instruction_);
}

TEST_F(RiscVVectorPermuteInstructionsTest, VrgatherVS16) {
  SetSemanticFunction(&Vrgather);
  AppendVectorRegisterOperands({kVs2}, {});
  AppendRegisterOperands<RV32Register>({kRs1Name}, {});
  AppendVectorRegisterOperands({kVmask}, {kVd});
  VrgatherVSHelper<uint16_t>(this, instruction_);
}

TEST_F(RiscVVectorPermuteInstructionsTest, VrgatherVS32) {
  SetSemanticFunction(&Vrgather);
  AppendVectorRegisterOperands({kVs2}, {});
  AppendRegisterOperands<RV32Register>({kRs1Name}, {});
  AppendVectorRegisterOperands({kVmask}, {kVd});
  VrgatherVSHelper<uint32_t>(this, instruction_);
}

TEST_F(RiscVVectorPermuteInstructionsTest, VrgatherVS64) {
  SetSemanticFunction(&Vrgather);
  AppendVectorRegisterOperands({kVs2}, {});
  AppendRegisterOperands<RV32Register>({kRs1Name}, {});
  AppendVectorRegisterOperands({kVmask}, {kVd});
  VrgatherVSHelper<uint64_t>(this, instruction_);
}

// Test vrgatherei16 instruction for SEW values of 1, 2, 4, and 8 bytes.
TEST_F(RiscVVectorPermuteInstructionsTest, Vrgatherei16VV8) {
  SetSemanticFunction(&Vrgatherei16);
  AppendVectorRegisterOperands({kVs2, kVs1, kVmask}, {kVd});
  VrgatherVVHelper<uint8_t, uint16_t>(this, instruction_);
}

TEST_F(RiscVVectorPermuteInstructionsTest, Vrgatherei16VV16) {
  SetSemanticFunction(&Vrgatherei16);
  AppendVectorRegisterOperands({kVs2, kVs1, kVmask}, {kVd});
  VrgatherVVHelper<uint16_t, uint16_t>(this, instruction_);
}

TEST_F(RiscVVectorPermuteInstructionsTest, Vrgatherei16VV32) {
  SetSemanticFunction(&Vrgatherei16);
  AppendVectorRegisterOperands({kVs2, kVs1, kVmask}, {kVd});
  VrgatherVVHelper<uint32_t, uint16_t>(this, instruction_);
}

TEST_F(RiscVVectorPermuteInstructionsTest, Vrgatherei16VV64) {
  SetSemanticFunction(&Vrgatherei16);
  AppendVectorRegisterOperands({kVs2, kVs1, kVmask}, {kVd});
  VrgatherVVHelper<uint64_t, uint16_t>(this, instruction_);
}

// Helper function for slideup/down instructions.
template <typename T>
void SlideHelper(RiscVVectorPermuteInstructionsTest* tester, Instruction* inst,
                 bool is_slide_up) {
  auto* rv_vector = tester->rv_vector();
  uint32_t vtype =
      (kSewSettingsByByteSize[sizeof(T)] << 3) | kLmulSettingByLogSize[7];
  tester->ConfigureVectorUnit(vtype, 2048);
  int vlen = rv_vector->vector_length();
  int max_vlen = rv_vector->max_vector_length();
  int num_values_per_reg = kVectorLengthInBytes / sizeof(T);
  // Initialize vs2 to random values.
  for (int reg = 0; reg < 8; reg++) {
    auto src_span = tester->vreg()[kVs2 + reg]->data_buffer()->Get<T>();
    for (int i = 0; i < num_values_per_reg; i++) {
      src_span[i] = tester->RandomValue<T>();
    }
  }
  tester->SetVectorRegisterValues<uint8_t>({{kVmaskName, kA5Mask}});
  // Try 20 different shift values randomly.
  for (int num = 0; num < 20; num++) {
    RV32Register::ValueType shift_value =
        absl::Uniform(absl::IntervalClosed, tester->bitgen(), 0, 2 * vlen);
    tester->SetRegisterValues<RV32Register::ValueType>(
        {{kRs1Name, shift_value}});
    // Initialize the destination register set.
    int value_indx = 0;
    for (int reg = 0; reg < 8; reg++) {
      auto dst_span = tester->vreg()[kVd + reg]->data_buffer()->Get<T>();
      for (int i = 0; i < num_values_per_reg; i++) {
        dst_span[i] = value_indx++;
      }
    }
    inst->Execute();
    for (int i = 0; i < vlen; i++) {
      int value_reg_offset = i / num_values_per_reg;
      int value_elem_index = i % num_values_per_reg;
      int mask_index = i >> 8;
      int mask_offset = i & 0b111;
      bool mask_value = (kA5Mask[mask_index] >> mask_offset) & 0b1;
      T dst = tester->vreg()[kVd + value_reg_offset]->data_buffer()->Get<T>(
          value_elem_index);
      if (is_slide_up) {  // For slide up instruction.
        if ((i < shift_value) || (!mask_value)) {
          // If it's an inactive element, or the index is less than the shift
          // amount, the element is unchanged.
          T value = static_cast<T>(i);
          EXPECT_EQ(value, dst) << "indx: " << i;
        } else {
          // Active elements are shifted up by 'shift_value'.
          int src_reg_offset = (i - shift_value) / num_values_per_reg;
          int src_reg_index = (i - shift_value) % num_values_per_reg;
          T src = tester->vreg()[kVs2 + src_reg_offset]->data_buffer()->Get<T>(
              src_reg_index);
          EXPECT_EQ(src, dst) << "indx: " << i;
        }
      } else {  // This is slide down.
        if (mask_value) {
          if (i + shift_value >= max_vlen) {
            // Active elements originating beyond max_vlen are 0.
            EXPECT_EQ(0, dst) << "indx: " << i;
          } else {
            // Active elements are shifted down by 'shift_value'.
            int src_reg_offset = (i + shift_value) / num_values_per_reg;
            int src_reg_index = (i + shift_value) % num_values_per_reg;
            T src =
                tester->vreg()[kVs2 + src_reg_offset]->data_buffer()->Get<T>(
                    src_reg_index);
            EXPECT_EQ(src, dst) << "indx: " << i;
          }
        } else {
          // All inactive elements are unchanged.
          T src = tester->vreg()[kVd + value_reg_offset]->data_buffer()->Get<T>(
              value_elem_index);
          EXPECT_EQ(src, dst) << "indx: " << i;
        }
      }
    }
  }
}

// Test vslideup instruction for SEW values of 1, 2, 4, and 8 bytes.
TEST_F(RiscVVectorPermuteInstructionsTest, Vslideup8) {
  SetSemanticFunction(&Vslideup);
  AppendVectorRegisterOperands({kVs2}, {});
  AppendRegisterOperands<RV32Register>({kRs1Name}, {});
  AppendVectorRegisterOperands({kVmask}, {kVd});
  SlideHelper<uint8_t>(this, instruction_, /*is_slide_up*/ true);
}

TEST_F(RiscVVectorPermuteInstructionsTest, Vslideup16) {
  SetSemanticFunction(&Vslideup);
  AppendVectorRegisterOperands({kVs2}, {});
  AppendRegisterOperands<RV32Register>({kRs1Name}, {});
  AppendVectorRegisterOperands({kVmask}, {kVd});
  SlideHelper<uint16_t>(this, instruction_, /*is_slide_up*/ true);
}

TEST_F(RiscVVectorPermuteInstructionsTest, Vslideup32) {
  SetSemanticFunction(&Vslideup);
  AppendVectorRegisterOperands({kVs2}, {});
  AppendRegisterOperands<RV32Register>({kRs1Name}, {});
  AppendVectorRegisterOperands({kVmask}, {kVd});
  SlideHelper<uint32_t>(this, instruction_, /*is_slide_up*/ true);
}

TEST_F(RiscVVectorPermuteInstructionsTest, Vslideup64) {
  SetSemanticFunction(&Vslideup);
  AppendVectorRegisterOperands({kVs2}, {});
  AppendRegisterOperands<RV32Register>({kRs1Name}, {});
  AppendVectorRegisterOperands({kVmask}, {kVd});
  SlideHelper<uint64_t>(this, instruction_, /*is_slide_up*/ true);
}

// Test vslidedown instruction for SEW values of 1, 2, 4, and 8 bytes.
TEST_F(RiscVVectorPermuteInstructionsTest, Vslidedown8) {
  SetSemanticFunction(&Vslidedown);
  AppendVectorRegisterOperands({kVs2}, {});
  AppendRegisterOperands<RV32Register>({kRs1Name}, {});
  AppendVectorRegisterOperands({kVmask}, {kVd});
  SlideHelper<uint8_t>(this, instruction_, /*is_slide_up*/ false);
}

TEST_F(RiscVVectorPermuteInstructionsTest, Vslidedown16) {
  SetSemanticFunction(&Vslidedown);
  AppendVectorRegisterOperands({kVs2}, {});
  AppendRegisterOperands<RV32Register>({kRs1Name}, {});
  AppendVectorRegisterOperands({kVmask}, {kVd});
  SlideHelper<uint16_t>(this, instruction_, /*is_slide_up*/ false);
}

TEST_F(RiscVVectorPermuteInstructionsTest, Vslidedown32) {
  SetSemanticFunction(&Vslidedown);
  AppendVectorRegisterOperands({kVs2}, {});
  AppendRegisterOperands<RV32Register>({kRs1Name}, {});
  AppendVectorRegisterOperands({kVmask}, {kVd});
  SlideHelper<uint32_t>(this, instruction_, /*is_slide_up*/ false);
}

TEST_F(RiscVVectorPermuteInstructionsTest, Vslidedown64) {
  SetSemanticFunction(&Vslidedown);
  AppendVectorRegisterOperands({kVs2}, {});
  AppendRegisterOperands<RV32Register>({kRs1Name}, {});
  AppendVectorRegisterOperands({kVmask}, {kVd});
  SlideHelper<uint64_t>(this, instruction_, /*is_slide_up*/ false);
}

template <typename T>
void Slide1Helper(RiscVVectorPermuteInstructionsTest* tester, Instruction* inst,
                  bool is_slide_up) {
  auto* rv_vector = tester->rv_vector();
  uint32_t vtype =
      (kSewSettingsByByteSize[sizeof(T)] << 3) | kLmulSettingByLogSize[7];
  tester->ConfigureVectorUnit(vtype, 2048);
  int vlen = rv_vector->vector_length();
  int max_vlen = rv_vector->max_vector_length();
  int num_values_per_reg = kVectorLengthInBytes / sizeof(T);
  // Initialize vs2 to random values.
  for (int reg = kVs2; reg < kVs2 + 8; reg++) {
    auto span = tester->vreg()[reg]->data_buffer()->Get<T>();
    for (int i = 0; i < num_values_per_reg; i++) {
      span[i] = tester->RandomValue<T>();
    }
  }
  tester->SetVectorRegisterValues<uint8_t>({{kVmaskName, kA5Mask}});
  // Try 20 different shift values randomly.
  for (int num = 0; num < 20; num++) {
    RV32Register::ValueType fill_in_value =
        absl::Uniform(absl::IntervalClosed, tester->bitgen(), 0, 2 * vlen);
    tester->SetRegisterValues<RV32Register::ValueType>(
        {{kRs1Name, fill_in_value}});
    fill_in_value = static_cast<T>(fill_in_value);
    inst->Execute();
    for (int i = 0; i < vlen; i++) {
      int value_reg_offset = i / num_values_per_reg;
      int value_elem_index = i % num_values_per_reg;
      int mask_index = i >> 8;
      int mask_offset = i & 0b111;
      bool mask_value = (kA5Mask[mask_index] >> mask_offset) & 0b1;
      T dst = tester->vreg()[kVd + value_reg_offset]->data_buffer()->Get<T>(
          value_elem_index);
      if (is_slide_up) {
        if (!mask_value) {  // Inactive elements are unchanged.
          T src = tester->vreg()[kVd + value_reg_offset]->data_buffer()->Get<T>(
              value_elem_index);
          EXPECT_EQ(src, dst) << "i: " << i;
        } else {
          if (i == 0) {  // The first value should match the fill-in.
            EXPECT_EQ(fill_in_value, dst) << "i: " << i;
          } else {  // Other values are shifted by 1.
            int src_reg_offset = (i - 1) / num_values_per_reg;
            int src_reg_index = (i - 1) % num_values_per_reg;
            T src =
                tester->vreg()[kVs2 + src_reg_offset]->data_buffer()->Get<T>(
                    src_reg_index);
            EXPECT_EQ(src, dst) << "i: " << i;
          }
        }
      } else {  // This is slide down.
        if (mask_value) {
          if (i + 1 >= max_vlen) {  // The last value should match the fill-in.
            EXPECT_EQ(fill_in_value, dst);
          } else {  // Other elements are shifted by 1.
            int src_reg_offset = (i + 1) / num_values_per_reg;
            int src_reg_index = (i + 1) % num_values_per_reg;
            T src =
                tester->vreg()[kVs2 + src_reg_offset]->data_buffer()->Get<T>(
                    src_reg_index);
            EXPECT_EQ(src, dst) << "i: " << i;
          }
        } else {  // Inactive elements are unchanged.
          T src = tester->vreg()[kVd + value_reg_offset]->data_buffer()->Get<T>(
              value_elem_index);
          EXPECT_EQ(src, dst) << "i: " << i;
        }
      }
    }
  }
}

// Test vslide1up instruction for SEW values of 1, 2, 4, and 8 bytes.
TEST_F(RiscVVectorPermuteInstructionsTest, Vslide1up8) {
  SetSemanticFunction(&Vslide1up);
  AppendVectorRegisterOperands({kVs2}, {});
  AppendRegisterOperands<RV32Register>({kRs1Name}, {});
  AppendVectorRegisterOperands({kVmask}, {kVd});
  Slide1Helper<uint8_t>(this, instruction_, /*is_slide_up*/ true);
}

TEST_F(RiscVVectorPermuteInstructionsTest, Vslide1up16) {
  SetSemanticFunction(&Vslide1up);
  AppendVectorRegisterOperands({kVs2}, {});
  AppendRegisterOperands<RV32Register>({kRs1Name}, {});
  AppendVectorRegisterOperands({kVmask}, {kVd});
  Slide1Helper<uint16_t>(this, instruction_, /*is_slide_up*/ true);
}

TEST_F(RiscVVectorPermuteInstructionsTest, Vslide1up32) {
  SetSemanticFunction(&Vslide1up);
  AppendVectorRegisterOperands({kVs2}, {});
  AppendRegisterOperands<RV32Register>({kRs1Name}, {});
  AppendVectorRegisterOperands({kVmask}, {kVd});
  Slide1Helper<uint32_t>(this, instruction_, /*is_slide_up*/ true);
}

TEST_F(RiscVVectorPermuteInstructionsTest, Vslide1up64) {
  SetSemanticFunction(&Vslide1up);
  AppendVectorRegisterOperands({kVs2}, {});
  AppendRegisterOperands<RV32Register>({kRs1Name}, {});
  AppendVectorRegisterOperands({kVmask}, {kVd});
  Slide1Helper<uint64_t>(this, instruction_, /*is_slide_up*/ true);
}
// Test vslide1down instruction for SEW values of 1, 2, 4, and 8 bytes.
TEST_F(RiscVVectorPermuteInstructionsTest, Vslide1down8) {
  SetSemanticFunction(&Vslide1down);
  AppendVectorRegisterOperands({kVs2}, {});
  AppendRegisterOperands<RV32Register>({kRs1Name}, {});
  AppendVectorRegisterOperands({kVmask}, {kVd});
  Slide1Helper<uint8_t>(this, instruction_, /*is_slide_up*/ false);
}

TEST_F(RiscVVectorPermuteInstructionsTest, Vslide1down16) {
  SetSemanticFunction(&Vslide1down);
  AppendVectorRegisterOperands({kVs2}, {});
  AppendRegisterOperands<RV32Register>({kRs1Name}, {});
  AppendVectorRegisterOperands({kVmask}, {kVd});
  Slide1Helper<uint16_t>(this, instruction_, /*is_slide_up*/ false);
}

TEST_F(RiscVVectorPermuteInstructionsTest, Vslide1down32) {
  SetSemanticFunction(&Vslide1down);
  AppendVectorRegisterOperands({kVs2}, {});
  AppendRegisterOperands<RV32Register>({kRs1Name}, {});
  AppendVectorRegisterOperands({kVmask}, {kVd});
  Slide1Helper<uint32_t>(this, instruction_, /*is_slide_up*/ false);
}

TEST_F(RiscVVectorPermuteInstructionsTest, Vslide1down64) {
  SetSemanticFunction(&Vslide1down);
  AppendVectorRegisterOperands({kVs2}, {});
  AppendRegisterOperands<RV32Register>({kRs1Name}, {});
  AppendVectorRegisterOperands({kVmask}, {kVd});
  Slide1Helper<uint64_t>(this, instruction_, /*is_slide_up*/ false);
}

template <typename T>
void CompressHelper(RiscVVectorPermuteInstructionsTest* tester,
                    Instruction* inst) {
  auto* rv_vector = tester->rv_vector();
  uint32_t vtype =
      (kSewSettingsByByteSize[sizeof(T)] << 3) | kLmulSettingByLogSize[7];
  tester->ConfigureVectorUnit(vtype, 2048);
  int vlen = rv_vector->vector_length();
  int num_values_per_reg = kVectorLengthInBytes / sizeof(T);
  auto vd_span = tester->vreg()[kVd]->data_buffer()->Get<T>();
  std::vector<T> origin_vd_values(vd_span.begin(), vd_span.end());
  // Initialize vs2 to random values.
  for (int reg = kVs2; reg < kVs2 + 8; reg++) {
    auto span = tester->vreg()[reg]->data_buffer()->Get<T>();
    for (int i = 0; i < num_values_per_reg; i++) {
      span[i] = tester->RandomValue<T>();
    }
  }
  tester->SetVectorRegisterValues<uint8_t>({{kVmaskName, kA5Mask}});
  inst->Execute();
  // First check all the elements that were compressed (mask bit true).
  int offset = 0;
  for (int i = 0; i < vlen; i++) {
    int value_reg_offset = i / num_values_per_reg;
    int value_elem_index = i % num_values_per_reg;
    int mask_index = i >> 8;
    int mask_offset = i & 0b111;
    bool mask_value = (kA5Mask[mask_index] >> mask_offset) & 0b1;
    if (mask_value) {
      T src = tester->vreg()[kVs2 + value_reg_offset]->data_buffer()->Get<T>(
          value_elem_index);
      int dst_reg_index = offset / num_values_per_reg;
      int dst_element_index = offset % num_values_per_reg;
      T dst = tester->vreg()[kVd + dst_reg_index]->data_buffer()->Get<T>(
          dst_element_index);
      EXPECT_EQ(src, dst) << "index: " << i;
      offset++;
    }
  }
  // The remaining elements are unchanged.
  for (int i = offset; i < vlen; i++) {
    int value_reg_index = i / num_values_per_reg;
    int value_elem_index = i % num_values_per_reg;
    T src = origin_vd_values[value_elem_index];
    T dst = tester->vreg()[kVd + value_reg_index]->data_buffer()->Get<T>(
        value_elem_index);
    EXPECT_EQ(src, dst) << "index: " << i;
  }
}

// Test compress instruction for SEW of 8, 16, 32, and 64.
TEST_F(RiscVVectorPermuteInstructionsTest, Vcompress8) {
  SetSemanticFunction(&Vcompress);
  AppendVectorRegisterOperands({kVs2, kVmask}, {kVd});
  CompressHelper<uint8_t>(this, instruction_);
}

TEST_F(RiscVVectorPermuteInstructionsTest, Vcompress16) {
  SetSemanticFunction(&Vcompress);
  AppendVectorRegisterOperands({kVs2, kVmask}, {kVd});
  CompressHelper<uint16_t>(this, instruction_);
}

TEST_F(RiscVVectorPermuteInstructionsTest, Vcompress32) {
  SetSemanticFunction(&Vcompress);
  AppendVectorRegisterOperands({kVs2, kVmask}, {kVd});
  CompressHelper<uint32_t>(this, instruction_);
}

TEST_F(RiscVVectorPermuteInstructionsTest, Vcompress64) {
  SetSemanticFunction(&Vcompress);
  AppendVectorRegisterOperands({kVs2, kVmask}, {kVd});
  CompressHelper<uint64_t>(this, instruction_);
}

}  // namespace
