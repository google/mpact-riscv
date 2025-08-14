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

#include "riscv/riscv_vector_memory_instructions.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <ios>
#include <string>
#include <vector>

#include "absl/functional/bind_front.h"
#include "absl/numeric/bits.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"

// This file contains the test fixture and tests for testing RiscV vector
// memory instructions.

namespace {

using ::absl::Span;
using ::mpact::sim::generic::ImmediateOperand;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::generic::RegisterBase;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVVectorState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RV32VectorDestinationOperand;
using ::mpact::sim::riscv::RV32VectorSourceOperand;
using ::mpact::sim::riscv::RVVectorRegister;
using ::mpact::sim::util::FlatDemandMemory;
using ::std::tuple;

// Semantic functions.
using ::mpact::sim::riscv::VlChild;
using ::mpact::sim::riscv::VlIndexed;
using ::mpact::sim::riscv::Vlm;
using ::mpact::sim::riscv::VlRegister;
using ::mpact::sim::riscv::VlSegment;
using ::mpact::sim::riscv::VlSegmentChild;
using ::mpact::sim::riscv::VlSegmentIndexed;
using ::mpact::sim::riscv::VlSegmentStrided;
using ::mpact::sim::riscv::VlStrided;
using ::mpact::sim::riscv::Vsetvl;
using ::mpact::sim::riscv::VsIndexed;
using ::mpact::sim::riscv::Vsm;
using ::mpact::sim::riscv::VsRegister;
using ::mpact::sim::riscv::VsSegment;
using ::mpact::sim::riscv::VsSegmentIndexed;
using ::mpact::sim::riscv::VsSegmentStrided;
using ::mpact::sim::riscv::VsStrided;

// Constants used in the tests.
constexpr int kVectorLengthInBits = 512;
constexpr int kVectorLengthInBytes = kVectorLengthInBits / 8;
constexpr uint32_t kInstAddress = 0x1000;
constexpr uint32_t kDataLoadAddress = 0x1'0000;
constexpr uint32_t kDataStoreAddress = 0x8'0000;
constexpr char kRs1Name[] = "x1";
constexpr int kRs1 = 1;
constexpr char kRs2Name[] = "x2";
constexpr char kRs3Name[] = "x3";
constexpr char kRdName[] = "x8";
constexpr int kRd = 8;
constexpr int kVmask = 1;
constexpr char kVmaskName[] = "v1";
constexpr int kVd = 8;
constexpr int kVs1 = 16;
constexpr int kVs2 = 24;

// Setting bits and corresponding values for lmul and sew.
constexpr int kLmulSettings[7] = {0b101, 0b110, 0b111, 0b000,
                                  0b001, 0b010, 0b011};
constexpr int kLmul8Values[7] = {1, 2, 4, 8, 16, 32, 64};
constexpr int kLmulSettingByLogSize[] = {0,     0b101, 0b110, 0b111,
                                         0b000, 0b001, 0b010, 0b011};
constexpr int kSewSettings[4] = {0b000, 0b001, 0b010, 0b011};
constexpr int kSewValues[4] = {1, 2, 4, 8};
constexpr int kSewSettingsByByteSize[] = {0, 0b000, 0b001, 0,    0b010,
                                          0, 0,     0,     0b011};

// Don't need to set every byte, as only the low bits are used for mask values.
constexpr uint8_t kA5Mask[kVectorLengthInBytes] = {
    0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5,
    0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5,
    0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5,
    0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5,
    0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5,
    0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5,
};

// Test fixture class. This class allows for more convenient manipulations
// of instructions to test the semantic functions.
class RV32VInstructionsTest : public testing::Test {
 public:
  RV32VInstructionsTest() {
    memory_ = new FlatDemandMemory(0);
    state_ = new RiscVState("test", RiscVXlen::RV32, memory_);
    rv_vector_ = new RiscVVectorState(state_, kVectorLengthInBytes);
    instruction_ = new Instruction(kInstAddress, state_);
    instruction_->set_size(4);
    child_instruction_ = new Instruction(kInstAddress, state_);
    child_instruction_->set_size(4);
    // Initialize a portion of memory with a known pattern.
    auto* db = state_->db_factory()->Allocate(8192);
    auto span = db->Get<uint8_t>();
    for (int i = 0; i < 8192; i++) {
      span[i] = i & 0xff;
    }
    memory_->Store(kDataLoadAddress - 4096, db);
    db->DecRef();
    for (int i = 1; i < 32; i++) {
      xreg_[i] = state_->GetRegister<RV32Register>(absl::StrCat("x", i)).first;
    }
    for (int i = 1; i < 32; i++) {
      vreg_[i] =
          state_->GetRegister<RVVectorRegister>(absl::StrCat("v", i)).first;
    }
  }

  ~RV32VInstructionsTest() override {
    delete state_;
    delete rv_vector_;
    instruction_->DecRef();
    child_instruction_->DecRef();
    delete memory_;
  }

  // Creates immediate operands with the values from the vector and appends them
  // to the given instruction.
  template <typename T>
  void AppendImmediateOperands(Instruction* inst,
                               const std::vector<T>& values) {
    for (auto value : values) {
      auto* src = new ImmediateOperand<T>(value);
      inst->AppendSource(src);
    }
  }

  // Creates immediate operands with the values from the vector and appends them
  // to the default instruction.
  template <typename T>
  void AppendImmediateOperands(const std::vector<T>& values) {
    AppendImmediateOperands<T>(instruction_, values);
  }

  // Creates source and destination scalar register operands for the registers
  // named in the two vectors and append them to the given instruction.
  void AppendRegisterOperands(Instruction* inst,
                              const std::vector<std::string>& sources,
                              const std::vector<std::string>& destinations) {
    for (auto& reg_name : sources) {
      auto* reg = state_->GetRegister<RV32Register>(reg_name).first;
      inst->AppendSource(reg->CreateSourceOperand());
    }
    for (auto& reg_name : destinations) {
      auto* reg = state_->GetRegister<RV32Register>(reg_name).first;
      inst->AppendDestination(reg->CreateDestinationOperand(0));
    }
  }

  // Creates source and destination scalar register operands for the registers
  // named in the two vectors and append them to the default instruction.
  void AppendRegisterOperands(const std::vector<std::string>& sources,
                              const std::vector<std::string>& destinations) {
    AppendRegisterOperands(instruction_, sources, destinations);
  }

  // Returns the value of the named vector register.
  template <typename T>
  T GetRegisterValue(absl::string_view vreg_name) {
    auto* reg = state_->GetRegister<RV32Register>(vreg_name).first;
    return reg->data_buffer()->Get<T>();
  }

  // named register and sets it to the corresponding value.
  template <typename T>
  void SetRegisterValues(
      const std::vector<tuple<std::string, const T>>& values) {
    for (auto& [reg_name, value] : values) {
      auto* reg = state_->GetRegister<RV32Register>(reg_name).first;
      auto* db = state_->db_factory()->Allocate<RV32Register::ValueType>(1);
      db->Set<T>(0, value);
      reg->SetDataBuffer(db);
      db->DecRef();
    }
  }

  // Creates source and destination scalar register operands for the registers
  // named in the two vectors and append them to the given instruction.
  void AppendVectorRegisterOperands(Instruction* inst,
                                    const std::vector<int>& sources,
                                    const std::vector<int>& destinations) {
    for (auto& reg_no : sources) {
      std::vector<RegisterBase*> reg_vec;
      for (int i = 0; (i < 8) && (i + reg_no < 32); i++) {
        std::string reg_name = absl::StrCat("v", i + reg_no);
        reg_vec.push_back(
            state_->GetRegister<RVVectorRegister>(reg_name).first);
      }
      auto* op = new RV32VectorSourceOperand(absl::Span<RegisterBase*>(reg_vec),
                                             absl::StrCat("v", reg_no));
      inst->AppendSource(op);
    }
    for (auto& reg_no : destinations) {
      std::vector<RegisterBase*> reg_vec;
      for (int i = 0; (i < 8) && (i + reg_no < 32); i++) {
        std::string reg_name = absl::StrCat("v", i + reg_no);
        reg_vec.push_back(
            state_->GetRegister<RVVectorRegister>(reg_name).first);
      }
      auto* op = new RV32VectorDestinationOperand(
          absl::Span<RegisterBase*>(reg_vec), 0, absl::StrCat("v", reg_no));
      inst->AppendDestination(op);
    }
  }
  // Creates source and destination scalar register operands for the registers
  // named in the two vectors and append them to the default instruction.
  void AppendVectorRegisterOperands(const std::vector<int>& sources,
                                    const std::vector<int>& destinations) {
    AppendVectorRegisterOperands(instruction_, sources, destinations);
  }

  // Returns the value of the named vector register.
  template <typename T>
  T GetVectorRegisterValue(absl::string_view reg_name) {
    auto* reg = state_->GetRegister<RVVectorRegister>(reg_name).first;
    return reg->data_buffer()->Get<T>(0);
  }

  // Set a vector register value. Takes a vector of tuples of register names and
  // spans of values, fetches each register and sets it to the corresponding
  // value.
  template <typename T>
  void SetVectorRegisterValues(
      const std::vector<tuple<std::string, Span<const T>>>& values) {
    for (auto& [vreg_name, span] : values) {
      auto* vreg = state_->GetRegister<RVVectorRegister>(vreg_name).first;
      auto* db = state_->db_factory()->MakeCopyOf(vreg->data_buffer());
      db->template Set<T>(span);
      vreg->SetDataBuffer(db);
      db->DecRef();
    }
  }

  // Initializes the semantic function of the instruction object.
  void SetSemanticFunction(Instruction* inst,
                           Instruction::SemanticFunction fcn) {
    inst->set_semantic_function(fcn);
  }

  // Initializes the semantic function for the default instruction.
  void SetSemanticFunction(Instruction::SemanticFunction fcn) {
    instruction_->set_semantic_function(fcn);
  }

  // Sets the default child instruction as the child of the default instruction.
  void SetChildInstruction() { instruction_->AppendChild(child_instruction_); }

  // Initializes the semantic function for the default child instruction.
  void SetChildSemanticFunction(Instruction::SemanticFunction fcn) {
    child_instruction_->set_semantic_function(fcn);
  }

  // Configure the vector unit according to the vtype and vlen values.
  void ConfigureVectorUnit(uint32_t vtype, uint32_t vlen) {
    Instruction* inst = new Instruction(state_);
    AppendImmediateOperands<uint32_t>(inst, {vlen, vtype});
    SetSemanticFunction(inst, absl::bind_front(&Vsetvl, true, false));
    inst->Execute(nullptr);
    inst->DecRef();
  }

  template <typename T>
  T ComputeValue(int address) {
    T value = 0;
    uint8_t* ptr = reinterpret_cast<uint8_t*>(&value);
    for (int j = 0; j < sizeof(T); j++) {
      ptr[j] = (address + j) & 0xff;
    }
    return value;
  }

  template <typename T>
  void VectorLoadUnitStridedHelper() {
    // Set up instructions.
    AppendRegisterOperands({kRs1Name, kRs2Name}, {});
    AppendVectorRegisterOperands({kVmask}, {});
    SetSemanticFunction(absl::bind_front(&VlStrided,
                                         /*element_width*/ sizeof(T)));
    // Add the child instruction that performs the register write-back.
    SetChildInstruction();
    SetChildSemanticFunction(&VlChild);
    AppendVectorRegisterOperands(child_instruction_, {}, {kVd});
    // Set up register values.
    SetRegisterValues<uint32_t>({{kRs1Name, kDataLoadAddress}});
    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});
    // Iterate over different lmul values.
    for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
      SetRegisterValues<int32_t>({{kRs2Name, sizeof(T)}});
      uint32_t vtype =
          (kSewSettingsByByteSize[sizeof(T)] << 3) | kLmulSettings[lmul_index];
      int lmul8 = kLmul8Values[lmul_index];
      int num_values = kVectorLengthInBytes * lmul8 / (sizeof(T) * 8);
      ConfigureVectorUnit(vtype, /*vlen*/ 1024);
      // Execute instruction.
      instruction_->Execute(nullptr);

      // Check register values.
      int count = 0;
      for (int reg = kVd; reg < kVd + 8; reg++) {
        auto span = vreg_[reg]->data_buffer()->Get<T>();
        for (int i = 0; i < kVectorLengthInBytes / sizeof(T); i++) {
          int mask_index = count / 8;
          int mask_offset = count % 8;
          bool mask = (kA5Mask[mask_index] >> mask_offset) & 0x1;
          if (mask && (count < num_values)) {
            // First compute the expected value, then compare it.
            T value = ComputeValue<T>(4096 + count * sizeof(T));
            EXPECT_EQ(value, span[i])
                << "element size " << sizeof(T) << " LMUL8 " << lmul8
                << " Count " << count << " Reg " << reg << " value " << i;
          } else {
            // The remainder of the values should be zero.
            EXPECT_EQ(0, span[i])
                << "element size " << sizeof(T) << " LMUL8 " << lmul8
                << " Count " << count << " Reg " << reg << " value " << i;
          }
          count++;
        }
      }
    }
  }

  template <typename T>
  void VectorLoadStridedHelper() {
    const int strides[5] = {1, 4, 0, -1, -3};
    // Set up instructions.
    AppendRegisterOperands({kRs1Name, kRs2Name}, {});
    AppendVectorRegisterOperands({kVmask}, {});
    SetSemanticFunction(absl::bind_front(&VlStrided,
                                         /*element_width*/ sizeof(T)));
    // Add the child instruction that performs the register write-back.
    SetChildInstruction();
    SetChildSemanticFunction(&VlChild);
    AppendVectorRegisterOperands(child_instruction_, {}, {kVd});
    // Set up register values.
    SetRegisterValues<uint32_t>({{kRs1Name, kDataLoadAddress}});
    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});
    // Iterate over different lmul values.
    for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
      // Try different strides.
      for (int s = 0; s < 5; s++) {
        int32_t stride = strides[s];
        SetRegisterValues<int32_t>({{kRs2Name, stride}});
        // Configure vector unit.
        uint32_t vtype = (kSewSettingsByByteSize[sizeof(T)] << 3) |
                         kLmulSettings[lmul_index];
        int lmul8 = kLmul8Values[lmul_index];
        int num_values = kVectorLengthInBytes * lmul8 / (sizeof(T) * 8);
        ConfigureVectorUnit(vtype, /*vlen*/ 1024);
        // Execute instruction.
        instruction_->Execute(nullptr);

        // Check register values.
        int count = 0;
        for (int reg = kVd; reg < kVd + 8; reg++) {
          auto span = vreg_[reg]->data_buffer()->Get<T>();
          for (int i = 0; i < kVectorLengthInBytes / sizeof(T); i++) {
            int mask_index = count / 8;
            int mask_offset = count % 8;
            bool mask = (kA5Mask[mask_index] >> mask_offset) & 0x1;
            if (mask && (count < num_values)) {
              // First compute the expected value, then compare it.
              T value = ComputeValue<T>(4096 + count * stride);
              EXPECT_EQ(value, span[i])
                  << "element size " << sizeof(T) << " stride: " << stride
                  << " LMUL8 " << lmul8 << " Count " << count << " Reg " << reg
                  << " value " << i;
            } else {
              // The remainder of the values should be zero.
              EXPECT_EQ(0, span[i])
                  << "element size " << sizeof(T) << " stride: " << stride
                  << " LMUL8 " << lmul8 << " Count " << count << " Reg " << reg
                  << " value " << i;
            }
            count++;
          }
        }
      }
    }
  }

  template <typename T>
  T IndexValue(int i) {
    T offset = ~i & 0b1111;
    T val = (i & ~0b1111) | offset;
    return val * sizeof(T);
  }

  // Helper function for testing vector load indexed instructions.
  template <typename IndexType, typename ValueType>
  void VectorLoadIndexedHelper() {
    // Set up instructions.
    AppendRegisterOperands({kRs1Name}, {});
    AppendVectorRegisterOperands({kVs2, kVmask}, {});
    SetSemanticFunction(absl::bind_front(&VlIndexed,
                                         /*index_width*/ sizeof(IndexType)));
    // Add the child instruction that performs the register write-back.
    SetChildInstruction();
    SetChildSemanticFunction(&VlChild);
    AppendVectorRegisterOperands(child_instruction_, {}, {kVd});
    // Set up register values.
    SetRegisterValues<uint32_t>({{kRs1Name, kDataLoadAddress}});
    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});

    // Iterate over different lmul values.
    for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
      // Configure vector unit.
      uint32_t vtype = (kSewSettingsByByteSize[sizeof(ValueType)] << 3) |
                       kLmulSettings[lmul_index];
      ConfigureVectorUnit(vtype, /*vlen*/ 1024);
      int lmul8 = kLmul8Values[lmul_index];
      int index_emul8 = lmul8 * sizeof(IndexType) / sizeof(ValueType);

      if ((index_emul8 == 0) || (index_emul8 > 64)) {
        // The index vector length is illegal.
        instruction_->Execute(nullptr);
        EXPECT_TRUE(rv_vector_->vector_exception());
        rv_vector_->clear_vector_exception();
        continue;
      }
      EXPECT_FALSE(rv_vector_->vector_exception());

      int num_values = kVectorLengthInBytes * lmul8 / (sizeof(ValueType) * 8);
      // Set up index vector values.
      int values_per_reg = kVectorLengthInBytes / sizeof(IndexType);
      for (int i = 0; i < num_values; i++) {
        int reg_index = kVs2 + i / values_per_reg;
        int reg_offset = i % values_per_reg;
        auto index_span = vreg_[reg_index]->data_buffer()->Get<IndexType>();
        index_span[reg_offset] = IndexValue<ValueType>(i);
      }
      // Execute instruction.
      instruction_->Execute(nullptr);

      // Check register values.
      int count = 0;
      for (int reg = kVd; reg < kVd + 8; reg++) {
        auto span = vreg_[reg]->data_buffer()->Get<ValueType>();
        for (int i = 0; i < kVectorLengthInBytes / sizeof(ValueType); i++) {
          int mask_index = count / 8;
          int mask_offset = count % 8;
          bool mask = (kA5Mask[mask_index] >> mask_offset) & 0x1;
          if (mask && (count < num_values)) {
            // Compare expected value.
            auto value =
                ComputeValue<ValueType>(4096 + IndexValue<ValueType>(count));
            EXPECT_EQ(value, span[i])
                << "element size " << sizeof(ValueType) << " index: " << index
                << " LMUL8 " << lmul8 << " Count " << count << " reg " << reg;
          } else {
            // The remainder of the values should be zero.
            EXPECT_EQ(0, span[i])
                << "element size " << sizeof(ValueType) << " index: " << index
                << " LMUL8 " << lmul8 << " count " << count << " reg " << reg;
          }
          count++;
        }
      }
    }
  }

  // Helper function to test vector load segment strided instructions.
  template <typename T>
  void VectorLoadSegmentHelper() {
    // Set up instructions.
    AppendRegisterOperands({kRs1Name}, {});
    AppendVectorRegisterOperands({kVmask}, {});
    AppendRegisterOperands({kRs3Name}, {});
    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});
    SetSemanticFunction(absl::bind_front(&VlSegment,
                                         /*element_width*/ sizeof(T)));
    // Add the child instruction that performs the register write-back.
    SetChildInstruction();
    SetChildSemanticFunction(
        absl::bind_front(&VlSegmentChild, /*element_width*/ sizeof(T)));
    AppendRegisterOperands(child_instruction_, {kRs3Name}, {});
    AppendVectorRegisterOperands(child_instruction_, {}, {kVd});
    // Set up register values.
    SetRegisterValues<uint32_t>({{kRs1Name, kDataLoadAddress}});
    // Iterate over legal values in the nf field.
    for (int nf = 1; nf < 8; nf++) {
      int num_fields = nf + 1;
      // Iterate over different lmul values.
      SetRegisterValues<int32_t>({{kRs3Name, nf}});
      for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
        // Configure vector unit, set the sew to the element width.
        uint32_t vtype = (kSewSettingsByByteSize[sizeof(T)] << 3) |
                         kLmulSettings[lmul_index];
        int lmul8 = kLmul8Values[lmul_index];
        int num_values = kVectorLengthInBytes * lmul8 / (sizeof(T) * 8);
        ConfigureVectorUnit(vtype, /*vlen*/ 1024);
        // Execute instruction.
        instruction_->Execute(nullptr);

        if (lmul8 * num_fields > 64) {
          EXPECT_TRUE(rv_vector_->vector_exception());
          rv_vector_->clear_vector_exception();
          continue;
        }
        EXPECT_FALSE(rv_vector_->vector_exception());

        // Check register values.
        int count = 0;
        // Fields are in consecutive (groups of) registers. First compute the
        // number of registers for each field.
        int regs_per_field = ::std::max(1, lmul8 / 8);
        for (int field = 0; field < num_fields; field++) {
          int start_reg = kVd + field * regs_per_field;
          int max_reg = start_reg + regs_per_field;
          count = 0;
          for (int reg = start_reg; reg < max_reg; reg++) {
            auto span = vreg_[reg]->data_buffer()->Get<T>();
            int num_reg_elements =
                std::min(kVectorLengthInBytes / sizeof(T),
                         kVectorLengthInBytes * lmul8 / (sizeof(T) * 8));
            for (int i = 0; i < num_reg_elements; i++) {
              int mask_index = count / 8;
              int mask_offset = count % 8;
              bool mask = (kA5Mask[mask_index] >> mask_offset) & 0x1;
              if (mask && (count < num_values)) {
                int address =
                    4096 + count * sizeof(T) * num_fields + field * sizeof(T);
                T value = ComputeValue<T>(address);
                EXPECT_EQ(value, span[i])
                    << "element size " << sizeof(T) << " LMUL8 " << lmul8
                    << " Count " << count << " Reg " << reg << " value " << i;
              } else {
                // The remainder of the values should be zero.
                EXPECT_EQ(0, span[i])
                    << "element size " << sizeof(T) << " LMUL8 " << lmul8
                    << " Count " << count << " Reg " << reg << " value " << i;
              }
              count++;
            }
          }
        }
      }
    }
  }

  // Helper function to test vector load segment strided instructions.
  template <typename T>
  void VectorLoadStridedSegmentHelper() {
    const int strides[5] = {1, 4, 0, -1, -3};
    // Set up instructions.
    // Base address and stride.
    AppendRegisterOperands({kRs1Name, kRs2Name}, {});
    // Vector mask register.
    AppendVectorRegisterOperands({kVmask}, {});
    // Operand to hold the number of fields.
    AppendRegisterOperands({kRs3Name}, {});
    // Initialize the mask.
    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});
    // Bind semantic function.
    SetSemanticFunction(absl::bind_front(&VlSegmentStrided,
                                         /*element_width*/ sizeof(T)));
    // Add the child instruction that performs the register write-back.
    SetChildInstruction();
    SetChildSemanticFunction(
        absl::bind_front(&VlSegmentChild, /*element_width*/ sizeof(T)));
    // Number of fields.
    AppendRegisterOperands(child_instruction_, {kRs3Name}, {});
    // Destination vector register operand.
    AppendVectorRegisterOperands(child_instruction_, {}, {kVd});

    // Set up register values.
    // Base address.
    SetRegisterValues<uint32_t>({{kRs1Name, kDataLoadAddress}});
    // Iterate over legal values in the nf field.
    for (int nf = 1; nf < 8; nf++) {
      int num_fields = nf + 1;
      // Set the number of fields.
      SetRegisterValues<int32_t>({{kRs3Name, nf}});
      // Iterate over different lmul values.
      for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
        // Configure vector unit, set the sew to the element width.
        uint32_t vtype = (kSewSettingsByByteSize[sizeof(T)] << 3) |
                         kLmulSettings[lmul_index];
        int lmul8 = kLmul8Values[lmul_index];
        int num_values = kVectorLengthInBytes * lmul8 / (sizeof(T) * 8);
        ConfigureVectorUnit(vtype, /*vlen*/ 1024);

        // Try different strides.
        for (int s = 0; s < 5; s++) {
          int32_t stride = strides[s] * num_fields * sizeof(T);
          // Set the stride.
          SetRegisterValues<int32_t>({{kRs2Name, stride}});
          // Execute instruction.
          instruction_->Execute(nullptr);

          if (lmul8 * num_fields > 64) {
            EXPECT_TRUE(rv_vector_->vector_exception());
            rv_vector_->clear_vector_exception();
            continue;
          }
          EXPECT_FALSE(rv_vector_->vector_exception());

          // Check register values.
          // Fields are in consecutive (groups of) registers. First compute the
          // number of registers for each field.
          int regs_per_field = ::std::max(1, lmul8 / 8);
          for (int field = 0; field < num_fields; field++) {
            int start_reg = kVd + field * regs_per_field;
            int max_reg = start_reg + regs_per_field;
            int count = 0;
            for (int reg = start_reg; reg < max_reg; reg++) {
              auto span = vreg_[reg]->data_buffer()->Get<T>();
              int num_reg_elements =
                  std::min(kVectorLengthInBytes / sizeof(T),
                           kVectorLengthInBytes * lmul8 / (sizeof(T) * 8));
              for (int i = 0; i < num_reg_elements; i++) {
                int mask_index = count / 8;
                int mask_offset = count % 8;
                bool mask = (kA5Mask[mask_index] >> mask_offset) & 0x1;
                if (mask && (count < num_values)) {
                  // First compute the expected value, then compare it.
                  int address = 4096 + stride * count + field * sizeof(T);
                  T value = ComputeValue<T>(address);

                  EXPECT_EQ(value, span[i])
                      << "element size " << sizeof(T) << " stride: " << stride
                      << " LMUL8 " << lmul8 << " Count " << count << " Reg "
                      << reg << " value " << i;
                } else {
                  // The remainder of the values should be zero.
                  EXPECT_EQ(0, span[i])
                      << "element size " << sizeof(T) << " stride: " << stride
                      << " LMUL8 " << lmul8 << " Count " << count << " Reg "
                      << reg << " value " << i;
                }
                count++;
              }
            }
          }
        }
      }
    }
  }

  // Helper function to test vector load segment indexed instructions.
  template <typename IndexType, typename ValueType>
  void VectorLoadIndexedSegmentHelper() {
    // Set up instructions.
    // Base address and stride.
    AppendRegisterOperands({kRs1Name}, {});
    // Vector mask register.
    AppendVectorRegisterOperands({kVs2, kVmask}, {});
    // Operand to hold the number of fields.
    AppendRegisterOperands({kRs3Name}, {});
    // Initialize the mask.
    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});
    // Bind semantic function.
    SetSemanticFunction(absl::bind_front(&VlSegmentIndexed,
                                         /*element_width*/ sizeof(IndexType)));
    // Add the child instruction that performs the register write-back.
    SetChildInstruction();
    SetChildSemanticFunction(
        absl::bind_front(&VlSegmentChild,
                         /*element_width*/ sizeof(ValueType)));
    // Number of fields.
    AppendRegisterOperands(child_instruction_, {kRs3Name}, {});
    // Destination vector register operand.
    AppendVectorRegisterOperands(child_instruction_, {}, {kVd});

    // Set up register values.
    // Base address.
    SetRegisterValues<uint32_t>({{kRs1Name, kDataLoadAddress}});
    // Iterate over legal values in the nf field.
    for (int nf = 1; nf < 8; nf++) {
      int num_fields = nf + 1;
      // Set the number of fields.
      SetRegisterValues<int32_t>({{kRs3Name, nf}});
      // Iterate over different lmul values.
      for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
        rv_vector_->clear_vector_exception();
        // Configure vector unit, set the sew to the element width.
        uint32_t vtype = (kSewSettingsByByteSize[sizeof(ValueType)] << 3) |
                         kLmulSettings[lmul_index];
        int lmul8 = kLmul8Values[lmul_index];
        ConfigureVectorUnit(vtype, /*vlen*/ 1024);
        int index_emul8 = lmul8 * sizeof(IndexType) / sizeof(ValueType);
        int num_values = kVectorLengthInBytes * lmul8 / (sizeof(ValueType) * 8);

        if ((index_emul8 == 0) || (index_emul8 > 64)) {
          // The index vector length is illegal.
          instruction_->Execute(nullptr);
          EXPECT_TRUE(rv_vector_->vector_exception());
          continue;
        }
        if (lmul8 * num_fields > 64) {
          instruction_->Execute(nullptr);
          EXPECT_TRUE(rv_vector_->vector_exception());
          continue;
        }

        // Set up index vector values.
        int values_per_reg = kVectorLengthInBytes / sizeof(IndexType);
        for (int i = 0; i < num_values; i++) {
          int reg_index = kVs2 + i / values_per_reg;
          int reg_offset = i % values_per_reg;
          auto index_span = vreg_[reg_index]->data_buffer()->Get<IndexType>();
          index_span[reg_offset] = IndexValue<ValueType>(i);
        }

        // Execute instruction.
        instruction_->Execute(nullptr);
        EXPECT_FALSE(rv_vector_->vector_exception());

        // Check register values.
        // Fields are in consecutive (groups of) registers. First compute the
        // number of registers for each field.
        int regs_per_field = ::std::max(1, lmul8 / 8);
        for (int field = 0; field < num_fields; field++) {
          int start_reg = kVd + field * regs_per_field;
          int max_reg = start_reg + regs_per_field;
          int count = 0;
          for (int reg = start_reg; reg < max_reg; reg++) {
            auto span = vreg_[reg]->data_buffer()->Get<ValueType>();
            int num_reg_elements = std::min(
                kVectorLengthInBytes / sizeof(ValueType),
                kVectorLengthInBytes * lmul8 / (sizeof(ValueType) * 8));
            for (int i = 0; i < num_reg_elements; i++) {
              int mask_index = count / 8;
              int mask_offset = count % 8;
              bool mask = (kA5Mask[mask_index] >> mask_offset) & 0x1;
              if (mask && (count < num_values)) {
                // First compute the expected value, then compare it.
                int address = 4096 + IndexValue<IndexType>(count) +
                              field * sizeof(ValueType);
                ValueType value = ComputeValue<ValueType>(address);

                EXPECT_EQ(value, span[i])
                    << "element size " << sizeof(ValueType) << " LMUL8 "
                    << lmul8 << " Count " << count << " Reg " << reg
                    << " value " << i;
              } else {
                // The remainder of the values should be zero.
                EXPECT_EQ(0, span[i])
                    << "element size " << sizeof(ValueType) << " LMUL8 "
                    << lmul8 << " Count " << count << " Reg " << reg
                    << " value " << i;
              }
              count++;
            }
          }
        }
      }
    }
  }

  template <typename T>
  void VectorStoreStridedHelper() {
    const int strides[5] = {1, 4, 8, -1, -3};
    // Set up instructions.
    AppendVectorRegisterOperands({kVs1}, {});
    AppendRegisterOperands({kRs1Name, kRs2Name}, {});
    AppendVectorRegisterOperands({kVmask}, {});
    SetSemanticFunction(absl::bind_front(&VsStrided,
                                         /*element_width*/ sizeof(T)));
    // Set up register values.
    SetRegisterValues<uint32_t>({{kRs1Name, kDataStoreAddress}});
    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});
    // Set the store data register elements to be consecutive integers.
    for (int reg = 0; reg < 8; reg++) {
      auto reg_span = vreg_[reg + kVs1]->data_buffer()->Get<T>();
      for (int i = 0; i < reg_span.size(); i++) {
        reg_span[i] = static_cast<T>(reg * reg_span.size() + i + 1);
      }
    }
    // Iterate over different lmul values.
    for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
      // Try different strides.
      for (int s = 0; s < 5; s++) {
        int32_t stride = strides[s] * sizeof(T);
        SetRegisterValues<int32_t>({{kRs2Name, stride}});
        // Configure vector unit.
        uint32_t vtype = (kSewSettingsByByteSize[sizeof(T)] << 3) |
                         kLmulSettings[lmul_index];
        int lmul8 = kLmul8Values[lmul_index];
        int num_values = kVectorLengthInBytes * lmul8 / (sizeof(T) * 8);
        ConfigureVectorUnit(vtype, /*vlen*/ 1024);
        // Execute instruction.
        instruction_->Execute(nullptr);

        // Check memory values.
        auto* data_db = state_->db_factory()->Allocate<T>(1);
        uint64_t base = kDataStoreAddress;
        T value = 1;
        for (int i = 0; i < 8 * kVectorLengthInBytes / sizeof(T); i++) {
          data_db->template Set<T>(0, 0);
          state_->LoadMemory(instruction_, base, data_db, nullptr, nullptr);
          int mask_index = i / 8;
          int mask_offset = i % 8;
          bool mask = (kA5Mask[mask_index] >> mask_offset) & 0x1;
          if (mask && (i < num_values)) {
            EXPECT_EQ(data_db->template Get<T>(0), static_cast<T>(value))
                << "index: " << i << " element_size: " << sizeof(T)
                << " lmul8: " << lmul8 << " stride: " << stride;
          } else {
            EXPECT_EQ(data_db->template Get<T>(0), 0) << "index: " << i;
          }
          base += stride;
          value++;
        }
        data_db->DecRef();
        // Clear memory.
        data_db = state_->db_factory()->Allocate(0x4000);
        memset(data_db->raw_ptr(), 0, 0x4000);
        state_->StoreMemory(instruction_, kDataStoreAddress - 0x2000, data_db);
        data_db->DecRef();
      }
    }
  }

  template <typename IndexType, typename ValueType>
  void VectorStoreIndexedHelper() {
    // Set up instructions.
    AppendVectorRegisterOperands({kVs1}, {});
    AppendRegisterOperands({kRs1Name}, {});
    AppendVectorRegisterOperands({kVs2, kVmask}, {});
    SetSemanticFunction(absl::bind_front(&VsIndexed,
                                         /*index_width*/ sizeof(IndexType)));

    // Set up register values.
    SetRegisterValues<uint32_t>({{kRs1Name, kDataStoreAddress}});
    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});

    int values_per_reg = kVectorLengthInBytes / sizeof(ValueType);
    int index_values_per_reg = kVectorLengthInBytes / sizeof(IndexType);
    for (int reg = 0; reg < 8; reg++) {
      for (int i = 0; i < values_per_reg; i++) {
        vreg_[kVs1 + reg]->data_buffer()->Set<ValueType>(
            i, reg * values_per_reg + i);
      }
    }

    auto* data_db = state_->db_factory()->Allocate<ValueType>(1);
    // Iterate over different lmul values.
    for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
      // Configure vector unit.
      uint32_t vtype = (kSewSettingsByByteSize[sizeof(ValueType)] << 3) |
                       kLmulSettings[lmul_index];
      ConfigureVectorUnit(vtype, /*vlen*/ 1024);
      int lmul8 = kLmul8Values[lmul_index];
      int index_emul8 = lmul8 * sizeof(IndexType) / sizeof(ValueType);
      int num_values = kVectorLengthInBytes * lmul8 / (sizeof(ValueType) * 8);

      // Skip if the number of values is greater than the offset representation.
      // This only happens for uint8_t.
      if (num_values > 256) continue;

      // Check the index vector length.
      if ((index_emul8 == 0) || (index_emul8 > 64)) {
        // The index vector length is illegal.
        instruction_->Execute(nullptr);
        EXPECT_TRUE(rv_vector_->vector_exception());
        rv_vector_->clear_vector_exception();
        continue;
      }

      for (int i = 0; i < num_values; i++) {
        int reg = i / index_values_per_reg;
        int element = i % index_values_per_reg;
        vreg_[kVs2 + reg]->data_buffer()->Set<IndexType>(
            element, IndexValue<IndexType>(i));
      }

      // Execute instruction.
      instruction_->Execute(nullptr);

      // Check results.
      EXPECT_FALSE(rv_vector_->vector_exception());

      // Check register values.
      for (int i = 0; i < 8 * kVectorLengthInBytes / sizeof(ValueType); i++) {
        uint64_t address = kDataStoreAddress + IndexValue<IndexType>(i);
        state_->LoadMemory(instruction_, address, data_db, nullptr, nullptr);
        int mask_index = i / 8;
        int mask_offset = i % 8;
        bool mask = (kA5Mask[mask_index] >> mask_offset) & 0x1;
        if (mask && (i < num_values)) {
          EXPECT_EQ(data_db->template Get<ValueType>(0), i)
              << "reg[" << i / values_per_reg << "][" << i % values_per_reg
              << "]";
        } else {
          EXPECT_EQ(data_db->template Get<ValueType>(0), 0)
              << "reg[" << i / values_per_reg << "][" << i % values_per_reg
              << "]";
        }
        // Clear the memory location.
        data_db->template Set<ValueType>(0, 0);
        state_->StoreMemory(instruction_, address, data_db);
      }
    }
    data_db->DecRef();
  }

  // Helper function to test vector load segment strided instructions.
  template <typename T>
  void VectorStoreSegmentHelper() {
    // Set up instructions.
    // Store data register.
    AppendVectorRegisterOperands({kVs1}, {});
    // Base address and stride.
    AppendRegisterOperands({kRs1Name}, {});
    // Vector mask register.
    AppendVectorRegisterOperands({kVmask}, {});
    // Operand to hold the number of fields.
    AppendRegisterOperands({kRs3Name}, {});
    // Bind semantic function.
    SetSemanticFunction(absl::bind_front(&VsSegment,
                                         /*element_width*/ sizeof(T)));

    // Set up register values.
    // Set the store data register elements to be consecutive integers.
    for (int reg = 0; reg < 8; reg++) {
      auto reg_span = vreg_[reg + kVs1]->data_buffer()->Get<T>();
      for (int i = 0; i < reg_span.size(); i++) {
        reg_span[i] = static_cast<T>(reg * reg_span.size() + i + 1);
      }
    }
    // Base address.
    SetRegisterValues<uint32_t>({{kRs1Name, kDataStoreAddress}});
    // Initialize the mask.
    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});

    int num_values_per_register = kVectorLengthInBytes / sizeof(T);
    // Can load all the data in one load, so set the data_db size accordingly.
    auto* data_db =
        state_->db_factory()->Allocate<uint8_t>(kVectorLengthInBytes * 8);
    // Iterate over legal values in the nf field.
    for (int nf = 1; nf < 8; nf++) {
      int num_fields = nf + 1;
      // Set the number of fields in the source operand.
      SetRegisterValues<int32_t>({{kRs3Name, nf}});
      // Iterate over different lmul values.
      for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
        // Configure vector unit, set the sew to the element width.
        uint32_t vtype = (kSewSettingsByByteSize[sizeof(T)] << 3) |
                         kLmulSettings[lmul_index];
        int lmul8 = kLmul8Values[lmul_index];
        ConfigureVectorUnit(vtype, /*vlen*/ 1024);
        // Clear the memory.
        uint64_t base = kDataStoreAddress;
        std::memset(data_db->raw_ptr(), 0, data_db->template size<uint8_t>());
        state_->StoreMemory(instruction_, base, data_db);

        // Execute instruction.
        instruction_->Execute(nullptr);

        if (lmul8 * num_fields > 64) {
          EXPECT_TRUE(rv_vector_->vector_exception());
          rv_vector_->clear_vector_exception();
          continue;
        }
        int emul = sizeof(T) * lmul8 / rv_vector_->selected_element_width();
        if (emul == 0 || emul > 64) {
          EXPECT_TRUE(rv_vector_->vector_exception());
          rv_vector_->clear_vector_exception();
          continue;
        }
        EXPECT_FALSE(rv_vector_->vector_exception());
        // Check memory values.
        T value = 1;
        int vlen = rv_vector_->vector_length();
        int num_regs = std::max(1, lmul8 / 8);
        // Load the store data.
        state_->LoadMemory(instruction_, base, data_db, nullptr, nullptr);
        // Iterate over fields.
        for (int field = 0; field < num_fields; field++) {
          // Iterate over the registers used for each field.
          int segment_no = 0;
          for (int reg = 0; reg < num_regs; reg++) {
            // Iterate over segments within each register.
            for (int i = 0; i < num_values_per_register; i++) {
              // Get the mask value.
              int mask_index = segment_no >> 3;
              int mask_offset = segment_no & 0b111;
              bool mask = (kA5Mask[mask_index] >> mask_offset) & 0x1;
              T mem_value =
                  data_db->template Get<T>(segment_no * num_fields + field);
              if (mask && (segment_no < vlen)) {
                EXPECT_EQ(mem_value, static_cast<T>(value))
                    << " segment_no: " << segment_no << " field: " << field
                    << " index: " << i << " element_size: " << sizeof(T)
                    << " lmul8: " << lmul8 << " mask: " << mask;
                // Zero the memory location.
                data_db->template Set<T>(0, 0);
              } else {
                EXPECT_EQ(mem_value, 0)
                    << " segment_no: " << segment_no << " field: " << field
                    << " index: " << i << " element_size: " << sizeof(T)
                    << " lmul8: " << lmul8 << " mask: " << mask;
              }
              value++;
              segment_no++;
            }
          }
        }
        if (HasFailure()) {
          data_db->DecRef();
          return;
        }
      }
    }
    data_db->DecRef();
  }

  // Helper function to test vector load segment strided instructions.
  template <typename T>
  void VectorStoreStridedSegmentHelper() {
    const int strides[5] = {1, 4, 8, -1, -3};
    // Set up instructions.
    // Store data register.
    AppendVectorRegisterOperands({kVs1}, {});
    // Base address and stride.
    AppendRegisterOperands({kRs1Name, kRs2Name}, {});
    // Vector mask register.
    AppendVectorRegisterOperands({kVmask}, {});
    // Operand to hold the number of fields.
    AppendRegisterOperands({kRs3Name}, {});
    // Bind semantic function.
    SetSemanticFunction(absl::bind_front(&VsSegmentStrided,
                                         /*element_width*/ sizeof(T)));

    // Set up register values.
    // Set the store data register elements to be consecutive integers.
    for (int reg = 0; reg < 8; reg++) {
      auto reg_span = vreg_[reg + kVs1]->data_buffer()->Get<T>();
      for (int i = 0; i < reg_span.size(); i++) {
        reg_span[i] = static_cast<T>(reg * reg_span.size() + i + 1);
      }
    }
    // Base address.
    SetRegisterValues<uint32_t>({{kRs1Name, kDataStoreAddress}});
    // Initialize the mask.
    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});

    int num_values_per_register = kVectorLengthInBytes / sizeof(T);

    auto* data_db = state_->db_factory()->Allocate<T>(1);
    // Iterate over legal values in the nf field.
    for (int nf = 1; nf < 8; nf++) {
      int num_fields = nf + 1;
      // Set the number of fields in the source operand.
      SetRegisterValues<int32_t>({{kRs3Name, nf}});
      // Iterate over different lmul values.
      for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
        // Configure vector unit, set the sew to the element width.
        uint32_t vtype = (kSewSettingsByByteSize[sizeof(T)] << 3) |
                         kLmulSettings[lmul_index];
        int lmul8 = kLmul8Values[lmul_index];
        ConfigureVectorUnit(vtype, /*vlen*/ 1024);

        // Try different strides.
        for (int s : strides) {
          int32_t stride = s * num_fields * sizeof(T);
          // Set the stride.
          SetRegisterValues<int32_t>({{kRs2Name, stride}});
          // Execute instruction.
          instruction_->Execute(nullptr);

          if (lmul8 * num_fields > 64) {
            EXPECT_TRUE(rv_vector_->vector_exception());
            rv_vector_->clear_vector_exception();
            continue;
          }
          int emul = sizeof(T) * lmul8 / rv_vector_->selected_element_width();
          if (emul == 0 || emul > 64) {
            EXPECT_TRUE(rv_vector_->vector_exception());
            rv_vector_->clear_vector_exception();
            continue;
          }
          EXPECT_FALSE(rv_vector_->vector_exception());
          // Check memory values.
          uint64_t base = kDataStoreAddress;
          T value = 1;
          int vlen = rv_vector_->vector_length();
          int num_regs = std::max(1, lmul8 / 8);
          // Iterate over fields.
          for (int field = 0; field < num_fields; field++) {
            uint64_t address = base + field * sizeof(T);
            // Iterate over the registers used for each field.
            int segment_no = 0;
            for (int reg = 0; reg < num_regs; reg++) {
              // Iterate over segments within each register.
              for (int i = 0; i < num_values_per_register; i++) {
                // Load the data.
                data_db->template Set<T>(0, 0);
                state_->LoadMemory(instruction_, address, data_db, nullptr,
                                   nullptr);
                // Get the mask value.
                int mask_index = segment_no >> 3;
                int mask_offset = segment_no & 0b111;
                bool mask = (kA5Mask[mask_index] >> mask_offset) & 0x1;
                if (mask && (segment_no < vlen)) {
                  EXPECT_EQ(data_db->template Get<T>(0), static_cast<T>(value))
                      << std::hex << "address: 0x" << address << std::dec
                      << " segment_no: " << segment_no << " field: " << field
                      << " index: " << i << " element_size: " << sizeof(T)
                      << " lmul8: " << lmul8 << " stride: " << stride;
                  // Zero the memory location.
                  data_db->template Set<T>(0, 0);
                  state_->StoreMemory(instruction_, address, data_db);
                } else {
                  EXPECT_EQ(data_db->template Get<T>(0), 0)
                      << std::hex << "address: 0x" << address << std::dec
                      << " segment_no: " << segment_no << " field: " << field
                      << " index: " << i << " element_size: " << sizeof(T)
                      << " lmul8: " << lmul8 << " stride: " << stride
                      << " mask: " << mask;
                }
                value++;
                address += stride;
                segment_no++;
              }
            }
          }
          if (HasFailure()) {
            data_db->DecRef();
            return;
          }
        }
      }
    }
    data_db->DecRef();
  }

  // Helper function to test vector load segment strided instructions.
  template <typename T, typename I>
  void VectorStoreIndexedSegmentHelper() {
    // Ensure that the IndexType is signed.
    using IndexType = typename std::make_signed<I>::type;
    // Set up instructions.
    // Store data register.
    AppendVectorRegisterOperands({kVs1}, {});
    // Base address and stride.
    AppendRegisterOperands({kRs1Name}, {});
    // Vector index register and vector mask register.
    AppendVectorRegisterOperands({kVs2, kVmask}, {});
    // Operand to hold the number of fields.
    AppendRegisterOperands({kRs3Name}, {});
    // Bind semantic function.
    SetSemanticFunction(
        absl::bind_front(&VsSegmentIndexed, /*index_width*/ sizeof(IndexType)));

    // Set up register values.
    // Set the store data register elements to be consecutive integers.
    for (int reg = 0; reg < 8; reg++) {
      auto reg_span = vreg_[reg + kVs1]->data_buffer()->Get<T>();
      for (int i = 0; i < reg_span.size(); i++) {
        reg_span[i] = static_cast<T>(reg * reg_span.size() + i + 1);
      }
    }
    // Base address.
    SetRegisterValues<uint32_t>({{kRs1Name, kDataStoreAddress}});
    // Initialize the mask.
    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});
    // Index values.
    int index_values_per_reg = kVectorLengthInBytes / sizeof(IndexType);
    int num_values_per_register = kVectorLengthInBytes / sizeof(T);

    auto* data_db = state_->db_factory()->Allocate<T>(1);
    // Iterate over legal values in the nf field.
    for (int nf = 1; nf < 8; nf++) {
      int num_fields = nf + 1;

      // Set the number of fields in the source operand.
      SetRegisterValues<int32_t>({{kRs3Name, nf}});
      // Iterate over different lmul values.
      for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
        // Configure vector unit, set the sew to the element width.
        uint32_t vtype = (kSewSettingsByByteSize[sizeof(T)] << 3) |
                         kLmulSettings[lmul_index];
        int lmul8 = kLmul8Values[lmul_index];

        // Set up Index vector.
        // Max number of segments (for testing) is limited by the range of the
        // index type. For byte indices, the range is only +/- 128 for each
        // index. Since the index isn't scaled, this means that the max number
        // of segments for byte indices is 256 / (sizeof(T) * nf)
        int segment_size = sizeof(T) * num_fields;
        int max_segments =
            kVectorLengthInBytes * lmul8 / (8 * num_fields * sizeof(T));
        if (sizeof(IndexType) == 1) {
          max_segments = std::min(256 / num_fields, max_segments);
        }
        ConfigureVectorUnit(vtype, max_segments);

        int emul8 =
            sizeof(IndexType) * lmul8 / rv_vector_->selected_element_width();
        // Make sure not to write too many indices. At this point the emul
        // value may still be "illegal", so avoid a "crash" due to writing
        // the data_buffer out of range.
        int max_indices = kVectorLengthInBytes *
                          std::min(8, std::max(1, emul8 / 8)) /
                          sizeof(IndexType);
        if (max_indices > max_segments) {
          max_indices = max_segments;
        }
        for (int i = 0; i < max_indices; i++) {
          int reg = i / index_values_per_reg;
          int element = i % index_values_per_reg;
          // Scale index by the segment size to avoid writing to the same
          // location twice.
          vreg_[kVs2 + reg]->data_buffer()->Set<IndexType>(
              element, IndexValue<IndexType>(i) * segment_size);
        }
        // Execute instruction.
        instruction_->Execute(nullptr);

        // Check for exceptions when they should be set, and verify no
        // exception otherwise.
        if (lmul8 * num_fields > 64) {
          EXPECT_TRUE(rv_vector_->vector_exception())
              << "emul8: " << emul8 << "  lmul8: " << lmul8;
          rv_vector_->clear_vector_exception();
          continue;
        }
        if (emul8 == 0 || emul8 > 64) {
          EXPECT_TRUE(rv_vector_->vector_exception())
              << "emul8: " << emul8 << "  lmul8: " << lmul8;
          rv_vector_->clear_vector_exception();
          continue;
        }
        EXPECT_FALSE(rv_vector_->vector_exception())
            << "emul8: " << emul8 << "  lmul8: " << lmul8;

        // Check memory values.
        uint64_t base = kDataStoreAddress;
        T value = 1;
        int vlen = rv_vector_->vector_length();
        // Iterate over fields.
        for (int field = 0; field < num_fields; field++) {
          // Expected value starts at 1 for the first register element and
          // increments from there. The following computes the expected value
          // of segment 0 for each field.
          value = field * kVectorLengthInBytes * std::max(1, lmul8 / 8) /
                      sizeof(T) +
                  1;
          for (int segment = 0; segment < max_segments; segment++) {
            int index_reg = segment / index_values_per_reg;
            int index_no = segment % index_values_per_reg;
            // Load the data.
            int64_t index =
                vreg_[kVs2 + index_reg]->data_buffer()->Get<IndexType>(
                    index_no);
            uint64_t address = base + field * sizeof(T) + index;
            state_->LoadMemory(instruction_, address, data_db, nullptr,
                               nullptr);
            int element = segment % num_values_per_register;
            // Get the mask value.
            int mask_index = segment >> 3;
            int mask_offset = segment & 0b111;
            bool mask = (kA5Mask[mask_index] >> mask_offset) & 0x1;
            if (mask && (segment < vlen)) {
              EXPECT_EQ(data_db->template Get<T>(0), static_cast<T>(value))
                  << std::hex << "address: 0x" << address << std::dec
                  << " index: " << index << " segment_no: " << segment
                  << " field: " << field << " i: " << element
                  << " element_size: " << sizeof(T) << " lmul8: " << lmul8
                  << " num_fields: " << num_fields;
              // Zero the memory location.
              data_db->template Set<T>(0, 0);
              state_->StoreMemory(instruction_, address, data_db);
            } else {
              EXPECT_EQ(data_db->template Get<T>(0), 0)
                  << std::hex << "address: 0x" << address << std::dec
                  << " index: " << index << " segment_no: " << segment
                  << " field: " << field << " i: " << element
                  << " element_size: " << sizeof(T) << " lmul8: " << lmul8
                  << " mask: " << mask;
            }
            value++;
          }
        }
        if (HasFailure()) {
          data_db->DecRef();
          return;
        }
      }
    }
    data_db->DecRef();
  }

 protected:
  RV32Register* xreg_[32];
  RVVectorRegister* vreg_[32];
  RiscVState* state_;
  Instruction* instruction_;
  Instruction* child_instruction_;
  FlatDemandMemory* memory_;
  RiscVVectorState* rv_vector_;
};

// Test the vector configuration set instructions. There are three separate
// versions depending on whether Rs1 is X0 or not, of if Rd is X0.
// The first handles the case when Rs1 is not X0.
TEST_F(RV32VInstructionsTest, VsetvlNN) {
  AppendRegisterOperands({kRs1Name, kRs2Name}, {kRdName});
  SetSemanticFunction(absl::bind_front(&Vsetvl,
                                       /*rd_zero*/ false,
                                       /*rs1_zero*/ false));
  for (int lmul = 0; lmul < 7; lmul++) {
    for (int sew = 0; sew < 4; sew++) {
      for (int vlen_select = 0; vlen_select < 2; vlen_select++) {
        // Try vlen below max and above.
        uint32_t vlen = (vlen_select == 0) ? 16 : 1024;
        uint32_t vma = (lmul & 1) ? 0b1'0'000'000 : 0;
        uint32_t vta = (sew & 1) ? 0b0'1'000'000 : 0;
        uint32_t vtype =
            vma | vta | (kSewSettings[sew] << 3) | kLmulSettings[lmul];

        SetRegisterValues<uint32_t>(
            {{kRs1Name, vlen}, {kRs2Name, vtype}, {kRdName, 0}});

        // Execute instruction.
        instruction_->Execute(nullptr);

        // Check results.
        uint32_t expected_vlen =
            std::min<uint32_t>(vlen, kVectorLengthInBytes * kLmul8Values[lmul] /
                                         (8 * kSewValues[sew]));
        EXPECT_EQ(xreg_[kRd]->data_buffer()->Get<uint32_t>(0), expected_vlen)
            << "LMUL: " << kLmul8Values[lmul] << " SEW: " << kSewValues[sew]
            << " AVL: " << vlen;
        EXPECT_EQ(rv_vector_->vector_length(), expected_vlen);
        EXPECT_EQ(rv_vector_->vector_mask_agnostic(), vma != 0);
        EXPECT_EQ(rv_vector_->vector_tail_agnostic(), vta != 0);
        EXPECT_EQ(rv_vector_->vector_length_multiplier(), kLmul8Values[lmul]);
        EXPECT_EQ(rv_vector_->selected_element_width(), kSewValues[sew]);
      }
    }
  }
}

// The case when Rd is X0, but not Rs1.
TEST_F(RV32VInstructionsTest, VsetvlZN) {
  AppendRegisterOperands({kRs1Name, kRs2Name}, {kRdName});
  SetSemanticFunction(absl::bind_front(&Vsetvl,
                                       /*rd_zero*/ true, /*rs1_zero*/ false));
  for (int lmul = 0; lmul < 7; lmul++) {
    for (int sew = 0; sew < 4; sew++) {
      for (int vlen_select = 0; vlen_select < 2; vlen_select++) {
        // Try vlen below max and above.
        uint32_t vlen = (vlen_select == 0) ? 16 : 1024;
        uint32_t vma = (lmul & 1) ? 0b1'0'000'000 : 0;
        uint32_t vta = (sew & 1) ? 0b0'1'000'000 : 0;
        uint32_t vtype =
            vma | vta | (kSewSettings[sew] << 3) | kLmulSettings[lmul];

        SetRegisterValues<uint32_t>(
            {{kRs1Name, vlen}, {kRs2Name, vtype}, {kRdName, 0}});

        // Execute instruction.
        instruction_->Execute(nullptr);

        // Check results.
        uint32_t expected_vlen =
            std::min<uint32_t>(vlen, kVectorLengthInBytes * kLmul8Values[lmul] /
                                         (8 * kSewValues[sew]));
        EXPECT_EQ(xreg_[kRd]->data_buffer()->Get<uint32_t>(0), 0);
        EXPECT_EQ(rv_vector_->vector_length(), expected_vlen);
        EXPECT_EQ(rv_vector_->vector_mask_agnostic(), vma != 0);
        EXPECT_EQ(rv_vector_->vector_tail_agnostic(), vta != 0);
        EXPECT_EQ(rv_vector_->vector_length_multiplier(), kLmul8Values[lmul]);
        EXPECT_EQ(rv_vector_->selected_element_width(), kSewValues[sew]);
      }
    }
  }
}

// The case when Rd is not X0, but Rs1 is X0.
TEST_F(RV32VInstructionsTest, VsetvlNZ) {
  AppendRegisterOperands({kRs1Name, kRs2Name}, {kRdName});
  SetSemanticFunction(absl::bind_front(&Vsetvl,
                                       /*rd_zero*/ false, /*rs1_zero*/ true));
  for (int lmul = 0; lmul < 7; lmul++) {
    for (int sew = 0; sew < 4; sew++) {
      for (int vlen_select = 0; vlen_select < 2; vlen_select++) {
        // Try vlen below max and above.
        uint32_t vlen = (vlen_select == 0) ? 16 : 1024;
        uint32_t vma = (lmul & 1) ? 0b1'0'000'000 : 0;
        uint32_t vta = (sew & 1) ? 0b0'1'000'000 : 0;
        uint32_t vtype =
            vma | vta | (kSewSettings[sew] << 3) | kLmulSettings[lmul];

        SetRegisterValues<uint32_t>(
            {{kRs1Name, vlen}, {kRs2Name, vtype}, {kRdName, 0}});

        // Execute instruction.
        instruction_->Execute(nullptr);

        // Check results.
        // In this case, vlen is vlen max.
        uint32_t expected_vlen =
            kVectorLengthInBytes * kLmul8Values[lmul] / (8 * kSewValues[sew]);
        EXPECT_EQ(xreg_[kRd]->data_buffer()->Get<uint32_t>(0), expected_vlen);
        EXPECT_EQ(rv_vector_->vector_length(), expected_vlen);
        EXPECT_EQ(rv_vector_->vector_mask_agnostic(), vma != 0);
        EXPECT_EQ(rv_vector_->vector_tail_agnostic(), vta != 0);
        EXPECT_EQ(rv_vector_->vector_length_multiplier(), kLmul8Values[lmul]);
        EXPECT_EQ(rv_vector_->selected_element_width(), kSewValues[sew]);
      }
    }
  }
}

// The case when Rd and Rs1 are X0. In this case we are testing if an invalid
// vector operation happens, which occurs if the max vector length changes due
// to the new value of vtype.
TEST_F(RV32VInstructionsTest, VsetvlZZ) {
  AppendRegisterOperands({kRs1Name, kRs2Name}, {kRdName});
  SetSemanticFunction(absl::bind_front(&Vsetvl,
                                       /*rd_zero*/ true, /*rs1_zero*/ true));
  // Iterate over vector lengths.
  for (int vlen = 512; vlen > 8; vlen /= 2) {
    // First set the appropriate vector type for sew = 1 byte.
    uint32_t lmul8 = vlen * 8 / kVectorLengthInBytes;
    ASSERT_LE(lmul8, 64);
    ASSERT_GE(lmul8, 1);
    int lmul8_log2 = absl::bit_width<uint32_t>(lmul8);
    int lmul_setting = kLmulSettingByLogSize[lmul8_log2];
    // Set vtype for this vector length.
    rv_vector_->SetVectorType(lmul_setting);
    int max_vector_length = rv_vector_->max_vector_length();
    for (int lmul = 0; lmul < 7; lmul++) {
      for (int sew = 0; sew < 4; sew++) {
        // Clear any exception.
        rv_vector_->clear_vector_exception();
        // Set up the vtype to try to set using the instruction.
        uint32_t vma = (lmul & 1) ? 0b1'0'000'000 : 0;
        uint32_t vta = (sew & 1) ? 0b0'1'000'000 : 0;
        uint32_t vtype =
            vma | vta | (kSewSettings[sew] << 3) | kLmulSettings[lmul];

        SetRegisterValues<uint32_t>(
            {{kRs1Name, 0xdeadbeef}, {kRs2Name, vtype}, {kRdName, 0xdeadbeef}});

        // Execute instruction.
        instruction_->Execute(nullptr);

        // Check results.
        uint32_t new_vlen =
            kVectorLengthInBytes * kLmul8Values[lmul] / (8 * kSewValues[sew]);
        // If vlen changes, then we expect an error and no change.
        if (new_vlen != max_vector_length) {
          EXPECT_TRUE(rv_vector_->vector_exception())
              << "vlen: " << max_vector_length
              << " lmul: " << kLmul8Values[lmul] << " sew: " << kSewValues[sew];
        } else {
          // Otherwise, check that the values are as expected.
          EXPECT_FALSE(rv_vector_->vector_exception());
          EXPECT_EQ(rv_vector_->vector_length_multiplier(), kLmul8Values[lmul]);
          EXPECT_EQ(rv_vector_->selected_element_width(), kSewValues[sew]);
          EXPECT_EQ(rv_vector_->vector_mask_agnostic(), vma != 0);
          EXPECT_EQ(rv_vector_->vector_tail_agnostic(), vta != 0);
        }
        // No change in registers or vector length.
        EXPECT_EQ(xreg_[kRs1]->data_buffer()->Get<uint32_t>(0), 0xdeadbeef);
        EXPECT_EQ(xreg_[kRd]->data_buffer()->Get<uint32_t>(0), 0xdeadbeef);
        EXPECT_EQ(rv_vector_->max_vector_length(), max_vector_length);
      }
    }
  }
}

// This tests the semantic function for the VleN and VlseN instructions. VleN
// is just a unit stride Vlse.
TEST_F(RV32VInstructionsTest, Vle8) { VectorLoadUnitStridedHelper<uint8_t>(); }

TEST_F(RV32VInstructionsTest, Vle16) {
  VectorLoadUnitStridedHelper<uint16_t>();
}

TEST_F(RV32VInstructionsTest, Vle32) {
  VectorLoadUnitStridedHelper<uint32_t>();
}

TEST_F(RV32VInstructionsTest, Vle64) {
  VectorLoadUnitStridedHelper<uint64_t>();
}

TEST_F(RV32VInstructionsTest, Vlse8) { VectorLoadStridedHelper<uint8_t>(); }

TEST_F(RV32VInstructionsTest, Vlse16) { VectorLoadStridedHelper<uint16_t>(); }

TEST_F(RV32VInstructionsTest, Vlse32) { VectorLoadStridedHelper<uint32_t>(); }

TEST_F(RV32VInstructionsTest, Vlse64) { VectorLoadStridedHelper<uint64_t>(); }

// Test of vector load mask.
TEST_F(RV32VInstructionsTest, Vlm) {
  // Set up operands and register values.
  AppendRegisterOperands({kRs1Name}, {});
  SetSemanticFunction(&Vlm);
  SetChildInstruction();
  AppendVectorRegisterOperands(child_instruction_, {}, {kVd});
  SetChildSemanticFunction(&VlChild);
  SetRegisterValues<uint32_t>({{kRs1Name, kDataLoadAddress}});
  // Execute instruction.
  instruction_->Execute(nullptr);
  EXPECT_FALSE(rv_vector_->vector_exception());
  auto span = vreg_[kVd]->data_buffer()->Get<uint8_t>();
  for (int i = 0; i < kVectorLengthInBytes; i++) {
    EXPECT_EQ(i & 0xff, span[i]) << "element: " << i;
  }
}

// Test of vector load register. Loads 1, 2, 4 or 8 registers.
TEST_F(RV32VInstructionsTest, VlRegister) {
  // Set up operands and register values.
  AppendRegisterOperands({kRs1Name}, {});
  SetChildInstruction();
  AppendVectorRegisterOperands(child_instruction_, {}, {kVd});
  SetChildSemanticFunction(&VlChild);
  SetRegisterValues<uint32_t>({{kRs1Name, kDataLoadAddress}});
  // Test 1, 2, 4 and 8 register versions.
  for (int num_reg = 1; num_reg <= 8; num_reg *= 2) {
    SetSemanticFunction(
        absl::bind_front(&VlRegister, num_reg, /*element_width*/ 1));
    // Execute instruction.
    instruction_->Execute();
    // Check values.

    for (int reg = kVd; reg < num_reg; reg++) {
      auto span = vreg_[reg]->data_buffer()->Get<uint8_t>();
      for (int i = 0; i < kVectorLengthInBytes; i++) {
        EXPECT_EQ(span[i], i & 0xff)
            << absl::StrCat("Reg: ", reg, " element ", i);
      }
    }
  }
}

// Indexed loads directly encode the element width of the index value. The
// width of the load value is determined by sew (selected element width).
TEST_F(RV32VInstructionsTest, VlIndexed8_8) {
  VectorLoadIndexedHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, VlIndexed8_16) {
  VectorLoadIndexedHelper<uint8_t, uint16_t>();
}

TEST_F(RV32VInstructionsTest, VlIndexed8_32) {
  VectorLoadIndexedHelper<uint8_t, uint32_t>();
}

TEST_F(RV32VInstructionsTest, VlIndexed8_64) {
  VectorLoadIndexedHelper<uint8_t, uint64_t>();
}

TEST_F(RV32VInstructionsTest, VlIndexed16_8) {
  VectorLoadIndexedHelper<uint16_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, VlIndexed16_16) {
  VectorLoadIndexedHelper<uint16_t, uint16_t>();
}

TEST_F(RV32VInstructionsTest, VlIndexed16_32) {
  VectorLoadIndexedHelper<uint16_t, uint32_t>();
}

TEST_F(RV32VInstructionsTest, VlIndexed16_64) {
  VectorLoadIndexedHelper<uint16_t, uint64_t>();
}

TEST_F(RV32VInstructionsTest, VlIndexed32_8) {
  VectorLoadIndexedHelper<uint32_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, VlIndexed32_16) {
  VectorLoadIndexedHelper<uint32_t, uint16_t>();
}

TEST_F(RV32VInstructionsTest, VlIndexed32_32) {
  VectorLoadIndexedHelper<uint32_t, uint32_t>();
}

TEST_F(RV32VInstructionsTest, VlIndexed32_64) {
  VectorLoadIndexedHelper<uint32_t, uint64_t>();
}

TEST_F(RV32VInstructionsTest, VlIndexed64_8) {
  VectorLoadIndexedHelper<uint64_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, VlIndexed64_16) {
  VectorLoadIndexedHelper<uint64_t, uint16_t>();
}

TEST_F(RV32VInstructionsTest, VlIndexed64_32) {
  VectorLoadIndexedHelper<uint64_t, uint32_t>();
}

TEST_F(RV32VInstructionsTest, VlIndexed64_64) {
  VectorLoadIndexedHelper<uint64_t, uint64_t>();
}

// Test vector load segment unit stride.
TEST_F(RV32VInstructionsTest, Vlsege8) { VectorLoadSegmentHelper<uint8_t>(); }

TEST_F(RV32VInstructionsTest, Vlsege16) { VectorLoadSegmentHelper<uint16_t>(); }

TEST_F(RV32VInstructionsTest, Vlsege32) { VectorLoadSegmentHelper<uint32_t>(); }

TEST_F(RV32VInstructionsTest, Vlsege64) { VectorLoadSegmentHelper<uint64_t>(); }

// Test vector load segment, strided.
TEST_F(RV32VInstructionsTest, Vlssege8) {
  VectorLoadStridedSegmentHelper<uint8_t>();
}

TEST_F(RV32VInstructionsTest, Vlssege16) {
  VectorLoadStridedSegmentHelper<uint16_t>();
}

TEST_F(RV32VInstructionsTest, Vlssege32) {
  VectorLoadStridedSegmentHelper<uint32_t>();
}

TEST_F(RV32VInstructionsTest, Vlssege64) {
  VectorLoadStridedSegmentHelper<uint64_t>();
}

// Test vector load segment, indexed.
TEST_F(RV32VInstructionsTest, Vluxsegei8_8) {
  VectorLoadIndexedSegmentHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, Vluxsegei8_16) {
  VectorLoadIndexedSegmentHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, Vluxsegei8_32) {
  VectorLoadIndexedSegmentHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, Vluxsegei8_64) {
  VectorLoadIndexedSegmentHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, Vluxsegei16_8) {
  VectorLoadIndexedSegmentHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, Vluxsegei16_16) {
  VectorLoadIndexedSegmentHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, Vluxsegei16_32) {
  VectorLoadIndexedSegmentHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, Vluxsegei16_64) {
  VectorLoadIndexedSegmentHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, Vluxsegei32_8) {
  VectorLoadIndexedSegmentHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, Vluxsegei32_16) {
  VectorLoadIndexedSegmentHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, Vluxsegei32_32) {
  VectorLoadIndexedSegmentHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, Vluxsegei32_64) {
  VectorLoadIndexedSegmentHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, Vluxsegei64_8) {
  VectorLoadIndexedSegmentHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, Vluxsegei64_16) {
  VectorLoadIndexedSegmentHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, Vluxsegei64_32) {
  VectorLoadIndexedSegmentHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, Vluxsegei64_64) {
  VectorLoadIndexedSegmentHelper<uint8_t, uint8_t>();
}

// Test Vector store strided.

TEST_F(RV32VInstructionsTest, Vsse8) { VectorStoreStridedHelper<uint8_t>(); }

TEST_F(RV32VInstructionsTest, Vsse16) { VectorStoreStridedHelper<uint16_t>(); }

TEST_F(RV32VInstructionsTest, Vsse32) { VectorStoreStridedHelper<uint32_t>(); }

TEST_F(RV32VInstructionsTest, Vsse64) { VectorStoreStridedHelper<uint64_t>(); }

TEST_F(RV32VInstructionsTest, Vsm) {
  ConfigureVectorUnit(0b0'0'000'000, /*vlen*/ 1024);
  // Set up operands and register values.
  AppendVectorRegisterOperands({kVs1}, {});
  AppendRegisterOperands({kRs1Name}, {});
  SetSemanticFunction(&Vsm);
  SetRegisterValues<uint32_t>({{kRs1Name, kDataStoreAddress}});
  for (int i = 0; i < kVectorLengthInBytes; i++) {
    vreg_[kVs1]->data_buffer()->Set<uint8_t>(i, i);
  }
  // Execute instruction.
  instruction_->Execute(nullptr);

  // Verify result.
  EXPECT_FALSE(rv_vector_->vector_exception());
  auto* data_db = state_->db_factory()->Allocate<uint8_t>(kVectorLengthInBytes);
  state_->LoadMemory(instruction_, kDataStoreAddress, data_db, nullptr,
                     nullptr);
  auto span = data_db->Get<uint8_t>();
  for (int i = 0; i < kVectorLengthInBytes; i++) {
    EXPECT_EQ(static_cast<int>(span[i]), i);
  }
  data_db->DecRef();
}

// Tests of indexed stores, cross product of index types with value types.
TEST_F(RV32VInstructionsTest, VsIndexed8_8) {
  VectorStoreIndexedHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, VsIndexed8_16) {
  VectorStoreIndexedHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, VsIndexed8_32) {
  VectorStoreIndexedHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, VsIndexed8_64) {
  VectorStoreIndexedHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, VsIndexed16_8) {
  VectorStoreIndexedHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, VsIndexed16_16) {
  VectorStoreIndexedHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, VsIndexed16_32) {
  VectorStoreIndexedHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, VsIndexed16_64) {
  VectorStoreIndexedHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, VsIndexed32_8) {
  VectorStoreIndexedHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, VsIndexed32_16) {
  VectorStoreIndexedHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, VsIndexed32_32) {
  VectorStoreIndexedHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, VsIndexed32_64) {
  VectorStoreIndexedHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, VsIndexed64_8) {
  VectorStoreIndexedHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, VsIndexed64_16) {
  VectorStoreIndexedHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, VsIndexed64_32) {
  VectorStoreIndexedHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, VsIndexed64_64) {
  VectorStoreIndexedHelper<uint8_t, uint8_t>();
}

TEST_F(RV32VInstructionsTest, VsRegister) {
  ConfigureVectorUnit(0b0'0'000'000, /*vlen*/ 1024);
  int num_elem = kVectorLengthInBytes / sizeof(uint64_t);
  // Set up operands and register values.
  AppendVectorRegisterOperands({kVs1}, {});
  for (int reg = 0; reg < 8; reg++) {
    for (int i = 0; i < num_elem; i++) {
      vreg_[kVs1 + reg]->data_buffer()->Set<uint64_t>(i, reg * num_elem + i);
    }
  }
  AppendRegisterOperands({kRs1Name}, {});
  SetRegisterValues<uint32_t>({{kRs1Name, kDataStoreAddress}});

  auto data_db = state_->db_factory()->Allocate(8 * kVectorLengthInBytes);
  for (int num_regs = 1; num_regs <= 8; num_regs++) {
    SetSemanticFunction(absl::bind_front(&VsRegister, num_regs));

    // Execute instruction.
    instruction_->Execute();

    // Verify results.
    EXPECT_FALSE(rv_vector_->vector_exception());
    uint64_t base = kDataStoreAddress;
    for (int reg = 0; reg < 8; reg++) {
      state_->LoadMemory(instruction_, base, data_db, nullptr, nullptr);
      auto span = data_db->Get<uint64_t>();
      for (int i = 0; i < num_elem; i++) {
        if (reg < num_regs) {
          EXPECT_EQ(span[i], reg * num_elem + i)
              << "reg[" << reg << "][" << i << "]";
        } else {
          EXPECT_EQ(span[i], 0) << "reg[" << reg << "][" << i << "]";
        }
      }
      base += kVectorLengthInBytes;
    }
    // Clear Memory.
    memset(data_db->raw_ptr(), 0, 8 * kVectorLengthInBytes);
    state_->StoreMemory(instruction_, kDataStoreAddress, data_db);
  }
  data_db->DecRef();
}

// Test vector store segment unit stride.
TEST_F(RV32VInstructionsTest, VsSegment8) {
  VectorStoreSegmentHelper<uint8_t>();
}

TEST_F(RV32VInstructionsTest, VsSegment16) {
  VectorStoreSegmentHelper<uint16_t>();
}

TEST_F(RV32VInstructionsTest, VsSegment32) {
  VectorStoreSegmentHelper<uint32_t>();
}

TEST_F(RV32VInstructionsTest, VsSegment64) {
  VectorStoreSegmentHelper<uint64_t>();
}

// Test vector store segment strided.
TEST_F(RV32VInstructionsTest, VsSegmentStrided8) {
  VectorStoreStridedSegmentHelper<uint8_t>();
}
TEST_F(RV32VInstructionsTest, VsSegmentStrided16) {
  VectorStoreStridedSegmentHelper<uint16_t>();
}
TEST_F(RV32VInstructionsTest, VsSegmentStrided32) {
  VectorStoreStridedSegmentHelper<uint32_t>();
}
TEST_F(RV32VInstructionsTest, VsSegmentStrided64) {
  VectorStoreStridedSegmentHelper<uint64_t>();
}

// Test vector store segment indexed. Test each
// combination of element size and index size.
TEST_F(RV32VInstructionsTest, VsSegmentIndexed8_8) {
  VectorStoreIndexedSegmentHelper<uint8_t, int8_t>();
}

TEST_F(RV32VInstructionsTest, VsSegmentIndexed8_16) {
  VectorStoreIndexedSegmentHelper<uint8_t, int16_t>();
}

TEST_F(RV32VInstructionsTest, VsSegmentIndexed8_32) {
  VectorStoreIndexedSegmentHelper<uint8_t, int32_t>();
}

TEST_F(RV32VInstructionsTest, VsSegmentIndexed8_64) {
  VectorStoreIndexedSegmentHelper<uint8_t, int64_t>();
}

TEST_F(RV32VInstructionsTest, VsSegmentIndexed16_8) {
  VectorStoreIndexedSegmentHelper<uint16_t, int8_t>();
}

TEST_F(RV32VInstructionsTest, VsSegmentIndexed16_16) {
  VectorStoreIndexedSegmentHelper<uint16_t, int16_t>();
}

TEST_F(RV32VInstructionsTest, VsSegmentIndexed16_32) {
  VectorStoreIndexedSegmentHelper<uint16_t, int32_t>();
}

TEST_F(RV32VInstructionsTest, VsSegmentIndexed16_64) {
  VectorStoreIndexedSegmentHelper<uint16_t, int64_t>();
}

TEST_F(RV32VInstructionsTest, VsSegmentIndexed32_8) {
  VectorStoreIndexedSegmentHelper<uint32_t, int8_t>();
}

TEST_F(RV32VInstructionsTest, VsSegmentIndexed32_16) {
  VectorStoreIndexedSegmentHelper<uint32_t, int16_t>();
}

TEST_F(RV32VInstructionsTest, VsSegmentIndexed32_32) {
  VectorStoreIndexedSegmentHelper<uint32_t, int32_t>();
}

TEST_F(RV32VInstructionsTest, VsSegmentIndexed32_64) {
  VectorStoreIndexedSegmentHelper<uint32_t, int64_t>();
}

TEST_F(RV32VInstructionsTest, VsSegmentIndexed64_8) {
  VectorStoreIndexedSegmentHelper<uint64_t, int8_t>();
}

TEST_F(RV32VInstructionsTest, VsSegmentIndexed64_16) {
  VectorStoreIndexedSegmentHelper<uint64_t, int16_t>();
}

TEST_F(RV32VInstructionsTest, VsSegmentIndexed64_32) {
  VectorStoreIndexedSegmentHelper<uint64_t, int32_t>();
}

TEST_F(RV32VInstructionsTest, VsSegmentIndexed64_64) {
  VectorStoreIndexedSegmentHelper<uint64_t, int64_t>();
}
}  // namespace
