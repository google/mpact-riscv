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

#ifndef MPACT_RISCV_RISCV_TEST_RISCV_VECTOR_INSTRUCTIONS_TEST_BASE_H_
#define MPACT_RISCV_RISCV_TEST_RISCV_VECTOR_INSTRUCTIONS_TEST_BASE_H_

#include <algorithm>
#include <ios>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "absl/functional/bind_front.h"
#include "absl/random/random.h"
#include "absl/strings/string_view.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_memory_instructions.h"
#include "riscv/riscv_vector_state.h"

// This file defines commonly used constants in the vector instruction tests
// as well as a base class for the vector instruction test fixtures. This base
// class contains methods that make it more convenient to write vector
// instruction test cases, and provide "harnesses" to test the functionality
// of individual instructions across different lmul values, vstart values,
// and vector length values.

namespace mpact {
namespace sim {
namespace riscv {
namespace test {

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

// Constants used in the tests.
constexpr int kVectorLengthInBits = 512;
constexpr int kVectorLengthInBytes = kVectorLengthInBits / 8;
constexpr uint32_t kInstAddress = 0x1000;
constexpr uint32_t kDataLoadAddress = 0x1'0000;
constexpr uint32_t kDataStoreAddress = 0x2'0000;
constexpr char kRs1Name[] = "x1";
constexpr int kRs1 = 1;
constexpr char kRs2Name[] = "x2";
constexpr char kRs3Name[] = "x3";
constexpr char kRdName[] = "x8";
constexpr int kRd = 8;
constexpr int kVmask = 1;
constexpr char kVmaskName[] = "v1";
constexpr int kVd = 8;
constexpr char kVdName[] = "v8";
constexpr int kVs1 = 16;
constexpr char kVs1Name[] = "v16";
constexpr int kVs2 = 24;
constexpr char kVs2Name[] = "v24";

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
// This is the base class for vector instruction test fixtures. It implements
// generic methods for testing and supporting testing of the RiscV vector
// instructions.
class RiscVVectorInstructionsTestBase : public testing::Test {
 public:
  RiscVVectorInstructionsTestBase() {
    memory_ = new FlatDemandMemory(0);
    state_ = new RiscVState("test", RiscVXlen::RV32, memory_);
    rv_vector_ = new RiscVVectorState(state_, kVectorLengthInBytes);
    instruction_ = new Instruction(kInstAddress, state_);
    instruction_->set_size(4);
    child_instruction_ = new Instruction(kInstAddress, state_);
    child_instruction_->set_size(4);
    // Initialize a portion of memory with a known pattern.
    auto *db = state_->db_factory()->Allocate(8192);
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
      freg_[i] = state_->GetRegister<RVFpRegister>(absl::StrCat("f", i)).first;
    }
    for (int i = 1; i < 32; i++) {
      vreg_[i] =
          state_->GetRegister<RVVectorRegister>(absl::StrCat("v", i)).first;
    }
  }

  ~RiscVVectorInstructionsTestBase() override {
    delete state_;
    delete rv_vector_;
    instruction_->DecRef();
    child_instruction_->DecRef();
    delete memory_;
  }

  // Clear the instruction instance and allocate a new one.
  void ResetInstruction() {
    instruction_->DecRef();
    instruction_ = new Instruction(kInstAddress, state_);
    instruction_->set_size(4);
  }

  // Creates immediate operands with the values from the vector and appends them
  // to the given instruction.
  template <typename T>
  void AppendImmediateOperands(Instruction *inst,
                               const std::vector<T> &values) {
    for (auto value : values) {
      auto *src = new ImmediateOperand<T>(value);
      inst->AppendSource(src);
    }
  }

  // Creates immediate operands with the values from the vector and appends them
  // to the default instruction.
  template <typename T>
  void AppendImmediateOperands(const std::vector<T> &values) {
    AppendImmediateOperands<T>(instruction_, values);
  }

  // Creates source and destination scalar register operands for the registers
  // named in the two vectors and append them to the given instruction.
  void AppendRegisterOperands(Instruction *inst,
                              const std::vector<std::string> &sources,
                              const std::vector<std::string> &destinations) {
    for (auto &reg_name : sources) {
      auto *reg = state_->GetRegister<RV32Register>(reg_name).first;
      inst->AppendSource(reg->CreateSourceOperand());
    }
    for (auto &reg_name : destinations) {
      auto *reg = state_->GetRegister<RV32Register>(reg_name).first;
      inst->AppendDestination(reg->CreateDestinationOperand(0));
    }
  }

  // Creates source and destination scalar register operands for the registers
  // named in the two vectors and append them to the default instruction.
  void AppendRegisterOperands(const std::vector<std::string> &sources,
                              const std::vector<std::string> &destinations) {
    AppendRegisterOperands(instruction_, sources, destinations);
  }

  // Returns the value of the named vector register.
  template <typename T>
  T GetRegisterValue(absl::string_view vreg_name) {
    auto *reg = state_->GetRegister<RV32Register>(vreg_name).first;
    return reg->data_buffer()->Get<T>();
  }

  // named register and sets it to the corresponding value.
  template <typename T, typename RegisterType = RV32Register>
  void SetRegisterValues(
      const std::vector<tuple<std::string, const T>> &values) {
    for (auto &[reg_name, value] : values) {
      auto *reg = state_->GetRegister<RegisterType>(reg_name).first;
      auto *db =
          state_->db_factory()->Allocate<typename RegisterType::ValueType>(1);
      db->template Set<T>(0, value);
      reg->SetDataBuffer(db);
      db->DecRef();
    }
  }

  // Creates source and destination scalar register operands for the registers
  // named in the two vectors and append them to the given instruction.
  void AppendVectorRegisterOperands(Instruction *inst,
                                    const std::vector<int> &sources,
                                    const std::vector<int> &destinations) {
    for (auto &reg_no : sources) {
      std::vector<RegisterBase *> reg_vec;
      for (int i = 0; (i < 8) && (i + reg_no < 32); i++) {
        std::string reg_name = absl::StrCat("v", i + reg_no);
        reg_vec.push_back(
            state_->GetRegister<RVVectorRegister>(reg_name).first);
      }
      auto *op = new RV32VectorSourceOperand(
          absl::Span<RegisterBase *>(reg_vec), absl::StrCat("v", reg_no));
      inst->AppendSource(op);
    }
    for (auto &reg_no : destinations) {
      std::vector<RegisterBase *> reg_vec;
      for (int i = 0; (i < 8) && (i + reg_no < 32); i++) {
        std::string reg_name = absl::StrCat("v", i + reg_no);
        reg_vec.push_back(
            state_->GetRegister<RVVectorRegister>(reg_name).first);
      }
      auto *op = new RV32VectorDestinationOperand(
          absl::Span<RegisterBase *>(reg_vec), 0, absl::StrCat("v", reg_no));
      inst->AppendDestination(op);
    }
  }
  // Creates source and destination scalar register operands for the registers
  // named in the two vectors and append them to the default instruction.
  void AppendVectorRegisterOperands(const std::vector<int> &sources,
                                    const std::vector<int> &destinations) {
    AppendVectorRegisterOperands(instruction_, sources, destinations);
  }

  // Returns the value of the named vector register.
  template <typename T>
  T GetVectorRegisterValue(absl::string_view reg_name) {
    auto *reg = state_->GetRegister<RVVectorRegister>(reg_name).first;
    return reg->data_buffer()->Get<T>(0);
  }

  // Set a vector register value. Takes a vector of tuples of register names and
  // spans of values, fetches each register and sets it to the corresponding
  // value.
  template <typename T>
  void SetVectorRegisterValues(
      const std::vector<tuple<std::string, Span<const T>>> &values) {
    for (auto &[vreg_name, span] : values) {
      auto *vreg = state_->GetRegister<RVVectorRegister>(vreg_name).first;
      auto *db = state_->db_factory()->MakeCopyOf(vreg->data_buffer());
      db->template Set<T>(span);
      vreg->SetDataBuffer(db);
      db->DecRef();
    }
  }

  // Initializes the semantic function of the instruction object.
  void SetSemanticFunction(Instruction *inst,
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
    Instruction *inst = new Instruction(state_);
    AppendImmediateOperands<uint32_t>(inst, {vlen, vtype});
    SetSemanticFunction(inst, absl::bind_front(&Vsetvl, true, false));
    inst->Execute(nullptr);
    inst->DecRef();
  }

  // Clear count registers in the register group, starting at start.
  void ClearVectorRegisterGroup(int start, int count) {
    for (int reg = start; (reg < start + count) && (reg < 32); reg++) {
      memset(vreg_[reg]->data_buffer()->raw_ptr(), 0, kVectorLengthInBytes);
    }
  }

  // Create a random value in the valid range for the type.
  template <typename T>
  T RandomValue() {
    return absl::Uniform(absl::IntervalClosed, bitgen_,
                         std::numeric_limits<T>::lowest(),
                         std::numeric_limits<T>::max());
  }

  // Fill the span with random values.
  template <typename T>
  void FillArrayWithRandomValues(absl::Span<T> span) {
    for (auto &val : span) {
      val = RandomValue<T>();
    }
  }

  // Helper function for testing unary vector-vector instructions.
  template <typename Vd, typename Vs2>
  void UnaryOpTestHelperV(absl::string_view name, int sew, Instruction *inst,
                          std::function<Vd(Vs2)> operation) {
    int byte_sew = sew / 8;
    if (byte_sew != sizeof(Vd) && byte_sew != sizeof(Vs2)) {
      FAIL() << name << ": selected element width != any operand types"
             << "sew: " << sew << " Vd: " << sizeof(Vd)
             << " Vs2: " << sizeof(Vs2);
      return;
    }
    // Number of elements per vector register.
    constexpr int vs2_size = kVectorLengthInBytes / sizeof(Vs2);
    // Input values for 8 registers.
    Vs2 vs2_value[vs2_size * 8];
    auto vs2_span = Span<Vs2>(vs2_value);
    AppendVectorRegisterOperands({kVs2, kVmask}, {kVd});
    // Initialize input values.
    FillArrayWithRandomValues<Vs2>(vs2_span);
    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});
    for (int i = 0; i < 8; i++) {
      auto vs2_name = absl::StrCat("v", kVs2 + i);
      SetVectorRegisterValues<Vs2>(
          {{vs2_name, vs2_span.subspan(vs2_size * i, vs2_size)}});
    }
    for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
      for (int vstart_count = 0; vstart_count < 4; vstart_count++) {
        for (int vlen_count = 0; vlen_count < 4; vlen_count++) {
          int lmul8 = kLmul8Values[lmul_index];
          int lmul8_vd = lmul8 * sizeof(Vd) / byte_sew;
          int lmul8_vs2 = lmul8 * sizeof(Vs2) / byte_sew;
          int num_values = lmul8 * kVectorLengthInBytes / (8 * byte_sew);
          int vstart = 0;
          if (vstart_count > 0) {
            vstart = absl::Uniform(absl::IntervalOpen, bitgen_, 0, num_values);
          }
          // Set vlen, but leave vlen high at least once.
          int vlen = 1024;
          if (vlen_count > 0) {
            vlen = absl::Uniform(absl::IntervalOpenClosed, bitgen_, vstart,
                                 num_values);
          }
          num_values = std::min(num_values, vlen);
          ASSERT_TRUE(vlen > vstart);
          // Configure vector unit for different lmul settings.
          uint32_t vtype = (kSewSettingsByByteSize[byte_sew] << 3) |
                           kLmulSettings[lmul_index];
          ConfigureVectorUnit(vtype, vlen);
          rv_vector_->set_vstart(vstart);
          ClearVectorRegisterGroup(kVd, 8);

          inst->Execute();
          if (lmul8_vd < 1 || lmul8_vd > 64) {
            EXPECT_TRUE(rv_vector_->vector_exception())
                << "lmul8: vd: " << lmul8_vd;
            rv_vector_->clear_vector_exception();
            continue;
          }
          if (lmul8_vs2 < 1 || lmul8_vs2 > 64) {
            EXPECT_TRUE(rv_vector_->vector_exception())
                << "lmul8: vs2: " << lmul8_vs2;
            rv_vector_->clear_vector_exception();
            continue;
          }

          EXPECT_FALSE(rv_vector_->vector_exception());
          EXPECT_EQ(rv_vector_->vstart(), 0);
          int count = 0;
          for (int reg = kVd; reg < kVd + 8; reg++) {
            for (int i = 0; i < kVectorLengthInBytes / sizeof(Vd); i++) {
              int mask_index = count >> 3;
              int mask_offset = count & 0b111;
              bool mask_value =
                  ((kA5Mask[mask_index] >> mask_offset) & 0b1) != 0;
              if ((count >= vstart) && mask_value && (count < num_values)) {
                EXPECT_EQ(operation(vs2_value[count]),
                          vreg_[reg]->data_buffer()->Get<Vd>(i))
                    << absl::StrCat(name, "[", count, "] != reg[", reg, "][", i,
                                    "]   lmul8(", lmul8, ")");
              } else {
                EXPECT_EQ(0, vreg_[reg]->data_buffer()->Get<Vd>(i))
                    << absl::StrCat(name, "  0 != reg[", reg, "][", i,
                                    "]   lmul8(", lmul8, ")");
              }
              count++;
            }
          }
        }
      }
    }
  }

  // Helper function for testing vector-vector instructions that use the value
  // of the mask bit.
  template <typename Vd, typename Vs2, typename Vs1>
  void BinaryOpWithMaskTestHelperVV(
      absl::string_view name, int sew, Instruction *inst,
      std::function<Vd(Vs2, Vs1, bool)> operation) {
    int byte_sew = sew / 8;
    if (byte_sew != sizeof(Vd) && byte_sew != sizeof(Vs2) &&
        byte_sew != sizeof(Vs1)) {
      FAIL() << name << ": selected element width != any operand types"
             << "sew: " << sew << " Vd: " << sizeof(Vd)
             << " Vs2: " << sizeof(Vs2) << " Vs1: " << sizeof(Vs1);
      return;
    }
    // Number of elements per vector register.
    constexpr int vs2_size = kVectorLengthInBytes / sizeof(Vs2);
    constexpr int vs1_size = kVectorLengthInBytes / sizeof(Vs1);
    // Input values for 8 registers.
    Vs2 vs2_value[vs2_size * 8];
    auto vs2_span = Span<Vs2>(vs2_value);
    Vs1 vs1_value[vs1_size * 8];
    auto vs1_span = Span<Vs1>(vs1_value);
    AppendVectorRegisterOperands({kVs2, kVs1, kVmask}, {kVd});
    // Initialize input values.
    FillArrayWithRandomValues<Vs2>(vs2_span);
    FillArrayWithRandomValues<Vs1>(vs1_span);
    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});
    for (int i = 0; i < 8; i++) {
      auto vs1_name = absl::StrCat("v", kVs1 + i);
      auto vs2_name = absl::StrCat("v", kVs2 + i);
      SetVectorRegisterValues<Vs2>(
          {{vs2_name, vs2_span.subspan(vs2_size * i, vs2_size)}});
      SetVectorRegisterValues<Vs1>(
          {{vs1_name, vs1_span.subspan(vs1_size * i, vs1_size)}});
    }
    // Iterate across the different lmul values.
    for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
      // Try different vstart values.
      for (int vstart_count = 0; vstart_count < 4; vstart_count++) {
        for (int vlen_count = 0; vlen_count < 4; vlen_count++) {
          int lmul8 = kLmul8Values[lmul_index];
          int lmul8_vd = lmul8 * sizeof(Vd) / byte_sew;
          int lmul8_vs2 = lmul8 * sizeof(Vs2) / byte_sew;
          int lmul8_vs1 = lmul8 * sizeof(Vs1) / byte_sew;
          int num_values = lmul8 * kVectorLengthInBytes / (8 * byte_sew);
          int vstart = 0;
          if (vstart_count > 0) {
            vstart = absl::Uniform(absl::IntervalOpen, bitgen_, 0, num_values);
          }
          // Set vlen, but leave vlen high at least once.
          int vlen = 1024;
          if (vlen_count > 0) {
            vlen = absl::Uniform(absl::IntervalOpenClosed, bitgen_, vstart,
                                 num_values);
          }
          num_values = std::min(num_values, vlen);
          // Configure vector unit for different lmul settings.
          uint32_t vtype = (kSewSettingsByByteSize[byte_sew] << 3) |
                           kLmulSettings[lmul_index];
          ConfigureVectorUnit(vtype, vlen);
          rv_vector_->set_vstart(vstart);
          ClearVectorRegisterGroup(kVd, 8);

          inst->Execute();

          if ((std::min(std::min(lmul8_vs2, lmul8_vs1), lmul8_vd) < 1) ||
              (std::max(std::max(lmul8_vs2, lmul8_vs1), lmul8_vd) > 64)) {
            EXPECT_TRUE(rv_vector_->vector_exception());
            rv_vector_->clear_vector_exception();
            continue;
          }

          EXPECT_FALSE(rv_vector_->vector_exception());
          EXPECT_EQ(rv_vector_->vstart(), 0);
          int count = 0;
          for (int reg = kVd; reg < kVd + 8; reg++) {
            for (int i = 0; i < kVectorLengthInBytes / sizeof(Vd); i++) {
              int mask_index = count >> 3;
              int mask_offset = count & 0b111;
              bool mask_value =
                  ((kA5Mask[mask_index] >> mask_offset) & 0b1) != 0;
              if ((count >= vstart) && (count < num_values)) {
                EXPECT_EQ(
                    operation(vs2_value[count], vs1_value[count], mask_value),
                    vreg_[reg]->data_buffer()->Get<Vd>(i))
                    << std::hex << (int64_t)vs2_value[count] << ", "
                    << (int64_t)vs1_value[count] << "   " << std::dec
                    << (int64_t)vs2_value[count] << ", "
                    << (int64_t)vs1_value[count] << "   "
                    << absl::StrCat(name, "[", count, "] != reg[", reg, "][", i,
                                    "]   lmul8(", lmul8, ")  vstart(", vstart,
                                    ")");
              } else {
                EXPECT_EQ(0, vreg_[reg]->data_buffer()->Get<Vd>(i))
                    << absl::StrCat(name, "  0 != reg[", reg, "][", i,
                                    "]   lmul8(", lmul8, ")  vstart(", vstart,
                                    ")");
              }
              count++;
            }
          }
          if (HasFailure()) return;
        }
      }
    }
  }

  // Helper function for testing vector-vector instructions that do not
  // use the value of the mask bit.
  template <typename Vd, typename Vs2, typename Vs1>
  void BinaryOpTestHelperVV(absl::string_view name, int sew, Instruction *inst,
                            std::function<Vd(Vs2, Vs1)> operation) {
    BinaryOpWithMaskTestHelperVV<Vd, Vs2, Vs1>(
        name, sew, inst, [operation](Vs2 vs2, Vs1 vs1, bool mask_value) -> Vd {
          if (mask_value) {
            return operation(vs2, vs1);
          }
          return 0;
        });
  }

  // Helper function for testing vector-scalar/immediate instructions that use
  // the value of the mask bit.
  template <typename Vd, typename Vs2, typename Rs1>
  void BinaryOpWithMaskTestHelperVX(
      absl::string_view name, int sew, Instruction *inst,
      std::function<Vd(Vs2, Rs1, bool)> operation) {
    int byte_sew = sew / 8;
    if (byte_sew != sizeof(Vd) && byte_sew != sizeof(Vs2) &&
        byte_sew != sizeof(Rs1)) {
      FAIL() << name << ": selected element width != any operand types"
             << "sew: " << sew << " Vd: " << sizeof(Vd)
             << " Vs2: " << sizeof(Vs2) << " Rs1: " << sizeof(Rs1);
      return;
    }
    // Number of elements per vector register.
    constexpr int vs2_size = kVectorLengthInBytes / sizeof(Vs2);
    // Input values for 8 registers.
    Vs2 vs2_value[vs2_size * 8];
    auto vs2_span = Span<Vs2>(vs2_value);
    AppendVectorRegisterOperands({kVs2}, {});
    AppendRegisterOperands({kRs1Name}, {});
    AppendVectorRegisterOperands({kVmask}, {kVd});
    // Initialize input values.
    FillArrayWithRandomValues<Vs2>(vs2_span);
    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});
    for (int i = 0; i < 8; i++) {
      auto vs2_name = absl::StrCat("v", kVs2 + i);
      SetVectorRegisterValues<Vs2>(
          {{vs2_name, vs2_span.subspan(vs2_size * i, vs2_size)}});
    }
    // Iterate across the different lmul values.
    for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
      // Try different vstart values.
      for (int vstart_count = 0; vstart_count < 4; vstart_count++) {
        for (int vlen_count = 0; vlen_count < 4; vlen_count++) {
          int lmul8 = kLmul8Values[lmul_index];
          int lmul8_vd = lmul8 * sizeof(Vd) / byte_sew;
          int lmul8_vs2 = lmul8 * sizeof(Vs2) / byte_sew;
          int num_values = lmul8 * kVectorLengthInBytes / (8 * byte_sew);
          // Set vstart, but leave vstart at 0 at least once.
          int vstart = 0;
          if (vstart_count > 0) {
            vstart = absl::Uniform(absl::IntervalOpen, bitgen_, 0, num_values);
          }
          // Set vlen, but leave vlen high at least once.
          int vlen = 1024;
          if (vlen_count > 0) {
            vlen = absl::Uniform(absl::IntervalOpenClosed, bitgen_, vstart,
                                 num_values);
          }
          num_values = std::min(num_values, vlen);
          ASSERT_TRUE(vlen > vstart);
          // Configure vector unit for the different lmul settings.
          uint32_t vtype = (kSewSettingsByByteSize[byte_sew] << 3) |
                           kLmulSettings[lmul_index];
          ConfigureVectorUnit(vtype, vlen);
          rv_vector_->set_vstart(vstart);
          ClearVectorRegisterGroup(kVd, 8);

          // Generate a new rs1 value.
          RV32Register::ValueType rs1_reg_value =
              RandomValue<RV32Register::ValueType>();
          SetRegisterValues<RV32Register::ValueType>(
              {{kRs1Name, rs1_reg_value}});
          // Cast the value to the appropriate width, sign-extending if need
          // be.
          Rs1 rs1_value = static_cast<Rs1>(
              static_cast<
                  typename SameSignedType<RV32Register::ValueType, Rs1>::type>(
                  rs1_reg_value));

          inst->Execute();
          if ((std::min(lmul8_vs2, lmul8_vd) < 1) ||
              (std::max(lmul8_vs2, lmul8_vd) > 64)) {
            EXPECT_TRUE(rv_vector_->vector_exception());
            rv_vector_->clear_vector_exception();
            continue;
          }

          EXPECT_FALSE(rv_vector_->vector_exception());
          EXPECT_EQ(rv_vector_->vstart(), 0);
          int count = 0;
          for (int reg = kVd; reg < kVd + 8; reg++) {
            for (int i = 0; i < kVectorLengthInBytes / sizeof(Vd); i++) {
              int mask_index = count >> 3;
              int mask_offset = count & 0b111;
              bool mask_value =
                  ((kA5Mask[mask_index] >> mask_offset) & 0b1) != 0;
              // Compare elements that are between vstart and vlen for which
              // the mask is true.
              if ((count >= vstart) && (count < num_values)) {
                Vd expected_value = operation(
                    vs2_value[count], static_cast<Rs1>(rs1_value), mask_value);
                Vd inst_value = vreg_[reg]->data_buffer()->Get<Vd>(i);
                EXPECT_EQ(expected_value, inst_value) << absl::StrCat(
                    name, " [", count, "] != reg[", reg, "][", i, "]   lmul8(",
                    lmul8, ")   op(", absl::Hex(vs2_value[count]), ", ",
                    absl::Hex(static_cast<Rs1>(rs1_value)),
                    ") vreg: ", absl::Hex(inst_value));
              } else {
                // The others should be zero.
                EXPECT_EQ(0, vreg_[reg]->data_buffer()->Get<Vd>(i))
                    << absl::StrCat(name, "  0 != reg[", reg, "][", i,
                                    "]   lmul8(", lmul8, ")");
              }
              count++;
            }
          }
          if (HasFailure()) return;
        }
      }
    }
  }

  // Templated helper function that tests vector-scalar instructions that do
  // not use the value of the mask bit.
  template <typename Vd, typename Vs2, typename Vs1>
  void BinaryOpTestHelperVX(absl::string_view name, int sew, Instruction *inst,
                            std::function<Vd(Vs2, Vs1)> operation) {
    BinaryOpWithMaskTestHelperVX<Vd, Vs2, Vs1>(
        name, sew, inst, [operation](Vs2 vs2, Vs1 vs1, bool mask_value) -> Vd {
          if (mask_value) {
            return operation(vs2, vs1);
          }
          return 0;
        });
  }

  // Helper function for testing vector-vector instructions that use the value
  // of the mask bit.
  template <typename Vd, typename Vs2, typename Vs1>
  void TernaryOpWithMaskTestHelperVV(
      absl::string_view name, int sew, Instruction *inst,
      std::function<Vd(Vs2, Vs1, Vd, bool)> operation) {
    int byte_sew = sew / 8;
    if (byte_sew != sizeof(Vd) && byte_sew != sizeof(Vs2) &&
        byte_sew != sizeof(Vs1)) {
      FAIL() << name << ": selected element width != any operand types"
             << "sew: " << sew << " Vd: " << sizeof(Vd)
             << " Vs2: " << sizeof(Vs2) << " Vs1: " << sizeof(Vs1);
      return;
    }
    // Number of elements per vector register.
    constexpr int vd_size = kVectorLengthInBytes / sizeof(Vd);
    constexpr int vs2_size = kVectorLengthInBytes / sizeof(Vs2);
    constexpr int vs1_size = kVectorLengthInBytes / sizeof(Vs1);
    // Input values for 8 registers.
    Vd vd_value[vd_size * 8];
    auto vd_span = Span<Vd>(vd_value);
    Vs2 vs2_value[vs2_size * 8];
    auto vs2_span = Span<Vs2>(vs2_value);
    Vs1 vs1_value[vs1_size * 8];
    auto vs1_span = Span<Vs1>(vs1_value);
    AppendVectorRegisterOperands({kVs2, kVs1, kVd, kVmask}, {kVd});
    // Initialize input values.
    FillArrayWithRandomValues<Vd>(vd_span);
    FillArrayWithRandomValues<Vs2>(vs2_span);
    FillArrayWithRandomValues<Vs1>(vs1_span);
    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});
    for (int i = 0; i < 8; i++) {
      auto vs1_name = absl::StrCat("v", kVs1 + i);
      auto vs2_name = absl::StrCat("v", kVs2 + i);
      SetVectorRegisterValues<Vs2>(
          {{vs2_name, vs2_span.subspan(vs2_size * i, vs2_size)}});
      SetVectorRegisterValues<Vs1>(
          {{vs1_name, vs1_span.subspan(vs1_size * i, vs1_size)}});
    }
    // Iterate across the different lmul values.
    for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
      // Try different vstart values.
      for (int vstart_count = 0; vstart_count < 4; vstart_count++) {
        for (int vlen_count = 0; vlen_count < 4; vlen_count++) {
          int lmul8 = kLmul8Values[lmul_index];
          int lmul8_vd = lmul8 * sizeof(Vd) / byte_sew;
          int lmul8_vs2 = lmul8 * sizeof(Vs2) / byte_sew;
          int lmul8_vs1 = lmul8 * sizeof(Vs1) / byte_sew;
          int num_values = lmul8 * kVectorLengthInBytes / (8 * byte_sew);
          int vstart = 0;
          if (vstart_count > 0) {
            vstart = absl::Uniform(absl::IntervalOpen, bitgen_, 0, num_values);
          }
          // Set vlen, but leave vlen high at least once.
          int vlen = 1024;
          if (vlen_count > 0) {
            vlen = absl::Uniform(absl::IntervalOpenClosed, bitgen_, vstart,
                                 num_values);
          }
          num_values = std::min(num_values, vlen);
          // Configure vector unit for different lmul settings.
          uint32_t vtype = (kSewSettingsByByteSize[byte_sew] << 3) |
                           kLmulSettings[lmul_index];
          ConfigureVectorUnit(vtype, vlen);
          rv_vector_->set_vstart(vstart);

          // Reset Vd values, since the previous instruction execution
          // overwrites them.
          for (int i = 0; i < 8; i++) {
            auto vd_name = absl::StrCat("v", kVd + i);
            SetVectorRegisterValues<Vd>(
                {{vd_name, vd_span.subspan(vd_size * i, vd_size)}});
          }

          inst->Execute();

          if ((std::min(std::min(lmul8_vs2, lmul8_vs1), lmul8_vd) < 1) ||
              (std::max(std::max(lmul8_vs2, lmul8_vs1), lmul8_vd) > 64)) {
            EXPECT_TRUE(rv_vector_->vector_exception());
            rv_vector_->clear_vector_exception();
            continue;
          }

          EXPECT_FALSE(rv_vector_->vector_exception());
          int count = 0;
          int reg_offset = count * byte_sew / kVectorLengthInBytes;
          for (int reg = kVd + reg_offset; reg < kVd + 8; reg++) {
            for (int i = 0; i < kVectorLengthInBytes / sizeof(Vd); i++) {
              int mask_index = count >> 3;
              int mask_offset = count & 0b111;
              bool mask_value =
                  ((kA5Mask[mask_index] >> mask_offset) & 0b1) != 0;
              if ((count >= vstart) && (count < num_values)) {
                EXPECT_EQ(operation(vs2_value[count], vs1_value[count],
                                    vd_value[count], mask_value),
                          vreg_[reg]->data_buffer()->Get<Vd>(i))
                    << "mask: " << mask_value << " (" << std::hex
                    << (int64_t)vs2_value[count] << ", "
                    << (int64_t)vs1_value[count] << ")  (" << std::dec
                    << (int64_t)vs2_value[count] << ", "
                    << (int64_t)vs1_value[count] << ", "
                    << (int64_t)vd_value[count] << ")    "
                    << absl::StrCat(name, "[", count, "] != reg[", reg, "][", i,
                                    "]   lmul8(", lmul8, ")  vstart(", vstart,
                                    ")");
              } else {
                EXPECT_EQ(vd_value[count],
                          vreg_[reg]->data_buffer()->Get<Vd>(i))
                    << absl::StrCat(name, "  0 != reg[", reg, "][", i,
                                    "]   lmul8(", lmul8, ")  vstart(", vstart,
                                    ")");
              }
              count++;
            }
          }
          if (HasFailure()) return;
        }
      }
    }
  }

  // Helper function for testing vector-vector instructions that do not
  // use the value of the mask bit.
  template <typename Vd, typename Vs2, typename Vs1>
  void TernaryOpTestHelperVV(absl::string_view name, int sew, Instruction *inst,
                             std::function<Vd(Vs2, Vs1, Vd)> operation) {
    TernaryOpWithMaskTestHelperVV<Vd, Vs2, Vs1>(
        name, sew, inst,
        [operation](Vs2 vs2, Vs1 vs1, Vd vd, bool mask_value) -> Vd {
          if (mask_value) {
            return operation(vs2, vs1, vd);
          }
          return vd;
        });
  }

  // Helper function for testing vector-scalar/immediate instructions that use
  // the value of the mask bit.
  template <typename Vd, typename Vs2, typename Rs1>
  void TernaryOpWithMaskTestHelperVX(
      absl::string_view name, int sew, Instruction *inst,
      std::function<Vd(Vs2, Rs1, Vd, bool)> operation) {
    int byte_sew = sew / 8;
    if (byte_sew != sizeof(Vd) && byte_sew != sizeof(Vs2) &&
        byte_sew != sizeof(Rs1)) {
      FAIL() << name << ": selected element width != any operand types"
             << "sew: " << sew << " Vd: " << sizeof(Vd)
             << " Vs2: " << sizeof(Vs2) << " Rs1: " << sizeof(Rs1);
      return;
    }
    // Number of elements per vector register.
    constexpr int vd_size = kVectorLengthInBytes / sizeof(Vd);
    constexpr int vs2_size = kVectorLengthInBytes / sizeof(Vs2);
    // Input values for 8 registers.
    Vd vd_value[vd_size * 8];
    auto vd_span = Span<Vd>(vd_value);
    Vs2 vs2_value[vs2_size * 8];
    auto vs2_span = Span<Vs2>(vs2_value);
    AppendVectorRegisterOperands({kVs2}, {});
    AppendRegisterOperands({kRs1Name}, {});
    AppendVectorRegisterOperands({kVd, kVmask}, {kVd});
    // Initialize input values.
    FillArrayWithRandomValues<Vd>(vd_span);
    FillArrayWithRandomValues<Vs2>(vs2_span);
    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});
    for (int i = 0; i < 8; i++) {
      auto vs2_name = absl::StrCat("v", kVs2 + i);
      SetVectorRegisterValues<Vs2>(
          {{vs2_name, vs2_span.subspan(vs2_size * i, vs2_size)}});
    }
    // Iterate across the different lmul values.
    for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
      // Try different vstart values.
      for (int vstart_count = 0; vstart_count < 4; vstart_count++) {
        for (int vlen_count = 0; vlen_count < 4; vlen_count++) {
          int lmul8 = kLmul8Values[lmul_index];
          int lmul8_vd = lmul8 * sizeof(Vd) / byte_sew;
          int lmul8_vs2 = lmul8 * sizeof(Vs2) / byte_sew;
          int num_values = lmul8 * kVectorLengthInBytes / (8 * byte_sew);
          // Set vstart, but leave vstart at 0 at least once.
          int vstart = 0;
          if (vstart_count > 0) {
            vstart = absl::Uniform(absl::IntervalOpen, bitgen_, 0, num_values);
          }
          // Set vlen, but leave vlen high at least once.
          int vlen = 1024;
          if (vlen_count > 0) {
            vlen = absl::Uniform(absl::IntervalOpenClosed, bitgen_, vstart,
                                 num_values);
          }
          num_values = std::min(num_values, vlen);
          ASSERT_TRUE(vlen > vstart);
          // Configure vector unit for the different lmul settings.
          uint32_t vtype = (kSewSettingsByByteSize[byte_sew] << 3) |
                           kLmulSettings[lmul_index];
          ConfigureVectorUnit(vtype, vlen);
          rv_vector_->set_vstart(vstart);

          // Reset Vd values, since the previous instruction execution
          // overwrites them.
          for (int i = 0; i < 8; i++) {
            auto vd_name = absl::StrCat("v", kVd + i);
            SetVectorRegisterValues<Vd>(
                {{vd_name, vd_span.subspan(vd_size * i, vd_size)}});
          }

          // Generate a new rs1 value.
          RV32Register::ValueType rs1_reg_value =
              RandomValue<RV32Register::ValueType>();
          SetRegisterValues<RV32Register::ValueType>(
              {{kRs1Name, rs1_reg_value}});
          // Cast the value to the appropriate width, sign-extending if need
          // be.
          Rs1 rs1_value = static_cast<Rs1>(
              static_cast<
                  typename SameSignedType<RV32Register::ValueType, Rs1>::type>(
                  rs1_reg_value));

          inst->Execute();
          if ((std::min(lmul8_vs2, lmul8_vd) < 1) ||
              (std::max(lmul8_vs2, lmul8_vd) > 64)) {
            EXPECT_TRUE(rv_vector_->vector_exception());
            rv_vector_->clear_vector_exception();
            continue;
          }

          EXPECT_FALSE(rv_vector_->vector_exception());
          EXPECT_EQ(rv_vector_->vstart(), 0);
          int count = 0;
          for (int reg = kVd; reg < kVd + 8; reg++) {
            for (int i = 0; i < kVectorLengthInBytes / sizeof(Vd); i++) {
              int mask_index = count >> 3;
              int mask_offset = count & 0b111;
              bool mask_value =
                  ((kA5Mask[mask_index] >> mask_offset) & 0b1) != 0;
              // Compare elements that are between vstart and vlen for which
              // the mask is true.
              if ((count >= vstart) && (count < num_values)) {
                Vd expected_value =
                    operation(vs2_value[count], static_cast<Rs1>(rs1_value),
                              vd_value[count], mask_value);
                Vd inst_value = vreg_[reg]->data_buffer()->Get<Vd>(i);
                EXPECT_EQ(expected_value, inst_value)
                    << "mask: " << mask_value << " (" << std::hex
                    << (int64_t)vs2_value[count] << ", " << (int64_t)rs1_value
                    << ")  (" << std::dec << (int64_t)vs2_value[count] << ", "
                    << (int64_t)rs1_value << ", " << (int64_t)vd_value[count]
                    << ")    "
                    << absl::StrCat(name, "[", count, "] != reg[", reg, "][", i,
                                    "]   lmul8(", lmul8, ")  vstart(", vstart,
                                    ")");
              } else {
                // The others should be zero.
                EXPECT_EQ(vd_span[count], vreg_[reg]->data_buffer()->Get<Vd>(i))
                    << absl::StrCat(name, "  0 != reg[", reg, "][", i,
                                    "]   lmul8(", lmul8, ")");
              }
              count++;
            }
          }
          if (HasFailure()) return;
        }
      }
    }
  }

  // Templated helper function that tests vector-scalar instructions that do
  // not use the value of the mask bit.
  template <typename Vd, typename Vs2, typename Rs1>
  void TernaryOpTestHelperVX(absl::string_view name, int sew, Instruction *inst,
                             std::function<Vd(Vs2, Rs1, Vd)> operation) {
    TernaryOpWithMaskTestHelperVX<Vd, Vs2, Rs1>(
        name, sew, inst,
        [operation](Vs2 vs2, Rs1 rs1, Vd vd, bool mask_value) -> Vd {
          if (mask_value) {
            return operation(vs2, rs1, vd);
          }
          return vd;
        });
  }

  // Helper function for testing binary mask vector-vector instructions that
  // use the mask bit.
  template <typename Vs2, typename Vs1>
  void BinaryMaskOpWithMaskTestHelperVV(
      absl::string_view name, int sew, Instruction *inst,
      std::function<uint8_t(Vs2, Vs1, bool)> operation) {
    int byte_sew = sew / 8;
    if (byte_sew != sizeof(Vs2) && byte_sew != sizeof(Vs1)) {
      FAIL() << name << ": selected element width != any operand types"
             << "sew: " << sew << " Vs2: " << sizeof(Vs2)
             << " Vs1: " << sizeof(Vs1);
      return;
    }
    // Number of elements per vector register.
    constexpr int vs2_size = kVectorLengthInBytes / sizeof(Vs2);
    constexpr int vs1_size = kVectorLengthInBytes / sizeof(Vs1);
    // Input values for 8 registers.
    Vs2 vs2_value[vs2_size * 8];
    auto vs2_span = Span<Vs2>(vs2_value);
    Vs1 vs1_value[vs1_size * 8];
    auto vs1_span = Span<Vs1>(vs1_value);
    AppendVectorRegisterOperands({kVs2, kVs1, kVmask}, {kVd});
    // Initialize input values.
    FillArrayWithRandomValues<Vs2>(vs2_span);
    FillArrayWithRandomValues<Vs1>(vs1_span);
    // Make every third value the same (at least if the types are same sized).
    for (int i = 0; i < std::min(vs1_size, vs2_size); i += 3) {
      vs1_span[i] = static_cast<Vs1>(vs2_span[i]);
    }

    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});
    for (int i = 0; i < 8; i++) {
      auto vs2_name = absl::StrCat("v", kVs2 + i);
      SetVectorRegisterValues<Vs2>(
          {{vs2_name, vs2_span.subspan(vs2_size * i, vs2_size)}});
      auto vs1_name = absl::StrCat("v", kVs1 + i);
      SetVectorRegisterValues<Vs1>(
          {{vs1_name, vs1_span.subspan(vs1_size * i, vs1_size)}});
    }
    for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
      for (int vstart_count = 0; vstart_count < 4; vstart_count++) {
        for (int vlen_count = 0; vlen_count < 4; vlen_count++) {
          ClearVectorRegisterGroup(kVd, 8);
          int lmul8 = kLmul8Values[lmul_index];
          int lmul8_vs2 = lmul8 * sizeof(Vs2) / byte_sew;
          int num_values = lmul8 * kVectorLengthInBytes / (8 * byte_sew);
          int vstart = 0;
          if (vstart_count > 0) {
            vstart = absl::Uniform(absl::IntervalOpen, bitgen_, 0, num_values);
          }
          // Set vlen, but leave vlen high at least once.
          int vlen = 1024;
          if (vlen_count > 0) {
            vlen = absl::Uniform(absl::IntervalOpenClosed, bitgen_, vstart,
                                 num_values);
          }
          num_values = std::min(num_values, vlen);
          ASSERT_TRUE(vlen > vstart);
          // Configure vector unit for different lmul settings.
          uint32_t vtype = (kSewSettingsByByteSize[byte_sew] << 3) |
                           kLmulSettings[lmul_index];
          ConfigureVectorUnit(vtype, vlen);
          rv_vector_->set_vstart(vstart);

          inst->Execute();
          if ((lmul8_vs2 < 1) || (lmul8_vs2 > 64)) {
            EXPECT_TRUE(rv_vector_->vector_exception());
            rv_vector_->clear_vector_exception();
            continue;
          }

          EXPECT_FALSE(rv_vector_->vector_exception());
          EXPECT_EQ(rv_vector_->vstart(), 0);
          auto dest_span = vreg_[kVd]->data_buffer()->Get<uint8_t>();
          for (int i = 0; i < kVectorLengthInBytes * 8; i++) {
            int mask_index = i >> 3;
            int mask_offset = i & 0b111;
            bool mask_value = ((kA5Mask[mask_index] >> mask_offset) & 0b1) != 0;
            uint8_t inst_value = dest_span[i >> 3];
            inst_value = (inst_value >> mask_offset) & 0b1;
            if ((i >= vstart) && (i < num_values)) {
              uint8_t expected_value =
                  operation(vs2_value[i], vs1_value[i], mask_value);
              EXPECT_EQ(expected_value, inst_value) << absl::StrCat(
                  name, "[", i, "] != reg[][", i, "]   lmul8(", lmul8,
                  ")  vstart(", vstart, ")  num_values(", num_values, ")");
            } else {
              EXPECT_EQ(0, inst_value) << absl::StrCat(
                  name, "[", i, "]  0 != reg[][", i, "]   lmul8(", lmul8,
                  ")  vstart(", vstart, ")  num_values(", num_values, ")");
            }
          }
          if (HasFailure()) return;
        }
      }
    }
  }

  // Helper function for testing binary mask vector-vector instructions that do
  // not use the mask bit.
  template <typename Vs2, typename Vs1>
  void BinaryMaskOpTestHelperVV(absl::string_view name, int sew,
                                Instruction *inst,
                                std::function<uint8_t(Vs2, Vs1)> operation) {
    BinaryMaskOpWithMaskTestHelperVV<Vs2, Vs1>(
        name, sew, inst,
        [operation](Vs2 vs2, Vs1 vs1, bool mask_value) -> uint8_t {
          if (mask_value) {
            return operation(vs2, vs1);
          }
          return 0;
        });
  }

  // Helper function for testing mask vector-scalar/immediate instructions that
  // use the mask bit.
  template <typename Vs2, typename Rs1>
  void BinaryMaskOpWithMaskTestHelperVX(
      absl::string_view name, int sew, Instruction *inst,
      std::function<uint8_t(Vs2, Rs1, bool)> operation) {
    int byte_sew = sew / 8;
    if (byte_sew != sizeof(Vs2) && byte_sew != sizeof(Rs1)) {
      FAIL() << name << ": selected element width != any operand types"
             << "sew: " << sew << " Vs2: " << sizeof(Vs2)
             << " Rs1: " << sizeof(Rs1);
      return;
    }
    // Number of elements per vector register.
    constexpr int vs2_size = kVectorLengthInBytes / sizeof(Vs2);
    // Input values for 8 registers.
    Vs2 vs2_value[vs2_size * 8];
    auto vs2_span = Span<Vs2>(vs2_value);
    AppendVectorRegisterOperands({kVs2}, {});
    AppendRegisterOperands({kRs1Name}, {});
    AppendVectorRegisterOperands({kVmask}, {kVd});
    // Initialize input values.
    FillArrayWithRandomValues<Vs2>(vs2_span);
    SetVectorRegisterValues<uint8_t>(
        {{kVmaskName, Span<const uint8_t>(kA5Mask)}});
    for (int i = 0; i < 8; i++) {
      auto vs2_name = absl::StrCat("v", kVs2 + i);
      SetVectorRegisterValues<Vs2>(
          {{vs2_name, vs2_span.subspan(vs2_size * i, vs2_size)}});
    }
    for (int lmul_index = 0; lmul_index < 7; lmul_index++) {
      for (int vstart_count = 0; vstart_count < 4; vstart_count++) {
        for (int vlen_count = 0; vlen_count < 4; vlen_count++) {
          ClearVectorRegisterGroup(kVd, 8);
          int lmul8 = kLmul8Values[lmul_index];
          int lmul8_vs2 = lmul8 * sizeof(Vs2) / byte_sew;
          int num_values = lmul8 * kVectorLengthInBytes / (8 * byte_sew);
          int vstart = 0;
          if (vstart_count > 0) {
            vstart = absl::Uniform(absl::IntervalOpen, bitgen_, 0, num_values);
          }
          // Set vlen, but leave vlen high at least once.
          int vlen = 1024;
          if (vlen_count > 0) {
            vlen = absl::Uniform(absl::IntervalOpenClosed, bitgen_, vstart,
                                 num_values);
          }
          num_values = std::min(num_values, vlen);
          ASSERT_TRUE(vlen > vstart);
          // Configure vector unit for different lmul settings.
          uint32_t vtype = (kSewSettingsByByteSize[byte_sew] << 3) |
                           kLmulSettings[lmul_index];
          ConfigureVectorUnit(vtype, vlen);
          rv_vector_->set_vstart(vstart);

          // Generate a new rs1 value.
          RV32Register::ValueType rs1_reg_value =
              RandomValue<RV32Register::ValueType>();
          SetRegisterValues<RV32Register::ValueType>(
              {{kRs1Name, rs1_reg_value}});
          // Cast the value to the appropriate width, sign-extending if need be.
          Rs1 rs1_value = static_cast<Rs1>(
              static_cast<
                  typename SameSignedType<RV32Register::ValueType, Rs1>::type>(
                  rs1_reg_value));
          inst->Execute();
          if ((lmul8_vs2 < 1) || (lmul8_vs2 > 64)) {
            EXPECT_TRUE(rv_vector_->vector_exception());
            rv_vector_->clear_vector_exception();
            continue;
          }

          EXPECT_FALSE(rv_vector_->vector_exception());
          EXPECT_EQ(rv_vector_->vstart(), 0);
          auto dest_span = vreg_[kVd]->data_buffer()->Get<uint8_t>();
          for (int i = 0; i < kVectorLengthInBytes * 8; i++) {
            int mask_index = i >> 3;
            int mask_offset = i & 0b111;
            bool mask_value = ((kA5Mask[mask_index] >> mask_offset) & 0b1) != 0;
            uint8_t inst_value = dest_span[i >> 3];
            inst_value = (inst_value >> mask_offset) & 0b1;
            if ((i >= vstart) && (i < num_values)) {
              uint8_t expected_value =
                  operation(vs2_value[i], rs1_value, mask_value);
              EXPECT_EQ(expected_value, inst_value) << absl::StrCat(
                  name, "[i] != reg[0][", i, "]   lmul8(", lmul8, ")");
            } else {
              EXPECT_EQ(0, inst_value) << absl::StrCat(
                  name, "  0 != reg[0][", i, "]   lmul8(", lmul8, ")");
            }
          }
          if (HasFailure()) return;
        }
      }
    }
  }

  // Helper function for testing mask vector-vector instructions that do not
  // use the mask bit.
  template <typename Vs2, typename Vs1>
  void BinaryMaskOpTestHelperVX(absl::string_view name, int sew,
                                Instruction *inst,
                                std::function<uint8_t(Vs2, Vs1)> operation) {
    BinaryMaskOpWithMaskTestHelperVX<Vs2, Vs1>(
        name, sew, inst,
        [operation](Vs2 vs2, Vs1 vs1, bool mask_value) -> uint8_t {
          if (mask_value) {
            return operation(vs2, vs1);
          }
          return 0;
        });
  }

  // Helper function to compute the rounding output bit.
  template <typename T>
  T RoundBits(int num_bits, T lost_bits) {
    bool bit_d =
        (num_bits == 0) ? false : ((lost_bits >> (num_bits - 1)) & 0b1) != 0;
    bool bit_d_minus_1 =
        (num_bits < 2) ? false : ((lost_bits >> (num_bits - 2)) & 0b1) != 0;
    bool bits_d_minus_2_to_0 =
        (num_bits < 3) ? false
                       : (lost_bits & ~(std::numeric_limits<uint64_t>::max()
                                        << (num_bits - 2))) != 0;
    bool bits_d_minus_1_to_0 =
        (num_bits < 2) ? false
                       : (lost_bits & ~(std::numeric_limits<uint64_t>::max()
                                        << (num_bits - 1))) != 0;
    switch (rv_vector_->vxrm()) {
      case 0:
        return bit_d_minus_1;
      case 1:
        return bit_d_minus_1 & (bits_d_minus_2_to_0 | bit_d);
      case 2:
        return 0;
      case 3:
        return !bit_d & bits_d_minus_1_to_0;
      default:
        return 0;
    }
  }

  RiscVVectorState *rv_vector() const { return rv_vector_; }
  absl::Span<RVVectorRegister *> vreg() {
    return absl::Span<RVVectorRegister *>(vreg_);
  }
  absl::Span<RV32Register *> xreg() {
    return absl::Span<RV32Register *>(xreg_);
  }
  absl::BitGen &bitgen() { return bitgen_; }
  Instruction *instruction() { return instruction_; }

 protected:
  RV32Register *xreg_[32];
  RVVectorRegister *vreg_[32];
  RVFpRegister *freg_[32];
  RiscVState *state_;
  Instruction *instruction_;
  Instruction *child_instruction_;
  FlatDemandMemory *memory_;
  RiscVVectorState *rv_vector_;
  absl::BitGen bitgen_;
};

}  // namespace test
}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // MPACT_RISCV_RISCV_TEST_RISCV_VECTOR_INSTRUCTIONS_TEST_BASE_H_
