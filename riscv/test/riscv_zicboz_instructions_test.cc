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

#include "riscv/riscv_zicboz_instructions.h"

#include <cstdint>
#include <type_traits>
#include <vector>

#include "absl/functional/bind_front.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"

// This file contains tests for the Zicboz `cbo.zero` instruction on both RV32
// and RV64.

namespace {

using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::ExceptionCode;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RV64Register;
using ::testing::Each;

constexpr absl::string_view kRs1Name = "x1";
constexpr uint64_t kInstAddress = 0x2468;
constexpr uint64_t kMemBase = 0x1000;
constexpr uint64_t kMaxPhysicalAddress = 0xFFFF;
// Cache block size bound to `cbo.zero` in the tests.
constexpr int kCacheBlockSizeBytes = 64;
// The starting address of the middle block within a three-block region spanning
// from `kMemBase` to `kMemBase + 3 * kCacheBlockSizeBytes` to verify that
// adjacent blocks remain untouched.
constexpr uint64_t kTargetBlock = kMemBase + kCacheBlockSizeBytes;
// Non-zero fill pattern to initialize the memory.
constexpr uint8_t kFilledValue = 0xA5;

template <typename Reg>
class RiscVZicbozInstructionTest : public testing::Test {
 protected:
  using UIntReg = typename Reg::ValueType;

  RiscVZicbozInstructionTest()
      : memory_(/*memory_size_in_units=*/0xffff'ffff'ffff'ffffULL,
                /*base_address=*/0, /*addressable_unit_size=*/1, kFilledValue),
        state_("test",
               std::is_same_v<Reg, RV32Register> ? RiscVXlen::RV32
                                                 : RiscVXlen::RV64,
               &memory_),
        instruction_(kInstAddress, &state_) {
    instruction_.set_size(4);
    instruction_.AppendSource(state_.template GetRegister<Reg>(kRs1Name)
                                  .first->CreateSourceOperand());
    SetCacheBlockSize(kCacheBlockSizeBytes);
    state_.set_on_trap([this](bool is_interrupt, uint64_t trap_value,
                              uint64_t exception_code, uint64_t epc,
                              const Instruction* inst) {
      trap_taken_ = true;
      trap_value_ = trap_value;
      trap_code_ = exception_code;
      return true;
    });
  }

  // Binds `cache_block_size` to the `cbo.zero` semantic function under test.
  void SetCacheBlockSize(int cache_block_size) {
    using CboZeroFn = void (*)(int, const Instruction*);
    constexpr CboZeroFn kCboZero = std::is_same_v<Reg, RV32Register>
                                       ? &mpact::sim::riscv::RV32::RiscVCboZero
                                       : &mpact::sim::riscv::RV64::RiscVCboZero;
    instruction_.set_semantic_function(
        absl::bind_front(kCboZero, cache_block_size));
  }

  // Sets `x1` (`rs1`) to `address` and executes `cbo.zero 0(x1)`.
  void ExecuteCboZero(UIntReg address) {
    Reg* const reg = state_.template GetRegister<Reg>(kRs1Name).first;
    DataBuffer* const db = state_.db_factory()->template Allocate<UIntReg>(1);
    db->template Set<UIntReg>(0, address);
    reg->SetDataBuffer(db);
    db->DecRef();
    instruction_.Execute(nullptr);
  }

  // Reads and returns one `kCacheBlockSizeBytes`-byte block at `block_address`.
  std::vector<uint8_t> ReadBlock(uint64_t block_address) {
    DataBuffer* const db = state_.db_factory()->Allocate(kCacheBlockSizeBytes);
    memory_.Load(block_address, db, nullptr, nullptr);
    std::vector<uint8_t> bytes(kCacheBlockSizeBytes);
    db->CopyTo(bytes.data());
    db->DecRef();
    return bytes;
  }

  mpact::sim::util::FlatDemandMemory memory_;
  RiscVState state_;
  Instruction instruction_;
  bool trap_taken_ = false;
  uint64_t trap_value_ = 0;
  uint64_t trap_code_ = 0;
};

using TestTypes = testing::Types<RV32Register, RV64Register>;
TYPED_TEST_SUITE(RiscVZicbozInstructionTest, TestTypes);

// Verifies that `cbo.zero` with a block-aligned address zeros that block and
// leaves the preceding and following blocks unmodified.
TYPED_TEST(RiscVZicbozInstructionTest, AlignedAddressZerosBlock) {
  this->ExecuteCboZero(kTargetBlock);

  EXPECT_THAT(this->ReadBlock(kTargetBlock), Each(0));
  EXPECT_THAT(this->ReadBlock(kTargetBlock - kCacheBlockSizeBytes),
              Each(kFilledValue));
  EXPECT_THAT(this->ReadBlock(kTargetBlock + kCacheBlockSizeBytes),
              Each(kFilledValue));
  EXPECT_FALSE(this->trap_taken_);
}

// Verifies that an unaligned address within a cache block is aligned down to
// zero the entire containing block.
TYPED_TEST(RiscVZicbozInstructionTest, UnalignedAddressZerosContainingBlock) {
  constexpr uint64_t kMisalignment = 0x2B;
  this->ExecuteCboZero(kTargetBlock + kMisalignment);

  EXPECT_THAT(this->ReadBlock(kTargetBlock), Each(0));
  EXPECT_THAT(this->ReadBlock(kTargetBlock - kCacheBlockSizeBytes),
              Each(kFilledValue));
  EXPECT_THAT(this->ReadBlock(kTargetBlock + kCacheBlockSizeBytes),
              Each(kFilledValue));
  EXPECT_FALSE(this->trap_taken_);
}

// Verifies that addressing the last byte of a cache block still zeros the
// entire containing block without spilling into the next block.
TYPED_TEST(RiscVZicbozInstructionTest, LastByteOfBlockZerosWholeBlock) {
  this->ExecuteCboZero(kTargetBlock + kCacheBlockSizeBytes - 1);

  EXPECT_THAT(this->ReadBlock(kTargetBlock), Each(0));
  EXPECT_THAT(this->ReadBlock(kTargetBlock - kCacheBlockSizeBytes),
              Each(kFilledValue));
  EXPECT_THAT(this->ReadBlock(kTargetBlock + kCacheBlockSizeBytes),
              Each(kFilledValue));
  EXPECT_FALSE(this->trap_taken_);
}

// Verifies that `cbo.zero` at address `0` zeros the first cache block.
TYPED_TEST(RiscVZicbozInstructionTest, ZeroBaseAddressZerosFirstBlock) {
  this->ExecuteCboZero(0);

  EXPECT_THAT(this->ReadBlock(0), Each(0));
  EXPECT_THAT(this->ReadBlock(kCacheBlockSizeBytes), Each(kFilledValue));
  EXPECT_FALSE(this->trap_taken_);
}

// Verifies that an address exceeding the maximum physical address raises a
// store access fault.
TYPED_TEST(RiscVZicbozInstructionTest, BeyondMaxPhysicalAddressTraps) {
  this->state_.set_max_physical_address(kMaxPhysicalAddress);
  constexpr uint64_t kFaultingBlock =
      kMaxPhysicalAddress + 1 + kCacheBlockSizeBytes;
  constexpr uint64_t kUnalignedOffset = 3;

  this->ExecuteCboZero(kFaultingBlock + kUnalignedOffset);

  EXPECT_TRUE(this->trap_taken_);
  EXPECT_EQ(this->trap_code_,
            static_cast<uint64_t>(ExceptionCode::kStoreAccessFault));
  EXPECT_EQ(this->trap_value_, kFaultingBlock);
  EXPECT_THAT(this->ReadBlock(kFaultingBlock), Each(kFilledValue));
}

// Verifies that `cbo.zero` zeros exactly one block of the cache block size
// bound to the semantic function rather than a fixed size.
TYPED_TEST(RiscVZicbozInstructionTest, ZerosBlockOfBoundCacheBlockSize) {
  constexpr int kSmallCacheBlockSizeBytes = 32;
  constexpr uint64_t kUnalignedOffset = 5;
  this->SetCacheBlockSize(kSmallCacheBlockSizeBytes);

  this->ExecuteCboZero(kTargetBlock + kSmallCacheBlockSizeBytes +
                       kUnalignedOffset);

  const std::vector<uint8_t> block = this->ReadBlock(kTargetBlock);
  EXPECT_THAT(absl::MakeConstSpan(block).first(kSmallCacheBlockSizeBytes),
              Each(kFilledValue));
  EXPECT_THAT(absl::MakeConstSpan(block).subspan(kSmallCacheBlockSizeBytes),
              Each(0));
  EXPECT_THAT(this->ReadBlock(kTargetBlock + kCacheBlockSizeBytes),
              Each(kFilledValue));
  EXPECT_FALSE(this->trap_taken_);
}

}  // namespace
