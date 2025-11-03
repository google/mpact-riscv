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

#include "riscv/riscv32_decoder.h"

#include <cstdint>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "elfio/elf_types.hpp"
#include "elfio/elfio.hpp"
#include "elfio/elfio_section.hpp"
#include "elfio/elfio_symbols.hpp"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/program_loader/elf_program_loader.h"
#include "riscv/riscv32g_enums.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_state.h"

namespace {

using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.
using ::mpact::sim::riscv::IsaExtensions;
using ::mpact::sim::riscv::RiscVCsrEnum;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::isa32::OpcodeEnum;

constexpr uint16_t kCNopOpcode = 0b000'0'00000'00000'01;

constexpr char kFileName[] = "hello_world.elf";

// The depot path to the test directory.
constexpr char kDepotPath[] = "riscv/test/";

using SymbolAccessor = ELFIO::symbol_section_accessor_template<ELFIO::section>;

class RiscV32DecoderTest : public testing::Test {
 protected:
  RiscV32DecoderTest()
      : state_("riscv32_test", RiscVXlen::RV32, &memory_),
        loader_(&memory_),
        decoder_(&state_, &memory_) {
    const std::string input_file =
        absl::StrCat(kDepotPath, "testfiles/", kFileName);
    auto result = loader_.LoadProgram(input_file);
    CHECK_OK(result.status());
    elf_reader_.load(input_file);
    auto* symtab = elf_reader_.sections[".symtab"];
    CHECK_NE(symtab, nullptr);
    symbol_accessor_ = new SymbolAccessor(elf_reader_, symtab);
  }

  ~RiscV32DecoderTest() override { delete symbol_accessor_; }

  ELFIO::elfio elf_reader_;
  mpact::sim::util::FlatDemandMemory memory_;
  mpact::sim::riscv::RiscVState state_;
  mpact::sim::util::ElfProgramLoader loader_;
  mpact::sim::riscv::RiscV32Decoder decoder_;
  mpact::sim::generic::DataBufferFactory db_factory_;
  SymbolAccessor* symbol_accessor_;
};

// This test is really pretty simple. It decodes the instructions in "main".
// The goal of this test is not so much to ensure that the decoder is accurate,
// but that the decoder returns a non-null instruction object for each address
// in main, and that executing this instruction does not generate an error.
TEST_F(RiscV32DecoderTest, HelloWorldMain) {
  ELFIO::Elf64_Addr value;
  ELFIO::Elf_Xword size;
  unsigned char bind;
  unsigned char type;
  ELFIO::Elf_Half section_index;
  unsigned char other;
  bool success = symbol_accessor_->get_symbol("main", value, size, bind, type,
                                              section_index, other);
  ASSERT_TRUE(success);
  uint64_t address = value;
  while (address < value + size) {
    EXPECT_FALSE(state_.program_error_controller()->HasError());
    auto* inst = decoder_.DecodeInstruction(address);
    ASSERT_NE(inst, nullptr);
    inst->Execute(nullptr);
    if (state_.program_error_controller()->HasError()) {
      auto errvec = state_.program_error_controller()->GetUnmaskedErrorNames();
      for (auto& err : errvec) {
        LOG(INFO) << "Error: " << err;
        auto msgvec = state_.program_error_controller()->GetErrorMessages(err);
        for (auto& msg : msgvec) {
          LOG(INFO) << "    " << msg;
        }
      }
    }
    EXPECT_FALSE(state_.program_error_controller()->HasError());
    state_.program_error_controller()->ClearAll();
    address += inst->size();
    inst->DecRef();
    state_.AdvanceDelayLines();
  }
}

// Even with a bad address, a valid instruction object should be returned.
TEST_F(RiscV32DecoderTest, BadAddress) {
  auto* inst = decoder_.DecodeInstruction(0x4321);
  ASSERT_NE(inst, nullptr);
  inst->Execute(nullptr);
  inst->DecRef();
}

TEST_F(RiscV32DecoderTest, BadAddressNoCompressedInstructions) {
  // Write opcode to memory.
  auto* db = db_factory_.Allocate<uint16_t>(1);
  db->Set<uint16_t>(0, kCNopOpcode);
  memory_.Store(0x4322, db);
  db->DecRef();
  // Get a handle to the MISA CSR.
  auto res = state_.csr_set()->GetCsr(*RiscVCsrEnum::kMIsa);
  CHECK_OK(res.status());
  auto* misa_csr = res.value();
  misa_csr->Set(misa_csr->GetUint64() | *IsaExtensions::kCompressed);
  // If the C bit is set, then the decoder should succeed.
  auto* inst = decoder_.DecodeInstruction(0x4322);
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->opcode(), static_cast<uint32_t>(OpcodeEnum::kCnop));
  inst->DecRef();
  misa_csr->Set(misa_csr->GetUint64() & ~*IsaExtensions::kCompressed);
  // If the C bit is not set, then the decoder should generate an invalid
  // address exception.
  inst = decoder_.DecodeInstruction(0x4322);
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->opcode(), static_cast<uint32_t>(OpcodeEnum::kNone));
  inst->DecRef();
}

}  // namespace
