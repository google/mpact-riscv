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

#include "riscv/riscv64_decoder.h"

#include <string>

#include "googlemock/include/gmock/gmock.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "elfio/elfio.hpp"
#include "elfio/elfio_section.hpp"
#include "elfio/elfio_symbols.hpp"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/program_loader/elf_program_loader.h"

namespace {

using ::mpact::sim::riscv::RiscVXlen;

constexpr char kFileName[] = "hello_world_64.elf";

// The depot path to the test directory.
constexpr char kDepotPath[] = "riscv/test/";

using SymbolAccessor = ELFIO::symbol_section_accessor_template<ELFIO::section>;

class RiscV64DecoderTest : public testing::Test {
 protected:
  RiscV64DecoderTest()
      : state_("riscv64_test", RiscVXlen::RV64),
        memory_(0),
        loader_(&memory_),
        decoder_(&state_, &memory_) {
    const std::string input_file =
        absl::StrCat(kDepotPath, "testfiles/", kFileName);
    auto result = loader_.LoadProgram(input_file);
    CHECK_OK(result.status());
    elf_reader_.load(input_file);
    auto *symtab = elf_reader_.sections[".symtab"];
    CHECK_NE(symtab, nullptr);
    symbol_accessor_ = new SymbolAccessor(elf_reader_, symtab);
  }

  ~RiscV64DecoderTest() override { delete symbol_accessor_; }

  ELFIO::elfio elf_reader_;
  mpact::sim::riscv::RiscVState state_;
  mpact::sim::util::FlatDemandMemory memory_;
  mpact::sim::util::ElfProgramLoader loader_;
  mpact::sim::riscv::RiscV64Decoder decoder_;
  SymbolAccessor *symbol_accessor_;
};

// This test is really pretty simple. It decodes the instructions in "main".
// The goal of this test is not so much to ensure that the decoder is accurate,
// but that the decoder returns a non-null instruction object for each address
// in main, and that executing this instruction does not generate an error.
TEST_F(RiscV64DecoderTest, HelloWorldMain) {
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
    auto *inst = decoder_.DecodeInstruction(address);
    ASSERT_NE(inst, nullptr);
    inst->Execute(nullptr);
    if (state_.program_error_controller()->HasError()) {
      auto errvec = state_.program_error_controller()->GetUnmaskedErrorNames();
      for (auto &err : errvec) {
        LOG(INFO) << "Error: " << err;
        auto msgvec = state_.program_error_controller()->GetErrorMessages(err);
        for (auto &msg : msgvec) {
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
TEST_F(RiscV64DecoderTest, BadAddress) {
  auto *inst = decoder_.DecodeInstruction(0x4321);
  ASSERT_NE(inst, nullptr);
  inst->Execute(nullptr);
  inst->DecRef();
}

}  // namespace
