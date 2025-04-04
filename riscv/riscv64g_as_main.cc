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

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "elfio/elf_types.hpp"
#include "elfio/elfio.hpp"
#include "elfio/elfio_dump.hpp"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/asm/opcode_assembler_interface.h"
#include "mpact/sim/util/asm/resolver_interface.h"
#include "mpact/sim/util/asm/simple_assembler.h"
#include "re2/re2.h"
#include "riscv/riscv64g_bin_encoder_interface.h"
#include "riscv/riscv64g_encoder.h"

using ::mpact::sim::riscv::isa64::RiscV64GBinEncoderInterface;
using ::mpact::sim::riscv::isa64::Riscv64gSlotMatcher;
using ::mpact::sim::util::assembler::SimpleAssembler;

// This file implements the main function for the generated assembler for
// RiscV64G.

namespace {

using ::mpact::sim::generic::operator*;  // NOLINT(misc-unused-using-decls)
using ::mpact::sim::util::assembler::OpcodeAssemblerInterface;
using ::mpact::sim::util::assembler::RelocationInfo;
using ::mpact::sim::util::assembler::ResolverInterface;
using AddSymbolCallback =
    ::mpact::sim::util::assembler::OpcodeAssemblerInterface::AddSymbolCallback;

// This class implements the byte oriented OpcodeAssemblerInterface, converting
// from the bit interface provided by the SlotMatcher interface. Since there is
// only one slot in the RiscV64G ISA, only one slot matcher is needed.
class RiscV64GAssembler : public OpcodeAssemblerInterface {
 public:
  RiscV64GAssembler(Riscv64gSlotMatcher* matcher)
      : label_re_("^(\\S+)\\s*:"), matcher_(matcher) {};
  ~RiscV64GAssembler() override = default;
  absl::StatusOr<size_t> Encode(
      uint64_t address, absl::string_view text,
      AddSymbolCallback add_symbol_callback, ResolverInterface* resolver,
      std::vector<uint8_t>& bytes,
      std::vector<RelocationInfo>& relocations) override {
    // First check to see if there is a label, if so, add it to the symbol table
    // with the current address.
    std::string label;
    if (RE2::Consume(&text, label_re_, &label)) {
      auto status = add_symbol_callback(label, address, 0, STT_NOTYPE, 0, 0);
      if (!status.ok()) return status;
    }
    auto res = matcher_->Encode(address, text, 0, resolver, relocations);
    if (!res.status().ok()) return res.status();
    auto [value, size] = res.value();
    union {
      uint64_t i;
      uint8_t b[sizeof(uint64_t)];
    } u;
    u.i = value;
    for (int i = 0; i < size / 8; ++i) {
      bytes.push_back(u.b[i]);
    }
    return bytes.size();
  }

 private:
  RE2 label_re_;
  Riscv64gSlotMatcher* matcher_;
};

}  // namespace

// This flag dumps the info for the generated ELF file.
ABSL_FLAG(bool, dump_elf, false, "Dump the ELF file");
// Produce a relocatable file as opposed to an executable.
ABSL_FLAG(bool, c, false, "Produce a relocatable file");
// Specify the output file name.
ABSL_FLAG(std::optional<std::string>, o, std::nullopt, "Output file name");

// Supported RiscV ELF flags (TSO memory ordering and compact encodings).
enum class RiscVElfFlags {
  kNone = 0,
  kRiscvTso = 0x0001,
  kRiscvRvc = 0x0010,
};

int main(int argc, char* argv[]) {
  auto arg_vec = absl::ParseCommandLine(argc, argv);

  if (arg_vec.size() > 2) {
    std::cout << "Too many arguments\n";
    return 1;
  }

  std::istream* is;

  if (arg_vec.size() == 1) {
    is = &std::cin;
  } else {
    is = new std::fstream(arg_vec[1], std::fstream::in);
  }

  RiscV64GBinEncoderInterface bin_encoder_interface;
  Riscv64gSlotMatcher matcher(&bin_encoder_interface);
  RiscV64GAssembler riscv_64g_assembler(&matcher);
  CHECK_OK(matcher.Initialize());
  // Instantiate the assembler.
  SimpleAssembler assembler("#", ELFCLASS64, &riscv_64g_assembler);
  // Set up the abi and the machine type.
  assembler.writer().set_os_abi(ELFOSABI_LINUX);
  assembler.writer().set_machine(EM_RISCV);
  // Set the appropriate ELF header flags.
  assembler.writer().set_flags(*RiscVElfFlags::kRiscvTso |
                               *RiscVElfFlags::kRiscvRvc);
  // Perform the first pass of parsing the assembly code.
  auto status = assembler.Parse(*is);
  if (!status.ok()) {
    std::cout << "Failed to parse assembly: " << status.message() << "\n";
    return 1;
  }
  // Perform the second pass of parsing the assembly code. This pass will either
  // generate an executable or a relocatable file.
  std::string output_file_name;
  if (absl::GetFlag(FLAGS_c)) {
    status = assembler.CreateRelocatable();
    if (arg_vec.size() == 1) {
      output_file_name = "stdin.o";
    } else {
      std::string input_file_name = arg_vec[1];
      auto dot_pos = input_file_name.find_last_of('.');
      if (dot_pos == std::string::npos) {
        output_file_name = absl::StrCat(input_file_name, ".o");
      } else {
        output_file_name =
            absl::StrCat(input_file_name.substr(0, dot_pos), ".o");
      }
    }
  } else {
    status = assembler.CreateExecutable(0x1000, "main");
    output_file_name = "a.out";
  }
  if (!status.ok()) {
    std::cout << "Assembly failure: " << status.message() << "\n";
    return 1;
  }
  // Write out the output file.
  if (absl::GetFlag(FLAGS_o).has_value()) {
    output_file_name = absl::GetFlag(FLAGS_o).value();
  }
  std::ofstream output_file(output_file_name);
  if (!output_file.is_open()) {
    std::cout << "Failed to open output file: " << output_file_name << "\n";
    return 1;
  }
  status = assembler.Write(output_file);
  if (!status.ok()) {
    std::cout << "Failed to write output file: " << status.message() << "\n";
    return 1;
  }
  output_file.close();
  if (is != &std::cin) delete is;

  // Dump the ELF info if requested.
  if (absl::GetFlag(FLAGS_dump_elf)) {
    ELFIO::elfio reader;
    if (!reader.load(output_file_name)) {
      std::cout << "Failed to load output file: " << output_file_name << "\n";
      return 1;
    }

    ELFIO::dump::header(std::cout, reader);
    ELFIO::dump::section_headers(std::cout, reader);
    ELFIO::dump::segment_headers(std::cout, reader);
    ELFIO::dump::symbol_tables(std::cout, reader);
    ELFIO::dump::notes(std::cout, reader);
    ELFIO::dump::modinfo(std::cout, reader);
    ELFIO::dump::dynamic_tags(std::cout, reader);
    ELFIO::dump::section_datas(std::cout, reader);
    ELFIO::dump::segment_datas(std::cout, reader);
  }
  return 0;
}
