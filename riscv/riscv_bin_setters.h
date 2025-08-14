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

#ifndef THIRD_PARTY_MPACT_RISCV_RISCV_BIN_SETTERS_H_
#define THIRD_PARTY_MPACT_RISCV_RISCV_BIN_SETTERS_H_

#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mpact/sim/util/asm/opcode_assembler_interface.h"
#include "mpact/sim/util/asm/resolver_interface.h"
#include "re2/re2.h"
#include "riscv/riscv_getter_helpers.h"

// This file contains various setters for the RiscV binary encoder that is used
// by the assembler to map from operand text strings to integer values.

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::util::assembler::RelocationInfo;
using ::mpact::sim::util::assembler::ResolverInterface;

// Initializer lists for maps for different subsets of registers, mapping from
// the register name to the register number.
constexpr std::initializer_list<const std::pair<absl::string_view, uint64_t>>
    kRegisterList = {
        {"x0", 0},   {"x1", 1},   {"x2", 2},   {"x3", 3},   {"x4", 4},
        {"x5", 5},   {"x6", 6},   {"x7", 7},   {"x8", 8},   {"x9", 9},
        {"x10", 10}, {"x11", 11}, {"x12", 12}, {"x13", 13}, {"x14", 14},
        {"x15", 15}, {"x16", 16}, {"x17", 17}, {"x18", 18}, {"x19", 19},
        {"x20", 20}, {"x21", 21}, {"x22", 22}, {"x23", 23}, {"x24", 24},
        {"x25", 25}, {"x26", 26}, {"x27", 27}, {"x28", 28}, {"x29", 29},
        {"x30", 30}, {"x31", 31}, {"zero", 0}, {"ra", 1},   {"sp", 2},
        {"gp", 3},   {"tp", 4},   {"t0", 5},   {"t1", 6},   {"t2", 7},
        {"s0", 8},   {"s1", 9},   {"a0", 10},  {"a1", 11},  {"a2", 12},
        {"a3", 13},  {"a4", 14},  {"a5", 15},  {"a6", 16},  {"a7", 17},
        {"s2", 18},  {"s3", 19},  {"s4", 20},  {"s5", 21},  {"s6", 22},
        {"s7", 23},  {"s8", 24},  {"s9", 25},  {"s10", 26}, {"s11", 27},
        {"t3", 28},  {"t4", 29},  {"t5", 30},  {"t6", 31}};

constexpr std::initializer_list<const std::pair<absl::string_view, uint64_t>>
    kCRegisterList = {
        {"x8", 8},   {"x9", 9},   {"x10", 10}, {"x11", 11},
        {"x12", 12}, {"x13", 13}, {"x14", 14}, {"x15", 15},
        {"s0", 8},   {"s1", 9},   {"a0", 10},  {"a1", 11},
        {"a2", 12},  {"a3", 13},  {"a4", 14},  {"a5", 15},
};

constexpr std::initializer_list<const std::pair<absl::string_view, uint64_t>>
    kFRegisterList = {
        {"f0", 0},   {"f1", 1},   {"f2", 2},    {"f3", 3},    {"f4", 4},
        {"f5", 5},   {"f6", 6},   {"f7", 7},    {"f8", 8},    {"f9", 9},
        {"f10", 10}, {"f11", 11}, {"f12", 12},  {"f13", 13},  {"f14", 14},
        {"f15", 15}, {"f16", 16}, {"f17", 17},  {"f18", 18},  {"f19", 19},
        {"f20", 20}, {"f21", 21}, {"f22", 22},  {"f23", 23},  {"f24", 24},
        {"f25", 25}, {"f26", 26}, {"f27", 27},  {"f28", 28},  {"f29", 29},
        {"f30", 30}, {"f31", 31}, {"ft0", 0},   {"ft1", 1},   {"ft2", 2},
        {"ft3", 3},  {"ft4", 4},  {"ft5", 5},   {"ft6", 6},   {"ft7", 7},
        {"fs0", 8},  {"fs1", 9},  {"fa0", 10},  {"fa1", 11},  {"fa2", 12},
        {"fa3", 13}, {"fa4", 14}, {"fa5", 15},  {"fa6", 16},  {"fa7", 17},
        {"fs2", 18}, {"fs3", 19}, {"fs4", 20},  {"fs5", 21},  {"fs6", 22},
        {"fs7", 23}, {"fs8", 24}, {"fs9", 25},  {"fs10", 26}, {"fs11", 27},
        {"ft8", 28}, {"ft9", 29}, {"ft10", 30}, {"ft11", 31}};

constexpr std::initializer_list<const std::pair<absl::string_view, uint64_t>>
    kDRegisterList = {
        {"d0", 0},   {"d1", 1},   {"d2", 2},    {"d3", 3},    {"d4", 4},
        {"d5", 5},   {"d6", 6},   {"d7", 7},    {"d8", 8},    {"d9", 9},
        {"d10", 10}, {"d11", 11}, {"d12", 12},  {"d13", 13},  {"d14", 14},
        {"d15", 15}, {"d16", 16}, {"d17", 17},  {"d18", 18},  {"d19", 19},
        {"d20", 20}, {"d21", 21}, {"d22", 22},  {"d23", 23},  {"d24", 24},
        {"d25", 25}, {"d26", 26}, {"d27", 27},  {"d28", 28},  {"d29", 29},
        {"d30", 30}, {"d31", 31}, {"dt0", 0},   {"dt1", 1},   {"dt2", 2},
        {"dt3", 3},  {"dt4", 4},  {"dt5", 5},   {"dt6", 6},   {"dt7", 7},
        {"ds0", 8},  {"ds1", 9},  {"da0", 10},  {"da1", 11},  {"da2", 12},
        {"da3", 13}, {"da4", 14}, {"da5", 15},  {"da6", 16},  {"da7", 17},
        {"ds2", 18}, {"ds3", 19}, {"ds4", 20},  {"ds5", 21},  {"ds6", 22},
        {"ds7", 23}, {"ds8", 24}, {"ds9", 25},  {"ds10", 26}, {"ds11", 27},
        {"dt8", 28}, {"dt9", 29}, {"dt10", 30}, {"dt11", 31}};

constexpr std::initializer_list<const std::pair<absl::string_view, uint64_t>>
    kDCRegisterList = {
        {"d8", 8},   {"d9", 9},   {"d10", 10}, {"d11", 11},
        {"d12", 12}, {"d13", 13}, {"d14", 14}, {"d15", 15},
        {"ds0", 8},  {"ds1", 9},  {"da0", 10}, {"da1", 11},
        {"da2", 12}, {"da3", 13}, {"da4", 14}, {"da5", 15},
};

// This is the initializer list for the map from CSR register names to register
// numbers.
constexpr std::initializer_list<const std::pair<absl::string_view, uint64_t>>
    kCsrRegisterList = {
        {"fflags", 0x001},
        {"frm", 0x002},
        {"fcsr", 0x003},
        {"cycle", 0xc00},
        {"time", 0xc01},
        {"instret", 0xc02},
        {"hpmcounter3", 0xc03},
        {"hpmcounter4", 0xc04},
        {"hpmcounter5", 0xc05},
        {"hpmcounter6", 0xc06},
        {"hpmcounter7", 0xc07},
        {"hpmcounter8", 0xc08},
        {"hpmcounter9", 0xc09},
        {"hpmcounter10", 0xc0a},
        {"hpmcounter11", 0xc0b},
        {"hpmcounter12", 0xc0c},
        {"hpmcounter13", 0xc0d},
        {"hpmcounter14", 0xc0e},
        {"hpmcounter15", 0xc0f},
        {"hpmcounter16", 0xc10},
        {"hpmcounter17", 0xc11},
        {"hpmcounter18", 0xc12},
        {"hpmcounter19", 0xc13},
        {"hpmcounter20", 0xc14},
        {"hpmcounter21", 0xc15},
        {"hpmcounter22", 0xc16},
        {"hpmcounter23", 0xc17},
        {"hpmcounter24", 0xc18},
        {"hpmcounter25", 0xc19},
        {"hpmcounter26", 0xc1a},
        {"hpmcounter27", 0xc1b},
        {"hpmcounter28", 0xc1c},
        {"hpmcounter29", 0xc1d},
        {"hpmcounter30", 0xc1e},
        {"hpmcounter31", 0xc1f},
        {"cycleh", 0xc80},
        {"timeh", 0xc81},
        {"instreth", 0xc82},
        {"hpmcounter3h", 0xc83},
        {"hpmcounter4h", 0xc84},
        {"hpmcounter5h", 0xc85},
        {"hpmcounter6h", 0xc86},
        {"hpmcounter7h", 0xc87},
        {"hpmcounter8h", 0xc88},
        {"hpmcounter9h", 0xc89},
        {"hpmcounter10h", 0xc8a},
        {"hpmcounter11h", 0xc8b},
        {"hpmcounter12h", 0xc8c},
        {"hpmcounter13h", 0xc8d},
        {"hpmcounter14h", 0xc8e},
        {"hpmcounter15h", 0xc8f},
        {"hpmcounter16h", 0xc90},
        {"hpmcounter17h", 0xc91},
        {"hpmcounter18h", 0xc92},
        {"hpmcounter19h", 0xc93},
        {"hpmcounter20h", 0xc94},
        {"hpmcounter2h1", 0xc95},
        {"hpmcounter22h", 0xc96},
        {"hpmcounter23h", 0xc97},
        {"hpmcounter24h", 0xc98},
        {"hpmcounter25h", 0xc99},
        {"hpmcounter26h", 0xc9a},
        {"hpmcounter27h", 0xc9b},
        {"hpmcounter28h", 0xc9c},
        {"hpmcounter29h", 0xc9d},
        {"hpmcounter30h", 0xc9e},
        {"hpmcounter3h1", 0xc9f},
        {"sstatus", 0x100},
        {"sie", 0x104},
        {"stvec", 0x105},
        {"stcounteren", 0x106},
        {"senvcfg", 0x10a},
        {"scountinhibit", 0x120},
        {"sscratch", 0x140},
        {"sepc", 0x141},
        {"scause", 0x142},
        {"stval", 0x143},
        {"sip", 0x144},
        {"scountovf", 0xda0},
        {"satp", 0x180},
        {"scontext", 0x5a8},
        {"sstateen0", 0x10c},
        {"sstateen1", 0x10d},
        {"sstateen2", 0x10e},
        {"sstateen3", 0x10f},
        {"hstatus", 0x600},
        {"hedeleg", 0x602},
        {"hideleg", 0x603},
        {"hie", 0x604},
        {"hcounteren", 0x606},
        {"hgeie", 0x607},
        {"hedelegh", 0x612},
        {"htval", 0x643},
        {"hip", 0x644},
        {"hvip", 0x645},
        {"htinst", 0x64a},
        {"hgeip", 0xe12},
        {"henvcfg", 0x60a},
        {"henvcfgh", 0x61a},
        {"hgatp", 0x680},
        {"hcontext", 0x6a8},
        {"htimedelta", 0x605},
        {"htimedeltah", 0x615},
        {"hstateen0", 0x60c},
        {"hstateen1", 0x60d},
        {"hstateen2", 0x60e},
        {"hstateen3", 0x60f},
        {"hstateen0h", 0x61c},
        {"hstateen1h", 0x61d},
        {"hstateen2h", 0x61e},
        {"hstateen3h", 0x61f},
        {"vsstatus", 0x200},
        {"vsie", 0x204},
        {"vstvec", 0x205},
        {"vsscratch", 0x240},
        {"vsepc", 0x241},
        {"vscause", 0x242},
        {"vstval", 0x243},
        {"vsip", 0x244},
        {"vsatp", 0x280},
        {"mvendorid", 0xf11},
        {"marchid", 0xf12},
        {"mimpid", 0xf13},
        {"mhartid", 0xf14},
        {"mconfigptr", 0xf15},
        {"mstatus", 0x300},
        {"misa", 0x301},
        {"medeleg", 0x302},
        {"mideleg", 0x303},
        {"mie", 0x304},
        {"mcounteren", 0x306},
        {"mstatush", 0x310},
        {"medelegh", 0x312},
        {"mscratch", 0x340},
        {"mepc", 0x341},
        {"mcause", 0x342},
        {"mtval", 0x343},
        {"mip", 0x344},
        {"mtinst", 0x34a},
        {"mtval2", 0x34b},
        {"menvcfg", 0x30a},
        {"menvcfgh", 0x31a},
        {"mseccfg", 0x747},
        {"mseccfgh", 0x757},
        {"pmpcfg0", 0x3a0},
        {"pmpcfg1", 0x3a1},
        {"pmpcfg2", 0x3a2},
        {"pmpcfg3", 0x3a3},
        {"pmpcfg4", 0x3a4},
        {"pmpcfg5", 0x3a5},
        {"pmpcfg6", 0x3a6},
        {"pmpcfg7", 0x3a7},
        {"pmpcfg8", 0x3a8},
        {"pmpcfg9", 0x3a9},
        {"pmpcfg10", 0x3aa},
        {"pmpcfg11", 0x3ab},
        {"pmpcfg12", 0x3ac},
        {"pmpcfg13", 0x3ad},
        {"pmpcfg14", 0x3ae},
        {"pmpcfg15", 0x3af},
        {"pmpaddr0", 0x3b0},
        {"pmpaddr1", 0x3b1},
        {"pmpaddr2", 0x3b2},
        {"pmpaddr3", 0x3b3},
        {"pmpaddr4", 0x3b4},
        {"pmpaddr5", 0x3b5},
        {"pmpaddr6", 0x3b6},
        {"pmpaddr7", 0x3b7},
        {"pmpaddr8", 0x3b8},
        {"pmpaddr9", 0x3b9},
        {"pmpaddr10", 0x3ba},
        {"pmpaddr11", 0x3bb},
        {"pmpaddr12", 0x3bc},
        {"pmpaddr13", 0x3bd},
        {"pmpaddr14", 0x3be},
        {"pmpaddr15", 0x3bf},
        {"pmpaddr16", 0x3c0},
        {"pmpaddr17", 0x3c1},
        {"pmpaddr18", 0x3c2},
        {"pmpaddr19", 0x3c3},
        {"pmpaddr20", 0x3c4},
        {"pmpaddr21", 0x3c5},
        {"pmpaddr22", 0x3c6},
        {"pmpaddr23", 0x3c7},
        {"pmpaddr24", 0x3c8},
        {"pmpaddr25", 0x3c9},
        {"pmpaddr26", 0x3ca},
        {"pmpaddr27", 0x3cb},
        {"pmpaddr28", 0x3cc},
        {"pmpaddr29", 0x3cd},
        {"pmpaddr30", 0x3ce},
        {"pmpaddr31", 0x3cf},
        {"pmpaddr32", 0x3d0},
        {"pmpaddr33", 0x3d1},
        {"pmpaddr34", 0x3d2},
        {"pmpaddr35", 0x3d3},
        {"pmpaddr36", 0x3d4},
        {"pmpaddr37", 0x3d5},
        {"pmpaddr38", 0x3d6},
        {"pmpaddr39", 0x3d7},
        {"pmpaddr40", 0x3d8},
        {"pmpaddr41", 0x3d9},
        {"pmpaddr42", 0x3da},
        {"pmpaddr43", 0x3db},
        {"pmpaddr44", 0x3dc},
        {"pmpaddr45", 0x3dd},
        {"pmpaddr46", 0x3de},
        {"pmpaddr47", 0x3df},
        {"pmpaddr48", 0x3e0},
        {"pmpaddr49", 0x3e1},
        {"pmpaddr50", 0x3e2},
        {"pmpaddr51", 0x3e3},
        {"pmpaddr52", 0x3e4},
        {"pmpaddr53", 0x3e5},
        {"pmpaddr54", 0x3e6},
        {"pmpaddr55", 0x3e7},
        {"pmpaddr56", 0x3e8},
        {"pmpaddr57", 0x3e9},
        {"pmpaddr58", 0x3ea},
        {"pmpaddr59", 0x3eb},
        {"pmpaddr60", 0x3ec},
        {"pmpaddr61", 0x3ed},
        {"pmpaddr62", 0x3ee},
        {"pmpaddr63", 0x3ef},
        {"mstateen0", 0x30c},
        {"mstateen1", 0x30d},
        {"mstateen2", 0x30e},
        {"mstateen3", 0x30f},
        {"mstateen0h", 0x31c},
        {"mstateen1h", 0x31d},
        {"mstateen2h", 0x31e},
        {"mstateen3h", 0x31f},
        {"mnscratch", 0x740},
        {"mnepc", 0x741},
        {"mncause", 0x742},
        {"mnstatus", 0x743},
        {"mcycle", 0xb00},
        {"minstret", 0xb02},
        {"mhpmcounter3", 0xb03},
        {"mhpmcounter4", 0xb04},
        {"mhpmcounter5", 0xb05},
        {"mhpmcounter6", 0xb06},
        {"mhpmcounter7", 0xb07},
        {"mhpmcounter8", 0xb08},
        {"mhpmcounter9", 0xb09},
        {"mhpmcounter10", 0xb0a},
        {"mhpmcounter11", 0xb0b},
        {"mhpmcounter12", 0xb0c},
        {"mhpmcounter13", 0xb0d},
        {"mhpmcounter14", 0xb0e},
        {"mhpmcounter15", 0xb0f},
        {"mhpmcounter16", 0xb10},
        {"mhpmcounter17", 0xb11},
        {"mhpmcounter18", 0xb12},
        {"mhpmcounter19", 0xb13},
        {"mhpmcounter20", 0xb14},
        {"mhpmcounter21", 0xb15},
        {"mhpmcounter22", 0xb16},
        {"mhpmcounter23", 0xb17},
        {"mhpmcounter24", 0xb18},
        {"mhpmcounter25", 0xb19},
        {"mhpmcounter26", 0xb1a},
        {"mhpmcounter27", 0xb1b},
        {"mhpmcounter28", 0xb1c},
        {"mhpmcounter29", 0xb1d},
        {"mhpmcounter30", 0xb1e},
        {"mhpmcounter31", 0xb1f},
        {"mcycleh", 0xb80},
        {"minstreth", 0xb82},
        {"mhpmcounter3h", 0xb83},
        {"mhpmcounter4h", 0xb84},
        {"mhpmcounter5h", 0xb85},
        {"mhpmcounter6h", 0xb86},
        {"mhpmcounter7h", 0xb87},
        {"mhpmcounter8h", 0xb88},
        {"mhpmcounter9h", 0xb89},
        {"mhpmcounter10h", 0xb8a},
        {"mhpmcounter11h", 0xb8b},
        {"mhpmcounter12h", 0xb8c},
        {"mhpmcounter13h", 0xb8d},
        {"mhpmcounter14h", 0xb8e},
        {"mhpmcounter15h", 0xb8f},
        {"mhpmcounter16h", 0xb90},
        {"mhpmcounter17h", 0xb91},
        {"mhpmcounter18h", 0xb92},
        {"mhpmcounter19h", 0xb93},
        {"mhpmcounter20h", 0xb94},
        {"mhpmcounter21h", 0xb95},
        {"mhpmcounter22h", 0xb96},
        {"mhpmcounter23h", 0xb97},
        {"mhpmcounter24h", 0xb98},
        {"mhpmcounter25h", 0xb99},
        {"mhpmcounter26h", 0xb9a},
        {"mhpmcounter27h", 0xb9b},
        {"mhpmcounter28h", 0xb9c},
        {"mhpmcounter29h", 0xb9d},
        {"mhpmcounter30h", 0xb9e},
        {"mhpmcounter31h", 0xb9f},
        {"mcountinhibit", 0x320},
        {"mhpmevent3", 0x323},
        {"mhpmevent4", 0x324},
        {"mhpmevent5", 0x325},
        {"mhpmevent6", 0x326},
        {"mhpmevent7", 0x327},
        {"mhpmevent8", 0x328},
        {"mhpmevent9", 0x329},
        {"mhpmevent10", 0x32a},
        {"mhpmevent11", 0x32b},
        {"mhpmevent12", 0x32c},
        {"mhpmevent13", 0x32d},
        {"mhpmevent14", 0x32e},
        {"mhpmevent15", 0x32f},
        {"mhpmevent16", 0x330},
        {"mhpmevent17", 0x331},
        {"mhpmevent18", 0x332},
        {"mhpmevent19", 0x333},
        {"mhpmevent20", 0x334},
        {"mhpmevent21", 0x335},
        {"mhpmevent22", 0x336},
        {"mhpmevent23", 0x337},
        {"mhpmevent24", 0x338},
        {"mhpmevent25", 0x339},
        {"mhpmevent26", 0x33a},
        {"mhpmevent27", 0x33b},
        {"mhpmevent28", 0x33c},
        {"mhpmevent29", 0x33d},
        {"mhpmevent30", 0x33e},
        {"mhpmevent31", 0x33f},
        {"mhpmevent3h", 0x723},
        {"mhpmevent4h", 0x724},
        {"mhpmevent5h", 0x725},
        {"mhpmevent6h", 0x726},
        {"mhpmevent7h", 0x727},
        {"mhpmevent8h", 0x728},
        {"mhpmevent9h", 0x729},
        {"mhpmevent10h", 0x72a},
        {"mhpmevent11h", 0x72b},
        {"mhpmevent12h", 0x72c},
        {"mhpmevent13h", 0x72d},
        {"mhpmevent14h", 0x72e},
        {"mhpmevent15h", 0x72f},
        {"mhpmevent16h", 0x730},
        {"mhpmevent17h", 0x731},
        {"mhpmevent18h", 0x732},
        {"mhpmevent19h", 0x733},
        {"mhpmevent20h", 0x734},
        {"mhpmevent21h", 0x735},
        {"mhpmevent22h", 0x736},
        {"mhpmevent23h", 0x737},
        {"mhpmevent24h", 0x738},
        {"mhpmevent25h", 0x739},
        {"mhpmevent26h", 0x73a},
        {"mhpmevent27h", 0x73b},
        {"mhpmevent28h", 0x73c},
        {"mhpmevent29h", 0x73d},
        {"mhpmevent30h", 0x73e},
        {"mhpmevent31h", 0x73f},
};

// A helper function to convert a text string to an integer. The function takes
// either numeric literals (hexadecimal or decimal), symbol names, or relocation
// functions, e.g., %hi(<symbol name>).
template <typename T>
absl::StatusOr<T> SimpleTextToInt(absl::string_view op_text,
                                  ResolverInterface* resolver) {
  T value;
  static RE2 hex_re("^\\s*0x([0-9a-fA-F]+)\\s*$");
  static RE2 dec_re("^\\s*(-?[0-9]+)\\s*$");
  static RE2 relo_re("^\\s*\\%[a-zA-Z0-9_]+\\s*\\(([a-zA-Z0-9_]+)\\s*\\)\\s*$");
  static RE2 symbol_re("^\\s*([a-zA-Z0-9_]+)\\s*$");
  std::string str;
  std::string text(op_text);
  // First see if the operand is a relocation function, and extract the text
  // argument. A relocation function is on the form of %name(arg).
  if (RE2::FullMatch(op_text, relo_re, &str)) {
    text = str;
  }
  // Extract the hex immediate.
  if (RE2::FullMatch(text, hex_re, &str)) {
    if (absl::SimpleHexAtoi(str, &value)) return value;
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid hexadecimal immediate: ", text));
  }
  // Extract the decimal immediate.
  if (RE2::FullMatch(text, dec_re, &str)) {
    if (absl::SimpleAtoi(str, &value)) return value;
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid decimal immediate: ", text));
  }
  // Extract the symbol.
  if (RE2::FullMatch(text, symbol_re, &str)) {
    if (resolver != nullptr) {
      auto res = resolver->Resolve(str);
      if (!res.ok()) {
        return res.status();
      }
      return static_cast<T>(res.value());
    }
  }
  return absl::InvalidArgumentError(absl::StrCat("Invalid argument: ", text));
}

using ValueMap = absl::flat_hash_map<absl::string_view, uint64_t>;

// This function adds the bin setters for the source operands to the given map.
template <typename Enum, typename Map, typename Encoder>
void AddRiscvSourceOpBinSetters(Map& map) {
  Insert(map, *Enum::kAAq,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map = {{"", 0}, {".aq", 1}};
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid source operand: ", text));
           }
           return Encoder::AType::InsertAq(iter->second, 0ULL);
         });
  Insert(map, *Enum::kARl,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map = {{"", 0}, {".rl", 1}};
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid source operand: ", text));
           }
           return Encoder::AType::InsertRl(iter->second, 0ULL);
         });
  Insert(map, *Enum::kBImm12,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<uint32_t>(text, resolver);
           if (!res.ok()) return res.status();
           uint32_t delta = res.value() - address;
           return Encoder::BType::InsertBImm(delta, 0ULL);
         });
  Insert(map, *Enum::kC3drs2,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kDCRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid source operand: ", text));
           }
           return Encoder::CS::InsertCsRs2(iter->second, 0ULL);
         });
  Insert(map, *Enum::kC3rs1,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kCRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid source operand: ", text));
           }
           return Encoder::CL::InsertClRs1(iter->second, 0ULL);
         });
  Insert(map, *Enum::kC3rs2,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kCRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid source operand: ", text));
           }
           return Encoder::CS::InsertCsRs2(iter->second, 0ULL);
         });
  Insert(map, *Enum::kCSRUimm5,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<uint32_t>(text, resolver);
           if (!res.ok()) return res.status();
           return Encoder::IType::InsertIUimm5(res.value(), 0ULL);
         });
  Insert(map, *Enum::kCdrs2,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kDRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid source operand: ", text));
           }
           return Encoder::CR::InsertRs2(iter->second, 0ULL);
         });
  Insert(map, *Enum::kCrs1,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid source operand: ", text));
           }
           return Encoder::CR::InsertRs1(iter->second, 0ULL);
         });
  Insert(map, *Enum::kCrs2,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid source operand: ", text));
           }
           return Encoder::CR::InsertRs2(iter->second, 0ULL);
         });
  Insert(map, *Enum::kCsr,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kCsrRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid source operand: ", text));
           }
           return Encoder::IType::InsertUImm12(iter->second, 0ULL);
         });
  Insert(map, *Enum::kDrs1,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kDRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid source operand: ", text));
           }
           return Encoder::RType::InsertRs1(iter->second, 0ULL);
         });
  Insert(map, *Enum::kDrs2,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kDRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid source operand: ", text));
           }
           return Encoder::RType::InsertRs2(iter->second, 0ULL);
         });
  Insert(map, *Enum::kDrs3,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kDRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid source operand: ", text));
           }
           return Encoder::R4Type::InsertRs3(iter->second, 0ULL);
         });
  Insert(map, *Enum::kFrs1,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kFRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid source operand: ", text));
           }
           return Encoder::RType::InsertRs1(iter->second, 0ULL);
         });
  Insert(map, *Enum::kFrs2,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kFRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid source operand: ", text));
           }
           return Encoder::RType::InsertRs1(iter->second, 0ULL);
         });
  Insert(map, *Enum::kFrs3,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kFRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid source operand: ", text));
           }
           return Encoder::R4Type::InsertRs3(iter->second, 0ULL);
         });
  Insert(map, *Enum::kICbImm8,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<int32_t>(text, resolver);
           if (!res.ok()) return res.status();
           uint32_t delta = res.value() - address;
           return Encoder::CB::InsertBimm(delta, 0ULL);
         });
  Insert(map, *Enum::kICiImm6,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<int32_t>(text, resolver);
           if (!res.ok()) return res.status();
           return Encoder::CI::InsertImm6(res.value(), 0ULL);
         });
  Insert(map, *Enum::kICiImm612,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<int32_t>(text, resolver);
           if (!res.ok()) return res.status();
           return Encoder::CI::InsertImm18(res.value(), 0ULL);
         });
  Insert(map, *Enum::kICiImm6x16,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<int32_t>(text, resolver);
           if (!res.ok()) return res.status();
           return Encoder::CI::InsertCiImm10(res.value(), 0ULL);
         });
  Insert(map, *Enum::kICiUimm6,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<int32_t>(text, resolver);
           if (!res.ok()) return res.status();
           return Encoder::CI::InsertUimm6(res.value(), 0ULL);
         });
  Insert(map, *Enum::kICiUimm6x4,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<int32_t>(text, resolver);
           if (!res.ok()) return res.status();
           return Encoder::CI::InsertCiImmW(res.value(), 0ULL);
         });
  Insert(map, *Enum::kICiUimm6x8,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<int32_t>(text, resolver);
           if (!res.ok()) return res.status();
           return Encoder::CI::InsertCiImmD(res.value(), 0ULL);
         });
  Insert(map, *Enum::kICiwUimm8x4,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<int32_t>(text, resolver);
           if (!res.ok()) return res.status();
           return Encoder::CIW::InsertCiwImm10(res.value(), 0ULL);
         });
  Insert(map, *Enum::kICjImm11,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<int32_t>(text, resolver);
           if (!res.ok()) return res.status();
           auto delta = res.value() - address;
           return Encoder::CJ::InsertJimm(delta, 0ULL);
         });
  Insert(map, *Enum::kIClUimm5x4,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<uint32_t>(text, resolver);
           if (!res.ok()) return res.status();
           return Encoder::CL::InsertClImmW(res.value(), 0ULL);
         });
  Insert(map, *Enum::kIClUimm5x8,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<int32_t>(text, resolver);
           if (!res.ok()) return res.status();
           return Encoder::CL::InsertClImmD(res.value(), 0ULL);
         });
  Insert(map, *Enum::kICssUimm6x4,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<uint32_t>(text, resolver);
           if (!res.ok()) return res.status();
           return Encoder::CS::InsertCsImmW(res.value(), 0ULL);
         });
  Insert(map, *Enum::kICssUimm6x8,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<uint32_t>(text, resolver);
           if (!res.ok()) return res.status();
           return Encoder::CS::InsertCsImmD(res.value(), 0ULL);
         });
  Insert(map, *Enum::kIImm12,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<int32_t>(text, resolver);
           if (!res.ok()) return res.status();
           return Encoder::IType::InsertImm12(res.value(), 0ULL);
         });
  Insert(map, *Enum::kIUimm5,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<uint32_t>(text, resolver);
           if (!res.ok()) return res.status();
           return Encoder::RType::InsertRUimm5(res.value(), 0ULL);
         });
  Insert(map, *Enum::kIUimm6,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<uint32_t>(text, resolver);
           if (!res.ok()) return res.status();
           return Encoder::RSType::InsertRUimm6(res.value(), 0ULL);
         });
  Insert(map, *Enum::kJImm12,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<int32_t>(text, resolver);
           if (!res.ok()) return res.status();
           return Encoder::IType::InsertImm12(res.value(), 0ULL);
         });
  Insert(map, *Enum::kJImm20,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<int32_t>(text, resolver);
           if (!res.ok()) return res.status();
           uint32_t delta = res.value() - address;
           auto value = Encoder::JType::InsertJImm(delta, 0ULL);
           return value;
         });
  Insert(map, *Enum::kPred,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<uint32_t>(text, resolver);
           if (!res.ok()) return res.status();
           return Encoder::Fence::InsertPred(res.value(), 0ULL);
         });
  Insert(map, *Enum::kRd,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid source operand: ", text));
           }
           return Encoder::RType::InsertRd(iter->second, 0ULL);
         });
  Insert(map, *Enum::kRm,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid source operand: ", text));
           }
           return Encoder::R4Type::InsertRs3(iter->second, 0ULL);
         });
  Insert(map, *Enum::kRs1,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid source operand: ", text));
           }
           return Encoder::RType::InsertRs1(iter->second, 0ULL);
         });
  Insert(map, *Enum::kRs2,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid source operand: ", text));
           }
           return Encoder::RType::InsertRs2(iter->second, 0ULL);
         });
  Insert(map, *Enum::kSImm12,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<uint32_t>(text, resolver);
           if (!res.ok()) return res.status();
           return Encoder::SType::InsertSImm(res.value(), 0ULL);
         });
  Insert(map, *Enum::kSucc,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<uint32_t>(text, resolver);
           if (!res.ok()) return res.status();
           return Encoder::Fence::InsertSucc(res.value(), 0ULL);
         });
  Insert(map, *Enum::kUImm20,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           auto res = SimpleTextToInt<uint32_t>(text, resolver);
           if (!res.ok()) return res.status();
           return Encoder::UType::InsertUImm(res.value(), 0ULL);
         });
}

// This function adds the destination operand setters for the RiscV ISA to the
// given map.
template <typename Enum, typename Map, typename Encoder>
void AddRiscvDestOpBinSetters(Map& map) {
  Insert(map, *Enum::kC3drd,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kDCRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid destination operand: ", text));
           }
           return Encoder::CL::InsertClRd(iter->second, 0ULL);
         });
  Insert(map, *Enum::kC3rd,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kCRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid destination operand: ", text));
           }
           return Encoder::CL::InsertClRd(iter->second, 0ULL);
         });
  Insert(map, *Enum::kC3rs1,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kCRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid destination operand: ", text));
           }
           return Encoder::CS::InsertCsRs1(iter->second, 0ULL);
         });
  Insert(map, *Enum::kCsr,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kCsrRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid source operand: ", text));
           }
           return Encoder::IType::InsertUImm12(iter->second, 0ULL);
         });
  Insert(map, *Enum::kDrd,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kDRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid destination operand: ", text));
           }
           return Encoder::RType::InsertRd(iter->second, 0ULL);
         });
  Insert(map, *Enum::kFrd,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kFRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid destination operand: ", text));
           }
           return Encoder::RType::InsertRd(iter->second, 0ULL);
         });
  Insert(map, *Enum::kRd,
         [](uint64_t address, absl::string_view text,
            ResolverInterface* resolver) -> absl::StatusOr<uint64_t> {
           static ValueMap map(kRegisterList);
           auto iter = map.find(text);
           if (iter == map.end()) {
             return absl::InvalidArgumentError(
                 absl::StrCat("Invalid destination operand: ", text));
           }
           return Encoder::RType::InsertRd(iter->second, 0ULL);
         });
}

// These functions add the appropriate relocation entry to the relocations
// vector if the operand (in text) requires it.
namespace internal {

absl::Status RelocateAddiIImm12(uint64_t address, absl::string_view text,
                                ResolverInterface* resolver,
                                std::vector<RelocationInfo>& relocations);
absl::Status RelocateJJImm20(uint64_t address, absl::string_view text,
                             ResolverInterface* resolver,
                             std::vector<RelocationInfo>& relocations);
absl::Status RelocateJrJImm12(uint64_t address, absl::string_view text,
                              ResolverInterface* resolver,
                              std::vector<RelocationInfo>& relocations);
absl::Status RelocateLuiUImm20(uint64_t address, absl::string_view text,
                               ResolverInterface* resolver,
                               std::vector<RelocationInfo>& relocations);
absl::Status RelocateSdSImm12(uint64_t address, absl::string_view text,
                              ResolverInterface* resolver,
                              std::vector<RelocationInfo>& relocations);
absl::Status RelocateAuipcUImm20(uint64_t address, absl::string_view text,
                                 ResolverInterface* resolver,
                                 std::vector<RelocationInfo>& relocations);

}  // namespace internal

// This function adds the source operand relocation setters for the RiscV ISA to
// the given map. Notice that the key in the map is the tuple consisting of the
// opcode and the source operand enum values.
template <typename OpcodeEnum, typename SourceOpEnum, typename Map>
void AddRiscvSourceOpRelocationSetters(Map& map) {
  Insert(map, OpcodeEnum::kAddi, SourceOpEnum::kIImm12,
         internal::RelocateAddiIImm12);
  Insert(map, OpcodeEnum::kJal, SourceOpEnum::kJImm20,
         internal::RelocateJJImm20);
  Insert(map, OpcodeEnum::kJ, SourceOpEnum::kJImm20, internal::RelocateJJImm20);
  Insert(map, OpcodeEnum::kJr, SourceOpEnum::kJImm12,
         internal::RelocateJrJImm12);
  Insert(map, OpcodeEnum::kLui, SourceOpEnum::kUImm20,
         internal::RelocateLuiUImm20);
  Insert(map, OpcodeEnum::kSd, SourceOpEnum::kSImm12,
         internal::RelocateSdSImm12);
  Insert(map, OpcodeEnum::kJalr, SourceOpEnum::kJImm12,
         internal::RelocateJrJImm12);
  Insert(map, OpcodeEnum::kAuipc, SourceOpEnum::kUImm20,
         internal::RelocateAuipcUImm20);
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact

#endif  // THIRD_PARTY_MPACT_RISCV_RISCV_BIN_SETTERS_H_
