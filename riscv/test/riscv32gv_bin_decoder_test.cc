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

#include "riscv/riscv32gv_bin_decoder.h"

#include "googlemock/include/gmock/gmock.h"
#include "riscv/riscv32gv_enums.h"

namespace {

// This file contains tests for verifying that the vector instructions in
// RiscV32GV are decoded properly by the generated decoder, thus validating
// the binary coder descriptions.

using OpcodeEnum = mpact::sim::riscv::isa32v::OpcodeEnum;
using mpact::sim::riscv::encoding::DecodeRiscV32GV;

// Constexpr for vector instruction opcodes for RV32GV instructions grouped by
// isa group.

// RV32V
// Vector config instructions
constexpr uint32_t kVsetvli_xn = 0b0'00000000000'00001'111'00000'1010111;
constexpr uint32_t kVsetvli_nz = 0b0'00000000000'00000'111'00001'1010111;
constexpr uint32_t kVsetvli_zz = 0b0'00000000000'00000'111'00000'1010111;
constexpr uint32_t kVsetivli = 0b11'0000000000'00000'111'00000'1010111;
constexpr uint32_t kVsetvl_xn = 0b1000000'00000'00001'111'00000'1010111;
constexpr uint32_t kVsetvl_nz = 0b1000000'00000'00000'111'00001'1010111;
constexpr uint32_t kVsetvl_zz = 0b1000000'00000'00000'111'00000'1010111;
// Vector load instructions
// Unit stride.
constexpr uint32_t kVle8 = 0b0000000'00000'00000'000'00000'000'0111;
constexpr uint32_t kVle16 = 0b0000000'00000'00000'101'00000'000'0111;
constexpr uint32_t kVle32 = 0b0000000'00000'00000'110'00000'000'0111;
constexpr uint32_t kVle64 = 0b0000000'00000'00000'111'00000'000'0111;
// Load vector mask.
constexpr uint32_t kVlm = 0b0000000'01011'00000'000'00000'000'0111;
// Load unit stride, fault first.
constexpr uint32_t kVle8ff = 0b0000000'10000'00000'000'00000'000'0111;
constexpr uint32_t kVle16ff = 0b0000000'10000'00000'101'00000'000'0111;
constexpr uint32_t kVle32ff = 0b0000000'10000'00000'110'00000'000'0111;
constexpr uint32_t kVle64ff = 0b0000000'10000'00000'111'00000'000'0111;
// Load whole register.
constexpr uint32_t kVl1re8 = 0b000'000'1'01000'00000'000'00000'000'0111;
constexpr uint32_t kVl1re16 = 0b000'000'1'01000'00000'101'00000'000'0111;
constexpr uint32_t kVl1re32 = 0b000'000'1'01000'00000'110'00000'000'0111;
constexpr uint32_t kVl1re64 = 0b000'000'1'01000'00000'111'00000'000'0111;
constexpr uint32_t kVl2re8 = 0b001'000'1'01000'00000'000'00000'000'0111;
constexpr uint32_t kVl2re16 = 0b001'000'1'01000'00000'101'00000'000'0111;
constexpr uint32_t kVl2re32 = 0b001'000'1'01000'00000'110'00000'000'0111;
constexpr uint32_t kVl2re64 = 0b001'000'1'01000'00000'111'00000'000'0111;
constexpr uint32_t kVl4re8 = 0b011'000'1'01000'00000'000'00000'000'0111;
constexpr uint32_t kVl4re16 = 0b011'000'1'01000'00000'101'00000'000'0111;
constexpr uint32_t kVl4re32 = 0b011'000'1'01000'00000'110'00000'000'0111;
constexpr uint32_t kVl4re64 = 0b011'000'1'01000'00000'111'00000'000'0111;
constexpr uint32_t kVl8re8 = 0b111'000'1'01000'00000'000'00000'000'0111;
constexpr uint32_t kVl8re16 = 0b111'000'1'01000'00000'101'00000'000'0111;
constexpr uint32_t kVl8re32 = 0b111'000'1'01000'00000'110'00000'000'0111;
constexpr uint32_t kVl8re64 = 0b111'000'1'01000'00000'111'00000'000'0111;
// Load strided.
constexpr uint32_t kVlse8 = 0b0000'10'0'00000'00000'000'00000'000'0111;
constexpr uint32_t kVlse16 = 0b0000'10'0'00000'00000'101'00000'000'0111;
constexpr uint32_t kVlse32 = 0b0000'10'0'00000'00000'110'00000'000'0111;
constexpr uint32_t kVlse64 = 0b0000'10'0'00000'00000'111'00000'000'0111;
// Load indexed, unordered.
constexpr uint32_t kVluxei8 = 0b0000'01'0'00000'00000'000'00000'000'0111;
constexpr uint32_t kVluxei16 = 0b0000'01'0'00000'00000'101'00000'000'0111;
constexpr uint32_t kVluxei32 = 0b0000'01'0'00000'00000'110'00000'000'0111;
constexpr uint32_t kVluxei64 = 0b0000'01'0'00000'00000'111'00000'000'0111;
// Load indexed, ordered.
constexpr uint32_t kVloxei8 = 0b0000'11'0'00000'00000'000'00000'000'0111;
constexpr uint32_t kVloxei16 = 0b0000'11'0'00000'00000'101'00000'000'0111;
constexpr uint32_t kVloxei32 = 0b0000'11'0'00000'00000'110'00000'000'0111;
constexpr uint32_t kVloxei64 = 0b0000'11'0'00000'00000'111'00000'000'0111;
// Load segment, unit stride.
constexpr uint32_t kVlsegNFe8 = 0b000'0'00'1'00000'00000'000'00000'000'0111;
constexpr uint32_t kVlsegNFe16 = 0b000'0'00'1'00000'00000'101'00000'000'0111;
constexpr uint32_t kVlsegNFe32 = 0b000'0'00'1'00000'00000'110'00000'000'0111;
constexpr uint32_t kVlsegNFe64 = 0b000'0'00'1'00000'00000'111'00000'000'0111;
// Load segment, strided.
constexpr uint32_t kVlssegNFe8 = 0b000'0'10'1'00000'00000'000'00000'000'0111;
constexpr uint32_t kVlssegNFe16 = 0b000'0'10'1'00000'00000'101'00000'000'0111;
constexpr uint32_t kVlssegNFe32 = 0b000'0'10'1'00000'00000'110'00000'000'0111;
constexpr uint32_t kVlssegNFe64 = 0b000'0'10'1'00000'00000'111'00000'000'0111;
// Load segment, indexed, unordered.
constexpr uint32_t kVluxsegNFei8 = 0b000'0'01'1'00000'00000'000'00000'000'0111;
constexpr uint32_t kVluxsegNFei16 = 0b000'0'01'1'00000'00000'101'00000'000'0111;
constexpr uint32_t kVluxsegNFei32 = 0b000'0'01'1'00000'00000'110'00000'000'0111;
constexpr uint32_t kVluxsegNFei64 = 0b000'0'01'1'00000'00000'111'00000'000'0111;
// Load segment, indexed, ordered.
constexpr uint32_t kVloxsegNFei8 = 0b000'0'11'1'00000'00000'000'00000'000'0111;
constexpr uint32_t kVloxsegNFei16 = 0b000'0'11'1'00000'00000'101'00000'000'0111;
constexpr uint32_t kVloxsegNFei32 = 0b000'0'11'1'00000'00000'110'00000'000'0111;
constexpr uint32_t kVloxsegNFei64 = 0b000'0'11'1'00000'00000'111'00000'000'0111;

// Vector store instructions
// Unit stride.
constexpr uint32_t kVse8 = 0b0000000'00000'00000'000'00000'0100111;
constexpr uint32_t kVse16 = 0b0000000'00000'00000'101'00000'0100111;
constexpr uint32_t kVse32 = 0b0000000'00000'00000'110'00000'0100111;
constexpr uint32_t kVse64 = 0b0000000'00000'00000'111'00000'0100111;
// Store vector mask.
constexpr uint32_t kVsm = 0b0000000'01011'00000'000'00000'0100111;
// Store, unit stride, fault first.
constexpr uint32_t kVse8ff = 0b0000000'10000'00000'000'00000'0100111;
constexpr uint32_t kVse16ff = 0b0000000'10000'00000'101'00000'0100111;
constexpr uint32_t kVse32ff = 0b0000000'10000'00000'110'00000'0100111;
constexpr uint32_t kVse64ff = 0b0000000'10000'00000'111'00000'0100111;
// Store whole register.
constexpr uint32_t kVs1re8 = 0b000'000'1'01000'00000'000'00000'010'0111;
constexpr uint32_t kVs1re16 = 0b000'000'1'01000'00000'101'00000'010'0111;
constexpr uint32_t kVs1re32 = 0b000'000'1'01000'00000'110'00000'010'0111;
constexpr uint32_t kVs1re64 = 0b000'000'1'01000'00000'111'00000'010'0111;
constexpr uint32_t kVs2re8 = 0b001'000'1'01000'00000'000'00000'010'0111;
constexpr uint32_t kVs2re16 = 0b001'000'1'01000'00000'101'00000'010'0111;
constexpr uint32_t kVs2re32 = 0b001'000'1'01000'00000'110'00000'010'0111;
constexpr uint32_t kVs2re64 = 0b001'000'1'01000'00000'111'00000'010'0111;
constexpr uint32_t kVs4re8 = 0b011'000'1'01000'00000'000'00000'010'0111;
constexpr uint32_t kVs4re16 = 0b011'000'1'01000'00000'101'00000'010'0111;
constexpr uint32_t kVs4re32 = 0b011'000'1'01000'00000'110'00000'010'0111;
constexpr uint32_t kVs4re64 = 0b011'000'1'01000'00000'111'00000'010'0111;
constexpr uint32_t kVs8re8 = 0b111'000'1'01000'00000'000'00000'010'0111;
constexpr uint32_t kVs8re16 = 0b111'000'1'01000'00000'101'00000'010'0111;
constexpr uint32_t kVs8re32 = 0b111'000'1'01000'00000'110'00000'010'0111;
constexpr uint32_t kVs8re64 = 0b111'000'1'01000'00000'111'00000'010'0111;
// Store strided.
constexpr uint32_t kVsse8 = 0b0000'10'0'00000'00000'000'00000'0100111;
constexpr uint32_t kVsse16 = 0b0000'10'0'00000'00000'101'00000'0100111;
constexpr uint32_t kVsse32 = 0b0000'10'0'00000'00000'110'00000'0100111;
constexpr uint32_t kVsse64 = 0b0000'10'0'00000'00000'111'00000'0100111;
// Store indexed, unordered.
constexpr uint32_t kVsuxei8 = 0b0000'01'0'00000'00000'000'00000'0100111;
constexpr uint32_t kVsuxei16 = 0b0000'01'0'00000'00000'101'00000'0100111;
constexpr uint32_t kVsuxei32 = 0b0000'01'0'00000'00000'110'00000'0100111;
constexpr uint32_t kVsuxei64 = 0b0000'01'0'00000'00000'111'00000'0100111;
// Stored indexed, ordered.
constexpr uint32_t kVsoxei8 = 0b0000'11'0'00000'00000'000'00000'0100111;
constexpr uint32_t kVsoxei16 = 0b0000'11'0'00000'00000'101'00000'0100111;
constexpr uint32_t kVsoxei32 = 0b0000'11'0'00000'00000'110'00000'0100111;
constexpr uint32_t kVsoxei64 = 0b0000'11'0'00000'00000'111'00000'0100111;
// Store segment, unit stride.
// Store segment, strided.
// Store segment, indexed, unordered.
// Store segment, indexed, ordered.
// Integer vector-vector, vector-scalar OPIVV, OPIVX, OPIVI
constexpr uint32_t kVaddVv = 0b000000'0'0000000000'000'00000'1010111;
constexpr uint32_t kVaddVx = 0b000000'0'0000000000'100'00000'1010111;
constexpr uint32_t kVaddVi = 0b000000'0'0000000000'011'00000'1010111;
constexpr uint32_t kVsubVv = 0b000010'0'0000000000'000'00000'1010111;
constexpr uint32_t kVsubVx = 0b000010'0'0000000000'100'00000'1010111;
constexpr uint32_t kVrsubVx = 0b000011'0'0000000000'100'00000'1010111;
constexpr uint32_t kVrsubVi = 0b000011'0'0000000000'011'00000'1010111;
constexpr uint32_t kVminuVv = 0b000100'0'0000000000'000'00000'1010111;
constexpr uint32_t kVminuVx = 0b000100'0'0000000000'100'00000'1010111;
constexpr uint32_t kVminVv = 0b000101'0'0000000000'000'00000'1010111;
constexpr uint32_t kVminVx = 0b000101'0'0000000000'100'00000'1010111;
constexpr uint32_t kVmaxuVv = 0b000110'0'0000000000'000'00000'1010111;
constexpr uint32_t kVmaxuVx = 0b000110'0'0000000000'100'00000'1010111;
constexpr uint32_t kVmaxVv = 0b000111'0'0000000000'000'00000'1010111;
constexpr uint32_t kVmaxVx = 0b000111'0'0000000000'100'00000'1010111;
constexpr uint32_t kVandVv = 0b001001'0'0000000000'000'00000'1010111;
constexpr uint32_t kVandVx = 0b001001'0'0000000000'100'00000'1010111;
constexpr uint32_t kVandVi = 0b001001'0'0000000000'011'00000'1010111;
constexpr uint32_t kVorVv = 0b001010'0'0000000000'000'00000'1010111;
constexpr uint32_t kVorVx = 0b001010'0'0000000000'100'00000'1010111;
constexpr uint32_t kVorVi = 0b001010'0'0000000000'011'00000'1010111;
constexpr uint32_t kVxorVv = 0b001011'0'0000000000'000'00000'1010111;
constexpr uint32_t kVxorVx = 0b001011'0'0000000000'100'00000'1010111;
constexpr uint32_t kVxorVi = 0b001011'0'0000000000'011'00000'1010111;
constexpr uint32_t kVrgatherVv = 0b001100'0'0000000000'000'00000'1010111;
constexpr uint32_t kVrgatherVx = 0b001100'0'0000000000'100'00000'1010111;
constexpr uint32_t kVrgatherVi = 0b001100'0'0000000000'011'00000'1010111;
constexpr uint32_t kVrgatherei16Vv = 0b001110'0'0000000000'000'00000'1010111;
constexpr uint32_t kVslideupVx = 0b001110'0'0000000000'100'00000'1010111;
constexpr uint32_t kVslideupVi = 0b001110'0'0000000000'011'00000'1010111;
constexpr uint32_t kVslidedownVx = 0b001111'0'0000000000'100'00000'1010111;
constexpr uint32_t kVslidedownVi = 0b001111'0'0000000000'011'00000'1010111;
constexpr uint32_t kVadcVv = 0b010000'0'0000000000'000'00001'1010111;
constexpr uint32_t kVadcVx = 0b010000'0'0000000000'100'00001'1010111;
constexpr uint32_t kVadcVi = 0b010000'0'0000000000'011'00001'1010111;
constexpr uint32_t kVmadcVv = 0b010001'0'0000000000'000'00000'1010111;
constexpr uint32_t kVmadcVx = 0b010001'0'0000000000'100'00000'1010111;
constexpr uint32_t kVmadcVi = 0b010001'0'0000000000'011'00000'1010111;
constexpr uint32_t kVsbcVv = 0b010010'0'0000000000'000'00001'1010111;
constexpr uint32_t kVsbcVx = 0b010010'0'0000000000'100'00001'1010111;
constexpr uint32_t kVmsbcVv = 0b010011'0'0000000000'000'00000'1010111;
constexpr uint32_t kVmsbcVx = 0b010011'0'0000000000'100'00000'1010111;
constexpr uint32_t kVmergeVv = 0b010111'0'0000000000'000'00000'1010111;
constexpr uint32_t kVmergeVx = 0b010111'0'0000000000'100'00000'1010111;
constexpr uint32_t kVmergeVi = 0b010111'0'0000000000'011'00000'1010111;
constexpr uint32_t kVmvVv = 0b010111'1'0000000000'000'00000'1010111;
constexpr uint32_t kVmvVx = 0b010111'1'0000000000'100'00000'1010111;
constexpr uint32_t kVmvVi = 0b010111'1'0000000000'011'00000'1010111;
constexpr uint32_t kVmseqVv = 0b011000'0'0000000000'000'00000'1010111;
constexpr uint32_t kVmseqVx = 0b011000'0'0000000000'100'00000'1010111;
constexpr uint32_t kVmseqVi = 0b011000'0'0000000000'011'00000'1010111;
constexpr uint32_t kVmsneVv = 0b011001'0'0000000000'000'00000'1010111;
constexpr uint32_t kVmsneVx = 0b011001'0'0000000000'100'00000'1010111;
constexpr uint32_t kVmsneVi = 0b011001'0'0000000000'011'00000'1010111;
constexpr uint32_t kVmsltuVv = 0b011010'0'0000000000'000'00000'1010111;
constexpr uint32_t kVmsltuVx = 0b011010'0'0000000000'100'00000'1010111;
constexpr uint32_t kVmsltVv = 0b011011'0'0000000000'000'00000'1010111;
constexpr uint32_t kVmsltVx = 0b011011'0'0000000000'100'00000'1010111;
constexpr uint32_t kVmsleuVv = 0b011100'0'0000000000'000'00000'1010111;
constexpr uint32_t kVmsleuVx = 0b011100'0'0000000000'100'00000'1010111;
constexpr uint32_t kVmsleuVi = 0b011100'0'0000000000'011'00000'1010111;
constexpr uint32_t kVmsleVv = 0b011101'0'0000000000'000'00000'1010111;
constexpr uint32_t kVmsleVx = 0b011101'0'0000000000'100'00000'1010111;
constexpr uint32_t kVmsleVi = 0b011101'0'0000000000'011'00000'1010111;
constexpr uint32_t kVmsgtuVx = 0b011110'0'0000000000'100'00000'1010111;
constexpr uint32_t kVmsgtuVi = 0b011110'0'0000000000'011'00000'1010111;
constexpr uint32_t kVmsgtVx = 0b011111'0'0000000000'100'00000'1010111;
constexpr uint32_t kVmsgtVi = 0b011111'0'0000000000'011'00000'1010111;
constexpr uint32_t kVsadduVv = 0b100000'0'0000000000'000'00000'1010111;
constexpr uint32_t kVsadduVx = 0b100000'0'0000000000'100'00000'1010111;
constexpr uint32_t kVsadduVi = 0b100000'0'0000000000'011'00000'1010111;
constexpr uint32_t kVsaddVv = 0b100001'0'0000000000'000'00000'1010111;
constexpr uint32_t kVsaddVx = 0b100001'0'0000000000'100'00000'1010111;
constexpr uint32_t kVsaddVi = 0b100001'0'0000000000'011'00000'1010111;
constexpr uint32_t kVssubuVv = 0b100010'0'0000000000'000'00000'1010111;
constexpr uint32_t kVssubuVx = 0b100010'0'0000000000'100'00000'1010111;
constexpr uint32_t kVssubVv = 0b100011'0'0000000000'000'00000'1010111;
constexpr uint32_t kVssubVx = 0b100011'0'0000000000'100'00000'1010111;
constexpr uint32_t kVsllVv = 0b100101'0'0000000000'000'00000'1010111;
constexpr uint32_t kVsllVx = 0b100101'0'0000000000'100'00000'1010111;
constexpr uint32_t kVsllVi = 0b100101'0'0000000000'011'00000'1010111;
constexpr uint32_t kVsmulVv = 0b100111'0'0000000000'000'00000'1010111;
constexpr uint32_t kVsmulVx = 0b100111'0'0000000000'100'00000'1010111;
constexpr uint32_t kVmv1rVi = 0b100111'0'00000'00000'011'00000'1010111;
constexpr uint32_t kVmv2rVi = 0b100111'0'00000'00001'011'00000'1010111;
constexpr uint32_t kVmv4rVi = 0b100111'0'00000'00011'011'00000'1010111;
constexpr uint32_t kVmv8rVi = 0b100111'0'00000'00111'011'00000'1010111;
constexpr uint32_t kVsrlVv = 0b101000'0'0000000000'000'00000'1010111;
constexpr uint32_t kVsrlVx = 0b101000'0'0000000000'100'00000'1010111;
constexpr uint32_t kVsrlVi = 0b101000'0'0000000000'011'00000'1010111;
constexpr uint32_t kVsraVv = 0b101001'0'0000000000'000'00000'1010111;
constexpr uint32_t kVsraVx = 0b101001'0'0000000000'100'00000'1010111;
constexpr uint32_t kVsraVi = 0b101001'0'0000000000'011'00000'1010111;
constexpr uint32_t kVssrlVv = 0b101010'0'0000000000'000'00000'1010111;
constexpr uint32_t kVssrlVx = 0b101010'0'0000000000'100'00000'1010111;
constexpr uint32_t kVssrlVi = 0b101010'0'0000000000'011'00000'1010111;
constexpr uint32_t kVssraVv = 0b101011'0'0000000000'000'00000'1010111;
constexpr uint32_t kVssraVx = 0b101011'0'0000000000'100'00000'1010111;
constexpr uint32_t kVssraVi = 0b101011'0'0000000000'011'00000'1010111;
constexpr uint32_t kVnsrlVv = 0b101100'0'0000000000'000'00000'1010111;
constexpr uint32_t kVnsrlVx = 0b101100'0'0000000000'100'00000'1010111;
constexpr uint32_t kVnsrlVi = 0b101100'0'0000000000'011'00000'1010111;
constexpr uint32_t kVnsraVv = 0b101101'0'0000000000'000'00000'1010111;
constexpr uint32_t kVnsraVx = 0b101101'0'0000000000'100'00000'1010111;
constexpr uint32_t kVnsraVi = 0b101101'0'0000000000'011'00000'1010111;
constexpr uint32_t kVnclipuVv = 0b101110'0'0000000000'000'00000'1010111;
constexpr uint32_t kVnclipuVx = 0b101110'0'0000000000'100'00000'1010111;
constexpr uint32_t kVnclipuVi = 0b101110'0'0000000000'011'00000'1010111;
constexpr uint32_t kVnclipVv = 0b101111'0'0000000000'000'00000'1010111;
constexpr uint32_t kVnclipVx = 0b101111'0'0000000000'100'00000'1010111;
constexpr uint32_t kVnclipVi = 0b101111'0'0000000000'011'00000'1010111;
constexpr uint32_t kVwredsumuVv = 0b110000'0'0000000000'000'00000'1010111;
constexpr uint32_t kVwredsumVv = 0b110001'0'0000000000'000'00000'1010111;

// Integer Vector OPMVV, OPMVX
constexpr uint32_t kVredsumVv = 0b000000'0'0000000000'010'00000'1010111;
constexpr uint32_t kVredandVv = 0b000001'0'0000000000'010'00000'1010111;
constexpr uint32_t kVredorVv = 0b000010'0'0000000000'010'00000'1010111;
constexpr uint32_t kVredxorVv = 0b000011'0'0000000000'010'00000'1010111;
constexpr uint32_t kVredminuVv = 0b000100'0'0000000000'010'00000'1010111;
constexpr uint32_t kVredminVv = 0b000101'0'0000000000'010'00000'1010111;
constexpr uint32_t kVredmaxuVv = 0b000110'0'0000000000'010'00000'1010111;
constexpr uint32_t kVredmaxVv = 0b000111'0'0000000000'010'00000'1010111;
constexpr uint32_t kVaadduVv = 0b001000'0'0000000000'010'00000'1010111;
constexpr uint32_t kVaadduVx = 0b001000'0'0000000000'110'00000'1010111;
constexpr uint32_t kVaaddVv = 0b001001'0'0000000000'010'00000'1010111;
constexpr uint32_t kVaaddVx = 0b001001'0'0000000000'110'00000'1010111;
constexpr uint32_t kVasubuVv = 0b001010'0'0000000000'010'00000'1010111;
constexpr uint32_t kVasubuVx = 0b001010'0'0000000000'110'00000'1010111;
constexpr uint32_t kVasubVv = 0b001011'0'0000000000'010'00000'1010111;
constexpr uint32_t kVasubVx = 0b001011'0'0000000000'110'00000'1010111;
constexpr uint32_t kVslide1upVx = 0b001110'0'0000000000'110'00000'1010111;
constexpr uint32_t kVslide1downVx = 0b001111'0'0000000000'110'00000'1010111;
constexpr uint32_t kVcompressVv = 0b010111'0'0000000000'010'00000'1010111;
constexpr uint32_t kVmandnotVv = 0b011000'0'0000000000'010'00000'1010111;
constexpr uint32_t kVmandVv = 0b011001'0'0000000000'010'00000'1010111;
constexpr uint32_t kVmorVv = 0b011010'0'0000000000'010'00000'1010111;
constexpr uint32_t kVmxorVv = 0b011011'0'0000000000'010'00000'1010111;
constexpr uint32_t kVmornotVv = 0b011100'0'0000000000'010'00000'1010111;
constexpr uint32_t kVmnandVv = 0b011101'0'0000000000'010'00000'1010111;
constexpr uint32_t kVmnorVv = 0b011110'0'0000000000'010'00000'1010111;
constexpr uint32_t kVmxnorVv = 0b011111'0'0000000000'010'00000'1010111;

constexpr uint32_t kVdivuVv = 0b100000'0'0000000000'010'00000'1010111;
constexpr uint32_t kVdivuVx = 0b100000'0'0000000000'110'00000'1010111;
constexpr uint32_t kVdivVv = 0b100001'0'0000000000'010'00000'1010111;
constexpr uint32_t kVdivVx = 0b100001'0'0000000000'110'00000'1010111;
constexpr uint32_t kVremuVv = 0b100010'0'0000000000'010'00000'1010111;
constexpr uint32_t kVremuVx = 0b100010'0'0000000000'110'00000'1010111;
constexpr uint32_t kVremVv = 0b100011'0'0000000000'010'00000'1010111;
constexpr uint32_t kVremVx = 0b100011'0'0000000000'110'00000'1010111;
constexpr uint32_t kVmulhuVv = 0b100100'0'0000000000'010'00000'1010111;
constexpr uint32_t kVmulhuVx = 0b100100'0'0000000000'110'00000'1010111;
constexpr uint32_t kVmulVv = 0b100101'0'0000000000'010'00000'1010111;
constexpr uint32_t kVmulVx = 0b100101'0'0000000000'110'00000'1010111;
constexpr uint32_t kVmulhsuVv = 0b100110'0'0000000000'010'00000'1010111;
constexpr uint32_t kVmulhsuVx = 0b100110'0'0000000000'110'00000'1010111;
constexpr uint32_t kVmulhVv = 0b100111'0'0000000000'010'00000'1010111;
constexpr uint32_t kVmulhVx = 0b100111'0'0000000000'110'00000'1010111;
constexpr uint32_t kVmaddVv = 0b101001'0'0000000000'010'00000'1010111;
constexpr uint32_t kVmaddVx = 0b101001'0'0000000000'110'00000'1010111;
constexpr uint32_t kVnmsubVv = 0b101011'0'0000000000'010'00000'1010111;
constexpr uint32_t kVnmsubVx = 0b101011'0'0000000000'110'00000'1010111;
constexpr uint32_t kVmaccVv = 0b101101'0'0000000000'010'00000'1010111;
constexpr uint32_t kVmaccVx = 0b101101'0'0000000000'110'00000'1010111;
constexpr uint32_t kVnmsacVv = 0b101111'0'0000000000'010'00000'1010111;
constexpr uint32_t kVnmsacVx = 0b101111'0'0000000000'110'00000'1010111;
constexpr uint32_t kVwadduVv = 0b110000'0'0000000000'010'00000'1010111;
constexpr uint32_t kVwadduVx = 0b110000'0'0000000000'110'00000'1010111;
constexpr uint32_t kVwaddVv = 0b110001'0'0000000000'010'00000'1010111;
constexpr uint32_t kVwaddVx = 0b110001'0'0000000000'110'00000'1010111;
constexpr uint32_t kVwsubuVv = 0b110010'0'0000000000'010'00000'1010111;
constexpr uint32_t kVwsubuVx = 0b110010'0'0000000000'110'00000'1010111;
constexpr uint32_t kVwsubVv = 0b110011'0'0000000000'010'00000'1010111;
constexpr uint32_t kVwsubVx = 0b110011'0'0000000000'110'00000'1010111;
constexpr uint32_t kVwadduWVv = 0b110100'0'0000000000'010'00000'1010111;
constexpr uint32_t kVwadduWVx = 0b110100'0'0000000000'110'00000'1010111;
constexpr uint32_t kVwaddWVv = 0b110101'0'0000000000'010'00000'1010111;
constexpr uint32_t kVwaddWVx = 0b110101'0'0000000000'110'00000'1010111;
constexpr uint32_t kVwsubuWVv = 0b110110'0'0000000000'010'00000'1010111;
constexpr uint32_t kVwsubuWVx = 0b110110'0'0000000000'110'00000'1010111;
constexpr uint32_t kVwsubWVv = 0b110111'0'0000000000'010'00000'1010111;
constexpr uint32_t kVwsubWVx = 0b110111'0'0000000000'110'00000'1010111;
constexpr uint32_t kVwmuluVv = 0b111000'0'0000000000'010'00000'1010111;
constexpr uint32_t kVwmuluVx = 0b111000'0'0000000000'110'00000'1010111;
constexpr uint32_t kVwmulsuVv = 0b111010'0'0000000000'010'00000'1010111;
constexpr uint32_t kVwmulsuVx = 0b111010'0'0000000000'110'00000'1010111;
constexpr uint32_t kVwmulVv = 0b111011'0'0000000000'010'00000'1010111;
constexpr uint32_t kVwmulVx = 0b111011'0'0000000000'110'00000'1010111;
constexpr uint32_t kVwmaccuVv = 0b111100'0'0000000000'010'00000'1010111;
constexpr uint32_t kVwmaccuVx = 0b111100'0'0000000000'110'00000'1010111;
constexpr uint32_t kVwmaccVv = 0b111101'0'0000000000'010'00000'1010111;
constexpr uint32_t kVwmaccVx = 0b111101'0'0000000000'110'00000'1010111;
constexpr uint32_t kVwmaccusVv = 0b111110'0'0000000000'010'00000'1010111;
constexpr uint32_t kVwmaccusVx = 0b111110'0'0000000000'110'00000'1010111;
constexpr uint32_t kVwmaccsuVv = 0b111111'0'0000000000'010'00000'1010111;
constexpr uint32_t kVwmaccsuVx = 0b111111'0'0000000000'110'00000'1010111;

// FP Vector OPFVV, OPFVF
constexpr uint32_t kVfaddVv = 0b000000'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfaddVf = 0b000000'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfredusumVv = 0b000001'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfsubVv = 0b000010'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfsubVf = 0b000010'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfredosumVv = 0b000011'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfminVv = 0b000100'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfminVf = 0b000100'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfredminVv = 0b000101'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfmaxVv = 0b000110'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfmaxVf = 0b000110'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfredmaxVv = 0b000111'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfsgnjVv = 0b001000'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfsgnjVf = 0b001000'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfsgnjnVv = 0b001001'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfsgnjnVf = 0b001001'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfsgnjxVv = 0b001010'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfsgnjxVf = 0b001010'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfslide1upVf = 0b001110'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfslide1downVf = 0b001111'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfmergeVf = 0b010111'0'0000000000'101'00000'1010111;
constexpr uint32_t kVmfeqVv = 0b011000'0'0000000000'001'00000'1010111;
constexpr uint32_t kVmfeqVf = 0b011000'0'0000000000'101'00000'1010111;
constexpr uint32_t kVmfleVv = 0b011001'0'0000000000'001'00000'1010111;
constexpr uint32_t kVmfleVf = 0b011001'0'0000000000'101'00000'1010111;
constexpr uint32_t kVmfltVv = 0b011011'0'0000000000'001'00000'1010111;
constexpr uint32_t kVmfltVf = 0b011011'0'0000000000'101'00000'1010111;
constexpr uint32_t kVmfneVv = 0b011100'0'0000000000'001'00000'1010111;
constexpr uint32_t kVmfneVf = 0b011100'0'0000000000'101'00000'1010111;
constexpr uint32_t kVmfgtVf = 0b011101'0'0000000000'101'00000'1010111;
constexpr uint32_t kVmfgeVf = 0b011111'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfdivVv = 0b100000'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfdivVf = 0b100000'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfrdivVf = 0b100001'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfmulVv = 0b100100'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfmulVf = 0b100100'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfrsubVf = 0b100111'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfmaddVv = 0b101000'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfmaddVf = 0b101000'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfnmaddVv = 0b101001'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfnmaddVf = 0b101001'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfmsubVv = 0b101010'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfmsubVf = 0b101010'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfnmsubVv = 0b101011'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfnmsubVf = 0b101011'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfmaccVv = 0b101100'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfmaccVf = 0b101100'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfnmaccVv = 0b101101'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfnmaccVf = 0b101101'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfmsacVv = 0b101110'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfmsacVf = 0b101110'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfnmsacVv = 0b101111'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfnmsacVf = 0b101111'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfwaddVv = 0b110000'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfwaddVf = 0b110000'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfwredusumVv = 0b110001'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfwsubVv = 0b110010'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfwsubVf = 0b110010'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfwredosumVv = 0b110011'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfwaddWVv = 0b110100'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfwaddWVf = 0b110100'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfwsubWVv = 0b110110'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfwsubWVf = 0b110110'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfwmulVv = 0b111000'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfwmulVf = 0b111000'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfwmaccVv = 0b111100'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfwmaccVf = 0b111100'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfwnmaccVv = 0b111101'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfwnmaccVf = 0b111101'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfwmsacVv = 0b111110'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfwmsacVf = 0b111110'0'0000000000'101'00000'1010111;
constexpr uint32_t kVfwnmsacVv = 0b111111'0'0000000000'001'00000'1010111;
constexpr uint32_t kVfwnmsacVf = 0b111111'0'0000000000'101'00000'1010111;

// VWXUNARY0
constexpr uint32_t kVmvXS = 0b010000'0'00000'00000'010'00000'1010111;
constexpr uint32_t kVcpop = 0b010000'0'00000'10000'010'00000'1010111;
constexpr uint32_t kVfirst = 0b010000'0'00000'10001'010'00000'1010111;

// VRXUNARY0
constexpr uint32_t kVmvSX = 0b010000'0'00000'00000'110'00000'1010111;

// VXUNARY0
constexpr uint32_t kVzextVf8 = 0b010010'0'00000'00010'010'00000'1010111;
constexpr uint32_t kVsextVf8 = 0b010010'0'00000'00011'010'00000'1010111;
constexpr uint32_t kVzextVf4 = 0b010010'0'00000'00100'010'00000'1010111;
constexpr uint32_t kVsextVf4 = 0b010010'0'00000'00101'010'00000'1010111;
constexpr uint32_t kVzextVf2 = 0b010010'0'00000'00110'010'00000'1010111;
constexpr uint32_t kVsextVf2 = 0b010010'0'00000'00111'010'00000'1010111;

// VMUNARY0
constexpr uint32_t kVmsbf = 0b010100'0'00000'00001'010'00000'1010111;
constexpr uint32_t kVmsof = 0b010100'0'00000'00010'010'00000'1010111;
constexpr uint32_t kVmsif = 0b010100'0'00000'00011'010'00000'1010111;
constexpr uint32_t kViota = 0b010100'0'00000'10000'010'00000'1010111;
constexpr uint32_t kVid = 0b010100'0'00000'10001'010'00000'1010111;

// VWFUNARY0
constexpr uint32_t kVfmvFS = 0b010000'0'00000'00000'001'00000'1010111;

// VRFUNARY0
constexpr uint32_t kVfmvSF = 0b010000'0'00000'00000'101'00000'1010111;

// VFUNARY0
constexpr uint32_t kVfcvtXuFV = 0b010010'0'00000'00000'001'00000'1010111;
constexpr uint32_t kVfcvtXFV = 0b010010'0'00000'00001'001'00000'1010111;
constexpr uint32_t kVfcvtFXuV = 0b010010'0'00000'00010'001'00000'1010111;
constexpr uint32_t kVfcvtFXV = 0b010010'0'00000'00011'001'00000'1010111;
constexpr uint32_t kVfcvtRtzXuFV = 0b010010'0'00000'00110'001'00000'1010111;
constexpr uint32_t kVfcvtRtzXFV = 0b010010'0'00000'00111'001'00000'1010111;

constexpr uint32_t kVfwcvtXuFV = 0b010010'0'00000'01000'001'00000'1010111;
constexpr uint32_t kVfwcvtXFV = 0b010010'0'00000'01001'001'00000'1010111;
constexpr uint32_t kVfwcvtFXuV = 0b010010'0'00000'01010'001'00000'1010111;
constexpr uint32_t kVfwcvtFXV = 0b010010'0'00000'01011'001'00000'1010111;
constexpr uint32_t kVfwcvtFFV = 0b010010'0'00000'01100'001'00000'1010111;
constexpr uint32_t kVfwcvtRtzXuFV = 0b010010'0'00000'01110'001'00000'1010111;
constexpr uint32_t kVfwcvtRtzXFV = 0b010010'0'00000'01111'001'00000'1010111;

constexpr uint32_t kVfncvtXuFW = 0b010010'0'00000'10000'001'00000'1010111;
constexpr uint32_t kVfncvtXFW = 0b010010'0'00000'10001'001'00000'1010111;
constexpr uint32_t kVfncvtFXuW = 0b010010'0'00000'10010'001'00000'1010111;
constexpr uint32_t kVfncvtFXW = 0b010010'0'00000'10011'001'00000'1010111;
constexpr uint32_t kVfncvtFFW = 0b010010'0'00000'10100'001'00000'1010111;
constexpr uint32_t kVfncvtRodFFW = 0b010010'0'00000'10101'001'00000'1010111;
constexpr uint32_t kVfncvtRtzXuFW = 0b010010'0'00000'10110'001'00000'1010111;
constexpr uint32_t kVfncvtRtzXFW = 0b010010'0'00000'10111'001'00000'1010111;

// VFUNARY1
constexpr uint32_t kVfsqrtV = 0b010011'0'00000'00000'001'00000'1010111;
constexpr uint32_t kVfrsqrt7V = 0b010011'0'00000'00100'001'00000'1010111;
constexpr uint32_t kVfrec7V = 0b010011'0'00000'00101'001'00000'1010111;
constexpr uint32_t kVfclassV = 0b010011'0'00000'10000'001'00000'1010111;

constexpr int kRdValue = 1;
constexpr int kSuccValue = 0xf;
constexpr int kPredValue = 0xf;

static uint32_t SetRd(uint32_t iword, uint32_t rdval) {
  return (iword | ((rdval & 0x1f) << 7));
}

static uint32_t SetRs1(uint32_t iword, uint32_t rsval) {
  return (iword | ((rsval & 0x1f) << 15));
}

static uint32_t SetPred(uint32_t iword, uint32_t pred) {
  return (iword | ((pred & 0xf) << 24));
}

static uint32_t SetSucc(uint32_t iword, uint32_t succ) {
  return (iword | ((succ & 0xf) << 20));
}

static uint32_t Set16Rd(uint32_t iword, uint32_t val) {
  return (iword | ((val & 0x1f) << 7));
}

static uint32_t Set16Rs2(uint32_t iword, uint32_t val) {
  return (iword | ((val & 0x1f) << 2));
}

static uint32_t SetNf(uint32_t iword, uint32_t nf) {
  return (iword | ((nf & 0x7) << 29));
}

// This test is only used to make sure all functions and variables are used,
// instead of deleting them and then having to put them back later as more
// encodings are added.
TEST(RiscV32GVBinDecoderTest, None) {
  (void)kRdValue;
  (void)kSuccValue;
  (void)kPredValue;
  (void)SetRd(0, 0);
  (void)SetRs1(0, 0);
  (void)SetPred(0, 0);
  (void)SetSucc(0, 0);
  (void)Set16Rd(0, 0);
  (void)Set16Rs2(0, 0);
}

TEST(RiscV32GVBinDecoderTest, VectorConfig) {
  EXPECT_EQ(DecodeRiscV32GV(kVsetvli_xn), OpcodeEnum::kVsetvliXn);
  EXPECT_EQ(DecodeRiscV32GV(kVsetvli_nz), OpcodeEnum::kVsetvliNz);
  EXPECT_EQ(DecodeRiscV32GV(kVsetvli_zz), OpcodeEnum::kVsetvliZz);
  EXPECT_EQ(DecodeRiscV32GV(kVsetivli), OpcodeEnum::kVsetivli);
  EXPECT_EQ(DecodeRiscV32GV(kVsetvl_xn), OpcodeEnum::kVsetvlXn);
  EXPECT_EQ(DecodeRiscV32GV(kVsetvl_nz), OpcodeEnum::kVsetvlNz);
  EXPECT_EQ(DecodeRiscV32GV(kVsetvl_zz), OpcodeEnum::kVsetvlZz);
}

TEST(RiscV32GVBinDecoderTest, VectorLoads) {
  EXPECT_EQ(DecodeRiscV32GV(kVle8), OpcodeEnum::kVle8);
  EXPECT_EQ(DecodeRiscV32GV(kVle16), OpcodeEnum::kVle16);
  EXPECT_EQ(DecodeRiscV32GV(kVle32), OpcodeEnum::kVle32);
  EXPECT_EQ(DecodeRiscV32GV(kVle64), OpcodeEnum::kVle64);

  EXPECT_EQ(DecodeRiscV32GV(kVlm), OpcodeEnum::kVlm);

  EXPECT_EQ(DecodeRiscV32GV(kVle8ff), OpcodeEnum::kVle8ff);
  EXPECT_EQ(DecodeRiscV32GV(kVle16ff), OpcodeEnum::kVle16ff);
  EXPECT_EQ(DecodeRiscV32GV(kVle32ff), OpcodeEnum::kVle32ff);
  EXPECT_EQ(DecodeRiscV32GV(kVle64ff), OpcodeEnum::kVle64ff);

  EXPECT_EQ(DecodeRiscV32GV(kVl1re8), OpcodeEnum::kVl1re8);
  EXPECT_EQ(DecodeRiscV32GV(kVl1re16), OpcodeEnum::kVl1re16);
  EXPECT_EQ(DecodeRiscV32GV(kVl1re32), OpcodeEnum::kVl1re32);
  EXPECT_EQ(DecodeRiscV32GV(kVl1re64), OpcodeEnum::kVl1re64);
  EXPECT_EQ(DecodeRiscV32GV(kVl2re8), OpcodeEnum::kVl2re8);
  EXPECT_EQ(DecodeRiscV32GV(kVl2re16), OpcodeEnum::kVl2re16);
  EXPECT_EQ(DecodeRiscV32GV(kVl2re32), OpcodeEnum::kVl2re32);
  EXPECT_EQ(DecodeRiscV32GV(kVl2re64), OpcodeEnum::kVl2re64);
  EXPECT_EQ(DecodeRiscV32GV(kVl4re8), OpcodeEnum::kVl4re8);
  EXPECT_EQ(DecodeRiscV32GV(kVl4re16), OpcodeEnum::kVl4re16);
  EXPECT_EQ(DecodeRiscV32GV(kVl4re32), OpcodeEnum::kVl4re32);
  EXPECT_EQ(DecodeRiscV32GV(kVl4re64), OpcodeEnum::kVl4re64);
  EXPECT_EQ(DecodeRiscV32GV(kVl8re8), OpcodeEnum::kVl8re8);
  EXPECT_EQ(DecodeRiscV32GV(kVl8re16), OpcodeEnum::kVl8re16);
  EXPECT_EQ(DecodeRiscV32GV(kVl8re32), OpcodeEnum::kVl8re32);
  EXPECT_EQ(DecodeRiscV32GV(kVl8re64), OpcodeEnum::kVl8re64);

  EXPECT_EQ(DecodeRiscV32GV(kVlse8), OpcodeEnum::kVlse8);
  EXPECT_EQ(DecodeRiscV32GV(kVlse16), OpcodeEnum::kVlse16);
  EXPECT_EQ(DecodeRiscV32GV(kVlse32), OpcodeEnum::kVlse32);
  EXPECT_EQ(DecodeRiscV32GV(kVlse64), OpcodeEnum::kVlse64);

  EXPECT_EQ(DecodeRiscV32GV(kVluxei8), OpcodeEnum::kVluxei8);
  EXPECT_EQ(DecodeRiscV32GV(kVluxei16), OpcodeEnum::kVluxei16);
  EXPECT_EQ(DecodeRiscV32GV(kVluxei32), OpcodeEnum::kVluxei32);
  EXPECT_EQ(DecodeRiscV32GV(kVluxei64), OpcodeEnum::kVluxei64);

  EXPECT_EQ(DecodeRiscV32GV(kVloxei8), OpcodeEnum::kVloxei8);
  EXPECT_EQ(DecodeRiscV32GV(kVloxei16), OpcodeEnum::kVloxei16);
  EXPECT_EQ(DecodeRiscV32GV(kVloxei32), OpcodeEnum::kVloxei32);
  EXPECT_EQ(DecodeRiscV32GV(kVloxei64), OpcodeEnum::kVloxei64);

  for (int nf = 1; nf < 8; nf++) {
    EXPECT_EQ(DecodeRiscV32GV(SetNf(kVlsegNFe8, nf)), OpcodeEnum::kVlsege8);
    EXPECT_EQ(DecodeRiscV32GV(SetNf(kVlsegNFe16, nf)), OpcodeEnum::kVlsege16);
    EXPECT_EQ(DecodeRiscV32GV(SetNf(kVlsegNFe32, nf)), OpcodeEnum::kVlsege32);
    EXPECT_EQ(DecodeRiscV32GV(SetNf(kVlsegNFe64, nf)), OpcodeEnum::kVlsege64);
  }

  for (int nf = 1; nf < 8; nf++) {
    EXPECT_EQ(DecodeRiscV32GV(SetNf(kVlssegNFe8, nf)), OpcodeEnum::kVlssege8);
    EXPECT_EQ(DecodeRiscV32GV(SetNf(kVlssegNFe16, nf)), OpcodeEnum::kVlssege16);
    EXPECT_EQ(DecodeRiscV32GV(SetNf(kVlssegNFe32, nf)), OpcodeEnum::kVlssege32);
    EXPECT_EQ(DecodeRiscV32GV(SetNf(kVlssegNFe64, nf)), OpcodeEnum::kVlssege64);
  }

  for (int nf = 1; nf < 8; nf++) {
    EXPECT_EQ(DecodeRiscV32GV(SetNf(kVluxsegNFei8, nf)),
              OpcodeEnum::kVluxsegei8);
    EXPECT_EQ(DecodeRiscV32GV(SetNf(kVluxsegNFei16, nf)),
              OpcodeEnum::kVluxsegei16);
    EXPECT_EQ(DecodeRiscV32GV(SetNf(kVluxsegNFei32, nf)),
              OpcodeEnum::kVluxsegei32);
    EXPECT_EQ(DecodeRiscV32GV(SetNf(kVluxsegNFei64, nf)),
              OpcodeEnum::kVluxsegei64);
  }

  for (int nf = 1; nf < 8; nf++) {
    EXPECT_EQ(DecodeRiscV32GV(SetNf(kVloxsegNFei8, nf)),
              OpcodeEnum::kVloxsegei8);
    EXPECT_EQ(DecodeRiscV32GV(SetNf(kVloxsegNFei16, nf)),
              OpcodeEnum::kVloxsegei16);
    EXPECT_EQ(DecodeRiscV32GV(SetNf(kVloxsegNFei32, nf)),
              OpcodeEnum::kVloxsegei32);
    EXPECT_EQ(DecodeRiscV32GV(SetNf(kVloxsegNFei64, nf)),
              OpcodeEnum::kVloxsegei64);
  }
}

TEST(RiscV32GVBinDecoderTest, VectorStores) {
  // Unit stride.
  EXPECT_EQ(DecodeRiscV32GV(kVse8), OpcodeEnum::kVse8);
  EXPECT_EQ(DecodeRiscV32GV(kVse16), OpcodeEnum::kVse16);
  EXPECT_EQ(DecodeRiscV32GV(kVse32), OpcodeEnum::kVse32);
  EXPECT_EQ(DecodeRiscV32GV(kVse64), OpcodeEnum::kVse64);
  // Store mask.
  EXPECT_EQ(DecodeRiscV32GV(kVsm), OpcodeEnum::kVsm);
  // Unit stride, fail first.
  EXPECT_EQ(DecodeRiscV32GV(kVse8ff), OpcodeEnum::kVse8ff);
  EXPECT_EQ(DecodeRiscV32GV(kVse16ff), OpcodeEnum::kVse16ff);
  EXPECT_EQ(DecodeRiscV32GV(kVse32ff), OpcodeEnum::kVse32ff);
  EXPECT_EQ(DecodeRiscV32GV(kVse64ff), OpcodeEnum::kVse64ff);
  // Store whole register.
  EXPECT_EQ(DecodeRiscV32GV(kVs1re8), OpcodeEnum::kVs1re8);
  EXPECT_EQ(DecodeRiscV32GV(kVs1re16), OpcodeEnum::kVs1re16);
  EXPECT_EQ(DecodeRiscV32GV(kVs1re32), OpcodeEnum::kVs1re32);
  EXPECT_EQ(DecodeRiscV32GV(kVs1re64), OpcodeEnum::kVs1re64);
  EXPECT_EQ(DecodeRiscV32GV(kVs2re8), OpcodeEnum::kVs2re8);
  EXPECT_EQ(DecodeRiscV32GV(kVs2re16), OpcodeEnum::kVs2re16);
  EXPECT_EQ(DecodeRiscV32GV(kVs2re32), OpcodeEnum::kVs2re32);
  EXPECT_EQ(DecodeRiscV32GV(kVs2re64), OpcodeEnum::kVs2re64);
  EXPECT_EQ(DecodeRiscV32GV(kVs4re8), OpcodeEnum::kVs4re8);
  EXPECT_EQ(DecodeRiscV32GV(kVs4re16), OpcodeEnum::kVs4re16);
  EXPECT_EQ(DecodeRiscV32GV(kVs4re32), OpcodeEnum::kVs4re32);
  EXPECT_EQ(DecodeRiscV32GV(kVs4re64), OpcodeEnum::kVs4re64);
  EXPECT_EQ(DecodeRiscV32GV(kVs8re8), OpcodeEnum::kVs8re8);
  EXPECT_EQ(DecodeRiscV32GV(kVs8re16), OpcodeEnum::kVs8re16);
  EXPECT_EQ(DecodeRiscV32GV(kVs8re32), OpcodeEnum::kVs8re32);
  EXPECT_EQ(DecodeRiscV32GV(kVs8re64), OpcodeEnum::kVs8re64);
  // Store strided.
  EXPECT_EQ(DecodeRiscV32GV(kVsse8), OpcodeEnum::kVsse8);
  EXPECT_EQ(DecodeRiscV32GV(kVsse16), OpcodeEnum::kVsse16);
  EXPECT_EQ(DecodeRiscV32GV(kVsse32), OpcodeEnum::kVsse32);
  EXPECT_EQ(DecodeRiscV32GV(kVsse64), OpcodeEnum::kVsse64);
  // Store indexed, unordered.
  EXPECT_EQ(DecodeRiscV32GV(kVsuxei8), OpcodeEnum::kVsuxei8);
  EXPECT_EQ(DecodeRiscV32GV(kVsuxei16), OpcodeEnum::kVsuxei16);
  EXPECT_EQ(DecodeRiscV32GV(kVsuxei32), OpcodeEnum::kVsuxei32);
  EXPECT_EQ(DecodeRiscV32GV(kVsuxei64), OpcodeEnum::kVsuxei64);
  // Store indexed, ordered.
  EXPECT_EQ(DecodeRiscV32GV(kVsoxei8), OpcodeEnum::kVsoxei8);
  EXPECT_EQ(DecodeRiscV32GV(kVsoxei16), OpcodeEnum::kVsoxei16);
  EXPECT_EQ(DecodeRiscV32GV(kVsoxei32), OpcodeEnum::kVsoxei32);
  EXPECT_EQ(DecodeRiscV32GV(kVsoxei64), OpcodeEnum::kVsoxei64);
}

TEST(RiscV32GVBinDecoderTest, OPIVV_OPIVX_OPIVI) {
  EXPECT_EQ(DecodeRiscV32GV(kVaddVv), OpcodeEnum::kVaddVv);
  EXPECT_EQ(DecodeRiscV32GV(kVaddVx), OpcodeEnum::kVaddVx);
  EXPECT_EQ(DecodeRiscV32GV(kVaddVi), OpcodeEnum::kVaddVi);
  EXPECT_EQ(DecodeRiscV32GV(kVsubVv), OpcodeEnum::kVsubVv);
  EXPECT_EQ(DecodeRiscV32GV(kVsubVx), OpcodeEnum::kVsubVx);
  EXPECT_EQ(DecodeRiscV32GV(kVrsubVx), OpcodeEnum::kVrsubVx);
  EXPECT_EQ(DecodeRiscV32GV(kVrsubVi), OpcodeEnum::kVrsubVi);
  EXPECT_EQ(DecodeRiscV32GV(kVminuVv), OpcodeEnum::kVminuVv);
  EXPECT_EQ(DecodeRiscV32GV(kVminuVx), OpcodeEnum::kVminuVx);
  EXPECT_EQ(DecodeRiscV32GV(kVminVv), OpcodeEnum::kVminVv);
  EXPECT_EQ(DecodeRiscV32GV(kVminVx), OpcodeEnum::kVminVx);
  EXPECT_EQ(DecodeRiscV32GV(kVmaxuVv), OpcodeEnum::kVmaxuVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmaxuVx), OpcodeEnum::kVmaxuVx);
  EXPECT_EQ(DecodeRiscV32GV(kVmaxVv), OpcodeEnum::kVmaxVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmaxVx), OpcodeEnum::kVmaxVx);
  EXPECT_EQ(DecodeRiscV32GV(kVandVv), OpcodeEnum::kVandVv);
  EXPECT_EQ(DecodeRiscV32GV(kVandVx), OpcodeEnum::kVandVx);
  EXPECT_EQ(DecodeRiscV32GV(kVandVi), OpcodeEnum::kVandVi);
  EXPECT_EQ(DecodeRiscV32GV(kVorVv), OpcodeEnum::kVorVv);
  EXPECT_EQ(DecodeRiscV32GV(kVorVx), OpcodeEnum::kVorVx);
  EXPECT_EQ(DecodeRiscV32GV(kVorVi), OpcodeEnum::kVorVi);
  EXPECT_EQ(DecodeRiscV32GV(kVxorVv), OpcodeEnum::kVxorVv);
  EXPECT_EQ(DecodeRiscV32GV(kVxorVx), OpcodeEnum::kVxorVx);
  EXPECT_EQ(DecodeRiscV32GV(kVxorVi), OpcodeEnum::kVxorVi);
  EXPECT_EQ(DecodeRiscV32GV(kVrgatherVv), OpcodeEnum::kVrgatherVv);
  EXPECT_EQ(DecodeRiscV32GV(kVrgatherVx), OpcodeEnum::kVrgatherVx);
  EXPECT_EQ(DecodeRiscV32GV(kVrgatherVi), OpcodeEnum::kVrgatherVi);
  EXPECT_EQ(DecodeRiscV32GV(kVrgatherei16Vv), OpcodeEnum::kVrgatherei16Vv);
  EXPECT_EQ(DecodeRiscV32GV(kVslideupVx), OpcodeEnum::kVslideupVx);
  EXPECT_EQ(DecodeRiscV32GV(kVslideupVi), OpcodeEnum::kVslideupVi);
  EXPECT_EQ(DecodeRiscV32GV(kVslidedownVx), OpcodeEnum::kVslidedownVx);
  EXPECT_EQ(DecodeRiscV32GV(kVslidedownVi), OpcodeEnum::kVslidedownVi);
  EXPECT_EQ(DecodeRiscV32GV(kVadcVv), OpcodeEnum::kVadcVv);
  EXPECT_EQ(DecodeRiscV32GV(kVadcVx), OpcodeEnum::kVadcVx);
  EXPECT_EQ(DecodeRiscV32GV(kVadcVi), OpcodeEnum::kVadcVi);
  EXPECT_EQ(DecodeRiscV32GV(kVmadcVv), OpcodeEnum::kVmadcVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmadcVx), OpcodeEnum::kVmadcVx);
  EXPECT_EQ(DecodeRiscV32GV(kVmadcVi), OpcodeEnum::kVmadcVi);
  EXPECT_EQ(DecodeRiscV32GV(kVsbcVv), OpcodeEnum::kVsbcVv);
  EXPECT_EQ(DecodeRiscV32GV(kVsbcVx), OpcodeEnum::kVsbcVx);
  EXPECT_EQ(DecodeRiscV32GV(kVmsbcVv), OpcodeEnum::kVmsbcVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmsbcVx), OpcodeEnum::kVmsbcVx);
  EXPECT_EQ(DecodeRiscV32GV(kVmergeVv), OpcodeEnum::kVmergeVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmergeVx), OpcodeEnum::kVmergeVx);
  EXPECT_EQ(DecodeRiscV32GV(kVmergeVi), OpcodeEnum::kVmergeVi);
  EXPECT_EQ(DecodeRiscV32GV(kVmvVv), OpcodeEnum::kVmvVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmvVx), OpcodeEnum::kVmvVx);
  EXPECT_EQ(DecodeRiscV32GV(kVmvVi), OpcodeEnum::kVmvVi);
  EXPECT_EQ(DecodeRiscV32GV(kVmseqVv), OpcodeEnum::kVmseqVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmseqVx), OpcodeEnum::kVmseqVx);
  EXPECT_EQ(DecodeRiscV32GV(kVmseqVi), OpcodeEnum::kVmseqVi);
  EXPECT_EQ(DecodeRiscV32GV(kVmsneVv), OpcodeEnum::kVmsneVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmsneVx), OpcodeEnum::kVmsneVx);
  EXPECT_EQ(DecodeRiscV32GV(kVmsneVi), OpcodeEnum::kVmsneVi);
  EXPECT_EQ(DecodeRiscV32GV(kVmsltuVv), OpcodeEnum::kVmsltuVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmsltuVx), OpcodeEnum::kVmsltuVx);
  EXPECT_EQ(DecodeRiscV32GV(kVmsltVv), OpcodeEnum::kVmsltVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmsltVx), OpcodeEnum::kVmsltVx);
  EXPECT_EQ(DecodeRiscV32GV(kVmsleuVv), OpcodeEnum::kVmsleuVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmsleuVx), OpcodeEnum::kVmsleuVx);
  EXPECT_EQ(DecodeRiscV32GV(kVmsleuVi), OpcodeEnum::kVmsleuVi);
  EXPECT_EQ(DecodeRiscV32GV(kVmsleVv), OpcodeEnum::kVmsleVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmsleVx), OpcodeEnum::kVmsleVx);
  EXPECT_EQ(DecodeRiscV32GV(kVmsleVi), OpcodeEnum::kVmsleVi);
  EXPECT_EQ(DecodeRiscV32GV(kVmsgtuVx), OpcodeEnum::kVmsgtuVx);
  EXPECT_EQ(DecodeRiscV32GV(kVmsgtuVi), OpcodeEnum::kVmsgtuVi);
  EXPECT_EQ(DecodeRiscV32GV(kVmsgtVx), OpcodeEnum::kVmsgtVx);
  EXPECT_EQ(DecodeRiscV32GV(kVmsgtVi), OpcodeEnum::kVmsgtVi);
  EXPECT_EQ(DecodeRiscV32GV(kVsadduVv), OpcodeEnum::kVsadduVv);
  EXPECT_EQ(DecodeRiscV32GV(kVsadduVx), OpcodeEnum::kVsadduVx);
  EXPECT_EQ(DecodeRiscV32GV(kVsadduVi), OpcodeEnum::kVsadduVi);
  EXPECT_EQ(DecodeRiscV32GV(kVsaddVv), OpcodeEnum::kVsaddVv);
  EXPECT_EQ(DecodeRiscV32GV(kVsaddVx), OpcodeEnum::kVsaddVx);
  EXPECT_EQ(DecodeRiscV32GV(kVsaddVi), OpcodeEnum::kVsaddVi);
  EXPECT_EQ(DecodeRiscV32GV(kVssubuVv), OpcodeEnum::kVssubuVv);
  EXPECT_EQ(DecodeRiscV32GV(kVssubuVx), OpcodeEnum::kVssubuVx);
  EXPECT_EQ(DecodeRiscV32GV(kVssubVv), OpcodeEnum::kVssubVv);
  EXPECT_EQ(DecodeRiscV32GV(kVssubVx), OpcodeEnum::kVssubVx);
  EXPECT_EQ(DecodeRiscV32GV(kVsllVv), OpcodeEnum::kVsllVv);
  EXPECT_EQ(DecodeRiscV32GV(kVsllVx), OpcodeEnum::kVsllVx);
  EXPECT_EQ(DecodeRiscV32GV(kVsllVi), OpcodeEnum::kVsllVi);
  EXPECT_EQ(DecodeRiscV32GV(kVsmulVv), OpcodeEnum::kVsmulVv);
  EXPECT_EQ(DecodeRiscV32GV(kVsmulVx), OpcodeEnum::kVsmulVx);
  EXPECT_EQ(DecodeRiscV32GV(kVmv1rVi), OpcodeEnum::kVmv1rVi);
  EXPECT_EQ(DecodeRiscV32GV(kVmv2rVi), OpcodeEnum::kVmv2rVi);
  EXPECT_EQ(DecodeRiscV32GV(kVmv4rVi), OpcodeEnum::kVmv4rVi);
  EXPECT_EQ(DecodeRiscV32GV(kVmv8rVi), OpcodeEnum::kVmv8rVi);
  EXPECT_EQ(DecodeRiscV32GV(kVsrlVv), OpcodeEnum::kVsrlVv);
  EXPECT_EQ(DecodeRiscV32GV(kVsrlVx), OpcodeEnum::kVsrlVx);
  EXPECT_EQ(DecodeRiscV32GV(kVsrlVi), OpcodeEnum::kVsrlVi);
  EXPECT_EQ(DecodeRiscV32GV(kVsraVv), OpcodeEnum::kVsraVv);
  EXPECT_EQ(DecodeRiscV32GV(kVsraVx), OpcodeEnum::kVsraVx);
  EXPECT_EQ(DecodeRiscV32GV(kVsraVi), OpcodeEnum::kVsraVi);
  EXPECT_EQ(DecodeRiscV32GV(kVssrlVv), OpcodeEnum::kVssrlVv);
  EXPECT_EQ(DecodeRiscV32GV(kVssrlVx), OpcodeEnum::kVssrlVx);
  EXPECT_EQ(DecodeRiscV32GV(kVssrlVi), OpcodeEnum::kVssrlVi);
  EXPECT_EQ(DecodeRiscV32GV(kVssraVv), OpcodeEnum::kVssraVv);
  EXPECT_EQ(DecodeRiscV32GV(kVssraVx), OpcodeEnum::kVssraVx);
  EXPECT_EQ(DecodeRiscV32GV(kVssraVi), OpcodeEnum::kVssraVi);
  EXPECT_EQ(DecodeRiscV32GV(kVnsrlVv), OpcodeEnum::kVnsrlVv);
  EXPECT_EQ(DecodeRiscV32GV(kVnsrlVx), OpcodeEnum::kVnsrlVx);
  EXPECT_EQ(DecodeRiscV32GV(kVnsrlVi), OpcodeEnum::kVnsrlVi);
  EXPECT_EQ(DecodeRiscV32GV(kVnsraVv), OpcodeEnum::kVnsraVv);
  EXPECT_EQ(DecodeRiscV32GV(kVnsraVx), OpcodeEnum::kVnsraVx);
  EXPECT_EQ(DecodeRiscV32GV(kVnsraVi), OpcodeEnum::kVnsraVi);
  EXPECT_EQ(DecodeRiscV32GV(kVnclipuVv), OpcodeEnum::kVnclipuVv);
  EXPECT_EQ(DecodeRiscV32GV(kVnclipuVx), OpcodeEnum::kVnclipuVx);
  EXPECT_EQ(DecodeRiscV32GV(kVnclipuVi), OpcodeEnum::kVnclipuVi);
  EXPECT_EQ(DecodeRiscV32GV(kVnclipVv), OpcodeEnum::kVnclipVv);
  EXPECT_EQ(DecodeRiscV32GV(kVnclipVx), OpcodeEnum::kVnclipVx);
  EXPECT_EQ(DecodeRiscV32GV(kVnclipVi), OpcodeEnum::kVnclipVi);
  EXPECT_EQ(DecodeRiscV32GV(kVwredsumuVv), OpcodeEnum::kVwredsumuVv);
  EXPECT_EQ(DecodeRiscV32GV(kVwredsumVv), OpcodeEnum::kVwredsumVv);
}

TEST(RiscV32GVBinDecoderTest, OPPMVV_OPMVX) {
  EXPECT_EQ(DecodeRiscV32GV(kVredsumVv), OpcodeEnum::kVredsumVv);
  EXPECT_EQ(DecodeRiscV32GV(kVredandVv), OpcodeEnum::kVredandVv);
  EXPECT_EQ(DecodeRiscV32GV(kVredorVv), OpcodeEnum::kVredorVv);
  EXPECT_EQ(DecodeRiscV32GV(kVredxorVv), OpcodeEnum::kVredxorVv);
  EXPECT_EQ(DecodeRiscV32GV(kVredminuVv), OpcodeEnum::kVredminuVv);
  EXPECT_EQ(DecodeRiscV32GV(kVredminVv), OpcodeEnum::kVredminVv);
  EXPECT_EQ(DecodeRiscV32GV(kVredmaxuVv), OpcodeEnum::kVredmaxuVv);
  EXPECT_EQ(DecodeRiscV32GV(kVredmaxVv), OpcodeEnum::kVredmaxVv);
  EXPECT_EQ(DecodeRiscV32GV(kVaadduVv), OpcodeEnum::kVaadduVv);
  EXPECT_EQ(DecodeRiscV32GV(kVaadduVx), OpcodeEnum::kVaadduVx);
  EXPECT_EQ(DecodeRiscV32GV(kVaaddVv), OpcodeEnum::kVaaddVv);
  EXPECT_EQ(DecodeRiscV32GV(kVaaddVx), OpcodeEnum::kVaaddVx);
  EXPECT_EQ(DecodeRiscV32GV(kVasubuVv), OpcodeEnum::kVasubuVv);
  EXPECT_EQ(DecodeRiscV32GV(kVasubuVx), OpcodeEnum::kVasubuVx);
  EXPECT_EQ(DecodeRiscV32GV(kVasubVv), OpcodeEnum::kVasubVv);
  EXPECT_EQ(DecodeRiscV32GV(kVasubVx), OpcodeEnum::kVasubVx);
  EXPECT_EQ(DecodeRiscV32GV(kVslide1upVx), OpcodeEnum::kVslide1upVx);
  EXPECT_EQ(DecodeRiscV32GV(kVslide1downVx), OpcodeEnum::kVslide1downVx);
  EXPECT_EQ(DecodeRiscV32GV(kVcompressVv), OpcodeEnum::kVcompressVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmandnotVv), OpcodeEnum::kVmandnotVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmandVv), OpcodeEnum::kVmandVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmorVv), OpcodeEnum::kVmorVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmxorVv), OpcodeEnum::kVmxorVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmornotVv), OpcodeEnum::kVmornotVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmnandVv), OpcodeEnum::kVmnandVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmnorVv), OpcodeEnum::kVmnorVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmxnorVv), OpcodeEnum::kVmxnorVv);
  EXPECT_EQ(DecodeRiscV32GV(kVdivuVv), OpcodeEnum::kVdivuVv);
  EXPECT_EQ(DecodeRiscV32GV(kVdivuVx), OpcodeEnum::kVdivuVx);
  EXPECT_EQ(DecodeRiscV32GV(kVdivVv), OpcodeEnum::kVdivVv);
  EXPECT_EQ(DecodeRiscV32GV(kVdivVx), OpcodeEnum::kVdivVx);
  EXPECT_EQ(DecodeRiscV32GV(kVremuVv), OpcodeEnum::kVremuVv);
  EXPECT_EQ(DecodeRiscV32GV(kVremuVx), OpcodeEnum::kVremuVx);
  EXPECT_EQ(DecodeRiscV32GV(kVremVv), OpcodeEnum::kVremVv);
  EXPECT_EQ(DecodeRiscV32GV(kVremVx), OpcodeEnum::kVremVx);
  EXPECT_EQ(DecodeRiscV32GV(kVmulhuVv), OpcodeEnum::kVmulhuVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmulhuVx), OpcodeEnum::kVmulhuVx);
  EXPECT_EQ(DecodeRiscV32GV(kVmulVv), OpcodeEnum::kVmulVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmulVx), OpcodeEnum::kVmulVx);
  EXPECT_EQ(DecodeRiscV32GV(kVmulhsuVv), OpcodeEnum::kVmulhsuVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmulhsuVx), OpcodeEnum::kVmulhsuVx);
  EXPECT_EQ(DecodeRiscV32GV(kVmulhVv), OpcodeEnum::kVmulhVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmulhVx), OpcodeEnum::kVmulhVx);
  EXPECT_EQ(DecodeRiscV32GV(kVmaddVv), OpcodeEnum::kVmaddVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmaddVx), OpcodeEnum::kVmaddVx);
  EXPECT_EQ(DecodeRiscV32GV(kVnmsubVv), OpcodeEnum::kVnmsubVv);
  EXPECT_EQ(DecodeRiscV32GV(kVnmsubVx), OpcodeEnum::kVnmsubVx);
  EXPECT_EQ(DecodeRiscV32GV(kVmaccVv), OpcodeEnum::kVmaccVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmaccVx), OpcodeEnum::kVmaccVx);
  EXPECT_EQ(DecodeRiscV32GV(kVnmsacVv), OpcodeEnum::kVnmsacVv);
  EXPECT_EQ(DecodeRiscV32GV(kVnmsacVx), OpcodeEnum::kVnmsacVx);
  EXPECT_EQ(DecodeRiscV32GV(kVwadduVv), OpcodeEnum::kVwadduVv);
  EXPECT_EQ(DecodeRiscV32GV(kVwadduVx), OpcodeEnum::kVwadduVx);
  EXPECT_EQ(DecodeRiscV32GV(kVwaddVv), OpcodeEnum::kVwaddVv);
  EXPECT_EQ(DecodeRiscV32GV(kVwaddVx), OpcodeEnum::kVwaddVx);
  EXPECT_EQ(DecodeRiscV32GV(kVwsubuVv), OpcodeEnum::kVwsubuVv);
  EXPECT_EQ(DecodeRiscV32GV(kVwsubuVx), OpcodeEnum::kVwsubuVx);
  EXPECT_EQ(DecodeRiscV32GV(kVwsubVv), OpcodeEnum::kVwsubVv);
  EXPECT_EQ(DecodeRiscV32GV(kVwsubVx), OpcodeEnum::kVwsubVx);
  EXPECT_EQ(DecodeRiscV32GV(kVwadduWVv), OpcodeEnum::kVwadduWVv);
  EXPECT_EQ(DecodeRiscV32GV(kVwadduWVx), OpcodeEnum::kVwadduWVx);
  EXPECT_EQ(DecodeRiscV32GV(kVwaddWVv), OpcodeEnum::kVwaddWVv);
  EXPECT_EQ(DecodeRiscV32GV(kVwaddWVx), OpcodeEnum::kVwaddWVx);
  EXPECT_EQ(DecodeRiscV32GV(kVwsubuWVv), OpcodeEnum::kVwsubuWVv);
  EXPECT_EQ(DecodeRiscV32GV(kVwsubuWVx), OpcodeEnum::kVwsubuWVx);
  EXPECT_EQ(DecodeRiscV32GV(kVwsubWVv), OpcodeEnum::kVwsubWVv);
  EXPECT_EQ(DecodeRiscV32GV(kVwsubWVx), OpcodeEnum::kVwsubWVx);
  EXPECT_EQ(DecodeRiscV32GV(kVwmuluVv), OpcodeEnum::kVwmuluVv);
  EXPECT_EQ(DecodeRiscV32GV(kVwmuluVx), OpcodeEnum::kVwmuluVx);
  EXPECT_EQ(DecodeRiscV32GV(kVwmulsuVv), OpcodeEnum::kVwmulsuVv);
  EXPECT_EQ(DecodeRiscV32GV(kVwmulsuVx), OpcodeEnum::kVwmulsuVx);
  EXPECT_EQ(DecodeRiscV32GV(kVwmulVv), OpcodeEnum::kVwmulVv);
  EXPECT_EQ(DecodeRiscV32GV(kVwmulVx), OpcodeEnum::kVwmulVx);
  EXPECT_EQ(DecodeRiscV32GV(kVwmaccuVv), OpcodeEnum::kVwmaccuVv);
  EXPECT_EQ(DecodeRiscV32GV(kVwmaccuVx), OpcodeEnum::kVwmaccuVx);
  EXPECT_EQ(DecodeRiscV32GV(kVwmaccVv), OpcodeEnum::kVwmaccVv);
  EXPECT_EQ(DecodeRiscV32GV(kVwmaccVx), OpcodeEnum::kVwmaccVx);
  EXPECT_EQ(DecodeRiscV32GV(kVwmaccusVv), OpcodeEnum::kVwmaccusVv);
  EXPECT_EQ(DecodeRiscV32GV(kVwmaccusVx), OpcodeEnum::kVwmaccusVx);
  EXPECT_EQ(DecodeRiscV32GV(kVwmaccsuVv), OpcodeEnum::kVwmaccsuVv);
  EXPECT_EQ(DecodeRiscV32GV(kVwmaccsuVx), OpcodeEnum::kVwmaccsuVx);
}

TEST(RiscV32GVBinDecoderTest, OPPFVV_OPFVF) {
  EXPECT_EQ(DecodeRiscV32GV(kVfaddVv), OpcodeEnum::kVfaddVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfaddVf), OpcodeEnum::kVfaddVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfredusumVv), OpcodeEnum::kVfredusumVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfsubVv), OpcodeEnum::kVfsubVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfsubVf), OpcodeEnum::kVfsubVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfredosumVv), OpcodeEnum::kVfredosumVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfminVv), OpcodeEnum::kVfminVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfminVf), OpcodeEnum::kVfminVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfredminVv), OpcodeEnum::kVfredminVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfmaxVv), OpcodeEnum::kVfmaxVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfmaxVf), OpcodeEnum::kVfmaxVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfredmaxVv), OpcodeEnum::kVfredmaxVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfsgnjVv), OpcodeEnum::kVfsgnjVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfsgnjVf), OpcodeEnum::kVfsgnjVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfsgnjnVv), OpcodeEnum::kVfsgnjnVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfsgnjnVf), OpcodeEnum::kVfsgnjnVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfsgnjxVv), OpcodeEnum::kVfsgnjxVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfsgnjxVf), OpcodeEnum::kVfsgnjxVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfslide1upVf), OpcodeEnum::kVfslide1upVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfslide1downVf), OpcodeEnum::kVfslide1downVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfmergeVf), OpcodeEnum::kVfmergeVf);
  EXPECT_EQ(DecodeRiscV32GV(kVmfeqVv), OpcodeEnum::kVmfeqVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmfeqVf), OpcodeEnum::kVmfeqVf);
  EXPECT_EQ(DecodeRiscV32GV(kVmfleVv), OpcodeEnum::kVmfleVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmfleVf), OpcodeEnum::kVmfleVf);
  EXPECT_EQ(DecodeRiscV32GV(kVmfltVv), OpcodeEnum::kVmfltVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmfltVf), OpcodeEnum::kVmfltVf);
  EXPECT_EQ(DecodeRiscV32GV(kVmfneVv), OpcodeEnum::kVmfneVv);
  EXPECT_EQ(DecodeRiscV32GV(kVmfneVf), OpcodeEnum::kVmfneVf);
  EXPECT_EQ(DecodeRiscV32GV(kVmfgtVf), OpcodeEnum::kVmfgtVf);
  EXPECT_EQ(DecodeRiscV32GV(kVmfgeVf), OpcodeEnum::kVmfgeVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfdivVv), OpcodeEnum::kVfdivVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfdivVf), OpcodeEnum::kVfdivVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfrdivVf), OpcodeEnum::kVfrdivVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfmulVv), OpcodeEnum::kVfmulVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfmulVf), OpcodeEnum::kVfmulVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfrsubVf), OpcodeEnum::kVfrsubVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfmaddVv), OpcodeEnum::kVfmaddVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfmaddVf), OpcodeEnum::kVfmaddVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfnmaddVv), OpcodeEnum::kVfnmaddVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfnmaddVf), OpcodeEnum::kVfnmaddVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfmsubVv), OpcodeEnum::kVfmsubVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfmsubVf), OpcodeEnum::kVfmsubVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfnmsubVv), OpcodeEnum::kVfnmsubVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfnmsubVf), OpcodeEnum::kVfnmsubVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfmaccVv), OpcodeEnum::kVfmaccVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfmaccVf), OpcodeEnum::kVfmaccVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfnmaccVv), OpcodeEnum::kVfnmaccVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfnmaccVf), OpcodeEnum::kVfnmaccVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfmsacVv), OpcodeEnum::kVfmsacVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfmsacVf), OpcodeEnum::kVfmsacVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfnmsacVv), OpcodeEnum::kVfnmsacVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfnmsacVf), OpcodeEnum::kVfnmsacVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfwaddVv), OpcodeEnum::kVfwaddVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfwaddVf), OpcodeEnum::kVfwaddVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfwredusumVv), OpcodeEnum::kVfwredusumVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfwsubVv), OpcodeEnum::kVfwsubVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfwsubVf), OpcodeEnum::kVfwsubVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfwredosumVv), OpcodeEnum::kVfwredosumVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfwaddWVv), OpcodeEnum::kVfwaddWVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfwaddWVf), OpcodeEnum::kVfwaddWVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfwsubWVv), OpcodeEnum::kVfwsubWVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfwsubWVf), OpcodeEnum::kVfwsubWVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfwmulVv), OpcodeEnum::kVfwmulVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfwmulVf), OpcodeEnum::kVfwmulVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfwmaccVv), OpcodeEnum::kVfwmaccVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfwmaccVf), OpcodeEnum::kVfwmaccVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfwnmaccVv), OpcodeEnum::kVfwnmaccVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfwnmaccVf), OpcodeEnum::kVfwnmaccVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfwmsacVv), OpcodeEnum::kVfwmsacVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfwmsacVf), OpcodeEnum::kVfwmsacVf);
  EXPECT_EQ(DecodeRiscV32GV(kVfwnmsacVv), OpcodeEnum::kVfwnmsacVv);
  EXPECT_EQ(DecodeRiscV32GV(kVfwnmsacVf), OpcodeEnum::kVfwnmsacVf);
}

TEST(RiscV32GVBinDecoderTest, VWXUNARY0) {
  EXPECT_EQ(DecodeRiscV32GV(kVmvXS), OpcodeEnum::kVmvXS);
  EXPECT_EQ(DecodeRiscV32GV(kVcpop), OpcodeEnum::kVcpop);
  EXPECT_EQ(DecodeRiscV32GV(kVfirst), OpcodeEnum::kVfirst);
}

TEST(RiscV32GVBinDecoderTest, VRXUNARY0) {
  EXPECT_EQ(DecodeRiscV32GV(kVmvSX), OpcodeEnum::kVmvSX);
}

TEST(RiscV32GVBinDecoderTest, VXUNARY0) {
  EXPECT_EQ(DecodeRiscV32GV(kVzextVf8), OpcodeEnum::kVzextVf8);
  EXPECT_EQ(DecodeRiscV32GV(kVsextVf8), OpcodeEnum::kVsextVf8);
  EXPECT_EQ(DecodeRiscV32GV(kVzextVf4), OpcodeEnum::kVzextVf4);
  EXPECT_EQ(DecodeRiscV32GV(kVsextVf4), OpcodeEnum::kVsextVf4);
  EXPECT_EQ(DecodeRiscV32GV(kVzextVf2), OpcodeEnum::kVzextVf2);
  EXPECT_EQ(DecodeRiscV32GV(kVsextVf2), OpcodeEnum::kVsextVf2);
}

TEST(RiscV32GVBinDecoderTest, VMUNARY) {
  EXPECT_EQ(DecodeRiscV32GV(kVmsbf), OpcodeEnum::kVmsbf);
  EXPECT_EQ(DecodeRiscV32GV(kVmsof), OpcodeEnum::kVmsof);
  EXPECT_EQ(DecodeRiscV32GV(kVmsif), OpcodeEnum::kVmsif);
  EXPECT_EQ(DecodeRiscV32GV(kViota), OpcodeEnum::kViota);
  EXPECT_EQ(DecodeRiscV32GV(kVid), OpcodeEnum::kVid);
}

TEST(RiscV32GVBinDecoderTest, VWFUNARY0) {
  EXPECT_EQ(DecodeRiscV32GV(kVfmvFS), OpcodeEnum::kVfmvFS);
}

TEST(RiscV32GVBinDecoderTest, VRFUNARY0) {
  EXPECT_EQ(DecodeRiscV32GV(kVfmvSF), OpcodeEnum::kVfmvSF);
}

TEST(RiscV32GVBinDecoderTest, VFUNARY0) {
  EXPECT_EQ(DecodeRiscV32GV(kVfcvtXuFV), OpcodeEnum::kVfcvtXuFV);
  EXPECT_EQ(DecodeRiscV32GV(kVfcvtXFV), OpcodeEnum::kVfcvtXFV);
  EXPECT_EQ(DecodeRiscV32GV(kVfcvtFXuV), OpcodeEnum::kVfcvtFXuV);
  EXPECT_EQ(DecodeRiscV32GV(kVfcvtFXV), OpcodeEnum::kVfcvtFXV);
  EXPECT_EQ(DecodeRiscV32GV(kVfcvtRtzXuFV), OpcodeEnum::kVfcvtRtzXuFV);
  EXPECT_EQ(DecodeRiscV32GV(kVfcvtRtzXFV), OpcodeEnum::kVfcvtRtzXFV);
  EXPECT_EQ(DecodeRiscV32GV(kVfwcvtXuFV), OpcodeEnum::kVfwcvtXuFV);
  EXPECT_EQ(DecodeRiscV32GV(kVfwcvtXFV), OpcodeEnum::kVfwcvtXFV);
  EXPECT_EQ(DecodeRiscV32GV(kVfwcvtFXuV), OpcodeEnum::kVfwcvtFXuV);
  EXPECT_EQ(DecodeRiscV32GV(kVfwcvtFXV), OpcodeEnum::kVfwcvtFXV);
  EXPECT_EQ(DecodeRiscV32GV(kVfwcvtFFV), OpcodeEnum::kVfwcvtFFV);
  EXPECT_EQ(DecodeRiscV32GV(kVfwcvtRtzXuFV), OpcodeEnum::kVfwcvtRtzXuFV);
  EXPECT_EQ(DecodeRiscV32GV(kVfwcvtRtzXFV), OpcodeEnum::kVfwcvtRtzXFV);
  EXPECT_EQ(DecodeRiscV32GV(kVfncvtXuFW), OpcodeEnum::kVfncvtXuFW);
  EXPECT_EQ(DecodeRiscV32GV(kVfncvtXFW), OpcodeEnum::kVfncvtXFW);
  EXPECT_EQ(DecodeRiscV32GV(kVfncvtFXuW), OpcodeEnum::kVfncvtFXuW);
  EXPECT_EQ(DecodeRiscV32GV(kVfncvtFXW), OpcodeEnum::kVfncvtFXW);
  EXPECT_EQ(DecodeRiscV32GV(kVfncvtFFW), OpcodeEnum::kVfncvtFFW);
  EXPECT_EQ(DecodeRiscV32GV(kVfncvtRodFFW), OpcodeEnum::kVfncvtRodFFW);
  EXPECT_EQ(DecodeRiscV32GV(kVfncvtRtzXuFW), OpcodeEnum::kVfncvtRtzXuFW);
  EXPECT_EQ(DecodeRiscV32GV(kVfncvtRtzXFW), OpcodeEnum::kVfncvtRtzXFW);
}

TEST(RiscV32GVBinDecoderTest, VFUNARY1) {
  EXPECT_EQ(DecodeRiscV32GV(kVfsqrtV), OpcodeEnum::kVfsqrtV);
  EXPECT_EQ(DecodeRiscV32GV(kVfrsqrt7V), OpcodeEnum::kVfrsqrt7V);
  EXPECT_EQ(DecodeRiscV32GV(kVfrec7V), OpcodeEnum::kVfrec7V);
  EXPECT_EQ(DecodeRiscV32GV(kVfclassV), OpcodeEnum::kVfclassV);
}

}  // namespace
