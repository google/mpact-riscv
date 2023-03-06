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

#include "riscv/riscv_debug_info.h"

#include "mpact/sim/generic/type_helpers.h"

namespace mpact {
namespace sim {
namespace riscv {

constexpr char kPcName[] = "pc";

constexpr char kX0Name[] = "x0";
constexpr char kX1Name[] = "x1";
constexpr char kX2Name[] = "x2";
constexpr char kX3Name[] = "x3";
constexpr char kX4Name[] = "x4";
constexpr char kX5Name[] = "x5";
constexpr char kX6Name[] = "x6";
constexpr char kX7Name[] = "x7";
constexpr char kX8Name[] = "x8";
constexpr char kX9Name[] = "x9";
constexpr char kX10Name[] = "x10";
constexpr char kX11Name[] = "x11";
constexpr char kX12Name[] = "x12";
constexpr char kX13Name[] = "x13";
constexpr char kX14Name[] = "x14";
constexpr char kX15Name[] = "x15";
constexpr char kX16Name[] = "x16";
constexpr char kX17Name[] = "x17";
constexpr char kX18Name[] = "x18";
constexpr char kX19Name[] = "x19";
constexpr char kX20Name[] = "x20";
constexpr char kX21Name[] = "x21";
constexpr char kX22Name[] = "x22";
constexpr char kX23Name[] = "x23";
constexpr char kX24Name[] = "x24";
constexpr char kX25Name[] = "x25";
constexpr char kX26Name[] = "x26";
constexpr char kX27Name[] = "x27";
constexpr char kX28Name[] = "x28";
constexpr char kX29Name[] = "x29";
constexpr char kX30Name[] = "x30";
constexpr char kX31Name[] = "x31";

constexpr char kF0Name[] = "f0";
constexpr char kF1Name[] = "f1";
constexpr char kF2Name[] = "f2";
constexpr char kF3Name[] = "f3";
constexpr char kF4Name[] = "f4";
constexpr char kF5Name[] = "f5";
constexpr char kF6Name[] = "f6";
constexpr char kF7Name[] = "f7";
constexpr char kF8Name[] = "f8";
constexpr char kF9Name[] = "f9";
constexpr char kF10Name[] = "f10";
constexpr char kF11Name[] = "f11";
constexpr char kF12Name[] = "f12";
constexpr char kF13Name[] = "f13";
constexpr char kF14Name[] = "f14";
constexpr char kF15Name[] = "f15";
constexpr char kF16Name[] = "f16";
constexpr char kF17Name[] = "f17";
constexpr char kF18Name[] = "f18";
constexpr char kF19Name[] = "f19";
constexpr char kF20Name[] = "f20";
constexpr char kF21Name[] = "f21";
constexpr char kF22Name[] = "f22";
constexpr char kF23Name[] = "f23";
constexpr char kF24Name[] = "f24";
constexpr char kF25Name[] = "f25";
constexpr char kF26Name[] = "f26";
constexpr char kF27Name[] = "f27";
constexpr char kF28Name[] = "f28";
constexpr char kF29Name[] = "f29";
constexpr char kF30Name[] = "f30";
constexpr char kF31Name[] = "f31";

constexpr char kV0Name[] = "v0";
constexpr char kV1Name[] = "v1";
constexpr char kV2Name[] = "v2";
constexpr char kV3Name[] = "v3";
constexpr char kV4Name[] = "v4";
constexpr char kV5Name[] = "v5";
constexpr char kV6Name[] = "v6";
constexpr char kV7Name[] = "v7";
constexpr char kV8Name[] = "v8";
constexpr char kV9Name[] = "v9";
constexpr char kV10Name[] = "v10";
constexpr char kV11Name[] = "v11";
constexpr char kV12Name[] = "v12";
constexpr char kV13Name[] = "v13";
constexpr char kV14Name[] = "v14";
constexpr char kV15Name[] = "v15";
constexpr char kV16Name[] = "v16";
constexpr char kV17Name[] = "v17";
constexpr char kV18Name[] = "v18";
constexpr char kV19Name[] = "v19";
constexpr char kV20Name[] = "v20";
constexpr char kV21Name[] = "v21";
constexpr char kV22Name[] = "v22";
constexpr char kV23Name[] = "v23";
constexpr char kV24Name[] = "v24";
constexpr char kV25Name[] = "v25";
constexpr char kV26Name[] = "v26";
constexpr char kV27Name[] = "v27";
constexpr char kV28Name[] = "v28";
constexpr char kV29Name[] = "v29";
constexpr char kV30Name[] = "v30";
constexpr char kV31Name[] = "v31";

RiscVDebugInfo::RiscVDebugInfo()
    : debug_register_map_({{*DebugRegisterEnum::kPc, kPcName},
                           {*DebugRegisterEnum::kX0, kX0Name},
                           {*DebugRegisterEnum::kX1, kX1Name},
                           {*DebugRegisterEnum::kX2, kX2Name},
                           {*DebugRegisterEnum::kX3, kX3Name},
                           {*DebugRegisterEnum::kX4, kX4Name},
                           {*DebugRegisterEnum::kX5, kX5Name},
                           {*DebugRegisterEnum::kX6, kX6Name},
                           {*DebugRegisterEnum::kX7, kX7Name},
                           {*DebugRegisterEnum::kX8, kX8Name},
                           {*DebugRegisterEnum::kX9, kX9Name},
                           {*DebugRegisterEnum::kX10, kX10Name},
                           {*DebugRegisterEnum::kX11, kX11Name},
                           {*DebugRegisterEnum::kX12, kX12Name},
                           {*DebugRegisterEnum::kX13, kX13Name},
                           {*DebugRegisterEnum::kX14, kX14Name},
                           {*DebugRegisterEnum::kX15, kX15Name},
                           {*DebugRegisterEnum::kX16, kX16Name},
                           {*DebugRegisterEnum::kX17, kX17Name},
                           {*DebugRegisterEnum::kX18, kX18Name},
                           {*DebugRegisterEnum::kX19, kX19Name},
                           {*DebugRegisterEnum::kX20, kX20Name},
                           {*DebugRegisterEnum::kX21, kX21Name},
                           {*DebugRegisterEnum::kX22, kX22Name},
                           {*DebugRegisterEnum::kX23, kX23Name},
                           {*DebugRegisterEnum::kX24, kX24Name},
                           {*DebugRegisterEnum::kX25, kX25Name},
                           {*DebugRegisterEnum::kX26, kX26Name},
                           {*DebugRegisterEnum::kX27, kX27Name},
                           {*DebugRegisterEnum::kX28, kX28Name},
                           {*DebugRegisterEnum::kX29, kX29Name},
                           {*DebugRegisterEnum::kX30, kX30Name},
                           {*DebugRegisterEnum::kX31, kX31Name},
                           {*DebugRegisterEnum::kF0, kF0Name},
                           {*DebugRegisterEnum::kF1, kF1Name},
                           {*DebugRegisterEnum::kF2, kF2Name},
                           {*DebugRegisterEnum::kF3, kF3Name},
                           {*DebugRegisterEnum::kF4, kF4Name},
                           {*DebugRegisterEnum::kF5, kF5Name},
                           {*DebugRegisterEnum::kF6, kF6Name},
                           {*DebugRegisterEnum::kF7, kF7Name},
                           {*DebugRegisterEnum::kF8, kF8Name},
                           {*DebugRegisterEnum::kF9, kF9Name},
                           {*DebugRegisterEnum::kF10, kF10Name},
                           {*DebugRegisterEnum::kF11, kF11Name},
                           {*DebugRegisterEnum::kF12, kF12Name},
                           {*DebugRegisterEnum::kF13, kF13Name},
                           {*DebugRegisterEnum::kF14, kF14Name},
                           {*DebugRegisterEnum::kF15, kF15Name},
                           {*DebugRegisterEnum::kF16, kF16Name},
                           {*DebugRegisterEnum::kF17, kF17Name},
                           {*DebugRegisterEnum::kF18, kF18Name},
                           {*DebugRegisterEnum::kF19, kF19Name},
                           {*DebugRegisterEnum::kF20, kF20Name},
                           {*DebugRegisterEnum::kF21, kF21Name},
                           {*DebugRegisterEnum::kF22, kF22Name},
                           {*DebugRegisterEnum::kF23, kF23Name},
                           {*DebugRegisterEnum::kF24, kF24Name},
                           {*DebugRegisterEnum::kF25, kF25Name},
                           {*DebugRegisterEnum::kF26, kF26Name},
                           {*DebugRegisterEnum::kF27, kF27Name},
                           {*DebugRegisterEnum::kF28, kF28Name},
                           {*DebugRegisterEnum::kF29, kF29Name},
                           {*DebugRegisterEnum::kF30, kF30Name},
                           {*DebugRegisterEnum::kF31, kF31Name},
                           {*DebugRegisterEnum::kV0, kV0Name},
                           {*DebugRegisterEnum::kV1, kV1Name},
                           {*DebugRegisterEnum::kV2, kV2Name},
                           {*DebugRegisterEnum::kV3, kV3Name},
                           {*DebugRegisterEnum::kV4, kV4Name},
                           {*DebugRegisterEnum::kV5, kV5Name},
                           {*DebugRegisterEnum::kV6, kV6Name},
                           {*DebugRegisterEnum::kV7, kV7Name},
                           {*DebugRegisterEnum::kV8, kV8Name},
                           {*DebugRegisterEnum::kV9, kV9Name},
                           {*DebugRegisterEnum::kV10, kV10Name},
                           {*DebugRegisterEnum::kV11, kV11Name},
                           {*DebugRegisterEnum::kV12, kV12Name},
                           {*DebugRegisterEnum::kV13, kV13Name},
                           {*DebugRegisterEnum::kV14, kV14Name},
                           {*DebugRegisterEnum::kV15, kV15Name},
                           {*DebugRegisterEnum::kV16, kV16Name},
                           {*DebugRegisterEnum::kV17, kV17Name},
                           {*DebugRegisterEnum::kV18, kV18Name},
                           {*DebugRegisterEnum::kV19, kV19Name},
                           {*DebugRegisterEnum::kV20, kV20Name},
                           {*DebugRegisterEnum::kV21, kV21Name},
                           {*DebugRegisterEnum::kV22, kV22Name},
                           {*DebugRegisterEnum::kV23, kV23Name},
                           {*DebugRegisterEnum::kV24, kV24Name},
                           {*DebugRegisterEnum::kV25, kV25Name},
                           {*DebugRegisterEnum::kV26, kV26Name},
                           {*DebugRegisterEnum::kV27, kV27Name},
                           {*DebugRegisterEnum::kV28, kV28Name},
                           {*DebugRegisterEnum::kV29, kV29Name},
                           {*DebugRegisterEnum::kV30, kV30Name},
                           {*DebugRegisterEnum::kV31, kV31Name}}) {}

RiscVDebugInfo *RiscVDebugInfo::Instance() {
  static RiscVDebugInfo *instance_ = nullptr;
  if (instance_ == nullptr) {
    instance_ = new RiscVDebugInfo();
  }
  return instance_;
}

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
