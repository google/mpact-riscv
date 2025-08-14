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

#include "riscv/riscv_m_instructions.h"

#include <limits>
#include <type_traits>

#include "absl/numeric/int128.h"
#include "mpact/sim/generic/instruction_helpers.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_register.h"

namespace mpact {
namespace sim {
namespace riscv {

using ::mpact::sim::generic::BinaryOp;
using ::mpact::sim::generic::NarrowType;
using ::mpact::sim::generic::WideType;

namespace RV32 {

using RegType = RV32Register;
using UintReg = typename std::make_unsigned<RV32Register::ValueType>::type;
using WideUintReg = typename WideType<UintReg>::type;
using IntReg = typename std::make_signed<RV32Register::ValueType>::type;
using WideIntReg = typename WideType<IntReg>::type;

void MMul(Instruction* instruction) {
  BinaryOp<UintReg, WideIntReg>(instruction, [](WideIntReg a_wide,
                                                WideIntReg b_wide) {
    WideIntReg c_wide = a_wide * b_wide;
    return static_cast<UintReg>((c_wide & std::numeric_limits<UintReg>::max()));
  });
}

void MMulh(Instruction* instruction) {
  BinaryOp<IntReg>(instruction, [](WideIntReg a_wide, WideIntReg b_wide) {
    WideIntReg c_wide = a_wide * b_wide;
    return static_cast<IntReg>(c_wide >> 32);
  });
}

void MMulhu(Instruction* instruction) {
  BinaryOp<UintReg>(instruction, [](WideUintReg a_wide, WideUintReg b_wide) {
    WideUintReg c_wide = a_wide * b_wide;
    return static_cast<UintReg>(c_wide >> 32);
  });
}

void MMulhsu(Instruction* instruction) {
  BinaryOp<UintReg, WideIntReg, WideUintReg>(
      instruction, [](WideIntReg a_wide, WideUintReg b_wide) {
        WideIntReg c_wide = a_wide * b_wide;
        return static_cast<IntReg>(c_wide >> 32);
      });
}

void MDiv(Instruction* instruction) {
  BinaryOp<IntReg>(instruction, [](IntReg a, IntReg b) -> IntReg {
    if (b == 0) return -1;
    if ((b == -1) && (a == std::numeric_limits<IntReg>::min())) {
      return std::numeric_limits<IntReg>::min();
    }
    return a / b;
  });
}

void MDivu(Instruction* instruction) {
  BinaryOp<UintReg>(instruction, [](UintReg a, UintReg b) -> UintReg {
    if (b == 0) return std::numeric_limits<UintReg>::max();
    return a / b;
  });
}

void MRem(Instruction* instruction) {
  BinaryOp<IntReg>(instruction, [](IntReg a, IntReg b) -> IntReg {
    if (b == 0) return a;
    if ((b == -1) && (a == std::numeric_limits<IntReg>::min())) {
      return 0;
    }
    return a % b;
  });
}

void MRemu(Instruction* instruction) {
  BinaryOp<UintReg>(instruction, [](UintReg a, UintReg b) {
    if (b == 0) return a;
    return a % b;
  });
}

}  // namespace RV32

namespace RV64 {

using RegType = RV64Register;
using UintReg = typename std::make_unsigned<RV64Register::ValueType>::type;
using WideUintReg = typename WideType<UintReg>::type;
using NarrowUintReg = typename NarrowType<UintReg>::type;
using IntReg = typename std::make_signed<RV64Register::ValueType>::type;
using WideIntReg = typename WideType<IntReg>::type;
using NarrowIntReg = typename NarrowType<IntReg>::type;

void MMul(Instruction* instruction) {
  BinaryOp<UintReg, WideIntReg>(instruction, [](WideIntReg a_wide,
                                                WideIntReg b_wide) {
    WideIntReg c_wide = a_wide * b_wide;
    return static_cast<UintReg>(c_wide & std::numeric_limits<UintReg>::max());
  });
}

void MMulh(Instruction* instruction) {
  BinaryOp<IntReg>(instruction, [](WideIntReg a_wide, WideIntReg b_wide) {
    WideIntReg c_wide = a_wide * b_wide;
    return static_cast<IntReg>(c_wide >> 64);
  });
}

void MMulhu(Instruction* instruction) {
  BinaryOp<UintReg>(instruction, [](WideUintReg a_wide, WideUintReg b_wide) {
    WideUintReg c_wide = a_wide * b_wide;
    return static_cast<UintReg>(c_wide >> 64);
  });
}

void MMulhsu(Instruction* instruction) {
  BinaryOp<UintReg, WideIntReg, WideUintReg>(
      instruction, [](WideIntReg a_wide, WideUintReg b_wide) {
        bool negate = false;
        WideUintReg a_u;
        if (a_wide < 0) {
          negate = true;
          a_wide = -a_wide;
        }
        a_u = static_cast<WideUintReg>(a_wide);
        WideUintReg c_u_wide = a_u * b_wide;
        WideIntReg c_wide = static_cast<WideIntReg>(c_u_wide);
        if (negate) {
          c_wide = -c_wide;
        }
        IntReg c = static_cast<IntReg>(c_wide >> 64);
        return c;
      });
}

void MDiv(Instruction* instruction) {
  BinaryOp<IntReg>(instruction, [](IntReg a, IntReg b) -> IntReg {
    if (b == 0) return -1;
    if ((b == -1) && (a == std::numeric_limits<IntReg>::min())) {
      return std::numeric_limits<IntReg>::min();
    }
    return a / b;
  });
}

void MDivu(Instruction* instruction) {
  BinaryOp<UintReg>(instruction, [](UintReg a, UintReg b) -> UintReg {
    if (b == 0) return std::numeric_limits<UintReg>::max();
    return a / b;
  });
}

void MRem(Instruction* instruction) {
  BinaryOp<IntReg>(instruction, [](IntReg a, IntReg b) -> IntReg {
    if (b == 0) return a;
    if ((b == -1) && (a == std::numeric_limits<IntReg>::min())) {
      return 0;
    }
    return a % b;
  });
}

void MRemu(Instruction* instruction) {
  BinaryOp<UintReg>(instruction, [](UintReg a, UintReg b) {
    if (b == 0) return a;
    return a % b;
  });
}

void MMulw(Instruction* instruction) {
  BinaryOp<IntReg, NarrowIntReg>(instruction,
                                 [](NarrowIntReg a, NarrowIntReg b) -> IntReg {
                                   NarrowIntReg c = a * b;
                                   IntReg c_wide = static_cast<IntReg>(c);
                                   return c_wide;
                                 });
}

void MDivw(Instruction* instruction) {
  BinaryOp<IntReg, NarrowIntReg>(
      instruction, [](NarrowIntReg a, NarrowIntReg b) -> IntReg {
        if (b == 0) return static_cast<IntReg>(-1);
        if ((b == -1) && (a == std::numeric_limits<NarrowIntReg>::min())) {
          return static_cast<IntReg>(std::numeric_limits<NarrowIntReg>::min());
        }
        auto c = a / b;
        return static_cast<IntReg>(c);
      });
}

void MDivuw(Instruction* instruction) {
  BinaryOp<IntReg, NarrowUintReg>(
      instruction, [](NarrowUintReg a, NarrowUintReg b) -> IntReg {
        if (b == 0) return std::numeric_limits<UintReg>::max();
        return static_cast<IntReg>(static_cast<NarrowIntReg>(a / b));
      });
}

void MRemw(Instruction* instruction) {
  BinaryOp<IntReg, NarrowIntReg>(
      instruction, [](NarrowIntReg a, NarrowIntReg b) -> IntReg {
        if (b == 0) return static_cast<IntReg>(a);
        if ((b == -1) && (a == std::numeric_limits<NarrowIntReg>::min())) {
          return 0;
        }
        return static_cast<IntReg>(a % b);
      });
}

void MRemuw(Instruction* instruction) {
  BinaryOp<IntReg, NarrowUintReg>(
      instruction, [](NarrowUintReg a, NarrowUintReg b) -> IntReg {
        if (b == 0) return static_cast<IntReg>(static_cast<NarrowIntReg>(a));
        return static_cast<IntReg>(static_cast<NarrowIntReg>(a % b));
      });
}

}  // namespace RV64

}  // namespace riscv
}  // namespace sim
}  // namespace mpact
