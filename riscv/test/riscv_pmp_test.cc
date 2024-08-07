#include "riscv/riscv_pmp.h"

#include <cstdint>

#include "absl/strings/str_cat.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/type_helpers.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_state.h"

namespace {

using ::mpact::sim::riscv::RiscVCsrEnum;
using ::mpact::sim::riscv::RiscVPmp;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::generic::operator*;  // NOLINT: used below (clang error).

// Test that the expected PMP CSRs are created.
TEST(RiscVPmpTest, CreatePmpCsrs32) {
  RiscVState state("test", RiscVXlen::RV32, nullptr, nullptr);
  RiscVPmp pmp(&state);
  pmp.CreatePmpCsrs<uint32_t, RiscVCsrEnum>(state.csr_set());
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(
        state.csr_set()->GetCsr(absl::StrCat("pmpcfg", i)).status().ok());
    EXPECT_TRUE(
        state.csr_set()->GetCsr(*RiscVCsrEnum::kPmpCfg0 + i).status().ok());
  }
  for (int i = 0; i < 16; ++i) {
    EXPECT_TRUE(
        state.csr_set()->GetCsr(absl::StrCat("pmpaddr", i)).status().ok());
    EXPECT_TRUE(
        state.csr_set()->GetCsr(*RiscVCsrEnum::kPmpAddr0 + i).status().ok());
  }
}

// Test that the expected PMP CSRs are created.
TEST(RiscVPmpTest, CreatePmpCsrs64) {
  RiscVState state("test", RiscVXlen::RV64, nullptr, nullptr);
  RiscVPmp pmp(&state);
  pmp.CreatePmpCsrs<uint64_t, RiscVCsrEnum>(state.csr_set());
  for (int i = 0; i < 4; ++i) {
    // For RV64, only the even numbered PMP configuration registers are created.
    if (i & 0x1) {
      EXPECT_FALSE(
          state.csr_set()->GetCsr(absl::StrCat("pmpcfg", i)).status().ok());
      EXPECT_FALSE(
          state.csr_set()->GetCsr(*RiscVCsrEnum::kPmpCfg0 + i).status().ok());
    } else {
      EXPECT_TRUE(
          state.csr_set()->GetCsr(absl::StrCat("pmpcfg", i)).status().ok());
      EXPECT_TRUE(
          state.csr_set()->GetCsr(*RiscVCsrEnum::kPmpCfg0 + i).status().ok());
    }
  }
  for (int i = 0; i < 16; ++i) {
    EXPECT_TRUE(
        state.csr_set()->GetCsr(absl::StrCat("pmpaddr", i)).status().ok());
    EXPECT_TRUE(
        state.csr_set()->GetCsr(*RiscVCsrEnum::kPmpAddr0 + i).status().ok());
  }
}

}  // namespace
