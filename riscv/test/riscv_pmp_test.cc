#include "riscv/riscv_pmp.h"

#include <cstdint>

#include "absl/strings/str_cat.h"
#include "googlemock/include/gmock/gmock.h"  // IWYU pragma: keep
#include "googletest/include/gtest/gtest.h"
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
  for (int i = 0; i < 16; ++i) {
    EXPECT_TRUE(
        state.csr_set()->GetCsr(absl::StrCat("pmpcfg", i)).status().ok());
    EXPECT_TRUE(
        state.csr_set()->GetCsr(*RiscVCsrEnum::kPmpCfg0 + i).status().ok());
  }
  for (int i = 0; i < 64; ++i) {
    EXPECT_TRUE(
        state.csr_set()->GetCsr(absl::StrCat("pmpaddr", i)).status().ok());
    EXPECT_TRUE(
        state.csr_set()->GetCsr(*RiscVCsrEnum::kPmpAddr0 + i).status().ok());
  }
  // Verify specific RV32 standard CSR addresses.
  EXPECT_EQ(*RiscVCsrEnum::kPmpCfg0, 0x3a0);
  EXPECT_EQ(*RiscVCsrEnum::kPmpCfg15, 0x3af);
  EXPECT_EQ(*RiscVCsrEnum::kPmpAddr0, 0x3b0);
  EXPECT_EQ(*RiscVCsrEnum::kPmpAddr16, 0x3c0);
  EXPECT_EQ(*RiscVCsrEnum::kPmpAddr31, 0x3cf);
  EXPECT_EQ(*RiscVCsrEnum::kPmpAddr63, 0x3ef);
}

// Test that the expected PMP CSRs are created.
TEST(RiscVPmpTest, CreatePmpCsrs64) {
  RiscVState state("test", RiscVXlen::RV64, nullptr, nullptr);
  RiscVPmp pmp(&state);
  pmp.CreatePmpCsrs<uint64_t, RiscVCsrEnum>(state.csr_set());
  for (int i = 0; i < 16; ++i) {
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
  for (int i = 0; i < 64; ++i) {
    EXPECT_TRUE(
        state.csr_set()->GetCsr(absl::StrCat("pmpaddr", i)).status().ok());
    EXPECT_TRUE(
        state.csr_set()->GetCsr(*RiscVCsrEnum::kPmpAddr0 + i).status().ok());
  }
}

}  // namespace
