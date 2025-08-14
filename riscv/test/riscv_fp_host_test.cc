#include "riscv/riscv_fp_host.h"

#include <cstdint>
#include <ios>

#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/generic/instruction.h"

namespace {

using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::GetHostFloatingPointInterface;
using ::mpact::sim::riscv::HostFloatingPointInterface;
using ::mpact::sim::riscv::ScopedFPStatus;

class RiscVFPHostTest : public testing::Test {
 protected:
  RiscVFPHostTest() {
    host_fp_interface_ = GetHostFloatingPointInterface();
    instruction_ = new Instruction(0x1000, nullptr);
    instruction_->set_size(4);
  }

  ~RiscVFPHostTest() override {
    delete host_fp_interface_;
    delete instruction_;
  }

  HostFloatingPointInterface* host_fp_interface_;
  Instruction* instruction_;
};

static float sum = 0.0;

static void simple_add(const Instruction* inst) { sum += 1.0; }

TEST_F(RiscVFPHostTest, SetGetCsr) {
  // Run through all the different values of the csr status bits.
  for (uint32_t i = 0; i < (1 << 5); i++) {
    host_fp_interface_->SetRiscVFcsr(i);
    EXPECT_EQ(host_fp_interface_->GetRiscVFcsr(), i) << std::hex << i;
  }
}

TEST_F(RiscVFPHostTest, CheckCsrIsPreserved) {
  // Execute a simple add instruction (which will not modify the status bits).
  // Make sure the status bits stay the same.
  for (uint32_t status = 0; status < (1 << 5); status++) {
    // Rounding mode is round-to-nearest.
    host_fp_interface_->SetRiscVFcsr(status);
    instruction_->set_semantic_function(&simple_add);
    {
      ScopedFPStatus set_fp_status(host_fp_interface_);
      instruction_->Execute(nullptr);
    }
    EXPECT_EQ(host_fp_interface_->GetRiscVFcsr(), status) << std::hex << status;
  }
}

}  // namespace
