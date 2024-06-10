#include <dlfcn.h>

#include <string>

#include "absl/debugging/leak_check.h"
#include "absl/strings/str_cat.h"
#include "googlemock/include/gmock/gmock.h"

namespace {

constexpr char kFileName[] = "librenode_mpact_riscv32.so";

constexpr char kDepotPath[] = "riscv/";

class LibRenodeMpactCheriotSoTest : public ::testing::Test {
 protected:
  LibRenodeMpactCheriotSoTest() {
    std::string path = absl::StrCat(kDepotPath, kFileName);
    absl::LeakCheckDisabler disabler;  // Ignore leaks from dlopen.
    lib_ = dlopen(path.c_str(), RTLD_LAZY);
  }

  ~LibRenodeMpactCheriotSoTest() { dlclose(lib_); }

  void *lib_ = nullptr;
};

TEST_F(LibRenodeMpactCheriotSoTest, Construct) {
  EXPECT_NE(dlsym(lib_, "construct"), nullptr);
}

TEST_F(LibRenodeMpactCheriotSoTest, ConstructWithSysbus) {
  EXPECT_NE(dlsym(lib_, "construct_with_sysbus"), nullptr);
}

TEST_F(LibRenodeMpactCheriotSoTest, Connect) {
  EXPECT_NE(dlsym(lib_, "connect"), nullptr);
}

TEST_F(LibRenodeMpactCheriotSoTest, ConnectWithSysbus) {
  EXPECT_NE(dlsym(lib_, "connect_with_sysbus"), nullptr);
}

TEST_F(LibRenodeMpactCheriotSoTest, Destruct) {
  EXPECT_NE(dlsym(lib_, "destruct"), nullptr);
}

TEST_F(LibRenodeMpactCheriotSoTest, GetRegInfoSize) {
  EXPECT_NE(dlsym(lib_, "get_reg_info_size"), nullptr);
}

TEST_F(LibRenodeMpactCheriotSoTest, GetRegInfo) {
  EXPECT_NE(dlsym(lib_, "get_reg_info"), nullptr);
}

TEST_F(LibRenodeMpactCheriotSoTest, LoadElf) {
  EXPECT_NE(dlsym(lib_, "load_elf"), nullptr);
}

TEST_F(LibRenodeMpactCheriotSoTest, ReadRegister) {
  EXPECT_NE(dlsym(lib_, "read_register"), nullptr);
}

TEST_F(LibRenodeMpactCheriotSoTest, WriteRegister) {
  EXPECT_NE(dlsym(lib_, "write_register"), nullptr);
}

TEST_F(LibRenodeMpactCheriotSoTest, ReadMemory) {
  EXPECT_NE(dlsym(lib_, "read_memory"), nullptr);
}

TEST_F(LibRenodeMpactCheriotSoTest, WriteMemory) {
  EXPECT_NE(dlsym(lib_, "write_memory"), nullptr);
}

TEST_F(LibRenodeMpactCheriotSoTest, Reset) {
  EXPECT_NE(dlsym(lib_, "reset"), nullptr);
}

TEST_F(LibRenodeMpactCheriotSoTest, Step) {
  EXPECT_NE(dlsym(lib_, "step"), nullptr);
}

TEST_F(LibRenodeMpactCheriotSoTest, SetConfig) {
  EXPECT_NE(dlsym(lib_, "set_config"), nullptr);
}

TEST_F(LibRenodeMpactCheriotSoTest, SetIrqValue) {
  EXPECT_NE(dlsym(lib_, "set_irq_value"), nullptr);
}

}  // namespace
