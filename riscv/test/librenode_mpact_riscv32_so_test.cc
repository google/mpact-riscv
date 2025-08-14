#include <dlfcn.h>

#include <cstdint>
#include <cstring>
#include <string>

#include "absl/debugging/leak_check.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "googlemock/include/gmock/gmock.h"
#include "mpact/sim/util/renode/renode_debug_interface.h"

namespace {

constexpr char kFileName[] = "librenode_mpact_riscv32.so";
constexpr char kDepotPath[] = "riscv/";
constexpr char kExecFileName[] = "hello_world_arm.elf";

using ::mpact::sim::util::renode::RenodeCpuRegister;

class LibRenodeMpactRiscV32SoTest : public ::testing::Test {
 protected:
  LibRenodeMpactRiscV32SoTest() {
    std::string path = absl::StrCat(kDepotPath, kFileName);
    absl::LeakCheckDisabler disabler;  // Ignore leaks from dlopen.
    lib_ = dlopen(path.c_str(), RTLD_LAZY);
    CHECK_NE(lib_, nullptr);
  }

  ~LibRenodeMpactRiscV32SoTest() { dlclose(lib_); }

  void* lib_ = nullptr;
};

TEST_F(LibRenodeMpactRiscV32SoTest, Construct) {
  EXPECT_NE(dlsym(lib_, "construct"), nullptr);
}

TEST_F(LibRenodeMpactRiscV32SoTest, ConstructWithSysbus) {
  EXPECT_NE(dlsym(lib_, "construct_with_sysbus"), nullptr);
}

TEST_F(LibRenodeMpactRiscV32SoTest, Connect) {
  EXPECT_NE(dlsym(lib_, "connect"), nullptr);
}

TEST_F(LibRenodeMpactRiscV32SoTest, ConnectWithSysbus) {
  EXPECT_NE(dlsym(lib_, "connect_with_sysbus"), nullptr);
}

TEST_F(LibRenodeMpactRiscV32SoTest, Destruct) {
  EXPECT_NE(dlsym(lib_, "destruct"), nullptr);
}

TEST_F(LibRenodeMpactRiscV32SoTest, GetRegInfoSize) {
  EXPECT_NE(dlsym(lib_, "get_reg_info_size"), nullptr);
}

TEST_F(LibRenodeMpactRiscV32SoTest, GetRegInfo) {
  EXPECT_NE(dlsym(lib_, "get_reg_info"), nullptr);
}

TEST_F(LibRenodeMpactRiscV32SoTest, LoadElf) {
  EXPECT_NE(dlsym(lib_, "load_elf"), nullptr);
}

TEST_F(LibRenodeMpactRiscV32SoTest, ReadRegister) {
  EXPECT_NE(dlsym(lib_, "read_register"), nullptr);
}

TEST_F(LibRenodeMpactRiscV32SoTest, WriteRegister) {
  EXPECT_NE(dlsym(lib_, "write_register"), nullptr);
}

TEST_F(LibRenodeMpactRiscV32SoTest, ReadMemory) {
  EXPECT_NE(dlsym(lib_, "read_memory"), nullptr);
}

TEST_F(LibRenodeMpactRiscV32SoTest, WriteMemory) {
  EXPECT_NE(dlsym(lib_, "write_memory"), nullptr);
}

TEST_F(LibRenodeMpactRiscV32SoTest, Reset) {
  EXPECT_NE(dlsym(lib_, "reset"), nullptr);
}

TEST_F(LibRenodeMpactRiscV32SoTest, Step) {
  EXPECT_NE(dlsym(lib_, "step"), nullptr);
}

TEST_F(LibRenodeMpactRiscV32SoTest, SetConfig) {
  EXPECT_NE(dlsym(lib_, "set_config"), nullptr);
}

TEST_F(LibRenodeMpactRiscV32SoTest, SetIrqValue) {
  EXPECT_NE(dlsym(lib_, "set_irq_value"), nullptr);
}

using ConstructType = int32_t (*)(char*, int32_t);
using StepType = uint64_t (*)(int32_t, uint64_t, int32_t*);
using SetConfigType = int32_t (*)(int32_t, const char*[], const char*[],
                                  int32_t);
using LoadElfType = uint64_t (*)(int32_t, const char*, bool, int32_t*);
using GetRegInfoSize = int32_t (*)(int32_t);
using GetRegInfo = int32_t (*)(int32_t, int32_t, char* name, void* info);
using WriteRegisterType = void (*)(int32_t, int32_t, uint64_t);
using DestructType = void (*)(int32_t);

TEST_F(LibRenodeMpactRiscV32SoTest, RunProgram) {
  absl::LeakCheckDisabler disabler;
  // Load the function pointers.
  ConstructType construct =
      reinterpret_cast<ConstructType>(dlsym(lib_, "construct"));
  StepType step = reinterpret_cast<StepType>(dlsym(lib_, "step"));
  SetConfigType set_config =
      reinterpret_cast<SetConfigType>(dlsym(lib_, "set_config"));
  LoadElfType load_elf = reinterpret_cast<LoadElfType>(dlsym(lib_, "load_elf"));
  GetRegInfoSize get_reg_info_size =
      reinterpret_cast<GetRegInfoSize>(dlsym(lib_, "get_reg_info_size"));
  GetRegInfo get_reg_info =
      reinterpret_cast<GetRegInfo>(dlsym(lib_, "get_reg_info"));
  WriteRegisterType write_register =
      reinterpret_cast<WriteRegisterType>(dlsym(lib_, "write_register"));
  DestructType destruct =
      reinterpret_cast<DestructType>(dlsym(lib_, "destruct"));
  // Verify that the function pointers are valid.
  CHECK_NE(construct, nullptr);
  CHECK_NE(step, nullptr);
  CHECK_NE(set_config, nullptr);
  CHECK_NE(load_elf, nullptr);
  CHECK_NE(get_reg_info_size, nullptr);
  CHECK_NE(get_reg_info, nullptr);
  // Construct a simulator instance.
  char cpu_type[] = "Mpact.RiscV32";
  int32_t id = construct(cpu_type, 256);
  CHECK_GE(id, 0);
  // Load the program.
  int32_t status = 0;
  std::string path =
      absl::StrCat(kDepotPath, "/test/testfiles/", kExecFileName);
  uint64_t entry_pt =
      load_elf(id, path.c_str(), /*for_symbols_only=*/false, &status);
  CHECK_GE(status, 0);
  int num_regs = get_reg_info_size(id);
  CHECK_GT(num_regs, 0);
  // Set configuration items.
  const char* kInstProfile = "instProfile";
  const char* kMemProfile = "memProfile";
  const char* kStackEnd = "stackEnd";
  const char* kMemoryBase = "memoryBase";
  const char* kMemorySize = "memorySize";

  const char* kConfigItems[] = {kInstProfile, kMemProfile, kStackEnd,
                                kMemoryBase, kMemorySize};
  const char* kConfigValues[] = {"1", "1", "0x00030000", "0x00000000",
                                 "0x10000000"};
  status = set_config(id, kConfigItems, kConfigValues, 5);
  CHECK_EQ(status, 0);
  // Set the pc to the entry point.
  for (int i = 0; i < num_regs; ++i) {
    char name[256];
    RenodeCpuRegister reg_info;
    status = get_reg_info(id, i, name, &reg_info);
    if (strncmp(name, "pc", 2) == 0) {
      write_register(id, reg_info.index, entry_pt);
      break;
    }
  }
  CHECK_EQ(status, 0);
  // Capture stdout.
  testing::internal::CaptureStdout();
  // Step the program until it completes.
  uint64_t total_stepped = 0;
  uint64_t num_stepped;
  do {
    num_stepped = step(id, 10'000, &status);
    total_stepped += num_stepped;
  } while (num_stepped > 10'000 && status == 0 && total_stepped < 100'000);
  EXPECT_EQ("Hello World! 5\n", testing::internal::GetCapturedStdout());
  destruct(id);
}

}  // namespace
