# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Unit tests for RiscV simulator classes.

package(default_applicable_licenses = ["//:license"])

config_setting(
    name = "darwin_arm64_cpu",
    values = {"cpu": "darwin_arm64"},
)

cc_library(
    name = "riscv_vector_instructions_test_base",
    testonly = True,
    hdrs = ["riscv_vector_instructions_test_base.h"],
    deps = [
        "//riscv:riscv_state",
        "//riscv:riscv_v",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "riscv_vector_fp_test_utilities",
    testonly = True,
    hdrs = ["riscv_vector_fp_test_utilities.h"],
    deps = [
        ":riscv_vector_instructions_test_base",
        "//riscv:riscv_state",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
    ],
)

cc_library(
    name = "riscv_fp_test_base",
    testonly = True,
    hdrs = ["riscv_fp_test_base.h"],
    deps = [
        "//riscv:riscv_state",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "riscv_vector_true_test",
    size = "small",
    srcs = [
        "riscv_vector_true_test.cc",
    ],
    deps = [
        "//riscv:riscv_state",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "riscv_csr_test",
    size = "small",
    srcs = [
        "riscv_csr_test.cc",
    ],
    deps = [
        "//riscv:riscv_state",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "riscv_state_test",
    size = "small",
    srcs = [
        "riscv_state_test.cc",
    ],
    deps = [
        "//riscv:riscv_state",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "riscv_zicsr_instructions_test",
    size = "small",
    srcs = [
        "riscv_zicsr_instructions_test.cc",
    ],
    deps = [
        "//riscv:riscv_g",
        "//riscv:riscv_state",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core",
    ],
)

cc_test(
    name = "riscv_i_instructions_test",
    size = "small",
    srcs = [
        "riscv_i_instructions_test.cc",
    ],
    deps = [
        "//riscv:riscv_g",
        "//riscv:riscv_state",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_test(
    name = "riscv_a_instructions_test",
    size = "small",
    srcs = [
        "riscv_a_instructions_test.cc",
    ],
    deps = [
        "//riscv:riscv_g",
        "//riscv:riscv_state",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "riscv_b_instructions_test",
    size = "small",
    srcs = ["riscv_b_instructions_test.cc"],
    deps = [
        "//riscv:riscv_g",
        "//riscv:riscv_state",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_test(
    name = "riscv_d_instructions_test",
    size = "small",
    srcs = ["riscv_d_instructions_test.cc"],
    copts = [
        "-ffp-model=strict",
    ] + select({
        "darwin_arm64_cpu": [],
        "//conditions:default": ["-fprotect-parens"],
    }),
    tags = ["not_run:arm"],
    deps = [
        ":riscv_fp_test_base",
        "//riscv:riscv_g",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_test(
    name = "riscv_f_instructions_test",
    size = "small",
    srcs = ["riscv_f_instructions_test.cc"],
    copts = [
        "-ffp-model=strict",
    ] + select({
        "darwin_arm64_cpu": [],
        "//conditions:default": ["-fprotect-parens"],
    }),
    tags = ["not_run:arm"],
    deps = [
        ":riscv_fp_test_base",
        "//riscv:riscv_g",
        "//riscv:riscv_state",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_test(
    name = "riscv32g_encoding_test",
    size = "small",
    srcs = [
        "riscv32g_encoding_test.cc",
    ],
    deps = [
        "//riscv:riscv32g_decoder",
        "//riscv:riscv_state",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "riscv64g_encoding_test",
    size = "small",
    srcs = [
        "riscv64g_encoding_test.cc",
    ],
    deps = [
        "//riscv:riscv64g_decoder",
        "//riscv:riscv64g_isa",
        "//riscv:riscv_state",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "riscv32_htif_semihost_test",
    size = "small",
    srcs = [
        "riscv32_htif_semihost_test.cc",
    ],
    deps = [
        "//riscv:riscv32_htif_semihost",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "riscv_arm_semihost_test",
    size = "small",
    srcs = [
        "riscv_arm_semihost_test.cc",
    ],
    data = [
        "testfiles/test_file.txt",
    ],
    deps = [
        "//riscv:riscv_arm_semihost",
        "//riscv:riscv_state",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "riscv_breakpoint_test",
    size = "small",
    srcs = [
        "riscv_breakpoint_test.cc",
    ],
    deps = [
        "//riscv:riscv_breakpoint",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:component",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "riscv32_decoder_test",
    size = "small",
    srcs = [
        "riscv32_decoder_test.cc",
    ],
    data = [
        "testfiles/hello_world.elf",
    ],
    deps = [
        "//riscv:riscv32g_decoder",
        "//riscv:riscv_state",
        "@com_github_serge1_elfio//:elfio",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
    ],
)

cc_test(
    name = "riscv64_decoder_test",
    size = "small",
    srcs = [
        "riscv64_decoder_test.cc",
    ],
    data = [
        "testfiles/hello_world_64.elf",
    ],
    deps = [
        "//riscv:riscv64g_decoder",
        "//riscv:riscv_state",
        "@com_github_serge1_elfio//:elfio",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
    ],
)

cc_test(
    name = "riscv32gv_bin_decoder_test",
    size = "small",
    srcs = [
        "riscv32gv_bin_decoder_test.cc",
    ],
    deps = [
        "//riscv:riscv32gv_bin_fmt",
        "//riscv:riscv32gv_isa",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "riscv_vector_memory_instructions_test",
    size = "small",
    srcs = [
        "riscv_vector_memory_instructions_test.cc",
    ],
    deps = [
        "//riscv:riscv_state",
        "//riscv:riscv_v",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "riscv_vector_opi_instructions_test",
    size = "small",
    srcs = [
        "riscv_vector_opi_instructions_test.cc",
    ],
    deps = [
        ":riscv_vector_instructions_test_base",
        "//riscv:riscv_state",
        "//riscv:riscv_v",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_test(
    name = "riscv_vector_opm_instructions_test",
    size = "small",
    srcs = [
        "riscv_vector_opm_instructions_test.cc",
    ],
    deps = [
        ":riscv_vector_instructions_test_base",
        "//riscv:riscv_state",
        "//riscv:riscv_v",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_test(
    name = "riscv_vector_reduction_instructions_test",
    size = "small",
    srcs = [
        "riscv_vector_reduction_instructions_test.cc",
    ],
    deps = [
        ":riscv_vector_instructions_test_base",
        "//riscv:riscv_state",
        "//riscv:riscv_v",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
    ],
)

cc_test(
    name = "riscv_vector_unary_instructions_test",
    size = "small",
    srcs = [
        "riscv_vector_unary_instructions_test.cc",
    ],
    deps = [
        ":riscv_vector_instructions_test_base",
        "//riscv:riscv_state",
        "//riscv:riscv_v",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
    ],
)

cc_test(
    name = "riscv_vector_fp_unary_instructions_test",
    size = "small",
    srcs = [
        "riscv_vector_fp_unary_instructions_test.cc",
    ],
    copts = [
        "-ffp-model=strict",
    ] + select({
        "darwin_arm64_cpu": [],
        "//conditions:default": ["-fprotect-parens"],
    }),
    deps = [
        ":riscv_vector_fp_test_utilities",
        ":riscv_vector_instructions_test_base",
        "//riscv:riscv_state",
        "//riscv:riscv_v",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
    ],
)

cc_test(
    name = "riscv_vector_fp_instructions_test",
    size = "small",
    srcs = [
        "riscv_vector_fp_instructions_test.cc",
    ],
    copts = [
        "-ffp-model=strict",
    ] + select({
        "darwin_arm64_cpu": [],
        "//conditions:default": ["-fprotect-parens"],
    }),
    deps = [
        ":riscv_vector_fp_test_utilities",
        ":riscv_vector_instructions_test_base",
        "//riscv:riscv_state",
        "//riscv:riscv_v",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
    ],
)

cc_test(
    name = "riscv_vector_fp_compare_instructions_test",
    size = "small",
    srcs = [
        "riscv_vector_fp_compare_instructions_test.cc",
    ],
    copts = [
        "-ffp-model=strict",
    ] + select({
        "darwin_arm64_cpu": [],
        "//conditions:default": ["-fprotect-parens"],
    }),
    deps = [
        ":riscv_vector_fp_test_utilities",
        ":riscv_vector_instructions_test_base",
        "//riscv:riscv_state",
        "//riscv:riscv_v",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_test(
    name = "riscv_vector_fp_reduction_instructions_test",
    size = "small",
    srcs = [
        "riscv_vector_fp_reduction_instructions_test.cc",
    ],
    copts = [
        "-ffp-model=strict",
    ] + select({
        "darwin_arm64_cpu": [],
        "//conditions:default": ["-fprotect-parens"],
    }),
    deps = [
        ":riscv_vector_fp_test_utilities",
        ":riscv_vector_instructions_test_base",
        "//riscv:riscv_state",
        "//riscv:riscv_v",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_test(
    name = "riscv_vector_permute_instructions_test",
    size = "small",
    srcs = [
        "riscv_vector_permute_instructions_test.cc",
    ],
    deps = [
        ":riscv_vector_instructions_test_base",
        "//riscv:riscv_state",
        "//riscv:riscv_v",
        "@com_google_absl//absl/random",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_test(
    name = "riscv_top_test",
    size = "small",
    srcs = [
        "riscv_top_test.cc",
    ],
    data = [
        "testfiles/hello_world.elf",
        "testfiles/hello_world_64.elf",
        "testfiles/hello_world_arm.elf",
    ],
    deps = [
        "//riscv:riscv32_htif_semihost",
        "//riscv:riscv_arm_semihost",
        "//riscv:riscv_state",
        "//riscv:riscv_top",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
    ],
)

cc_test(
    name = "riscv_misa_test",
    size = "small",
    srcs = [
        "riscv_misa_test.cc",
    ],
    deps = [
        "//riscv:riscv_state",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "riscv_clint_test",
    size = "small",
    srcs = [
        "riscv_clint_test.cc",
    ],
    deps = [
        "//riscv:riscv_clint",
        "//riscv:riscv_state",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:counters",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)
