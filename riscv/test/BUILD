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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_applicable_licenses = ["//:license"])

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
        "//riscv:riscv_fp_state",
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
        "//riscv:riscv_fp_state",
        "//riscv:riscv_state",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
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
        "@com_google_mpact-sim//mpact/sim/util/memory",
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
        "@com_google_mpact-sim//mpact/sim/util/memory",
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
        "@com_google_mpact-sim//mpact/sim/util/memory",
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
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/util/memory",
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
        "@com_google_mpact-sim//mpact/sim/util/memory",
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
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "riscv_d_instructions_test",
    size = "small",
    srcs = ["riscv_d_instructions_test.cc"],
    copts = select({
        "darwin_arm64_cpu": ["-ffp-model=strict"],
        "//conditions:default": [
            "-ffp-model=strict",
            "-fprotect-parens",
        ],
    }),
    tags = ["not_run:arm"],
    deps = [
        ":riscv_fp_test_base",
        "//riscv:riscv_fp_state",
        "//riscv:riscv_g",
        "//riscv:riscv_state",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
    ],
)

cc_test(
    name = "riscv_f_instructions_test",
    size = "small",
    srcs = ["riscv_f_instructions_test.cc"],
    copts = select({
        "darwin_arm64_cpu": ["-ffp-model=strict"],
        "//conditions:default": [
            "-ffp-model=strict",
            "-fprotect-parens",
        ],
    }),
    tags = ["not_run:arm"],
    deps = [
        ":riscv_fp_test_base",
        "//riscv:riscv_fp_state",
        "//riscv:riscv_g",
        "//riscv:riscv_state",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
    ],
)

cc_test(
    name = "riscv_zfh_instructions_test",
    size = "small",
    srcs = ["riscv_zfh_instructions_test.cc"],
    copts = select({
        "darwin_arm64_cpu": ["-ffp-model=strict"],
        "//conditions:default": [
            "-ffp-model=strict",
            "-fprotect-parens",
        ],
    }),
    tags = ["not_run:arm"],
    deps = [
        ":riscv_fp_test_base",
        "//riscv:riscv_fp_state",
        "//riscv:riscv_g",
        "//riscv:riscv_state",
        "//riscv:riscv_zfh_instructions",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
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
        "//riscv:riscv32g_isa",
        "//riscv:riscv_state",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "riscv32g_vec_encoding_test",
    size = "small",
    srcs = [
        "riscv32g_vec_encoding_test.cc",
    ],
    deps = [
        "//riscv:riscv32g_vec_decoder",
        "//riscv:riscv32gv_isa",
        "//riscv:riscv_state",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "riscv32gzb_encoding_test",
    size = "small",
    srcs = [
        "riscv32gzb_encoding_test.cc",
    ],
    deps = [
        "//riscv:riscv32g_bitmanip_decoder",
        "//riscv:riscv32gzb_isa",
        "//riscv:riscv_state",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "riscv32gzb_vec_encoding_test",
    size = "small",
    srcs = [
        "riscv32gzb_vec_encoding_test.cc",
    ],
    deps = [
        "//riscv:riscv32gvzb_isa",
        "//riscv:riscv32gzb_vec_decoder",
        "//riscv:riscv_state",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
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
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "riscv64g_vec_encoding_test",
    size = "small",
    srcs = [
        "riscv64g_vec_encoding_test.cc",
    ],
    deps = [
        "//riscv:riscv64g_vec_decoder",
        "//riscv:riscv64gv_isa",
        "//riscv:riscv_state",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "riscv64gzb_encoding_test",
    size = "small",
    srcs = [
        "riscv64gzb_encoding_test.cc",
    ],
    deps = [
        "//riscv:riscv64g_bitmanip_decoder",
        "//riscv:riscv64gzb_isa",
        "//riscv:riscv_state",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "riscv64gzb_vec_encoding_test",
    size = "small",
    srcs = [
        "riscv64gzb_vec_encoding_test.cc",
    ],
    deps = [
        "//riscv:riscv64gvzb_isa",
        "//riscv:riscv64gzb_vec_decoder",
        "//riscv:riscv_state",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "zvbb_encoding_test",
    size = "small",
    srcs = [
        "zvbb_encoding_test.cc",
    ],
    deps = [
        "//riscv:riscv_state",
        "//riscv:zvbb_decoder",
        "//riscv:zvbb_isa",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "zfh_encoding_test",
    size = "small",
    srcs = [
        "riscv_getters_zfh.h",
        "zfh_encoding.h",
        "zfh_encoding_test.cc",
    ],
    deps = [
        "//riscv:riscv_encoding_common",
        "//riscv:riscv_getters",
        "//riscv:riscv_state",
        "//riscv:zfh32_bin_fmt",
        "//riscv:zfh32_isa",
        "//riscv:zfh64_bin_fmt",
        "//riscv:zfh64_isa",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
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
        "@com_google_absl//absl/strings",
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
        "//riscv:riscv_action_point_memory_interface",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:action_points",
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
        "//riscv:riscv32g_isa",
        "//riscv:riscv_state",
        "@com_github_serge1_elfio//:elfio",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
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
        "//riscv:riscv64g_isa",
        "//riscv:riscv_state",
        "@com_github_serge1_elfio//:elfio",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:log_sink_registry",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_google_mpact-sim//mpact/sim/util/other:log_sink",
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
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
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
        "@com_google_absl//absl/types:span",
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
    copts = select({
        "darwin_arm64_cpu": ["-ffp-model=strict"],
        "//conditions:default": [
            "-ffp-model=strict",
            "-fprotect-parens",
        ],
    }),
    deps = [
        ":riscv_vector_fp_test_utilities",
        ":riscv_vector_instructions_test_base",
        "//riscv:riscv_fp_state",
        "//riscv:riscv_state",
        "//riscv:riscv_v",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
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
    copts = select({
        "darwin_arm64_cpu": ["-ffp-model=strict"],
        "//conditions:default": [
            "-ffp-model=strict",
            "-fprotect-parens",
        ],
    }),
    deps = [
        ":riscv_vector_fp_test_utilities",
        ":riscv_vector_instructions_test_base",
        "//riscv:riscv_fp_state",
        "//riscv:riscv_state",
        "//riscv:riscv_v",
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
    copts = select({
        "darwin_arm64_cpu": ["-ffp-model=strict"],
        "//conditions:default": [
            "-ffp-model=strict",
            "-fprotect-parens",
        ],
    }),
    deps = [
        ":riscv_vector_fp_test_utilities",
        ":riscv_vector_instructions_test_base",
        "//riscv:riscv_fp_state",
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
    copts = select({
        "darwin_arm64_cpu": ["-ffp-model=strict"],
        "//conditions:default": [
            "-ffp-model=strict",
            "-fprotect-parens",
        ],
    }),
    deps = [
        ":riscv_vector_fp_test_utilities",
        ":riscv_vector_instructions_test_base",
        "//riscv:riscv_fp_state",
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
        "//riscv:riscv32g_decoder",
        "//riscv:riscv64g_decoder",
        "//riscv:riscv_arm_semihost",
        "//riscv:riscv_state",
        "//riscv:riscv_top",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core",
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
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "riscv_plic_test",
    size = "small",
    srcs = [
        "riscv_plic_test.cc",
    ],
    deps = [
        "//riscv:riscv_plic",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core",
    ],
)

cc_test(
    name = "librenode_mpact_riscv32.so_test",
    size = "small",
    srcs = ["librenode_mpact_riscv32_so_test.cc"],
    data = [
        "testfiles/hello_world_arm.elf",
        "//riscv:renode_mpact_riscv32",
    ],
    tags = ["notap"],
    deps = [
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/util/renode:renode_debug_interface",
    ],
)

cc_test(
    name = "librenode_mpact_riscv64.so_test",
    size = "small",
    srcs = ["librenode_mpact_riscv64_so_test.cc"],
    data = [
        "testfiles/hello_world_64.elf",
        "//riscv:renode_mpact_riscv64",
    ],
    tags = ["notap"],
    deps = [
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/util/renode:renode_debug_interface",
    ],
)

cc_test(
    name = "riscv_fp_host_test",
    size = "small",
    srcs = ["riscv_fp_host_test.cc"],
    deps = [
        "//riscv:riscv_fp_state",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_test(
    name = "riscv_pmp_test",
    size = "small",
    srcs = ["riscv_pmp_test.cc"],
    deps = [
        "//riscv:riscv_state",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
    ],
)

cc_test(
    name = "riscv32_bitmanip_instructions_test",
    size = "small",
    srcs = ["riscv32_bitmanip_instructions_test.cc"],
    deps = [
        "//riscv:riscv_bitmanip_instructions",
        "//riscv:riscv_state",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_test(
    name = "riscv64_bitmanip_instructions_test",
    size = "small",
    srcs = ["riscv64_bitmanip_instructions_test.cc"],
    deps = [
        "//riscv:riscv_bitmanip_instructions",
        "//riscv:riscv_state",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_test(
    name = "riscv_counter_csr_test",
    size = "small",
    srcs = ["riscv_counter_csr_test.cc"],
    deps = [
        "//riscv:riscv_state",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:counters",
    ],
)

cc_test(
    name = "riscv_zicond_instructions_test",
    size = "small",
    srcs = ["riscv_zicond_instructions_test.cc"],
    deps = [
        "//riscv:riscv_g",
        "//riscv:riscv_state",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_test(
    name = "riscv_zc_instructions_test",
    size = "small",
    srcs = ["riscv_zc_instructions_test.cc"],
    deps = [
        "//riscv:riscv_state",
        "//riscv:rvm23_instructions",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "riscv_zc64_instructions_test",
    size = "small",
    srcs = ["riscv_zc64_instructions_test.cc"],
    deps = [
        "//riscv:riscv_state",
        "//riscv:rvm23_instructions",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "riscv_vector_basic_bit_manipulation_instructions_test",
    size = "small",
    srcs = [
        "riscv_vector_basic_bit_manipulation_test.cc",
    ],
    deps = [
        ":riscv_vector_instructions_test_base",
        "//riscv:riscv_state",
        "//riscv:riscv_vector_basic_bit_manipulation_instructions",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
    ],
)

config_setting(
    name = "arm_cpu",
    values = {"cpu": "arm"},
)

config_setting(
    name = "aarch64",
    values = {"cpu": "aarch64"},
)

config_setting(
    name = "darwin_arm64_cpu",
    values = {"cpu": "darwin_arm64"},
)
