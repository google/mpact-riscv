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

# Build rules for RiscV simulator code.

load("@com_google_mpact-sim//mpact/sim/decoder:mpact_sim_isa.bzl", "mpact_bin_fmt_decoder", "mpact_isa_decoder")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files([
    "riscv32g.bin_fmt",
    "riscv32g.isa",
    "riscv32v.bin_fmt",
    "riscv32v.isa",
    "riscv64g.bin_fmt",
    "riscv64g.isa",
    "riscv64v.bin_fmt",
    "riscv64v.isa",
    "riscv_vector.isa",
    "riscv_vector.bin_fmt",
])

config_setting(
    name = "arm_cpu",
    values = {"cpu": "arm"},
)

cc_library(
    name = "riscv_state",
    srcs = [
        "riscv_csr.cc",
        "riscv_fp_state.cc",
        "riscv_misa.cc",
        "riscv_register.cc",
        "riscv_sim_csrs.cc",
        "riscv_state.cc",
        "riscv_vector_state.cc",
        "riscv_xip_xie.cc",
        "riscv_xstatus.cc",
    ] + select({
        "arm_cpu": ["riscv_fp_host_arm.cc"],
        "//conditions:default": ["riscv_fp_host_x86.cc"],
    }),
    hdrs = [
        "riscv_csr.h",
        "riscv_fp_host.h",
        "riscv_fp_info.h",
        "riscv_fp_state.h",
        "riscv_misa.h",
        "riscv_register.h",
        "riscv_register_aliases.h",
        "riscv_sim_csrs.h",
        "riscv_state.h",
        "riscv_vector_state.h",
        "riscv_xip_xie.h",
        "riscv_xstatus.h",
    ],
    copts = [
        "-O3",
        "-ffp-model=strict",
        "-fprotect-parens",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "riscv_g",
    srcs = [
        "riscv_a_instructions.cc",
        "riscv_b_instructions.cc",
        "riscv_d_instructions.cc",
        "riscv_f_instructions.cc",
        "riscv_i_instructions.cc",
        "riscv_m_instructions.cc",
        "riscv_priv_instructions.cc",
        "riscv_zfencei_instructions.cc",
        "riscv_zicsr_instructions.cc",
    ],
    hdrs = [
        "riscv_a_instructions.h",
        "riscv_b_instructions.h",
        "riscv_d_instructions.h",
        "riscv_f_instructions.h",
        "riscv_i_instructions.h",
        "riscv_instruction_helpers.h",
        "riscv_m_instructions.h",
        "riscv_priv_instructions.h",
        "riscv_zfencei_instructions.h",
        "riscv_zicsr_instructions.h",
    ],
    copts = [
        "-O3",
        "-ffp-model=strict",
        "-fprotect-parens",
    ],
    deps = [
        ":riscv_state",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "riscv_v",
    srcs = [
        "riscv_vector_fp_compare_instructions.cc",
        "riscv_vector_fp_instructions.cc",
        "riscv_vector_fp_reduction_instructions.cc",
        "riscv_vector_fp_unary_instructions.cc",
        "riscv_vector_memory_instructions.cc",
        "riscv_vector_opi_instructions.cc",
        "riscv_vector_opm_instructions.cc",
        "riscv_vector_permute_instructions.cc",
        "riscv_vector_reduction_instructions.cc",
        "riscv_vector_unary_instructions.cc",
    ],
    hdrs = [
        "riscv_vector_fp_compare_instructions.h",
        "riscv_vector_fp_instructions.h",
        "riscv_vector_fp_reduction_instructions.h",
        "riscv_vector_fp_unary_instructions.h",
        "riscv_vector_instruction_helpers.h",
        "riscv_vector_memory_instructions.h",
        "riscv_vector_opi_instructions.h",
        "riscv_vector_opm_instructions.h",
        "riscv_vector_permute_instructions.h",
        "riscv_vector_reduction_instructions.h",
        "riscv_vector_unary_instructions.h",
    ],
    copts = [
        "-O3",
        "-ffp-model=strict",
        "-fprotect-parens",
    ],
    deps = [
        ":riscv_g",
        ":riscv_state",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
    ],
)

mpact_isa_decoder(
    name = "riscv32g_isa",
    src = "riscv32g.isa",
    includes = [],
    isa_name = "RiscV32G",
    deps = [
        ":riscv_g",
        "@com_google_absl//absl/functional:bind_front",
    ],
)

mpact_bin_fmt_decoder(
    name = "riscv32g_bin_fmt",
    src = "riscv32g.bin_fmt",
    decoder_name = "RiscV32G",
    includes = [
        "riscv32v.bin_fmt",
    ],
    deps = [
        ":riscv32g_isa",
    ],
)

mpact_isa_decoder(
    name = "riscv32gv_isa",
    src = "riscv32v.isa",
    includes = [
        "riscv32g.isa",
        "riscv_vector.isa",
    ],
    isa_name = "RiscV32GV",
    prefix = "riscv32gv",
    deps = [
        ":riscv_g",
        ":riscv_v",
        "@com_google_absl//absl/functional:bind_front",
    ],
)

mpact_bin_fmt_decoder(
    name = "riscv32gv_bin_fmt",
    src = "riscv32v.bin_fmt",
    decoder_name = "RiscV32GV",
    includes = [
        "riscv32g.bin_fmt",
        "riscv_vector.bin_fmt",
    ],
    prefix = "riscv32gv",
    deps = [
        ":riscv32gv_isa",
    ],
)

mpact_isa_decoder(
    name = "riscv64g_isa",
    src = "riscv64g.isa",
    includes = [],
    isa_name = "RiscV64G",
    deps = [
        ":riscv_g",
        ":riscv_v",
        "@com_google_absl//absl/functional:bind_front",
    ],
)

mpact_bin_fmt_decoder(
    name = "riscv64g_bin_fmt",
    src = "riscv64g.bin_fmt",
    decoder_name = "RiscV64G",
    includes = [
    ],
    prefix = "riscv64g",
    deps = [
        ":riscv64g_isa",
    ],
)

mpact_isa_decoder(
    name = "riscv64gv_isa",
    src = "riscv64v.isa",
    includes = [
        "riscv64g.isa",
        "riscv_vector.isa",
    ],
    isa_name = "RiscV64GV",
    prefix = "riscv64gv",
    deps = [
        ":riscv_g",
        ":riscv_v",
        "@com_google_absl//absl/functional:bind_front",
    ],
)

mpact_bin_fmt_decoder(
    name = "riscv64gv_bin_fmt",
    src = "riscv64v.bin_fmt",
    decoder_name = "RiscV64GV",
    includes = [
        "riscv64g.bin_fmt",
        "riscv_vector.bin_fmt",
    ],
    prefix = "riscv64gv",
    deps = [
        ":riscv64gv_isa",
    ],
)

cc_library(
    name = "riscv32g_decoder",
    srcs = [
        "riscv32_decoder.cc",
        "riscv32g_encoding.cc",
    ],
    hdrs = [
        "riscv32_decoder.h",
        "riscv32g_encoding.h",
    ],
    copts = ["-O3"],
    deps = [
        ":riscv32g_bin_fmt",
        ":riscv32g_isa",
        ":riscv_g",
        ":riscv_state",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:program_error",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "riscv32g_vec_decoder",
    srcs = [
        "riscv32g_vec_decoder.cc",
        "riscv32g_vec_encoding.cc",
    ],
    hdrs = [
        "riscv32g_vec_decoder.h",
        "riscv32g_vec_encoding.h",
    ],
    copts = ["-O3"],
    deps = [
        ":riscv32g_bin_fmt",
        ":riscv32g_isa",
        ":riscv32gv_bin_fmt",
        ":riscv32gv_isa",
        ":riscv_g",
        ":riscv_state",
        ":riscv_v",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:program_error",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "riscv64g_decoder",
    srcs = [
        "riscv64_decoder.cc",
        "riscv64g_encoding.cc",
    ],
    hdrs = [
        "riscv64_decoder.h",
        "riscv64g_encoding.h",
    ],
    copts = ["-O3"],
    deps = [
        ":riscv64g_bin_fmt",
        ":riscv64g_isa",
        ":riscv_g",
        ":riscv_state",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:program_error",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "riscv64g_vec_decoder",
    srcs = [
        "riscv64g_vec_decoder.cc",
        "riscv64g_vec_encoding.cc",
    ],
    hdrs = [
        "riscv64g_vec_decoder.h",
        "riscv64g_vec_encoding.h",
    ],
    copts = ["-O3"],
    deps = [
        ":riscv64gv_bin_fmt",
        ":riscv64gv_isa",
        ":riscv_state",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:program_error",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "riscv_top",
    srcs = [
        "riscv_top.cc",
    ],
    hdrs = [
        "riscv_top.h",
    ],
    copts = ["-O3"],
    deps = [
        ":riscv32g_decoder",
        ":riscv32g_isa",
        ":riscv64g_decoder",
        ":riscv64g_isa",
        ":riscv_breakpoint",
        ":riscv_debug_info",
        ":riscv_state",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:component",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
        "@com_google_mpact-sim//mpact/sim/generic:counters",
        "@com_google_mpact-sim//mpact/sim/generic:decode_cache",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "riscv_arm_semihost",
    srcs = ["riscv_arm_semihost.cc"],
    hdrs = ["riscv_arm_semihost.h"],
    copts = ["-O3"],
    deps = [
        ":riscv_state",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "riscv32_htif_semihost",
    srcs = [
        "riscv32_htif_semihost.cc",
    ],
    hdrs = [
        "riscv32_htif_semihost.h",
    ],
    copts = ["-O3"],
    deps = [
        ":riscv_g",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "riscv_breakpoint",
    srcs = [
        "riscv_breakpoint.cc",
    ],
    hdrs = [
        "riscv_breakpoint.h",
    ],
    copts = ["-O3"],
    deps = [
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-sim//mpact/sim/generic:component",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "debug_command_shell",
    srcs = [
        "debug_command_shell.cc",
    ],
    hdrs = [
        "debug_command_shell.h",
    ],
    copts = ["-O3"],
    deps = [
        ":stoull_wrapper",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "riscv_debug_info",
    srcs = ["riscv_debug_info.cc"],
    hdrs = ["riscv_debug_info.h"],
    copts = ["-O3"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
    ],
)

cc_library(
    name = "stoull_wrapper",
    srcs = [
        "stoull_wrapper.cc",
    ],
    hdrs = [
        "stoull_wrapper.h",
    ],
    copts = [
        "-O3",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_binary(
    name = "rv32g_sim",
    srcs = [
        "rv32g_sim.cc",
    ],
    copts = ["-O3"],
    deps = [
        ":debug_command_shell",
        ":riscv32_htif_semihost",
        ":riscv_arm_semihost",
        ":riscv_top",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:counters",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/proto:component_data_cc_proto",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_binary(
    name = "rv64g_sim",
    srcs = [
        "rv64g_sim.cc",
    ],
    copts = ["-O3"],
    deps = [
        ":debug_command_shell",
        ":riscv_arm_semihost",
        ":riscv_top",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:counters",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/proto:component_data_cc_proto",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "riscv_test_mem_watcher",
    srcs = [
        "riscv_test_mem_watcher.cc",
    ],
    hdrs = [
        "riscv_test_mem_watcher.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_binary(
    name = "rv32g_test_sim",
    srcs = [
        "rv32g_test_sim.cc",
    ],
    copts = ["-O3"],
    deps = [
        ":riscv_top",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
    ],
)

cc_binary(
    name = "rv64g_test_sim",
    srcs = [
        "rv64g_test_sim.cc",
    ],
    copts = ["-O3"],
    deps = [
        ":riscv_state",
        ":riscv_test_mem_watcher",
        ":riscv_top",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
    ],
)
