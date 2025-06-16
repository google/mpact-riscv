# Copyright 2024 Google LLC
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
    "riscv32gzb.bin_fmt",
    "riscv32gzb.isa",
    "riscv32v.bin_fmt",
    "riscv32v.isa",
    "riscv32zb.bin_fmt",
    "riscv32zb.isa",
    "riscv64gzb.bin_fmt",
    "riscv64gzb.isa",
    "riscv64g.bin_fmt",
    "riscv64g.isa",
    "riscv64v.bin_fmt",
    "riscv64v.isa",
    "riscv64zb.bin_fmt",
    "riscv64zb.isa",
    "riscv_format16.bin_fmt",
    "riscv_format32.bin_fmt",
    "riscv_vector.bin_fmt",
    "riscv_vector.isa",
])

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

cc_library(
    name = "riscv_state",
    srcs = [
        "riscv_csr.cc",
        "riscv_misa.cc",
        "riscv_register.cc",
        "riscv_sim_csrs.cc",
        "riscv_state.cc",
        "riscv_vector_state.cc",
        "riscv_xip_xie.cc",
        "riscv_xstatus.cc",
    ],
    hdrs = [
        "riscv_counter_csr.h",
        "riscv_csr.h",
        "riscv_jvt.h",
        "riscv_misa.h",
        "riscv_pmp.h",
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
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:counters",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "riscv_fp_state",
    srcs = select({
        "arm_cpu": [
            "riscv_fp_host_arm.cc",
            "riscv_fp_state.cc",
        ],
        "aarch64": [
            "riscv_fp_host_arm.cc",
            "riscv_fp_state.cc",
        ],
        "darwin_arm64_cpu": [
            "riscv_fp_host_arm.cc",
            "riscv_fp_state.cc",
        ],
        "//conditions:default": [
            "riscv_fp_host_x86.cc",
            "riscv_fp_state.cc",
        ],
    }),
    hdrs = [
        "riscv_fp_host.h",
        "riscv_fp_info.h",
        "riscv_fp_state.h",
    ],
    copts = select({
        "darwin_arm64_cpu": [
            "-O3",
            "-ffp-model=strict",
        ],
        "//conditions:default": [
            "-O3",
            "-ffp-model=strict",
            "-fprotect-parens",
        ],
    }),
    deps = [
        ":riscv_state",
        "@com_google_absl//absl/log",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
    ],
)

cc_library(
    name = "riscv_g",
    srcs = [
        "riscv_a_instructions.cc",
        "riscv_d_instructions.cc",
        "riscv_f_instructions.cc",
        "riscv_i_instructions.cc",
        "riscv_m_instructions.cc",
        "riscv_priv_instructions.cc",
        "riscv_zfencei_instructions.cc",
        "riscv_zicond_instructions.cc",
        "riscv_zicsr_instructions.cc",
    ],
    hdrs = [
        "riscv_a_instructions.h",
        "riscv_d_instructions.h",
        "riscv_f_instructions.h",
        "riscv_i_instructions.h",
        "riscv_instruction_helpers.h",
        "riscv_m_instructions.h",
        "riscv_priv_instructions.h",
        "riscv_zfencei_instructions.h",
        "riscv_zicond_instructions.h",
        "riscv_zicsr_instructions.h",
    ],
    copts = select({
        "darwin_arm64_cpu": [
            "-O3",
            "-ffp-model=strict",
        ],
        "//conditions:default": [
            "-O3",
            "-ffp-model=strict",
            "-fprotect-parens",
        ],
    }),
    deps = [
        ":riscv_fp_state",
        ":riscv_state",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "riscv_bitmanip_instructions",
    srcs = [
        "riscv_bitmanip_instructions.cc",
    ],
    hdrs = [
        "riscv_bitmanip_instructions.h",
    ],
    copts = [
        "-O3",
        "-ffp-model=strict",
    ],
    deps = [
        ":riscv_g",
        ":riscv_state",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
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
    copts = select({
        "darwin_arm64_cpu": [
            "-O3",
            "-ffp-model=strict",
        ],
        "//conditions:default": [
            "-O3",
            "-ffp-model=strict",
            "-fprotect-parens",
        ],
    }),
    deps = [
        ":riscv_fp_state",
        ":riscv_g",
        ":riscv_state",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
    ],
)

cc_library(
    name = "rvm23_instructions",
    srcs = [
        "riscv_zc_instructions.cc",
        "riscv_zhintpause_instructions.cc",
        "riscv_zicbop_instructions.cc",
        "riscv_zicond_instructions.cc",
        "riscv_zihintntl_instructions.cc",
        "riscv_zimop_instructions.cc",
    ],
    hdrs = [
        "riscv_zc_instructions.h",
        "riscv_zhintpause_instructions.h",
        "riscv_zicbop_instructions.h",
        "riscv_zicond_instructions.h",
        "riscv_zihintntl_instructions.h",
        "riscv_zimop_instructions.h",
    ],
    copts = [
        "-O3",
        "-ffp-model=strict",
    ],
    deps = [
        ":riscv_g",
        ":riscv_state",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

# TODO(julianmb): Remove this target once there is a rva23_instructions target.
cc_library(
    name = "riscv_vector_basic_bit_manipulation_instructions",
    srcs = [
        "riscv_vector_basic_bit_manipulation_instructions.cc",
    ],
    hdrs = [
        "riscv_vector_basic_bit_manipulation_instructions.h",
        "riscv_vector_instruction_helpers.h",
    ],
    copts = [
        "-O3",
        "-ffp-model=strict",
    ],
    deps = [
        ":riscv_state",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/types:span",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
    ],
)

cc_library(
    name = "riscv_zfh_instructions",
    srcs = ["riscv_zfh_instructions.cc"],
    hdrs = [
        "riscv_instruction_helpers.h",
        "riscv_zfh_instructions.h",
    ],
    copts = [
        "-ffp-model=strict",
        "-O3",
    ],
    deps = [
        ":riscv_fp_state",
        ":riscv_state",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
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
        "riscv_format16.bin_fmt",
        "riscv_format32.bin_fmt",
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
        "riscv32gzb.isa",
        "riscv32zb.isa",
        "riscv_vector.isa",
    ],
    isa_name = "RiscV32GV",
    prefix = "riscv32gv",
    deps = [
        ":riscv_bitmanip_instructions",
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
        "riscv32zb.bin_fmt",
        "riscv_format16.bin_fmt",
        "riscv_format32.bin_fmt",
        "riscv_vector.bin_fmt",
    ],
    prefix = "riscv32gv",
    deps = [
        ":riscv32gv_isa",
    ],
)

mpact_isa_decoder(
    name = "riscv32gzb_isa",
    src = "riscv32gzb.isa",
    includes = [
        "riscv32g.isa",
        "riscv32zb.isa",
    ],
    isa_name = "RiscV32GZB",
    prefix = "riscv32gzb",
    deps = [
        ":riscv_bitmanip_instructions",
        ":riscv_g",
        ":riscv_v",
        "@com_google_absl//absl/functional:bind_front",
    ],
)

mpact_bin_fmt_decoder(
    name = "riscv32gzb_bin_fmt",
    src = "riscv32gzb.bin_fmt",
    decoder_name = "RiscV32GZB",
    includes = [
        "riscv32g.bin_fmt",
        "riscv32zb.bin_fmt",
        "riscv_format16.bin_fmt",
        "riscv_format32.bin_fmt",
    ],
    prefix = "riscv32gzb",
    deps = [
        ":riscv32gzb_isa",
    ],
)

mpact_isa_decoder(
    name = "riscv32gvzb_isa",
    src = "riscv32v.isa",
    includes = [
        "riscv32g.isa",
        "riscv32gzb.isa",
        "riscv32zb.isa",
        "riscv_vector.isa",
    ],
    isa_name = "RiscV32GVZB",
    prefix = "riscv32gvzb",
    deps = [
        ":riscv_bitmanip_instructions",
        ":riscv_g",
        ":riscv_v",
        "@com_google_absl//absl/functional:bind_front",
    ],
)

mpact_bin_fmt_decoder(
    name = "riscv32gvzb_bin_fmt",
    src = "riscv32v.bin_fmt",
    decoder_name = "RiscV32GVZB",
    includes = [
        "riscv32g.bin_fmt",
        "riscv32zb.bin_fmt",
        "riscv_format16.bin_fmt",
        "riscv_format32.bin_fmt",
        "riscv_vector.bin_fmt",
    ],
    prefix = "riscv32gvzb",
    deps = [
        ":riscv32gvzb_isa",
    ],
)

mpact_isa_decoder(
    name = "rvm23_isa",
    src = "rvm23.isa",
    includes = [
        "riscv32g.isa",
        "riscv32v.isa",
        "riscv32zb.isa",
        "riscv_vector.isa",
        "riscv_zc.isa",
        "riscv_zhintpause.isa",
        "riscv_zicbop.isa",
        "riscv_zicond.isa",
        "riscv_zihintntl.isa",
        "riscv_zimop.isa",
    ],
    isa_name = "RVM23",
    prefix = "rvm23",
    deps = [
        ":riscv_bitmanip_instructions",
        ":riscv_g",
        ":riscv_v",
        ":rvm23_instructions",
        "@com_google_absl//absl/functional:bind_front",
    ],
)

mpact_bin_fmt_decoder(
    name = "rvm23_bin_fmt",
    src = "rvm23.bin_fmt",
    decoder_name = "RVM23",
    includes = [
        "riscv32g.bin_fmt",
        "riscv32v.bin_fmt",
        "riscv32zb.bin_fmt",
        "riscv_format16.bin_fmt",
        "riscv_format32.bin_fmt",
        "riscv_zc.bin_fmt",
        "riscv_zhintpause.bin_fmt",
        "riscv_zicbop.bin_fmt",
        "riscv_zicond.bin_fmt",
        "riscv_zihintntl.bin_fmt",
        "riscv_zimop.bin_fmt",
    ],
    prefix = "rvm23",
    deps = [
        ":rvm23_isa",
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
        "riscv_format16.bin_fmt",
        "riscv_format32.bin_fmt",
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
        "riscv32zb.isa",
        "riscv64g.isa",
        "riscv64gzb.isa",
        "riscv64zb.isa",
        "riscv_vector.isa",
    ],
    isa_name = "RiscV64GV",
    prefix = "riscv64gv",
    deps = [
        ":riscv_bitmanip_instructions",
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
        "riscv32zb.bin_fmt",
        "riscv64g.bin_fmt",
        "riscv64zb.bin_fmt",
        "riscv_format16.bin_fmt",
        "riscv_format32.bin_fmt",
        "riscv_vector.bin_fmt",
    ],
    prefix = "riscv64gv",
    deps = [
        ":riscv64gv_isa",
    ],
)

mpact_isa_decoder(
    name = "riscv64gvzb_isa",
    src = "riscv64v.isa",
    includes = [
        "riscv32zb.isa",
        "riscv64g.isa",
        "riscv64gzb.isa",
        "riscv64zb.isa",
        "riscv_vector.isa",
    ],
    isa_name = "RiscV64GVZB",
    prefix = "riscv64gvzb",
    deps = [
        ":riscv_bitmanip_instructions",
        ":riscv_g",
        ":riscv_v",
        "@com_google_absl//absl/functional:bind_front",
    ],
)

mpact_bin_fmt_decoder(
    name = "riscv64gvzb_bin_fmt",
    src = "riscv64v.bin_fmt",
    decoder_name = "RiscV64GVZB",
    includes = [
        "riscv32zb.bin_fmt",
        "riscv64g.bin_fmt",
        "riscv64zb.bin_fmt",
        "riscv_format16.bin_fmt",
        "riscv_format32.bin_fmt",
        "riscv_vector.bin_fmt",
    ],
    prefix = "riscv64gvzb",
    deps = [
        ":riscv64gvzb_isa",
    ],
)

mpact_isa_decoder(
    name = "riscv64gzb_isa",
    src = "riscv64gzb.isa",
    includes = [
        "riscv32zb.isa",
        "riscv64g.isa",
        "riscv64zb.isa",
    ],
    isa_name = "RiscV64GZB",
    prefix = "riscv64gzb",
    deps = [
        ":riscv_bitmanip_instructions",
        ":riscv_g",
        ":riscv_v",
        "@com_google_absl//absl/functional:bind_front",
    ],
)

mpact_bin_fmt_decoder(
    name = "riscv64gzb_bin_fmt",
    src = "riscv64gzb.bin_fmt",
    decoder_name = "RiscV64GZB",
    includes = [
        "riscv32zb.bin_fmt",
        "riscv64g.bin_fmt",
        "riscv64zb.bin_fmt",
        "riscv_format16.bin_fmt",
        "riscv_format32.bin_fmt",
    ],
    prefix = "riscv64gzb",
    deps = [
        ":riscv64gzb_isa",
    ],
)

mpact_isa_decoder(
    name = "zvbb_isa",
    src = "riscv_zvbb.isa",
    includes = [
        "riscv_vector.isa",
    ],
    isa_name = "ZVBB",
    prefix = "zvbb",
    deps = [
        ":riscv_v",
        ":riscv_vector_basic_bit_manipulation_instructions",
        "@com_google_absl//absl/functional:bind_front",
    ],
)

mpact_bin_fmt_decoder(
    name = "zvbb_bin_fmt",
    src = "riscv_zvbb.bin_fmt",
    decoder_name = "ZVBB",
    includes = [
        "riscv32g.bin_fmt",
        "riscv_format16.bin_fmt",
        "riscv_format32.bin_fmt",
        "riscv_vector.bin_fmt",
    ],
    prefix = "zvbb",
    deps = [
        ":zvbb_isa",
    ],
)

mpact_isa_decoder(
    name = "zfh32_isa",
    src = "riscv_zfh.isa",
    includes = [],
    isa_name = "ZFH32",
    prefix = "zfh32",
    deps = [
        ":riscv_g",
        ":riscv_zfh_instructions",
        "@com_google_absl//absl/functional:bind_front",
    ],
)

mpact_isa_decoder(
    name = "zfh64_isa",
    src = "riscv_zfh.isa",
    includes = [],
    isa_name = "ZFH64",
    prefix = "zfh64",
    deps = [
        ":riscv_g",
        ":riscv_zfh_instructions",
        "@com_google_absl//absl/functional:bind_front",
    ],
)

mpact_bin_fmt_decoder(
    name = "zfh32_bin_fmt",
    src = "riscv_zfh32.bin_fmt",
    decoder_name = "ZFH",
    includes = [
        "riscv32g.bin_fmt",
        "riscv_format16.bin_fmt",
        "riscv_format32.bin_fmt",
    ],
    prefix = "zfh32",
    deps = [
        ":zfh32_isa",
    ],
)

mpact_bin_fmt_decoder(
    name = "zfh64_bin_fmt",
    src = "riscv_zfh64.bin_fmt",
    decoder_name = "ZFH",
    includes = [
        "riscv32g.bin_fmt",
        "riscv_format16.bin_fmt",
        "riscv_format32.bin_fmt",
    ],
    prefix = "zfh64",
    deps = [
        ":zfh64_isa",
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
        ":riscv_state",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:program_error",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "riscv_getters",
    hdrs = [
        "riscv_getter_helpers.h",
        "riscv_getters.h",
        "riscv_getters_rv32.h",
        "riscv_getters_rv64.h",
        "riscv_getters_vector.h",
        "riscv_getters_zba.h",
        "riscv_getters_zbb32.h",
        "riscv_getters_zbb64.h",
        "riscv_getters_zvbb.h",
    ],
    deps = [
        ":riscv_encoding_common",
        ":riscv_state",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
    ],
)

cc_library(
    name = "riscv_encoding_common",
    hdrs = ["riscv_encoding_common.h"],
    deps = ["@com_google_mpact-sim//mpact/sim/generic:core"],
)

cc_library(
    name = "riscv32g_bitmanip_decoder",
    srcs = [
        "riscv32g_bitmanip_decoder.cc",
        "riscv32gzb_encoding.cc",
    ],
    hdrs = [
        "riscv32g_bitmanip_decoder.h",
        "riscv32gzb_encoding.h",
    ],
    copts = ["-O3"],
    deps = [
        ":riscv32gzb_bin_fmt",
        ":riscv32gzb_isa",
        ":riscv_encoding_common",
        ":riscv_getters",
        ":riscv_state",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "riscv64g_bitmanip_decoder",
    srcs = [
        "riscv64g_bitmanip_decoder.cc",
        "riscv64gzb_encoding.cc",
    ],
    hdrs = [
        "riscv64g_bitmanip_decoder.h",
        "riscv64gzb_encoding.h",
    ],
    copts = ["-O3"],
    deps = [
        ":riscv64gzb_bin_fmt",
        ":riscv64gzb_isa",
        ":riscv_encoding_common",
        ":riscv_getters",
        ":riscv_state",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
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
        ":riscv32gv_bin_fmt",
        ":riscv32gv_isa",
        ":riscv_state",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:program_error",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "riscv32gzb_vec_decoder",
    srcs = [
        "riscv32gzb_vec_decoder.cc",
        "riscv32gzb_vec_encoding.cc",
    ],
    hdrs = [
        "riscv32gzb_vec_decoder.h",
        "riscv32gzb_vec_encoding.h",
    ],
    copts = ["-O3"],
    deps = [
        ":riscv32gvzb_bin_fmt",
        ":riscv32gvzb_isa",
        ":riscv_encoding_common",
        ":riscv_getters",
        ":riscv_state",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:program_error",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
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
        ":riscv_state",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:program_error",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "riscv64g_encoder",
    srcs = [
        "riscv64g_bin_encoder_interface.cc",
        "riscv_bin_setters.cc",
    ],
    hdrs = [
        "riscv64g_bin_encoder_interface.h",
        "riscv_bin_setters.h",
    ],
    deps = [
        ":riscv64g_bin_fmt",
        ":riscv64g_isa",
        ":riscv_getters",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/asm",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_binary(
    name = "riscv64g_as",
    srcs = ["riscv64g_as_main.cc"],
    copts = ["-O3"],
    deps = [
        ":riscv64g_encoder",
        ":riscv64g_isa",
        "@com_github_serge1_elfio//:elfio",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/asm",
        "@com_google_mpact-sim//mpact/sim/util/asm:simple_assembler",
        "@com_googlesource_code_re2//:re2",
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
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:program_error",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "riscv64gzb_vec_decoder",
    srcs = [
        "riscv64gzb_vec_decoder.cc",
        "riscv64gzb_vec_encoding.cc",
    ],
    hdrs = [
        "riscv64gzb_vec_decoder.h",
        "riscv64gzb_vec_encoding.h",
    ],
    copts = ["-O3"],
    deps = [
        ":riscv64gvzb_bin_fmt",
        ":riscv64gvzb_isa",
        ":riscv_encoding_common",
        ":riscv_getters",
        ":riscv_state",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:program_error",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "zvbb_decoder",
    srcs = [
        "zvbb_decoder.cc",
        "zvbb_encoding.cc",
    ],
    hdrs = [
        "zvbb_decoder.h",
        "zvbb_encoding.h",
    ],
    copts = ["-O3"],
    deps = [
        ":riscv_encoding_common",
        ":riscv_getters",
        ":riscv_state",
        ":riscv_vector_basic_bit_manipulation_instructions",
        ":zvbb_bin_fmt",
        ":zvbb_isa",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:program_error",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
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
        ":riscv_action_point_memory_interface",
        ":riscv_debug_interface",
        ":riscv_fp_state",
        ":riscv_state",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_mpact-sim//mpact/sim/generic:action_points",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:component",
        "@com_google_mpact-sim//mpact/sim/generic:config",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
        "@com_google_mpact-sim//mpact/sim/generic:counters",
        "@com_google_mpact-sim//mpact/sim/generic:decode_cache",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_google_mpact-sim//mpact/sim/util/memory:cache",
    ],
)

cc_library(
    name = "riscv_arm_semihost",
    srcs = ["riscv_arm_semihost.cc"],
    hdrs = ["riscv_arm_semihost.h"],
    copts = ["-O3"],
    deps = [
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "riscv_action_point_memory_interface",
    srcs = [
        "riscv_action_point_memory_interface.cc",
    ],
    hdrs = [
        "riscv_action_point_memory_interface.h",
    ],
    copts = ["-O3"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-sim//mpact/sim/generic:action_points",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "riscv_debug_interface",
    hdrs = [
        "riscv_debug_interface.h",
    ],
    deps = [
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
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
        ":riscv_debug_interface",
        ":riscv_top",
        ":stoull_wrapper",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
        "@com_google_mpact-sim//mpact/sim/generic:debug_command_shell_interface",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
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
        ":riscv32g_bitmanip_decoder",
        ":riscv32g_decoder",
        ":riscv_arm_semihost",
        ":riscv_fp_state",
        ":riscv_state",
        ":riscv_top",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
        "@com_google_mpact-sim//mpact/sim/generic:counters",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/proto:component_data_cc_proto",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
        "@com_google_protobuf//:protobuf",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_binary(
    name = "rv32gv_sim",
    srcs = [
        "rv32gv_sim.cc",
    ],
    copts = ["-O3"],
    deps = [
        ":debug_command_shell",
        ":riscv32_htif_semihost",
        ":riscv32g_vec_decoder",
        ":riscv32gzb_vec_decoder",
        ":riscv_arm_semihost",
        ":riscv_fp_state",
        ":riscv_state",
        ":riscv_top",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
        "@com_google_mpact-sim//mpact/sim/generic:counters",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/proto:component_data_cc_proto",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
        "@com_google_protobuf//:protobuf",
        "@com_googlesource_code_re2//:re2",
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
        ":riscv64g_decoder",
        ":riscv_arm_semihost",
        ":riscv_fp_state",
        ":riscv_state",
        ":riscv_top",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
        "@com_google_mpact-sim//mpact/sim/generic:counters",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/proto:component_data_cc_proto",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
        "@com_google_protobuf//:protobuf",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_binary(
    name = "rv64gv_sim",
    srcs = [
        "rv64gv_sim.cc",
    ],
    copts = ["-O3"],
    deps = [
        ":debug_command_shell",
        ":riscv64g_vec_decoder",
        ":riscv64gzb_vec_decoder",
        ":riscv_arm_semihost",
        ":riscv_fp_state",
        ":riscv_state",
        ":riscv_top",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
        "@com_google_mpact-sim//mpact/sim/generic:counters",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/proto:component_data_cc_proto",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
        "@com_google_protobuf//:protobuf",
        "@com_googlesource_code_re2//:re2",
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

cc_library(
    name = "riscv_clint",
    srcs = [
        "riscv_clint.cc",
    ],
    hdrs = [
        "riscv_clint.h",
    ],
    deps = [
        ":riscv_state",
        "@com_google_absl//absl/log",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:counters",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "riscv_plic",
    srcs = [
        "riscv_plic.cc",
    ],
    hdrs = [
        "riscv_plic.h",
    ],
    deps = [
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_binary(
    name = "rv32g_test_sim",
    srcs = [
        "rv32g_test_sim.cc",
    ],
    copts = ["-O3"],
    deps = [
        ":riscv32g_decoder",
        ":riscv_fp_state",
        ":riscv_state",
        ":riscv_top",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
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
        ":riscv64g_decoder",
        ":riscv_state",
        ":riscv_test_mem_watcher",
        ":riscv_top",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
    ],
)

cc_library(
    name = "instrumentation",
    srcs = [
        "riscv_instrumentation_control.cc",
    ],
    hdrs = [
        "riscv_instrumentation_control.h",
    ],
    deps = [
        ":debug_command_shell",
        ":riscv_top",
        ":stoull_wrapper",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "riscv_renode",
    srcs = [
        "riscv_cli_forwarder.cc",
        "riscv_renode.cc",
        "riscv_renode_cli_top.cc",
        "riscv_renode_register_info.cc",
    ],
    hdrs = [
        "riscv_cli_forwarder.h",
        "riscv_renode.h",
        "riscv_renode_cli_top.h",
        "riscv_renode_register_info.h",
    ],
    deps = [
        ":debug_command_shell",
        ":instrumentation",
        ":riscv32g_decoder",
        ":riscv64g_decoder",
        ":riscv_arm_semihost",
        ":riscv_clint",
        ":riscv_debug_info",
        ":riscv_debug_interface",
        ":riscv_state",
        ":riscv_top",
        ":stoull_wrapper",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/proto:component_data_cc_proto",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_google_mpact-sim//mpact/sim/util/other:instruction_profiler",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
        "@com_google_mpact-sim//mpact/sim/util/renode:renode_debug_interface",
        "@com_google_mpact-sim//mpact/sim/util/renode:socket_cli",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_binary(
    name = "renode_mpact_riscv32",
    srcs = [
        "riscv32_renode.cc",
        "riscv32_renode.h",
    ],
    # List the symbols for the functions called by renode as undefined.
    linkopts = [
        "-u construct",
        "-u construct_with_sysbus",
        "-u connect",
        "-u connect_with_sysbus",
        "-u destruct",
        "-u get_reg_info_size",
        "-u get_reg_info",
        "-u load_elf",
        "-u read_register",
        "-u write_register",
        "-u read_memory",
        "-u write_memory",
        "-u reset",
        "-u step",
        "-u set_config",
        "-u set_irq_value",
    ],
    linkshared = True,
    linkstatic = True,
    deps = [
        ":riscv_renode",
        ":riscv_state",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_google_mpact-sim//mpact/sim/util/renode",
        "@com_google_mpact-sim//mpact/sim/util/renode:renode_debug_interface",
    ],
)

cc_binary(
    name = "renode_mpact_riscv64",
    srcs = [
        "riscv64_renode.cc",
        "riscv64_renode.h",
    ],
    # List the symbols for the functions called by renode as undefined.
    linkopts = [
        "-u construct",
        "-u construct_with_sysbus",
        "-u connect",
        "-u connect_with_sysbus",
        "-u destruct",
        "-u get_reg_info_size",
        "-u get_reg_info",
        "-u load_elf",
        "-u read_register",
        "-u write_register",
        "-u read_memory",
        "-u write_memory",
        "-u reset",
        "-u step",
        "-u set_config",
        "-u set_irq_value",
    ],
    linkshared = True,
    linkstatic = True,
    deps = [
        ":riscv_renode",
        ":riscv_state",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_google_mpact-sim//mpact/sim/util/renode",
        "@com_google_mpact-sim//mpact/sim/util/renode:renode_debug_interface",
    ],
)
