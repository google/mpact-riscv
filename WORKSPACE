# Copyright 2023 - 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

workspace(name = "com_google_mpact-riscv")

# First load the immediate repo dependencies (mpact-sim).
load("@com_google_mpact-riscv//:repos.bzl", "mpact_riscv_repos")

mpact_riscv_repos()

# Next load in the transitive repo dependencies.
load("@com_google_mpact-riscv//:dep_repos.bzl", "mpact_riscv_dep_repos")

mpact_riscv_dep_repos()

# Call the deps function. It will call any other dependent deps functions.
load("@com_google_mpact-riscv//:deps.bzl", "mpact_riscv_deps");

mpact_riscv_deps()

# Load the protobuf deps from mpact_sim
load("@com_google_mpact-sim//:protobuf_deps.bzl", "mpact_sim_protobuf_deps")

mpact_sim_protobuf_deps()

