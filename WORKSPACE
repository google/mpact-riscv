# Copyright 2023 Google LLC
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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# MPACT-Sim repo
http_archive(
    name = "com_google_mpact-sim",
    sha256 = "97eda611a23b898626ae7ecec523501f2f94a51d1c39b2dbf11a153b4b73be7e",
    strip_prefix = "mpact-sim-06435b2083fbe6bb5a69005e7db68c6ab1d0817b",
    url = "https://github.com/google/mpact-sim/archive/06435b2083fbe6bb5a69005e7db68c6ab1d0817b.tar.gz",
)


load("@com_google_mpact-sim//:repos.bzl", "mpact_sim_repos")

mpact_sim_repos()

load("@com_google_mpact-sim//:deps.bzl", "mpact_sim_deps")

mpact_sim_deps()
