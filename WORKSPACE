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
    sha256 = "f39cbbe26df267a6d0bd64756e46c8606d08b3f78cb407e3943862f79237bbf8",
    strip_prefix = "mpact-sim-0b5f0d18c69434e63f0ca20465067b5ef61f5046",
    url = "https://github.com/google/mpact-sim/archive/0b5f0d18c69434e63f0ca20465067b5ef61f5046.tar.gz",
)

load("@com_google_mpact-sim//:repos.bzl", "mpact_sim_repos")

mpact_sim_repos()

load("@com_google_mpact-sim//:deps.bzl", "mpact_sim_deps")

mpact_sim_deps()
