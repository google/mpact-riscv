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
    sha256 = "1f0e6ea27b0487a5d997f85efaebdf60a1f1dbb478c30b25d3c6b41e9d4b4028",
    strip_prefix = "mpact-sim-e1b7b0adeb53875908995674a8555b68c4821903",
    url = "https://github.com/google/mpact-sim/archive/e1b7b0adeb53875908995674a8555b68c4821903.tar.gz",
)


load("@com_google_mpact-sim//:repos.bzl", "mpact_sim_repos")

mpact_sim_repos()

load("@com_google_mpact-sim//:deps.bzl", "mpact_sim_deps")

mpact_sim_deps()
