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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

# Additional bazel rules.
http_archive(
    name = "bazel_skylib",
    sha256 = "b8a1527901774180afc798aeb28c4634bdccf19c4d98e7bdd1ce79d1fe9aaad7",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/1.4.1/bazel-skylib-1.4.1.tar.gz"],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

# MPACT-Sim repo
http_archive(
    name = "com_google_mpact-sim",
    sha256 = "8464cf3a56cbd33f73cb83943ce5ce51547cede06278546bdb187cf19fe917b2",
    strip_prefix = "mpact-sim-0.0.2",
    url = "https://github.com/google/mpact-sim/archive/refs/tags/0.0.2.tar.gz",
)

load("//@mpact-sim//:repos.bzl", "mpact_sim_repos")

mpact_sim_repos()

load("//@mpact-sim//:deps.bzl", "mpact_sim_deps")

mpact_sim_deps()

# Google re2
http_archive(
    name = "com_google_re2",
    sha256 = "7a9a4824958586980926a300b4717202485c4b4115ac031822e29aa4ef207e48",
    strip_prefix = "re2-2023-03-01",
    urls = ["https://github.com/google/re2/archive/refs/tags/2023-03-01.tar.gz"],
)
