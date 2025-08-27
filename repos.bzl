# Copyright 2024 Google LLC
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

"""Load dependent repositories"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def mpact_riscv_repos():
    """ Load dependencies needed to use mpact-riscv as a 3rd-party consumer"""

    if not native.existing_rule("com_google_mpact-sim"):
        http_archive(
            name = "com_google_mpact-sim",
            sha256 = "4e36793123257a004c30c6bfc9cf5c86575cdc00336b0efb56f5262bb0e9a614",
            strip_prefix = "mpact-sim-ced9aa21b5ba0f0f6d4c3ffa5875947f281eb3d8",
            url = "https://github.com/google/mpact-sim/archive/ced9aa21b5ba0f0f6d4c3ffa5875947f281eb3d8.tar.gz",
        )
