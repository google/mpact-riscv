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
            sha256 = "a8e7132b7f7c947634e2564fdacd3eb4e1876907e63a03dfa3d072962702aa81",
            strip_prefix = "mpact-sim-9c2db57c8dc8bdb7a5e76a97639f373c07739900",
            url = "https://github.com/google/mpact-sim/archive/9c2db57c8dc8bdb7a5e76a97639f373c07739900.tar.gz",
        )
