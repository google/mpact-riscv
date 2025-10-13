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
            sha256 = "11c09c59eadca3a1d54a2fe026e419bdc80037c50dd0bbbc38fa8e112001998c",
            strip_prefix = "mpact-sim-3cd45f0735627c80593b9ff6fe87033a7c9b9089",
            url = "https://github.com/google/mpact-sim/archive/3cd45f0735627c80593b9ff6fe87033a7c9b9089.tar.gz",
        )
