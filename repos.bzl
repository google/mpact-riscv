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
            sha256 = "cd5fc8c02a89427a1ae07b9218e42ee5508d25e4e4dce9b06aee5675b92cec48",
            strip_prefix = "mpact-sim-43fbc5a8ca1cf4a1bfb66d11460ee1339145ab65",
            url = "https://github.com/google/mpact-sim/archive/43fbc5a8ca1cf4a1bfb66d11460ee1339145ab65.tar.gz",
        )
