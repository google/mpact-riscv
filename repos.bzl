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
            sha256 = "da500b75747a7f35ad9ebbeae3afa902ad186ad6f5554f70efd37e6f0382125d",
            strip_prefix = "mpact-sim-2f9f9c9b5221c8d11fd6c7e8a3b365e9ae9883f7",
            url = "https://github.com/google/mpact-sim/archive/2f9f9c9b5221c8d11fd6c7e8a3b365e9ae9883f7.tar.gz",
        )
