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

"""Set up extra repositories with the dependencies"""

load("@com_google_mpact-sim//:deps.bzl", "mpact_sim_deps")

def mpact_riscv_deps():
    """ Extra dependencies to finish setting up Google repositories"""

    mpact_sim_deps()
