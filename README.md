# MPACT-RiscV

MPACT-RiscV is an implementation of an instruction set simulator for the RiscV
instruction set architecture created using the
[MPACT-Sim](https://github.com/google/mpact-sim) simulator tools and framework,
for which there are tutorials at
[Google for Developers](https://developers.google.com/mpact-sim).

The instruction set is described in a set of `.isa` files, and the instruction
encoding in a corresponding set of `.bin_fmt` files. The instruction semantics
are implemented in a set of of functions distributed over a large set of C++
files `*_instructions.{cc,h}`.

The top level simulator control is implemented in `riscv_top.{cc,h}`, and the
simulators are instantiated from main in `rv{32,64}g_sim.cc`.

## Building

### Bazel

MPACT-Sim utilizes the [Bazel](https://bazel.build/) build system. The easiest
way to install bazel is to use
[Bazelisk](https://github.com/bazelbuild/bazelisk), a wrapper for Bazel that
automates selecting and downloading the right version of bazel. Use `brew
install bazelisk` on macOS, `choco install bazelisk` on Windows, and on linux,
download the Bazelisk binary, add it to your `PATH`, then alias bazel to the
bazelisk binary.

### Java

MPACT-Sim depends on Java, so a reasonable JRE has to be installed. For macOS,
run `brew install java`, on linux `sudo apt install default-jre`, and on Windows
follow the appropriate instructions at [java.com](https://java.com).

### Build and Test

To build the mpact-sim libraries, use the command `bazel build ...:all` from the
top level directory. To run the tests, use the command `bazel test ...:all`

The build targets `rv32g_sim` and `rv64g_sim` provide RiscV simulators for the
32 bit and 64 bit variants of RiscV scalar architectures respectively. Decoding
and execution support for vector instructions is included, but has not been
built into these targets, though adding them to these simulators is not
particularly daunting.
