# CMake toolchain file for cross-compiling to RISC-V 64-bit
#
# Usage:
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/riscv64-gcc.cmake ..

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Cross-compiler
set(CMAKE_C_COMPILER   riscv64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)

# Search paths for cross-compiled libraries
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# RISC-V specific flags
set(CMAKE_CXX_FLAGS_INIT "-march=rv64gc -mabi=lp64d")
set(CMAKE_C_FLAGS_INIT   "-march=rv64gc -mabi=lp64d")

# Use QEMU user-mode emulation to run cross-compiled binaries
find_program(QEMU_RV64 qemu-riscv64-static qemu-riscv64)
if(QEMU_RV64)
	set(CMAKE_CROSSCOMPILING_EMULATOR "${QEMU_RV64}" "-L" "/usr/riscv64-linux-gnu")
endif()
