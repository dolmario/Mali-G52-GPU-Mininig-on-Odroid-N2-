cmake_minimum_required(VERSION 3.10)
project(rinhash_gpu C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -march=native")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Dependencies ----
find_package(OpenCL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENSSL REQUIRED openssl)
find_package(Threads REQUIRED)

# ---- Argon2 (REF ONLY, ohne SIMD/opt.c) ----
# Erwartete Dateien/Ordner:
#   src/argon2.c
#   src/core.c
#   src/encoding.c
#   src/ref.c
#   src/thread.c
#   src/blake2/blake2b.c
#   src/blake2/blake2.h
#   src/blake2/blake2-impl.h
add_library(argon2 STATIC
  src/argon2.c
  src/core.c
  src/encoding.c
  src/ref.c
  src/thread.c
  src/blake2/blake2b.c
)
target_include_directories(argon2 PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/blake2
)
target_link_libraries(argon2 PUBLIC Threads::Threads)

# ---- Miner ----
add_executable(rin_ocl
  src/main.c
  src/blake3.c
  src/blake3_portable.c
  src/blake3_dispatch.c
  src/blake3_neon.c      # AArch64/NEON ok
  # x86-SIMD-Quellen NICHT einbinden (sse2/sse41/avx2/avx512)
)

target_include_directories(rin_ocl PRIVATE
  ${OpenCL_INCLUDE_DIRS}
  ${OPENSSL_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/blake2
)

link_directories(${OPENSSL_LIBRARY_DIRS})

target_link_libraries(rin_ocl PRIVATE
  ${OpenCL_LIBRARIES}
  ${OPENSSL_LIBRARIES}
  argon2
  m
  pthread
)

# Kernel ins Build-Verzeichnis kopieren
configure_file(${CMAKE_SOURCE_DIR}/src/rinhash_argon2d.cl
               ${CMAKE_BINARY_DIR}/rinhash_argon2d.cl COPYONLY)
