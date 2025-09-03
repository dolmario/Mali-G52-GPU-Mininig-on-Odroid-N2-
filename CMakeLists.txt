cmake_minimum_required(VERSION 3.10)
project(rin_ocl C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -march=native -Wno-unused-result")

find_package(OpenCL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENSSL REQUIRED openssl)

include_directories(${OpenCL_INCLUDE_DIRS} ${OPENSSL_INCLUDE_DIRS})
link_directories(${OPENSSL_LIBRARY_DIRS})

add_executable(rin_ocl 
    src/main.c 
    src/blake3.c
    src/blake3_portable.c
    src/blake3_dispatch.c
    src/blake3_neon.c
)

target_link_libraries(rin_ocl ${OpenCL_LIBRARIES} ${OPENSSL_LIBRARIES} m pthread)
configure_file(${CMAKE_SOURCE_DIR}/kernels/rinhash_argon2d.cl ${CMAKE_BINARY_DIR}/rinhash_argon2d.cl COPYONLY)
