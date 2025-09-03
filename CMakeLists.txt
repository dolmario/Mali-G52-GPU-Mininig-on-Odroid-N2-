cmake_minimum_required(VERSION 3.10)
project(rinhash_gpu C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -march=native")

find_package(OpenCL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENSSL REQUIRED openssl)

include_directories(${OpenCL_INCLUDE_DIRS} ${OPENSSL_INCLUDE_DIRS})
link_directories(${OPENSSL_LIBRARY_DIRS})

add_executable(rin_ocl
    src/main.c
    blake3.c
    blake3_portable.c
)

target_link_libraries(rin_ocl ${OpenCL_LIBRARIES} ${OPENSSL_LIBRARIES} m pthread)

# Kernel ins Build-Verzeichnis kopieren
configure_file(${CMAKE_SOURCE_DIR}/src/rinhash_argon2d.cl ${CMAKE_BINARY_DIR}/rinhash_argon2d.cl COPYONLY)

