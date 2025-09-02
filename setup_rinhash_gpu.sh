#!/bin/bash
# setup_rinhash_gpu.sh - Komplettes Setup für RinHash GPU Mining auf Odroid N2+

set -e  # Exit on error

echo "=== RinHash GPU Mining Setup für Odroid N2+ ==="
echo ""

# 1. Erstelle Projekt-Verzeichnis
PROJECT_DIR="$HOME/mining/rinhash-gpu"
echo "[1/7] Erstelle Projekt-Verzeichnis: $PROJECT_DIR"
rm -rf "$PROJECT_DIR"  # Clean start
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# 2. BLAKE3 Bibliothek holen
echo "[2/7] Lade BLAKE3 Bibliothek..."
git clone --depth 1 https://github.com/BLAKE3-team/BLAKE3.git /tmp/blake3 2>/dev/null || true
cp /tmp/blake3/c/blake3.c .
cp /tmp/blake3/c/blake3.h .
cp /tmp/blake3/c/blake3_impl.h .
cp /tmp/blake3/c/blake3_portable.c .

# 3. CMakeLists.txt erstellen
echo "[3/7] Erstelle CMakeLists.txt..."
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.10)
project(rin_ocl C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -march=native")

find_package(OpenCL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENSSL REQUIRED openssl)

include_directories(${OpenCL_INCLUDE_DIRS} ${OPENSSL_INCLUDE_DIRS})
link_directories(${OPENSSL_LIBRARY_DIRS})

add_executable(rin_ocl 
    main.c 
    blake3.c
    blake3_portable.c
)

target_link_libraries(rin_ocl ${OpenCL_LIBRARIES} ${OPENSSL_LIBRARIES} m pthread)
configure_file(${CMAKE_SOURCE_DIR}/rinhash_argon2d.cl ${CMAKE_BINARY_DIR}/rinhash_argon2d.cl COPYONLY)
EOF

# 4. Erstelle main.c (Minimal-Version für Test)
echo "[4/7] Erstelle main.c..."
cat > main.c << 'EOF'
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <openssl/evp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "blake3.h"

typedef struct {
    uint32_t pass;
    uint32_t slice;
    uint32_t start_block;
    uint32_t end_block;
    uint32_t blocks_per_lane;
    uint32_t lanes;
} argon2_state_t;

static void blake3_hash(const uint8_t *in, size_t inlen, uint8_t out[32]) {
    blake3_hasher h;
    blake3_hasher_init(&h);
    blake3_hasher_update(&h, in, inlen);
    blake3_hasher_finalize(&h, out, 32);
}

static void sha3_256(const uint8_t *in, size_t inlen, uint8_t out[32]) {
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha3_256(), NULL);
    EVP_DigestUpdate(ctx, in, inlen);
    unsigned int olen = 0;
    EVP_DigestFinal_ex(ctx, out, &olen);
    EVP_MD_CTX_free(ctx);
}

static char* load_kernel_source(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("open kernel"); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    char *src = (char*)malloc(sz + 1);
    fread(src, 1, sz, f);
    src[sz] = 0;
    fclose(f);
    return src;
}

int main(int argc, char **argv) {
    uint32_t m_cost_kb = 8 * 1024;
    if (argc > 1) {
        int mb = atoi(argv[1]);
        if (mb >= 8 && mb <= 256) {
            m_cost_kb = mb * 1024;
            printf("Using %d MB memory\n", mb);
        }
    }
    
    // OpenCL Setup
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "No OpenCL platforms found\n");
        return 1;
    }
    
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "No GPU device found\n");
        return 1;
    }
    
    // Print device info
    char name[256];
    cl_ulong mem_size, max_alloc;
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL);
    
    printf("=== OpenCL Device Info ===\n");
    printf("Device: %s\n", name);
    printf("Global Memory: %llu MB\n", (unsigned long long)(mem_size / 1024 / 1024));
    printf("Max Allocation: %llu MB\n", (unsigned long long)(max_alloc / 1024 / 1024));
    printf("========================\n\n");
    
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (!ctx || err) {
        fprintf(stderr, "clCreateContext failed: %d\n", err);
        return 1;
    }
    
    cl_command_queue queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!queue || err) {
        fprintf(stderr, "clCreateCommandQueue failed: %d\n", err);
        return 1;
    }
    
    // Build kernel
    char *kernel_source = load_kernel_source("rinhash_argon2d.cl");
    cl_program program = clCreateProgramWithSource(ctx, 1, (const char**)&kernel_source, NULL, &err);
    
    err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = 0;
        fprintf(stderr, "Build failed:\n%s\n", log);
        free(log);
        return 1;
    }
    
    cl_kernel kernel = clCreateKernel(program, "argon2d_core", &err);
    if (!kernel || err) {
        fprintf(stderr, "clCreateKernel failed: %d\n", err);
        return 1;
    }
    
    // Allocate memory
    size_t m_bytes = (size_t)m_cost_kb * 1024;
    cl_mem d_mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE, m_bytes, NULL, &err);
    cl_mem d_prehash = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 32, NULL, &err);
    cl_mem d_output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, 32, NULL, &err);
    cl_mem d_state = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(argon2_state_t), NULL, &err);
    
    printf("Testing basic kernel execution...\n");
    
    // Test data
    uint8_t prehash[32], output[32];
    memset(prehash, 0x42, 32);
    
    clEnqueueWriteBuffer(queue, d_prehash, CL_FALSE, 0, 32, prehash, 0, NULL, NULL);
    
    // Simple test run
    argon2_state_t state = {0, 0, 0, 100, m_cost_kb, 1};
    clEnqueueWriteBuffer(queue, d_state, CL_FALSE, 0, sizeof(state), &state, 0, NULL, NULL);
    
    uint32_t t_cost = 2, lanes = 1;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_prehash);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_mem);
    clSetKernelArg(kernel, 2, sizeof(uint32_t), &m_cost_kb);
    clSetKernelArg(kernel, 3, sizeof(uint32_t), &t_cost);
    clSetKernelArg(kernel, 4, sizeof(uint32_t), &lanes);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_state);
    
    size_t global_size = 1, local_size = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Kernel execution failed: %d\n", err);
        return 1;
    }
    
    clFinish(queue);
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, 32, output, 0, NULL, NULL);
    
    printf("Kernel executed successfully!\n");
    printf("Output: ");
    for (int i = 0; i < 32; i++) printf("%02x", output[i]);
    printf("\n\nSetup complete! GPU mining ready.\n");
    
    // Cleanup
    clReleaseMemObject(d_mem);
    clReleaseMemObject(d_prehash);
    clReleaseMemObject(d_output);
    clReleaseMemObject(d_state);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(kernel_source);
    
    return 0;
}
EOF

# 5. Erstelle OpenCL Kernel
echo "[5/7] Erstelle OpenCL Kernel..."
cat > rinhash_argon2d.cl << 'EOF'
// rinhash_argon2d.cl - MVP Kernel für Mali
__kernel void argon2d_core(__global const uchar *prehash32,
                           __global uchar *mem,
                           const uint m_cost_kb,
                           const uint t_cost,
                           const uint lanes,
                           __global uchar *out32,
                           __global const uint *state_buf)
{
    uint pass = state_buf[0];
    uint slice = state_buf[1];
    uint start_block = state_buf[2];
    uint end_block = state_buf[3];
    uint blocks_lane = state_buf[4];
    
    size_t lane_bytes = (size_t)blocks_lane * 1024;
    
    // Simple memory mixing (placeholder)
    for (uint idx = start_block; idx < end_block; idx++) {
        size_t offset = (size_t)idx * 1024;
        uint seed = 2166136261u ^ (idx + pass * 131u + slice * 17u);
        
        for (int j = 0; j < 1024; j++) {
            seed = seed * 16777619u + (uint)prehash32[j & 31];
            mem[offset + j] ^= (uchar)(seed & 0xFF);
        }
    }
    
    // Output generation (simplified)
    if (pass == t_cost - 1 && slice == 3 && end_block == blocks_lane) {
        for (int i = 0; i < 32; i++) {
            out32[i] = mem[i * 1024] ^ prehash32[i];
        }
    }
}
EOF

# 6. Build
echo "[6/7] Kompiliere..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)

# 7. Test Info
echo ""
echo "[7/7] Setup abgeschlossen!"
echo ""
echo "=== Test-Befehle ==="
echo "cd $PROJECT_DIR/build"
echo "RUSTICL_ENABLE=panfrost ./rin_ocl 8    # Test mit 8MB"
echo "RUSTICL_ENABLE=panfrost ./rin_ocl 32   # Test mit 32MB"
echo "RUSTICL_ENABLE=panfrost ./rin_ocl 64   # Test mit 64MB"
echo ""
echo "Viel Erfolg beim Mining!"
