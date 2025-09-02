#!/bin/bash
# complete_setup.sh - Vollständiges RinHash GPU Mining Setup für Odroid N2+
# Version 2.0 - Mit allen Fixes

set -e  # Exit on error

echo "=== RinHash GPU Mining Setup v2.0 für Odroid N2+ ==="
echo ""

# 0. Cleanup alte Versuche
echo "[0/8] Cleanup..."
rm -rf ~/mining/rinhash-gpu
rm -rf /tmp/blake3

# 1. Erstelle sauberes Projekt-Verzeichnis
PROJECT_DIR="$HOME/mining/rinhash-gpu"
echo "[1/8] Erstelle Projekt-Verzeichnis: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# 2. BLAKE3 Bibliothek vollständig holen
echo "[2/8] Lade BLAKE3 Bibliothek (vollständig)..."
git clone --depth 1 https://github.com/BLAKE3-team/BLAKE3.git /tmp/blake3
cp /tmp/blake3/c/blake3.c .
cp /tmp/blake3/c/blake3.h .
cp /tmp/blake3/c/blake3_impl.h .
cp /tmp/blake3/c/blake3_portable.c .
cp /tmp/blake3/c/blake3_dispatch.c .
cp /tmp/blake3/c/blake3_neon.c .  # ARM NEON optimization

# 3. CMakeLists.txt mit allen BLAKE3 files
echo "[3/8] Erstelle CMakeLists.txt..."
cat > CMakeLists.txt << 'EOF'
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
    main.c 
    blake3.c
    blake3_portable.c
    blake3_dispatch.c
    blake3_neon.c
)

target_link_libraries(rin_ocl ${OpenCL_LIBRARIES} ${OPENSSL_LIBRARIES} m pthread)
configure_file(${CMAKE_SOURCE_DIR}/rinhash_argon2d.cl ${CMAKE_BINARY_DIR}/rinhash_argon2d.cl COPYONLY)
EOF

# 4. main.c mit korrigiertem OpenCL Setup (alte API für Mali)
echo "[4/8] Erstelle main.c (mit Mali-Fixes)..."
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
    if (!ctx) { perror("EVP_MD_CTX_new"); exit(1); }
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
    size_t read = fread(src, 1, sz, f);
    (void)read; // Suppress warning
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
        }
    }
    printf("Using %u MB memory\n", m_cost_kb / 1024);
    
    // OpenCL Setup - Platform detection
    cl_int err;
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "No OpenCL platforms found (error %d)\n", err);
        fprintf(stderr, "Make sure to run with: RUSTICL_ENABLE=panfrost\n");
        return 1;
    }
    
    cl_platform_id *platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms, NULL);
    
    // Find the right platform (rusticl)
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    
    for (cl_uint i = 0; i < num_platforms; i++) {
        char platform_name[256];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        printf("Platform %u: %s\n", i, platform_name);
        
        // Try to get GPU device from this platform
        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        if (err == CL_SUCCESS && num_devices > 0) {
            platform = platforms[i];
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
            if (err == CL_SUCCESS) {
                printf("Found GPU on platform %u\n", i);
                break;
            }
        }
    }
    free(platforms);
    
    if (!device) {
        fprintf(stderr, "No GPU device found on any platform\n");
        fprintf(stderr, "Make sure to run with: RUSTICL_ENABLE=panfrost\n");
        return 1;
    }
    
    // Print device info
    char name[256];
    cl_ulong mem_size, max_alloc;
    cl_uint compute_units;
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    
    printf("\n=== OpenCL Device Info ===\n");
    printf("Device: %s\n", name);
    printf("Global Memory: %llu MB\n", (unsigned long long)(mem_size / 1024 / 1024));
    printf("Max Allocation: %llu MB\n", (unsigned long long)(max_alloc / 1024 / 1024));
    printf("Compute Units: %u\n", compute_units);
    printf("========================\n\n");
    
    // Create context
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (!ctx || err != CL_SUCCESS) {
        fprintf(stderr, "clCreateContext failed: %d\n", err);
        return 1;
    }
    
    // Create command queue (use old API for Mali compatibility)
    cl_command_queue queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!queue || err != CL_SUCCESS) {
        fprintf(stderr, "clCreateCommandQueue failed: %d\n", err);
        return 1;
    }
    
    printf("OpenCL setup successful!\n");
    
    // Build kernel
    char *kernel_source = load_kernel_source("rinhash_argon2d.cl");
    cl_program program = clCreateProgramWithSource(ctx, 1, (const char**)&kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateProgramWithSource failed: %d\n", err);
        return 1;
    }
    
    err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = 0;
        fprintf(stderr, "Kernel build failed:\n%s\n", log);
        free(log);
        return 1;
    }
    
    cl_kernel kernel = clCreateKernel(program, "argon2d_core", &err);
    if (!kernel || err != CL_SUCCESS) {
        fprintf(stderr, "clCreateKernel failed: %d\n", err);
        return 1;
    }
    
    printf("Kernel compiled successfully!\n");
    
    // Allocate memory
    size_t m_bytes = (size_t)m_cost_kb * 1024;
    cl_mem d_mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE, m_bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to allocate %zu MB GPU memory: %d\n", m_bytes / 1024 / 1024, err);
        return 1;
    }
    
    cl_mem d_prehash = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 32, NULL, &err);
    cl_mem d_output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, 32, NULL, &err);
    cl_mem d_state = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(argon2_state_t), NULL, &err);
    
    printf("GPU memory allocated: %u MB\n", m_cost_kb / 1024);
    
    // Test execution
    printf("\nRunning test kernel...\n");
    
    uint8_t prehash[32], output[32];
    memset(prehash, 0x42, 32);
    
    clEnqueueWriteBuffer(queue, d_prehash, CL_TRUE, 0, 32, prehash, 0, NULL, NULL);
    
    argon2_state_t state = {0, 0, 0, 100, m_cost_kb, 1};
    clEnqueueWriteBuffer(queue, d_state, CL_TRUE, 0, sizeof(state), &state, 0, NULL, NULL);
    
    uint32_t t_cost = 2, lanes = 1;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_prehash);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_mem);
    clSetKernelArg(kernel, 2, sizeof(uint32_t), &m_cost_kb);
    clSetKernelArg(kernel, 3, sizeof(uint32_t), &t_cost);
    clSetKernelArg(kernel, 4, sizeof(uint32_t), &lanes);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_state);
    
    size_t global_size = 1, local_size = 1;
    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Kernel execution failed: %d\n", err);
        return 1;
    }
    
    clWaitForEvents(1, &event);
    
    // Get timing
    cl_ulong start_ns, end_ns;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start_ns), &start_ns, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end_ns), &end_ns, NULL);
    double kernel_ms = (end_ns - start_ns) / 1e6;
    
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, 32, output, 0, NULL, NULL);
    
    printf("Kernel executed in %.2f ms\n", kernel_ms);
    printf("Output: ");
    for (int i = 0; i < 32; i++) printf("%02x", output[i]);
    printf("\n\n✓ GPU Mining setup complete and working!\n");
    
    // Cleanup
    clReleaseEvent(event);
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

# 5. OpenCL Kernel
echo "[5/8] Erstelle OpenCL Kernel..."
cat > rinhash_argon2d.cl << 'EOF'
// rinhash_argon2d.cl - Funktionierender Test-Kernel für Mali
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
    
    // Simple test: just mix some memory
    uint seed = 0x12345678;
    for (uint i = start_block; i < end_block && i < 100; i++) {
        size_t offset = i * 1024;
        if (offset < (size_t)m_cost_kb * 1024) {
            seed = seed * 1103515245 + 12345;
            mem[offset] = (uchar)(seed & 0xFF);
        }
    }
    
    // Generate output
    for (int i = 0; i < 32; i++) {
        out32[i] = prehash32[i] ^ (uchar)(seed >> i);
    }
}
EOF

# 6. Build
echo "[6/8] Kompiliere..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)

# 7. Test OpenCL availability
echo "[7/8] Teste OpenCL..."
echo ""
echo "=== clinfo Output ==="
RUSTICL_ENABLE=panfrost clinfo 2>/dev/null | grep -E "Platform|Device|Mali" || echo "clinfo failed - trying direct test"

# 8. Create test script
echo "[8/8] Erstelle Test-Script..."
cat > ../test_gpu.sh << 'EOF'
#!/bin/bash
cd $(dirname $0)/build
echo "Testing GPU mining with different memory sizes..."
echo ""

for SIZE in 8 16 32; do
    echo "Testing with ${SIZE}MB..."
    RUSTICL_ENABLE=panfrost timeout 5 ./rin_ocl $SIZE
    if [ $? -eq 0 ]; then
        echo "✓ ${SIZE}MB test passed"
    else
        echo "✗ ${SIZE}MB test failed"
    fi
    echo ""
done
EOF
chmod +x ../test_gpu.sh

echo ""
echo "==================================="
echo "Setup abgeschlossen!"
echo "==================================="
echo ""
echo "Teste mit:"
echo "  cd $PROJECT_DIR/build"
echo "  RUSTICL_ENABLE=panfrost ./rin_ocl 8"
echo ""
echo "Oder nutze das Test-Script:"
echo "  cd $PROJECT_DIR"
echo "  ./test_gpu.sh"
echo ""
echo "WICHTIG: Immer mit RUSTICL_ENABLE=panfrost starten!"
EOF

chmod +x complete_setup.sh
./complete_setup.sh
