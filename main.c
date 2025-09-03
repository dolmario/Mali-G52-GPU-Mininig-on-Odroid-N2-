// main.c - RinHash GPU Mining mit Kernel-Chunking für Mali Watchdog
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <openssl/evp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "blake3.h"

// Argon2 state für Chunking
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
    if (1 != EVP_DigestInit_ex(ctx, EVP_sha3_256(), NULL)) { perror("EVP_DigestInit"); exit(1); }
    if (1 != EVP_DigestUpdate(ctx, in, inlen)) { perror("EVP_DigestUpdate"); exit(1); }
    unsigned int olen = 0;
    if (1 != EVP_DigestFinal_ex(ctx, out, &olen)) { perror("EVP_DigestFinal"); exit(1); }
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

static int check_target(const uint8_t h[32], const uint8_t t[32]) {
    // Little-endian comparison für Mining (wichtig!)
    for (int i = 0; i < 32; i++) {
        if (h[i] < t[i]) return 1;
        if (h[i] > t[i]) return 0;
    }
    return 1;
}

static void print_device_info(cl_device_id dev) {
    char name[256];
    cl_ulong mem_size, max_alloc;
    cl_uint compute_units;
    
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    
    printf("=== OpenCL Device Info ===\n");
    printf("Device: %s\n", name);
    printf("Global Memory: %llu MB\n", (unsigned long long)(mem_size / 1024 / 1024));
    printf("Max Allocation: %llu MB\n", (unsigned long long)(max_alloc / 1024 / 1024));
    printf("Compute Units: %u\n", compute_units);
    printf("========================\n\n");
}

int main(int argc, char **argv) {
    // Parse memory size from args (default 8MB)
    uint32_t m_cost_kb = 8 * 1024;
    if (argc > 1) {
        int mb = atoi(argv[1]);
        if (mb >= 8 && mb <= 256) {
            m_cost_kb = mb * 1024;
            printf("Using %d MB memory\n", mb);
        }
    }
    
    // Target für Testing (sehr leicht)
    uint8_t target[32];
    memset(target, 0xFF, 32);
    target[0] = 0x00;  // Little-endian: 0x00FFFFFF...
    target[1] = 0x0F;
    
    // 80-Byte Block Header
    struct __attribute__((packed)) {
        uint32_t version;
        uint8_t prev_hash[32];
        uint8_t merkle_root[32];
        uint32_t timestamp;
        uint32_t bits;
        uint32_t nonce;
    } header;
    
    header.version = 1;
    memset(header.prev_hash, 0xAA, 32);
    memset(header.merkle_root, 0xBB, 32);
    header.timestamp = (uint32_t)time(NULL);
    header.bits = 0x1d00ffff;
    
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
    
    print_device_info(device);
    
    // Check max allocation
    cl_ulong max_alloc;
    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL);
    if (m_cost_kb * 1024 > max_alloc) {
        fprintf(stderr, "Warning: Requested %u MB exceeds max allocation %llu MB\n",
                m_cost_kb / 1024, (unsigned long long)(max_alloc / 1024 / 1024));
        m_cost_kb = (uint32_t)(max_alloc / 1024 / 2);  // Use half of max
        printf("Reduced to %u MB\n", m_cost_kb / 1024);
    }
    
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (!ctx || err) {
        fprintf(stderr, "clCreateContext failed: %d\n", err);
        return 1;
    }

    // ALTE API für Mali Kompatibilität (ohne Profiling)
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (!queue || err) {
        fprintf(stderr, "clCreateCommandQueue failed: %d\n", err);
        return 1;
    }
    
    // Build kernel
    char *kernel_source = load_kernel_source("rinhash_argon2d.cl");
    cl_program program = clCreateProgramWithSource(ctx, 1, (const char**)&kernel_source, NULL, &err);
    if (!program || err) {
        fprintf(stderr, "clCreateProgramWithSource failed: %d\n", err);
        return 1;
    }
    
    // Build with CL1.2 (wichtig für Mali!)
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
    
    // Allocate GPU memory
    size_t m_bytes = (size_t)m_cost_kb * 1024;
    cl_mem d_mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE, m_bytes, NULL, &err);
    if (!d_mem || err) {
        fprintf(stderr, "Failed to allocate %zu MB GPU memory: %d\n", m_bytes / 1024 / 1024, err);
        return 1;
    }
    
    cl_mem d_prehash = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 32, NULL, &err);
    cl_mem d_output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, 32, NULL, &err);
    cl_mem d_state = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(argon2_state_t), NULL, &err);
    
    // Argon2 parameters
    uint32_t t_cost = 2;  // Passes
    uint32_t lanes = 1;   // Single lane for Mali
    uint32_t blocks_per_lane = m_cost_kb;  // 1 block = 1 KiB
    
    // Chunking parameters (kritisch für Watchdog!)
    uint32_t chunks_per_pass;
    if (m_cost_kb <= 16 * 1024) chunks_per_pass = 8;       // 16MB: 8 chunks
    else if (m_cost_kb <= 64 * 1024) chunks_per_pass = 16; // 64MB: 16 chunks  
    else chunks_per_pass = 32;                              // 256MB: 32 chunks
    uint32_t slice_len = blocks_per_lane / 4;  // 4 slices in Argon2
    
    printf("Memory: %u MB, Blocks: %u, Chunks/pass: %u\n", 
           m_cost_kb / 1024, blocks_per_lane, chunks_per_pass);
    printf("Starting mining...\n\n");
    
    uint32_t shares_found = 0;
    clock_t start_time = clock();
    uint8_t prehash[32], argon_out[32], final_hash[32];
    
    // Mining loop
    for (uint32_t nonce = 0; nonce < 0xFFFFFFFF; nonce++) {
        if (nonce % 100 == 0) {
            double elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            if (elapsed > 0) {
                double hashrate = nonce / elapsed;
                printf("\rNonce: %u | Hashrate: %.2f H/s | Shares: %u", 
                       nonce, hashrate, shares_found);
                fflush(stdout);
            }
        }
        
        // Update nonce
        header.nonce = nonce;
        
        // Step 1: BLAKE3
        blake3_hash((uint8_t*)&header, 80, prehash);
        clEnqueueWriteBuffer(queue, d_prehash, CL_FALSE, 0, 32, prehash, 0, NULL, NULL);
        
        // Step 2: Argon2d (chunked!)
        for (uint32_t pass = 0; pass < t_cost; pass++) {
            for (uint32_t slice = 0; slice < 4; slice++) {
                uint32_t slice_start = slice * slice_len;
                uint32_t slice_end = (slice + 1) * slice_len;
                
                uint32_t chunk_size = slice_len / (chunks_per_pass / 4);
                if (chunk_size < 1) chunk_size = 1;
                
                for (uint32_t off = slice_start; off < slice_end; off += chunk_size) {
                    argon2_state_t state = {
                        .pass = pass,
                        .slice = slice,
                        .start_block = off,
                        .end_block = (off + chunk_size > slice_end) ? slice_end : off + chunk_size,
                        .blocks_per_lane = blocks_per_lane,
                        .lanes = lanes
                    };
                    
                    // Upload state
                    clEnqueueWriteBuffer(queue, d_state, CL_FALSE, 0, sizeof(state), &state, 0, NULL, NULL);
                    
                    // Set kernel args
                    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_prehash);
                    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_mem);
                    clSetKernelArg(kernel, 2, sizeof(uint32_t), &m_cost_kb);
                    clSetKernelArg(kernel, 3, sizeof(uint32_t), &t_cost);
                    clSetKernelArg(kernel, 4, sizeof(uint32_t), &lanes);
                    clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_output);
                    clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_state);
                    
                    // Execute chunk
                    size_t global_size = 1, local_size = 1;
                    cl_event event;
                    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, 
                                                 &global_size, &local_size, 0, NULL, &event);
                    if (err != CL_SUCCESS) {
                        fprintf(stderr, "\nKernel failed: %d\n", err);
                        goto cleanup;
                    }
                    
                    // Wait for chunk completion (wichtig!)
                    clWaitForEvents(1, &event);
                    
                    // Optional: Profile chunk time
                    if (nonce == 0 && pass == 0) {  // Only profile first nonce
                        cl_ulong start_ns, end_ns;
                        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, 
                                               sizeof(start_ns), &start_ns, NULL);
                        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, 
                                               sizeof(end_ns), &end_ns, NULL);
                        double chunk_ms = (end_ns - start_ns) / 1e6;
                        if (chunk_ms > 1000) {
                            printf("\nWarning: Chunk took %.1f ms (reduce chunk size!)\n", chunk_ms);
                        }
                    }
                    clReleaseEvent(event);
                }
            }
        }
        
        // Get Argon2d output
        clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, 32, argon_out, 0, NULL, NULL);
        
        // Step 3: SHA3-256
        sha3_256(argon_out, 32, final_hash);
        
        // Check target
        if (check_target(final_hash, target)) {
            shares_found++;
            printf("\n\n*** SHARE FOUND! ***\n");
            printf("Nonce: %u\n", nonce);
            printf("Hash: ");
            for (int i = 0; i < 32; i++) printf("%02x", final_hash[i]);
            printf("\n\n");
        }
    }
    
cleanup:
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
