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
    size_t bytes_read = fread(src, 1, sz, f);
    (void)bytes_read;
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
    
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    
    clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "No GPU device found\n");
        return 1;
    }
    
    char name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    printf("Device: %s\n", name);
    
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateCommandQueue failed: %d\n", err);
        return 1;
    }
    
    printf("OpenCL initialized successfully!\n");
    
    char *kernel_source = load_kernel_source("rinhash_argon2d.cl");
    cl_program program = clCreateProgramWithSource(ctx, 1, (const char**)&kernel_source, NULL, &err);
    
    err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build failed:\n%s\n", log);
        free(log);
        return 1;
    }
    
    cl_kernel kernel = clCreateKernel(program, "argon2d_core", &err);
    
    size_t m_bytes = (size_t)m_cost_kb * 1024;
    cl_mem d_mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE, m_bytes, NULL, &err);
    cl_mem d_prehash = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 32, NULL, &err);
    cl_mem d_output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, 32, NULL, &err);
    cl_mem d_state = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(argon2_state_t), NULL, &err);
    
    uint8_t prehash[32], output[32];
    memset(prehash, 0x42, 32);
    
    argon2_state_t state = {0, 0, 0, 100, m_cost_kb, 1};
    
    clEnqueueWriteBuffer(queue, d_prehash, CL_TRUE, 0, 32, prehash, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_state, CL_TRUE, 0, sizeof(state), &state, 0, NULL, NULL);
    
    uint32_t t_cost = 2, lanes = 1;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_prehash);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_mem);
    clSetKernelArg(kernel, 2, sizeof(uint32_t), &m_cost_kb);
    clSetKernelArg(kernel, 3, sizeof(uint32_t), &t_cost);
    clSetKernelArg(kernel, 4, sizeof(uint32_t), &lanes);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_state);
    
    size_t global_size = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Kernel execution failed: %d\n", err);
        return 1;
    }
    
    clFinish(queue);
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, 32, output, 0, NULL, NULL);
    
    printf("Test output: ");
    for (int i = 0; i < 32; i++) printf("%02x", output[i]);
    printf("\n\nGPU Mining ready!\n");
    
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
