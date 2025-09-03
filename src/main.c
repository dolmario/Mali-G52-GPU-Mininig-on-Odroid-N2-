#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <openssl/evp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "blake3.h"

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
    // Standard: 64 MB
    uint32_t m_cost_kb = 64 * 1024;
    if (argc > 1) {
        int mb = atoi(argv[1]);
        if (mb >= 8 && mb <= 256) m_cost_kb = mb * 1024;
    }
    printf("Using %u MB memory\n", m_cost_kb / 1024);

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "No GPU device found\n"); return 1; }

    char name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    printf("Device: %s\n", name);

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "clCreateCommandQueue failed: %d\n", err); return 1; }

    printf("OpenCL initialized successfully!\n");

    // Kernel laden
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
    cl_mem d_mem     = clCreateBuffer(ctx, CL_MEM_READ_WRITE, m_bytes, NULL, &err);
    cl_mem d_prehash = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 32, NULL, &err);
    cl_mem d_output  = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, 32, NULL, &err);

    // Dummy Header → Prehash
    uint8_t header[80];
    memset(header, 0xAB, 80);
    uint8_t prehash[32], output[32];
    blake3_hash(header, 80, prehash);

    clEnqueueWriteBuffer(queue, d_prehash, CL_TRUE, 0, 32, prehash, 0, NULL, NULL);

    uint32_t t_cost = 2, lanes = 1;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_prehash);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_mem);
    clSetKernelArg(kernel, 2, sizeof(uint32_t), &m_cost_kb);
    clSetKernelArg(kernel, 3, sizeof(uint32_t), &t_cost);
    clSetKernelArg(kernel, 4, sizeof(uint32_t), &lanes);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_output);

    size_t global_size = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "Kernel execution failed: %d\n", err); return 1; }

    clFinish(queue);
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, 32, output, 0, NULL, NULL);

    // Final SHA3
    uint8_t final_hash[32];
    sha3_256(output, 32, final_hash);

    printf("Final RinHash: ");
    for (int i = 0; i < 32; i++) printf("%02x", final_hash[i]);
    printf("\n");

    // Cleanup
    clReleaseMemObject(d_mem);
    clReleaseMemObject(d_prehash);
    clReleaseMemObject(d_output);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(kernel_source);
    return 0;
}
