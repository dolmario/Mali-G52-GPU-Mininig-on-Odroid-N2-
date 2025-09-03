#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <stdint.h>
#include <pthread.h>
#include <sys/select.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include "blake3.h"

// ============================ Konfig =============================
static const char *POOL = "rinhash.eu.mine.zergpool.com";
static const int   PORT = 7148;
static const char *WAL  = "DTXoRQ7Zpw3FVRW2DWkLrM9Skj4Gp9SeSj";
static const char *PASS = "c=DOGE,ID=n2plus";

// Chunk-Größe in Blöcken (1 Block = 1 KB)
// 2048 ist ein guter Start für Mali, 4096 probieren wenn stabil.
#define CHUNK_BLOCKS 2048
static const uint32_t T_COST = 2;    // Argon2d-Pässe (RinHash nutzt i.d.R. 2)

// ============================ Utils ==============================
static void blake3_hash32(const uint8_t *in, size_t len, uint8_t out[32]) {
    blake3_hasher h; blake3_hasher_init(&h);
    blake3_hasher_update(&h, in, len);
    blake3_hasher_finalize(&h, out, 32);
}

static void sha3_256(const uint8_t *in, size_t len, uint8_t out[32]) {
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha3_256(), NULL);
    EVP_DigestUpdate(ctx, in, len);
    unsigned int olen = 0;
    EVP_DigestFinal_ex(ctx, out, &olen);
    EVP_MD_CTX_free(ctx);
}

static void sha256_once(const uint8_t *in, size_t len, uint8_t out[32]) {
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha256(), NULL);
    EVP_DigestUpdate(ctx, in, len);
    unsigned int olen = 0;
    EVP_DigestFinal_ex(ctx, out, &olen);
    EVP_MD_CTX_free(ctx);
}
static void double_sha256(const uint8_t *in, size_t len, uint8_t out[32]) {
    uint8_t t[32]; sha256_once(in, len, t); sha256_once(t, 32, out);
}

static int hex2bin(const char *hex, uint8_t *out, size_t outlen) {
    for (size_t i=0;i<outlen;i++) {
        unsigned v; if (sscanf(hex + 2*i, "%2x", &v) != 1) return 0;
        out[i] = (uint8_t)v;
    }
    return 1;
}
static void flip32(uint8_t *p){ for (int i=0;i<16;i++){ uint8_t t=p[i]; p[i]=p[31-i]; p[31-i]=t; } }

static void target_from_nbits(uint32_t nbits, uint8_t target[32]) {
    memset(target, 0, 32);
    uint32_t exp = nbits >> 24;
    uint32_t mant = nbits & 0xFFFFFFu;
    if (exp <= 3) {
        mant >>= 8*(3-exp);
        target[29] = (mant >> 16) & 0xFF;
        target[30] = (mant >> 8)  & 0xFF;
        target[31] = mant & 0xFF;
    } else {
        int idx = 32 - exp; if (idx < 0) idx = 0;
        if (idx + 3 <= 32) {
            target[idx]   = (mant >> 16) & 0xFF;
            target[idx+1] = (mant >> 8)  & 0xFF;
            target[idx+2] = mant & 0xFF;
        }
    }
}

// ========================== Stratum ==============================
typedef struct {
    char host[256];
    int  port;
    int  sock;
    char wallet[128];
    char pass[128];
    char extranonce1[64];
    uint32_t extranonce2_size;
} stratum_ctx_t;

typedef struct {
    char     job_id[128];
    char     prevhash_hex[65];      // BE hex
    char     coinb1_hex[2048];
    char     coinb2_hex[2048];
    char     merkle_hex[16][65];
    int      merkle_count;
    uint32_t version;
    uint32_t nbits;
    uint32_t ntime;
    int      clean;
} stratum_job_t;

static int sock_connect(const char *host, int port) {
    struct addrinfo hints = {0}, *res=NULL,*p=NULL; 
    char portstr[16];
    snprintf(portstr,sizeof portstr,"%d",port);
    hints.ai_socktype = SOCK_STREAM; hints.ai_family = AF_UNSPEC;
    if (getaddrinfo(host,portstr,&hints,&res)!=0) return -1;
    int s=-1;
    for (p=res;p;p=p->ai_next) {
        s = socket(p->ai_family,p->ai_socktype,p->ai_protocol);
        if (s<0) continue;
        if (connect(s,p->ai_addr,p->ai_addrlen)==0) break;
        close(s); s=-1;
    }
    freeaddrinfo(res);
    if (s >= 0) {
        int flags = fcntl(s, F_GETFL, 0);
        fcntl(s, F_SETFL, flags | O_NONBLOCK);
    }
    return s;
}

static int send_line(int s, const char *line) {
    size_t L=strlen(line),o=0; 
    while(o<L){ ssize_t n=send(s,line+o,L-o,0); if(n<=0) return 0; o+=n; }
    return 1;
}

static int recv_line_nonblocking(int s, char *buf, size_t cap, int timeout_ms) {
    fd_set rfds; struct timeval tv;
    FD_ZERO(&rfds); FD_SET(s, &rfds);
    tv.tv_sec = timeout_ms/1000; tv.tv_usec = (timeout_ms%1000)*1000;
    int sel = select(s+1,&rfds,NULL,NULL,&tv);
    if (sel <= 0) return 0;
    size_t o=0;
    while (o+1<cap) {
        char c; ssize_t n = recv(s,&c,1,MSG_DONTWAIT);
        if (n <= 0) break;
        buf[o++] = c; if (c=='\n') break;
    }
    buf[o]=0; return (int)o;
}

static int get_next_quoted(const char **pp, const char *end, char *out, size_t cap) {
    const char *p = *pp, *q1=NULL,*q2=NULL;
    for (; p < end; p++){ if(*p=='"'){ q1=p; break; } }
    if (!q1) return 0;
    for (p=q1+1; p<end; p++){ if(*p=='"'){ q2=p; break; } }
    if (!q2) return 0;
    size_t L=(size_t)(q2-(q1+1)); if (L>=cap) L=cap-1;
    memcpy(out,q1+1,L); out[L]=0; *pp = q2+1; return 1;
}

static int stratum_parse_notify(const char *line, stratum_job_t *J){
    if (!strstr(line,"\"mining.notify\"")) return 0;
    memset(J,0,sizeof *J);
    const char *pp = strstr(line,"\"params\""); if(!pp) return 0;
    const char *lb = strchr(pp,'['); const char *rb = lb?strrchr(pp,']'):NULL;
    if(!lb||!rb||rb<=lb) return 0;
    const char *p = lb+1;
    if(!get_next_quoted(&p,rb,J->job_id,sizeof J->job_id)) return 0;
    if(!get_next_quoted(&p,rb,J->prevhash_hex,sizeof J->prevhash_hex)) return 0;
    if(!get_next_quoted(&p,rb,J->coinb1_hex,sizeof J->coinb1_hex)) return 0;
    if(!get_next_quoted(&p,rb,J->coinb2_hex,sizeof J->coinb2_hex)) return 0;

    // merkle array
    J->merkle_count = 0;
    const char *m_lb=strchr(p,'['), *m_rb=NULL;
    if (m_lb) { const char *scan=m_lb+1; for(;scan<rb;scan++){ if(*scan==']'){ m_rb=scan; break; }}}
    if (m_lb && m_rb && m_rb>m_lb) {
        const char *mp = m_lb+1;
        while (J->merkle_count < 16) {
            char tmp[65]; if(!get_next_quoted(&mp,m_rb,tmp,sizeof tmp)) break;
            snprintf(J->merkle_hex[J->merkle_count],65,"%s",tmp);
            J->merkle_count++;
        }
        p = m_rb+1;
    }

    // version, nbits, ntime (als hex in Strings)
    char vhex[16]={0}, nbhex[16]={0}, nth[16]={0};
    if(!get_next_quoted(&p,rb,vhex,sizeof vhex)) return 0; sscanf(vhex,"%x",&J->version);
    if(!get_next_quoted(&p,rb,nbhex,sizeof nbhex)) return 0; sscanf(nbhex,"%x",&J->nbits);
    if(!get_next_quoted(&p,rb,nth,sizeof nth))    return 0; sscanf(nth,"%x",&J->ntime);

    J->clean = strstr(p,"true")!=NULL;
    return 1;
}

static int stratum_connect(stratum_ctx_t *C,const char *host,int port,const char *user,const char *pass){
    memset(C,0,sizeof *C);
    snprintf(C->host,sizeof C->host,"%s",host); C->port=port;
    snprintf(C->wallet,sizeof C->wallet,"%s",user);
    snprintf(C->pass,sizeof C->pass,"%s",pass);
    C->sock = sock_connect(host,port); if (C->sock<0) return 0;

    char sub[256];
    snprintf(sub,sizeof sub,"{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"rin-ocl/0.2\"]}\n");
    if(!send_line(C->sock,sub)) return 0;

    // warte auf result mit extranonce
    char line[8192]; int ok=0;
    for (int i=0;i<50;i++){
        int n = recv_line_nonblocking(C->sock,line,sizeof line,100);
        if (n>0 && strstr(line,"\"result\"")) { ok=1; break; }
        usleep(100000);
    }
    if (!ok) return 0;

    // sehr einfache Extraktion: letztes Zahlenfeld = extranonce2_size
    const char *lb=strchr(line,'['); const char *rb=lb?strrchr(line,']'):NULL;
    if(lb&&rb){
        // extranonce1 (zweites Quoted)
        const char *p=lb+1; char dummy[256];
        get_next_quoted(&p,rb,dummy,sizeof dummy);        // sub_details
        get_next_quoted(&p,rb,C->extranonce1,sizeof C->extranonce1);
        const char *lc=strrchr(lb,',');
        C->extranonce2_size = lc ? (uint32_t)strtoul(lc+1,NULL,10) : 4;
        if(C->extranonce2_size==0) C->extranonce2_size=4;
    }

    char auth[512];
    snprintf(auth,sizeof auth,"{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\"%s\",\"%s\"]}\n",C->wallet,C->pass);
    if(!send_line(C->sock,auth)) return 0;

    // ack oder erste notify
    for (int i=0;i<50;i++){
        int n = recv_line_nonblocking(C->sock,line,sizeof line,100);
        if (n>0 && (strstr(line,"\"result\":true") || strstr(line,"\"mining.notify\""))) break;
        usleep(100000);
    }

    printf("Connected to %s:%d\nExtranonce1=%s, extranonce2_size=%u\n",
           C->host,C->port,C->extranonce1,C->extranonce2_size);
    return 1;
}

static int stratum_get_job(stratum_ctx_t *C, stratum_job_t *J){
    char line[16384];
    int n = recv_line_nonblocking(C->sock,line,sizeof line,10);
    if (n>0) {
        if (stratum_parse_notify(line,J)) {
            printf("New job: %s (clean=%d)\n", J->job_id, J->clean);
            return 1;
        }
    }
    return 0;
}

static int stratum_submit(stratum_ctx_t *C,const stratum_job_t *J,
                          const char *extranonce2_hex,uint32_t ntime_le,uint32_t nonce_le){
    char ntime_hex[9], nonce_hex[9];
    snprintf(ntime_hex,9,"%08x",ntime_le);
    snprintf(nonce_hex,9,"%08x",nonce_le);
    char req[512];
    snprintf(req,sizeof req,
      "{\"id\":4,\"method\":\"mining.submit\",\"params\":[\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"]}\n",
      C->wallet, J->job_id, extranonce2_hex, ntime_hex, nonce_hex);
    return send_line(C->sock,req);
}

// merkle (LE) aus coinbase-hash (BE) + branches (hex BE)
static void build_merkle_root_le(const uint8_t cb_be[32], char merkle_hex[][65], int mcount, uint8_t out_le[32]) {
    uint8_t h_le[32]; for (int i=0;i<32;i++) h_le[i]=cb_be[31-i];
    for (int i=0;i<mcount;i++){
        uint8_t br_be[32], br_le[32], cat[64], dh[32];
        if(!hex2bin(merkle_hex[i], br_be, 32)) memset(br_be,0,32);
        for(int k=0;k<32;k++) br_le[k]=br_be[31-k];
        memcpy(cat,h_le,32); memcpy(cat+32,br_le,32);
        double_sha256(cat,64,dh);
        for(int k=0;k<32;k++) h_le[k]=dh[31-k];
    }
    memcpy(out_le,h_le,32);
}

static void build_header_le(const stratum_job_t *J, const uint8_t prevhash_le[32],
                            const uint8_t merkleroot_le[32], uint32_t ntime, uint32_t nbits,
                            uint32_t nonce, uint8_t out80[80]){
    memset(out80,0,80);
    memcpy(out80+0,&J->version,4);
    memcpy(out80+4,prevhash_le,32);
    memcpy(out80+36,merkleroot_le,32);
    memcpy(out80+68,&ntime,4);
    memcpy(out80+72,&nbits,4);
    memcpy(out80+76,&nonce,4);
}

// ============================ OpenCL =============================
static char* load_kernel_source(const char *path) {
    FILE *f=fopen(path,"rb"); if(!f){ perror("kernel"); exit(1); }
    fseek(f,0,SEEK_END); long sz=ftell(f); rewind(f);
    char *src=(char*)malloc(sz+1); fread(src,1,sz,f); src[sz]=0; fclose(f); return src;
}

// ============================== MAIN ============================
int main(int argc, char **argv) {
    // Speicher (MB) -> Blöcke
    uint32_t m_cost_kb = 64*1024;
    if (argc>1){ int mb=atoi(argv[1]); if(mb>=8 && mb<=256) m_cost_kb=mb*1024; }
    printf("Using %u MB memory\n", m_cost_kb/1024);

    // OpenCL Setup
    cl_int err; cl_platform_id plat; cl_device_id dev;
    clGetPlatformIDs(1,&plat,NULL);
    err = clGetDeviceIDs(plat,CL_DEVICE_TYPE_GPU,1,&dev,NULL);
    if (err!=CL_SUCCESS){ fprintf(stderr,"No GPU device found\n"); return 1; }

    char devname[256]; clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof devname,devname,NULL);
    printf("Device: %s\n", devname);

    cl_context ctx = clCreateContext(NULL,1,&dev,NULL,NULL,&err);
    cl_command_queue q = clCreateCommandQueue(ctx,dev,0,&err);
    if (err!=CL_SUCCESS){ fprintf(stderr,"clCreateCommandQueue: %d\n",err); return 1; }

    char *ksrc = load_kernel_source("rinhash_argon2d.cl");
    cl_program prog = clCreateProgramWithSource(ctx,1,(const char**)&ksrc,NULL,&err);
    err = clBuildProgram(prog,1,&dev,"-cl-std=CL1.2",NULL,NULL);
    if (err!=CL_SUCCESS){
        size_t L; clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,0,NULL,&L);
        char *log=(char*)malloc(L+1); clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,L,log,NULL);
        log[L]=0; fprintf(stderr,"Build failed:\n%s\n",log); free(log); return 1;
    }
    cl_kernel krn = clCreateKernel(prog,"argon2d_core",&err);

    size_t m_bytes = (size_t)m_cost_kb * 1024;
    cl_mem d_mem   = clCreateBuffer(ctx,CL_MEM_READ_WRITE,m_bytes,NULL,&err);
    cl_mem d_phash = clCreateBuffer(ctx,CL_MEM_READ_ONLY, 32,NULL,&err);
    cl_mem d_out   = clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,32,NULL,&err);

    // Kernel-arg „fixe“ Slots binden (wir setzen die variablen später pro Aufruf)
    // arg0: prehash, arg1: mem, arg2: blocks, arg3: pass, arg4: start, arg5: end, arg6: out, arg7: do_init
    clSetKernelArg(krn,0,sizeof(d_phash), &d_phash);
    clSetKernelArg(krn,1,sizeof(d_mem),   &d_mem);
    clSetKernelArg(krn,2,sizeof(uint32_t),&m_cost_kb);
    clSetKernelArg(krn,6,sizeof(d_out),   &d_out);

    // Stratum
    stratum_ctx_t S; if(!stratum_connect(&S,POOL,PORT,WAL,PASS)){ fprintf(stderr,"Stratum connect failed\n"); return 1; }

    uint8_t prevhash_le[32], merkleroot_le[32], target_be[32];
    stratum_job_t J={0}, Jnew={0}; int have_job=0;

    // Mining Loop
    time_t last_stat=time(NULL); uint64_t hashes=0;

   while (1) {
    // === Job holen (non-blocking) ===
    if (stratum_get_job(&S, &Jnew)) {
        J = Jnew; have_job = 1;
        uint8_t prev_be[32]; hex2bin(J.prevhash_hex, prev_be, 32);
        for (int i = 0; i < 32; i++) prevhash_le[i] = prev_be[31 - i];
        target_from_nbits(J.nbits, target_be);
        printf("Job %s ready. nbits=%08x ntime=%08x\n", J.job_id, J.nbits, J.ntime);
    }
    if (!have_job) { usleep(100000); continue; }

    // === Coinbase vorbereiten (ohne en2) ===
    uint8_t coinb1[2048], coinb2[2048];
    size_t cb1 = strlen(J.coinb1_hex) / 2, cb2 = strlen(J.coinb2_hex) / 2;
    hex2bin(J.coinb1_hex, coinb1, cb1);
    hex2bin(J.coinb2_hex, coinb2, cb2);
    uint8_t en1[64]; size_t en1b = strlen(S.extranonce1) / 2; if (en1b > 64) en1b = 64;
    hex2bin(S.extranonce1, en1, en1b);

    // === extranonce2 bauen ===
    static uint32_t en2_counter = 1;
    char en2_hex[64];
    snprintf(en2_hex, sizeof en2_hex, "%0*x", S.extranonce2_size * 2, en2_counter++);
    uint8_t en2[64]; hex2bin(en2_hex, en2, S.extranonce2_size);

    // === Coinbase = coinb1 + en1 + en2 + coinb2 ===
    uint8_t coinbase[4096]; size_t off = 0;
    memcpy(coinbase + off, coinb1, cb1); off += cb1;
    memcpy(coinbase + off, en1, en1b);   off += en1b;
    memcpy(coinbase + off, en2, S.extranonce2_size); off += S.extranonce2_size;
    memcpy(coinbase + off, coinb2, cb2); off += cb2;

    // === Coinbase Hash (double-SHA256, BE) ===
    uint8_t cbh_be[32]; double_sha256(coinbase, off, cbh_be);

    // === Merkle-Root in LE ===
    build_merkle_root_le(cbh_be, J.merkle_hex, J.merkle_count, merkleroot_le);

    // === Nonce-Range (klein halten wegen Mali-Watchdog) ===
    const uint32_t NONCES_PER_ITER = 20000;
    for (uint32_t nonce = 0; nonce < NONCES_PER_ITER; nonce++) {
        // --- Header bauen ---
        uint8_t header[80];
        build_header_le(&J, prevhash_le, merkleroot_le, J.ntime, J.nbits, nonce, header);

        // --- Prehash (BLAKE3) ---
        uint8_t prehash[32]; blake3_hash32(header, 80, prehash);
        clEnqueueWriteBuffer(q, d_phash, CL_TRUE, 0, 32, prehash, 0, NULL, NULL);

        // --- Argon2d: Passes + Slices + Chunks ---
        size_t G = 1;
        for (uint32_t pass = 0; pass < T_COST; pass++) {
            for (uint32_t slice = 0; slice < 4; slice++) {
                for (uint32_t start = slice * (m_cost_kb / 4);
                     start < (slice + 1) * (m_cost_kb / 4);
                     start += CHUNK_BLOCKS)
                {
                    uint32_t end = start + CHUNK_BLOCKS;
                    if (end > (slice + 1) * (m_cost_kb / 4))
                        end = (slice + 1) * (m_cost_kb / 4);

                    uint32_t do_init = (pass == 0 && slice == 0 && start == 0) ? 1U : 0U;

                    clSetKernelArg(krn, 3, sizeof(uint32_t), &pass);
                    clSetKernelArg(krn, 4, sizeof(uint32_t), &slice);
                    clSetKernelArg(krn, 5, sizeof(uint32_t), &start);
                    clSetKernelArg(krn, 6, sizeof(uint32_t), &end);
                    clSetKernelArg(krn, 8, sizeof(uint32_t), &do_init);

                    err = clEnqueueNDRangeKernel(q, krn, 1, NULL, &G, NULL, 0, NULL, NULL);
                    if (err != CL_SUCCESS) { fprintf(stderr, "Kernel failed: %d\n", err); break; }
                    clFlush(q);
                }
            }
        }
        clFinish(q);

        // --- Ergebnis 32B holen und finalisieren ---
        uint8_t argon_out[32]; clEnqueueReadBuffer(q, d_out, CL_TRUE, 0, 32, argon_out, 0, NULL, NULL);
        uint8_t final_hash[32]; sha3_256(argon_out, 32, final_hash);

        // --- Target-Vergleich (BE) ---
        int ok = 1;
        for (int i = 0; i < 32; i++) {
            if (final_hash[i] < target_be[i]) { ok = 1; break; }
            if (final_hash[i] > target_be[i]) { ok = 0; break; }
        }
        if (ok) {
            if (stratum_submit(&S, &J, en2_hex, J.ntime, nonce)) {
                printf("\nFOUND share  ntime=%08x nonce=%08x\n", J.ntime, nonce);
            } else {
                fprintf(stderr, "\nSubmit failed\n");
            }
        }

        hashes++;

                // --- Stats & neue Jobs ---
        if ((nonce % 1000) == 0) {
            stratum_job_t tmp;
            if (stratum_get_job(&S, &tmp)) {
                if (strcmp(tmp.job_id, J.job_id) != 0 || tmp.clean) {
                    J = tmp;
                    uint8_t prev_be2[32]; hex2bin(J.prevhash_hex, prev_be2, 32);
                    for (int i = 0; i < 32; i++) prevhash_le[i] = prev_be2[31 - i];
                    target_from_nbits(J.nbits, target_be);
                    printf("\nSwitch to job %s\n", J.job_id);
                    break;
                }
            }
            time_t now = time(NULL);
            if (now - last_stat >= 10) {
                double rate = (double)hashes / (double)(now - last_stat);
                printf("Hashrate: %.1f H/s | Job: %s\r", rate, J.job_id);
                fflush(stdout);
                last_stat = now; hashes = 0;
            }
        }
    }

    // === Cleanup (normal nicht erreicht) ===
    clReleaseMemObject(d_mem);
    clReleaseMemObject(d_phash);
    clReleaseMemObject(d_out);
    clReleaseKernel(krn);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    free(ksrc);
    close(S.sock);

    return 0;
}
