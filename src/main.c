// src/main.c – RinHash GPU Miner (Batch-GPU Mode) – FULL FILE

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <stdint.h>
#include <pthread.h>
#include <sys/select.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include "blake3.h"
#include "argon2.h"

// ============================ Konfig =============================
static const char *POOL_CANDIDATES[] = {
    "rinhash.eu.mine.zergpool.com",
    "rinhash.mine.zergpool.com",
    "rinhash.na.mine.zergpool.com",
    "rinhash.asia.mine.zergpool.com",
    NULL
};
static const int   PORT_DEFAULT = 7148;
static const char *WAL  = "DTXoRQ7Zpw3FVRW2DWkLrM9Skj4Gp9SeSj";
static const char *PASS = "c=DOGE,ID=n2plus";

// RinHash-Parameter (Spec)
#define CHUNK_BLOCKS 256
static const cl_uint T_COST    = 2;   // Argon2d-Pässe
static const cl_uint M_COST_KB = 64;  // 64 KiB
static const cl_uint ARGON_VER = 0x13;

// Batch-Konfiguration
static cl_uint g_batch_size = 256; // Default 256, via RIN_BATCH steuerbar
static cl_uint g_wgs = 0;          // 0 = Treiber wählt, via RIN_WGS steuerbar

// Debug / Optionen (per Env)
static int g_debug = 0;
static int g_use_batch = 1;  // Batch-Kernel verwenden
static int g_did_debug_for_job = 0; // Debug nur einmal pro Job

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
static void hexdump(const char* name, const uint8_t* b, size_t n){
    printf("%s=", name);
    for (size_t i=0;i<n;i++) printf("%02x", b[i]);
    printf("\n");
}

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
    cl_uint extranonce2_size;
} stratum_ctx_t;

typedef struct {
    char     job_id[128];
    char     prevhash_hex[65];      // BE hex
    char     coinb1_hex[4096];
    char     coinb2_hex[4096];
    char     merkle_hex[16][65];
    int      merkle_count;
    uint32_t version;
    uint32_t nbits;
    uint32_t ntime;
    int      clean;
} stratum_job_t;

static int set_nonblock(int s, int on) {
    int flags = fcntl(s, F_GETFL, 0);
    if (flags < 0) return -1;
    if (on) flags |= O_NONBLOCK; else flags &= ~O_NONBLOCK;
    return fcntl(s, F_SETFL, flags);
}
static int is_numeric_ip(const char *s) {
    struct in6_addr a6; struct in_addr a4;
    return inet_pton(AF_INET, s, &a4) == 1 || inet_pton(AF_INET6, s, &a6) == 1;
}
static int sock_connect_verbose(const char *host, int port, int timeout_ms) {
    if (is_numeric_ip(host)) {
        int s = socket(AF_INET, SOCK_STREAM, 0);
        if (s < 0) { perror("socket"); return -1; }
        struct sockaddr_in sa; memset(&sa,0,sizeof sa);
        sa.sin_family = AF_INET; sa.sin_port = htons((uint16_t)port);
        if (inet_pton(AF_INET, host, &sa.sin_addr) != 1) { close(s); return -1; }
        if (set_nonblock(s, 1) != 0) { perror("fcntl"); close(s); return -1; }
        int rc = connect(s, (struct sockaddr*)&sa, sizeof sa);
        if (rc != 0 && errno != EINPROGRESS) { perror("connect"); close(s); return -1; }
        fd_set wfds; FD_ZERO(&wfds); FD_SET(s, &wfds);
        struct timeval tv = { .tv_sec = timeout_ms/1000, .tv_usec = (timeout_ms%1000)*1000 };
        rc = select(s+1, NULL, &wfds, NULL, &tv);
        if (rc <= 0) { if (rc==0) fprintf(stderr,"connect timeout %s\n", host); else perror("select"); close(s); return -1; }
        int soerr=0; socklen_t slen=sizeof soerr;
        if (getsockopt(s,SOL_SOCKET,SO_ERROR,&soerr,&slen)<0 || soerr!=0){ if(soerr) fprintf(stderr,"connect error: %s\n", strerror(soerr)); else perror("getsockopt"); close(s); return -1; }
        fprintf(stderr,"Connected to %s (numeric IP)\n", host);
        return s;
    }
    struct addrinfo hints = {0}, *res=NULL, *p=NULL;
    char portstr[16];
    snprintf(portstr, sizeof portstr, "%d", port);
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family   = AF_UNSPEC;
    hints.ai_flags    = AI_NUMERICSERV | AI_ADDRCONFIG;

    int ga = getaddrinfo(host, portstr, &hints, &res);
    if (ga != 0 || !res) {
        fprintf(stderr, "DNS resolve failed for %s:%s - %s\n", host, portstr, gai_strerror(ga));
        return -1;
    }

    int s = -1;
    for (p = res; p; p = p->ai_next) {
        char addrbuf[128] = {0};
        void *aptr = NULL;
        if (p->ai_family == AF_INET)  aptr = &((struct sockaddr_in*)p->ai_addr)->sin_addr;
        if (p->ai_family == AF_INET6) aptr = &((struct sockaddr_in6*)p->ai_addr)->sin6_addr;
        if (aptr) inet_ntop(p->ai_family, aptr, addrbuf, sizeof addrbuf);

        s = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (s < 0) { perror("socket"); continue; }

        if (set_nonblock(s, 1) != 0) { perror("fcntl(O_NONBLOCK)"); close(s); s=-1; continue; }

        int rc = connect(s, p->ai_addr, p->ai_addrlen);
        if (rc != 0 && errno != EINPROGRESS) {
            fprintf(stderr, "connect() to %s (%s) failed: %s\n", host, addrbuf[0]?addrbuf:"?", strerror(errno));
            close(s); s=-1; continue;
        }
        fd_set wfds; FD_ZERO(&wfds); FD_SET(s, &wfds);
        struct timeval tv = { .tv_sec = timeout_ms/1000, .tv_usec = (timeout_ms%1000)*1000 };
        rc = select(s+1, NULL, &wfds, NULL, &tv);
        if (rc <= 0) {
            if (rc == 0) fprintf(stderr, "connect() timeout to %s (%s)\n", host, addrbuf[0]?addrbuf:"?");
            else perror("select(connect)");
            close(s); s = -1; continue;
        }
        int soerr=0; socklen_t slen=sizeof soerr;
        if (getsockopt(s, SOL_SOCKET, SO_ERROR, &soerr, &slen) < 0 || soerr != 0) {
            fprintf(stderr, "connect() error after select to %s (%s): %s\n", host, addrbuf[0]?addrbuf:"?", (soerr?strerror(soerr):"getsockopt failed"));
            close(s); s = -1; continue;
        }
        fprintf(stderr, "Connected to %s (%s)\n", host, addrbuf[0]?addrbuf:"?");
        break;
    }

    freeaddrinfo(res);
    if (s < 0) {
        fprintf(stderr, "All address candidates for %s:%d failed.\n", host, port);
        return -1;
    }
    return s;
}

static int send_line_verbose(int s, const char *line) {
    size_t L = strlen(line), o = 0;
    while (o < L) {
        ssize_t n = send(s, line + o, L - o, 0);
        if (n <= 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) { usleep(1000); continue; }
            fprintf(stderr, "send() failed: %s\n", strerror(errno));
            return 0;
        }
        o += (size_t)n;
    }
    return 1;
}

// -------- robuster Zeilenpuffer --------
static char inbuf[65536];
static size_t inlen = 0;

static int recv_into_buffer(int s, int timeout_ms) {
    fd_set rfds; struct timeval tv;
    FD_ZERO(&rfds); FD_SET(s, &rfds);
    tv.tv_sec = timeout_ms/1000; tv.tv_usec = (timeout_ms%1000)*1000;
    int sel = select(s+1,&rfds,NULL,NULL,&tv);
    if (sel <= 0) return 0;
    char tmp[8192];
    ssize_t n = recv(s,tmp,sizeof tmp,MSG_DONTWAIT);
    if (n <= 0) return 0;
    if (inlen + (size_t)n >= sizeof inbuf) inlen = 0; // Reset bei Überlauf
    memcpy(inbuf + inlen, tmp, (size_t)n); inlen += (size_t)n;
    return 1;
}
static int next_line(char *out, size_t cap) {
    for (size_t i=0;i<inlen;i++) if (inbuf[i]=='\n') {
        size_t L = i+1; if (L >= cap) L = cap-1;
        memcpy(out, inbuf, L); out[L]=0;
        memmove(inbuf, inbuf+i+1, inlen-(i+1));
        inlen -= (i+1);
        return 1;
    }
    return 0;
}

// --- Subscribe-Parsing (extranonce sauber) ---
static const char* find_after_result_array_first(const char *s) {
    const char *p = strstr(s, "\"result\""); if (!p) return NULL;
    p = strchr(p, '['); if (!p) return NULL;
    int depth = 0; int first_closed = 0;
    for (; *p; p++){
        if (*p == '[') depth++;
        else if (*p == ']') {
            depth--;
            if (depth == 1 && !first_closed) { first_closed = 1; return p+1; }
        }
    }
    return NULL;
}
static int parse_subscribe_result(const char *line, char *ex1, size_t ex1cap, cl_uint *ex2sz){
    const char *after_first = find_after_result_array_first(line);
    if (!after_first) return 0;
    const char *q1 = strchr(after_first, '"'); if(!q1) return 0;
    const char *q2 = strchr(q1+1, '"');       if(!q2) return 0;
    size_t L = (size_t)(q2 - (q1+1)); if (L >= ex1cap) L = ex1cap-1;
    memcpy(ex1, q1+1, L); ex1[L] = 0;

    const char *p = q2+1;
    while (*p && (*p==' ' || *p==',' || *p==']')) p++;
    unsigned long v = strtoul(p, NULL, 10);
    if (v == 0 || v > 32) v = 4;
    *ex2sz = (cl_uint)v;
    return 1;
}

static int stratum_connect_one(stratum_ctx_t *C,const char *host,int port,const char *user,const char *pass){
    memset(C,0,sizeof *C);
    snprintf(C->host,sizeof C->host,"%s",host); C->port=port;
    snprintf(C->wallet,sizeof C->wallet,"%s",user);
    snprintf(C->pass,sizeof C->pass,"%s",pass);

    C->sock = sock_connect_verbose(host, port, 5000);
    if (C->sock < 0) return 0;

    char sub[256];
    snprintf(sub,sizeof sub,"{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"rin-ocl/0.9\"]}\n");
    if(!send_line_verbose(C->sock,sub)) { fprintf(stderr,"Stratum send subscribe failed\n"); close(C->sock); return 0; }

    time_t t0=time(NULL); char line[16384];
    int have_ex = 0;
    while (time(NULL)-t0 < 5) {
        recv_into_buffer(C->sock, 500);
        while (next_line(line,sizeof line)) {
            if (strstr(line,"\"result\"") && !strstr(line,"\"method\"")) {
                if (parse_subscribe_result(line, C->extranonce1, sizeof C->extranonce1, &C->extranonce2_size)) {
                    have_ex = 1; break;
                }
            }
        }
        if (have_ex) break;
    }
    if (!have_ex) {
        fprintf(stderr, "No subscribe result within timeout (5s)\n");
        close(C->sock); return 0;
    }
    if (!C->extranonce1[0]) snprintf(C->extranonce1,sizeof C->extranonce1,"00000000");
    if (!C->extranonce2_size) C->extranonce2_size = 4;

    char auth[512];
    snprintf(auth,sizeof auth,"{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\"%s\",\"%s\"]}\n",C->wallet,C->pass);
    if(!send_line_verbose(C->sock,auth)) { fprintf(stderr,"Stratum send authorize failed\n"); close(C->sock); return 0; }

    t0=time(NULL);
    while (time(NULL)-t0 < 2) { recv_into_buffer(C->sock, 200); while(next_line(line,sizeof line)){} }

    printf("Connected to %s:%d\nExtranonce1=%s, extranonce2_size=%u\n",
           C->host,C->port,C->extranonce1,(unsigned)C->extranonce2_size);
    return 1;
}

static int stratum_connect_any(stratum_ctx_t *C, const char **hosts, int port, const char *user, const char *pass){
    const char *env_host = getenv("POOL_HOST");
    const char *env_port = getenv("POOL_PORT");
    if (env_host && env_host[0]) {
        int p = (env_port && env_port[0]) ? atoi(env_port) : port;
        fprintf(stderr, "Trying POOL_HOST from ENV: %s:%d\n", env_host, p);
        if (stratum_connect_one(C, env_host, p, user, pass)) return 1;
        fprintf(stderr, "ENV host failed, falling back to default list…\n");
    }
    for (int i = 0; hosts[i]; i++) {
        fprintf(stderr, "Trying %s:%d …\n", hosts[i], port);
        if (stratum_connect_one(C, hosts[i], port, user, pass)) return 1;
    }
    return 0;
}

static int get_next_quoted(const char *pp, const char *end, char *out, size_t cap) {
    const char *p = pp, *q1=NULL,*q2=NULL;
    for (; p < end; p++){ if(*p=='"'){ q1=p; break; } }
    if (!q1) return 0;
    for (p=q1+1; p<end; p++){ if(*p=='"'){ q2=p; break; } }
    if (!q2) return 0;
    size_t L=(size_t)(q2-(q1+1)); if (L>=cap) L=cap-1;
    memcpy(out,q1+1,L); out[L]=0; return 1;
}

static int stratum_parse_notify(const char *line, stratum_job_t *J){
    if (!strstr(line,"\"mining.notify\"")) return 0;
    memset(J,0,sizeof *J);
    const char *pp = strstr(line,"\"params\""); if(!pp) return 0;
    const char *lb = strchr(pp,'['); const char *rb = lb?strrchr(pp,']'):NULL;
    if(!lb||!rb||rb<=lb) return 0;
    const char *p = lb+1;
    if(!get_next_quoted(p,rb,J->job_id,sizeof J->job_id)) return 0;
    if(!get_next_quoted(p,rb,J->prevhash_hex,sizeof J->prevhash_hex)) return 0;
    if(!get_next_quoted(p,rb,J->coinb1_hex,sizeof J->coinb1_hex)) return 0;
    if(!get_next_quoted(p,rb,J->coinb2_hex,sizeof J->coinb2_hex)) return 0;

    J->merkle_count = 0;
    const char *m_lb=strchr(p,'['), *m_rb=NULL;
    if (m_lb) { const char *scan=m_lb+1; for(;scan<rb;scan++){ if(*scan==']'){ m_rb=scan; break; }}}
    if (m_lb && m_rb && m_rb>m_lb) {
        const char *mp = m_lb+1;
        while (J->merkle_count < 16) {
            char tmp[65]; if(!get_next_quoted(mp,m_rb,tmp,sizeof tmp)) break;
            snprintf(J->merkle_hex[J->merkle_count],65,"%s",tmp);
            J->merkle_count++;
        }
        p = m_rb+1;
    }

    char vhex[16]={0}, nbhex[16]={0}, nth[16]={0};
    if(!get_next_quoted(p,rb,vhex,sizeof vhex)) return 0; sscanf(vhex,"%x",&J->version);
    if(!get_next_quoted(p,rb,nbhex,sizeof nbhex)) return 0; sscanf(nbhex,"%x",&J->nbits);
    if(!get_next_quoted(p,rb,nth,sizeof nth))    return 0; sscanf(nth,"%x",&J->ntime);

    J->clean = strstr(p,"true")!=NULL;
    return 1;
}

static int stratum_get_job(stratum_ctx_t *C, stratum_job_t *J){
    char line[16384]; int got=0;
    recv_into_buffer(C->sock, 0);
    while (next_line(line,sizeof line)) {
        if (strstr(line,"\"mining.notify\"")) {
            if (stratum_parse_notify(line,J)) { got=1; break; }
        }
    }
    return got;
}

static void stratum_wait_submit_ack(int sock) {
    char line[4096];
    uint64_t start = 0;
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    start = (uint64_t)ts.tv_sec*1000ull + ts.tv_nsec/1000000ull;

    while (1) {
        recv_into_buffer(sock, 0);
        while (next_line(line, sizeof line)) {
            if (strstr(line, "\"id\":4")) {
                if (strstr(line, "\"result\":true")) {
                    printf(" -> ACCEPTED\n");
                    fflush(stdout);
                    return;
                } else if (strstr(line, "\"error\"")) {
                    printf(" -> REJECTED: %s", line);
                    fflush(stdout);
                    return;
                }
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &ts);
        uint64_t now = (uint64_t)ts.tv_sec*1000ull + ts.tv_nsec/1000000ull;
        if (now - start > 1200) return;
        usleep(20000);
    }
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
    int ok = send_line_verbose(C->sock,req);
    if (ok) stratum_wait_submit_ack(C->sock);
    return ok;
}

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
static char* load_kernel_source_try(const char *path) {
    FILE *f=fopen(path,"rb");
    if(!f) return NULL;
    if (fseek(f,0,SEEK_END)!=0){ fclose(f); return NULL; }
    long sz=ftell(f); if (sz<0){ fclose(f); return NULL; }
    rewind(f);
    char *src=(char*)malloc((size_t)sz+1);
    if(!src){ fclose(f); return NULL; }
    size_t nread = fread(src,1,(size_t)sz,f);
    fclose(f);
    if (nread != (size_t)sz) { free(src); return NULL; }
    src[sz]=0;
    return src;
}
static char* load_kernel_source(const char *filename) {
    char *s = load_kernel_source_try(filename);
    if (s) return s;
    s = load_kernel_source_try("src/rinhash_argon2d.cl");     if (s) return s;
    s = load_kernel_source_try("kernels/rinhash_argon2d.cl"); if (s) return s;
    fprintf(stderr,"kernel not found: %s (also tried src/ and kernels/)\n", filename);
    exit(1);
}

static inline uint64_t mono_ms(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec*1000ull + (uint64_t)ts.tv_nsec/1000000ull;
}

static void check_arg(const char* name, cl_int e, int idx){
    if (e != CL_SUCCESS){
        fprintf(stderr, "clSetKernelArg(%s, idx=%d) failed: %d\n", name, idx, e);
        exit(1);
    }
}

static volatile sig_atomic_t g_stop = 0;
static void on_sigint(int sig){ (void)sig; g_stop = 1; }

// ============================== MAIN ============================
int main(int argc, char **argv) {
   signal(SIGINT, on_sigint);

   // ENV-Variablen lesen
   if (getenv("RIN_DEBUG")) g_debug = atoi(getenv("RIN_DEBUG"));
   if (getenv("RIN_BATCH")) {
       g_batch_size = (cl_uint)atoi(getenv("RIN_BATCH"));
       if (g_batch_size < 16) g_batch_size = 16;
       if (g_batch_size > 1024) g_batch_size = 1024;
   }
   if (getenv("RIN_WGS")) {
       g_wgs = (cl_uint)atoi(getenv("RIN_WGS"));
   }

   printf("=== RinHash Batch-GPU Miner ===\n");
   printf("Batch size: %u (via RIN_BATCH, default 256)\n", g_batch_size);
   printf("Work group size: %s\n", g_wgs ? "custom" : "auto (driver chooses)");
   printf("Memory: %u KiB per instance (spec-compliant)\n", M_COST_KB);
   printf("Argon2d params: t=%u, m=%u KiB, p=1, salt=RinCoinSalt, ver=0x%02x\n",
          T_COST, M_COST_KB, ARGON_VER);

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

   // Versuche Batch-Kernel, sonst Fallback
   cl_kernel batch_kernel = clCreateKernel(prog,"argon2d_batch",&err);
   cl_kernel fallback_kernel = NULL;
   if (err != CL_SUCCESS) {
       printf("Batch kernel not found, using fallback single-nonce mode\n");
       g_use_batch = 0;
       fallback_kernel = clCreateKernel(prog,"argon2d_core",&err);
       if (err != CL_SUCCESS) {
           fprintf(stderr,"Neither batch nor core kernel found!\n");
           return 1;
       }
   } else {
       printf("Using batch kernel for parallel processing\n");
   }

   // Batch-Puffer anlegen
   uint8_t *prehash_batch = (uint8_t*)malloc(32 * g_batch_size);
   uint8_t *argon_out_batch = (uint8_t*)malloc(32 * g_batch_size);
   
   cl_mem d_prehash = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  32 * g_batch_size, NULL, &err);
   cl_mem d_mem     = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 65536ULL * g_batch_size, NULL, &err);
   cl_mem d_out     = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, 32 * g_batch_size, NULL, &err);

   // Fallback-Puffer (falls Batch nicht geht)
   cl_mem d_phash_single = NULL, d_mem_single = NULL, d_out_single = NULL;
   if (!g_use_batch) {
       d_phash_single = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 32, NULL, &err);
       d_mem_single   = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 65536, NULL, &err);
       d_out_single   = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, 32, NULL, &err);
   }

   // ===== Stratum =====
   int PORT = getenv("POOL_PORT") ? atoi(getenv("POOL_PORT")) : PORT_DEFAULT;
   stratum_ctx_t S;
   if(!stratum_connect_any(&S, POOL_CANDIDATES, PORT, WAL, PASS)){
       fprintf(stderr,"Stratum connect failed (all hosts)\n");
       return 1;
   }

   uint8_t prevhash_le[32], merkleroot_le[32], target_be[32];
   stratum_job_t J={0}, Jnew={0}; int have_job=0;

   uint64_t total_hashes = 0;
   uint64_t t_poll  = mono_ms();
   uint64_t t_stats = mono_ms();
   
   static cl_uint batch_nonce_base = 0;
   static cl_uint en2_counter = 1;

   while (!g_stop) {
       // Stratum-Polling (alle 100ms)
       if (mono_ms() - t_poll >= 100) {
           while (stratum_get_job(&S, &Jnew)) {
               if (!have_job || strcmp(Jnew.job_id, J.job_id) != 0 || Jnew.clean) {
                   J = Jnew; have_job = 1;
                   g_did_debug_for_job = 0;  // Reset debug flag für neuen Job
                   batch_nonce_base = 0;     // Nonce-Zähler zurücksetzen
                   
                   uint8_t prev_be[32]; hex2bin(J.prevhash_hex, prev_be, 32);
                   for (int i = 0; i < 32; i++) prevhash_le[i] = prev_be[31 - i];
                   target_from_nbits(J.nbits, target_be);
                   printf("\n=== New Job: %s ===\n", J.job_id);
                   printf("nbits=%08x ntime=%08x\n", J.nbits, J.ntime);
                   fflush(stdout);
               }
           }
           t_poll = mono_ms();
       }
       if (!have_job) { usleep(10000); continue; }

       // === Batch-Mining ===
       if (g_use_batch) {
           // Coinbase/Merkle für diesen Batch
           uint8_t coinb1[4096], coinb2[4096];
           size_t cb1 = strlen(J.coinb1_hex) / 2, cb2 = strlen(J.coinb2_hex) / 2;
           hex2bin(J.coinb1_hex, coinb1, cb1);
           hex2bin(J.coinb2_hex, coinb2, cb2);

           uint8_t en1[64]; size_t en1b = strlen(S.extranonce1) / 2; 
           if (en1b > 64) en1b = 64;
           hex2bin(S.extranonce1, en1, en1b);
           
           // Extranonce2 für Batch
           char en2_hex[64];
           snprintf(en2_hex, sizeof en2_hex, "%0*x", S.extranonce2_size * 2, en2_counter++);
           uint8_t en2[64]; hex2bin(en2_hex, en2, S.extranonce2_size);

           uint8_t coinbase[8192]; size_t off = 0;
           memcpy(coinbase + off, coinb1, cb1); off += cb1;
           memcpy(coinbase + off, en1, en1b); off += en1b;
           memcpy(coinbase + off, en2, S.extranonce2_size); off += S.extranonce2_size;
           memcpy(coinbase + off, coinb2, cb2); off += cb2;

           uint8_t cbh_be[32]; double_sha256(coinbase, off, cbh_be);
           build_merkle_root_le(cbh_be, J.merkle_hex, J.merkle_count, merkleroot_le);

           // Batch mit fortlaufenden Nonces füllen
           for (cl_uint i = 0; i < g_batch_size; i++) {
               uint8_t header[80];
               uint32_t nonce = batch_nonce_base + i;
               build_header_le(&J, prevhash_le, merkleroot_le, J.ntime, J.nbits, nonce, header);
               blake3_hash32(header, 80, prehash_batch + i*32);
           }

           // Async Transfer zur GPU
           cl_event ev_w;
           clEnqueueWriteBuffer(q, d_prehash, CL_FALSE, 0, 32*g_batch_size, 
                              prehash_batch, 0, NULL, &ev_w);

           // Kernel-Args setzen
           err = clSetKernelArg(batch_kernel, 0, sizeof(cl_mem), &d_prehash);
           check_arg("prehashes", err, 0);
           err = clSetKernelArg(batch_kernel, 1, sizeof(cl_mem), &d_mem);
           check_arg("mem", err, 1);
           err = clSetKernelArg(batch_kernel, 2, sizeof(cl_mem), &d_out);
           check_arg("out32", err, 2);
           err = clSetKernelArg(batch_kernel, 3, sizeof(cl_uint), &g_batch_size);
           check_arg("num_items", err, 3);
           cl_uint blocks = M_COST_KB;  // 64
           err = clSetKernelArg(batch_kernel, 4, sizeof(cl_uint), &blocks);
           check_arg("blocks_per_lane", err, 4);
           err = clSetKernelArg(batch_kernel, 5, sizeof(cl_uint), &T_COST);
           check_arg("t_cost", err, 5);

           // Kernel starten (warte auf Write-Event)
           size_t global = g_batch_size;
           size_t local_size = g_wgs;
           size_t *local = (g_wgs > 0) ? &local_size : NULL;
           
           uint64_t t_kernel_start = mono_ms();
           err = clEnqueueNDRangeKernel(q, batch_kernel, 1, NULL, &global, local, 
                                       1, &ev_w, NULL);
           if (err != CL_SUCCESS) {
               fprintf(stderr, "Kernel launch failed: %d\n", err);
               exit(1);
           }
           clReleaseEvent(ev_w);
           clFinish(q);
           uint64_t t_kernel_end = mono_ms();

           // Ergebnisse lesen
           clEnqueueReadBuffer(q, d_out, CL_TRUE, 0, 32*g_batch_size, 
                             argon_out_batch, 0, NULL, NULL);

           // Debug beim ersten Batch des Jobs
           if (g_debug && !g_did_debug_for_job) {
               printf("[DEBUG] First batch of job %s:\n", J.job_id);
               hexdump("prehash[0]", prehash_batch, 32);
               hexdump("gpu_argon2[0]", argon_out_batch, 32);
               g_did_debug_for_job = 1;
           }

           // Ergebnisse prüfen
           for (cl_uint i = 0; i < g_batch_size; i++) {
               uint8_t final_hash[32];
               sha3_256(argon_out_batch + i*32, 32, final_hash);
               
               // Target-Vergleich (BE)
               int ok = 1;
               for (int j = 0; j < 32; j++) {
                   if (final_hash[j] < target_be[j]) { ok = 1; break; }
                   if (final_hash[j] > target_be[j]) { ok = 0; break; }
               }
               
               if (ok) {
                   uint32_t nonce = batch_nonce_base + i;
                   printf("\nFOUND share! ntime=%08x nonce=%08x", J.ntime, nonce);
                   fflush(stdout);
                   stratum_submit(&S, &J, en2_hex, J.ntime, nonce);
               }
           }

           batch_nonce_base += g_batch_size;
           total_hashes += g_batch_size;

           // Statistik
           if (mono_ms() - t_stats >= 5000) {
               double secs = (mono_ms() - t_stats) / 1000.0;
               double rate = total_hashes / secs;
               double kernel_ms = t_kernel_end - t_kernel_start;
               printf("\rHashrate: %.1f H/s | Batch time: %.1fms | Job: %s        ", 
                      rate, kernel_ms, J.job_id);
               fflush(stdout);
               t_stats = mono_ms();
               total_hashes = 0;
           }

       } else {
           // === Fallback: Single-Nonce Loop (alter Code) ===
           fprintf(stderr, "Fallback mode not fully implemented\n");
           break;
       }

       usleep(1000); // Kurze Pause zwischen Batches
   }

   // Cleanup
   if (batch_kernel) clReleaseKernel(batch_kernel);
   if (fallback_kernel) clReleaseKernel(fallback_kernel);
   clReleaseProgram(prog);
   clReleaseMemObject(d_prehash);
   clReleaseMemObject(d_mem);
   clReleaseMemObject(d_out);
   if (d_phash_single) clReleaseMemObject(d_phash_single);
   if (d_mem_single) clReleaseMemObject(d_mem_single);
   if (d_out_single) clReleaseMemObject(d_out_single);
   clReleaseCommandQueue(q);
   clReleaseContext(ctx);
   close(S.sock);
   free(ksrc);
   free(prehash_batch);
   free(argon_out_batch);
   
   printf("\nShutdown complete.\n");
   return 0;
}


