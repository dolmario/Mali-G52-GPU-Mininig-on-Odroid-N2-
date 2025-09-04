// src/main.c — RinHash GPU Miner (Hybrid: CPU init + GPU fill) — FULL FILE

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

// Debug / Optionen (per Env)
static int g_debug = 0;
static int g_cpu_init = 1; // Hybrid-Mode standardmäßig AN

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

// ======================= BLAKE2b (CPU, für H0/H′) =======================
typedef unsigned long long u64;
typedef unsigned int       u32;
typedef unsigned char      u8;

static inline u64 rotr64(u64 x, u32 n){ return (x >> n) | (x << (64 - n)); }

static const u64 B2B_IV[8] = {
  0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
  0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
  0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
  0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};
static const u32 B2B_SIGMA[12][16] = {
 { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
 {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3},
 {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4},
 { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8},
 { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13},
 { 2,12, 6,10, 4, 7,15,14, 1,13, 3, 9, 8, 5,11, 0},
 {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11},
 {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10},
 { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7,10, 4, 1, 5},
 {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0},
 { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
 {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3}
};

static inline void blake2b_init(u64 h[8], u32 outlen){
    for (int i=0;i<8;i++) h[i]=B2B_IV[i];
    h[0] ^= 0x01010000ULL ^ (u64)outlen;
}

#define GG(a,b,c,d,x,y) \
    a = a + b + x; d = rotr64(d ^ a, 32); \
    c = c + d;     b = rotr64(b ^ c, 24); \
    a = a + b + y; d = rotr64(d ^ a, 16); \
    c = c + d;     b = rotr64(b ^ c, 63);

static inline void blake2b_compress(u64 h[8], const u8 block[128], u64 t0, u64 t1, u64 f0){
    u64 m[16];
    for (int i=0;i<16;i++){
        int o=i*8;
        m[i] = ((u64)block[o]) | ((u64)block[o+1]<<8) | ((u64)block[o+2]<<16) | ((u64)block[o+3]<<24) |
               ((u64)block[o+4]<<32)| ((u64)block[o+5]<<40)| ((u64)block[o+6]<<48)| ((u64)block[o+7]<<56);
    }
    u64 v[16];
    for (int i=0;i<8;i++) v[i]=h[i];
    for (int i=0;i<8;i++) v[i+8]=B2B_IV[i];
    v[12] ^= t0; v[13] ^= t1;
    v[14] ^= f0;

    for (int r=0;r<12;r++){
        const u32* s = &B2B_SIGMA[r][0];
        GG(v[0],v[4],v[8], v[12], m[s[0]], m[s[1]]);
        GG(v[1],v[5],v[9], v[13], m[s[2]], m[s[3]]);
        GG(v[2],v[6],v[10],v[14], m[s[4]], m[s[5]]);
        GG(v[3],v[7],v[11],v[15], m[s[6]], m[s[7]]);
        GG(v[0],v[5],v[10],v[15], m[s[8]], m[s[9]]);
        GG(v[1],v[6],v[11],v[12], m[s[10]],m[s[11]]);
        GG(v[2],v[7],v[8], v[13], m[s[12]],m[s[13]]);
        GG(v[3],v[4],v[9], v[14], m[s[14]],m[s[15]]);
    }
    for (int i=0;i<8;i++) h[i] ^= v[i] ^ v[i+8];
}
#undef GG

// Hash für <=128B Input, out<=64
static inline void blake2b_hash(u8* out, u32 outlen, const u8* in, u32 inlen){
    u64 h[8]; blake2b_init(h, outlen);
    u8 block[128]={0};
    for (u32 i=0;i<inlen && i<128;i++) block[i]=in[i];
    u64 t0 = (u64)inlen, t1 = 0, f0 = ~((u64)0);
    blake2b_compress(h, block, t0, t1, f0);
    for (int i=0;i<8;i++){
        u64 v=h[i];
        int o=i*8;
        out[o+0]=(u8)(v); out[o+1]=(u8)(v>>8); out[o+2]=(u8)(v>>16); out[o+3]=(u8)(v>>24);
        out[o+4]=(u8)(v>>32); out[o+5]=(u8)(v>>40); out[o+6]=(u8)(v>>48); out[o+7]=(u8)(v>>56);
    }
    // outlen <=64, out ist bereits befüllt, ggf. abgeschnitten
}

// H' (blake2b_long)
static inline void st32p(u8* p, u32 v){ p[0]=(u8)v; p[1]=(u8)(v>>8); p[2]=(u8)(v>>16); p[3]=(u8)(v>>24); }

static inline void blake2b_long(u8* out, u32 outlen, const u8* in, u32 inlen){
    u8 tmp[64];
    u8 buf[4 + 256]; // 4B τ + in (klein bei uns)
    st32p(buf, outlen);
    for (u32 i=0;i<inlen;i++) buf[4+i]=in[i];

    if (outlen <= 64){
        blake2b_hash(out, outlen, buf, 4+inlen);
        return;
    }
    blake2b_hash(tmp, 64, buf, 4+inlen);
    u32 written = 0, remain = outlen;
    for (;;){
        u32 take = (remain > 32) ? 32 : remain;
        for (u32 i=0;i<take;i++) out[written+i]=tmp[i];
        written += take; remain -= take;
        if (!remain) break;
        blake2b_hash(tmp, 64, tmp, 64);
    }
}

// Argon2d H0 + erste zwei Blöcke (p=1, t=2, m=64 KiB, ver=0x13, salt="RinCoinSalt")
static void argon2d_cpu_first_blocks(const uint8_t pwd32[32], uint32_t blocks_per_lane,
                                     uint8_t B0[1024], uint8_t B1[1024]){
    const char *salt = "RinCoinSalt";
    const u32 saltlen = 11;
    u8 param[4*10 + 32 + 11]; // lanes, tau, m, t, version, type, |P|, P, |S|, S, |K|, |X|
    u32 off=0;
    st32p(param+off, 1);            off+=4;              // lanes
    st32p(param+off, 32);           off+=4;              // tau
    st32p(param+off, blocks_per_lane); off+=4;           // m (KB == blocks)
    st32p(param+off, T_COST);       off+=4;              // t
    st32p(param+off, ARGON_VER);    off+=4;              // version
    st32p(param+off, 0);            off+=4;              // type (0 = d)
    st32p(param+off, 32);           off+=4;              // |P|
    memcpy(param+off, pwd32, 32);   off+=32;             // P
    st32p(param+off, saltlen);      off+=4;              // |S|
    memcpy(param+off, salt, saltlen); off+=saltlen;      // S
    st32p(param+off, 0);            off+=4;              // |K|=0
    st32p(param+off, 0);            off+=4;              // |X|=0

    u8 H0[64];
    blake2b_hash(H0, 64, param, off);

    u8 inX[64+4+4];
    memcpy(inX, H0, 64);
    st32p(inX+64, 0); st32p(inX+68, 0); // idx=0, lane=0
    blake2b_long(B0, 1024, inX, 72);
    st32p(inX+64, 1); st32p(inX+68, 0); // idx=1, lane=0
    blake2b_long(B1, 1024, inX, 72);
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

   // ENV
   if (getenv("RIN_DEBUG"))    g_debug = atoi(getenv("RIN_DEBUG"));
   if (getenv("RIN_CPU_INIT")) g_cpu_init = atoi(getenv("RIN_CPU_INIT"));

   cl_uint m_cost_kb = M_COST_KB;
   if (argc > 1) printf("WARNING: Memory arg ignored. Using spec-compliant 64 KiB\n");
   printf("Using %u KiB memory (RinHash spec)\n", (unsigned)m_cost_kb);
   printf("CPU Argon2d params: t_cost=%u, m_cost=%u KiB, lanes=1, salt_mode=RinCoinSalt, ver=0x%02x\n",
          T_COST, m_cost_kb, ARGON_VER);

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
   if (err!=CL_SUCCESS){ fprintf(stderr,"clCreateKernel failed: %d\n", err); return 1; }

   // Speicher: lane memory + prehash32 + out32
   size_t m_bytes = (size_t)m_cost_kb * 1024;
   cl_mem d_mem   = clCreateBuffer(ctx,CL_MEM_READ_WRITE,m_bytes,NULL,&err);
   cl_mem d_phash = clCreateBuffer(ctx,CL_MEM_READ_ONLY, 32,NULL,&err);
   cl_mem d_out   = clCreateBuffer(ctx,CL_MEM_READ_WRITE,32,NULL,&err);

   // ===== Stratum =====
   int         PORT = getenv("POOL_PORT") ? atoi(getenv("POOL_PORT")) : PORT_DEFAULT;
   stratum_ctx_t S;
   if(!stratum_connect_any(&S, POOL_CANDIDATES, PORT, WAL, PASS)){
       fprintf(stderr,"Stratum connect failed (all hosts)\n");
       return 1;
   }

   uint8_t prevhash_le[32], merkleroot_le[32], target_be[32];
   stratum_job_t J={0}, Jnew={0}; int have_job=0;

   uint64_t hashes_window = 0;
   uint64_t t_poll  = mono_ms();
   uint64_t t_print = mono_ms();

   while (!g_stop) {
       // Polling
       if (mono_ms() - t_poll >= 100) {
           while (stratum_get_job(&S, &Jnew)) {
               if (!have_job || strcmp(Jnew.job_id, J.job_id) != 0 || Jnew.clean) {
                   J = Jnew; have_job = 1;
                   uint8_t prev_be[32]; hex2bin(J.prevhash_hex, prev_be, 32);
                   for (int i = 0; i < 32; i++) prevhash_le[i] = prev_be[31 - i];
                   target_from_nbits(J.nbits, target_be);
                   printf("Job %s ready. nbits=%08x ntime=%08x\n", J.job_id, J.nbits, J.ntime);
                   fflush(stdout);
               }
           }
           t_poll = mono_ms();
       }
       if (!have_job) { usleep(10000); continue; }

       // Coinbase, Merkle
       uint8_t coinb1[4096], coinb2[4096];
       size_t cb1 = strlen(J.coinb1_hex) / 2, cb2 = strlen(J.coinb2_hex) / 2;
       hex2bin(J.coinb1_hex, coinb1, cb1);
       hex2bin(J.coinb2_hex, coinb2, cb2);

       uint8_t en1[64]; size_t en1b = strlen(S.extranonce1) / 2; if (en1b > 64) en1b = 64;
       hex2bin(S.extranonce1, en1, en1b);
       static cl_uint en2_counter = 1;
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

       // === Nonce-Schleife ===
       const cl_uint NONCES_PER_ITER = 20000;

       for (cl_uint nonce = 0; nonce < NONCES_PER_ITER && !g_stop; nonce++) {
           uint8_t header[80];
           build_header_le(&J, prevhash_le, merkleroot_le, J.ntime, J.nbits, nonce, header);

           uint8_t prehash32[32]; blake3_hash32(header, 80, prehash32);

           // d_out nullen
           static const uint8_t zero32[32] = {0};
           clEnqueueWriteBuffer(q, d_out, CL_FALSE, 0, 32, zero32, 0, NULL, NULL);

           // prehash32 → GPU (nur relevant, falls do_init=1)
           cl_event write_ev;
           clEnqueueWriteBuffer(q, d_phash, CL_FALSE, 0, 32, prehash32, 0, NULL, &write_ev);
           clWaitForEvents(1, &write_ev);
           clReleaseEvent(write_ev);

           // CPU init (B0,B1) falls aktiviert
           cl_uint do_init = 1;
           if (g_cpu_init) {
               uint8_t B0[1024], B1[1024];
               argon2d_cpu_first_blocks(prehash32, m_cost_kb, B0, B1);
               clEnqueueWriteBuffer(q, d_mem, CL_TRUE, 0,     1024, B0, 0, NULL, NULL);
               clEnqueueWriteBuffer(q, d_mem, CL_TRUE, 1024,  1024, B1, 0, NULL, NULL);
               do_init = 0;
               if (g_debug && nonce==0){
                   printf("[DEBUG] CPU-init wrote B0/B1 for job %s\n", J.job_id);
               }
           }

           // ===== Kernel-Args =====
           err = clSetKernelArg(krn, 0, sizeof(cl_mem), &d_phash);         check_arg("prehash32", err, 0);
           err = clSetKernelArg(krn, 1, sizeof(cl_mem), &d_mem);           check_arg("mem", err, 1);
           err = clSetKernelArg(krn, 2, sizeof(cl_uint), &m_cost_kb);      check_arg("blocks_per_lane", err, 2);
           err = clSetKernelArg(krn, 7, sizeof(cl_mem), &d_out);           check_arg("out32", err, 7);
           err = clSetKernelArg(krn, 8, sizeof(cl_uint), &do_init);        check_arg("do_init", err, 8);

           // t=2, 4 Slices, CHUNK_BLOCKS
           for (cl_uint pass = 0; pass < T_COST; pass++) {
               for (cl_uint slice = 0; slice < 4; slice++) {
                   const cl_uint slice_begin = slice * (m_cost_kb / 4);
                   const cl_uint slice_end   = (slice + 1) * (m_cost_kb / 4);
                   for (cl_uint start = slice_begin; start < slice_end; start += CHUNK_BLOCKS) {
                       cl_uint end = start + CHUNK_BLOCKS; if (end > slice_end) end = slice_end;

                       err = clSetKernelArg(krn, 3, sizeof(cl_uint), &pass);   check_arg("pass_index", err, 3);
                       err = clSetKernelArg(krn, 4, sizeof(cl_uint), &slice);  check_arg("slice_index", err, 4);
                       err = clSetKernelArg(krn, 5, sizeof(cl_uint), &start);  check_arg("start_block", err, 5);
                       err = clSetKernelArg(krn, 6, sizeof(cl_uint), &end);    check_arg("end_block", err, 6);

                       size_t G = 1;
                       err = clEnqueueNDRangeKernel(q, krn, 1, NULL, &G, NULL, 0, NULL, NULL);
                       if (err != CL_SUCCESS) {
                           fprintf(stderr, "clEnqueueNDRangeKernel failed: %d\n", err);
                           exit(1);
                       }
                       clFlush(q);
                   }
               }
           }
           clFinish(q);

           // GPU-Argon2d-Tag (32B), dann SHA3-256
           uint8_t argon_gpu[32]; clEnqueueReadBuffer(q, d_out, CL_TRUE, 0, 32, argon_gpu, 0, NULL, NULL);
           if (g_debug && nonce==0){
               hexdump("prehash", prehash32, 32);
               hexdump("gpu_argon2", argon_gpu, 32);
           }

           uint8_t final_hash[32]; sha3_256(argon_gpu, 32, final_hash);

           // Target (BE) Vergleich (final_hash ist BE)
           int ok = 1;
           for (int i = 0; i < 32; i++) {
               if (final_hash[i] < target_be[i]) { ok = 1; break; }
               if (final_hash[i] > target_be[i]) { ok = 0; break; }
           }
           if (ok) {
               printf("FOUND share  ntime=%08x nonce=%08x", J.ntime, nonce);
               if (stratum_submit(&S, &J, en2_hex, J.ntime, nonce)) {
                   // ACK-Ausgabe im Submit
               } else {
                   printf(" -> Submit failed\n");
               }
           }

           // Stats
           hashes_window++;
           uint64_t now = mono_ms();
           static uint64_t t_print_local = 0;
           if (!t_print_local) t_print_local = now;
           if (now - t_print_local >= 5000) {
               double secs = (now - t_print_local) / 1000.0;
               double rate = hashes_window / secs;
               printf("Hashrate: %.1f H/s | Job: %s\r", rate, J.job_id[0]?J.job_id:"-");
               fflush(stdout);
               t_print_local = now;
               hashes_window = 0;
           }

           // sanftes Stratum-Polling
           if ((nonce % 1000) == 0 && mono_ms() - t_poll >= 100) {
               stratum_job_t Jtmp;
               while (stratum_get_job(&S, &Jtmp)) {
                   if (strcmp(Jtmp.job_id, J.job_id) != 0 || Jtmp.clean) {
                       J = Jtmp;
                       uint8_t prev_be2[32]; hex2bin(J.prevhash_hex, prev_be2, 32);
                       for (int i = 0; i < 32; i++) prevhash_le[i] = prev_be2[31 - i];
                       target_from_nbits(J.nbits, target_be);
                       printf("\nSwitch to job %s\n", J.job_id);
                       fflush(stdout);
                       break;
                   }
               }
               t_poll = mono_ms();
           }
       } // nonces
       usleep(5000);
   }

   clReleaseKernel(krn);
   clReleaseProgram(prog);
   clReleaseMemObject(d_mem);
   clReleaseMemObject(d_phash);
   clReleaseMemObject(d_out);
   clReleaseCommandQueue(q);
   clReleaseContext(ctx);
   close(S.sock);
   free(ksrc);
   return 0;
}


