// src/main.c — RinHash Batch-GPU Miner (Odroid N2+ / Mali-G52 Panfrost)
// Pipeline: BLAKE3(80B header LE) -> Argon2d (GPU) -> SHA3-256 (CPU) -> compare vs share target (BE) -> submit
// Stratum: Zergpool RinHash (subscribe/authorize/notify/set_difficulty/submit)
// - Kernel file rinhash_argon2d.cl remains unchanged.
// - Batch kernel preferred; single-kernel fallback retained.
// - Robust arg setup for batch kernel to avoid CL_INVALID_KERNEL_ARGS (-52).
// - Handles d=/sd= in POOL_PASS, optional RIN_FORCE_DIFF; quiet logs; RIN_DEBUG=1 shows first-batch debug.

#define CL_TARGET_OPENCL_VERSION 120
#ifndef CL_KERNEL_NUM_ARGS
#define CL_KERNEL_NUM_ARGS 0x1195
#endif
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
#include <math.h>

#include "blake3.h"

// ============================ Defaults =============================
static const char *POOL_CANDIDATES[] = {
    "rinhash.eu.mine.zergpool.com",
    "rinhash.mine.zergpool.com",
    "rinhash.na.mine.zergpool.com",
    "rinhash.asia.mine.zergpool.com",
    NULL
};
static const int   PORT_DEFAULT = 7148;

// Change via ENV: WALLET / POOL_PASS
static const char *DEFAULT_WAL  = "DTXoRQ7Zpw3FVRW2DWkLrM9Skj4Gp9SeSj"; // <— change for production
static const char *DEFAULT_PASS = "c=DOGE,ID=n2plus";                   // you can append sd= or d= via POOL_PASS

// Argon2d params
#define M_COST_KB 64
#define T_COST    2
#define LANES     1

// Batch default
#define BATCH_DEFAULT 256

// ============================ Globals / Debug =============================
static int g_debug = 0;

// Share difficulty & target (BE)
static double  g_share_diff = 1.0;
static uint8_t g_share_target_be[32] = {
    0x00,0x00,0x00,0x00, 0xFF,0xFF,0x00,0x00,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0
};
static int g_diff_locked = 0;     // 1 if RIN_FORCE_DIFF or 'd=' in pass; ignore set_difficulty

// ============================ Utils ==============================
static void blake3_hash32(const uint8_t *in, size_t len, uint8_t out[32]) {
    blake3_hasher h; blake3_hasher_init(&h);
    blake3_hasher_update(&h, in, len);
    blake3_hasher_finalize(&h, out, 32);
}

static void sha3_256_once(const uint8_t *in, size_t len, uint8_t out[32]) {
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

static uint64_t mono_ms(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec*1000ull + (uint64_t)ts.tv_nsec/1000000ull;
}

// ============================ 256-bit helpers =============================
typedef struct { uint32_t v[8]; } u256le;

static void u256le_set_diff1(u256le *a) {
    for (int i=0;i<8;i++) a->v[i]=0;
    a->v[6] = 0xFFFF0000u; // maps BE 00000000ffff0000........ in LE limbs
}

static void u256le_shl_bits(u256le *a, unsigned bits) {
    if (bits >= 256) { for(int i=0;i<8;i++) a->v[i]=0; return; }
    unsigned w = bits >> 5, b = bits & 31;
    uint32_t tmp[8] = {0};
    for (int i=7;i>=0;i--) {
        uint64_t acc = 0;
        if ((int)i - (int)w >= 0) {
            acc = (uint64_t)a->v[i - w] << b;
            if (b && (int)i - (int)w - 1 >= 0)
                acc |= (uint64_t)a->v[i - w - 1] >> (32 - b);
        }
        tmp[i] = (uint32_t)acc;
    }
    for (int i=0;i<8;i++) a->v[i]=tmp[i];
}

static void u256le_rshift_bits(u256le *a, unsigned bits) {
    if (bits >= 256) { for(int i=0;i<8;i++) a->v[i]=0; return; }
    unsigned w = bits >> 5, b = bits & 31;
    uint32_t tmp[8]={0};
    for (int i=0;i<8;i++) {
        uint64_t acc = 0;
        if (i + (int)w <= 7) {
            acc = (uint64_t)a->v[i + w] >> b;
            if (b && i + (int)w + 1 <= 7)
                acc |= (uint64_t)a->v[i + w + 1] << (32 - b);
        }
        tmp[i] = (uint32_t)acc;
    }
    for (int i=0;i<8;i++) a->v[i]=tmp[i];
}

static void u256le_div_u64(const u256le *a, uint64_t d, u256le *q) {
    uint64_t rem = 0;
    for (int i=7;i>=0;i--) {
        uint64_t cur = (rem << 32) | a->v[i];
        uint64_t qi  = cur / d;
        rem          = cur % d;
        q->v[i] = (uint32_t)qi;
    }
}

static void u256le_to_be_bytes(const u256le *a, uint8_t out_be[32]) {
    for (int i=0;i<8;i++) {
        uint32_t w = a->v[7-i];
        out_be[i*4+0] = (uint8_t)(w >> 24);
        out_be[i*4+1] = (uint8_t)(w >> 16);
        out_be[i*4+2] = (uint8_t)(w >>  8);
        out_be[i*4+3] = (uint8_t)(w      );
    }
}

static void diff_to_target(double diff, uint8_t out_be[32]) {
    if (diff <= 0.0) { memset(out_be, 0, 32); return; }
    int e;
    double f = frexp(diff, &e);                 // diff = f * 2^e, 0.5<=f<1
    uint64_t m = (uint64_t)ldexp(f, 52);        // 52-bit mantissa
    if (m == 0) { memset(out_be, 0xFF, 32); return; }
    int shift = 52 - e;                         // target = floor((diff1 << shift)/m)
    u256le num; u256le_set_diff1(&num);
    if (shift >= 0) u256le_shl_bits(&num, (unsigned)shift);
    else            u256le_rshift_bits(&num, (unsigned)(-shift));
    u256le q; u256le_div_u64(&num, m, &q);
    u256le_to_be_bytes(&q, out_be);
}

static int hash_meets_target_be(const uint8_t hash_be[32], const uint8_t target_be[32]) {
    for (int i = 0; i < 32; i++) {
        if (hash_be[i] < target_be[i]) return 1;
        if (hash_be[i] > target_be[i]) return 0;
    }
    return 1;
}

// ============================ Stratum ==============================
typedef struct {
    char host[256];
    int  port;
    int  sock;
    char wallet[256];
    char pass[512];
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
        if (set_nonblock(s, 1) != 0) { perror("fcntl(O_NONBLOCK)"); close(s); return -1; }
        int rc = connect(s, (struct sockaddr*)&sa, sizeof sa);
        if (rc != 0 && errno != EINPROGRESS) { perror("connect"); close(s); return -1; }
        fd_set wfds; FD_ZERO(&wfds); FD_SET(s, &wfds);
        struct timeval tv = { .tv_sec = timeout_ms/1000, .tv_usec = (timeout_ms%1000)*1000 };
        rc = select(s+1, NULL, &wfds, NULL, &tv);
        if (rc <= 0) { if (rc==0) fprintf(stderr,"connect timeout %s\n", host); else perror("select"); close(s); return -1; }
        int soerr=0; socklen_t slen=sizeof soerr;
        if (getsockopt(s,SOL_SOCKET,SO_ERROR,&soerr,&slen)<0 || soerr!=0){
            fprintf(stderr,"connect() error: %s\n", soerr?strerror(soerr):"getsockopt");
            close(s); return -1;
        }
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
        s = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (s < 0) continue;
        if (set_nonblock(s, 1) != 0) { close(s); s=-1; continue; }
        int rc = connect(s, p->ai_addr, p->ai_addrlen);
        if (rc != 0 && errno != EINPROGRESS) { close(s); s=-1; continue; }
        fd_set wfds; FD_ZERO(&wfds); FD_SET(s, &wfds);
        struct timeval tv = { .tv_sec = timeout_ms/1000, .tv_usec = (timeout_ms%1000)*1000 };
        rc = select(s+1, NULL, &wfds, NULL, &tv);
        if (rc <= 0) { close(s); s=-1; continue; }
        int soerr=0; socklen_t slen=sizeof soerr;
        if (getsockopt(s, SOL_SOCKET, SO_ERROR, &soerr, &slen) < 0 || soerr != 0) { close(s); s=-1; continue; }
        break;
    }

    freeaddrinfo(res);
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

// -------- Zeilenpuffer --------
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

// --- Subscribe parsing (extranonce) ---
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

// Parse set_difficulty line
static int parse_set_difficulty_line(const char *line, double *out_diff) {
    const char *p = strstr(line, "mining.set_difficulty");
    if (!p) return 0;
    p = strstr(p, "\"params\"");
    if (!p) return 0;
    p = strchr(p, '[');
    if (!p) return 0;
    p++;
    while (*p==' '||*p=='\t') p++;
    char *endp=NULL;
    double d = strtod(p, &endp);
    if (endp == p) return 0;
    *out_diff = d;
    return 1;
}

// Parse diff from password (d= locks; sd= start)
static int parse_diff_from_pass(const char *pass, double *out_diff, int *out_locked) {
    if (!pass || !*pass) return 0;
    const char *p = pass;
    double found = 0.0; int have = 0; int locked = 0;
    while (*p) {
        while (*p == ' ' || *p == ',' || *p == ';') p++;
        if (!*p) break;
        const char *k = p;
        while (*p && *p != ',' && *p != ';') p++;
        size_t L = (size_t)(p - k);
        if (L >= 2 && !strncmp(k,"d=",2)) {
            found = strtod(k+2, NULL); have=1; locked=1; // hard diff
        } else if (L >= 3 && !strncmp(k,"sd=",3)) {
            if (!locked) { found = strtod(k+3, NULL); have=1; }
        }
    }
    if (have && found > 0.0) {
        *out_diff = found;
        *out_locked = locked;
        return 1;
    }
    return 0;
}

static int stratum_connect_one(stratum_ctx_t *C,const char *host,int port,const char *user,const char *pass){
    memset(C,0,sizeof *C);
    snprintf(C->host,sizeof C->host,"%s",host); C->port=port;
    snprintf(C->wallet,sizeof C->wallet,"%s",user);
    snprintf(C->pass,sizeof C->pass,"%s",pass ? pass : "");

    C->sock = sock_connect_verbose(host, port, 5000);
    if (C->sock < 0) return 0;

    char sub[256];
    snprintf(sub,sizeof sub,"{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"rin-ocl/1.0\"]}\n");
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
            // set_difficulty can arrive any time
            double dtmp;
            if (!g_diff_locked && parse_set_difficulty_line(line, &dtmp)) {
                g_share_diff = dtmp;
                diff_to_target(g_share_diff, g_share_target_be);
                if (g_debug) {
                    printf("[DIFF] set_difficulty=%.8f\n", g_share_diff);
                    printf("[DIFF] share_target="); for(int i=0;i<32;i++) printf("%02x", g_share_target_be[i]); printf("\n");
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

    char auth[1536];
    int nw = snprintf(auth,sizeof auth,"{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\"%s\",\"%s\"]}\n",C->wallet,C->pass);
    if (nw <= 0 || (size_t)nw >= sizeof auth) { fprintf(stderr,"authorize snprintf overflow\n"); close(C->sock); return 0; }
    if(!send_line_verbose(C->sock,auth)) { fprintf(stderr,"Stratum send authorize failed\n"); close(C->sock); return 0; }

    // Drain early chatter briefly
    time_t t1=time(NULL);
    while (time(NULL)-t1 < 2) { recv_into_buffer(C->sock, 200); char dump[16384]; while(next_line(dump,sizeof dump)){} }

    printf("Connected. extranonce1=%s ex2_size=%u  (Batch=%u)\n", C->extranonce1, (unsigned)C->extranonce2_size, (unsigned)BATCH_DEFAULT);
    return 1;
}

static int stratum_connect_any(stratum_ctx_t *C, const char **hosts, int port, const char *user, const char *pass){
    const char *env_host = getenv("POOL_HOST");
    const char *env_port = getenv("POOL_PORT");
    if (env_host && env_host[0]) {
        int p = (env_port && env_port[0]) ? atoi(env_port) : port;
        if (stratum_connect_one(C, env_host, p, user, pass)) return 1;
        fprintf(stderr, "ENV host failed, falling back to default list…\n");
    }
    for (int i = 0; hosts[i]; i++) {
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

    // merkle array
    J->merkle_count = 0;
    const char *m_lb=strchr(p,'['), *m_rb=NULL;
    if (m_lb) { const char *scan=m_lb+1; for(;scan<rb;scan++){ if(*scan==']'){ m_rb=scan; break; }}}
    if (m_lb && m_rb && m_rb>m_lb) {
        const char *mp = m_lb+1;
        while (J->merkle_count < 16) {
            char tmp[65]; if(!get_next_quoted(mp,m_rb,tmp,sizeof tmp)) break;
            snprintf(J->merkle_hex[J->merkle_count],65,"%s",tmp);
            J->merkle_count++;
            mp = strchr(mp+1,'"'); if(!mp) break;
        }
        p = m_rb+1;
    }

    // version, nbits, ntime (hex strings)
    char vhex[16]={0}, nbhex[16]={0}, nth[16]={0};
    if(!get_next_quoted(p,rb,vhex,sizeof vhex)) return 0; sscanf(vhex,"%x",&J->version);
    if(!get_next_quoted(p,rb,nbhex,sizeof nbhex)) return 0; sscanf(nbhex,"%x",&J->nbits);
    if(!get_next_quoted(p,rb,nth,sizeof nth))    return 0; sscanf(nth,"%x",&J->ntime);

    J->clean = strstr(p,"true")!=NULL;
    return 1;
}

static void stratum_wait_submit_ack(int sock) {
    char line[4096];
    uint64_t start = mono_ms();
    while (mono_ms() - start <= 3000) {
        recv_into_buffer(sock, 0);
        while (next_line(line, sizeof line)) {
            if (strstr(line, "\"id\":4")) {
                if (strstr(line, "\"result\":true")) {
                    printf(" -> ACCEPTED\n"); fflush(stdout); return;
                } else if (strstr(line, "\"error\"")) {
                    printf(" -> REJECTED: %s", line); fflush(stdout); return;
                }
            }
            double dtmp;
            if (!g_diff_locked && parse_set_difficulty_line(line, &dtmp)) {
                g_share_diff = dtmp;
                diff_to_target(g_share_diff, g_share_target_be);
                if (g_debug) {
                    printf("[DIFF] set_difficulty=%.8f\n", g_share_diff);
                    printf("[DIFF] share_target="); for(int i=0;i<32;i++) printf("%02x", g_share_target_be[i]); printf("\n");
                }
            }
        }
        usleep(20000);
    }
}

static size_t build_submit_json(char *req, size_t cap,
                                const char *wallet, const char *job_id,
                                const char *ex2_hex, uint32_t ntime_le, uint32_t nonce_le) {
    char ntime_hex[9], nonce_hex[9];
    snprintf(ntime_hex, sizeof ntime_hex, "%08x", ntime_le);
    snprintf(nonce_hex, sizeof nonce_hex, "%08x", nonce_le);
    return (size_t)snprintf(req, cap,
        "{\"id\":4,\"method\":\"mining.submit\",\"params\":[\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"]}\n",
        wallet, job_id, ex2_hex, ntime_hex, nonce_hex);
}

static int stratum_submit(stratum_ctx_t *C,const stratum_job_t *J,
                          const char *extranonce2_hex,uint32_t ntime_le,uint32_t nonce_le){
    char req[1536];
    size_t n = build_submit_json(req, sizeof req, C->wallet, J->job_id, extranonce2_hex, ntime_le, nonce_le);
    if (n == 0 || n >= sizeof req) { fprintf(stderr,"submit snprintf overflow\n"); return 0; }
    int ok = send_line_verbose(C->sock,req);
    if (ok) stratum_wait_submit_ack(C->sock);
    return ok;
}

// nbits -> target (for info only)
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
        int idx = 32 - (int)exp; if (idx < 0) idx = 0;
        if (idx + 3 <= 32) {
            target[idx]   = (mant >> 16) & 0xFF;
            target[idx+1] = (mant >> 8)  & 0xFF;
            target[idx+2] = mant & 0xFF;
        }
    }
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

// ============================ OpenCL helpers ======================
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

static volatile sig_atomic_t g_stop = 0;
static void on_sigint(int sig){ (void)sig; g_stop = 1; }

// --- Batch arg setter that adapts to kernel signature to avoid CL_INVALID_KERNEL_ARGS (-52)
static int set_batch_kernel_args_adaptive(cl_kernel k_batch,
                                          cl_mem d_prehash, cl_mem d_mem, cl_mem d_out,
                                          cl_uint num_items, cl_uint blocks_per_lane) {
    cl_int err;
    err  = clSetKernelArg(k_batch, 0, sizeof(cl_mem), &d_prehash);
    err |= clSetKernelArg(k_batch, 1, sizeof(cl_mem), &d_mem);
    err |= clSetKernelArg(k_batch, 2, sizeof(cl_mem), &d_out);
    err |= clSetKernelArg(k_batch, 3, sizeof(cl_uint), &num_items);
    err |= clSetKernelArg(k_batch, 4, sizeof(cl_uint), &blocks_per_lane);
    if (err != CL_SUCCESS) return err;

    cl_uint num_args = 5;
    clGetKernelInfo(k_batch, CL_KERNEL_NUM_ARGS, sizeof(num_args), &num_args, NULL);
    if (num_args <= 5) return CL_SUCCESS;

    cl_uint tcost = T_COST;
    cl_uint lanes = LANES;
    cl_uint do_init = 1u;
    cl_uint zero = 0u;

    cl_uint extras[4] = { tcost, lanes, do_init, zero };
    for (cl_uint i = 5, j = 0; i < num_args && j < 4; ++i, ++j) {
        err = clSetKernelArg(k_batch, i, sizeof(cl_uint), &extras[j]);
        if (err == CL_INVALID_ARG_SIZE || err == CL_INVALID_ARG_VALUE) {
            cl_mem alt = (i % 2) ? d_out : d_mem;
            err = clSetKernelArg(k_batch, i, sizeof(cl_mem), &alt);
        }
        if (err != CL_SUCCESS) return err;
    }
    return CL_SUCCESS;
}

// ============================== MAIN ============================
int main(int argc, char **argv) {
    signal(SIGINT, on_sigint);

    // ENV
    if (getenv("RIN_DEBUG")) g_debug = atoi(getenv("RIN_DEBUG"));

    cl_uint BATCH = BATCH_DEFAULT;
    if (getenv("RIN_BATCH")) {
        int b = atoi(getenv("RIN_BATCH"));
        if (b > 0 && b <= 4096) BATCH = (cl_uint)b;
    }

    // Wallet/Pass from env
    const char *wallet_env = getenv("WALLET");
    const char *pass_env   = getenv("POOL_PASS");
    const char *WAL  = wallet_env && wallet_env[0] ? wallet_env : DEFAULT_WAL;
    const char *PASS = pass_env   && pass_env[0]   ? pass_env   : DEFAULT_PASS;

    // Optional forced share diff (locks)
    const char *force_d_env = getenv("RIN_FORCE_DIFF");
    if (force_d_env && force_d_env[0]) {
        g_share_diff = atof(force_d_env);
        if (g_share_diff < 1e-12) g_share_diff = 1e-12;
        diff_to_target(g_share_diff, g_share_target_be);
        g_diff_locked = 1;
        printf("[FORCE] share difficulty=%.12f (locked)\n", g_share_diff);
        if (g_debug) {
            printf("[FORCE] share_target="); for(int i=0;i<32;i++) printf("%02x", g_share_target_be[i]); printf("\n");
        }
    } else {
        // If pass contains d= or sd=, use it to set initial share target.
        double pdiff=0.0; int locked=0;
        if (parse_diff_from_pass(PASS, &pdiff, &locked) && pdiff > 0.0) {
            g_share_diff = pdiff;
            diff_to_target(g_share_diff, g_share_target_be);
            g_diff_locked = locked ? 1 : 0;
            printf("[PASS] initial difficulty=%.12f%s\n", g_share_diff, locked?" (locked via d=)":" (start via sd=)");
            if (g_debug) {
                printf("[PASS] share_target="); for(int i=0;i<32;i++) printf("%02x", g_share_target_be[i]); printf("\n");
            }
        }
    }

    printf("=== RinHash Batch-GPU Miner ===\n");
    printf("Batch size: %u\n", BATCH);
    printf("Wallet: %s\n", WAL);

    // OpenCL Setup
    cl_int err; cl_platform_id plat; cl_device_id dev;
    cl_uint np=0; clGetPlatformIDs(1,&plat,&np);
    if (np==0){ fprintf(stderr,"No OpenCL platform\n"); return 1; }
    err = clGetDeviceIDs(plat,CL_DEVICE_TYPE_GPU,1,&dev,NULL);
    if (err!=CL_SUCCESS){ fprintf(stderr,"No GPU device\n"); return 1; }

    char devname[256]={0}; clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof devname,devname,NULL);
    printf("Device: %s\n", devname);

    cl_context ctx = clCreateContext(NULL,1,&dev,NULL,NULL,&err);
    cl_command_queue q = clCreateCommandQueue(ctx,dev,0,&err);
    if (err!=CL_SUCCESS){ fprintf(stderr,"clCreateCommandQueue: %d\n",err); return 1; }

    // Build program
    char *ksrc = load_kernel_source("rinhash_argon2d.cl");
    cl_program prog = clCreateProgramWithSource(ctx,1,(const char**)&ksrc,NULL,&err);
    err = clBuildProgram(prog,1,&dev,"-cl-std=CL1.2",NULL,NULL);
    if (err!=CL_SUCCESS){
        size_t L; clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,0,NULL,&L);
        char *log=(char*)malloc(L+1); clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,L,log,NULL);
        log[L]=0; fprintf(stderr,"Build failed:\n%s\n",log); free(log); return 1;
    }

    // Prefer batch kernel
    cl_kernel k_batch = clCreateKernel(prog,"argon2d_batch",&err);
    int have_batch = (err == CL_SUCCESS);
    cl_kernel k_single = NULL;
    if (!have_batch) {
        k_single = clCreateKernel(prog,"argon2d_core",&err);
        if (err != CL_SUCCESS) { fprintf(stderr,"No usable kernel found\n"); return 1; }
        printf("Using single kernel (fallback)\n");
    } else {
        printf("Using batch kernel\n");
    }

    // Buffers
    cl_mem d_prehash = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 32u * BATCH, NULL, &err);
    cl_mem d_mem     = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 1024u * M_COST_KB * BATCH, NULL, &err);
    cl_mem d_out     = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 32u * BATCH, NULL, &err);
    if (err!=CL_SUCCESS){ fprintf(stderr,"clCreateBuffer failed: %d\n", err); return 1; }

    uint8_t *h_prehash = (uint8_t*)malloc(32u * BATCH);
    uint8_t *h_out     = (uint8_t*)malloc(32u * BATCH);
    if (!h_prehash || !h_out) { fprintf(stderr,"malloc failed\n"); return 1; }

    // ===== Stratum =====
    int PORT = getenv("POOL_PORT") ? atoi(getenv("POOL_PORT")) : PORT_DEFAULT;
    stratum_ctx_t S;
    if(!stratum_connect_any(&S, POOL_CANDIDATES, PORT, WAL, PASS)){
        fprintf(stderr,"Stratum connect failed\n");
        return 1;
    }

    uint8_t prevhash_le[32], target_block_be[32];
    stratum_job_t J={0}, Jnew={0}; int have_job=0;

    uint64_t hashes_window = 0;
    uint64_t t_rate = mono_ms();
    uint64_t t_poll = mono_ms();

    // extranonce2 handling (masked to size)
    static uint32_t extranonce2_counter = 1;
    uint32_t en2_mask = (S.extranonce2_size >= 4) ? 0xFFFFFFFFu : ((1u << (S.extranonce2_size*8)) - 1u);

    static uint32_t nonce_base = 0;
    int debug_shown_for_job = 0;

    while (!g_stop) {
        // Poll stratum
        if (mono_ms() - t_poll >= 50) {
            char line[16384];
            recv_into_buffer(S.sock, 0);
            while (next_line(line,sizeof line)) {
                double dtmp;
                if (!g_diff_locked && parse_set_difficulty_line(line, &dtmp)) {
                    g_share_diff = dtmp;
                    diff_to_target(g_share_diff, g_share_target_be);
                    if (g_debug) {
                        printf("[DIFF] set_difficulty=%.8f\n", g_share_diff);
                        printf("[DIFF] share_target="); for(int i=0;i<32;i++) printf("%02x", g_share_target_be[i]); printf("\n");
                    }
                    continue;
                }
                if (strstr(line,"\"mining.notify\"")) {
                    if (stratum_parse_notify(line,&Jnew)) {
                        J = Jnew; have_job = 1;
                        uint8_t prev_be[32]; hex2bin(J.prevhash_hex, prev_be, 32);
                        for (int i = 0; i < 32; i++) prevhash_le[i] = prev_be[31 - i];
                        target_from_nbits(J.nbits, target_block_be);
                        nonce_base = 0;
                        extranonce2_counter = 1; // reset en2 per new job
                        debug_shown_for_job = 0;
                        printf("\nNew Job id %s\n", J.job_id);
                    }
                }
            }
            t_poll = mono_ms();
        }
        if (!have_job) { usleep(10000); continue; }

        // ----- Coinbase / Merkle -----
        uint8_t coinb1[4096], coinb2[4096];
        size_t cb1 = strlen(J.coinb1_hex) / 2, cb2 = strlen(J.coinb2_hex) / 2;
        hex2bin(J.coinb1_hex, coinb1, cb1);
        hex2bin(J.coinb2_hex, coinb2, cb2);

        uint8_t en1[64]; size_t en1b = strlen(S.extranonce1) / 2; if (en1b > 64) en1b = 64;
        hex2bin(S.extranonce1, en1, en1b);

        uint32_t en2_val = (extranonce2_counter++) & en2_mask; // roll within size
        char en2_hex[64];
        snprintf(en2_hex, sizeof en2_hex, "%0*x", S.extranonce2_size * 2, en2_val);
        uint8_t en2[64]; hex2bin(en2_hex, en2, S.extranonce2_size);

        uint8_t coinbase[8192]; size_t off = 0;
        memcpy(coinbase + off, coinb1, cb1); off += cb1;
        memcpy(coinbase + off, en1, en1b); off += en1b;
        memcpy(coinbase + off, en2, S.extranonce2_size); off += S.extranonce2_size;
        memcpy(coinbase + off, coinb2, cb2); off += cb2;

        uint8_t cbh_be[32]; double_sha256(coinbase, off, cbh_be);

        uint8_t merkleroot_le[32];
        build_merkle_root_le(cbh_be, J.merkle_hex, J.merkle_count, merkleroot_le);

        // ----- Header+BLAKE3 für BATCH Nonces -----
        uint32_t ntime_le = J.ntime;
        for (cl_uint i=0;i<BATCH;i++) {
            uint32_t nonce = nonce_base + i;
            uint8_t header[80];
            build_header_le(&J, prevhash_le, merkleroot_le, ntime_le, J.nbits, nonce, header);
            blake3_hash32(header, 80, &h_prehash[i*32]);
        }
        nonce_base += BATCH;

        // ----- GPU: Argon2d -----
        err = clEnqueueWriteBuffer(q, d_prehash, CL_FALSE, 0, 32u*BATCH, h_prehash, 0, NULL, NULL);
        if (err!=CL_SUCCESS){ fprintf(stderr,"clEnqueueWriteBuffer(prehash)=%d\n", err); return 1; }

        uint32_t blocks_per_lane = M_COST_KB;
        size_t global = (size_t)BATCH;

        uint64_t t0 = mono_ms();

        int used_batch = 0;
        if (have_batch) {
            err = set_batch_kernel_args_adaptive(k_batch, d_prehash, d_mem, d_out, BATCH, blocks_per_lane);
            if (err == CL_SUCCESS) {
                err = clEnqueueNDRangeKernel(q, k_batch, 1, NULL, &global, NULL, 0, NULL, NULL);
                if (err == CL_SUCCESS) used_batch = 1;
                else fprintf(stderr,"clEnqueueNDRangeKernel(batch)=%d\n", err);
            } else {
                fprintf(stderr,"clSetKernelArg(batch)=%d\n", err);
            }
        }
        if (!used_batch) {
            if (!k_single) {
                k_single = clCreateKernel(prog,"argon2d_core",&err);
                if (err != CL_SUCCESS) { fprintf(stderr,"Fallback kernel create failed: %d\n", err); return 1; }
            }
            // __kernel void argon2d_core(__global const u8* prehash32, __global u8* mem,
            //  const u32 blocks_per_lane, const u32 pass_index, const u32 slice_index,
            //  const u32 start_block, const u32 end_block, __global u8* out32, const u32 do_init)
            cl_uint pass=0,slice=0,start=0,end=blocks_per_lane,do_init=1;
            err  = clSetKernelArg(k_single, 0, sizeof(cl_mem), &d_prehash);
            err |= clSetKernelArg(k_single, 1, sizeof(cl_mem), &d_mem);
            err |= clSetKernelArg(k_single, 2, sizeof(cl_uint), &blocks_per_lane);
            err |= clSetKernelArg(k_single, 3, sizeof(cl_uint), &pass);
            err |= clSetKernelArg(k_single, 4, sizeof(cl_uint), &slice);
            err |= clSetKernelArg(k_single, 5, sizeof(cl_uint), &start);
            err |= clSetKernelArg(k_single, 6, sizeof(cl_uint), &end);
            err |= clSetKernelArg(k_single, 7, sizeof(cl_mem), &d_out);
            err |= clSetKernelArg(k_single, 8, sizeof(cl_uint), &do_init);
            if (err!=CL_SUCCESS){ fprintf(stderr,"clSetKernelArg(single)=%d\n", err); return 1; }
            size_t one = 1;
            err = clEnqueueNDRangeKernel(q, k_single, 1, NULL, &one, NULL, 0, NULL, NULL);
            if (err!=CL_SUCCESS){ fprintf(stderr,"clEnqueueNDRangeKernel(single)=%d\n", err); return 1; }
        }

        clFinish(q);

        // Read results
        err = clEnqueueReadBuffer(q, d_out, CL_TRUE, 0, 32u*BATCH, h_out, 0, NULL, NULL);
        if (err!=CL_SUCCESS){ fprintf(stderr,"clEnqueueReadBuffer(out)=%d\n", err); return 1; }

        uint64_t t1 = mono_ms();
        double batch_ms = (double)(t1 - t0);
        hashes_window += used_batch ? BATCH : 1;

        // ----- Final SHA3 + share target check + submit -----
        uint8_t debug_final_first[32]; int debug_final_set=0;
        cl_uint loopN = used_batch ? BATCH : 1;

        // Keep the exact extranonce2 used for this batch for submits
        char en2_hex_submit[64];
        snprintf(en2_hex_submit, sizeof en2_hex_submit, "%0*x", S.extranonce2_size * 2, en2_val);

        for (cl_uint i=0;i<loopN;i++) {
            uint8_t final_hash[32];
            sha3_256_once(&h_out[i*32], 32, final_hash); // SHA3-256 digest (BE)

            if (!debug_final_set) { memcpy(debug_final_first, final_hash, 32); debug_final_set=1; }

            if (hash_meets_target_be(final_hash, g_share_target_be)) {
                uint32_t nonce = (nonce_base - (used_batch?BATCH:1)) + i;
                printf("FOUND share  job=%s ntime=%08x nonce=%08x", J.job_id, J.ntime, nonce);
                if (g_debug) printf(" ex2=%s", en2_hex_submit);
                fflush(stdout);
                stratum_submit(&S, &J, en2_hex_submit, J.ntime, nonce);
            }
        }

        if (g_debug && !debug_shown_for_job) {
            printf("[DEBUG] Job %s\n", J.job_id);
            hexdump("prehash[0]", &h_prehash[0], 32);
            hexdump("gpu_argon2[0]", &h_out[0], 32);
            hexdump("final_sha3[0]", debug_final_first, 32);
            hexdump("share_target", g_share_target_be, 32);
            debug_shown_for_job = 1;
        }

        // Hashrate every ~5s
        uint64_t now = mono_ms();
        if (now - t_rate >= 5000) {
            double secs = (now - t_rate) / 1000.0;
            double rate = (double)hashes_window / secs;
            printf("Hashrate: %.1f H/s | Batch: %.1fms | Job: %s\r",
                   rate, batch_ms, J.job_id[0]?J.job_id:"-");
            fflush(stdout);
            t_rate = now;
            hashes_window = 0;
        }
    }

    // Cleanup
    if (have_batch) clReleaseKernel(k_batch);
    if (k_single) clReleaseKernel(k_single);
    clReleaseProgram(prog);
    clReleaseMemObject(d_mem);
    clReleaseMemObject(d_prehash);
    clReleaseMemObject(d_out);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    free(ksrc);
    free(h_prehash);
    free(h_out);

    return 0;
}

