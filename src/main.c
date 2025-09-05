// src/main.c — RinHash Batch-GPU Miner (Odroid N2+ / Mali-G52 Panfrost)
// Pipeline: BLAKE3(80B header LE) -> Argon2d (GPU) -> SHA3-256 (CPU) -> compare vs share target (BE) -> submit
// Stratum: Zergpool RinHash (subscribe/authorize/notify/set_difficulty/submit)

#define CL_TARGET_OPENCL_VERSION 120
#ifndef CL_KERNEL_NUM_ARGS
#define CL_KERNEL_NUM_ARGS 0x1195
#endif
#include <CL/cl.h>

#include <openssl/evp.h>
#include <openssl/sha.h>
#include <openssl/bn.h>

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

static const char *DEFAULT_WAL  = "DTXoRQ7Zpw3FVRW2DWkLrM9Skj4Gp9SeSj";
static const char *DEFAULT_PASS = "c=DOGE,ID=n2plus";

// Argon2d params
#define M_COST_KB 64
#define T_COST    2
#define LANES     1

// Batch default
#define BATCH_DEFAULT 256

// ============================ Globals / Debug =============================

static const uint8_t DIFF1_TARGET_BE[32] = {
    0x00,0x00,0x00,0x00, 0xFF,0xFF,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
};

static int g_debug = 0;

// Share difficulty & target (BE)
static double  g_share_diff = 1.0;
static uint8_t g_share_target_be[32] = {
    0x00,0x00,0x00,0x00, 0xFF,0xFF,0x00,0x00,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0
};
static int g_diff_locked = 0;

// Statistics
static uint64_t g_total_hashes = 0;
static uint32_t g_best_leading_zeros = 0;
static uint32_t g_accepted_shares = 0;
static uint32_t g_rejected_shares = 0;

// Time rolling and job management
static uint32_t g_batch_counter = 0;
static volatile uint32_t g_job_gen = 0;
static volatile int g_last_notify_clean = 1;

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

// ============================ Debug Functions =============================
static void update_statistics(const uint8_t hash_be[32]) {
    g_total_hashes++;
    
    uint32_t zeros = 0;
    for (int i = 0; i < 32; i++) {
        if (hash_be[i] == 0) {
            zeros += 8;
        } else {
            uint8_t b = hash_be[i];
            while ((b & 0x80) == 0 && zeros < 256) {
                zeros++;
                b <<= 1;
            }
            break;
        }
    }
    
    if (zeros > g_best_leading_zeros) {
        g_best_leading_zeros = zeros;
        if (g_debug) {
            printf("New best: %u leading zero bits\n", zeros);
        }
    }
}

// ============================ Difficulty calculation =============================
static void diff_to_target(double diff, uint8_t out_be[32]) {
    if (diff <= 0) diff = 1e-12;

    BN_CTX *ctx = BN_CTX_new();
    BIGNUM *diff1 = BN_bin2bn(DIFF1_TARGET_BE, 32, NULL);
    BIGNUM *num   = BN_new();
    BIGNUM *den   = BN_new();
    BIGNUM *tgt   = BN_new();

    // Fixed-Point: *2^24, damit kleine diff-Werte exakt bleiben
    const unsigned SHIFT = 24;
    BN_copy(num, diff1);
    BN_lshift(num, num, SHIFT);  // diff1 * 2^24

    long double scaled = (long double)diff * (long double)(1ULL << SHIFT);
    uint64_t den64 = (scaled < 1.0L) ? 1ULL : (uint64_t)llroundl(scaled);
    BN_set_word(den, den64);

    BN_div(tgt, NULL, num, den, ctx);  // floor(diff1*2^24 / (diff*2^24))

    if (BN_is_zero(tgt)) BN_one(tgt);

#if OPENSSL_VERSION_NUMBER >= 0x10100000L
    BN_bn2binpad(tgt, out_be, 32);     // exakt 32-Byte big-endian
#else
    memset(out_be, 0, 32);
    int n = BN_num_bytes(tgt);
    if (n > 32) {
        uint8_t tmp[64]; int m = BN_bn2bin(tgt, tmp);
        memcpy(out_be, tmp + (m - 32), 32);
    } else {
        BN_bn2bin(tgt, out_be + (32 - n));
    }
#endif

    BN_free(diff1); BN_free(num); BN_free(den); BN_free(tgt); BN_CTX_free(ctx);
}

// 256-Bit Vergleich: Hash (BE) <= Target (BE) ?
static int hash_meets_target_be(const uint8_t hash_be[32], const uint8_t target_be[32]) {
    for (int i = 0; i < 32; i++) {
        if (hash_be[i] < target_be[i]) return 1; // Hash kleiner -> erfüllt Target
        if (hash_be[i] > target_be[i]) return 0; // Hash größer  -> nicht erfüllt
    }
    return 1; // exakt gleich
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
    char     prevhash_hex[65];
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

// Line buffer
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
    if (inlen + (size_t)n >= sizeof inbuf) inlen = 0;
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

// Subscribe parsing helper (advance pointer)
static int next_quoted(const char **pp, const char *end, char *out, size_t cap) {
    const char *p = *pp, *q1 = NULL, *q2 = NULL;
    for (; p < end; p++) if (*p == '"') { q1 = p; break; }
    if (!q1) return 0;
    for (p = q1 + 1; p < end; p++) if (*p == '"') { q2 = p; break; }
    if (!q2) return 0;
    size_t L = (size_t)(q2 - (q1 + 1)); if (L >= cap) L = cap - 1;
    memcpy(out, q1 + 1, L); out[L] = 0;
    *pp = q2 + 1;
    return 1;
}

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
    if (!strstr(line, "\"id\":1") || !strstr(line, "\"result\"")) return 0;
    const char *p = strstr(line, "\"result\""); if(!p) return 0;
    p = strchr(p, '['); if (!p) return 0; p++;
    if (*p != '[') return 0;
    int depth = 0;
    while (*p) {
        if (*p == '[') depth++;
        else if (*p == ']') { depth--; if (depth == 0) { p++; break; } }
        p++;
    }
    if (depth != 0) return 0;
    while (*p == ' ' || *p == '	' || *p == ',') p++;
    if (*p != '"') return 0;
    const char *q1 = ++p;
    while (*p && *p != '"') p++;
    if (*p != '"') return 0;
    size_t L = (size_t)(p - q1); if (L >= ex1cap) L = ex1cap - 1;
    memcpy(ex1, q1, L); ex1[L] = 0;
    p++;
    while (*p == ' ' || *p == '	' || *p == ',') p++;
    if (!(*p == '-' || (*p >= '0' && *p <= '9'))) return 0;
    unsigned long v = strtoul(p, NULL, 10);
    if (v == 0 || v > 32) v = 4;
    *ex2sz = (cl_uint)v;
    return 1;
}

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
            found = strtod(k+2, NULL); have=1; locked=1;
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
    snprintf(sub,sizeof sub,"{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"cpuminer-opt/3.21.0\"]}\n");
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
            double dtmp;
            if (!g_diff_locked && parse_set_difficulty_line(line, &dtmp)) {
                g_share_diff = dtmp;
                diff_to_target(g_share_diff, g_share_target_be);
                if (g_debug) {
                    printf("[DIFF] set_difficulty=%.8f\n", g_share_diff);
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

static int stratum_parse_notify(const char *line, stratum_job_t *J){
    if (!strstr(line,"\"mining.notify\"")) return 0;
    memset(J,0,sizeof *J);
    const char *pp = strstr(line,"\"params\""); if(!pp) return 0;
    const char *lb = strchr(pp,'['); const char *rb = lb?strrchr(pp,']'):NULL;
    if(!lb||!rb||rb<=lb) return 0;
    const char *p = lb+1;

    if(!next_quoted(&p,rb,J->job_id,sizeof J->job_id)) return 0;
    if(!next_quoted(&p,rb,J->prevhash_hex,sizeof J->prevhash_hex)) return 0;
    if(!next_quoted(&p,rb,J->coinb1_hex,sizeof J->coinb1_hex)) return 0;
    if(!next_quoted(&p,rb,J->coinb2_hex,sizeof J->coinb2_hex)) return 0;

    J->merkle_count = 0;
    // Parse merkle array
    const char *m_lb=strchr(p,'['), *m_rb=NULL;
    if (m_lb) {
        int depth=0;
        const char *scan=m_lb;
        for (; scan<rb; scan++){
            if (*scan=='[') depth++;
            else if (*scan==']') { depth--; if (depth==0){ m_rb=scan; break; } }
        }
    }
    if (m_lb && m_rb && m_rb>m_lb) {
        const char *mp = m_lb+1;
        while (J->merkle_count < 16) {
            char tmp[65];
            if(!next_quoted(&mp, m_rb, tmp, sizeof tmp)) break;
            snprintf(J->merkle_hex[J->merkle_count],65,"%s",tmp);
            J->merkle_count++;
        }
        p = m_rb+1;
    }

    char vhex[16]={0}, nbhex[16]={0}, nth[16]={0};
    if(!next_quoted(&p,rb,vhex,sizeof vhex)) return 0; sscanf(vhex,"%x",&J->version);
    if(!next_quoted(&p,rb,nbhex,sizeof nbhex)) return 0; sscanf(nbhex,"%x",&J->nbits);
    if(!next_quoted(&p,rb,nth,sizeof nth))     return 0; sscanf(nth,"%x",&J->ntime);

    J->clean = strstr(p,"true")!=NULL;
    return 1;
}

// Stratum will 8-stellige Hex-ZAHLEN für ntime/nonce (wie cpuminer), nicht LE-Byte-Hex.
static size_t build_submit_json(char *req, size_t cap,
                                const char *wallet, const char *job_id,
                                const char *ex2_hex, uint32_t ntime_le, uint32_t nonce_le)
{
    return (size_t)snprintf(
        req, cap,
        "{\"id\":4,\"method\":\"mining.submit\",\"params\":[\"%s\",\"%s\",\"%s\",\"%08x\",\"%08x\"]}\n",
        wallet, job_id, ex2_hex, (unsigned)ntime_le, (unsigned)nonce_le
    );
}

// Immer die Pool-Antwort zeigen, inkl. REJECT-Grund (JSON)
static void stratum_wait_submit_ack(int sock) {
    char line[4096];
    uint64_t start = mono_ms();

    while (mono_ms() - start <= 5000) {
        recv_into_buffer(sock, 250);

        while (next_line(line, sizeof line)) {
            if (strstr(line, "\"id\":4")) {
                if (strstr(line, "\"result\":true")) {
                    g_accepted_shares++;
                    fprintf(stderr, " -> ACCEPTED\n");
                    fflush(stdout);
                    return;
                }
                if (strstr(line, "\"error\"")) {
                    g_rejected_shares++;
                    fprintf(stderr, "[REJECT] %s", line);
                    fprintf(stderr, " -> REJECTED\n");
                    fflush(stdout);
                    return;
                }
            }
            // nebenbei reinkommende diff ggf. übernehmen
            double dtmp;
            if (!g_diff_locked && parse_set_difficulty_line(line, &dtmp)) {
                g_share_diff = dtmp;
                diff_to_target(g_share_diff, g_share_target_be);
                if (g_debug) fprintf(stderr, "[DIFF] set_difficulty=%.8f\n", g_share_diff);
            }
        }
        usleep(20000);
    }
    fprintf(stderr, "[WARN] submit ack timeout (>5s)\n");
}

static int stratum_submit(stratum_ctx_t *C, const stratum_job_t *J,
                          const char *extranonce2_hex, uint32_t ntime_le, uint32_t nonce_le)
{
    if (g_debug) {
        fprintf(stderr, "SUBMIT job=%s ex2=%s ntime=%08x nonce=%08x\n",
                J->job_id, extranonce2_hex, (unsigned)ntime_le, (unsigned)nonce_le);
    }
    char req[1536];
    size_t n = build_submit_json(req, sizeof req, C->wallet, J->job_id,
                                 extranonce2_hex, ntime_le, nonce_le);
    if (n == 0 || n >= sizeof req) {
        fprintf(stderr,"submit snprintf overflow\n");
        return 0;
    }
    int ok = send_line_verbose(C->sock, req);
    if (ok) stratum_wait_submit_ack(C->sock);
    return ok;
}

// Build merkle root from coinbase hash + branches
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

// Mali-G52 robust kernel arg setter
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
    cl_int info_err = clGetKernelInfo(k_batch, CL_KERNEL_NUM_ARGS, sizeof(num_args), &num_args, NULL);
    
    if (info_err != CL_SUCCESS || num_args > 16) {
        for (num_args = 5; num_args <= 8; num_args++) {
            cl_uint test = T_COST;
            if (clSetKernelArg(k_batch, num_args, sizeof(cl_uint), &test) == CL_INVALID_ARG_INDEX) {
                break;
            }
        }
    }
    
    if (num_args > 5) {
        cl_uint extras[] = {T_COST, LANES, 1, 0};
        for (cl_uint i = 5, j = 0; i < num_args && j < 4; i++, j++) {
            clSetKernelArg(k_batch, i, sizeof(cl_uint), &extras[j]);
        }
    }
    
    return CL_SUCCESS;
}

// Aligned buffer creation for Mali
static cl_mem create_aligned_buffer(cl_context ctx, cl_mem_flags flags, size_t size, cl_int *err) {
    size_t aligned_size = (size + 255) & ~255UL;
    
    cl_mem buf = clCreateBuffer(ctx, flags, aligned_size, NULL, err);
    if (*err != CL_SUCCESS) {
        if (aligned_size > size) {
            buf = clCreateBuffer(ctx, flags, size, NULL, err);
        }
    }
    return buf;
}

// ============================== MAIN ============================
int main(int argc, char **argv) {
    signal(SIGINT, on_sigint);

    if (getenv("RIN_DEBUG")) g_debug = atoi(getenv("RIN_DEBUG"));

        cl_uint BATCH = BATCH_DEFAULT;
    if (getenv("RIN_BATCH")) {
        int b = atoi(getenv("RIN_BATCH"));
        if (b > 0 && b <= 4096) BATCH = (cl_uint)b;
    }

    // --- NEW: chunk (Microbatch) ---
    cl_uint chunk = 64;
    if (getenv("RIN_CHUNK")) {
        int c = atoi(getenv("RIN_CHUNK"));
        if (c > 0) chunk = (cl_uint)c;
    }
    if (chunk > BATCH) chunk = BATCH;
    if (chunk < 1)     chunk = 1;

    const char *wallet_env = getenv("WALLET");
    const char *pass_env   = getenv("POOL_PASS");
    const char *WAL  = wallet_env && wallet_env[0] ? wallet_env : DEFAULT_WAL;
    const char *PASS = pass_env   && pass_env[0]   ? pass_env   : DEFAULT_PASS;

    // Parse d= / sd= aus POOL_PASS (Diff & Lock setzen)
    double pdiff = 0.0; int locked = 0;
    if (parse_diff_from_pass(PASS, &pdiff, &locked) && pdiff > 0.0) {
        g_share_diff = pdiff;
        diff_to_target(g_share_diff, g_share_target_be);
        g_diff_locked = locked;
        printf("Password diff: %.8f%s\n", g_share_diff, locked ? " (locked)" : " (start)");
    }

    const char *force_d_env = getenv("RIN_FORCE_DIFF");
    if (force_d_env && force_d_env[0]) {
        g_share_diff = atof(force_d_env);
        if (g_share_diff < 1e-12) g_share_diff = 1e-12;
        diff_to_target(g_share_diff, g_share_target_be);
        g_diff_locked = 1;
        printf("Forced diff: %.8f (locked)\n", g_share_diff);
    }

    printf("=== RinHash Batch-GPU Miner ===\n");
    printf("Batch size: %u  (chunk=%u)\n", BATCH, chunk);
    printf("Wallet: %s\n", WAL);


    // OpenCL Setup
    cl_int err; cl_platform_id plat; cl_device_id dev;
    cl_uint np = 0;
    clGetPlatformIDs(1, &plat, &np);
    if (np == 0) {
        fprintf(stderr, "No OpenCL platform\n");
        return 1;
    }
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "No GPU device\n");
        return 1;
    }

    char devname[256] = {0};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof devname, devname, NULL);
    printf("Device: %s\n", devname);

    // Check memory limits
    cl_ulong max_alloc = 0, max_mem = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(max_mem), &max_mem, NULL);

    size_t needed = (size_t)BATCH * (32 + 1024 * M_COST_KB + 32);
    if (needed > max_alloc) {
        printf("Warning: Need %zu MB, but max_alloc=%lu MB\n",
               needed / 1024 / 1024, (unsigned long)(max_alloc / 1024 / 1024));
        BATCH = (cl_uint)(max_alloc / (32 + 1024 * M_COST_KB + 32));
        if (BATCH < 32) BATCH = 32;
        printf("Auto-reducing batch to %u\n", BATCH);
        needed = (size_t)BATCH * (32 + 1024 * M_COST_KB + 32);
    }

    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);

    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
    if (err != CL_SUCCESS) {
        q = clCreateCommandQueue(ctx, dev, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Command queue creation failed: %d\n", err);
            return 1;
        }
    }

    // Build program with fallback options
    char *ksrc = load_kernel_source("rinhash_argon2d.cl");
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char**)&ksrc, NULL, &err);

    const char *build_opts[] = {
        "-cl-std=CL1.2 -cl-fast-relaxed-math -cl-mad-enable",
        "-cl-std=CL1.2 -cl-fast-relaxed-math",
        "-cl-std=CL1.2",
        "-cl-std=CL1.1",
        NULL
    };

    err = CL_BUILD_PROGRAM_FAILURE;
    for (int i = 0; build_opts[i] && err != CL_SUCCESS; i++) {
        err = clBuildProgram(prog, 1, &dev, build_opts[i], NULL, NULL);
        if (err == CL_SUCCESS) {
            if (g_debug) printf("Built with options: %s\n", build_opts[i]);
            break;
        }
    }

    if (err != CL_SUCCESS) {
        size_t L;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &L);
        char *log = (char*)malloc(L + 1);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, L, log, NULL);
        log[L] = 0;
        fprintf(stderr, "Build failed with all options:\n%s\n", log);
        free(log);
        return 1;
    }

    // Prefer batch kernel
    cl_kernel k_batch = clCreateKernel(prog, "argon2d_batch", &err);
    int have_batch = (err == CL_SUCCESS);
    cl_kernel k_single = NULL;
    if (!have_batch) {
        k_single = clCreateKernel(prog, "argon2d_core", &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "No usable kernel found\n");
            return 1;
        }
        printf("Using single kernel (fallback)\n");
    } else {
        printf("Using batch kernel\n");
    }

    // Create aligned buffers
    cl_mem d_prehash = create_aligned_buffer(ctx, CL_MEM_READ_ONLY, 32u * BATCH, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "prehash buffer failed: %d\n", err);
        return 1;
    }

    cl_mem d_mem = create_aligned_buffer(ctx, CL_MEM_READ_WRITE, 1024u * M_COST_KB * BATCH, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "mem buffer failed: %d\n", err);
        return 1;
    }

    cl_mem d_out = create_aligned_buffer(ctx, CL_MEM_WRITE_ONLY, 32u * BATCH, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "out buffer failed: %d\n", err);
        return 1;
    }

    uint8_t *h_prehash = (uint8_t*)malloc(32u * BATCH);
    uint8_t *h_out     = (uint8_t*)malloc(32u * BATCH);
    if (!h_prehash || !h_out) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    // Stratum
    int PORT = getenv("POOL_PORT") ? atoi(getenv("POOL_PORT")) : PORT_DEFAULT;
    stratum_ctx_t S;
    if (!stratum_connect_any(&S, POOL_CANDIDATES, PORT, WAL, PASS)) {
        fprintf(stderr, "Stratum connect failed\n");
        return 1;
    }

    uint8_t prevhash_le[32];
    stratum_job_t J = {0}, Jnew = {0};
    int have_job = 0;

    uint64_t hashes_window = 0;
    uint64_t t_rate = mono_ms();
    uint64_t t_poll = mono_ms();

    static uint32_t extranonce2_counter = 1;
    uint32_t en2_mask = (S.extranonce2_size >= 4) ? 0xFFFFFFFFu : ((1u << (S.extranonce2_size * 8)) - 1u);

    static uint32_t nonce_base = 0;
    int debug_shown_for_job = 0;

    while (!g_stop) {
        // Poll stratum
        if (mono_ms() - t_poll >= 50) {
            char line[16384];
            recv_into_buffer(S.sock, 0);
            while (next_line(line, sizeof line)) {
                double dtmp;
                if (!g_diff_locked && parse_set_difficulty_line(line, &dtmp)) {
                    g_share_diff = dtmp;
                    diff_to_target(g_share_diff, g_share_target_be);
                    if (g_debug) {
                        printf("[DIFF] set_difficulty=%.8f\n", g_share_diff);
                    }
                    continue;
                }
                if (strstr(line, "\"mining.notify\"")) {
                    if (stratum_parse_notify(line, &Jnew)) {
                        J = Jnew;
                        have_job = 1;
                        uint8_t prev_be[32];
                        hex2bin(J.prevhash_hex, prev_be, 32);
                        for (int i = 0; i < 32; i++) {
                            prevhash_le[i] = prev_be[31 - i];
                        }
                        nonce_base = 0;
                        extranonce2_counter = 1;
                        debug_shown_for_job = 0;

                        // Job generation and clean job tracking
                        g_job_gen++;
                        g_last_notify_clean = J.clean ? 1 : 0;
                        g_batch_counter = 0;

                        printf("New Job id %s%s\n", J.job_id, J.clean ? " (clean)" : "");
                    }
                }
            }
            t_poll = mono_ms();
        }
        if (!have_job) {
            usleep(10000);
            continue;
        }

        // Snapshot for clean detection
        uint32_t batch_gen = g_job_gen;
        uint32_t clean_at_start = g_last_notify_clean;

        // Coinbase / Merkle
        stratum_job_t batch_job = J;  // SNAPSHOT

        uint8_t coinb1[4096], coinb2[4096];
        size_t cb1 = strlen(batch_job.coinb1_hex) / 2;
        size_t cb2 = strlen(batch_job.coinb2_hex) / 2;
        hex2bin(batch_job.coinb1_hex, coinb1, cb1);
        hex2bin(batch_job.coinb2_hex, coinb2, cb2);

        uint8_t en1[64]; size_t en1b = strlen(S.extranonce1) / 2;
        if (en1b > 64) en1b = 64;
        hex2bin(S.extranonce1, en1, en1b);

        // fresh extranonce2 (mask auf ex2_size)
        uint32_t en2_mask = (S.extranonce2_size >= 4) ? 0xFFFFFFFFu : ((1u << (S.extranonce2_size * 8)) - 1u);
        
        // fresh extranonce2 - NUR beim ersten Chunk eines Jobs
        static uint32_t current_en2_for_job = 0;
        if (g_batch_counter == 0) {
            current_en2_for_job = (extranonce2_counter++) & en2_mask;
        }
        uint32_t en2_val = current_en2_for_job;

        // ---- WICHTIG: en2 BYTES -> coinbase ---- (LE Zählweise wie bisher)
        uint8_t en2[64] = {0};
        for (int i = 0; i < (int)S.extranonce2_size && i < 4; i++)
            en2[i] = (uint8_t)((en2_val >> (8 * i)) & 0xFF);

        // en2_hex GENAU aus diesen Bytes (gleiche Reihenfolge wie in coinbase!)
        char en2_hex[64] = {0};
        for (cl_uint i = 0; i < S.extranonce2_size; i++) {
            snprintf(en2_hex + i*2, sizeof en2_hex - i*2, "%02x", en2[i]);
        }

        // coinbase = coinb1 | en1 | en2 | coinb2
        uint8_t coinbase[8192]; size_t off = 0;
        memcpy(coinbase + off, coinb1, cb1); off += cb1;
        memcpy(coinbase + off, en1,    en1b); off += en1b;
        memcpy(coinbase + off, en2,    S.extranonce2_size); off += S.extranonce2_size;
        memcpy(coinbase + off, coinb2, cb2); off += cb2;

        // double-SHA256(coinbase) -> BE
        uint8_t cbh_be[32]; double_sha256(coinbase, off, cbh_be);

        // Merkle-Root (LE) für Header
        uint8_t merkleroot_le[32];
        build_merkle_root_le(cbh_be, batch_job.merkle_hex, batch_job.merkle_count, merkleroot_le);

        // Time rolling aus SNAPSHOT
        uint32_t submit_ntime = batch_job.ntime + (g_batch_counter % 15);

        // -------- Prehash (BLAKE3) nur für workN = chunk --------
        cl_uint workN = chunk;
        if (!have_batch) workN = 1;
        for (cl_uint i = 0; i < workN; i++) {
            uint32_t nonce = nonce_base + i;
            uint8_t header[80];
            build_header_le(&batch_job, prevhash_le, merkleroot_le,
                            submit_ntime, batch_job.nbits, nonce, header);
            blake3_hash32(header, 80, &h_prehash[i * 32]);
        }
        nonce_base += workN;

        // -------- GPU Launch: nur workN Bytes schieben/lesen --------
        err = clEnqueueWriteBuffer(q, d_prehash, CL_FALSE, 0, 32u * workN, h_prehash, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "clEnqueueWriteBuffer(prehash)=%d\n", err); return 1; }

        uint32_t blocks_per_lane = M_COST_KB;
        size_t global = (size_t)workN;

        uint64_t t0 = mono_ms();

        if (have_batch) {
            if (!(set_batch_kernel_args_adaptive(k_batch, d_prehash, d_mem, d_out, workN, blocks_per_lane) == CL_SUCCESS &&
                  clEnqueueNDRangeKernel(q, k_batch, 1, NULL, &global, NULL, 0, NULL, NULL) == CL_SUCCESS)) {
                fprintf(stderr, "clEnqueueNDRangeKernel(batch) failed, trying single...\n");
                // Fallback Single
                if (!k_single) { k_single = clCreateKernel(prog, "argon2d_core", &err);
                    if (err != CL_SUCCESS) { fprintf(stderr, "Fallback kernel create failed: %d\n", err); return 1; } }
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
                if (err != CL_SUCCESS || clEnqueueNDRangeKernel(q, k_single, 1, NULL, &(size_t){1}, NULL, 0, NULL, NULL) != CL_SUCCESS) {
                    fprintf(stderr, "clEnqueueNDRangeKernel(single) failed\n");
                    return 1;
                }
            }
        } else {
            if (!k_single) { k_single = clCreateKernel(prog, "argon2d_core", &err);
                if (err != CL_SUCCESS) { fprintf(stderr, "No usable kernel found\n"); return 1; } }
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
            if (err != CL_SUCCESS || clEnqueueNDRangeKernel(q, k_single, 1, NULL, &(size_t){1}, NULL, 0, NULL, NULL) != CL_SUCCESS) {
                fprintf(stderr, "clEnqueueNDRangeKernel(single) failed\n");
                return 1;
            }
        }

        clFinish(q);

        err = clEnqueueReadBuffer(q, d_out, CL_TRUE, 0, 32u * workN, h_out, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "clEnqueueReadBuffer(out)=%d\n", err); return 1; }

        uint64_t t1 = mono_ms();
        double batch_ms = (double)(t1 - t0);
        hashes_window += workN;

        // Clean: NUR verwerfen, wenn während des Launches clean kam
        if (!clean_at_start && g_last_notify_clean && g_job_gen != batch_gen) {
            goto after_submit;
        }

        // Final SHA3 + Target + Submit
        cl_uint loopN = workN;
        for (cl_uint i = 0; i < loopN; i++) {
            uint8_t final_hash[32];
            sha3_256_once(&h_out[i * 32], 32, final_hash);
            update_statistics(final_hash);

            if (hash_meets_target_be(final_hash, g_share_target_be)) {
                uint32_t nonce = (nonce_base - workN) + i;
                if (g_debug || g_accepted_shares + g_rejected_shares < 5) {
                    printf("\nSHARE FOUND! job=%s nonce=%08x", J.job_id, nonce);
                    fflush(stdout);
                }
                // exakt derselbe Snapshot + derselbe en2_hex + %08x-ntime/nonce
                stratum_submit(&S, &batch_job, en2_hex, submit_ntime, nonce);
            }
        }

after_submit:
        g_batch_counter++;


        if (g_debug && !debug_shown_for_job) {
            printf("Debug: job=%s best=%u zeros diff=%.6f batch=%u\n",
                   J.job_id, g_best_leading_zeros, g_share_diff, g_batch_counter);
            debug_shown_for_job = 1;
        }

        // Hashrate every ~5s
        uint64_t now = mono_ms();
        if (now - t_rate >= 5000) {
            double secs = (now - t_rate) / 1000.0;
            double rate = (double)hashes_window / secs;
            printf("Hashrate: %.1f H/s | Batch: %.1fms | Job: %s | Shares: %u/%u\r",
                   rate, batch_ms, J.job_id[0] ? J.job_id : "-", g_accepted_shares, g_rejected_shares);
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