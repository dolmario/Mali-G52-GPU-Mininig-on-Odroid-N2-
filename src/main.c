#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include "blake3.h"



// === Helper: JSON String Extractor ===
static int get_next_quoted(const char **pp, const char *end, char *out, size_t cap) {
    const char *p = *pp;
    const char *q1 = NULL, *q2 = NULL;

    // erstes "
    for (; p < end; p++) { if (*p == '"') { q1 = p; break; } }
    if (!q1 || q1 >= end) return 0;

    // zweites "
    for (p = q1 + 1; p < end; p++) { if (*p == '"') { q2 = p; break; } }
    if (!q2 || q2 > end) return 0;

    size_t L = (size_t)(q2 - (q1 + 1));
    if (L >= cap) L = cap - 1;
    memcpy(out, q1 + 1, L);
    out[L] = 0;

    *pp = q2 + 1;  // weiter hinter dem String
    return 1;
}



// ---------- Stratum + Job Strukturen ----------
typedef struct {
    char host[256];
    int  port;
    int  sock;
    char wallet[128];
    char pass[128];
    char extranonce1[64];     // hex
    uint32_t extranonce2_size;
} stratum_ctx_t;

typedef struct {
    char     job_id[128];
    char     prevhash_hex[65];      // BE hex
    char     coinb1_hex[1024];
    char     coinb2_hex[1024];
    char     merkle_hex[16][65];    // BE hex strings
    int      merkle_count;
    uint32_t version;               // parsed from hex
    uint32_t nbits;                 // parsed
    uint32_t ntime;                 // parsed
    int      clean;
} stratum_job_t;

// ---------- Utils ----------
static int hex2bin(const char *hex, uint8_t *out, size_t outlen) {
    for (size_t i=0;i<outlen;i++) {
        unsigned v;
        if (sscanf(hex + 2*i, "%2x", &v) != 1) return 0;
        out[i] = (uint8_t)v;
    }
    return 1;
}
static void flip32(uint8_t *p) { for (int i=0;i<16;i++){ uint8_t t=p[i]; p[i]=p[31-i]; p[31-i]=t; } }

static void sha256_once(const uint8_t *in, size_t len, uint8_t out[32]) {
    SHA256_CTX c; SHA256_Init(&c); SHA256_Update(&c, in, len); SHA256_Final(out, &c);
}
static void double_sha256(const uint8_t *in, size_t len, uint8_t out[32]) {
    uint8_t t[32]; sha256_once(in,len,t); sha256_once(t,32,out);
}
static void blake3_hash32(const uint8_t *in, size_t inlen, uint8_t out[32]) {
    blake3_hasher h; blake3_hasher_init(&h); blake3_hasher_update(&h,in,inlen); blake3_hasher_finalize(&h,out,32);
}
static void sha3_256(const uint8_t *in, size_t inlen, uint8_t out[32]) {
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha3_256(), NULL);
    EVP_DigestUpdate(ctx, in, inlen);
    unsigned olen=0; EVP_DigestFinal_ex(ctx, out, &olen); EVP_MD_CTX_free(ctx);
}

// nbits → 32B Target (big-endian) für direkten BE-Vergleich
static void target_from_nbits(uint32_t nbits, uint8_t target[32]) {
    memset(target, 0, 32);
    uint32_t exp  = nbits >> 24;
    uint32_t mant = nbits & 0xFFFFFFu;
    if (exp <= 3) {
        mant >>= 8*(3-exp);
        target[29] = (mant >> 16) & 0xFF;
        target[30] = (mant >> 8)  & 0xFF;
        target[31] = mant & 0xFF;
    } else {
        int idx = 32 - exp;
        if (idx < 0) idx = 0;
        if (idx + 3 <= 32) {
            target[idx]   = (mant >> 16) & 0xFF;
            target[idx+1] = (mant >> 8)  & 0xFF;
            target[idx+2] = mant & 0xFF;
        }
    }
}

// ---------- Stratum Minimal ----------
static int sock_connect(const char *host, int port) {
    struct addrinfo hints = {0}, *res=NULL,*p=NULL; char portstr[16];
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
    freeaddrinfo(res); return s;
}
static int send_line(int s, const char *line){ size_t L=strlen(line),o=0; while(o<L){ ssize_t n=send(s,line+o,L-o,0); if(n<=0) return 0; o+=n; } return 1; }
static int recv_line(int s, char *buf, size_t cap){
    size_t o=0; while(o+1<cap){ char c; ssize_t n=recv(s,&c,1,0); if(n<=0) return -1; buf[o++]=c; if(c=='\n') break; }
    buf[o]=0; return (int)o;
}

static int stratum_connect(stratum_ctx_t *C, const char *host,int port,const char *user,const char *pass){
    memset(C,0,sizeof *C);
    snprintf(C->host,sizeof C->host,"%s",host); C->port=port;
    snprintf(C->wallet,sizeof C->wallet,"%s",user);
    snprintf(C->pass,sizeof C->pass,"%s",pass);
    C->sock = sock_connect(host,port); if(C->sock<0) return 0;

    // subscribe
    char sub[256];
    snprintf(sub,sizeof sub,"{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"rin-ocl/0.1\"]}\n");
    if(!send_line(C->sock,sub)) return 0;

    // parse result: [sub_details, extranonce1, extranonce2_size]
    char line[8192];
    while(1){
        if(recv_line(C->sock,line,sizeof line)<=0) return 0;
        if(strstr(line,"\"result\"")){
            // extract extranonce1
            char *lb=strchr(line,'['); char *rb=strrchr(line,']');
            if(!lb||!rb) break;
            // find first quoted after sub_details (skip first [] by finding \", then next \")
            // crude: find second quoted string overall
            char *q=strchr(lb,'\"'); if(!q) break;
            q=strchr(q+1,'\"'); if(!q) break; // end of first quoted
            char *q1=strchr(q+1,'\"'); if(!q1) break;
            char *q2=strchr(q1+1,'\"'); if(!q2) break;
            size_t len = (size_t)(q2-(q1+1)); if(len>sizeof C->extranonce1-1) len=sizeof C->extranonce1-1;
            memcpy(C->extranonce1,q1+1,len); C->extranonce1[len]=0;

            // extranonce2_size: last number before ']'
            char *lc = strrchr(lb, ','); if(lc && lc<rb) C->extranonce2_size = (uint32_t)strtoul(lc+1,NULL,10);
            if(C->extranonce2_size==0) C->extranonce2_size=4;
            break;
        }
    }
    // authorize
    char auth[512];
    snprintf(auth,sizeof auth,"{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\"%s\",\"%s\"]}\n",C->wallet,C->pass);
    if(!send_line(C->sock,auth)) return 0;
    // read ack
    while(1){
        if(recv_line(C->sock,line,sizeof line)<=0) return 0;
        if(strstr(line,"\"result\":true")) break;
        if(strstr(line,"\"mining.notify\"")) break; // job may already come
    }
    printf("Stratum: extranonce1=%s, extranonce2_size=%u\n", C->extranonce1, C->extranonce2_size);
    return 1;
}

static int stratum_parse_notify(const char *line, stratum_job_t *J){
    if (!strstr(line, "\"mining.notify\"")) return 0;
    memset(J, 0, sizeof(*J));

    // params suchen
    const char *p_params = strstr(line, "\"params\"");
    if (!p_params) return 0;
    const char *lb = strchr(p_params, '[');
    const char *rb = lb ? strrchr(p_params, ']') : NULL;
    if (!lb || !rb || rb <= lb) return 0;

    // Lesezeiger innerhalb params
    const char *p = lb + 1;

    // 1: job_id
    if (!get_next_quoted(&p, rb, J->job_id, sizeof(J->job_id))) return 0;

    // 2: prevhash (BE hex)
    if (!get_next_quoted(&p, rb, J->prevhash_hex, sizeof(J->prevhash_hex))) return 0;

    // 3: coinb1 (hex)
    if (!get_next_quoted(&p, rb, J->coinb1_hex, sizeof(J->coinb1_hex))) return 0;

    // 4: coinb2 (hex)
    if (!get_next_quoted(&p, rb, J->coinb2_hex, sizeof(J->coinb2_hex))) return 0;

    // 5: merkle[] (Array aus Strings)
    // finde das nächste '[' und sein korrespondierendes ']'
    const char *m_lb = NULL, *m_rb = NULL;
    {
        const char *scan = p;
        for (; scan < rb; scan++) { if (*scan == '[') { m_lb = scan; break; } }
        if (m_lb) {
            for (scan = m_lb + 1; scan < rb; scan++) { if (*scan == ']') { m_rb = scan; break; } }
        }
    }
    J->merkle_count = 0;
    if (m_lb && m_rb && m_rb > m_lb) {
        const char *mp = m_lb + 1;
        while (J->merkle_count < 16) {
            char tmp[65];
            if (!get_next_quoted(&mp, m_rb, tmp, sizeof(tmp))) break;
            snprintf(J->merkle_hex[J->merkle_count], sizeof(J->merkle_hex[0]), "%s", tmp);
            J->merkle_count++;
        }
        p = m_rb + 1; // nach dem Merkle-Array weitermachen
    }

    // 6: version (quoted hex)
    {
        char vhex[16] = {0};
        if (!get_next_quoted(&p, rb, vhex, sizeof(vhex))) return 0;
        sscanf(vhex, "%x", &J->version);
    }

    // 7: nbits (quoted hex)
    {
        char nbhex[16] = {0};
        if (!get_next_quoted(&p, rb, nbhex, sizeof(nbhex))) return 0;
        sscanf(nbhex, "%x", &J->nbits);
    }

    // 8: ntime (quoted hex)
    {
        char nth[16] = {0};
        if (!get_next_quoted(&p, rb, nth, sizeof(nth))) return 0;
        sscanf(nth, "%x", &J->ntime);
    }

    // 9: clean_jobs (bool) – grob erkennen
    // wir schauen nur, ob irgendwo hinterher "true" / "false" steht
    J->clean = 0;
    {
        const char *boolp = strstr(p, "true");
        if (boolp && boolp < rb) J->clean = 1;
    }

    return 1;
}


static int stratum_get_job(stratum_ctx_t *C, stratum_job_t *J){
    char line[16384];
    while(1){
        int n = recv_line(C->sock,line,sizeof line);
        if(n<=0) return 0;
        if(stratum_parse_notify(line,J)) return 1;
        // ignore other
    }
}

static int stratum_submit(stratum_ctx_t *C, const stratum_job_t *J,
                          const char *extranonce2_hex, uint32_t ntime_le, uint32_t nonce_le){
    char ntime_hex[9], nonce_hex[9];
    snprintf(ntime_hex,sizeof ntime_hex,"%08x", ntime_le);
    snprintf(nonce_hex,sizeof nonce_hex,"%08x", nonce_le);
    char req[512];
    snprintf(req,sizeof req,
        "{\"id\":4,\"method\":\"mining.submit\",\"params\":[\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"]}\n",
        C->wallet, J->job_id, extranonce2_hex, ntime_hex, nonce_hex);
    return send_line(C->sock,req);
}

// ---------- Merkle-Root aus coinbase + Branches ----------
static void build_merkle_root_le(const uint8_t coinbase_hash_be[32],
                                 char merkle_hex[][65], int mcount,
                                 uint8_t out_le[32])
{
    // Start: coinbase_hash in LE wandeln
    uint8_t h_le[32];
    for(int i=0;i<32;i++) h_le[i] = coinbase_hash_be[31-i];

    for(int i=0;i<mcount;i++){
        uint8_t br_be[32], br_le[32], cat[64], dh[32];
        if(!hex2bin(merkle_hex[i], br_be, 32)) memset(br_be,0,32);
        for(int k=0;k<32;k++) br_le[k]=br_be[31-k];
        memcpy(cat, h_le, 32);
        memcpy(cat+32, br_le, 32);
        double_sha256(cat, 64, dh);      // dh ist BE
        for(int k=0;k<32;k++) h_le[k] = dh[31-k]; // wieder LE für nächste Runde
    }
    memcpy(out_le, h_le, 32);
}

// ---------- Header bauen ----------
static void build_header_le(const stratum_job_t *J, const uint8_t prevhash_le[32],
                            const uint8_t merkleroot_le[32], uint32_t ntime, uint32_t nbits,
                            uint32_t nonce, uint8_t out80[80])
{
    memset(out80,0,80);
    memcpy(out80+0,  &J->version, 4);      // LE
    memcpy(out80+4,  prevhash_le, 32);     // LE bytes
    memcpy(out80+36, merkleroot_le, 32);   // LE bytes
    memcpy(out80+68, &ntime, 4);           // LE
    memcpy(out80+72, &nbits, 4);           // LE
    memcpy(out80+76, &nonce, 4);           // LE
}

// ---------- OpenCL Helper ----------
static char* load_kernel_source(const char *path) {
    FILE *f=fopen(path,"rb"); if(!f){ perror("kernel"); exit(1); }
    fseek(f,0,SEEK_END); long sz=ftell(f); rewind(f);
    char *src=(char*)malloc(sz+1);
    fread(src,1,sz,f); src[sz]=0; fclose(f); return src;
}

// ================== MAIN ==================
int main(int argc, char **argv) {
    // Pool-Konfiguration (RinHash @ Zergpool)
    static const char *POOL = "rinhash.eu.mine.zergpool.com";
    static const int   PORT = 7148;
    static const char *WAL  = "DTXoRQ7Zpw3FVRW2DWkLrM9Skj4Gp9SeSj";
    static const char *PASS = "c=DOGE,ID=n2plus";



    // --- OpenCL Setup ---
    uint32_t m_cost_kb = 64*1024; // Default 64MB
    if (argc>1){ int mb=atoi(argv[1]); if(mb>=8 && mb<=256) m_cost_kb=mb*1024; }
    printf("Using %u MB memory\n", m_cost_kb/1024);

    cl_int err; cl_platform_id platform; cl_device_id device;
    clGetPlatformIDs(1,&platform,NULL);
    err = clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,1,&device,NULL);
    if(err!=CL_SUCCESS){ fprintf(stderr,"No GPU device found\n"); return 1; }

    char devname[256]; clGetDeviceInfo(device,CL_DEVICE_NAME,sizeof devname,devname,NULL);
    printf("Device: %s\n", devname);

    cl_context ctx = clCreateContext(NULL,1,&device,NULL,NULL,&err);
    cl_command_queue q = clCreateCommandQueue(ctx,device,0,&err);
    if(err!=CL_SUCCESS){ fprintf(stderr,"clCreateCommandQueue: %d\n",err); return 1; }

    char *ksrc = load_kernel_source("rinhash_argon2d.cl");
    cl_program prog = clCreateProgramWithSource(ctx,1,(const char**)&ksrc,NULL,&err);
    err = clBuildProgram(prog,1,&device,"-cl-std=CL1.2",NULL,NULL);
    if(err!=CL_SUCCESS){
        size_t L; clGetProgramBuildInfo(prog,device,CL_PROGRAM_BUILD_LOG,0,NULL,&L);
        char *log=(char*)malloc(L+1);
        clGetProgramBuildInfo(prog,device,CL_PROGRAM_BUILD_LOG,L,log,NULL); log[L]=0;
        fprintf(stderr,"Build failed:\n%s\n",log); free(log); return 1;
    }
    cl_kernel krn = clCreateKernel(prog,"argon2d_core",&err);

    size_t m_bytes = (size_t)m_cost_kb * 1024;
    cl_mem d_mem    = clCreateBuffer(ctx,CL_MEM_READ_WRITE,m_bytes,NULL,&err);
    cl_mem d_phash  = clCreateBuffer(ctx,CL_MEM_READ_ONLY, 32,NULL,&err);
    cl_mem d_out    = clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,32,NULL,&err);

    uint32_t t_cost=2, lanes=1;
    clSetKernelArg(krn,0,sizeof(d_phash),&d_phash);
    clSetKernelArg(krn,1,sizeof(d_mem),  &d_mem);
    clSetKernelArg(krn,2,sizeof(uint32_t),&m_cost_kb);
    clSetKernelArg(krn,3,sizeof(uint32_t),&t_cost);
    clSetKernelArg(krn,4,sizeof(uint32_t),&lanes);
    clSetKernelArg(krn,5,sizeof(d_out),  &d_out);

    // --- Stratum ---
    stratum_ctx_t S;
    if(!stratum_connect(&S,POOL,PORT,WAL,PASS)){ fprintf(stderr,"Stratum connect failed\n"); return 1; }
    printf("Connected to %s:%d\n", POOL, PORT);

    // prevhash LE vorbereiten pro Job
    uint8_t prevhash_le[32], merkleroot_le[32];

    // --- Mining Loop ---
    while(1){
        stratum_job_t J;
        if(!stratum_get_job(&S,&J)) { fprintf(stderr,"get_job failed\n"); sleep(2); continue; }

        // prevhash hex (BE) -> bin -> LE
        hex2bin(J.prevhash_hex, prevhash_le, 32); flip32(prevhash_le);

        // extranonce2 Schleife
        uint32_t en2_counter = 1;
        for(;; en2_counter++){
            // extranonce1 (hex vom Pool) -> bin
            uint8_t en1[64]; size_t en1b = strlen(S.extranonce1)/2; if(en1b>64) en1b=64;
            hex2bin(S.extranonce1, en1, en1b);
            // extranonce2 -> hex mit richtiger Länge (bytes = extranonce2_size)
            char en2_hex[64]; int en2_bytes = (int)S.extranonce2_size;
            {
                // hex-string mit führenden Nullen
                char tmp[64]; snprintf(tmp,sizeof tmp,"%0*x", en2_bytes*2, en2_counter);
                snprintf(en2_hex,sizeof en2_hex,"%s", tmp);
            }
            uint8_t en2[64]; hex2bin(en2_hex, en2, en2_bytes);

            // coinbase = coinb1 + en1 + en2 + coinb2
            uint8_t coinb1[1024], coinb2[1024];
            size_t cb1 = strlen(J.coinb1_hex)/2, cb2 = strlen(J.coinb2_hex)/2;
            hex2bin(J.coinb1_hex, coinb1, cb1);
            hex2bin(J.coinb2_hex, coinb2, cb2);
            uint8_t coinbase[4096]; size_t off=0;
            memcpy(coinbase+off,coinb1,cb1); off+=cb1;
            memcpy(coinbase+off,en1,en1b);   off+=en1b;
            memcpy(coinbase+off,en2,en2_bytes); off+=en2_bytes;
            memcpy(coinbase+off,coinb2,cb2); off+=cb2;

            // coinbase hash (dSHA256, BE)
            uint8_t coinbase_hash_be[32]; double_sha256(coinbase,off,coinbase_hash_be);

            // Merkle-Root in LE bauen
            build_merkle_root_le(coinbase_hash_be, J.merkle_hex, J.merkle_count, merkleroot_le);

            // Nonce Fenster
            uint8_t target_be[32]; target_from_nbits(J.nbits, target_be);

            // kleine Statistik
            time_t t0=time(NULL); unsigned hashes=0;

            for(uint32_t nonce=0; nonce<200000; nonce++){
                uint8_t header[80];
                build_header_le(&J, prevhash_le, merkleroot_le, J.ntime, J.nbits, nonce, header);

                // RinHash
                uint8_t ph[32]; blake3_hash32(header,80,ph);

                clEnqueueWriteBuffer(q,d_phash,CL_TRUE,0,32,ph,0,NULL,NULL);
                size_t G=1;
                err = clEnqueueNDRangeKernel(q,krn,1,NULL,&G,NULL,0,NULL,NULL);
                if(err!=CL_SUCCESS){ fprintf(stderr,"Kernel failed: %d\n",err); break; }
                clFinish(q);

                uint8_t argon_out[32]; clEnqueueReadBuffer(q,d_out,CL_TRUE,0,32,argon_out,0,NULL,NULL);
                uint8_t final_be[32]; sha3_256(argon_out,32,final_be);

                // Vergleich BE: final < target ?
                int less=0, greater=0;
                for(int i=0;i<32;i++){
                    if(final_be[i] < target_be[i]) { less=1; break; }
                    if(final_be[i] > target_be[i]) { greater=1; break; }
                }
                if(less && !greater){
                    // Share submit
                    char en2_hex_full[64]; snprintf(en2_hex_full,sizeof en2_hex_full,"%0*x", en2_bytes*2, en2_counter);
                    if(stratum_submit(&S,&J,en2_hex_full,J.ntime,nonce)){
                        printf("Share submitted: job=%s en2=%s nonce=%08x\n", J.job_id, en2_hex_full, nonce);
                    } else {
                        fprintf(stderr,"Submit failed\n");
                    }
                }

                hashes++;
                if((hashes % 5000)==0){
                    time_t dt=time(NULL)-t0; if(dt==0) dt=1;
                    printf("Speed ~ %u H/s | job %s\n", hashes/(unsigned)dt, J.job_id);
                }
            }
            // bei clean-jobs brich ab auf nächstes notify
            // (hier simpel: wir laufen weiter, bis neuer Job kommt)
        }
    }

    // Cleanup (theoretisch nie erreicht)
    clReleaseMemObject(d_mem); clReleaseMemObject(d_phash); clReleaseMemObject(d_out);
    clReleaseKernel(krn); clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx); free(ksrc);
    return 0;
}
