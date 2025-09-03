#pragma once
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    int sock;
    char extranonce1[64];   // hex
    uint32_t extranonce2_size;
    char subscription_id[64];
    char client_addr[128];  // e.g. "ghostrider.eu.mine.zergpool.com:5354"
    char user[128];         // wallet.worker or wallet
    char pass[128];         // c=DOGE,ID=n2plus
} stratum_ctx_t;

typedef struct {
    char job_id[128];
    char prevhash_hex[65];      // 32 bytes hex LE from pool (will convert)
    char coinb1_hex[256];
    char coinb2_hex[256];
    char merkle[16][65];        // up to 16 merkle branches (hex)
    int  merkle_count;
    uint32_t version;           // hex from pool, parse
    uint32_t nbits;             // compact target
    uint32_t ntime;             // network time
    bool clean;
} stratum_job_t;

bool stratum_connect(stratum_ctx_t *s, const char *host, int port, const char *user, const char *pass);
bool stratum_get_job(stratum_ctx_t *s, stratum_job_t *job);   // blocks until a job or ping
bool stratum_submit_share(stratum_ctx_t *s, const stratum_job_t *job,
                          const char *extranonce2_hex, uint32_t nonce_le, uint32_t ntime_le);
void stratum_close(stratum_ctx_t *s);
