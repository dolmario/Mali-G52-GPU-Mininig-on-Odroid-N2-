// job.c
#include "job.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

size_t hex2bin(const char *hex, uint8_t *out, size_t max) {
    size_t n = 0;
    for (; hex[0] && hex[1] && n<max; hex+=2) {
        unsigned v; if (sscanf(hex, "%2x", &v) != 1) break;
        out[n++] = (uint8_t)v;
    }
    return n;
}

void bin2hex(const uint8_t *in, size_t len, char *out, size_t outsz) {
    static const char *h="0123456789abcdef";
    size_t j=0; for (size_t i=0;i<len && j+1<outsz;i++) {
        if (j+2>=outsz) break;
        out[j++] = h[in[i]>>4];
        out[j++] = h[in[i]&15];
    }
    out[j]=0;
}

void flip32(uint8_t *p) {
    for (int i=0;i<16;i++) { uint8_t t=p[i]; p[i]=p[31-i]; p[31-i]=t; }
}

static void dbl_sha256(const uint8_t *in, size_t len, uint8_t out[32]); // (optional, falls du debug willst)

uint32_t target_from_nbits(uint32_t nbits, uint8_t target[32]) {
    memset(target, 0, 32);
    uint32_t exp = nbits >> 24;
    uint32_t mant = nbits & 0xFFFFFFu;
    if (exp <= 3) {
        uint32_t v = mant >> (8*(3-exp));
        target[31] = (uint8_t)(v      & 0xFF);
        target[30] = (uint8_t)((v>>8) & 0xFF);
        target[29] = (uint8_t)((v>>16)& 0xFF);
    } else {
        int i = 32 - exp;
        if (i < 0) i = 0;
        if (i+3 <= 32) {
            target[i]   = (uint8_t)((mant >> 16) & 0xFF);
            target[i+1] = (uint8_t)((mant >> 8)  & 0xFF);
            target[i+2] = (uint8_t)( mant        & 0xFF);
        }
    }
    // target ist big-endian; viele Miner vergleichen BE → ok. Wenn du LE brauchst, flip32(target).
    return 0;
}

bool build_coinbase(const stratum_ctx_t *s, const stratum_job_t *job,
                    const char *extranonce2_hex,
                    uint8_t *coinbase, size_t *coinbase_len)
{
    // coinbase = coinb1 + extranonce1 + extranonce2 + coinb2
    uint8_t cb1[512], cb2[512], en1[64], en2[64];
    size_t n1 = hex2bin(job->coinb1_hex, cb1, sizeof cb1);
    size_t n2 = hex2bin(job->coinb2_hex, cb2, sizeof cb2);
    size_t ne1 = hex2bin(s->extranonce1, en1, sizeof en1);
    size_t ne2 = hex2bin(extranonce2_hex, en2, sizeof en2);

    if (!n1 || !n2 || !ne1) return false;
    size_t off=0;
    memcpy(coinbase+off, cb1, n1); off+=n1;
    memcpy(coinbase+off, en1, ne1); off+=ne1;
    memcpy(coinbase+off, en2, ne2); off+=ne2;
    memcpy(coinbase+off, cb2, n2); off+=n2;
    *coinbase_len = off;
    return true;
}

static void sha256_once(const uint8_t *in, size_t len, uint8_t out[32]) {
    // OpenSSL single SHA256
    // (du nutzt OpenSSL eh schon; alternativ BLAKE3 hier nicht!)
    // Placeholder: implementiere falls du coinbase doppelt SHA256 brauchst.
    // Für Merkle-Root brauchst du dSHA256(leaf || leaf).
    // Du kannst OpenSSL SHA256 hier reinkopieren – ausgelassen, um kurz zu bleiben.
}

bool build_block_header(const stratum_job_t *job, const uint8_t merkle_root[32],
                        uint8_t header[80], uint32_t nonce_le)
{
    // version(4) | prevhash(32) | merkleroot(32) | ntime(4) | nbits(4) | nonce(4)
    memset(header, 0, 80);

    uint32_t v = job->version;
    memcpy(header+0, &v, 4); // little-endian ok

    // prevhash kommt vom Pool als big-endian hex; im Header muss little-endian rein:
    uint8_t prev[32]; hex2bin(job->prevhash_hex, prev, 32);
    flip32(prev);
    memcpy(header+4, prev, 32);

    // merkle_root kommt als BE; Header braucht LE:
    uint8_t mr[32]; memcpy(mr, merkle_root, 32);
    flip32(mr);
    memcpy(header+36, mr, 32);

    uint32_t ntime = job->ntime;
    memcpy(header+68, &ntime, 4);
    uint32_t nbits = job->nbits;
    memcpy(header+72, &nbits, 4);
    memcpy(header+76, &nonce_le, 4);

    return true;
}
