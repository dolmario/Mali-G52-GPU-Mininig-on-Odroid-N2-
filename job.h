// job.h
#pragma once
#include "net_stratum.h"
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

size_t hex2bin(const char *hex, uint8_t *out, size_t max);
void   bin2hex(const uint8_t *in, size_t len, char *out, size_t outsz);
void   flip32(uint8_t *p);                      // 32B LE <-> BE
uint32_t target_from_nbits(uint32_t nbits, uint8_t target[32]);

bool build_coinbase(const stratum_ctx_t *s, const stratum_job_t *job,
                    const char *extranonce2_hex,
                    uint8_t *coinbase, size_t *coinbase_len);

bool build_block_header(const stratum_job_t *job, const uint8_t merkle_root[32],
                        uint8_t header[80], uint32_t nonce_le);
