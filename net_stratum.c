#define _GNU_SOURCE
#include "net_stratum.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>

static int sock_connect(const char *host, int port) {
    struct addrinfo hints = {0}, *res = NULL, *p = NULL;
    char portstr[16]; snprintf(portstr, sizeof portstr, "%d", port);
    hints.ai_socktype = SOCK_STREAM; hints.ai_family = AF_UNSPEC;
    if (getaddrinfo(host, portstr, &hints, &res) != 0) return -1;
    int s = -1;
    for (p = res; p; p = p->ai_next) {
        s = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (s < 0) continue;
        if (connect(s, p->ai_addr, p->ai_addrlen) == 0) break;
        close(s); s = -1;
    }
    freeaddrinfo(res);
    return s;
}

static bool send_line(int sock, const char *line) {
    size_t len = strlen(line);
    size_t off = 0;
    while (off < len) {
        ssize_t n = send(sock, line + off, len - off, 0);
        if (n <= 0) return false;
        off += (size_t)n;
    }
    return true;
}

static int recv_line(int sock, char *buf, size_t bufsz) {
    size_t off = 0;
    while (off + 1 < bufsz) {
        char c;
        ssize_t n = recv(sock, &c, 1, 0);
        if (n <= 0) return -1;
        buf[off++] = c;
        if (c == '\n') break;
    }
    buf[off] = 0;
    return (int)off;
}

static const char* json_find(const char *s, const char *key) {
    // naive search: "\"key\""
    static char patt[128];
    snprintf(patt, sizeof patt, "\"%s\"", key);
    return strstr(s, patt);
}

static int hex_to_u32(const char *hex) {
    // parses small hex (like "1b0404cb")
    return (int)strtoul(hex, NULL, 16);
}

bool stratum_connect(stratum_ctx_t *s, const char *host, int port, const char *user, const char *pass) {
    memset(s, 0, sizeof *s);
    s->sock = sock_connect(host, port);
    if (s->sock < 0) return false;
    snprintf(s->client_addr, sizeof s->client_addr, "%s:%d", host, port);
    snprintf(s->user, sizeof s->user, "%s", user);
    snprintf(s->pass, sizeof s->pass, "%s", pass);

    // subscribe
    char sub[256];
    snprintf(sub, sizeof sub, "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"rin-ocl/0.1\",\"%s\"]}\n", s->client_addr);
    if (!send_line(s->sock, sub)) return false;

    // read lines until we get extranonce1
    char line[4096];
    while (1) {
        if (recv_line(s->sock, line, sizeof line) <= 0) return false;
        if (strstr(line, "\"mining.notify\"")) {
            // will be handled later by get_job
            // keep this line in a small buffer? simpler: just continue, we will request a new notify after auth
        }
        if (json_find(line, "result")) {
            // Try to extract extranonce1 and size (Zergpool returns: result: [subscription_details, extranonce1, extranonce2_size])
            // crude parse: find array after "result":
            char *r = strstr(line, "\"result\"");
            if (!r) continue;
            char *lb = strchr(r, '['); if (!lb) continue;
            char *rb = strchr(lb, ']'); if (!rb) continue;
            // tokenise by comma
            // we expect: [something, "extranonce1", extranonce2_size]
            char *q = strchr(lb, '"');
            if (!q || q > rb) continue;
            char *q2 = strchr(q+1, '"');
            if (!q2 || q2 > rb) continue;
            size_t elen = (size_t)(q2 - (q+1));
            if (elen >= sizeof s->extranonce1) elen = sizeof s->extranonce1 - 1;
            memcpy(s->extranonce1, q+1, elen); s->extranonce1[elen] = 0;

            // find extranonce2_size: search last comma then number
            char *last_comma = strrchr(lb, ',');
            if (last_comma && last_comma < rb) {
                s->extranonce2_size = (uint32_t)strtoul(last_comma+1, NULL, 10);
            } else {
                s->extranonce2_size = 4; // default fallback
            }
            break;
        }
    }

    // authorize
    char auth[512];
    snprintf(auth, sizeof auth, "{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\"%s\",\"%s\"]}\n", s->user, s->pass);
    if (!send_line(s->sock, auth)) return false;

    // read auth response (ignore result content)
    while (1) {
        if (recv_line(s->sock, line, sizeof line) <= 0) return false;
        if (strstr(line, "\"result\":true")) break;
        if (strstr(line, "\"mining.notify\"")) {
            // a job may already come; ignore here, get_job() will fetch a fresh one
            break;
        }
    }
    return true;
}

static void hex_copy(char *dst, const char *src, size_t maxlen) {
    size_t n = strnlen(src, maxlen-1);
    memcpy(dst, src, n); dst[n] = 0;
}

bool stratum_get_job(stratum_ctx_t *s, stratum_job_t *job) {
    memset(job, 0, sizeof *job);
    char line[16384];
    while (1) {
        if (recv_line(s->sock, line, sizeof line) <= 0) return false;
        if (!strstr(line, "\"mining.notify\"")) continue;

        // Parse fields from params:
        // "params":[job_id, prevhash, coinb1, coinb2, merkle, version, nbits, ntime, clean_jobs]
        char *pp = strstr(line, "\"params\"");
        if (!pp) continue;
        char *lb = strchr(pp, '['); if (!lb) continue;
        char *rb = strrchr(lb, ']'); if (!rb) continue;

        // naive split by quotes:
        // 0: job_id, 1: prevhash, 2: coinb1, 3: coinb2, 4.. merkle array (we’ll scan), then version/nbits/ntime as hex strings
        char *p = lb;
        // job_id
        p = strchr(p, '"'); if (!p || p>rb) continue; char *e = strchr(p+1, '"'); if (!e) continue;
        *e = 0; hex_copy(job->job_id, p+1, sizeof job->job_id); *e='"'; p=e+1;

        // prevhash
        p = strchr(p, '"'); if (!p||p>rb) continue; e = strchr(p+1, '"'); if (!e) continue;
        *e = 0; hex_copy(job->prevhash_hex, p+1, sizeof job->prevhash_hex); *e='"'; p=e+1;

        // coinb1
        p = strchr(p, '"'); if (!p||p>rb) continue; e = strchr(p+1, '"'); if (!e) continue;
        *e = 0; hex_copy(job->coinb1_hex, p+1, sizeof job->coinb1_hex); *e='"'; p=e+1;

        // coinb2
        p = strchr(p, '"'); if (!p||p>rb) continue; e = strchr(p+1, '"'); if (!e) continue;
        *e = 0; hex_copy(job->coinb2_hex, p+1, sizeof job->coinb2_hex); *e='"'; p=e+1;

        // merkle (array of hex strings) until we hit version (non-quoted hex or quoted; zergpool usually array of strings, then 3 strings)
        job->merkle_count = 0;
        while (1) {
            char *nextq = strchr(p, '"');
            if (!nextq || nextq > rb) break;
            char *nextq2 = strchr(nextq+1, '"'); if (!nextq2) break;
            // lookahead: if following char after nextq2 is ',' or ']' then it's likely a merkle entry (64 hex chars)
            size_t len = (size_t)(nextq2 - (nextq + 1));
            if (len >= 2 && len <= 64 && job->merkle_count < 16) {
                // take as merkle branch
                char tmp[70]; memset(tmp,0,sizeof tmp);
                memcpy(tmp, nextq+1, len); tmp[len]=0;
                hex_copy(job->merkle[job->merkle_count], tmp, sizeof job->merkle[0]);
                job->merkle_count++;
                p = nextq2 + 1;
            } else {
                break;
            }
        }

        // version, nbits, ntime are hex strings following merkle array.
        // Extract three quoted strings after merkle loop:
        for (int i=0;i<3;i++) {
            char *q1 = strchr(p, '"'); if (!q1||q1>rb) break;
            char *q2 = strchr(q1+1, '"'); if (!q2) break;
            char tmp[32]; size_t L = (size_t)(q2-(q1+1)); if (L>31) L=31;
            memcpy(tmp, q1+1, L); tmp[L]=0;
            if (i==0) job->version = (uint32_t)strtoul(tmp, NULL, 16);
            if (i==1) job->nbits   = (uint32_t)strtoul(tmp, NULL, 16);
            if (i==2) job->ntime   = (uint32_t)strtoul(tmp, NULL, 16);
            p = q2+1;
        }

        // clean flag
        job->clean = strstr(line, "true") && !strstr(line, "\"result\"");

        return true;
    }
}

bool stratum_submit_share(stratum_ctx_t *s, const stratum_job_t *job,
                          const char *extranonce2_hex, uint32_t nonce_le, uint32_t ntime_le)
{
    // Share submit: ["wallet.worker","job_id","extranonce2","ntime","nonce"]
    char req[512];
    char nonce_hex[9]; snprintf(nonce_hex, sizeof nonce_hex, "%08x", nonce_le);
    char ntime_hex[9]; snprintf(ntime_hex, sizeof ntime_hex, "%08x", ntime_le);

    snprintf(req, sizeof req,
        "{\"id\":4,\"method\":\"mining.submit\",\"params\":[\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"]}\n",
        s->user, job->job_id, extranonce2_hex, ntime_hex, nonce_hex);

    return send_line(s->sock, req);
}

void stratum_close(stratum_ctx_t *s) {
    if (s->sock >= 0) close(s->sock);
    s->sock = -1;
}
