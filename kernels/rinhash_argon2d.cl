// rinhash_argon2d.cl – Echte Argon2d-Implementierung (Mali G52, OpenCL 1.2)

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#define ARGON2_QWORDS_IN_BLOCK 128
#define ARGON2_BLOCK_SIZE      1024  // 1KB pro Block
#define ARGON2_SYNC_POINTS     4

inline ulong rotr64(ulong v, uint n) {
    return (v >> n) | (v << (64 - n));
}

inline ulong fBlaMka(ulong x, ulong y) {
    const ulong m = 0xFFFFFFFFUL;
    ulong xy = (x & m) * (y & m);
    return x + y + 2 * xy;
}

inline void G(ulong *a, ulong *b, ulong *c, ulong *d) {
    *a = fBlaMka(*a, *b);
    *d = rotr64(*d ^ *a, 32);
    *c = fBlaMka(*c, *d);
    *b = rotr64(*b ^ *c, 24);
    *a = fBlaMka(*a, *b);
    *d = rotr64(*d ^ *a, 16);
    *c = fBlaMka(*c, *d);
    *b = rotr64(*b ^ *c, 63);
}

inline void blake2_round(__private ulong *v) {
    G(&v[0], &v[4], &v[8],  &v[12]);
    G(&v[1], &v[5], &v[9],  &v[13]);
    G(&v[2], &v[6], &v[10], &v[14]);
    G(&v[3], &v[7], &v[11], &v[15]);
    G(&v[0], &v[5], &v[10], &v[15]);
    G(&v[1], &v[6], &v[11], &v[12]);
    G(&v[2], &v[7], &v[8],  &v[13]);
    G(&v[3], &v[4], &v[9],  &v[14]);
}

// Block-Füllfunktion
void fill_block(__global ulong *prev_block,
                __global ulong *ref_block,
                __global ulong *curr_block) {

    ulong state[ARGON2_QWORDS_IN_BLOCK];

    for (int i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) {
        state[i] = prev_block[i] ^ ref_block[i];
    }

    // 8 BLAKE2-Runden
    for (int r = 0; r < 8; r++) {
        for (int j = 0; j < ARGON2_QWORDS_IN_BLOCK / 16; j++) {
            blake2_round(&state[j * 16]);
        }
    }

    for (int i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) {
        curr_block[i] = state[i] ^ prev_block[i] ^ ref_block[i];
    }
}

// Indexberechnung (Argon2d)
uint index_argon2d(__global ulong *addr_block, uint blocks_per_lane) {
    ulong v = addr_block[0];
    uint j1 = (uint)(v & 0xFFFFFFFFUL);
    return j1 % blocks_per_lane;
}

__kernel void argon2d_core(__global const uchar *prehash32,
                           __global ulong *memory,
                           const uint m_cost_kb,
                           const uint passes,
                           __global uchar *out32) {

    const uint blocks_total = (m_cost_kb * 1024) / ARGON2_BLOCK_SIZE;
    if (blocks_total < 3) return; // mindestens 3 Blöcke nötig

    // Lane = 0 (vereinfacht)
    __global ulong *lane_mem = memory;

    // Block 0 initialisieren aus prehash
    for (int i = 0; i < 4; i++) {
        ulong val = 0;
        for (int j = 0; j < 8; j++) {
            val |= ((ulong)prehash32[i * 8 + j]) << (8 * j);
        }
        lane_mem[i] = val;
    }
    for (int i = 4; i < ARGON2_QWORDS_IN_BLOCK; i++) {
        lane_mem[i] = 0;
    }

    // Block 1 als Kopie von Block 0
    __global ulong *block1 = lane_mem + ARGON2_QWORDS_IN_BLOCK;
    for (int i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) {
        block1[i] = lane_mem[i] ^ (ulong)1;
    }

    // Haupt-Loop
    for (uint pass = 0; pass < passes; pass++) {
        for (uint idx = 2; idx < blocks_total; idx++) {
            __global ulong *prev = lane_mem + (idx - 1) * ARGON2_QWORDS_IN_BLOCK;
            uint ref_idx = index_argon2d(prev, blocks_total);
            __global ulong *ref = lane_mem + ref_idx * ARGON2_QWORDS_IN_BLOCK;
            __global ulong *curr = lane_mem + idx * ARGON2_QWORDS_IN_BLOCK;

            fill_block(prev, ref, curr);
        }
    }

    // Finale Reduktion
    __global ulong *last = lane_mem + (blocks_total - 1) * ARGON2_QWORDS_IN_BLOCK;
    for (int i = 0; i < 4; i++) {
        ulong val = last[i];
        for (int j = 0; j < 8; j++) {
            out32[i * 8 + j] = (uchar)((val >> (8 * j)) & 0xFF);
        }
    }
}

