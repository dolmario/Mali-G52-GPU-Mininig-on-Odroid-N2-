// OpenCL 1.2 RinHash Argon2d (angepasst für Mali-G52 mit Chunking)
#define ARGON2_BLOCK_SIZE      1024
#define ARGON2_QWORDS_IN_BLOCK 128
#define ARGON2_SYNC_POINTS     4

inline ulong rotr64(ulong x, uint n) { 
    return (x >> n) | (x << (64 - n)); 
}

inline ulong fBlaMka(ulong x, ulong y) {
    const ulong m = 0xFFFFFFFFUL;
    ulong xy = (x & m) * (y & m);
    return x + y + 2 * xy;
}

inline void G(__private ulong *a, __private ulong *b, __private ulong *c, __private ulong *d) {
    *a = fBlaMka(*a, *b); *d = rotr64(*d ^ *a, 32);
    *c = fBlaMka(*c, *d); *b = rotr64(*b ^ *c, 24);
    *a = fBlaMka(*a, *b); *d = rotr64(*d ^ *a, 16);
    *c = fBlaMka(*c, *d); *b = rotr64(*b ^ *c, 63);
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

// Argon2 Fill Block
inline void fill_block(__global ulong *prev, __global ulong *ref, __global ulong *next, int with_xor) {
    __private ulong state[ARGON2_QWORDS_IN_BLOCK];
    __private ulong block_r[ARGON2_QWORDS_IN_BLOCK];

    // XOR prev + ref
    for (int i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) {
        state[i] = prev[i] ^ ref[i];
        block_r[i] = ref[i];
    }

    // 8 BLAKE2 rounds
    for (int r = 0; r < 8; r++) {
        for (int j = 0; j < ARGON2_QWORDS_IN_BLOCK / 16; j++)
            blake2_round(&state[j*16]);
    }

    // XOR with input, write result
    for (int i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) {
        ulong v = state[i] ^ block_r[i];
        if (with_xor) v ^= next[i];
        next[i] = v;
    }
}

// Main Kernel
__kernel void argon2d_core(__global const uchar *prehash32,
                           __global uchar *mem,
                           const uint m_cost_kb,
                           const uint t_cost,
                           const uint lanes,
                           __global uchar *out32,
                           const uint start_block,
                           const uint end_block)
{
    (void)lanes; // MVP: single lane
    __global ulong *memory = (__global ulong*)mem;
    const uint blocks = m_cost_kb; // 1KB per block

    // Init first block only at pass=0, block=0
    if (start_block == 0) {
        __global ulong *first = memory;
        for (int i = 0; i < 4; i++) {
            ulong v = 0;
            for (int j = 0; j < 8; j++)
                v |= ((ulong)prehash32[i*8+j]) << (8*j);
            first[i] = v;
        }
        first[4] = (ulong)m_cost_kb;
        first[5] = (ulong)t_cost;
        first[6] = 0x13UL;  // Argon2d Version
        first[7] = 0UL;
        for (int i = 8; i < ARGON2_QWORDS_IN_BLOCK; i++) first[i] = 0;
    }

    // Main filling loop for this chunk
    for (uint pass = 0; pass < t_cost; pass++) {
        uint s = max((uint)1, start_block);
        uint e = min(end_block, blocks);

        for (uint idx = s; idx < e; idx++) {
            __global ulong *prev = memory + (idx-1)*ARGON2_QWORDS_IN_BLOCK;
            __global ulong *curr = memory + idx*ARGON2_QWORDS_IN_BLOCK;
            __global ulong *ref  = prev; // TODO: use get_ref_index()

            fill_block(prev, ref, curr, pass > 0);
        }
    }

    // Final output → out32
    __global ulong *last = memory + (blocks-1)*ARGON2_QWORDS_IN_BLOCK;
    for (int i = 0; i < 4; i++) {
        ulong v = last[i];
        for (int j = 0; j < 8; j++)
            out32[i*8+j] = (uchar)((v >> (8*j)) & 0xFF);
    }
    for (int i = 0; i < 32; i++)
        out32[i] ^= prehash32[i];
}


