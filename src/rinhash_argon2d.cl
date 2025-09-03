// OpenCL 1.2 RinHash Argon2d Core (Debug)
#define ARGON2_QWORDS_IN_BLOCK 128

inline ulong rotr64(ulong x, uint n) { return (x >> n) | (x << (64 - n)); }

inline void G(__private ulong *a, __private ulong *b, __private ulong *c, __private ulong *d) {
    *a = *a + *b; *d = rotr64(*d ^ *a, 32);
    *c = *c + *d; *b = rotr64(*b ^ *c, 24);
    *a = *a + *b; *d = rotr64(*d ^ *a, 16);
    *c = *c + *d; *b = rotr64(*b ^ *c, 63);
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

__kernel void argon2d_core(__global const uchar *prehash32,
                           __global uchar *mem,
                           const uint m_cost_kb,
                           const uint t_cost,
                           const uint lanes,
                           __global uchar *out32)
{
    (void)lanes; // single lane MVP
    __global ulong *memory = (__global ulong*)mem;
    const uint blocks = m_cost_kb; // 1KB per block

    // Block 0: aus prehash initialisieren (sehr vereinfachtes H0)
    for (int i = 0; i < 4; i++) {
        ulong v = 0;
        for (int j = 0; j < 8; j++)
            v |= ((ulong)prehash32[i*8+j]) << (j*8);
        memory[i] = v;
    }
    for (int i = 4; i < ARGON2_QWORDS_IN_BLOCK; i++) memory[i] = 0;

    for (uint pass = 0; pass < t_cost; pass++) {
        for (uint idx = 1; idx < blocks; idx++) {
            __global ulong *prev = memory + (idx-1)*ARGON2_QWORDS_IN_BLOCK;
            __global ulong *curr = memory + idx*ARGON2_QWORDS_IN_BLOCK;

            // in private kopieren + Salz
            __private ulong state[ARGON2_QWORDS_IN_BLOCK];
            for (int i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++)
                state[i] = prev[i] ^ ((ulong)idx * 0x9e3779b97f4a7c15UL);

            // Debug nur beim allerersten Arbeitsblock
            if (pass == 0 && idx == 1) {
                printf("DEBUG Prehash[0]=%02x\n", prehash32[0]);
                printf("DEBUG Seed[0]=%llu\n", (ulong)state[0]);
            }

            // 2 Runden BLAKE2-Mix (MVP – echt: mehr + Kompression)
            for (int r = 0; r < 2; r++)
                for (int j = 0; j < ARGON2_QWORDS_IN_BLOCK/16; j++)
                    blake2_round(&state[j*16]);

            // zurückschreiben
            for (int i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++)
                curr[i] = state[i];
        }
    }

    // final: 32B aus letztem Block (reduziert) + XOR prehash
    __global ulong *final_block = memory + (blocks-1)*ARGON2_QWORDS_IN_BLOCK;
    for (int i = 0; i < 4; i++) {
        ulong v = final_block[i];
        for (int j = 0; j < 8; j++)
            out32[i*8 + j] = (uchar)((v >> (j*8)) & 0xFF);
    }
    for (int i = 0; i < 32; i++) out32[i] ^= prehash32[i];
}



