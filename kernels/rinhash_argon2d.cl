// PHC-konformer Argon2d Kernel für RinHash (Mali-optimiert, i128-frei)

typedef ulong u64;
typedef uint  u32;
typedef uchar u8;

#define ARGON2_BLOCK_SIZE         1024
#define ARGON2_QWORDS_IN_BLOCK    128
#define ARGON2_SYNC_POINTS        4
#define ARGON2_VERSION            0x13

// Blake2b Konstanten
#define BLAKE2B_BLOCKBYTES 128
#define BLAKE2B_OUTBYTES   64

__constant u64 blake2b_IV[8] = {
    0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL,
    0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
    0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL,
    0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL
};

__constant u8 blake2b_sigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}
};

// ---- 64-bit Hilfsfunktionen (OHNE __int128) ----

inline u64 rotr64(u64 x, u32 n) {
    n &= 63U;
    return (x >> n) | (x << (64U - n));
}

// Low-64 von 64×64 via 32×32 Teilprodukte → i128-frei
inline u64 mul64_lo(u64 a, u64 b) {
    u32 a0 = (u32)(a & 0xffffffffUL);
    u32 a1 = (u32)(a >> 32);
    u32 b0 = (u32)(b & 0xffffffffUL);
    u32 b1 = (u32)(b >> 32);

    u64 p00 = (u64)a0 * (u64)b0;          // 32×32 -> 64
    u64 p01 = (u64)a0 * (u64)b1;
    u64 p10 = (u64)a1 * (u64)b0;
    u64 mid = (p01 + p10) << 32;          // nur low-64 behalten
    return p00 + mid;                      // mod 2^64
}

// BlaMka (PHC): a + b + 2 * (a*b mod 2^64)
inline u64 blamka(u64 x, u64 y) {
    return x + y + (mul64_lo(x, y) << 1);
}

// Argon2 G (BlaMka-Variante)
inline void argon2_G(u64 *a, u64 *b, u64 *c, u64 *d) {
    *a = blamka(*a, *b);
    *d = rotr64(*d ^ *a, 32);
    *c = blamka(*c, *d);
    *b = rotr64(*b ^ *c, 24);
    *a = blamka(*a, *b);
    *d = rotr64(*d ^ *a, 16);
    *c = blamka(*c, *d);
    *b = rotr64(*b ^ *c, 63);
}

// ---- Blake2b Kompression ----

inline void blake2b_G(u64* a, u64* b, u64* c, u64* d, u64 x, u64 y) {
    *a = *a + *b + x;
    *d = rotr64(*d ^ *a, 32);
    *c = *c + *d;
    *b = rotr64(*b ^ *c, 24);
    *a = *a + *b + y;
    *d = rotr64(*d ^ *a, 16);
    *c = *c + *d;
    *b = rotr64(*b ^ *c, 63);
}

inline void blake2b_compress(u64 h[8], const u64 m[16], u64 t, u64 f) {
    u64 v[16];

    for (int i=0;i<8;i++){ v[i]=h[i]; v[i+8]=blake2b_IV[i]; }
    v[12] ^= t;
    v[14] ^= f;

    for (int r=0;r<12;r++) {
        blake2b_G(&v[0], &v[4], &v[8],  &v[12], m[blake2b_sigma[r][0]],  m[blake2b_sigma[r][1]]);
        blake2b_G(&v[1], &v[5], &v[9],  &v[13], m[blake2b_sigma[r][2]],  m[blake2b_sigma[r][3]]);
        blake2b_G(&v[2], &v[6], &v[10], &v[14], m[blake2b_sigma[r][4]],  m[blake2b_sigma[r][5]]);
        blake2b_G(&v[3], &v[7], &v[11], &v[15], m[blake2b_sigma[r][6]],  m[blake2b_sigma[r][7]]);

        blake2b_G(&v[0], &v[5], &v[10], &v[15], m[blake2b_sigma[r][8]],  m[blake2b_sigma[r][9]]);
        blake2b_G(&v[1], &v[6], &v[11], &v[12], m[blake2b_sigma[r][10]], m[blake2b_sigma[r][11]]);
        blake2b_G(&v[2], &v[7], &v[8],  &v[13], m[blake2b_sigma[r][12]], m[blake2b_sigma[r][13]]);
        blake2b_G(&v[3], &v[4], &v[9],  &v[14], m[blake2b_sigma[r][14]], m[blake2b_sigma[r][15]]);
    }

    for (int i=0;i<8;i++) h[i] ^= v[i] ^ v[i+8];
}

// Hash genau 1024 Bytes → 32-Byte Blake2b (Input aus __global!)
inline void blake2b_hash32_1024(__global const u8 *in1024, __private u8 out32[32]) {
    u64 h[8];
    for (int i=0;i<8;i++) h[i] = blake2b_IV[i];
    h[0] ^= 0x01010000UL ^ 32UL;

    u64 t = 0;
    u64 m[16];

    for (int blk=0; blk<8; blk++) {
        __global const u8 *p = in1024 + (size_t)blk * 128U;
        for (int i=0;i<16;i++) {
            u64 w = 0;
            for (int b=0;b<8;b++) w |= ((u64)p[i*8 + b]) << (8*b); // LE
            m[i] = w;
        }
        t += 128;
        u64 f = (blk==7) ? 0xffffffffffffffffUL : 0UL;
        blake2b_compress(h, m, t, f);
    }

    for (int i=0;i<4;i++) {
        u64 w = h[i];
        out32[i*8 + 0] = (u8)( w        & 0xff);
        out32[i*8 + 1] = (u8)((w >>  8) & 0xff);
        out32[i*8 + 2] = (u8)((w >> 16) & 0xff);
        out32[i*8 + 3] = (u8)((w >> 24) & 0xff);
        out32[i*8 + 4] = (u8)((w >> 32) & 0xff);
        out32[i*8 + 5] = (u8)((w >> 40) & 0xff);
        out32[i*8 + 6] = (u8)((w >> 48) & 0xff);
        out32[i*8 + 7] = (u8)((w >> 56) & 0xff);
    }
}

// ---- Argon2d Bausteine ----

inline void fill_block(__global const u64 *prev_block,
                       __global const u64 *ref_block,
                       __global       u64 *next_block,
                       int with_xor)
{
    u64 R[ARGON2_QWORDS_IN_BLOCK];
    u64 Z[ARGON2_QWORDS_IN_BLOCK];

    for (int i=0;i<ARGON2_QWORDS_IN_BLOCK;i++) {
        u64 v = prev_block[i] ^ ref_block[i];
        R[i] = v;
        Z[i] = v;
    }

    for (int r=0;r<8;r++) {
        // columns (32 Vierergruppen)
        for (int j=0;j<32;j++) {
            int i0 = 4*j + 0;
            int i1 = 4*j + 1;
            int i2 = 4*j + 2;
            int i3 = 4*j + 3;
            argon2_G(&R[i0], &R[i1], &R[i2], &R[i3]);
        }
        // rows
        for (int j=0;j<32;j++) {
            int i0 = j +   0;
            int i1 = j +  32;
            int i2 = j +  64;
            int i3 = j +  96;
            argon2_G(&R[i0], &R[i1], &R[i2], &R[i3]);
        }
    }

    for (int i=0;i<ARGON2_QWORDS_IN_BLOCK;i++) {
        u64 v = Z[i] ^ R[i];
        next_block[i] = with_xor ? (next_block[i] ^ v) : v;
    }
}

// Referenzindex (ein Lane)
inline u32 index_alpha(u32 pass, u32 lane, u32 slice, u32 pos, u64 pseudo_rand, u32 same_lane, u32 blocks_per_lane) {
    (void)lane; (void)same_lane; // single lane
    u32 segment_length = blocks_per_lane / ARGON2_SYNC_POINTS;
    u32 reference_area_size;
    if (pass == 0) {
        if (slice == 0) {
            reference_area_size = pos - 1U;
        } else {
            reference_area_size = slice * segment_length + pos - 1U;
        }
    } else {
        reference_area_size = 3U * segment_length + pos - 1U;
    }
    u64 rel = pseudo_rand;
    rel = (rel * rel) >> 32;
    rel = reference_area_size - 1U - ((reference_area_size * rel) >> 32);
    u32 start_position = (pass != 0 && slice != 3) ? ((slice + 1U) * segment_length) : 0U;
    return (start_position + (u32)rel) % blocks_per_lane;
}

// ---- Kernel ----
// prehash32 bleibt in der Signatur (Hybrid ignoriert es)
__kernel void argon2d_core(
    __global const u8* prehash32,
    __global       u8* mem,
    const u32          blocks_per_lane,
    const u32          pass_index,
    const u32          slice_index,
    const u32          start_block,
    const u32          end_block,
    __global       u8* out32,
    const u32          do_init
){
    (void)prehash32;
    if (get_global_id(0) != 0) return;

    __global u64* mem64 = (__global u64*)mem;

    // Hybrid: CPU hat B[0], B[1] bereits geschrieben → keine Kernel-Init nötig
    (void)do_init;

    u32 seg_len = blocks_per_lane / ARGON2_SYNC_POINTS;

    for (u32 pos = start_block; pos < end_block && pos < blocks_per_lane; pos++) {
        if (pos < 2U) continue; // B0/B1 existieren

        u32 prev     = (pos == 0U) ? (blocks_per_lane - 1U) : (pos - 1U);
        u32 prev_off = prev * ARGON2_QWORDS_IN_BLOCK;
        u32 cur_off  = pos  * ARGON2_QWORDS_IN_BLOCK;

        u64 pseudo   = mem64[prev_off];

        u32 ref_idx  = index_alpha(pass_index, 0U, slice_index, pos % seg_len, pseudo, 1U, blocks_per_lane);
        u32 ref_off  = ref_idx * ARGON2_QWORDS_IN_BLOCK;

        fill_block(&mem64[prev_off], &mem64[ref_off], &mem64[cur_off], (pass_index > 0U) ? 1 : 0);
    }

    // Final: Tag = Blake2b-256(last 1024B block) → 32B nach out32
    if (pass_index == 1U && slice_index == 3U && end_block >= blocks_per_lane) {
        __global const u8 *blk_bytes = ( __global const u8* )(&mem64[(blocks_per_lane - 1U) * ARGON2_QWORDS_IN_BLOCK]);
        u8 tag[32];
        blake2b_hash32_1024(blk_bytes, tag);
        for (int i=0;i<32;i++) out32[i] = tag[i];
    }
}
