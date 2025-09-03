// rinhash_argon2d_real.cl — CL 1.2, Compile-Fixes (TYPEN/DECLS)
// HINWEIS: Das ist weiterhin ein vereinfachter Argon2d-Kern (MVP) mit Chunking-Interface.
// Ziel: erstmal sauber kompilieren & laufen. Feintuning/korrekter Argon2d-Core kommt danach.

// Konstanten
#define ARGON2_BLOCK_SIZE        1024           // Bytes pro Block
#define ARGON2_QWORDS_IN_BLOCK   128            // 128 * 8 = 1024
#define ARGON2_SYNC_POINTS       4

// 64-bit Rotate
inline ulong rotr64(ulong w, uint c) {
    return (w >> c) | (w << (64 - c));
}

// fBlaMka wie in Argon2
inline ulong fBlaMka(ulong x, ulong y) {
    const ulong m = (ulong)0xFFFFFFFFULL;       // untere 32 Bit
    ulong xy = (x & m) * (y & m);
    return x + y + (xy << 1);
}

// Blake2b G
inline void G(ulong *a, ulong *b, ulong *c, ulong *d) {
    *a = fBlaMka(*a, *b);
    *d = rotr64((*d) ^ (*a), 32);
    *c = fBlaMka(*c, *d);
    *b = rotr64((*b) ^ (*c), 24);
    *a = fBlaMka(*a, *b);
    *d = rotr64((*d) ^ (*a), 16);
    *c = fBlaMka(*c, *d);
    *b = rotr64((*b) ^ (*c), 63);
}

// Eine Blake2-Runde über 16er-Vektoren
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

// Vereinfachte Argon2-Block-Füllung (MVP)
// Achtung: noch NICHT bitgenau Argon2d-konform, dient nur zur Stabilitäts-/Leistungsprüfung.
inline void fill_block(__global ulong *prev_block,
                       __global ulong *ref_block,
                       __global ulong *next_block,
                       int with_xor)
{
    ulong state[ARGON2_QWORDS_IN_BLOCK];

    // XOR prev/ref in state
    for (int i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) {
        state[i] = prev_block[i] ^ ref_block[i];
    }

    // 8 „Rounds“ in 16er-Segmente (MVP)
    for (int r = 0; r < 8; r++) {
        for (int j = 0; j < ARGON2_QWORDS_IN_BLOCK / 16; j++) {
            blake2_round(&state[j * 16]);
        }
    }

    // optionales XOR mit next_block (chain)
    for (int i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) {
        ulong v = state[i];
        if (with_xor) v ^= next_block[i];
        next_block[i] = v;
    }
}

// Argon2d-Index (stark vereinfacht, aber syntaktisch korrekt; nutzt j1/j2)
inline uint get_ref_index(uint pass, uint lane, uint slice,
                          uint index, uint ref_lane,
                          __global ulong *memory, uint blocks_per_lane)
{
    // j1/j2: 32-Bit Teile aus dem vorigen Block
    uint j1, j2;

    // Vorheriger Block innerhalb der Lane
    uint prev_idx = (index == 0) ? (blocks_per_lane - 1) : (index - 1);
    __global ulong *addr_block = memory
        + (lane * blocks_per_lane + prev_idx) * ARGON2_QWORDS_IN_BLOCK;

    // Low/High 32 Bit aus 64-Bit-Wort:
    ulong w0 = addr_block[0];
    j1 = (uint)( w0        & (ulong)0xFFFFFFFFULL);
    j2 = (uint)((w0 >> 32) & (ulong)0xFFFFFFFFULL);

    // Referenzbereichsgröße (stark vereinfacht, passt für MVP)
    uint reference_area_size;
    if (pass == 0) {
        if (slice == 0) reference_area_size = (index > 0) ? (index - 1) : 0;
        else            reference_area_size = slice * (blocks_per_lane / ARGON2_SYNC_POINTS) + ((index > 0) ? (index - 1) : 0);
    } else {
        reference_area_size = blocks_per_lane - (blocks_per_lane / ARGON2_SYNC_POINTS) + ((index > 0) ? (index - 1) : 0);
        if (ref_lane != lane) {
            reference_area_size = (blocks_per_lane / ARGON2_SYNC_POINTS) * slice + index;
        }
    }
    if (reference_area_size == 0) return (lane * blocks_per_lane); // fallback

    // Mapping j1 → Index (Argon2-ähnlich, nicht exakt)
    ulong pr = (ulong)j1;
    pr = (pr * pr) >> 32;
    pr = (ulong)reference_area_size - 1UL - (( (ulong)reference_area_size * pr) >> 32);

    uint start_position = 0; // MVP
    return start_position + (uint)pr;
}

// Kernel mit Chunking-State
__kernel void argon2d_core(__global const uchar *prehash32,
                           __global uchar *mem,
                           const uint m_cost_kb,
                           const uint t_cost,
                           const uint lanes,
                           __global uchar *out32,
                           __global const uint *state_buf)
{
    // State lesen (6×uint)
    uint pass        = state_buf[0];
    uint slice       = state_buf[1];
    uint start_block = state_buf[2];
    uint end_block   = state_buf[3];
    uint blocks_lane = state_buf[4];
    // uint lanes_cnt = state_buf[5]; // wir nutzen lane=0

    __global ulong *memory = (__global ulong*)mem;
    uint lane = 0;
    size_t lane_offset_qw = (size_t)lane * (size_t)blocks_lane * (size_t)ARGON2_QWORDS_IN_BLOCK;

    // Init der ersten Blöcke beim allerersten Chunk
    if (pass == 0 && slice == 0 && start_block == 0) {
        __global ulong *b0 = memory + lane_offset_qw;
        // H0 aus prehash32 (vereinfachte Expansion auf 4×u64)
        for (int i = 0; i < 4; i++) {
            ulong v = (ulong)prehash32[i*8 + 0]
                    | ((ulong)prehash32[i*8 + 1] << 8)
                    | ((ulong)prehash32[i*8 + 2] << 16)
                    | ((ulong)prehash32[i*8 + 3] << 24)
                    | ((ulong)prehash32[i*8 + 4] << 32)
                    | ((ulong)prehash32[i*8 + 5] << 40)
                    | ((ulong)prehash32[i*8 + 6] << 48)
                    | ((ulong)prehash32[i*8 + 7] << 56);
            b0[i] = v;
        }
        for (int i = 4; i < ARGON2_QWORDS_IN_BLOCK; i++) b0[i] = 0UL;

        if (blocks_lane > 1) {
            __global ulong *b1 = b0 + ARGON2_QWORDS_IN_BLOCK;
            for (int i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) b1[i] = b0[i];
            b1[0] ^= (ulong)1; // Block-Zähler andeuten
        }
    }

    // Fenster verarbeiten
    for (uint idx = start_block; idx < end_block; ++idx) {
        // Erste zwei Blöcke im ersten Slice/Pas lassen wir wie initialisiert
        if (pass == 0 && slice == 0 && idx < 2) continue;

        __global ulong *curr = memory + lane_offset_qw + (size_t)idx * ARGON2_QWORDS_IN_BLOCK;
        __global ulong *prev = memory + lane_offset_qw + (size_t)((idx == 0 ? (blocks_lane - 1) : (idx - 1))) * ARGON2_QWORDS_IN_BLOCK;

        uint ref_index = get_ref_index(pass, lane, slice, idx, lane, memory, blocks_lane);
        __global ulong *ref  = memory + lane_offset_qw + (size_t)ref_index * ARGON2_QWORDS_IN_BLOCK;

        // MVP-Kompression
        fill_block(prev, ref, curr, pass > 0);
    }

    // Letzter Chunk erzeugt 32-Byte-Ausgabe
    if (pass == (t_cost - 1) && slice == 3 && end_block == blocks_lane) {
        __global ulong *final_block = memory + lane_offset_qw + (size_t)(blocks_lane - 1) * ARGON2_QWORDS_IN_BLOCK;

        // Einfache Reduktion (später: korrekte Finalisierung)
        for (int i = 0; i < 4; i++) {
            ulong val = final_block[i] ^ final_block[ARGON2_QWORDS_IN_BLOCK - 4 + i];
            // Write little-endian:
            out32[i*8 + 0] = (uchar)( val        & 0xFF);
            out32[i*8 + 1] = (uchar)((val >> 8 ) & 0xFF);
            out32[i*8 + 2] = (uchar)((val >> 16) & 0xFF);
            out32[i*8 + 3] = (uchar)((val >> 24) & 0xFF);
            out32[i*8 + 4] = (uchar)((val >> 32) & 0xFF);
            out32[i*8 + 5] = (uchar)((val >> 40) & 0xFF);
            out32[i*8 + 6] = (uchar)((val >> 48) & 0xFF);
            out32[i*8 + 7] = (uchar)((val >> 56) & 0xFF);
        }
        // Vorhash einmischen (MVP)
        for (int i = 0; i < 32; i++) out32[i] ^= prehash32[i];
    }
}
