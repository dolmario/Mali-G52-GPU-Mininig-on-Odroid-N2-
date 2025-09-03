// OpenCL 1.2 – Argon2d (Single-Lane) – sequenziell pro Chunk (start..end)
// Signatur:
// 0: __global const uchar* d_phash (32)
// 1: __global uchar*       d_mem   (m_cost_kb * 1024 bytes)
// 2: uint                  m_cost_kb  (== number of 1KB blocks)
// 3: uint                  pass
// 4: uint                  slice      (0..3)
// 5: uint                  start      (block index inkl.)
// 6: uint                  end        (block index exkl.)
// 7: __global uchar*       d_out      (32 bytes)
// 8: uint                  do_init

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#define BLOCK_BYTES 1024u
#define QWORDS_PER_BLOCK (BLOCK_BYTES / 8u)

typedef ulong u64;
typedef uint  u32;

// Rotations-Makro
inline u64 rotl64(u64 x, uint r) {
    return (x << r) | (x >> (64 - r));
}

// Einfacher 64-bit Mixer (kein kryptografischer Anspruch – SHA3 nachgelagert)
inline u64 mix64(u64 a, u64 b, u64 c) {
    a ^= rotl64(b, 13); a += c;
    b ^= rotl64(a, 17); b += a;
    c ^= rotl64(b, 43); c += b ^ 0x9e3779b97f4a7c15UL;
    return a ^ b ^ c;
}

// Typumwandlung: uchar* -> u64*
inline __global u64* as64(__global uchar* p) {
    return (__global u64*)p;
}

inline __global const u64* as64c(__global const uchar* p) {
    return (__global const u64*)p;
}

// Extrahiere J1, J2 aus den ersten 16 Bytes des vorherigen Blocks
inline void j1j2_from_prev(__global const u64* mem64, uint prev_base, uint words, uint* J1, uint* J2) {
    u64 v0 = mem64[prev_base + 0];
    u64 v1 = mem64[prev_base + 1];
    *J1 = (uint)(v0 & 0xFFFFFFFFu);
    *J2 = (uint)(v1 & 0xFFFFFFFFu);
}

// Wähle Referenzblock im Bereich [0, idx-1), außer (idx-1)
inline uint ref_index_single_lane(uint idx, uint J1, uint J2) {
    if (idx <= 1u) return 0u;
    uint r = (uint)(((ulong)J1 << 32) ^ (ulong)J2);
    uint cand = r % (idx - 1u);
    return cand;
}

// Berechne neuen Block: dst = (prev XOR ref) gemischt
inline void compute_block(__global u64* mem, uint dst_base, uint prev_base, uint ref_base) {
    for (uint i = 0; i < QWORDS_PER_BLOCK; i += 4) {
        u64 a = mem[prev_base + i + 0];
        u64 b = mem[prev_base + i + 1];
        u64 c = mem[ref_base  + i + 0];
        u64 d = mem[ref_base  + i + 1];
        u64 x0 = mix64(a, c, (u64)(i * 0x9e37u + 0));
        u64 x1 = mix64(b, d, (u64)(i * 0x85ebu + 1));
        mem[dst_base + i + 0] = x0;
        mem[dst_base + i + 1] = x1;

        u64 a2 = mem[prev_base + i + 2];
        u64 b2 = mem[prev_base + i + 3];
        u64 c2 = mem[ref_base  + i + 2];
        u64 d2 = mem[ref_base  + i + 3];
        u64 x2 = mix64(a2, c2, (u64)(i * 0x27d4u + 2));
        u64 x3 = mix64(b2, d2, (u64)(i * 0x94d0u + 3));
        mem[dst_base + i + 2] = x2;
        mem[dst_base + i + 3] = x3;
    }
}

__kernel void argon2d_core(
    __global const uchar* prehash32,   // arg 0
    __global uchar*       mem,         // arg 1
    const uint            blocks_per_lane, // arg 2 (m_cost_kb)
    const uint            pass_index,      // arg 3
    const uint            slice_index,     // arg 4
    const uint            start_block,     // arg 5
    const uint            end_block,       // arg 6
    __global uchar*       out32,           // arg 7
    const uint            do_init          // arg 8
) {
    // Nur Work-Item 0 arbeitet (sequenzieller Modus)
    if (get_global_id(0) != 0) return;

    __global u64* mem64 = as64(mem);
    __global u64* out64 = as64(out32);

    // --- Initialisierung der ersten beiden Blöcke ---
    if (do_init) {
        // Seed aus prehash32 (erste 16 Bytes)
        u64 s0 = ((u64)prehash32[0] << 56) |
                 ((u64)prehash32[1] << 48) |
                 ((u64)prehash32[2] << 40) |
                 ((u64)prehash32[3] << 32) |
                 ((u64)prehash32[4] << 24) |
                 ((u64)prehash32[5] << 16) |
                 ((u64)prehash32[6] << 8)  |
                 ((u64)prehash32[7]);

        u64 s1 = ((u64)prehash32[8]  << 56) |
                 ((u64)prehash32[9]  << 48) |
                 ((u64)prehash32[10] << 40) |
                 ((u64)prehash32[11] << 32) |
                 ((u64)prehash32[12] << 24) |
                 ((u64)prehash32[13] << 16) |
                 ((u64)prehash32[14] << 8)  |
                 ((u64)prehash32[15]);

        uint b0 = 0 * QWORDS_PER_BLOCK;
        uint b1 = 1 * QWORDS_PER_BLOCK;

        for (uint i = 0; i < QWORDS_PER_BLOCK; i++) {
            mem64[b0 + i] = mix64(s0 ^ (u64)i, s1 + (u64)i, (u64)(0xA5A5A5A5u + i));
            mem64[b1 + i] = mix64(s1 ^ (u64)i, s0 + (u64)i, (u64)(0x5A5A5A5Au + i));
        }

        // Fülle Lücke von Block 2 bis start_block mit einfachem Vorwärts-Modus
        for (uint idx = 2u; idx < start_block && idx < blocks_per_lane; ++idx) {
            uint prev = idx - 1u;
            uint dstb  = idx  * QWORDS_PER_BLOCK;
            uint prevb = prev * QWORDS_PER_BLOCK;
            uint refb  = 0;  // Einfache Referenz (konservativ)
            compute_block(mem64, dstb, prevb, refb);
        }
    }

    // --- Nullen des Ausgabepuffers ---
    for (int k = 0; k < 4; k++) {
        out64[k] = 0;
    }

    // --- Hauptlauf: sequenzielle Berechnung von start_block bis end_block-1 ---
    for (uint idx = start_block; idx < end_block && idx < blocks_per_lane; ++idx) {
        uint prev = (idx == 0) ? 0 : (idx - 1u);
        uint prevb = prev * QWORDS_PER_BLOCK;

        uint J1 = 0, J2 = 0;
        j1j2_from_prev(mem64, prevb, QWORDS_PER_BLOCK, &J1, &J2);
        uint ref = ref_index_single_lane(idx, J1, J2);

        uint dstb = idx  * QWORDS_PER_BLOCK;
        uint refb = ref  * QWORDS_PER_BLOCK;

        compute_block(mem64, dstb, prevb, refb);

        // Reduziere Ergebnis in out64 (XOR aller ersten 4 Wörter)
        out64[0] ^= mem64[dstb + 0];
        out64[1] ^= mem64[dstb + 1];
        out64[2] ^= mem64[dstb + 2];
        out64[3] ^= mem64[dstb + 3];
    }
}
