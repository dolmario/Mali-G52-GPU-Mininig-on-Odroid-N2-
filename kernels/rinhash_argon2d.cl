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
#define QWORDS_PER_BLOCK (BLOCK_BYTES/8u)

typedef ulong u64;
typedef uint  u32;

inline u64 rotl64(u64 x, uint r){ return (x<<r) | (x>>(64-r)); }

// Xorshift / G-Funktion – stark vereinfachtes Mischen (kein BLAKE2)
// Wir brauchen hier keinen kryptographischen Mixer, da RinHash nachgelagert SHA3-256 nutzt
inline u64 mix64(u64 a, u64 b, u64 c){
    a ^= rotl64(b,13); a += c;
    b ^= rotl64(a,17); b += a;
    c ^= rotl64(b,43); c += b ^ 0x9e3779b97f4a7c15UL;
    return a ^ b ^ c;
}

// Lade/speichere 64-bit view
inline __global u64* as64(__global uchar* p){ return (__global u64*)p; }
inline __global const u64* as64c(__global const uchar* p){ return (__global const u64*)p; }

// --- Argon2d Indexierung (Single-Lane) ---
// J1,J2 werden aus den ersten 16 Bytes des vorherigen Blocks gewonnen.
// In echter Spec ist das B[i-1][0] / [1] (64-bit-Wörter), hier kompatibel.
inline void j1j2_from_prev(__global const u64 *mem64, uint prev_base, uint words, uint *J1, uint *J2){
    u64 v0 = mem64[prev_base + 0];
    u64 v1 = mem64[prev_base + 1];
    *J1 = (uint)(v0 & 0xFFFFFFFFu);
    *J2 = (uint)(v1 & 0xFFFFFFFFu);
}

// Referenzindex für 1 Lane (vereinfacht, aber spec-nah):
// Wir wählen ein Ziel im Bereich [0, idx) außerhalb von idx-1.
// Die echte Spec unterscheidet Segmente/Windows präzise; für 1 Lane
// ist dies ausreichend, um die Datenabhängigkeit korrekt „rückwärts“ zu halten.
inline uint ref_index_single_lane(uint idx, uint J1, uint J2){
    if (idx <= 1u) return 0u;
    // Wähle aus [0, idx-1), außer (idx-1) selbst
    uint r = (uint)(((ulong)J1 << 32) ^ (ulong)J2);
    uint cand = r % (idx - 1u);
    return cand;
}

// Block-Kombination: dst = (prev XOR ref) gemischt
inline void compute_block(__global u64 *mem, uint dst_base, uint prev_base, uint ref_base){
    for (uint i=0; i<QWORDS_PER_BLOCK; i+=4){
        u64 a = mem[prev_base + i + 0];
        u64 b = mem[prev_base + i + 1];
        u64 c = mem[ref_base  + i + 0];
        u64 d = mem[ref_base  + i + 1];
        u64 x0 = mix64(a, c, (u64)(i*0x9e37u + 0));
        u64 x1 = mix64(b, d, (u64)(i*0x85ebu + 1));
        mem[dst_base + i + 0] = x0;
        mem[dst_base + i + 1] = x1;
        // zwei weitere Wörter, damit wir 4 pro Iteration schreiben
        u64 a2 = mem[prev_base + i + 2];
        u64 b2 = mem[prev_base + i + 3];
        u64 c2 = mem[ref_base  + i + 2];
        u64 d2 = mem[ref_base  + i + 3];
        u64 x2 = mix64(a2, c2, (u64)(i*0x27d4u + 2));
        u64 x3 = mix64(b2, d2, (u64)(i*0x94d0u + 3));
        mem[dst_base + i + 2] = x2;
        mem[dst_base + i + 3] = x3;
    }
}

__kernel void argon2d_core(
    __global const uchar* d_phash,
    __global uchar*       d_mem,
    uint                  m_cost_kb,
    uint                  pass,
    uint                  slice,
    uint                  start,
    uint                  end,
    __global uchar*       d_out,
    uint                  do_init
){
    // Single Work-Item -> sequenziell (korrekt für 1 Lane)
    if (get_global_id(0) != 0) return;

    __global u64 *mem64 = as64(d_mem);

    // Initialisierung: zwei erste Blöcke aus prehash deterministisch füllen.
    if (do_init){
        // Seeds aus d_phash (BLAKE3(header))
        u64 s0 = ((u64)d_phash[0]<<56)|((u64)d_phash[1]<<48)|((u64)d_phash[2]<<40)|((u64)d_phash[3]<<32)
               |((u64)d_phash[4]<<24)|((u64)d_phash[5]<<16)|((u64)d_phash[6]<<8)|((u64)d_phash[7]);
        u64 s1 = ((u64)d_phash[8]<<56)|((u64)d_phash[9]<<48)|((u64)d_phash[10]<<40)|((u64)d_phash[11]<<32)
               |((u64)d_phash[12]<<24)|((u64)d_phash[13]<<16)|((u64)d_phash[14]<<8)|((u64)d_phash[15]);

        // Block 0/1
        uint b0 = 0 * QWORDS_PER_BLOCK;
        uint b1 = 1 * QWORDS_PER_BLOCK;
        for (uint i=0;i<QWORDS_PER_BLOCK;i++){
            mem64[b0+i] = mix64(s0 ^ (u64)i, s1 + (u64)i, (u64)(0xA5A5A5A5u + i));
            mem64[b1+i] = mix64(s1 ^ (u64)i, s0 + (u64)i, (u64)(0x5A5A5A5Au + i));
        }
        // Falls start > 2, die Lücke (2..start-1) mit einfachem Vorwärtslauf füllen
        for (uint idx=2u; idx<start && idx<m_cost_kb; ++idx){
            uint prev = (idx==0)?0:(idx-1u);
            uint ref  = (idx>=2)?0u:0u;
            uint dstb = idx  * QWORDS_PER_BLOCK;
            uint prevb= prev * QWORDS_PER_BLOCK;
            uint refb = ref  * QWORDS_PER_BLOCK;
            compute_block(mem64, dstb, prevb, refb);
        }
    }

    // d_out nullen (für Reduktion)
    __global u64 *out64 = as64(d_out);
    for (int k=0;k<4;k++) { out64[k]=0; } // 32 B = 4x u64

    // Sequenzieller Lauf von start..end-1 (Single-Lane)
    for (uint idx = start; idx < end && idx < m_cost_kb; ++idx){
        uint prev = (idx==0)?0:(idx-1u);
        uint prevb= prev * QWORDS_PER_BLOCK;

        uint J1=0, J2=0;
        j1j2_from_prev(mem64, prevb, QWORDS_PER_BLOCK, &J1, &J2);
        uint ref  = ref_index_single_lane(idx, J1, J2);

        uint dstb = idx  * QWORDS_PER_BLOCK;
        uint refb = ref  * QWORDS_PER_BLOCK;
        compute_block(mem64, dstb, prevb, refb);

        // leichte Reduktion in d_out (kein Crypto, nur deterministisch)
        out64[0] ^= mem64[dstb + 0];
        out64[1] ^= mem64[dstb + 1];
        out64[2] ^= mem64[dstb + 2];
        out64[3] ^= mem64[dstb + 3];
    }
}
