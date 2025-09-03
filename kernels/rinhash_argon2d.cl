// OpenCL 1.2 Kernel – vereinfachter Argon2d-Core mit Chunking
// Signatur muss zu main.c passen:
// 0: __global const uchar* d_phash (32)
// 1: __global uchar*       d_mem   (m_cost_kb * 1024 bytes)
// 2: uint                  m_cost_kb
// 3: uint                  pass
// 4: uint                  slice
// 5: uint                  start
// 6: uint                  end
// 7: __global uchar*       d_out   (32)
// 8: uint                  do_init (1 bei allererstem Chunk/PASS0/SLICE0)

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#define BLOCK_BYTES 1024u
#define WORDS_PER_BLOCK (BLOCK_BYTES/4u)

// Kleines Inline-"Mixerle", KEIN echtes BLAKE/Keccak – nur deterministisches Mischen.
inline uint rotl32(uint x, uint r){ return (x<<r) | (x>>(32-r)); }
inline uint mix_word(uint a, uint b, uint c){
    a ^= rotl32(b, 13); a += c;
    a ^= rotl32(a, 17); a += b;
    return a ^ (c + 0x9e3779b9u);
}

// Fake-Index (Platzhalter) – nutzt vorherigen Blockinhalt
inline uint get_ref_index_simplified(__global const uint *mem32, uint idx){
    if (idx == 0) return 0u;
    // Nimm ein Wort aus dem vorherigen Block als Pseudozufall
    uint prev_base = (idx - 1u) * WORDS_PER_BLOCK;
    uint v = mem32[prev_base + (idx % WORDS_PER_BLOCK)];
    // Begrenzen auf [0, idx)
    return (v % idx);
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
    const uint gid = get_global_id(0);
    const uint idx = start + gid;
    if (idx >= end) return;

    // Gesamtspeicher in 32-bit Sicht
    __global uint *mem32 = (__global uint*)d_mem;

    // Initialisierung nur beim allerersten Chunk des allerersten Slices/Passes
    if (do_init && gid == 0) {
        // Fülle Block 0 und 1 deterministisch aus d_phash
        uint seed0 = ((uint)d_phash[ 0]<<24)|((uint)d_phash[ 1]<<16)|((uint)d_phash[ 2]<<8)|((uint)d_phash[ 3]);
        uint seed1 = ((uint)d_phash[28]<<24)|((uint)d_phash[29]<<16)|((uint)d_phash[30]<<8)|((uint)d_phash[31]);

        for (uint w=0; w<WORDS_PER_BLOCK; ++w) {
            mem32[0*WORDS_PER_BLOCK + w] = mix_word(seed0 ^ w, seed1, 0xA5A5A5A5u + w);
            mem32[1*WORDS_PER_BLOCK + w] = mix_word(seed1 ^ w, seed0, 0x5A5A5A5Au + w);
        }
        // Rest bis start mit etwas leichtem Muster belegen (optional)
        for (uint i=2; i<start && i<m_cost_kb; ++i){
            uint base = i*WORDS_PER_BLOCK;
            for (uint w=0; w<WORDS_PER_BLOCK; ++w){
                mem32[base + w] = mix_word(mem32[base - WORDS_PER_BLOCK + w],
                                           (uint)(i*0x9E37u + w), (uint)(w*0x85EBu + i));
            }
        }
    }

    // --- eigentliche Block-Berechnung für idx ---
    // Input-Quellen: vorheriger Block und "Referenzblock" (vereinfachter Index)
    uint ref = get_ref_index_simplified(mem32, idx==0 ? 0u : idx);
    uint prev = (idx==0) ? 0u : (idx-1u);
    uint base_dst  = idx  * WORDS_PER_BLOCK;
    uint base_prev = prev * WORDS_PER_BLOCK;
    uint base_ref  = ref  * WORDS_PER_BLOCK;

    // Mischen: 32-bit Worte kombinieren (sehr einfach, placeholder)
    for (uint w=0; w<WORDS_PER_BLOCK; ++w){
        uint a = mem32[base_prev + w];
        uint b = mem32[base_ref  + w];
        uint c = ((uint)idx << 16) ^ (uint)w;
        uint outw = mix_word(a, b, c);
        mem32[base_dst + w] = outw;
    }

    // "Digest"-Update (billiger 32-Byte Abdruck) – wir schreiben einfach
    // die ersten 8 Worte des aktuellen Blocks XOR-verdichtet in d_out.
    // Damit der Host immer etwas Deterministisches bekommt.
    // (Kein Barrier nötig: jede Work-Item macht atomare XORs auf d_out[0..31])
    // d_out ist byte*, wir adressieren aber als uint*:
    __global uint *out32 = (__global uint*)d_out;
    // Lasse nur das letzte Work-Item des Chunks schreiben? Nein: alle mischen atomar.
    // Dadurch hat der Host am Ende des Pass/Slice eine stabile, wenn auch simple, Reduktion.
    uint partial0 = mem32[base_dst + 0];
    uint partial1 = mem32[base_dst + 1];
    uint partial2 = mem32[base_dst + 2];
    uint partial3 = mem32[base_dst + 3];
    uint partial4 = mem32[base_dst + 4];
    uint partial5 = mem32[base_dst + 5];
    uint partial6 = mem32[base_dst + 6];
    uint partial7 = mem32[base_dst + 7];

    // Atomare XORs (OpenCL 1.2: 32-bit atomics ok)
    atomic_xor((volatile __global int*)&out32[0], (int)partial0);
    atomic_xor((volatile __global int*)&out32[1], (int)partial1);
    atomic_xor((volatile __global int*)&out32[2], (int)partial2);
    atomic_xor((volatile __global int*)&out32[3], (int)partial3);
    atomic_xor((volatile __global int*)&out32[4], (int)partial4);
    atomic_xor((volatile __global int*)&out32[5], (int)partial5);
    atomic_xor((volatile __global int*)&out32[6], (int)partial6);
    atomic_xor((volatile __global int*)&out32[7], (int)partial7);
}

