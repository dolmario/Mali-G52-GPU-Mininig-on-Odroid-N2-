// OpenCL 1.2 – RinHash Argon2d (single lane) mit richtiger Slice/Index-Logik
// und Chunking für Mali (start_block/end_block). MVP: 1 Lane, Addressing via j1/j2
// aus dem vorherigen Block (Wort 0). Für volle Konformität kann später der
// Address-Block (AddrGen) ergänzt werden.

#define ARGON2_QWORDS_IN_BLOCK 128   // 1024 Bytes
#define ARGON2_SYNC_POINTS 4         // 4 Slices

inline ulong rotr64(ulong x, uint n) { return (x >> n) | (x << (64 - n)); }

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

inline void fill_block(__global ulong *prev, __global ulong *ref, __global ulong *next, int with_xor) {
    __private ulong state[ARGON2_QWORDS_IN_BLOCK];
    __private ulong block_r[ARGON2_QWORDS_IN_BLOCK];

    // XOR prev/ref in private state
    for (int i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) {
        ulong v = prev[i] ^ ref[i];
        state[i] = v;
        block_r[i] = ref[i];
    }

    // 8 Runden Blake2 (Argon2-Style)
    for (int r = 0; r < 8; r++) {
        for (int j = 0; j < ARGON2_QWORDS_IN_BLOCK/16; j++)
            blake2_round(&state[j*16]);
    }

    // Finalisieren
    for (int i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) {
        ulong v = state[i] ^ block_r[i];
        if (with_xor) v ^= next[i];
        next[i] = v;
    }
}

// Argon2d-Indexierung (Single-Lane Version):
// Mappt j1/j2 aus prev_block[0] auf eine gültige Referenzposition im Slice-Bereich.
// Entspricht der Spezifikation näher als der alte MVP (keine zufällige/lineare Wahl).
inline uint argon2d_ref_index(uint pass, uint slice, uint idx,
                              uint blocks_per_lane,
                              __global ulong *memory)
{
    // Für Pass 0, Slice 0 ist die erlaubte Referenzfläche besonders eingeschränkt.
    // Allgemein: Segmentgröße = blocks_per_lane / ARGON2_SYNC_POINTS
    const uint segment_length = blocks_per_lane / ARGON2_SYNC_POINTS;
    const uint slice_start = slice * segment_length;

    // „Position in Segment“ (0..segment_length-1)
    uint pos_in_seg;
    if (pass == 0 && slice == 0) {
        // Im allerersten Segment beginnt man ab Block 2 (Block 0/1 sind init)
        if (idx < 2) return 0; // Fallback
        pos_in_seg = idx - 2;  // offset ab 2
        if (pos_in_seg >= segment_length) pos_in_seg = segment_length - 1;
    } else {
        // sonst: normalisiert auf Segmentanfang
        if (idx < slice_start) return slice_start; // Fallback
        pos_in_seg = idx - slice_start;
        if (pos_in_seg >= segment_length) pos_in_seg = segment_length - 1;
    }

    // Pseudo-Zufall aus vorherigem Block (Wort 0 → 64 Bit):
    // j1 = low32, j2 = high32
    __global ulong *prev = memory + (idx - 1) * ARGON2_QWORDS_IN_BLOCK;
    ulong w0 = prev[0];
    uint j1 = (uint)(w0 & 0xFFFFFFFFUL);
    // uint j2 = (uint)(w0 >> 32); // für Multi-Lane nötig; hier lane=0

    // Referenzbereichsgröße
    uint reference_area_size;
    if (pass == 0) {
        // Pass 0: innerhalb des bisherigen Bereichs des Segments, minus 1
        reference_area_size = (slice == 0) ? pos_in_seg : (slice * segment_length + pos_in_seg);
    } else {
        // Pass >0: kompletter Bereich bis jetzt, minus 1
        reference_area_size = blocks_per_lane - segment_length + pos_in_seg;
    }
    if (reference_area_size == 0) return (idx - 1); // Fallback

    // „Non-uniform mapping“ wie in der Spezifikation (Bernstein):
    // pseudo = (j1^2 >> 32);  ref_area - 1 - floor(ref_area * pseudo / 2^32)
    ulong pr = (ulong)j1; pr = (pr * pr) >> 32;
    uint rel = reference_area_size - 1 - (uint)((reference_area_size * pr) >> 32);

    // Startposition (in Pass 0 ab 0, sonst ab segment_length)
    uint start_position;
    if (pass == 0) {
        start_position = 0;
    } else {
        start_position = (slice == 0) ? 0 : segment_length * slice;
    }

    uint ref_index = start_position + rel;
    if (ref_index >= blocks_per_lane) ref_index %= blocks_per_lane;

    // In den ersten beiden Blöcken nicht auf sich selbst o. ähnliche Kantenfälle zeigen:
    if (ref_index == idx) {
        ref_index = (ref_index == 0) ? 1 : (ref_index - 1);
    }

    return ref_index;
}

__kernel void argon2d_core(
    __global const uchar *prehash32,   // 32 B BLAKE3(header)
    __global uchar *mem,               // Argon2 memory (blocks * 1024 B)
    const uint blocks_per_lane,        // m_cost_kb (1 Block = 1 KB)
    const uint pass_index,             // welcher Pass (0..t_cost-1)
    const uint slice_index,            // welcher Slice (0..3)
    const uint start_block,            // inkl. (Chunk-Start)
    const uint end_block,              // exkl. (Chunk-Ende)
    __global uchar *out32,             // 32 B Ergebnis (optional)
    const uint do_init                 // 1 = Block 0/1 initialisieren
){
    __global ulong *memory = (__global ulong*)mem;

    // Einmalige Initialisierung zu Pass 0, Slice 0, Start==0
    if (do_init) {
        // Block 0 aus prehash
        __global ulong *b0 = memory + 0;
        for (int i = 0; i < 4; i++) {
            ulong v = 0;
            for (int j = 0; j < 8; j++)
                v |= ((ulong)prehash32[i*8 + j]) << (8*j);
            b0[i] = v;
        }
        for (int i = 4; i < ARGON2_QWORDS_IN_BLOCK; i++) b0[i] = 0;

        // Block 1 (H') – einfach als Variation von Block 0 (MVP)
        if (blocks_per_lane > 1) {
            __global ulong *b1 = memory + ARGON2_QWORDS_IN_BLOCK;
            for (int i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++)
                b1[i] = b0[i] ^ (ulong)0xDEADBEEFDEADBEEFUL;
        }
    }

    // Segmentgrenzen
    const uint segment_length = blocks_per_lane / ARGON2_SYNC_POINTS;
    const uint seg_start = slice_index * segment_length;
    const uint seg_end   = seg_start + segment_length;

    // Chunk auf diesen Slice schneiden
    uint s = start_block; if (s < seg_start) s = seg_start;
    uint e = end_block;   if (e > seg_end)   e = seg_end;
    if (e > blocks_per_lane) e = blocks_per_lane;
    if (s >= e) return;

    // Beim allerersten Segment nicht Block 0 überschreiben
    if (pass_index == 0 && slice_index == 0 && s == 0) s = 2; // Block 0/1 sind Init

    for (uint idx = s; idx < e; idx++) {
        __global ulong *prev = memory + (idx - 1) * ARGON2_QWORDS_IN_BLOCK;
        uint ref_i = argon2d_ref_index(pass_index, slice_index, idx, blocks_per_lane, memory);
        __global ulong *ref  = memory + ref_i * ARGON2_QWORDS_IN_BLOCK;
        __global ulong *curr = memory + idx * ARGON2_QWORDS_IN_BLOCK;

        // with_xor ab Pass > 0
        fill_block(prev, ref, curr, (pass_index > 0));
    }

    // out32 wird vom Host nach komplettem letzten Pass erzeugt
}


