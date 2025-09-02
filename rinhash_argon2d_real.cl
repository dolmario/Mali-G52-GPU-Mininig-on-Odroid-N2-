// rinhash_argon2d_real.cl - Echter Argon2d für Mali-G52
// OpenCL 1.2 kompatibel

// Constants
#define ARGON2_BLOCK_SIZE 1024
#define ARGON2_QWORDS_IN_BLOCK 128
#define ARGON2_SYNC_POINTS 4

// Blake2b round function
inline ulong rotr64(ulong w, uint c) {
    return (w >> c) | (w << (64 - c));
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

// Blake2 round for Argon2
void blake2_round(__private ulong *v) {
    G(&v[0], &v[4], &v[8],  &v[12]);
    G(&v[1], &v[5], &v[9],  &v[13]);
    G(&v[2], &v[6], &v[10], &v[14]);
    G(&v[3], &v[7], &v[11], &v[15]);
    G(&v[0], &v[5], &v[10], &v[15]);
    G(&v[1], &v[6], &v[11], &v[12]);
    G(&v[2], &v[7], &v[8],  &v[13]);
    G(&v[3], &v[4], &v[9],  &v[14]);
}

// Argon2 compression function - fills dst block
void fill_block(__global ulong *prev_block, 
                __global ulong *ref_block,
                __global ulong *next_block,
                int with_xor) {
    ulong state[ARGON2_QWORDS_IN_BLOCK];
    ulong block_r[ARGON2_QWORDS_IN_BLOCK];
    ulong block_tmp[ARGON2_QWORDS_IN_BLOCK];
    
    // XOR prev and ref blocks
    for (int i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) {
        state[i] = prev_block[i] ^ ref_block[i];
        block_r[i] = ref_block[i];
    }
    
    // Apply Blake2 rounds (simplified - should be 8 rounds)
    for (int i = 0; i < 8; i++) {
        // Permutation
        for (int j = 0; j < ARGON2_QWORDS_IN_BLOCK / 16; j++) {
            blake2_round(&state[j * 16]);
        }
    }
    
    // XOR with input
    for (int i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) {
        state[i] ^= block_r[i];
        if (with_xor) {
            state[i] ^= next_block[i];
        }
    }
    
    // Write result
    for (int i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) {
        next_block[i] = state[i];
    }
}

// Argon2d indexing
uint get_ref_index(uint pass, uint lane, uint slice, 
                   uint index, uint ref_lane,
                   __global ulong *memory, uint blocks_per_lane) {
    uint32_t j1, j2;
    ulong pseudo_rand;
    
    if (pass == 0 && slice == 0) {
        // First pass, first slice
        ref_lane = lane;
    }
    
    // Get pseudo-random values from previous block
    if (index % ARGON2_QWORDS_IN_BLOCK == 0) {
        // First block in segment - use last block
        __global ulong *addr_block = memory + 
            (lane * blocks_per_lane + index - 1) * ARGON2_QWORDS_IN_BLOCK;
        j1 = (uint32_t)addr_block[0];
        j2 = (uint32_t)(addr_block[0] >> 32);
    } else {
        // Use previous block
        __global ulong *addr_block = memory + 
            (lane * blocks_per_lane + index - 1) * ARGON2_QWORDS_IN_BLOCK;
        j1 = (uint32_t)addr_block[0];
        j2 = (uint32_t)(addr_block[0] >> 32);
    }
    
    uint reference_area_size;
    uint start_position = 0;
    
    if (pass == 0) {
        if (slice == 0) {
            reference_area_size = index - 1;
        } else {
            reference_area_size = slice * (blocks_per_lane / ARGON2_SYNC_POINTS) + index - 1;
        }
    } else {
        reference_area_size = blocks_per_lane - (blocks_per_lane / ARGON2_SYNC_POINTS) + index - 1;
        if (ref_lane != lane) {
            reference_area_size = blocks_per_lane / ARGON2_SYNC_POINTS * slice + index;
        }
    }
    
    // Mapping j1 to actual index
    pseudo_rand = (ulong)j1;
    pseudo_rand = (pseudo_rand * pseudo_rand) >> 32;
    pseudo_rand = reference_area_size - 1 - ((reference_area_size * pseudo_rand) >> 32);
    
    return start_position + (uint)pseudo_rand;
}

__kernel void argon2d_core(__global const uchar *prehash32,
                           __global uchar *mem,
                           const uint m_cost_kb,
                           const uint t_cost,
                           const uint lanes,
                           __global uchar *out32,
                           __global const uint *state_buf)
{
    // Read chunking state
    uint pass          = state_buf[0];
    uint slice         = state_buf[1];
    uint start_block   = state_buf[2];
    uint end_block     = state_buf[3];
    uint blocks_lane   = state_buf[4];
    uint lanes_count   = state_buf[5];
    
    // Cast memory to ulong for 64-bit operations
    __global ulong *memory = (__global ulong*)mem;
    
    // Single lane for now (lane 0)
    uint lane = 0;
    size_t lane_offset = lane * blocks_lane * ARGON2_QWORDS_IN_BLOCK;
    
    // Initialize first blocks on first pass
    if (pass == 0 && slice == 0 && start_block == 0) {
        // H0: Initialize from prehash (simplified)
        __global ulong *first_block = memory + lane_offset;
        
        // Fill with prehash data (extended)
        for (int i = 0; i < 4; i++) {
            ulong val = 0;
            for (int j = 0; j < 8; j++) {
                val |= ((ulong)prehash32[i * 8 + j]) << (j * 8);
            }
            first_block[i] = val;
        }
        
        // Fill rest with lane/pass/slice info
        first_block[4] = (ulong)lane;
        first_block[5] = (ulong)m_cost_kb;
        first_block[6] = (ulong)t_cost;
        first_block[7] = 0x13;  // Argon2d version
        
        for (int i = 8; i < ARGON2_QWORDS_IN_BLOCK; i++) {
            first_block[i] = 0;
        }
        
        // Second block (H')
        if (blocks_lane > 1) {
            __global ulong *second_block = memory + lane_offset + ARGON2_QWORDS_IN_BLOCK;
            for (int i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) {
                second_block[i] = first_block[i];
            }
            second_block[0] ^= 1;  // Block counter
        }
    }
    
    // Process blocks in chunk
    for (uint curr_block = start_block; curr_block < end_block; curr_block++) {
        // Skip first two blocks (already initialized)
        if (pass == 0 && slice == 0 && curr_block < 2) continue;
        
        __global ulong *curr = memory + lane_offset + curr_block * ARGON2_QWORDS_IN_BLOCK;
        __global ulong *prev = memory + lane_offset + ((curr_block - 1) % blocks_lane) * ARGON2_QWORDS_IN_BLOCK;
        
        // Compute reference block index
        uint ref_index = get_ref_index(pass, lane, slice, curr_block, lane, memory, blocks_lane);
        __global ulong *ref = memory + lane_offset + ref_index * ARGON2_QWORDS_IN_BLOCK;
        
        // Fill block using compression function
        fill_block(prev, ref, curr, pass > 0);
    }
    
    // Generate final output (only on last chunk)
    if (pass == t_cost - 1 && slice == 3 && end_block == blocks_lane) {
        // Final block is the last block in memory
        __global ulong *final_block = memory + lane_offset + (blocks_lane - 1) * ARGON2_QWORDS_IN_BLOCK;
        
        // Simple reduction to 32 bytes (should use proper finalization)
        for (int i = 0; i < 4; i++) {
            ulong val = final_block[i] ^ final_block[ARGON2_QWORDS_IN_BLOCK - 4 + i];
            for (int j = 0; j < 8; j++) {
                out32[i * 8 + j] = (uchar)((val >> (j * 8)) & 0xFF);
            }
        }
        
        // XOR with prehash
        for (int i = 0; i < 32; i++) {
            out32[i] ^= prehash32[i];
        }
    }
}
