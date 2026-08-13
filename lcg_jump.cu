// ============================================================
//  Parallel LCG on GPU with Jump-Ahead Function
//  Supports 32 / 64 / 128 / 256-bit state, user-defined
//  a, c, m, seed, thread count, and output-per-thread.
//  Includes built-in verification against CPU brute force.
// ============================================================

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <cuda_runtime.h>

// ============================================================
//  256-bit unsigned integer (4 x 64-bit limbs), little-endian
//  Only supports add/mul with automatic mod-2^256 behaviour
// ============================================================
struct uint256_t {
    uint64_t w[4]; // w[0] = least significant limb
};

__host__ __device__ inline uint256_t u256_from_u64(uint64_t v) {
    uint256_t r{}; r.w[0] = v; r.w[1] = r.w[2] = r.w[3] = 0; return r;
}

__host__ __device__ inline uint256_t u256_add(const uint256_t &a, const uint256_t &b) {
    uint256_t r{};
    unsigned __int128 carry = 0;
    for (int i = 0; i < 4; i++) {
        unsigned __int128 s = (unsigned __int128)a.w[i] + b.w[i] + carry;
        r.w[i] = (uint64_t)s;
        carry = s >> 64;
    }
    return r; // overflow beyond 256 bits dropped -> mod 2^256
}

__host__ __device__ inline uint256_t u256_mul(const uint256_t &a, const uint256_t &b) {
    uint64_t res[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; i++) {
        unsigned __int128 carry = 0;
        for (int j = 0; j < 4 - i; j++) {
            unsigned __int128 prod = (unsigned __int128)a.w[i] * b.w[j];
            unsigned __int128 sum  = (unsigned __int128)res[i + j] +
                                      (prod & 0xFFFFFFFFFFFFFFFFULL) + carry;
            res[i + j] = (uint64_t)sum;
            carry = (sum >> 64) + (prod >> 64);
        }
        // carries beyond index 3 are beyond 256 bits -> dropped (mod 2^256)
    }
    uint256_t r{};
    r.w[0]=res[0]; r.w[1]=res[1]; r.w[2]=res[2]; r.w[3]=res[3];
    return r;
}

__host__ inline void u256_print(const uint256_t &v) {
    printf("0x");
    for (int i = 3; i >= 0; i--) printf("%016llx", (unsigned long long)v.w[i]);
}

// ============================================================
//  Generic templated jump function (32/64/128-bit types)
//  T*T and T+T must wrap around automatically (true for
//  uint32_t / uint64_t / unsigned __int128)
// ============================================================
template <typename T>
__host__ __device__ inline void lcg_jump_generic(T a, T c, uint64_t k, T &A_out, T &C_out) {
    T A = T(1), C = T(0);
    T a_i = a, c_i = c;
    while (k > 0) {
        if (k & 1ULL) {
            A = A * a_i;
            C = C * a_i + c_i;
        }
        c_i = c_i * (a_i + T(1));
        a_i = a_i * a_i;
        k >>= 1;
    }
    A_out = A; C_out = C;
}

// Specialization for uint256_t (no native operators, use helper functions)
__host__ __device__ inline void lcg_jump_256(uint256_t a, uint256_t c, uint64_t k,
                                              uint256_t &A_out, uint256_t &C_out) {
    uint256_t A = u256_from_u64(1), C = u256_from_u64(0);
    uint256_t a_i = a, c_i = c;
    while (k > 0) {
        if (k & 1ULL) {
            A = u256_mul(A, a_i);
            C = u256_add(u256_mul(C, a_i), c_i);
        }
        uint256_t a_plus_1 = u256_add(a_i, u256_from_u64(1));
        c_i = u256_mul(c_i, a_plus_1);
        a_i = u256_mul(a_i, a_i);
        k >>= 1;
    }
    A_out = A; C_out = C;
}

// ============================================================
//  Generic kernel (32/64/128-bit) — T2 = double-width type
//  used only when a custom (non power-of-two) modulus is set.
// ============================================================
template <typename T, typename T2>
__global__ void lcg_kernel(T seed, T a, T c, T m, bool use_mod,
                            int n_per_thread, int num_threads, T *out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    uint64_t start_index = (uint64_t)tid * n_per_thread;

    T A, C;
    lcg_jump_generic<T>(a, c, start_index, A, C);

    T state;
    if (use_mod) state = (T)(((T2)A * (T2)seed + (T2)C) % (T2)m);
    else         state = A * seed + C; // power-of-two modulus = free wraparound

    for (int i = 0; i < n_per_thread; i++) {
        out[(size_t)tid * n_per_thread + i] = state;
        if (use_mod) state = (T)(((T2)a * (T2)state + (T2)c) % (T2)m);
        else         state = a * state + c;
    }
}

// 256-bit kernel (modulus fixed at 2^256, always wraparound)
__global__ void lcg_kernel_256(uint256_t seed, uint256_t a, uint256_t c,
                                int n_per_thread, int num_threads, uint256_t *out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    uint64_t start_index = (uint64_t)tid * n_per_thread;

    uint256_t A, C;
    lcg_jump_256(a, c, start_index, A, C);

    uint256_t state = u256_add(u256_mul(A, seed), C);
    for (int i = 0; i < n_per_thread; i++) {
        out[(size_t)tid * n_per_thread + i] = state;
        state = u256_add(u256_mul(a, state), c);
    }
}

// ============================================================
//  CPU brute-force reference (for verification), 64-bit only
// ============================================================
uint64_t cpu_brute_force_64(uint64_t seed, uint64_t a, uint64_t c, uint64_t m,
                             bool use_mod, uint64_t steps) {
    uint64_t x = seed;
    for (uint64_t i = 0; i < steps; i++) {
        if (use_mod) x = (uint64_t)(((__uint128_t)a * x + c) % m);
        else         x = a * x + c; // wraps at 2^64
    }
    return x;
}

uint32_t cpu_brute_force_32(uint32_t seed, uint32_t a, uint32_t c, uint32_t m,
                             bool use_mod, uint64_t steps) {
    uint32_t x = seed;
    for (uint64_t i = 0; i < steps; i++) {
        if (use_mod) x = (uint32_t)(((uint64_t)a * x + c) % m);
        else         x = a * x + c; // wraps at 2^32
    }
    return x;
}

// ============================================================
//  Verification + printing helpers
// ============================================================
void verify_and_print_32(uint32_t *h_out, int num_threads, int n_per_thread,
                          uint32_t a, uint32_t c, uint32_t m, bool use_mod, uint32_t seed) {
    int show = n_per_thread < 5 ? n_per_thread : 5;
    printf("\n--- Per-thread output (first %d values each) ---\n", show);
    for (int t = 0; t < num_threads; t++) {
        printf("Thread %2d: ", t);
        for (int i = 0; i < show; i++) printf("%u ", h_out[(size_t)t * n_per_thread + i]);
        printf("\n");
    }

    // Boundary check: last value of thread t-1 stepped once == first value of thread t
    bool boundary_ok = true;
    for (int t = 1; t < num_threads; t++) {
        uint32_t last_prev = h_out[(size_t)(t - 1) * n_per_thread + (n_per_thread - 1)];
        uint32_t first_this = h_out[(size_t)t * n_per_thread];
        uint32_t expected = use_mod ? (uint32_t)(((uint64_t)a * last_prev + c) % m)
                                     : a * last_prev + c;
        if (expected != first_this) { boundary_ok = false; printf("MISMATCH at boundary %d->%d\n", t-1, t); }
    }
    printf(boundary_ok ? "Boundary check: PASSED\n" : "Boundary check: FAILED\n");

    // Brute-force cross-check on a sample thread
    int check_t = num_threads > 1 ? num_threads / 2 : 0;
    uint32_t expected_start = cpu_brute_force_32(seed, a, c, m, use_mod, (uint64_t)check_t * n_per_thread);
    uint32_t gpu_val = h_out[(size_t)check_t * n_per_thread];
    printf("CPU brute-force check (thread %d): expected=%u gpu=%u -> %s\n",
           check_t, expected_start, gpu_val, expected_start == gpu_val ? "MATCH" : "MISMATCH");

    // Duplicate-start check
    std::unordered_set<uint32_t> seen;
    bool dup = false;
    for (int t = 0; t < num_threads; t++) {
        uint32_t v = h_out[(size_t)t * n_per_thread];
        if (seen.count(v)) { dup = true; printf("Duplicate start value at thread %d\n", t); }
        seen.insert(v);
    }
    printf(dup ? "Uniqueness check: FAILED\n" : "Uniqueness check: PASSED\n");
}

void verify_and_print_64(uint64_t *h_out, int num_threads, int n_per_thread,
                          uint64_t a, uint64_t c, uint64_t m, bool use_mod, uint64_t seed) {
    int show = n_per_thread < 5 ? n_per_thread : 5;
    printf("\n--- Per-thread output (first %d values each) ---\n", show);
    for (int t = 0; t < num_threads; t++) {
        printf("Thread %2d: ", t);
        for (int i = 0; i < show; i++) printf("%llu ", (unsigned long long)h_out[(size_t)t * n_per_thread + i]);
        printf("\n");
    }

    bool boundary_ok = true;
    for (int t = 1; t < num_threads; t++) {
        uint64_t last_prev = h_out[(size_t)(t - 1) * n_per_thread + (n_per_thread - 1)];
        uint64_t first_this = h_out[(size_t)t * n_per_thread];
        uint64_t expected = use_mod ? (uint64_t)(((__uint128_t)a * last_prev + c) % m)
                                     : a * last_prev + c;
        if (expected != first_this) { boundary_ok = false; printf("MISMATCH at boundary %d->%d\n", t-1, t); }
    }
    printf(boundary_ok ? "Boundary check: PASSED\n" : "Boundary check: FAILED\n");

    int check_t = num_threads > 1 ? num_threads / 2 : 0;
    uint64_t expected_start = cpu_brute_force_64(seed, a, c, m, use_mod, (uint64_t)check_t * n_per_thread);
    uint64_t gpu_val = h_out[(size_t)check_t * n_per_thread];
    printf("CPU brute-force check (thread %d): expected=%llu gpu=%llu -> %s\n",
           check_t, (unsigned long long)expected_start, (unsigned long long)gpu_val,
           expected_start == gpu_val ? "MATCH" : "MISMATCH");

    std::unordered_set<uint64_t> seen;
    bool dup = false;
    for (int t = 0; t < num_threads; t++) {
        uint64_t v = h_out[(size_t)t * n_per_thread];
        if (seen.count(v)) { dup = true; printf("Duplicate start value at thread %d\n", t); }
        seen.insert(v);
    }
    printf(dup ? "Uniqueness check: FAILED\n" : "Uniqueness check: PASSED\n");
}

void print_only_128(unsigned __int128 *h_out, int num_threads, int n_per_thread) {
    int show = n_per_thread < 5 ? n_per_thread : 5;
    printf("\n--- Per-thread output (low 64 bits shown, first %d values each) ---\n", show);
    for (int t = 0; t < num_threads; t++) {
        printf("Thread %2d: ", t);
        for (int i = 0; i < show; i++) {
            unsigned __int128 v = h_out[(size_t)t * n_per_thread + i];
            printf("%llu ", (unsigned long long)v); // low 64 bits only (printf can't do 128-bit)
        }
        printf("\n");
    }
    printf("(Note: 128-bit values shown truncated to low 64 bits for display only.)\n");
}

void print_only_256(uint256_t *h_out, int num_threads, int n_per_thread) {
    int show = n_per_thread < 3 ? n_per_thread : 3;
    printf("\n--- Per-thread output (first %d values each) ---\n", show);
    for (int t = 0; t < num_threads; t++) {
        printf("Thread %2d:\n", t);
        for (int i = 0; i < show; i++) {
            printf("   ");
            u256_print(h_out[(size_t)t * n_per_thread + i]);
            printf("\n");
        }
    }
}

// ============================================================
//  Main: interactive input + dispatch to correct bit-width path
// ============================================================
int main()
{
    int bits;
    printf("=== Parallel LCG on GPU (jump-ahead based) ===\n");
    printf("Choose bit-width for LCG state (32 / 64 / 128 / 256): ");
    scanf("%d", &bits);

    unsigned long long a_in, c_in, seed_in, m_in = 0;
    int use_custom_mod = 0;

    printf("Enter multiplier (a): ");
    scanf("%llu", &a_in);
    printf("Enter increment (c): ");
    scanf("%llu", &c_in);
    printf("Enter seed: ");
    scanf("%llu", &seed_in);

    if (bits == 32 || bits == 64) {
        printf("Use custom modulus instead of 2^%d? (1 = yes, 0 = no): ", bits);
        scanf("%d", &use_custom_mod);
        if (use_custom_mod) {
            printf("Enter modulus (m): ");
            scanf("%llu", &m_in);
        }
    } else {
        printf("Note: for %d-bit mode, modulus is fixed at 2^%d (required for GPU efficiency).\n", bits, bits);
    }

    int num_threads, n_per_thread;
    printf("Enter number of GPU threads/cores to use (e.g. 50): ");
    scanf("%d", &num_threads);
    printf("Enter how many random numbers each thread should generate: ");
    scanf("%d", &n_per_thread);

    int threadsPerBlock = 32; // one warp
    int blocks = (num_threads + threadsPerBlock - 1) / threadsPerBlock;

    // ---------------- 32-bit ----------------
    if (bits == 32) {
        uint32_t *d_out;
        size_t bytes = (size_t)num_threads * n_per_thread * sizeof(uint32_t);
        cudaMalloc(&d_out, bytes);
        uint32_t m = use_custom_mod ? (uint32_t)m_in : 0;

        lcg_kernel<uint32_t, uint64_t><<<blocks, threadsPerBlock>>>(
            (uint32_t)seed_in, (uint32_t)a_in, (uint32_t)c_in, m,
            use_custom_mod, n_per_thread, num_threads, d_out);
        cudaDeviceSynchronize();

        uint32_t *h_out = new uint32_t[(size_t)num_threads * n_per_thread];
        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

        verify_and_print_32(h_out, num_threads, n_per_thread,
                             (uint32_t)a_in, (uint32_t)c_in, m, use_custom_mod, (uint32_t)seed_in);

        cudaFree(d_out); delete[] h_out;
    }
    // ---------------- 64-bit ----------------
    else if (bits == 64) {
        uint64_t *d_out;
        size_t bytes = (size_t)num_threads * n_per_thread * sizeof(uint64_t);
        cudaMalloc(&d_out, bytes);
        uint64_t m = use_custom_mod ? m_in : 0ULL;

        lcg_kernel<uint64_t, __uint128_t><<<blocks, threadsPerBlock>>>(
            (uint64_t)seed_in, (uint64_t)a_in, (uint64_t)c_in, m,
            use_custom_mod, n_per_thread, num_threads, d_out);
        cudaDeviceSynchronize();

        uint64_t *h_out = new uint64_t[(size_t)num_threads * n_per_thread];
        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

        verify_and_print_64(h_out, num_threads, n_per_thread,
                             (uint64_t)a_in, (uint64_t)c_in, m, use_custom_mod, (uint64_t)seed_in);

        cudaFree(d_out); delete[] h_out;
    }
    // ---------------- 128-bit ----------------
    else if (bits == 128) {
        unsigned __int128 *d_out;
        size_t bytes = (size_t)num_threads * n_per_thread * sizeof(unsigned __int128);
        cudaMalloc(&d_out, bytes);

        lcg_kernel<unsigned __int128, unsigned __int128><<<blocks, threadsPerBlock>>>(
            (unsigned __int128)seed_in, (unsigned __int128)a_in, (unsigned __int128)c_in,
            (unsigned __int128)0, false, n_per_thread, num_threads, d_out);
        cudaDeviceSynchronize();

        unsigned __int128 *h_out = new unsigned __int128[(size_t)num_threads * n_per_thread];
        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

        print_only_128(h_out, num_threads, n_per_thread);
        printf("(Note: 128-bit custom modulus not supported in this build; modulus fixed at 2^128.)\n");

        cudaFree(d_out); delete[] h_out;
    }
    // ---------------- 256-bit ----------------
    else if (bits == 256) {
        uint256_t *d_out;
        size_t bytes = (size_t)num_threads * n_per_thread * sizeof(uint256_t);
        cudaMalloc(&d_out, bytes);

        uint256_t seed256 = u256_from_u64(seed_in);
        uint256_t a256    = u256_from_u64(a_in);
        uint256_t c256    = u256_from_u64(c_in);

        lcg_kernel_256<<<blocks, threadsPerBlock>>>(seed256, a256, c256, n_per_thread, num_threads, d_out);
        cudaDeviceSynchronize();

        uint256_t *h_out = new uint256_t[(size_t)num_threads * n_per_thread];
        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

        print_only_256(h_out, num_threads, n_per_thread);
        printf("(Note: 256-bit custom modulus not supported in this build; modulus fixed at 2^256.)\n");

        cudaFree(d_out); delete[] h_out;
    }
    else {
        printf("Unsupported bit-width. Choose 32, 64, 128, or 256.\n");
        return 1;
    }

    printf("\nDone.\n");
    return 0;
}
