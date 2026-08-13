// ============================================================
//  Parallel LCG on GPU with Jump-Ahead Function
//  ------------------------------------------------------------
//  Features:
//    - Supports 32 / 64 / 128 / 256-bit LCG state
//    - User-defined a (multiplier), c (increment), seed, modulus
//    - Jump-ahead function for O(log k) parallel initialization
//    - Detailed on-screen instructions BEFORE asking for input
//    - Input range validation (prevents silent truncation bugs)
//    - Hull-Dobell full-period checker (power-of-two modulus)
//    - GPU profiling: kernel time, memory transfer time, total time
//    - Correctness verification: boundary check, CPU brute-force
//      cross-check, duplicate-start check
//    - CSV export of full output for Excel
// ============================================================

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
//  NOTE: This is the NO-JUMP-FUNCTION comparison version.
//  The O(log k) jump-ahead functions have been intentionally
//  removed. Each thread instead finds its starting point by
//  stepping the LCG one iteration at a time (O(k) per thread) --
//  see the kernels below. Compare timing against lcg_full_v2.cu.
// ============================================================

// ============================================================
//  Generic kernel (32/64/128-bit)
//  *** NO-JUMP VERSION ***
//  Each thread finds its starting point by stepping the LCG
//  ONE ITERATION AT A TIME from the seed, all the way up to its
//  start_index. This is O(k) per thread instead of O(log k) --
//  intentionally naive, for benchmarking against the jump version.
// ============================================================
template <typename T, typename T2>
__global__ void lcg_kernel(T seed, T a, T c, T m, bool use_mod,
                            int n_per_thread, int num_threads, T *out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    uint64_t start_index = (uint64_t)tid * n_per_thread;

    // --- NO JUMP FUNCTION: step forward one at a time ---
    T state = seed;
    for (uint64_t s = 0; s < start_index; s++) {
        if (use_mod) state = (T)(((T2)a * (T2)state + (T2)c) % (T2)m);
        else         state = a * state + c;
    }

    for (int i = 0; i < n_per_thread; i++) {
        out[(size_t)tid * n_per_thread + i] = state;
        if (use_mod) state = (T)(((T2)a * (T2)state + (T2)c) % (T2)m);
        else         state = a * state + c;
    }
}

__global__ void lcg_kernel_256(uint256_t seed, uint256_t a, uint256_t c,
                                int n_per_thread, int num_threads, uint256_t *out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    uint64_t start_index = (uint64_t)tid * n_per_thread;

    // --- NO JUMP FUNCTION: step forward one at a time ---
    uint256_t state = seed;
    for (uint64_t s = 0; s < start_index; s++) {
        state = u256_add(u256_mul(a, state), c);
    }

    for (int i = 0; i < n_per_thread; i++) {
        out[(size_t)tid * n_per_thread + i] = state;
        state = u256_add(u256_mul(a, state), c);
    }
}

// ============================================================
//  CPU brute-force reference (for verification)
// ============================================================
uint64_t cpu_brute_force_64(uint64_t seed, uint64_t a, uint64_t c, uint64_t m,
                             bool use_mod, uint64_t steps) {
    uint64_t x = seed;
    for (uint64_t i = 0; i < steps; i++) {
        if (use_mod) x = (uint64_t)(((__uint128_t)a * x + c) % m);
        else         x = a * x + c;
    }
    return x;
}

uint32_t cpu_brute_force_32(uint32_t seed, uint32_t a, uint32_t c, uint32_t m,
                             bool use_mod, uint64_t steps) {
    uint32_t x = seed;
    for (uint64_t i = 0; i < steps; i++) {
        if (use_mod) x = (uint32_t)(((uint64_t)a * x + c) % m);
        else         x = a * x + c;
    }
    return x;
}

// ============================================================
//  Hull-Dobell full-period checker (power-of-two modulus only)
// ============================================================
void check_hull_dobell_pow2(unsigned long long a, unsigned long long c, int bits) {
    printf("\n--- Hull-Dobell full-period check (modulus = 2^%d) ---\n", bits);
    bool c_ok = (c & 1ULL) == 1ULL;
    bool a_ok = ((a - 1) % 4ULL == 0ULL);

    if (c_ok) printf("  [OK]   c is odd -> gcd(c, m) = 1\n");
    else      printf("  [WARN] c is even -> gcd(c, m) != 1. Full period NOT guaranteed.\n");

    if (a_ok) printf("  [OK]   (a - 1) is divisible by 4\n");
    else      printf("  [WARN] (a - 1) is NOT divisible by 4. Full period NOT guaranteed.\n");

    if (c_ok && a_ok)
        printf("  Result: PASSES Hull-Dobell -> generator will achieve full period (2^%d).\n", bits);
    else
        printf("  Result: FAILS Hull-Dobell -> expect short cycles / fixed-point collapse.\n");
}

// ============================================================
//  Range validation
// ============================================================
bool value_fits_bits(unsigned long long v, int bits) {
    if (bits >= 64) return true;
    unsigned long long max_val = (bits == 32) ? 0xFFFFFFFFULL : 0xFFFFFFFFFFFFFFFFULL;
    return v <= max_val;
}

// ============================================================
//  Verification + printing helpers
// ============================================================
void verify_and_print_32(uint32_t *h_out, int num_threads, int n_per_thread,
                          uint32_t a, uint32_t c, uint32_t m, bool use_mod, uint32_t seed) {
    int show = n_per_thread < 10 ? n_per_thread : 10;
    printf("\n--- Per-thread output (first %d values each) ---\n", show);
    for (int t = 0; t < num_threads; t++) {
        printf("Thread %2d: ", t);
        for (int i = 0; i < show; i++) printf("%u ", h_out[(size_t)t * n_per_thread + i]);
        printf("\n");
    }

    bool boundary_ok = true;
    for (int t = 1; t < num_threads; t++) {
        uint32_t last_prev = h_out[(size_t)(t - 1) * n_per_thread + (n_per_thread - 1)];
        uint32_t first_this = h_out[(size_t)t * n_per_thread];
        uint32_t expected = use_mod ? (uint32_t)(((uint64_t)a * last_prev + c) % m)
                                     : a * last_prev + c;
        if (expected != first_this) { boundary_ok = false; printf("MISMATCH at boundary %d->%d\n", t-1, t); }
    }
    printf(boundary_ok ? "Boundary check: PASSED\n" : "Boundary check: FAILED\n");

    int check_t = num_threads > 1 ? num_threads / 2 : 0;
    uint32_t expected_start = cpu_brute_force_32(seed, a, c, m, use_mod, (uint64_t)check_t * n_per_thread);
    uint32_t gpu_val = h_out[(size_t)check_t * n_per_thread];
    printf("CPU brute-force check (thread %d): expected=%u gpu=%u -> %s\n",
           check_t, expected_start, gpu_val, expected_start == gpu_val ? "MATCH" : "MISMATCH");

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
    int show = n_per_thread < 10 ? n_per_thread : 10;
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
    int show = n_per_thread < 10 ? n_per_thread : 10;
    printf("\n--- Per-thread output (low 64 bits shown, first %d values each) ---\n", show);
    for (int t = 0; t < num_threads; t++) {
        printf("Thread %2d: ", t);
        for (int i = 0; i < show; i++) {
            unsigned __int128 v = h_out[(size_t)t * n_per_thread + i];
            printf("%llu ", (unsigned long long)v);
        }
        printf("\n");
    }
    printf("(Note: 128-bit values shown truncated to low 64 bits for display only.)\n");
}

void print_only_256(uint256_t *h_out, int num_threads, int n_per_thread) {
    int show = n_per_thread < 10 ? n_per_thread : 10;
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
//  Binary export -- the recommended format for millions+ numbers.
//  Format: [header][raw values, tightly packed]
//  Header (all uint32_t, 16 bytes total):
//    magic (0x4C434731 = "LCG1"), bits, num_threads, n_per_thread
//  Then num_threads * n_per_thread values written back-to-back,
//  each of exactly `bits/8` bytes (native binary layout).
//  256-bit values are written as 4 x uint64_t limbs (little-endian),
//  i.e. also 32 bytes each, no header change needed.
// ============================================================
struct BinHeader {
    uint32_t magic;       // 0x4C434731
    uint32_t bits;
    uint32_t num_threads;
    uint32_t n_per_thread;
};

template <typename T>
void export_to_binary(const char *filename, T *h_out, int num_threads, int n_per_thread, int bits) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) { printf("ERROR: could not open %s for writing.\n", filename); return; }

    BinHeader hdr{0x4C434731u, (uint32_t)bits, (uint32_t)num_threads, (uint32_t)n_per_thread};
    fwrite(&hdr, sizeof(BinHeader), 1, fp);

    size_t total = (size_t)num_threads * n_per_thread;
    fwrite(h_out, sizeof(T), total, fp); // single bulk write -- fast even for 100M+ numbers

    fclose(fp);
    double mb = (double)(total * sizeof(T)) / (1024.0 * 1024.0);
    printf("Full output (%zu numbers, %.2f MB) written to %s (binary)\n", total, mb, filename);
}

void export_to_binary_256(const char *filename, uint256_t *h_out, int num_threads, int n_per_thread) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) { printf("ERROR: could not open %s for writing.\n", filename); return; }

    BinHeader hdr{0x4C434731u, 256u, (uint32_t)num_threads, (uint32_t)n_per_thread};
    fwrite(&hdr, sizeof(BinHeader), 1, fp);

    size_t total = (size_t)num_threads * n_per_thread;
    fwrite(h_out, sizeof(uint256_t), total, fp); // each entry = 4 x uint64_t limbs = 32 bytes

    fclose(fp);
    double mb = (double)(total * sizeof(uint256_t)) / (1024.0 * 1024.0);
    printf("Full output (%zu numbers, %.2f MB) written to %s (binary)\n", total, mb, filename);
}

// ============================================================
//  CSV export (templated for 32/64/128-bit; 256-bit separate)
// ============================================================
template <typename T>
void export_to_csv(const char *filename, T *h_out, int num_threads, int n_per_thread) {
    std::ofstream file(filename);
    file << "thread,index,value\n";
    for (int t = 0; t < num_threads; t++) {
        for (int i = 0; i < n_per_thread; i++) {
            file << t << "," << i << "," << (unsigned long long)h_out[(size_t)t * n_per_thread + i] << "\n";
        }
    }
    file.close();
    printf("Full output written to %s\n", filename);
}

void export_to_csv_256(const char *filename, uint256_t *h_out, int num_threads, int n_per_thread) {
    std::ofstream file(filename);
    file << "thread,index,limb0,limb1,limb2,limb3\n";
    for (int t = 0; t < num_threads; t++) {
        for (int i = 0; i < n_per_thread; i++) {
            uint256_t v = h_out[(size_t)t * n_per_thread + i];
            file << t << "," << i << "," << v.w[0] << "," << v.w[1] << "," << v.w[2] << "," << v.w[3] << "\n";
        }
    }
    file.close();
    printf("Full output written to %s\n", filename);
}

// ============================================================
//  Pre-input instructions banner
// ============================================================
void print_instructions() {
    printf("================================================================\n");
    printf(" PARALLEL LCG ON GPU -- *** NO-JUMP-FUNCTION VERSION ***\n");
    printf(" (for benchmarking against the jump-ahead version)\n");
    printf("================================================================\n");
    printf("This program generates pseudo-random numbers using:\n");
    printf("   X(n+1) = (a * X(n) + c) mod m\n");
    printf("Unlike the jump-ahead version, each thread here finds its own\n");
    printf("starting point by stepping the LCG ONE ITERATION AT A TIME from\n");
    printf("the seed, all the way up to its assigned start index. This is\n");
    printf("O(k) work per thread instead of O(log k) -- intentionally naive,\n");
    printf("to demonstrate why the jump function matters for parallel LCG.\n\n");
    printf("WARNING: with large thread counts and/or large values-per-thread,\n");
    printf("         later threads may need MILLIONS of sequential steps just\n");
    printf("         to reach their starting point. Expect this version to be\n");
    printf("         dramatically slower than the jump-function version as\n");
    printf("         num_threads x n_per_thread grows.\n\n");

    printf("You will be asked for the following:\n\n");

    printf(" 1) BIT-WIDTH (32 / 64 / 128 / 256)\n");
    printf("    Determines the integer size used for the LCG state.\n");
    printf("    - 32/64-bit : full custom modulus supported.\n");
    printf("    - 128/256-bit: modulus is FIXED at 2^bits (no custom m).\n\n");

    printf(" 2) MULTIPLIER (a)\n");
    printf("    Must fit within the chosen bit-width, e.g.:\n");
    printf("      32-bit max value: 4294967295\n");
    printf("      64-bit max value: 18446744073709551615\n");
    printf("    Values that don't fit will be REJECTED (not silently truncated).\n");
    printf("    Tip: for a full-period generator with modulus 2^bits,\n");
    printf("         (a - 1) should be divisible by 4.\n\n");

    printf(" 3) INCREMENT (c)\n");
    printf("    Same size rule as 'a'.\n");
    printf("    Tip: for full period with modulus 2^bits, c must be ODD.\n\n");

    printf(" 4) SEED\n");
    printf("    Starting value of the sequence. Any value within bit-width.\n\n");

    printf(" 5) CUSTOM MODULUS (32/64-bit only)\n");
    printf("    Choose 'no' to use the fast default m = 2^bits (recommended,\n");
    printf("    free modulo via integer wraparound on GPU).\n");
    printf("    Choose 'yes' to enter your own m (e.g. to replicate a known\n");
    printf("    generator, or constrain output to a specific range).\n\n");

    printf(" 6) NUMBER OF GPU THREADS/CORES\n");
    printf("    How many independent parallel streams to launch.\n");
    printf("    Each thread gets a non-overlapping block of the sequence.\n\n");

    printf(" 7) VALUES PER THREAD\n");
    printf("    How many random numbers each thread should generate.\n");
    printf("    Total numbers produced = threads x values_per_thread.\n\n");

    printf("After input, the program will:\n");
    printf("  - Validate that a/c/seed fit the chosen bit-width\n");
    printf("  - Run a Hull-Dobell full-period check (default modulus only)\n");
    printf("  - Launch the GPU kernel and PROFILE timing (H2D copy,\n");
    printf("    kernel execution, D2H copy, total wall time)\n");
    printf("  - Print sample output per thread\n");
    printf("  - Run correctness checks (boundary, brute-force, uniqueness)\n");
    printf("  - Export the full output to lcg_output_nojump.csv\n");
    printf("================================================================\n\n");
}

// ============================================================
//  Main
// ============================================================
int main()
{
    print_instructions();

    int bits;
    printf("Choose bit-width for LCG state (32 / 64 / 128 / 256): ");
    scanf("%d", &bits);

    unsigned long long a_in, c_in, seed_in, m_in = 0;
    int use_custom_mod = 0;

    // ---- Ask modulus choice FIRST, so Hull-Dobell hints for a/c can be tailored ----
    if (bits == 32 || bits == 64) {
        printf("Use custom modulus instead of 2^%d? (1 = yes, 0 = no): ", bits);
        scanf("%d", &use_custom_mod);
        if (use_custom_mod) {
            printf("Enter modulus (m): ");
            scanf("%llu", &m_in);
        }
    } else {
        use_custom_mod = 0;
        printf("Note: for %d-bit mode, modulus is fixed at 2^%d (required for GPU efficiency).\n", bits, bits);
    }

    // ---- Per-field Hull-Dobell hint + input for 'a' ----
    printf("\n[Hull-Dobell condition for 'a']\n");
    if (!use_custom_mod) {
        printf("  Modulus m = 2^%d. For FULL PERIOD: (a - 1) must be divisible by 4\n", bits);
        printf("  (since 4 divides m whenever bits >= 2). Pick 'a' accordingly, e.g. a mod 4 == 1.\n");
    } else {
        printf("  Modulus m is custom (%llu). For FULL PERIOD: (a - 1) must be divisible by\n", m_in);
        printf("  EVERY prime factor of m, and by 4 if 4 divides m. This program does not\n");
        printf("  auto-factor custom m, so verify this yourself if exact period matters.\n");
    }
    printf("Enter multiplier (a): ");
    scanf("%llu", &a_in);

    // ---- Per-field Hull-Dobell hint + input for 'c' ----
    printf("\n[Hull-Dobell condition for 'c']\n");
    if (!use_custom_mod) {
        printf("  Modulus m = 2^%d. For FULL PERIOD: c and m must be coprime,\n", bits);
        printf("  i.e. gcd(c, m) = 1 -> simply pick 'c' to be ODD.\n");
    } else {
        printf("  Modulus m is custom (%llu). For FULL PERIOD: gcd(c, m) must equal 1.\n", m_in);
    }
    printf("Enter increment (c): ");
    scanf("%llu", &c_in);

    // ---- Seed (no Hull-Dobell condition applies to seed) ----
    printf("\n[Seed]\n  Any starting value within the chosen bit-width. Does not affect period length.\n");
    printf("Enter seed: ");
    scanf("%llu", &seed_in);

    // ---- Range validation: catch silent truncation before it happens ----
    if (bits == 32 || bits == 64) {
        if (!value_fits_bits(a_in, bits)) {
            printf("ERROR: multiplier 'a' = %llu does not fit in %d bits (max %llu).\n",
                   a_in, bits, bits == 32 ? 0xFFFFFFFFULL : 0xFFFFFFFFFFFFFFFFULL);
            printf("Choose a larger bit-width, or enter a smaller 'a'. Aborting.\n");
            return 1;
        }
        if (!value_fits_bits(c_in, bits)) {
            printf("ERROR: increment 'c' = %llu does not fit in %d bits (max %llu).\n",
                   c_in, bits, bits == 32 ? 0xFFFFFFFFULL : 0xFFFFFFFFFFFFFFFFULL);
            printf("Choose a larger bit-width, or enter a smaller 'c'. Aborting.\n");
            return 1;
        }
        if (!value_fits_bits(seed_in, bits)) {
            printf("ERROR: seed = %llu does not fit in %d bits. Aborting.\n", seed_in, bits);
            return 1;
        }
    }

    // ---- Final Hull-Dobell summary (power-of-two modulus case only) ----
    if (!use_custom_mod) {
        check_hull_dobell_pow2(a_in, c_in, bits);
    }

    int num_threads, n_per_thread;
    printf("Enter number of GPU threads/cores to use (e.g. 50): ");
    scanf("%d", &num_threads);
    printf("Enter how many random numbers each thread should generate: ");
    scanf("%d", &n_per_thread);

    // ---- Output format selection ----
    long long total_count = (long long)num_threads * n_per_thread;
    printf("\n[Output format]\n");
    printf("  Total numbers to generate: %lld\n", total_count);
    if (total_count > 1048576) {
        printf("  NOTE: this exceeds Excel's per-sheet row limit (1,048,576).\n");
        printf("        CSV/Excel export will NOT be able to hold all rows in one sheet.\n");
        printf("        Binary export has no such limit and is strongly recommended.\n");
    }
    printf("  1 = Binary only (.bin)  -- fastest, smallest, handles unlimited size [RECOMMENDED]\n");
    printf("  2 = CSV only (.csv)     -- human-readable, capped by Excel's row limit\n");
    printf("  3 = Both\n");
    int export_choice = 1;
    printf("Choose export format (1/2/3): ");
    scanf("%d", &export_choice);
    bool do_binary = (export_choice == 1 || export_choice == 3);
    bool do_csv    = (export_choice == 2 || export_choice == 3);

    int threadsPerBlock = 32; // one warp
    int blocks = (num_threads + threadsPerBlock - 1) / threadsPerBlock;

    // ---- Profiling events ----
    cudaEvent_t ev_start_total, ev_start_h2d, ev_end_h2d, ev_start_kernel, ev_end_kernel, ev_end_d2h, ev_end_total;
    cudaEventCreate(&ev_start_total);
    cudaEventCreate(&ev_start_h2d);
    cudaEventCreate(&ev_end_h2d);
    cudaEventCreate(&ev_start_kernel);
    cudaEventCreate(&ev_end_kernel);
    cudaEventCreate(&ev_end_d2h);
    cudaEventCreate(&ev_end_total);

    cudaEventRecord(ev_start_total);

    // ---------------- 32-bit ----------------
    if (bits == 32) {
        uint32_t *d_out;
        size_t bytes = (size_t)num_threads * n_per_thread * sizeof(uint32_t);

        cudaEventRecord(ev_start_h2d);
        cudaMalloc(&d_out, bytes); // no host->device input data needed (params passed by value)
        cudaEventRecord(ev_end_h2d);

        uint32_t m = use_custom_mod ? (uint32_t)m_in : 0;

        cudaEventRecord(ev_start_kernel);
        lcg_kernel<uint32_t, uint64_t><<<blocks, threadsPerBlock>>>(
            (uint32_t)seed_in, (uint32_t)a_in, (uint32_t)c_in, m,
            use_custom_mod, n_per_thread, num_threads, d_out);
        cudaEventRecord(ev_end_kernel);
        cudaEventSynchronize(ev_end_kernel);

        uint32_t *h_out = new uint32_t[(size_t)num_threads * n_per_thread];
        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(ev_end_d2h);
        cudaEventSynchronize(ev_end_d2h);

        verify_and_print_32(h_out, num_threads, n_per_thread,
                             (uint32_t)a_in, (uint32_t)c_in, m, use_custom_mod, (uint32_t)seed_in);
        if (do_binary) export_to_binary<uint32_t>("lcg_output_nojump.bin", h_out, num_threads, n_per_thread, bits);
        if (do_csv)    export_to_csv<uint32_t>("lcg_output_nojump.csv", h_out, num_threads, n_per_thread);

        cudaFree(d_out); delete[] h_out;
    }
    // ---------------- 64-bit ----------------
    else if (bits == 64) {
        uint64_t *d_out;
        size_t bytes = (size_t)num_threads * n_per_thread * sizeof(uint64_t);

        cudaEventRecord(ev_start_h2d);
        cudaMalloc(&d_out, bytes);
        cudaEventRecord(ev_end_h2d);

        uint64_t m = use_custom_mod ? m_in : 0ULL;

        cudaEventRecord(ev_start_kernel);
        lcg_kernel<uint64_t, __uint128_t><<<blocks, threadsPerBlock>>>(
            (uint64_t)seed_in, (uint64_t)a_in, (uint64_t)c_in, m,
            use_custom_mod, n_per_thread, num_threads, d_out);
        cudaEventRecord(ev_end_kernel);
        cudaEventSynchronize(ev_end_kernel);

        uint64_t *h_out = new uint64_t[(size_t)num_threads * n_per_thread];
        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(ev_end_d2h);
        cudaEventSynchronize(ev_end_d2h);

        verify_and_print_64(h_out, num_threads, n_per_thread,
                             (uint64_t)a_in, (uint64_t)c_in, m, use_custom_mod, (uint64_t)seed_in);
        if (do_binary) export_to_binary<uint64_t>("lcg_output_nojump.bin", h_out, num_threads, n_per_thread, bits);
        if (do_csv)    export_to_csv<uint64_t>("lcg_output_nojump.csv", h_out, num_threads, n_per_thread);

        cudaFree(d_out); delete[] h_out;
    }
    // ---------------- 128-bit ----------------
    else if (bits == 128) {
        unsigned __int128 *d_out;
        size_t bytes = (size_t)num_threads * n_per_thread * sizeof(unsigned __int128);

        cudaEventRecord(ev_start_h2d);
        cudaMalloc(&d_out, bytes);
        cudaEventRecord(ev_end_h2d);

        cudaEventRecord(ev_start_kernel);
        lcg_kernel<unsigned __int128, unsigned __int128><<<blocks, threadsPerBlock>>>(
            (unsigned __int128)seed_in, (unsigned __int128)a_in, (unsigned __int128)c_in,
            (unsigned __int128)0, false, n_per_thread, num_threads, d_out);
        cudaEventRecord(ev_end_kernel);
        cudaEventSynchronize(ev_end_kernel);

        unsigned __int128 *h_out = new unsigned __int128[(size_t)num_threads * n_per_thread];
        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(ev_end_d2h);
        cudaEventSynchronize(ev_end_d2h);

        print_only_128(h_out, num_threads, n_per_thread);
        printf("(Note: 128-bit custom modulus not supported; modulus fixed at 2^128.)\n");
        if (do_binary) export_to_binary<unsigned __int128>("lcg_output_nojump.bin", h_out, num_threads, n_per_thread, bits);
        if (do_csv)    export_to_csv<unsigned __int128>("lcg_output_nojump.csv", h_out, num_threads, n_per_thread);

        cudaFree(d_out); delete[] h_out;
    }
    // ---------------- 256-bit ----------------
    else if (bits == 256) {
        uint256_t *d_out;
        size_t bytes = (size_t)num_threads * n_per_thread * sizeof(uint256_t);

        cudaEventRecord(ev_start_h2d);
        cudaMalloc(&d_out, bytes);
        cudaEventRecord(ev_end_h2d);

        uint256_t seed256 = u256_from_u64(seed_in);
        uint256_t a256    = u256_from_u64(a_in);
        uint256_t c256    = u256_from_u64(c_in);

        cudaEventRecord(ev_start_kernel);
        lcg_kernel_256<<<blocks, threadsPerBlock>>>(seed256, a256, c256, n_per_thread, num_threads, d_out);
        cudaEventRecord(ev_end_kernel);
        cudaEventSynchronize(ev_end_kernel);

        uint256_t *h_out = new uint256_t[(size_t)num_threads * n_per_thread];
        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(ev_end_d2h);
        cudaEventSynchronize(ev_end_d2h);

        print_only_256(h_out, num_threads, n_per_thread);
        printf("(Note: 256-bit custom modulus not supported; modulus fixed at 2^256.)\n");
        if (do_binary) export_to_binary_256("lcg_output_nojump.bin", h_out, num_threads, n_per_thread);
        if (do_csv)    export_to_csv_256("lcg_output_nojump.csv", h_out, num_threads, n_per_thread);

        cudaFree(d_out); delete[] h_out;
    }
    else {
        printf("Unsupported bit-width. Choose 32, 64, 128, or 256.\n");
        return 1;
    }

    cudaEventRecord(ev_end_total);
    cudaEventSynchronize(ev_end_total);

    // ---- Print profiling summary ----
    float t_h2d = 0, t_kernel = 0, t_d2h = 0, t_total = 0;
    cudaEventElapsedTime(&t_h2d, ev_start_h2d, ev_end_h2d);
    cudaEventElapsedTime(&t_kernel, ev_start_kernel, ev_end_kernel);
    cudaEventElapsedTime(&t_d2h, ev_end_kernel, ev_end_d2h);
    cudaEventElapsedTime(&t_total, ev_start_total, ev_end_total);

    long long total_numbers = (long long)num_threads * n_per_thread;

    printf("\n--- GPU Profiling Summary (NO-JUMP-FUNCTION VERSION) ---\n");
    printf("  Device malloc time      : %8.4f ms\n", t_h2d);
    printf("  Kernel execution time   : %8.4f ms\n", t_kernel);
    printf("  Device->Host copy time  : %8.4f ms\n", t_d2h);
    printf("  Total wall time         : %8.4f ms\n", t_total);
    printf("  Total numbers generated : %lld\n", total_numbers);
    if (t_kernel > 0.0f) {
        double throughput = (double)total_numbers / (t_kernel / 1000.0);
        printf("  Kernel throughput       : %.2f numbers/sec\n", throughput);
    }

    cudaEventDestroy(ev_start_total);
    cudaEventDestroy(ev_start_h2d);
    cudaEventDestroy(ev_end_h2d);
    cudaEventDestroy(ev_start_kernel);
    cudaEventDestroy(ev_end_kernel);
    cudaEventDestroy(ev_end_d2h);
    cudaEventDestroy(ev_end_total);

    printf("\nDone.\n");
    return 0;
}
