// ============================================================
//  Compact Parallel LCG on GPU (jump-ahead based) -- 32/64-bit
//  X(n+1) = (a*X(n) + c) mod m | jump: X(n+k) = A_k*X(n) + C_k
// ============================================================
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <unordered_set>
#include <cuda_runtime.h>

// ---- Jump function: computes (A,C) for a k-step jump ----
template <typename T>
__host__ __device__ void lcg_jump(T a, T c, uint64_t k, T &A, T &C) {
    A = 1; C = 0;
    while (k > 0) {
        if (k & 1) { A = A * a; C = C * a + c; }
        c = c * (a + 1);
        a = a * a;
        k >>= 1;
    }
}

// ---- Kernel: each thread jumps to its start, then runs locally ----
template <typename T, typename T2>
__global__ void lcg_kernel(T seed, T a, T c, T m, bool use_mod, int n, int nt, T *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nt) return;
    T A, C;
    lcg_jump<T>(a, c, (uint64_t)tid * n, A, C);
    T state = use_mod ? (T)(((T2)A * seed + C) % m) : A * seed + C;
    for (int i = 0; i < n; i++) {
        out[(size_t)tid * n + i] = state;
        state = use_mod ? (T)(((T2)a * state + c) % m) : a * state + c;
    }
}

// ---- Hull-Dobell check (power-of-two modulus) ----
void check_hull_dobell(unsigned long long a, unsigned long long c, int bits) {
    bool c_ok = (c & 1) == 1, a_ok = ((a - 1) % 4) == 0;
    printf("\nHull-Dobell check (m=2^%d): c odd=%s, (a-1)%%4==0=%s -> %s\n",
           bits, c_ok ? "OK" : "FAIL", a_ok ? "OK" : "FAIL",
           (c_ok && a_ok) ? "FULL PERIOD" : "SHORT PERIOD RISK");
}

// ---- Verify + print + export ----
template <typename T>
void verify_print_export(T *h, int nt, int n, T a, T c, T m, bool use_mod, T seed) {
    int show = n < 10 ? n : 10;
    printf("\n--- First %d values per thread ---\n", show);
    for (int t = 0; t < nt; t++) {
        printf("Thread %2d: ", t);
        for (int i = 0; i < show; i++) printf("%llu ", (unsigned long long)h[(size_t)t * n + i]);
        printf("\n");
    }

    bool ok = true;
    for (int t = 1; t < nt; t++) {
        T last = h[(size_t)(t - 1) * n + (n - 1)];
        T expect = use_mod ? (T)(((unsigned __int128)a * last + c) % m) : a * last + c;
        if (expect != h[(size_t)t * n]) ok = false;
    }
    printf("Boundary check: %s\n", ok ? "PASSED" : "FAILED");

    std::unordered_set<T> seen;
    bool dup = false;
    for (int t = 0; t < nt; t++) {
        T v = h[(size_t)t * n];
        if (seen.count(v)) dup = true;
        seen.insert(v);
    }
    printf("Uniqueness check: %s\n", dup ? "FAILED (values repeat)" : "PASSED");

    std::ofstream f("lcg_output.csv");
    f << "thread,index,value\n";
    for (int t = 0; t < nt; t++)
        for (int i = 0; i < n; i++)
            f << t << "," << i << "," << (unsigned long long)h[(size_t)t * n + i] << "\n";
    f.close();
    printf("Full output written to lcg_output.csv\n");
}

int main() {
    printf("=== Compact Parallel LCG on GPU (jump-function based) ===\n");
    printf("Formula: X(n+1) = (a*X(n) + c) mod m\n");
    printf("Tips for full period (default modulus m=2^bits): c must be ODD,\n");
    printf("(a-1) must be divisible by 4. Suggested simple values: a=1664525,\n");
    printf("c=1013904223, seed=42.\n\n");

    int bits;
    printf("Bit-width (32/64): "); scanf("%d", &bits);

    unsigned long long a, c, seed, m = 0;
    printf("Multiplier (a): "); scanf("%llu", &a);
    printf("Increment (c): ");  scanf("%llu", &c);
    printf("Seed: ");           scanf("%llu", &seed);

    if (bits == 32 && (a > 0xFFFFFFFFULL || c > 0xFFFFFFFFULL)) {
        printf("ERROR: a or c too large for 32-bit. Use 64-bit or smaller values.\n");
        return 1;
    }

    int use_mod = 0;
    printf("Use custom modulus? (1=yes, 0=no, default 2^%d): ", bits);
    scanf("%d", &use_mod);
    if (use_mod) { printf("Modulus (m): "); scanf("%llu", &m); }
    else check_hull_dobell(a, c, bits);

    int nt, n;
    printf("Number of GPU threads: "); scanf("%d", &nt);
    printf("Values per thread: ");     scanf("%d", &n);

    int tpb = 32, blocks = (nt + tpb - 1) / tpb;
    cudaEvent_t t0, t1, t2, t3;
    cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventCreate(&t2); cudaEventCreate(&t3);
    cudaEventRecord(t0);

    if (bits == 32) {
        uint32_t *d_out, mm = (uint32_t)m;
        size_t bytes = (size_t)nt * n * sizeof(uint32_t);
        cudaMalloc(&d_out, bytes);
        cudaEventRecord(t1);
        lcg_kernel<uint32_t, uint64_t><<<blocks, tpb>>>((uint32_t)seed, (uint32_t)a, (uint32_t)c, mm, use_mod, n, nt, d_out);
        cudaEventRecord(t2); cudaEventSynchronize(t2);
        uint32_t *h_out = new uint32_t[(size_t)nt * n];
        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(t3); cudaEventSynchronize(t3);
        verify_print_export<uint32_t>(h_out, nt, n, (uint32_t)a, (uint32_t)c, mm, use_mod, (uint32_t)seed);
        cudaFree(d_out); delete[] h_out;
    } else {
        uint64_t *d_out, mm = m;
        size_t bytes = (size_t)nt * n * sizeof(uint64_t);
        cudaMalloc(&d_out, bytes);
        cudaEventRecord(t1);
        lcg_kernel<uint64_t, __uint128_t><<<blocks, tpb>>>(seed, a, c, mm, use_mod, n, nt, d_out);
        cudaEventRecord(t2); cudaEventSynchronize(t2);
        uint64_t *h_out = new uint64_t[(size_t)nt * n];
        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(t3); cudaEventSynchronize(t3);
        verify_print_export<uint64_t>(h_out, nt, n, a, c, mm, use_mod, seed);
        cudaFree(d_out); delete[] h_out;
    }

    float malloc_ms, kernel_ms, copy_ms;
    cudaEventElapsedTime(&malloc_ms, t0, t1);
    cudaEventElapsedTime(&kernel_ms, t1, t2);
    cudaEventElapsedTime(&copy_ms, t2, t3);
    printf("\n--- Profiling ---\n");
    printf("Malloc: %.4f ms | Kernel: %.4f ms | D2H copy: %.4f ms | Total numbers: %lld\n",
           malloc_ms, kernel_ms, copy_ms, (long long)nt * n);

    return 0;
}
