#include <hip/hip_runtime.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <tuple>

using namespace std;

template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, int M, int N, int K) {
    __shared__ T tileA[32][32];
    __shared__ T tileB[32][32];

    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;
    T sum = 0;

    for (int t = 0; t < (K + 31) / 32; ++t) {
        if (row < M && t * 32 + threadIdx.x < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * 32 + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0;

        if (t * 32 + threadIdx.y < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * 32 + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int i = 0; i < 32; ++i)
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

template<typename T>
void verifyResult(T *a, T *b, T *c, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            T sum = 0;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            if (abs(c[i * N + j] - sum) > 1e-3) {
                printf("Mismatch at (%d, %d): GPU = %f, CPU = %f\n", i, j, (float)c[i * N + j], (float)sum);
                return;
            }
        }
    }
    cout << "Result is correct!\n";
}

template<typename T>
void executeKernel(T *d_a, T *d_b, T *d_c, int M, int N, int K) {
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);

    hipLaunchKernelGGL(matmul_kernel<T>, grid, block, 0, 0, d_a, d_b, d_c, M, N, K);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);

    float gflops = 2.0f * M * N * K / (milliseconds * 1e6f);
    printf("Time: %.3f ms, GFLOPs: %.2f\n", milliseconds, gflops);

    hipEventDestroy(start);
    hipEventDestroy(stop);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code %s)\n", hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template<typename T>
void copyFromHostToDevice(T *h_a, T *h_b, T *d_a, T *d_b, int M, int N, int K) {
    hipMemcpy(d_a, h_a, M * K * sizeof(T), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, K * N * sizeof(T), hipMemcpyHostToDevice);
}

template<typename T>
void copyFromDeviceToHost(T *d_c, T *h_c, int M, int N) {
    hipMemcpy(h_c, d_c, M * N * sizeof(T), hipMemcpyDeviceToHost);
}

template<typename T>
void deallocateMemory(T *d_a, T *d_b, T *d_c) {
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
}

void cleanUpDevice() {
    hipDeviceReset();
}

std::tuple<int, int, int> parseCmdLineArgs(int argc, char *argv[]) {
    int M = 1024, N = 1024, K = 1024;
    for (int i = 1; i < argc; i++) {
        std::string option(argv[i]);
        std::string value(argv[++i]);
        if (option == "-m") M = std::stoi(value);
        else if (option == "-n") N = std::stoi(value);
        else if (option == "-k") K = std::stoi(value);
    }
    return {M, N, K};
}

int main(int argc, char *argv[]) {
    auto [M, N, K] = parseCmdLineArgs(argc, argv);

    int *h_a = (int *)malloc(M * K * sizeof(int));
    int *h_b = (int *)malloc(K * N * sizeof(int));
    int *h_c = (int *)malloc(M * N * sizeof(int));

    for (int i = 0; i < M * K; i++) h_a[i] = rand() % 10;
    for (int i = 0; i < K * N; i++) h_b[i] = rand() % 10;

    int *d_a, *d_b, *d_c;
    hipMalloc(&d_a, M * K * sizeof(int));
    hipMalloc(&d_b, K * N * sizeof(int));
    hipMalloc(&d_c, M * N * sizeof(int));

    copyFromHostToDevice<int>(h_a, h_b, d_a, d_b, M, N, K);
    executeKernel<int>(d_a, d_b, d_c, M, N, K);
    copyFromDeviceToHost<int>(d_c, h_c, M, N);
    verifyResult<int>(h_a, h_b, h_c, M, N, K);
    deallocateMemory<int>(d_a, d_b, d_c);
    cleanUpDevice();

    free(h_a); free(h_b); free(h_c);

    return 0;
}
