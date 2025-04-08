#include <hip/hip_runtime.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <tuple>
#include <string>

#define CHECK(cmd) \
  { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
      fprintf(stderr, "HIP error %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(e)); \
      exit(EXIT_FAILURE); \
    } \
  }

template<typename T>
__host__ void verifyResult(T *h_a, T *h_b, T *h_c, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      T sum = 0;
      for (int k = 0; k < K; k++) {
        sum += h_a[i * K + k] * h_b[k * N + j];
      }
      assert(h_c[i * N + j] == sum);
    }
  }
  printf("Correct!\n");
}

template<typename T, size_t BM, size_t BN, size_t BK, size_t TM>
__global__ void gemm_kernel(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, size_t M, size_t N, size_t K) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  __shared__ T As[BM * BK];
  __shared__ T Bs[BK * BN];

  const uint innerColA = threadIdx.x % BK;
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN;
  const uint innerRowB = threadIdx.x / BN;

  T threadResults[TM] = {0};

  const T* aTilePtr = A + cRow * BM * K;
  const T* bTilePtr = B + cCol * BN;
  T* cTilePtr = C + cRow * BM * N + cCol * BN;

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    As[innerRowA * BK + innerColA] = __ldg(&aTilePtr[innerRowA * K + innerColA]);
    Bs[innerRowB * BN + innerColB] = __ldg(&bTilePtr[innerRowB * N + innerColB]);
    __syncthreads();

    #pragma unroll
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      T tmpB = Bs[dotIdx * BN + threadCol];
      #pragma unroll
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
    aTilePtr += BK;
    bTilePtr += BK * N;
  }

  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    cTilePtr[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
  }
}

template<typename T>
__host__ void copyFromHostToDeviceAsync(T* h_a, T* h_b, T* d_a, T* d_b, size_t M, size_t N, size_t K, hipStream_t stream) {
  size_t a_bytes = sizeof(T) * M * K;
  size_t b_bytes = sizeof(T) * K * N;

  CHECK(hipMemcpyAsync(d_a, h_a, a_bytes, hipMemcpyHostToDevice, stream));
  CHECK(hipMemcpyAsync(d_b, h_b, b_bytes, hipMemcpyHostToDevice, stream));
}

template<typename T, const uint BM, const uint BN, const uint BK, const uint TM>
__host__ void executeKernel(T* d_a, T* d_b, T* d_c, size_t M, size_t N, size_t K) {
  dim3 block((BM * BN) / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  hipLaunchKernelGGL((gemm_kernel<T, BM, BN, BK, TM>), grid, block, 0, 0, d_a, d_b, d_c, M, N, K);
  CHECK(hipDeviceSynchronize());
}

template<typename T>
__host__ void copyFromDeviceToHostAsync(T* d_c, T* h_c, size_t M, size_t N, hipStream_t stream) {
  size_t c_bytes = sizeof(T) * M * N;
  CHECK(hipMemcpyAsync(h_c, d_c, c_bytes, hipMemcpyDeviceToHost, stream));
}

template<typename T>
__host__ void deallocateMemory(T* d_a, T* d_b, T* d_c) {
  CHECK(hipFree(d_a));
  CHECK(hipFree(d_b));
  CHECK(hipFree(d_c));
}

__host__ void cleanUpDevice() {
  CHECK(hipDeviceReset());
}

__host__ std::tuple<int, int, int> parseCmdLineArgs(int argc, char *argv[]) {
  int M = 1024, N = 1024, K = 1024;
  for (int i = 1; i < argc; i++){
    std::string option(argv[i]);
    std::string value(argv[i+1]);
    i++;
    if (option == "-m") M = std::stoi(value);
    else if (option == "-n") N = std::stoi(value);
    else if (option == "-k") K = std::stoi(value);
  }
  return {M, N, K};
}

int main(int argc, char *argv[]) {
  auto [M, N, K] = parseCmdLineArgs(argc, argv);

  float* h_a = (float*)malloc(M * K * sizeof(float));
  float* h_b = (float*)malloc(K * N * sizeof(float));
  float* h_c = (float*)malloc(M * N * sizeof(float));

  for (size_t i = 0; i < M * K; i++) h_a[i] = rand() % 10;
  for (size_t i = 0; i < K * N; i++) h_b[i] = rand() % 10;

  float *d_a, *d_b, *d_c;
  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));
  CHECK(hipMallocAsync(&d_a, M * K * sizeof(float), stream));
  CHECK(hipMallocAsync(&d_b, K * N * sizeof(float), stream));
  CHECK(hipMallocAsync(&d_c, M * N * sizeof(float), stream));

  copyFromHostToDeviceAsync<float>(h_a, h_b, d_a, d_b, M, N, K, stream);

  executeKernel<float, 64, 64, 8, 8>(d_a, d_b, d_c, M, N, K);

  copyFromDeviceToHostAsync<float>(d_c, h_c, M, N, stream);
  CHECK(hipStreamSynchronize(stream));

  verifyResult<float>(h_a, h_b, h_c, M, N, K);

  deallocateMemory<float>(d_a, d_b, d_c);
  cleanUpDevice();
  hipStreamDestroy(stream);

  free(h_a);
  free(h_b);
  free(h_c);
  return 0;
}
