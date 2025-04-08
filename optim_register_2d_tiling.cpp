#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <tuple>
#include <hip/hip_fp16.h>

#define HIP_CHECK(status) \
    if (status != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    }

template<typename T>
__host__ void verifyResult(T *h_a, T *h_b, T *h_c, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      T sum = 0;
      for (int k = 0; k < K; k++) {
        sum += h_a[i * K + k] * h_b[k * N + j];
      }
      assert(fabs(h_c[i * N + j] - sum) < 1e-3);
    }
  }
  printf("Correct!\n");
}

template<typename T, size_t BM, size_t BN, size_t BK, size_t TM, size_t TN>
__global__ void gemm_kernel(T* __restrict__ A, T* __restrict__ B, T* __restrict__ C, size_t M, size_t N, size_t K) {
  constexpr size_t numThreads = (BM * BN) / (TM * TN);
  static_assert(numThreads == 256, "Expected fixed thread count for performance tuning");

  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  __shared__ T As[BM * BK];
  __shared__ T Bs[BK * BN];

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  const uint strideA = numThreads / BK;

  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  const uint strideB = numThreads / BN;

  T threadResults[TM * TN] = {0};
  T regM[TM] = {0};
  T regN[TN] = {0};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      uint idx = (innerRowA + loadOffset) * K + innerColA;
      if ((innerRowA + loadOffset) < BM && innerColA < BK) {
        As[(innerRowA + loadOffset) * BK + innerColA] = __ldg(&A[idx]);
      }
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      uint idx = (innerRowB + loadOffset) * N + innerColB;
      if ((innerRowB + loadOffset) < BK && innerColB < BN) {
        Bs[(innerRowB + loadOffset) * BN + innerColB] = __ldg(&B[idx]);
      }
    }
    __syncthreads();

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
    A += BK;
    B += BK * N;
  }

  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = threadResults[resIdxM * TN + resIdxN];
    }
  }
}

template<typename T>
__host__ void copyFromHostToDevice(T* h_a, T* h_b, T* d_a, T* d_b, size_t M, size_t N , size_t K) {
  size_t a_bytes = sizeof(T) * M * K;
  size_t b_bytes = sizeof(T) * K * N;
  HIP_CHECK(hipMemcpyAsync(d_a, h_a, a_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpyAsync(d_b, h_b, b_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipDeviceSynchronize());
}

template<typename T, const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__host__ void executeKernel(T* d_a, T* d_b, T* d_c, size_t M, size_t N, size_t K) {
  dim3 block((BM * BN) / (TM * TN), 1, 1);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
  hipLaunchKernelGGL((gemm_kernel<T, BM, BN, BK, TM, TN>), grid, block, 0, 0, d_a, d_b, d_c, M, N, K);
  HIP_CHECK(hipDeviceSynchronize());
}

template<typename T>
__host__ void copyFromDeviceToHost(T* d_c, T* h_c, size_t M, size_t N) {
  size_t c_bytes = sizeof(T) * M * N;
  HIP_CHECK(hipMemcpyAsync(h_c, d_c, c_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
}

template<typename T>
__host__ void deallocateMemory(T* d_a, T* d_b, T* d_c) {
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));
}

__host__ void cleanUpDevice() {
  HIP_CHECK(hipDeviceReset());
}

__host__ std::tuple<int, int, int> parseCmdLineArgs(int argc, char *argv[]) {
  int M = 1024, N = 1024, K = 1024;
  for (int i = 1; i < argc; i++) {
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
  HIP_CHECK(hipMalloc((void**)&d_a, M * K * sizeof(float)));
  HIP_CHECK(hipMalloc((void**)&d_b, K * N * sizeof(float)));
  HIP_CHECK(hipMalloc((void**)&d_c, M * N * sizeof(float)));

  copyFromHostToDevice<float>(h_a, h_b, d_a, d_b, M, N, K);
  executeKernel<float, 128, 128, 16, 8, 8>(d_a, d_b, d_c, M, N, K);
  copyFromDeviceToHost<float>(d_c, h_c, M, N);
  verifyResult<float>(h_a, h_b, h_c, M, N, K);

  deallocateMemory<float>(d_a, d_b, d_c);
  cleanUpDevice();

  free(h_a);
  free(h_b);
  free(h_c);
  return 0;
}
