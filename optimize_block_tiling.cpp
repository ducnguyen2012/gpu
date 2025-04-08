#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <iostream>

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

template<typename T, const size_t bM, const size_t bN, const size_t bK>
__global__ void gemm_kernel(T* d_a, T* d_b, T* d_c, size_t M, size_t N, size_t K) {
  const size_t cRow = blockIdx.y * bM + threadIdx.y;
  const size_t cCol = blockIdx.x * bN + threadIdx.x;

  __shared__ T As[bM][bK];
  __shared__ T Bs[bK][bN];

  T sum = 0;

  for (int bkIdx = 0; bkIdx < K; bkIdx += bK) {
    if (cRow < M && (bkIdx + threadIdx.x) < K)
      As[threadIdx.y][threadIdx.x] = d_a[cRow * K + bkIdx + threadIdx.x];
    else
      As[threadIdx.y][threadIdx.x] = 0;

    if ((bkIdx + threadIdx.y) < K && cCol < N)
      Bs[threadIdx.y][threadIdx.x] = d_b[(bkIdx + threadIdx.y) * N + cCol];
    else
      Bs[threadIdx.y][threadIdx.x] = 0;

    __syncthreads();

    for (size_t dotIdx = 0; dotIdx < bK; ++dotIdx) {
      sum += As[threadIdx.y][dotIdx] * Bs[dotIdx][threadIdx.x];
    }

    __syncthreads();
  }

  if (cRow < M && cCol < N)
    d_c[cRow * N + cCol] = sum;
}


template<typename T>
__host__ void copyFromHostToDevice(T* h_a, T* h_b, T* d_a, T* d_b, size_t M, size_t N, size_t K) {
  size_t a_bytes = sizeof(T) * M * K;
  size_t b_bytes = sizeof(T) * K * N;
  hipError_t err = hipMemcpy(d_a, h_a, a_bytes, hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to copy h_a to d_a (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = hipMemcpy(d_b, h_b, b_bytes, hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to copy h_b to d_b (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
template<typename T, const size_t bM, const size_t bN, const size_t bK>
__host__ void executeKernel(T* d_a, T* d_b, T* d_c, size_t M, size_t N, size_t K) {
  dim3 block(bN, bM); // (x, y)
  dim3 grid((N + bN - 1) / bN, (M + bM - 1) / bM);
  hipLaunchKernelGGL((gemm_kernel<T, bM, bN, bK>), grid, block, 0, 0, d_a, d_b, d_c, M, N, K);
  hipDeviceSynchronize();
}


template<typename T>
__host__ void copyFromDeviceToHost(T* d_c, T* h_c, size_t M, size_t N) {
  size_t c_bytes = sizeof(T) * M * N;
  hipError_t err = hipMemcpy(h_c, d_c, c_bytes, hipMemcpyDeviceToHost);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to copy from d_c to h_c (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

template<typename T>
__host__ void deallocateMemory(T* d_a, T* d_b, T* d_c) {
  hipError_t err = hipFree(d_a);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to deallocate d_a (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = hipFree(d_b);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to deallocate d_b (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = hipFree(d_c);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to deallocate d_c (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__host__ void cleanUpDevice() {
  hipError_t err = hipDeviceReset();
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to clean up device (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

template<typename T>
__host__ std::tuple<int, int, int> parseCmdLineArgs(int argc, char *argv[]) {
  int M = 1024, N = 1024, K = 1024;
  for (int i = 1; i < argc; i++){
    std::string option(argv[i]);
    std::string value(argv[i+1]);
    i++;
    if (option.compare("-m") == 0)
      M = std::stoi(value);
    else if (option.compare("-n") == 0)
      N = std::stoi(value);
    else if (option.compare("-k") == 0)
      K = std::stoi(value);
  }
  return {M, N, K};
}

int main(int argc, char *argv[]) {
  auto parsedCmdLineArgsTuple = parseCmdLineArgs<float>(argc, argv);
  int M = std::get<0>(parsedCmdLineArgsTuple);
  int N = std::get<1>(parsedCmdLineArgsTuple);
  int K = std::get<2>(parsedCmdLineArgsTuple);

  float* h_a = (float*)malloc(M * K * sizeof(float));
  float* h_b = (float*)malloc(K * N * sizeof(float));
  float* h_c = (float*)malloc(M * N * sizeof(float));

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < K; j++) {
      h_a[i * K + j] = rand() % 10;
    }
  }
  for (size_t i = 0; i < K; i++) {
    for (size_t j = 0; j < N; j++) {
      h_b[i * N + j] = rand() % 10;
    }
  }

  float *d_a, *d_b, *d_c;
  hipError_t err = hipMalloc((void **)&d_a, M * K * sizeof(float));
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to allocate d_a (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = hipMalloc((void **)&d_b, K * N * sizeof(float));
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to allocate d_b (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = hipMalloc((void **)&d_c, M * N * sizeof(float));
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to allocate d_c (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  copyFromHostToDevice<float>(h_a, h_b, d_a, d_b, M, N, K);
  executeKernel<float, 32, 32, 32>(d_a, d_b, d_c, M, N, K);
  copyFromDeviceToHost<float>(d_c, h_c, M, N);
  verifyResult<float>(h_a, h_b, h_c, M, N, K);
  deallocateMemory<float>(d_a, d_b, d_c);
  cleanUpDevice();

  free(h_a);
  free(h_b);
  free(h_c);
  return 0;
}
