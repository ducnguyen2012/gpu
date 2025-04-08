// File: tiled-main.hip.cpp

#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <tuple>

// Kiểm tra kết quả trên CPU để xác nhận tính đúng đắn
template<typename T>
void verifyResult(T *h_a, T *h_b, T *h_c, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      T sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += h_a[i * K + k] * h_b[k * N + j];
      }
      assert(fabs(h_c[i * N + j] - sum) < 1e-3);
    }
  }
  std::cout << "✅ Correct!\n";
}

// Kernel GEMM sử dụng kỹ thuật chia tile
template<typename T, const size_t bM, const size_t bN, const size_t bK>
__global__ void gemm_kernel(T* d_a, T* d_b, T* d_c, size_t M, size_t N, size_t K) {
  __shared__ T As[bM * bK];
  __shared__ T Bs[bK * bN];

  size_t blockRow = blockIdx.y;
  size_t blockCol = blockIdx.x;

  size_t threadRow = threadIdx.x / bN;
  size_t threadCol = threadIdx.x % bN;

  T tmp = 0;

  for (int bk = 0; bk < K; bk += bK) {
    size_t globalARow = blockRow * bM + threadRow;
    size_t globalACol = bk + threadCol;

    size_t globalBRow = bk + threadRow;
    size_t globalBCol = blockCol * bN + threadCol;

    if (globalARow < M && globalACol < K)
      As[threadRow * bK + threadCol] = d_a[globalARow * K + globalACol];
    else
      As[threadRow * bK + threadCol] = 0;

    if (globalBRow < K && globalBCol < N)
      Bs[threadRow * bN + threadCol] = d_b[globalBRow * N + globalBCol];
    else
      Bs[threadRow * bN + threadCol] = 0;

    __syncthreads();

    for (size_t i = 0; i < bK; ++i) {
      tmp += As[threadRow * bK + i] * Bs[i * bN + threadCol];
    }

    __syncthreads();
  }

  size_t globalCRow = blockRow * bM + threadRow;
  size_t globalCCol = blockCol * bN + threadCol;

  if (globalCRow < M && globalCCol < N)
    d_c[globalCRow * N + globalCCol] = tmp;
}

// Copy dữ liệu từ host sang device
template<typename T>
void copyFromHostToDevice(T* h_a, T* h_b, T* d_a, T* d_b, size_t M, size_t N, size_t K) {
  hipMemcpy(d_a, h_a, sizeof(T) * M * K, hipMemcpyHostToDevice);
  hipMemcpy(d_b, h_b, sizeof(T) * K * N, hipMemcpyHostToDevice);
}

// Copy kết quả từ device sang host
template<typename T>
void copyFromDeviceToHost(T* d_c, T* h_c, size_t M, size_t N) {
  hipMemcpy(h_c, d_c, sizeof(T) * M * N, hipMemcpyDeviceToHost);
}

// Gọi kernel thực thi
template<typename T, const size_t bM, const size_t bN, const size_t bK>
void executeKernel(T* d_a, T* d_b, T* d_c, size_t M, size_t N, size_t K) {
  dim3 block(bM * bN);
  dim3 grid((N + bN - 1) / bN, (M + bM - 1) / bM);
  hipLaunchKernelGGL(gemm_kernel<T, bM, bN, bK>, grid, block, 0, 0, d_a, d_b, d_c, M, N, K);
  hipDeviceSynchronize();
}

template<typename T>
void deallocateMemory(T* d_a, T* d_b, T* d_c) {
  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_c);
}

std::tuple<int, int, int> parseCmdLineArgs(int argc, char *argv[]) {
  int M = 1024, N = 1024, K = 1024;
  for (int i = 1; i < argc; i += 2) {
    std::string opt = argv[i];
    std::string val = argv[i + 1];
    if (opt == "-m") M = std::stoi(val);
    else if (opt == "-n") N = std::stoi(val);
    else if (opt == "-k") K = std::stoi(val);
  }
  return {M, N, K};
}

int main(int argc, char *argv[]) {
  auto [M, N, K] = parseCmdLineArgs(argc, argv);

  float *h_a = (float*)malloc(sizeof(float) * M * K);
  float *h_b = (float*)malloc(sizeof(float) * K * N);
  float *h_c = (float*)malloc(sizeof(float) * M * N);

  for (size_t i = 0; i < M * K; i++) h_a[i] = rand() % 10;
  for (size_t i = 0; i < K * N; i++) h_b[i] = rand() % 10;

  float *d_a, *d_b, *d_c;
  hipMalloc(&d_a, sizeof(float) * M * K);
  hipMalloc(&d_b, sizeof(float) * K * N);
  hipMalloc(&d_c, sizeof(float) * M * N);

  copyFromHostToDevice(h_a, h_b, d_a, d_b, M, N, K);
  executeKernel<float, 32, 32, 32>(d_a, d_b, d_c, M, N, K);
  copyFromDeviceToHost(d_c, h_c, M, N);

  verifyResult(h_a, h_b, h_c, M, N, K);

  deallocateMemory(d_a, d_b, d_c);
  free(h_a);
  free(h_b);
  free(h_c);

  hipDeviceReset();
  return 0;
}
