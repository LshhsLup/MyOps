#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// macros defined in pytorch:
// https://github.com/pytorch/pytorch/blob/main/torch/headeronly/macros/Macros.h
#if defined(__CUDACC__)
  #define MYOPS_HOST_DEVICE __host__ __device__
  #define MYOPS_HOST __host__
  #define MYOPS_DEVICE __device__
  // constants from
  // (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications)
  // The maximum number of threads per multiprocessor is 1024 for Turing
  // architecture (7.5), 1536 for Geforce Ampere (8.6)/Jetson Orin (8.7), and
  // 2048 for all other architectures. You'll get warnings if you exceed these
  // constants. Hence, the following macros adjust the input values from the user
  // to resolve potential warnings.
  #if __CUDA_ARCH__ == 750
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1024;
  #elif __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 890 || \
      __CUDA_ARCH__ == 1200
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1536;
  #else
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 2048;
  #endif
// CUDA_MAX_THREADS_PER_BLOCK is same for all architectures currently
constexpr uint32_t CUDA_MAX_THREADS_PER_BLOCK = 1024;
// CUDA_THREADS_PER_BLOCK_FALLBACK is the "canonical fallback" choice of block
// size. 256 is a good number for this fallback and should give good occupancy
// and versatility across all architectures.
constexpr uint32_t CUDA_THREADS_PER_BLOCK_FALLBACK = 256;
  // NOTE: if you are thinking of constexpr-ify the inputs to launch bounds, it
  //       turns out that although __launch_bounds__ can take constexpr, it
  //       can't take a constexpr that has anything to do with templates.
  //       MYOPS_MAX_THREADS_PER_BLOCK
  //       and MYOPS_MIN_BLOCKS_PER_SM are kept as macros.
  // Suppose you were planning to write __launch_bounds__(a, b), based on your
  // performance tuning on a modern GPU. Instead, you should write
  // __launch_bounds__(MYOPS_MAX_THREADS_PER_BLOCK(a), MYOPS_MIN_BLOCKS_PER_SM(a, b)),
  // which will also properly respect limits on old architectures.
  #define MYOPS_MAX_THREADS_PER_BLOCK(val) \
    (((val) <= CUDA_MAX_THREADS_PER_BLOCK) ? (val) : CUDA_THREADS_PER_BLOCK_FALLBACK)
  #define MYOPS_MIN_BLOCKS_PER_SM(threads_per_block, blocks_per_sm)      \
    ((((threads_per_block) * (blocks_per_sm) <= CUDA_MAX_THREADS_PER_SM) \
          ? (blocks_per_sm)                                              \
          : ((CUDA_MAX_THREADS_PER_SM + (threads_per_block) - 1) / (threads_per_block))))
  // MYOPS_LAUNCH_BOUNDS is analogous to __launch_bounds__
  #define MYOPS_LAUNCH_BOUNDS_0 \
    __launch_bounds__(256, 4)  // default launch bounds that should give good occupancy and
                               // versatility across all architectures.
  #define MYOPS_LAUNCH_BOUNDS_1(max_threads_per_block) \
    __launch_bounds__((MYOPS_MAX_THREADS_PER_BLOCK((max_threads_per_block))))
  #define MYOPS_LAUNCH_BOUNDS_2(max_threads_per_block, min_blocks_per_sm)     \
    __launch_bounds__((MYOPS_MAX_THREADS_PER_BLOCK((max_threads_per_block))), \
                      (MYOPS_MIN_BLOCKS_PER_SM((max_threads_per_block), (min_blocks_per_sm))))
#else
  #define MYOPS_HOST_DEVICE
  #define MYOPS_HOST
  #define MYOPS_DEVICE
#endif
#define WARP_SIZE 32
#define MYOPS_CUDA_KERNEL_LAUNCH_CHECK()                                                  \
  do {                                                                                    \
    cudaError_t err = cudaGetLastError();                                                 \
    if (err != cudaSuccess) {                                                             \
      fprintf(stderr, "CUDA kernel launch error: %s at %s:%d\n", cudaGetErrorString(err), \
              __FILE__, __LINE__);                                                        \
      fflush(stderr);                                                                     \
      abort();                                                                            \
    }                                                                                     \
  } while (0)

#define MYOPS_CHECK_FAILED(msg)                                              \
  do {                                                                       \
    fprintf(stderr, "CHECK FAILED: %s at %s:%d\n", msg, __FILE__, __LINE__); \
    fflush(stderr);                                                          \
    abort();                                                                 \
  } while (0)

namespace myops {

constexpr uint32_t threads_per_block() {
  return WARP_SIZE * 4;
}

enum MyOpsDtype { MYOPS_DTYPE_FLOAT, MYOPS_DTYPE_HALF, MYOPS_DTYPE_BFLOAT16 };

// Type conversion helpers
template <typename T>
struct FloatConverter;

template <>
struct FloatConverter<float> {
  static __device__ float to_float(float x) { return x; }
  static __device__ float from_float(float x) { return x; }
};

template <>
struct FloatConverter<__half> {
  static __device__ float to_float(__half x) { return __half2float(x); }
  static __device__ __half from_float(float x) { return __float2half(x); }
};

template <>
struct FloatConverter<__nv_bfloat16> {
  static __device__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
  static __device__ __nv_bfloat16 from_float(float x) { return __float2bfloat16(x); }
};

}  // namespace myops
