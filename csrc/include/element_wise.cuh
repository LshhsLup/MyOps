#pragma once

#include <cstdint>
#include <functional>
#include <tuple>
#include <utility>
#include "common.h"
#include "function_traits.h"
#include "memory_access.cuh"

namespace myops {
// for vectorized
template <int io_size>
constexpr uint32_t elements_per_thread() {
  if constexpr (io_size == 1) {
    return 16;
  } else {
    return 8;
  }
}

template <typename args_t, size_t... Is>
constexpr auto sum_of_sizes(args_t args, std::index_sequence<Is...>) {
  if constexpr (sizeof...(Is) == 0) {
    return 0;
  } else {
    return (sizeof(std::tuple_element_t<Is, args_t>) + ...);
  }
}

template <typename func_t>
constexpr auto calc_io_size() {
  using traits = function_traits<func_t>;
  using args_t = typename traits::ArgsTuple;
  constexpr auto input_sizes =
      sum_of_sizes(args_t{}, std::make_index_sequence<std::tuple_size_v<args_t>>());
  constexpr auto output_sizes = sizeof(typename traits::return_type);
  return input_sizes + output_sizes;
}

template <int io_size>
constexpr auto io_block_work_size() {
  return threads_per_block() * elements_per_thread<io_size>();
}

template <typename funct, typename policy_t>
__device__ inline void elementwise_kernel_helper(funct func, policy_t policy) {
  using traits = function_traits<funct>;
  using return_t = typename traits::result_type;
  using args_t = typename traits::ArgsTuple;
  constexpr int elements_per_thread = policy_t::thread_work_size;
  int idx = blockIdx.x;
  return_t results[elements_per_thread];
  args_t args[elements_per_thread];

  // load
  policy.load(args, idx);

// compute
#pragma unroll
  for (int i = 0; i < elements_per_thread; ++i) {
    if (policy.check_inbounds(i)) {
      results[i] = std::apply(func, args[i]);
    }
  }

  // store
  policy.store(results, idx);
}

template <int vec_size, typename func_t, typename array_t>
MYOPS_LAUNCH_BOUNDS_1(threads_per_block())
__global__ void vectorized_elementwise_kernel(int N, func_t func, array_t data) {
  using traits = function_traits<func_t>;
  constexpr auto io_size = calc_io_size<func_t>();
  int remaining = N - io_block_work_size<io_size>() * blockIdx.x;
  if (remaining >= io_block_work_size<io_size>()) {
    auto policy = vectorized<vec_size, array_t, elements_per_thread<io_size>()>(data);
    elementwise_kernel_helper(func, policy);
  } else {
    auto loader = Loader();
    auto storer = Storer();
    auto policy = unroll<threads_per_block(), array_t, elements_per_thread<io_size>(),
                         decltype(loader), decltype(storer)>(data, remaining, loader, storer);
    elementwise_kernel_helper(func, policy);
  }
}

template <typename func_t,
          typename array_t,
          typename load_t,
          typename store_t,
          int elements_per_thread = 4>
MYOPS_LAUNCH_BOUNDS_1(threads_per_block())
__global__ void unrolled_elementwise_kernel(int N,
                                            func_t func,
                                            array_t data,
                                            load_t loader,
                                            store_t storer) {
  int remaining = N - elements_per_thread * threads_per_block() * blockIdx.x;
  auto policy = unroll<threads_per_block(), array_t, elements_per_thread, load_t, store_t>(
      data, remaining, loader, storer);
  elementwise_kernel_helper(func, policy);
}

template <typename func_t, typename array_t>
static inline void launch_vectorized_kernel(cudaStream_t stream,
                                            int64_t N,
                                            const func_t &f,
                                            array_t data) {
  using traits = function_traits<func_t>;
  constexpr auto io_size = calc_io_size<func_t>();
  using cpp_type = typename function_traits<func_t>::result_type;
  const uint16_t max_vec_size = max_vectorize_num<func_t>(data);
  uint16_t vec_size = 16 / static_cast<uint16_t>(sizeof(cpp_type));
  vec_size = std::min<uint16_t>(vec_size, max_vec_size);
  constexpr int thread_work_size = elements_per_thread<io_size>();
  constexpr int block_work_size = thread_work_size * threads_per_block();
  int64_t grid = (N + io_block_work_size<io_size>() - 1) / io_block_work_size<io_size>();
  switch (vec_size) {
    case 8:
      vectorized_elementwise_kernel<8, func_t, array_t>
          <<<dim3(grid), threads_per_block(), 0, stream>>>(N, f, data);
      MYOPS_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 4:
      vectorized_elementwise_kernel<4, func_t, array_t>
          <<<dim3(grid), threads_per_block(), 0, stream>>>(N, f, data);
      MYOPS_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 2:
      vectorized_elementwise_kernel<2, func_t, array_t>
          <<<dim3(grid), threads_per_block(), 0, stream>>>(N, f, data);
      MYOPS_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 1: {
      constexpr int elements_per_thread_unrolled = 4;
      constexpr int block_work_size_unrolled = elements_per_thread_unrolled * threads_per_block();
      int64_t grid_unrolled = (N + block_work_size_unrolled - 1) / block_work_size_unrolled;
      auto loader = Loader();
      auto storer = Storer();
      unrolled_elementwise_kernel<func_t, array_t, Loader, Storer, 4>
          <<<dim3(grid_unrolled), threads_per_block(), 0, stream>>>(N, f, data, loader, storer);
      MYOPS_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    }
    default:
      MYOPS_CHECK_FAILED("Unexpected vectorization size");
  }
}

template <typename scalar_t, int ArgsNum>
struct TensorArray {
  static_assert(ArgsNum >= 0, "ArgsNum must be non-negative");

  // ptrs[0] = output
  // ptrs[1..ArgsNum] = inputs
  std::array<char *, ArgsNum + 1> ptrs{};

  TensorArray() = default;

  template <typename... InPtrs, typename = std::enable_if_t<sizeof...(InPtrs) == ArgsNum>>
  TensorArray(scalar_t *out, const InPtrs *...ins)
      : ptrs{reinterpret_cast<char *>(out),
             const_cast<char *>(reinterpret_cast<const char *>(ins))...} {}

  // data[0]            -> out
  // data[1 + arg_idx]  -> input[arg_idx]
  MYOPS_HOST_DEVICE char *operator[](int idx) const { return ptrs[idx]; }
};

}  // namespace myops
