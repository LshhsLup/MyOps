#pragma once

#include <algorithm>
#include <array>
#include <cstdio>
#include <type_traits>
#include "common.h"
#include "function_traits.h"

// inspired by pytorch:
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/MemoryAccess.cuh

namespace myops {
// we want to doing something like this:
//    using args_t = typename traits::ArgsTuple;
//    args_t args;
//    #pragma unroll
//    for (int i = 0; i < traits::arity; i++) {
//      std::get<i>(args) = ....
//    }
template <template <int> typename func, int end, int current = 0>
struct constexpr_unroll {
  template <typename... Args>
  static inline MYOPS_HOST_DEVICE void with_args(Args &&...args) {
    func<current>::apply(std::forward<Args>(args)...);
    constexpr_unroll<func, end, current + 1>::with_args(args...);
  }
};

template <template <int> typename func, int end>
struct constexpr_unroll<func, end, end> {
  template <typename... Args>
  static inline MYOPS_HOST_DEVICE void with_args(Args &&...args) {}
};

// aligned vector generates vectorized load/store instructions on CUDA
template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) aligned_vector {
  T val[vec_size];
};

template <typename T>
inline MYOPS_HOST_DEVICE int max_vectorize_num(const int8_t *ptr) {
  uint64_t address = reinterpret_cast<uint64_t>(ptr);
  constexpr int vec2_alignment = std::alignment_of_v<aligned_vector<T, 2>>;
  constexpr int vec4_alignment = std::alignment_of_v<aligned_vector<T, 4>>;
  constexpr int vec8_alignment = std::alignment_of_v<aligned_vector<T, 8>>;
  if (address % vec8_alignment == 0) {
    return 8;
  } else if (address % vec4_alignment == 0) {
    return 4;
  } else if (address % vec2_alignment == 0) {
    return 2;
  }
  return 1;
}

template <typename T>
inline MYOPS_HOST_DEVICE int max_vectorize_num(char *ptr) {
  return max_vectorize_num<T>(reinterpret_cast<const int8_t *>(ptr));
}

template <int i>
struct max_vectorize_num_helper {
  template <typename array_t, typename traits>
  static MYOPS_HOST_DEVICE void apply(int &result, array_t ptrs, traits) {
    using arg_t = typename traits::template arg<i>::type;
    result = std::min(result, max_vectorize_num<arg_t>(ptrs[i + 1]));
  }
};

template <typename func_t, typename array_t>
inline int max_vectorize_num(array_t ptrs) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  constexpr int input_args_num = traits::input_args_num;
  int result = max_vectorize_num<return_t>(ptrs[0]);
  constexpr_unroll<max_vectorize_num_helper, input_args_num>::with_args(result, ptrs, traits{});
  return result;
}

template <typename T>
struct LoadImpl {
  MYOPS_HOST_DEVICE static T apply(const void *src) {
    if constexpr (std::is_same_v<T, bool>) {
      return static_cast<bool>(*reinterpret_cast<const signed char *>(src));
    } else {
      return *reinterpret_cast<const T *>(src);
    }
  }
};

template <typename T>
MYOPS_HOST_DEVICE constexpr T load(const void *src) {
  return LoadImpl<T>::apply(src);
}

template <typename T>
MYOPS_HOST_DEVICE constexpr T load(const T *src) {
  return LoadImpl<T>::apply(src);
}

struct Loader {
  template <typename T>
  __device__ T load(char *base_ptr, uint32_t offset) {
    return myops::load<T>(reinterpret_cast<const T *>(base_ptr) + offset);
  }
};

struct Storer {
  template <typename T>
  __device__ void store(T value, char *base_ptr, uint32_t offset) {
    *(reinterpret_cast<T *>(base_ptr) + offset) = value;
  }
};

template <int arg_index>
struct unroll_load_helper {
  template <typename args_t, typename policy_t, typename offset_t, typename loader_t>
  static __device__ void apply(policy_t &self,
                               args_t *args,
                               offset_t offset,
                               loader_t loader,
                               int j) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    std::get<arg_index>(args[j]) =
        loader.template load<arg_t>(self.data[1 + arg_index], offset[arg_index]);
  }
};

template <int threads_per_block,
          typename data_t,
          int elements_per_thread,
          typename load_t,
          typename store_t>
struct unroll {
  data_t data;
  int remaining;
  load_t loader;
  store_t storer;
  static constexpr int thread_work_size = elements_per_thread;
  static constexpr int block_work_size = threads_per_block * thread_work_size;

  __device__ unroll(data_t data, int remaining, load_t loader, store_t storer)
      : data(data), remaining(remaining), loader(loader), storer(storer) {}

  __device__ inline bool check_inbounds(int thread_work_element) {
    return (static_cast<int>(threadIdx.x + thread_work_element * threads_per_block) < remaining);
  }

  template <typename args_t>
  __device__ inline void load(args_t *args, int idx) {
    constexpr int input_args_num = std::tuple_size_v<args_t>;
    int thread_idx = threadIdx.x;
#pragma unroll
    for (int i = 0; i < elements_per_thread; ++i) {
      if (thread_idx < remaining) {
        int linear_idx = idx * block_work_size + thread_idx;
        using offset_type = std::array<uint32_t, std::max<int>(input_args_num, 1)>;
        offset_type offset;
#pragma unroll
        for (int arg = 0; arg < input_args_num; ++arg) {
          offset[arg] = linear_idx;
        }
        constexpr_unroll<unroll_load_helper, input_args_num>::with_args(*this, args, offset, loader,
                                                                        i);
        thread_idx += threads_per_block;
      }
    }
  }

  template <typename T>
  __device__ inline void store(T *from, int idx) {
    int thread_idx = threadIdx.x;
#pragma unroll
    for (int i = 0; i < elements_per_thread; ++i) {
      if (thread_idx < remaining) {
        int linear_idx = idx * block_work_size + thread_idx;
        uint32_t offset = linear_idx;
        storer.store(from[i], data[0], offset);
        thread_idx += threads_per_block;
      }
    }
  }
};

template <int vec_size, typename T>
__device__ aligned_vector<T, vec_size> load_vector(const T *base_ptr, uint32_t offset) {
  using vec_t = aligned_vector<T, vec_size>;
  auto *from = reinterpret_cast<const vec_t *>(base_ptr);
  return from[offset];
}

template <int arg_index>
struct vectorized_load_helper {
  template <typename args_t, typename policy_t>
  static __device__ void apply(policy_t &self, args_t *args, int idx, int block_work_size) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    auto ptr = reinterpret_cast<arg_t *>(self.data[1 + arg_index]) + idx * block_work_size;
    auto arg_accessor = [&args] __device__(int thread_unroll_idx) -> arg_t & {
      return std::get<arg_index>(args[thread_unroll_idx]);
    };
    self.load_single_arg(arg_accessor, ptr);
  }
};

template <int vec_size, typename data_t, int elements_per_thread>
struct vectorized {
  data_t data;
  static_assert(elements_per_thread % vec_size == 0,
                "The workload per thread must be a multiple of vec_size");
  static constexpr int thread_work_size = elements_per_thread;
  static constexpr int loop_size = elements_per_thread / vec_size;

  __device__ vectorized(data_t data) : data(data) {}

  __device__ inline bool check_inbounds(int thread_work_elem) { return true; }

  // accessor_t is callable/lambda
  template <typename accessor_t, typename scalar_t>
  __device__ inline void load_single_arg(accessor_t to, scalar_t *from) {
    int thread_idx = threadIdx.x;
#pragma unroll
    for (int i = 0; i < loop_size; ++i) {
      int index = thread_idx + i * threads_per_block();
      auto v = load_vector<vec_size>(from, index);
#pragma unroll
      for (int j = 0; j < vec_size; ++j) {
        to(vec_size * i + j) = v.val[j];
      }
    }
  }

  template <typename args_t>
  __device__ inline void load(args_t *args, int idx) {
    constexpr int input_args_num = std::tuple_size_v<args_t>;
    constexpr_unroll<vectorized_load_helper, input_args_num>::with_args(
        *this, args, idx, elements_per_thread * threads_per_block());
  }

  template <typename scalar_t>
  __device__ inline void store(scalar_t *from, int idx) {
    using vec_t = aligned_vector<scalar_t, vec_size>;
    scalar_t *to =
        reinterpret_cast<scalar_t *>(data[0]) + elements_per_thread * threads_per_block() * idx;
    vec_t *to_ = reinterpret_cast<vec_t *>(to);
    int thread_idx = threadIdx.x;
#pragma unroll
    for (int i = 0; i < loop_size; ++i) {
      int index = thread_idx + i * threads_per_block();
      vec_t v;
      for (int j = 0; j < vec_size; ++j) {
        v.val[j] = from[vec_size * i + j];
      }
      to_[index] = v;
    }
  }
};

}  // namespace myops
