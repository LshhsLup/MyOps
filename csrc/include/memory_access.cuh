#pragma once

#include "common.h"

// we want to doing something like this:
//    using args_t = typename traits::ArgsTuple;
//    args_t args;
//    #pragma unroll
//    for (int i = 0; i < traits::arity; i++) {
//      std::get<i>(args) = ....
//    }
template <template<int> typename func, int end, int current = 0>
struct constexpr_unroll {
  template <typename... Args>
  static inline MYOPS_HOST_DEVICE void with_args(Args&& ...args) {
    func<current>::apply(std::forward<Args>(args)...);
    constexpr_unroll<func, end, current+1>::with_args(args...);
  }
};

template<template<int> typename func, int end>
struct constexpr_unroll<func, end, end> {
  template<typename... Args>
  static inline MYOPS_HOST_DEVICE void with_args(Args&& ...args) {}
};

// aligned vector generates vectoized load/store instructions on CUDA
template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) aligned_vector {
  T val[vec_size];
};

template <typename T>
inline MYOPS_HOST_DEVICE int max_vectorzed_num (const int8_t *ptr) {
    
}
