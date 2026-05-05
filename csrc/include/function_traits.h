#pragma once

namespace myops {

// Inspired by pytorch:
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/detail/FunctionTraits.h

// a general function template that can get info of func_t:
// return_type, args_type and the num of args.
// func_t can be:
// 1. function: int(float, double)
// 2. function pointer: int(*)(float, double)
// 3. Lambda: [](float, double) -> int {};
// 4. functor: struct func {int operator()(float, double){}};
// 5. member function: void (class::*)(int) const
template <typename T>
struct function_traits : public function_traits<decltype(&T::operator())> {};

// free function
template <typename ReturnType, typename... Args>
struct function_traits<ReturnType(Args...)> {
  enum { input_args_num = sizeof...(Args) };
  using ArgsTuple = std::tuple<Args...>;
  using result_type = ReturnType;
  using return_type = ReturnType;

  template <size_t i>
  struct arg {
    using type = std::tuple_element_t<i, std::tuple<Args...>>;
  };
};

// Const class member functions
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType (ClassType::*)(Args...) const>
    : public function_traits<ReturnType(Args...)> {};

// Reference types
template <typename T>
struct function_traits<T &> : public function_traits<T> {};
template <typename T>
struct function_traits<T *> : public function_traits<T> {};

}  // namespace myops