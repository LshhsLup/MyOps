#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include "include/common.h"

namespace myops {
namespace torch_api {

class DeviceGuard {
 public:
  explicit DeviceGuard(const at::Tensor &tensor) {
    if (tensor.device().is_cuda()) {
      cudaSetDevice(tensor.device().index());
    }
  }
};

enum class TensorAttr { DEVICE, DTYPE, ALL };
enum class TensorDim { LASTDIM, ALL };

struct attr_t {
  int64_t device_id;
  at::ScalarType dtype;
};

inline void checkDevice(int64_t &device_id, const at::Tensor &tensor) {
  auto tensor_device_id = tensor.get_device();
  if (device_id == -1) {
    device_id = tensor_device_id;
    return;
  }
  TORCH_CHECK(tensor_device_id == device_id,
              "Tensor device id is not same, original device_id: ", device_id,
              "now device_id is: ", tensor_device_id);
}

inline void checkDtype(at::ScalarType &dtype, const at::Tensor &tensor) {
  auto tensor_dtype = tensor.scalar_type();
  if (dtype == at::ScalarType::Undefined) {
    dtype = tensor_dtype;
    return;
  }
  TORCH_CHECK(tensor_dtype == dtype, "Tensor dtype is not same. original dtype: ", dtype,
              ", now dtype is: ", tensor_dtype);
}

template <TensorAttr attr>
inline void checkTensorAttr(attr_t &attr_states, const at::Tensor &tensor) {
  if constexpr (attr == TensorAttr::DEVICE) {
    checkDevice(attr_states.device_id, tensor);
  } else if constexpr (attr == TensorAttr::DTYPE) {
    checkDtype(attr_states.dtype, tensor);
  } else if constexpr (attr == TensorAttr::ALL) {
    checkDevice(attr_states.device_id, tensor);
    checkDtype(attr_states.dtype, tensor);
  }
}

template <TensorAttr attr,
          typename T,
          typename = typename std::enable_if<
              std::is_same<typename std::decay<T>::type, at::Tensor>::value>::type>
void checkTensorSameWithSpecificAttr(attr_t &attr_states, const c10::optional<T> &tensor) {
  if (!tensor.has_value() || !tensor->defined()) return;
  auto temp_tensor = tensor.value();
  checkTensorAttr<attr>(attr_states, temp_tensor);
}

template <TensorAttr attr,
          typename T,
          typename = typename std::enable_if<
              std::is_same<typename std::decay<T>::type, at::Tensor>::value>::type>
void checkTensorSameWithSpecificAttr(attr_t &attr_states, const T &tensor) {
  if (!tensor.defined()) return;
  checkTensorAttr<attr>(attr_states, tensor);
}

template <TensorAttr attr, typename T, typename... Args>
void checkTensorSameWithSpecificAttr(attr_t &attr_states, const T &tensor, Args &&...args) {
  checkTensorSameWithSpecificAttr<attr>(attr_states, tensor);
  checkTensorSameWithSpecificAttr<attr>(attr_states, std::forward<Args>(args)...);
}

template <TensorAttr attr = TensorAttr::ALL, typename... Args>
void checkTensorSameAttr(Args &&...args) {
  attr_t attr_states = {-1, at::ScalarType::Undefined};
  checkTensorSameWithSpecificAttr<attr>(attr_states, std::forward<Args>(args)...);
}

#define CHECK_SHAPE(x, ...)                                   \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
              #x " must have shape (" #__VA_ARGS__ ")")

#define CHECK_TENSOR_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous.")

#define CHECK_OPTIONAL_TENSOR_CONTIGUOUS(x) \
  if (x.has_value()) TORCH_CHECK(x.value().is_contiguous(), #x " must be contiguous.")

template <TensorDim dim,
          typename T,
          typename = typename std::enable_if<
              std::is_same<typename std::decay<T>::type, at::Tensor>::value>::type>
void checkTensorContiguousImpl(const std::string &err_msg, const T &tensor) {
  if (!tensor.defined()) return;
  if (dim == TensorDim::ALL) {
    TORCH_CHECK(tensor.is_contiguous(), err_msg);
  } else if (dim == TensorDim::LASTDIM) {
    TORCH_CHECK(tensor.stride(-1) == 1 || tensor.size(-1) == 1, err_msg);
  }
}

template <TensorDim dim,
          typename T,
          typename = typename std::enable_if<
              std::is_same<typename std::decay<T>::type, at::Tensor>::value>::type>
void checkTensorContiguousImpl(const std::string &err_msg, const c10::optional<T> &tensor) {
  if (!tensor.has_value() || !tensor->defined()) return;
  if (dim == TensorDim::ALL) {
    TORCH_CHECK(!tensor.has_value() || tensor.value().is_contiguous(), err_msg);
  } else if (dim == TensorDim::LASTDIM) {
    TORCH_CHECK(
        !tensor.has_value() || tensor.value().stride(-1) == 1 || tensor.value().size(-1) == 1,
        err_msg);
  }
}

template <TensorDim dim, typename T, typename... Args>
void checkTensorContiguousImpl(const std::string &err_msg, const T &tensor, Args &&...args) {
  checkTensorContiguousImpl<dim>(err_msg, tensor);
  checkTensorContiguousImpl<dim>(err_msg, std::forward<Args>(args)...);
}

template <TensorDim dim = TensorDim::ALL, typename... Args>
void checkTensorContiguous(const std::string &err_msg, Args &&...args) {
  checkTensorContiguousImpl<dim>(err_msg, std::forward<Args>(args)...);
}

template <typename T = void>
inline T *getAtTensorPtr(const c10::optional<at::Tensor> &tensor) {
  return tensor.has_value() ? (T *)tensor.value().data_ptr() : (T *)nullptr;
}

template <typename T = void>
inline T *getAtTensorPtr(const at::Tensor &tensor) {
  return tensor.defined() ? (T *)tensor.data_ptr() : (T *)nullptr;
}

template <typename T, typename SRC>
inline std::vector<T> convertVectorDtype(const SRC &data) {
  std::vector<T> ret(data.begin(), data.end());
  return ret;
}

template <typename scalar_t>
struct PytorchCppTypeToMyType {};

template <>
struct PytorchCppTypeToMyType<float> {
  static constexpr MyOpsDtype value = MyOpsDtype::MYOPS_DTYPE_FLOAT;
};

template <>
struct PytorchCppTypeToMyType<at::Half> {
  static constexpr MyOpsDtype value = MyOpsDtype::MYOPS_DTYPE_HALF;
};

template <>
struct PytorchCppTypeToMyType<at::BFloat16> {
  static constexpr MyOpsDtype value = MyOpsDtype::MYOPS_DTYPE_BFLOAT16;
};

template <MyOpsDtype scalar_t>
struct MyTypeToPytorchCppType {};

template <>
struct MyTypeToPytorchCppType<MyOpsDtype::MYOPS_DTYPE_FLOAT> {
  using type = float;
};

template <>
struct MyTypeToPytorchCppType<MyOpsDtype::MYOPS_DTYPE_HALF> {
  using type = at::Half;
};

template <>
struct MyTypeToPytorchCppType<MyOpsDtype::MYOPS_DTYPE_BFLOAT16> {
  using type = at::BFloat16;
};

// ============================================================================
// Runtime dtype conversion
// ============================================================================

inline MyOpsDtype scalarTypeToMyOpsDtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:
      return MyOpsDtype::MYOPS_DTYPE_FLOAT;
    case at::kHalf:
      return MyOpsDtype::MYOPS_DTYPE_HALF;
    case at::kBFloat16:
      return MyOpsDtype::MYOPS_DTYPE_BFLOAT16;
    default:
      TORCH_CHECK(false, "Unsupported dtype: ", dtype);
  }
}

}  // namespace torch_api
}  // namespace myops