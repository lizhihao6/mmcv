// Copyright(c) OpenMMLab.All rights reserved.
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void nattenav_impl(const Tensor attn, const Tensor value, Tensor output) {
  DISPATCH_DEVICE_IMPL(nattenav_impl, attn, value, output);
}

void nattenav(const Tensor attn, const Tensor value, Tensor output) {
  nattenav_impl(attn, value, output);
}
