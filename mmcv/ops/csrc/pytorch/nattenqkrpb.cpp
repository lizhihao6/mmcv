// Copyright(c) OpenMMLab.All rights reserved.
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void nattenqkrpb_impl(const Tensor query, const Tensor key, const Tensor rpb, Tensor output) {
  DISPATCH_DEVICE_IMPL(nattenav_impl, query, key, rpb, output);
}

void nattenqkrpb(const Tensor query, const Tensor key, const Tensor rpb, Tensor output) {
  nattenav_impl(query, key, rpb, output);
}
