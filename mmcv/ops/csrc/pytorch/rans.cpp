// Copyright(c) OpenMMLab.All rights reserved.
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

#include <utils/rans/rans_inference.hpp>

BufferedRansEncoder buffered_rans_encoder_impl(const Tensor device_signatures) {
  return DISPATCH_DEVICE_IMPL(device_signatures);
}
BufferedRansEncoder buffered_rans_encoder(const Tensor device_signatures) {
  return buffered_rans_encoder_impl(device_signatures);
}

RansEncoder rans_encoder_impl(const Tensor device_signatures) {
  return DISPATCH_DEVICE_IMPL(device_signatures);
}
RansEncoder rans_encoder(const Tensor device_signatures) {
  return rans_encoder_impl(device_signatures);
}

RansDecoder rans_decoder_impl(const Tensor device_signatures) {
  return DISPATCH_DEVICE_IMPL(device_signatures);
}
RansDecoder rans_decoder(const Tensor device_signatures) {
  return rans_decoder_impl(device_signatures);
}
