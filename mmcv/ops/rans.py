# Copyright (c) OpenMMLab. All rights reserved.
from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['BufferedRansEncoder',
                                          'RansEncoder',
                                          'RansDecoder'])

BufferedRansEncoder = ext_module.BufferedRansEncoder
RansEncoder = ext_module.RansEncoder
RansDecoder = ext_module.RansDecoder
