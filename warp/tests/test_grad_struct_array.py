# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest


import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


# Test for issue #1174: Gradients not propagating through array of structs
@wp.struct
class ScalarStruct:
    a: wp.float32


@wp.kernel
def pack_struct_array_kernel(x: wp.array(dtype=wp.float32), y: wp.array(dtype=ScalarStruct)):
    i = wp.tid()
    y[i].a = x[i]


@wp.kernel
def loss_from_struct_array_kernel(y: wp.array(dtype=ScalarStruct), loss: wp.array(dtype=wp.float32)):
    i = wp.tid()
    loss[i] = y[i].a


def test_struct_array_gradient_propagation(_test, device):
    """Test that gradients propagate through array-of-structs (issue #1174)"""
    with wp.ScopedDevice(device):
        x = wp.ones(1, dtype=wp.float32, requires_grad=True)
        y = wp.zeros(1, dtype=ScalarStruct, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(kernel=pack_struct_array_kernel, dim=1, inputs=[x], outputs=[y])
            wp.launch(kernel=loss_from_struct_array_kernel, dim=1, inputs=[y], outputs=[loss])

        tape.backward(loss=loss)

        # Check that gradients propagate correctly
        assert_np_equal(y.grad.numpy()[0][0], 1.0, tol=1e-5)  # y.grad[0].a should be 1.0
        assert_np_equal(x.grad.numpy()[0], 1.0, tol=1e-5)  # x.grad[0] should be 1.0 (not 0.0!)


devices = get_test_devices()


class TestGrad(unittest.TestCase):
    pass


add_function_test(TestGrad, "test_struct_array_gradient_propagation", test_struct_array_gradient_propagation, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
