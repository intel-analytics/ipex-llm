/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.nn

import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

class FPNSpec extends FlatSpec with Matchers {
  "test" should "work correctly" in {
    val in_channels_list = Array(1, 2, 4)
    val out_channels = 2
    val model = FPN(in_channels_list, out_channels)

    /*
    val feature0x =
      Tensor(
        T(
          T(0.5417, 0.6189, 0.4326, 0.0828, 0.1496, 0.7923, 0.6045, 0.1090),
          T(0.8257, 0.3597, 0.0780, 0.0216, 0.7714, 0.9073, 0.0178, 0.7351),
          T(0.9778, 0.6250, 0.4430, 0.6592, 0.0397, 0.4242, 0.5416, 0.1251),
          T(0.9990, 0.0348, 0.2702, 0.3132, 0.0969, 0.1247, 0.0750, 0.8420),
          T(0.7007, 0.9568, 0.4179, 0.9242, 0.3360, 0.0589, 0.2398, 0.5842),
          T(0.4648, 0.7644, 0.1408, 0.4335, 0.6977, 0.3636, 0.0444, 0.1101),
          T(0.1859, 0.7598, 0.8315, 0.0346, 0.3133, 0.9713, 0.0894, 0.3608),
          T(0.5934, 0.2739, 0.6590, 0.6748, 0.0299, 0.3118, 0.6790, 0.4435)
        )).reshape(Array(1, 1, 8, 8))

    val feature1x =
      Tensor(
        T(
          T(
            T(0.0045, 0.4870, 0.0269, 0.6793),
            T(0.8730, 0.5867, 0.1536, 0.9963),
            T(0.8477, 0.5264, 0.0951, 0.6716),
            T(0.4876, 0.8124, 0.9424, 0.4415)
          ),
          T(
            T(0.0665, 0.3682, 0.4516, 0.6683),
            T(0.0632, 0.1431, 0.6429, 0.8086),
            T(0.8182, 0.7684, 0.7524, 0.7243),
            T(0.0616, 0.8783, 0.8404, 0.3362)
          )
        )
      )

    val feature2x =
      Tensor(
        T(
          T(
            T(0.6832, 0.7758),
            T(0.8380, 0.1164)
          ),
          T(
            T(0.5668, 0.3116),
            T(0.8366, 0.1092)
          ),
          T(
            T(0.1225, 0.3929),
            T(0.0450, 0.1906)
          ),
          T(
            T(0.6821, 0.8735),
            T(0.1885, 0.1869)
          )
        )
      )

    val inner_wei1 = Tensor(
      T(T(T(T(-1.4994, 0.1677))))
    )

    val inner_bias1 = Tensor(T(0, 0))

    val inner_wei2 = Tensor(
      T(T(T(T(0.5772, 0.4900),
            T(-0.1733, 1.1133)))))

    val inner_bias2 = Tensor(T(0, 0))

    val inner_wei3 = Tensor(
      T(T(T(T(-0.7950, 0.8102),
        T(0.0071, 0.7662),
        T(-0.7967, 0.4904),
        T(0.1939, 0.7008)))))

    val inner_bias3 = Tensor(0, 0)
    */

    val feature0 = Tensor(Array(1, in_channels_list(0), 8, 8)).randn()
    val feature1 = Tensor(Array(1, in_channels_list(1), 4, 4)).randn()
    val feature2 = Tensor(Array(1, in_channels_list(2), 2, 2)).randn()

    val input = T(feature0, feature1, feature2)
    val output = model.forward(input)
    val param = model.parameters()
    println()
  }
}
