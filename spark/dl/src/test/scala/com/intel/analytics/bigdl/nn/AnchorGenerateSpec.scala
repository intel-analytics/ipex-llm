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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}
import org.scalatest.{FlatSpec, Matchers}

class AnchorGenerateSpec extends FlatSpec with Matchers {
  "AnchorGenerate" should "be ok" in {
    val ratios = Array[Float](0.5f, 1.0f, 2.0f)
    val sizes = Array[Float](32f, 64f, 128f, 256f, 512f)
    val strides = Array[Float](4f, 8f, 16f, 32f, 64f)
    val anchor = new AnchorGenerate(sizes, ratios, strides)

    val input = Tensor[Float](T(T(T(T(0.7668, 0.1659, 0.4393, 0.2243),
      T(0.8935, 0.0497, 0.1780, 0.3011),
      T(0.1893, 0.9186, 0.2131, 0.3957)),

      T(T(0.6017, 0.4234, 0.5224, 0.4175),
        T(0.0340, 0.9157, 0.3079, 0.6269),
        T(0.8277, 0.6594, 0.0887, 0.4890)),

      T(T(0.5887, 0.7340, 0.8497, 0.9112),
        T(0.4847, 0.9436, 0.3904, 0.2499),
        T(0.3206, 0.9753, 0.7582, 0.6688)),

      T(T(0.2651, 0.2336, 0.5057, 0.5688),
        T(0.0634, 0.8993, 0.2732, 0.3397),
        T(0.1879, 0.5534, 0.2682, 0.9556)),

      T(T(0.9761, 0.5934, 0.3124, 0.9431),
        T(0.8519, 0.9815, 0.1132, 0.4783),
        T(0.4436, 0.3847, 0.4521, 0.5569)),

      T(T(0.9952, 0.0015, 0.0813, 0.4907),
        T(0.2130, 0.4603, 0.1386, 0.0277),
        T(0.5662, 0.3503, 0.6555, 0.7667)))))

    val imgInfo = Tensor[Float](T(T(20, 38)))

    val expectedOutput = Tensor[Float](T(T(-22, -10,  25,  13),
      T(-14, -14,  17,  17),
      T(-10, -22,  13,  25),
      T(-18, -10,  29,  13),
      T(-10, -14,  21,  17),
      T( -6, -22,  17,  25),
      T(-14, -10,  33,  13),
      T( -6, -14,  25,  17),
      T( -2, -22,  21,  25),
      T(-10, -10,  37,  13),
      T( -2, -14,  29,  17),
      T(  2, -22,  25,  25),
      T(-22,  -6,  25,  17),
      T(-14, -10,  17,  21),
      T(-10, -18,  13,  29),
      T(-18,  -6,  29,  17),
      T(-10, -10,  21,  21),
      T( -6, -18,  17,  29),
      T(-14,  -6,  33,  17),
      T( -6, -10,  25,  21),
      T( -2, -18,  21,  29),
      T(-10,  -6,  37,  17),
      T( -2, -10,  29,  21),
      T(  2, -18,  25,  29),
      T(-22,  -2,  25,  21),
      T(-14,  -6,  17,  25),
      T(-10, -14,  13,  33),
      T(-18,  -2,  29,  21),
      T(-10,  -6,  21,  25),
      T( -6, -14,  17,  33),
      T(-14,  -2,  33,  21),
      T( -6,  -6,  25,  25),
      T( -2, -14,  21,  33),
      T(-10,  -2,  37,  21),
      T( -2,  -6,  29,  25),
      T(  2, -14,  25,  33)))

    val out = anchor.forward(T(input, imgInfo))

    out.apply[Table](1).apply[Tensor[Float]](1) should be(expectedOutput)
    println("done")

  }
}
