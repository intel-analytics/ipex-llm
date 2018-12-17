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
package com.intel.analytics.bigdl.nn.tf

import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class BiasAddSpec extends FlatSpec with Matchers {
  "BiasAdd operation" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    val input =
      T(
        Tensor(T(
          T(
            T(1f, 2f, 3f),
            T(2f, 3f, 4f),
            T(3f, 4f, 5f)
          ),
          T(
            T(3f, 4f, 5f),
            T(2f, 3f, 4f),
            T(1f, 2f, 3f)
          ))),
        Tensor(T(3f, 2f, 1f))
      )

    val expectOutput = Tensor(
      T(
        T(
          T(4f, 4f, 4f),
          T(5f, 5f, 5f),
          T(6f, 6f, 6f)
        ),
        T(
          T(6f, 6f, 6f),
          T(5f, 5f, 5f),
          T(4f, 4f, 4f)
        )))
    val expectedGradValue = Tensor(
      T(
        T(
          T(4f, 4f, 4f),
          T(5f, 5f, 5f),
          T(6f, 6f, 6f)
        ),
        T(
          T(6f, 6f, 6f),
          T(5f, 5f, 5f),
          T(4f, 4f, 4f)
        )))

    val expectedGradBias = Tensor(T(30f, 30f, 30f))
    val layer = BiasAdd()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, output)
    output should be(expectOutput)
    gradInput[Tensor[Float]](1) should be(expectedGradValue)
    gradInput[Tensor[Float]](2) should be(expectedGradBias)
  }
}

class BiasAddSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val biasAddGrad = BiasAddGrad[Float](DataFormat.NCHW).
      setName("biasAddGrad")
    val input = Tensor[Float](2, 2, 2).apply1(_ => Random.nextFloat())
    runSerializationTest(biasAddGrad, input)
  }
}
