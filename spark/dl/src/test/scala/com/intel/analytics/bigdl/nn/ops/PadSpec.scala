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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class PadSpec extends FlatSpec with Matchers {
  "Pad operation" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    val input =
      Tensor(T(
        T(
          T(1f, 2f, 3f),
          T(4f, 5f, 6f)),
        T(
          T(1f, 2f, 3f),
          T(4f, 5f, 6f))
      ))
    val padding = Tensor[Int](T(T(1, 2), T(1, 2), T(1, 2)))

    val expectOutput = Tensor(
      T(
        T(
          T(0f, 0f, 0f, 0f, 0f, 0f),
          T(0f, 0f, 0f, 0f, 0f, 0f),
          T(0f, 0f, 0f, 0f, 0f, 0f),
          T(0f, 0f, 0f, 0f, 0f, 0f),
          T(0f, 0f, 0f, 0f, 0f, 0f)),
        T(
          T(0f, 0f, 0f, 0f, 0f, 0f),
          T(0f, 1f, 2f, 3f, 0f, 0f),
          T(0f, 4f, 5f, 6f, 0f, 0f),
          T(0f, 0f, 0f, 0f, 0f, 0f),
          T(0f, 0f, 0f, 0f, 0f, 0f)),
        T(
          T(0f, 0f, 0f, 0f, 0f, 0f),
          T(0f, 1f, 2f, 3f, 0f, 0f),
          T(0f, 4f, 5f, 6f, 0f, 0f),
          T(0f, 0f, 0f, 0f, 0f, 0f),
          T(0f, 0f, 0f, 0f, 0f, 0f)),
        T(
          T(0f, 0f, 0f, 0f, 0f, 0f),
          T(0f, 0f, 0f, 0f, 0f, 0f),
          T(0f, 0f, 0f, 0f, 0f, 0f),
          T(0f, 0f, 0f, 0f, 0f, 0f),
          T(0f, 0f, 0f, 0f, 0f, 0f)),
        T(
          T(0f, 0f, 0f, 0f, 0f, 0f),
          T(0f, 0f, 0f, 0f, 0f, 0f),
          T(0f, 0f, 0f, 0f, 0f, 0f),
          T(0f, 0f, 0f, 0f, 0f, 0f),
          T(0f, 0f, 0f, 0f, 0f, 0f)))
    )

    val output = Pad[Float, Float](mode = "CONSTANT", 0.0f).forward(T(input, padding))
    output should be(expectOutput)
  }
}

class PadSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val pad = Pad[Float, Float](mode = "CONSTANT", 0.0f).setName("pad")
    val inputTensor = Tensor[Float](2, 2, 3).apply1(_ => Random.nextFloat())
    val padding = Tensor[Int](T(T(1, 2), T(1, 2), T(1, 2)))
    val input = T(inputTensor, padding)
    runSerializationTest(pad, input, pad.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }
}
