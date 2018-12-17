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

class TileSpec extends FlatSpec with Matchers {
  "Tile operation" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    val input =
      T(
        Tensor(
          T(
            T(
              T(1f, 2f, 3f),
              T(2f, 2f, 4f),
              T(2f, 2f, 4f)),
            T(
              T(2f, 2f, 3f),
              T(2f, 2f, 4f),
              T(2f, 2f, 4f))
          )),
        Tensor[Int](T(2, 1, 2))
      )

    val expectOutput = Tensor(
      T(
        T(
          T(1f, 2f, 3f, 1f, 2f, 3f),
          T(2f, 2f, 4f, 2f, 2f, 4f),
          T(2f, 2f, 4f, 2f, 2f, 4f)),
        T(
          T(2f, 2f, 3f, 2f, 2f, 3f),
          T(2f, 2f, 4f, 2f, 2f, 4f),
          T(2f, 2f, 4f, 2f, 2f, 4f)),
        T(
          T(1f, 2f, 3f, 1f, 2f, 3f),
          T(2f, 2f, 4f, 2f, 2f, 4f),
          T(2f, 2f, 4f, 2f, 2f, 4f)),
        T(
          T(2f, 2f, 3f, 2f, 2f, 3f),
          T(2f, 2f, 4f, 2f, 2f, 4f),
          T(2f, 2f, 4f, 2f, 2f, 4f))
      ))

    val output = Tile().forward(input)
    output should be(expectOutput)
  }

  "Tile operation" should "handle empty multiples tensor" in {
    val scalar = Tensor.scalar(1)
    val multiply = Tensor[Int]()
    Tile[Float]().forward(T(scalar, multiply)) should be(Tensor.scalar(1))
  }
}

class TileSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val tile = Tile[Float]().setName("tileOps")
    val input = T(Tensor[Float](2, 3, 3).apply1(_ => Random.nextFloat()),
      Tensor[Int](T(2, 1, 2)))
    runSerializationTest(tile, input, tile.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }
}
