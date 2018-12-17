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

class SliceSpec extends FlatSpec with Matchers {
  "Slice operation" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    val input =
      Tensor(T(
        T(
          T(1, 2, 3),
          T(4, 5, 6)
        ),
        T(
          T(7, 8, 9),
          T(10, 11, 12)
        ),
        T(
          T(13, 14, 15),
          T(16, 17, 18)
        )
      ))

    val expectOutput =
      Tensor(T(
        T(
          T(5)
        ),
        T(
          T(11)
        )
      ))

    val output = Slice(begin = Array(0, 1, 1), size = Array(2, -1, 1)).forward(input)
    output should be(expectOutput)
  }
}

class SliceSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val slice = Slice[Float](begin = Array(0, 1, 1),
      size = Array(2, -1, 1)).setName("slice")
    val input = Tensor[Float](3, 2, 3).apply1(_ => Random.nextFloat())
    runSerializationTest(slice, input, slice.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }
}
