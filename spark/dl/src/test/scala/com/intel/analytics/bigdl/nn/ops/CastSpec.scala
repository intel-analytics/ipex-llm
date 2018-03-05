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

import com.intel.analytics.bigdl.nn.tf.Assign
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class CastSpec extends FlatSpec with Matchers {
  "Cast operation Float" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    val input =
        Tensor(T(1.5f, 2.1f, 3.1f))

    val expectOutput = Tensor[Int](T(1, 2, 3))

    val output = Cast[Float, Int]().forward(input)
    output should be(expectOutput)
  }

  "Cast operation Double" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val input =
      T(
        Tensor(T(1.0, 2.0, 3.0)),
        Tensor(T(2.0, 2.0, 4.0))
      )

    val expectOutput = Tensor(T(2.0, 2.0, 4.0))

    val output = new Assign().forward(input)
    output should be(expectOutput)
  }
}

class CastSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val cast = Cast[Float, Float]().setName("cast")
    val input = Tensor[Float](2, 2).apply1(_ => Random.nextFloat())
    runSerializationTest(cast, input, cast.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }
}
