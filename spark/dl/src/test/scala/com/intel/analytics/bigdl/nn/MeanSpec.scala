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
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class MeanSpec extends FlatSpec with Matchers {

  "mean" should "work correctly" in {
    val input = Tensor[Float](T(
      T(1.0f, 2.0f),
      T(3.0f, 4.0f)
    ))

    val layer = Mean[Float](dimension = 2)

    val expect = Tensor[Float](T(1.5f, 3.5f))

    layer.forward(input) should be(expect)
  }

  "mean" should "work correctly without squeeze" in {
    val input = Tensor[Float](T(
      T(1.0f, 2.0f),
      T(3.0f, 4.0f)
    ))

    val layer = Mean[Float](dimension = 2, squeeze = false)

    val expect = Tensor[Float](T(T(1.5f), T(3.5f)))

    layer.forward(input) should be(expect)
  }
}

class MeanSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val mean = Mean[Float](2).setName("mean")
    val input = Tensor[Float](5, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(mean, input)
  }
}
