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

import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class SplitAndSelectSpec extends FlatSpec with Matchers {
  "SplitAndSelect forward" should "be correct" in {
    val layer = SplitAndSelect(1, 3, 4)
    val input = Tensor(T(
      T(0.1f, 0.1f),
      T(0.2f, 0.2f),
      T(0.3f, 0.3f),
      T(0.4f, 0.4f),
      T(0.5f, 0.5f),
      T(0.6f, 0.6f),
      T(0.7f, 0.7f),
      T(0.8f, 0.8f)
    ))

    layer.forward(input) should be(Tensor(T(
      T(0.5f, 0.5f),
      T(0.6f, 0.6f)
    )))
  }

  "SplitAndSelect backward" should "be correct" in {
    val layer = SplitAndSelect(1, 3, 4)
    val input = Tensor(T(
      T(0.1f, 0.1f),
      T(0.2f, 0.2f),
      T(0.3f, 0.3f),
      T(0.4f, 0.4f),
      T(0.5f, 0.5f),
      T(0.6f, 0.6f),
      T(0.7f, 0.7f),
      T(0.8f, 0.8f)
    ))

    val gradOutput = Tensor(T(
      T(0.1f, 0.1f),
      T(0.5f, 0.5f)
    ))

    layer.forward(input) should be(Tensor(T(
      T(0.5f, 0.5f),
      T(0.6f, 0.6f)
    )))

    layer.backward(input, gradOutput) should be(Tensor(T(
      T(0.0f, 0.0f),
      T(0.0f, 0.0f),
      T(0.0f, 0.0f),
      T(0.0f, 0.0f),
      T(0.1f, 0.1f),
      T(0.5f, 0.5f),
      T(0.0f, 0.0f),
      T(0.0f, 0.0f)
    )))
  }
}

class SplitAndSelectSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val splitAndSelect = SplitAndSelect[Float](2, 1, 2).setName("splitSelect")
    val input = Tensor[Float](1, 6, 2).apply1(_ => Random.nextFloat())
    runSerializationTest(splitAndSelect, input)
  }
}
