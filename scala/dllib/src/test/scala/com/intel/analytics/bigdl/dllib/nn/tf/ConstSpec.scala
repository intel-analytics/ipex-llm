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

class ConstSpec extends FlatSpec with Matchers {
  "Const forward tensor" should "be correct" in {
    val value = Tensor(2, 3).rand()
    val layer = Const(value)
    val input = Tensor(4, 5).rand()
    layer.forward(input) should be(value)
  }

  "Const forward tensors" should "be correct" in {
    val value = Tensor(2, 3).rand()
    val layer = Const(value)
    val input = T(Tensor(4, 5).rand(), Tensor(3, 4).rand())
    layer.forward(input) should be(value)
  }
}

class ConstSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val value = Tensor[Float](3).apply1(_ => Random.nextFloat())
    val const = Const[Float, Float](value).setName("const")
    val input = Tensor[Float](3).apply1(_ => Random.nextFloat())
    runSerializationTest(const, input)
  }
}
