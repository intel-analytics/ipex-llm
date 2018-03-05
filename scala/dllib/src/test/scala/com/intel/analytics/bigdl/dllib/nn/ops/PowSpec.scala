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

class PowSpec extends FlatSpec with Matchers {

  "Power float" should "be right" in {
    val v = Tensor[Float](T(2))
    val t = Tensor[Float](T(1, 2, 3))
    val ops = Pow[Float]()
    ops.forward(T(t, v)) should be(Tensor[Float](T(1, 4, 9)))
  }

  "Power double" should "be right" in {
    val v = Tensor[Double](T(2))
    val t = Tensor[Double](T(1, 2, 3))
    val ops = Pow[Double]()
    ops.forward(T(t, v)) should be(Tensor[Double](T(1, 4, 9)))
  }
}

class PowSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val pow = Pow[Float]().setName("powOps")
    val v = Tensor[Float](T(2))
    val t = Tensor[Float](T(1, 2, 3))
    val input = (T(t, v))
    runSerializationTest(pow, input)
  }
}
