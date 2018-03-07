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

class SelectSpec extends FlatSpec with Matchers {
  "select" should "be correct when condition is true" in {
    val cond = Tensor.scalar[Boolean](true)
    val t = Tensor[Int](T(1))
    val e = Tensor[Int](T(2))

    val ops = Select[Float]()
    ops.forward(T(cond, t, e)) should be(t)
  }

  "select" should "be correct when condition is false" in {
    val cond = Tensor.scalar[Boolean](false)
    val t = Tensor[Int](T(1))
    val e = Tensor[Int](T(2))

    val ops = Select[Float]()
    ops.forward(T(cond, t, e)) should be(e)
  }
}

class SelectSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val select = Select[Float]().setName("select")
    val cond = Tensor.scalar[Boolean](true)
    val t = Tensor[Int](T(1))
    val e = Tensor[Int](T(2))
    val input = T(cond, t, e)
    runSerializationTest(select, input)
  }
}
