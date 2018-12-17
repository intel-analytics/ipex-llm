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


class ArgMaxSpec extends FlatSpec with Matchers {

  "ArgMax Float" should "work properly" in {
    val dataTensor = Tensor[Float](T(T(1.0f, 2.0f), T(3.0f, 4.0f)))
    val dimensionTensor = Tensor.scalar[Int](1)
    val input = T(dataTensor, dimensionTensor)
    val expectedOutput = Tensor[Int](T(2, 2))

    val layer = ArgMax[Double]()
    val result = layer.forward(input)

    result should be (expectedOutput)
  }

}

class ArgMaxSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val argMax = ArgMax[Float].setName("argMax")
    val dataTensor = Tensor[Float](T(T(1.0f, 2.0f), T(3.0f, 4.0f)))
    val dimensionTensor = Tensor.scalar[Int](1)
    val input = T(dataTensor, dimensionTensor)
    runSerializationTest(argMax, input)
  }
}
