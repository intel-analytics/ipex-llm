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
package com.intel.analytics.bigdl.nn.onnx

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}


class ReshapeSpec extends FlatSpec with Matchers {

  "Reshape" should "work" in {
    // case when shape is an attribute
    val inputTensor = Tensor[Float](4, 8).rand()
    val shape = Array[Int](2, 2, 8)
    var reshape = Reshape[Float](shape)
    var output = reshape.forward(inputTensor)

    output.size() should be (shape)

    // case when shape is an input
    reshape = Reshape()
    val shapeTensor = Tensor[Float](3)
    shapeTensor.setValue(1, 2)
    shapeTensor.setValue(2, 2)
    shapeTensor.setValue(3, 8)
    output = reshape.forward(T(inputTensor, shapeTensor))
    output.size() should be (shape)

  }
}

class ReshapeSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val reshape = Reshape[Float](Array(2, 2, 8))
    val input = Tensor[Float](4, 8).rand()
    runSerializationTest(reshape, input)
  }

}

