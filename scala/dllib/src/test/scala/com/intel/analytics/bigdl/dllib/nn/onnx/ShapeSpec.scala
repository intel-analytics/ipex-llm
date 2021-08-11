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
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}


class ShapeSpec extends FlatSpec with Matchers {

  "Shape" should "work" in {
    val inputTensor = Tensor[Float](20, 1, 9).rand()
    val shape = Shape[Float]()
    val output = shape.forward(inputTensor)
    val ans = Tensor[Float](3)
    ans.setValue(1, 20)
    ans.setValue(2, 1)
    ans.setValue(3, 9)

    output.nDimension() should be (1)
    output.nDimension() should be (ans.nDimension())
    output.size(1) should be (ans.size(1))

    (1 to output.size(1)).foreach(i => {
      output.valueAt(i) should be (ans.valueAt(i))
    })

  }
}

class ShapeSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val shape = Shape[Float]()
    val input = Tensor[Float](5).rand()
    runSerializationTest(shape, input)
  }

}

