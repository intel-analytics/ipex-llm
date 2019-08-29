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

class OnnxSumSpec extends FlatSpec with Matchers {
  "OnnxSum" should "work" in {
    val onnxSum = OnnxSum[Float]()
    val input = T(
      Tensor[Float](5, 5).rand(),
      Tensor[Float](5, 5).rand(),
      Tensor[Float](5, 5).rand()
    )
    val out = onnxSum.forward(input)

    // Todo: verify output
    println(out)
  }
}


class OnnxSumSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val onnxSum = OnnxSum[Float]()
    val input = T(
      Tensor[Float](5, 5).rand(),
      Tensor[Float](5, 5).rand(),
      Tensor[Float](5, 5).rand()
    )
    runSerializationTest(onnxSum, input)
  }
}
