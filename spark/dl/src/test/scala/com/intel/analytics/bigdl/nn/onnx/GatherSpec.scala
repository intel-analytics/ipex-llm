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

class GatherSpec extends FlatSpec with Matchers {

  "Gather" should "work" in {
    val data = Tensor[Float](3, 2)
    data.setValue(1, 1, 1.0f)
    data.setValue(1, 2, 1.2f)
    data.setValue(2, 1, 2.3f)
    data.setValue(2, 2, 3.4f)
    data.setValue(3, 1, 4.5f)
    data.setValue(3, 2, 5.7f)

    val indices = Tensor[Float](2, 2)
    indices.setValue(1, 1, 0f)
    indices.setValue(1, 2, 1f)
    indices.setValue(2, 1, 1f)
    indices.setValue(2, 2, 2f)

    val model = Gather[Float, Float]()
    val input = T(data, indices)
    val out = model.forward(input)

    // Todo: verify output
    println(out.toString)

  }
}


class GatherSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val gather = Gather[Float, Float]()
    val data = Tensor[Float](3, 2)
    data.setValue(1, 1, 1.0f)
    data.setValue(1, 2, 1.2f)
    data.setValue(2, 1, 2.3f)
    data.setValue(2, 2, 3.4f)
    data.setValue(3, 1, 4.5f)
    data.setValue(3, 2, 5.7f)

    val indices = Tensor[Float](2, 2)
    indices.setValue(1, 1, 0f)
    indices.setValue(1, 2, 1f)
    indices.setValue(2, 1, 1f)
    indices.setValue(2, 2, 2f)

    val model = Gather[Float, Float]()
    val input = T(data, indices)

    runSerializationTest(gather, input)
  }
}