/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest


class GetShapeSpec extends KerasBaseSpec {

  "GetShape" should "be test" in {
    val seq = Sequential[Float]()
    val input = InputLayer[Float](inputShape = Shape(3, 4), name = "input1")
    seq.add(input)
    val dense = Dense[Float](2, activation = "relu")
    seq.add(dense)
    val ss = new GetShape[Float]()
    seq.add(ss)
    seq.getOutputShape().toSingle().toArray should be(Array(3))
    val outShape = seq.forward(Tensor[Float](Array(2, 3, 4)).randn())
    outShape.toTensor[Float].storage().toArray should be(Array(2, 3, 2))
  }

}

class GetShapeSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val ss = new GetShape[Float](inputShape = Shape(3, 2))
    ss.build(Shape(2, 3, 2))
    val input = Tensor[Float](2, 3, 2).rand()
    runSerializationTest(ss, input)
  }
}
