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
import com.intel.analytics.bigdl.utils.{MultiShape, Shape, T}
import com.intel.analytics.zoo.pipeline.api.keras.models._
import com.intel.analytics.zoo.pipeline.api.keras.layers.Merge.merge
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest


class MergeSpec extends KerasBaseSpec {

  "Merge sum" should "work properly" in {
    val input1 = Tensor[Float](2, 4, 8).rand()
    val input2 = Tensor[Float](2, 4, 8).rand()
    val input = T(1 -> input1, 2 -> input2)
    val seq = Sequential[Float]()
    val l1 = InputLayer[Float](inputShape = Shape(4, 8))
    val l2 = InputLayer[Float](inputShape = Shape(4, 8))
    val layer = Merge[Float](layers = List(l1, l2), mode = "sum")
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 4, 8))
    seq.forward(input) should be (input1 + input2)
  }

  "Merge sub" should "work properly" in {
    val input1 = Tensor[Float](2, 3, 4).rand()
    val input2 = Tensor[Float](2, 3, 4).rand()
    val input = T(1 -> input1, 2 -> input2)
    val seq = Sequential[Float]()
    val l1 = InputLayer[Float](inputShape = Shape(3, 4))
    val l2 = InputLayer[Float](inputShape = Shape(3, 4))
    val layer = Merge[Float](layers = List(l1, l2), mode = "sub")
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 3, 4))
    seq.forward(input) should be (input1 - input2)
  }

  "Merge div" should "work properly" in {
    val input1 = Tensor[Float](2, 3, 4).rand()
    val input2 = Tensor[Float](2, 3, 4).rand()
    val input = T(1 -> input1, 2 -> input2)
    val seq = Sequential[Float]()
    val l1 = InputLayer[Float](inputShape = Shape(3, 4))
    val l2 = InputLayer[Float](inputShape = Shape(3, 4))
    val layer = Merge[Float](layers = List(l1, l2), mode = "div")
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 3, 4))
    seq.forward(input) should be (input1 / input2)
  }

  "Merge min" should "work properly" in {
    val input1 = Tensor[Float](3, 8).rand(0, 1)
    val input2 = Tensor[Float](3, 8).rand(1, 2)
    val input = T(1 -> input1, 2 -> input2)
    val seq = Sequential[Float]()
    val l1 = InputLayer[Float](inputShape = Shape(8))
    val l2 = InputLayer[Float](inputShape = Shape(8))
    val layer = Merge[Float](layers = List(l1, l2), mode = "min")
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 8))
    seq.forward(input) should be (input1)
  }

  "merge method" should "work correctly" in {
    val input1 = Tensor[Float](2, 8).rand()
    val input2 = Tensor[Float](2, 12).rand()
    val input = T(1 -> input1, 2 -> input2)
    val l1 = Input[Float](inputShape = Shape(8))
    val l2 = Input[Float](inputShape = Shape(12))
    val dense1 = Dense[Float](10).inputs(l1)
    val dense2 = Dense[Float](10).inputs(l2)
    val output = merge(inputs = List(dense1, dense2), mode = "sum")
    val model = Model[Float](Array(l1, l2), output)
    model.forward(input)
    model.getOutputShape().toSingle().toArray should be (Array(-1, 10))
  }

  "Merge with incompatible input shapes" should "raise an exception" in {
    intercept[RuntimeException] {
      val seq = Sequential[Float]()
      val l1 = InputLayer[Float](inputShape = Shape(4))
      val l2 = InputLayer[Float](inputShape = Shape(5))
      val layer = Merge[Float](layers = List(l1, l2), mode = "cosine",
        inputShape = MultiShape(List(Shape(4), Shape(4))))
      seq.add(layer)
    }
  }

  "Merge ave" should "work properly" in {
    val input1 = Tensor[Float](3, 10).rand()
    val input2 = Tensor[Float](3, 10).rand()
    val input3 = Tensor[Float](3, 10).rand()
    val input = T(1 -> input1, 2 -> input2, 3 -> input3)
    val seq = Sequential[Float]()
    val l1 = InputLayer[Float](inputShape = Shape(10))
    val l2 = InputLayer[Float](inputShape = Shape(10))
    val l3 = InputLayer[Float](inputShape = Shape(10))
    val layer = Merge[Float](layers = List(l1, l2, l3), mode = "ave")
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 10))
    seq.forward(input) should be ((input1 + input2 + input3)/3)
  }

  "Merge concat_0" should "work properly" in {
    val input1 = Tensor[Float](2, 3, 8).rand()
    val input2 = Tensor[Float](2, 3, 8).rand()
    val input = T(1 -> input1, 2 -> input2)
    val seq = Sequential[Float]()
    val l1 = InputLayer[Float](inputShape = Shape(3, 8))
    val l2 = InputLayer[Float](inputShape = Shape(3, 8))
    val layer = Merge[Float](layers = List(l1, l2), mode = "concat", concatAxis = 0)
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 3, 8))
    seq.forward(input)
  }

  "Merge concat" should "work properly" in {
    val input1 = Tensor[Float](2, 3, 8).rand()
    val input2 = Tensor[Float](2, 4, 8).rand()
    val input = T(1 -> input1, 2 -> input2)
    val seq = Sequential[Float]()
    val l1 = InputLayer[Float](inputShape = Shape(3, 8))
    val l2 = InputLayer[Float](inputShape = Shape(4, 8))
    val layer = Merge[Float](layers = List(l1, l2), mode = "concat", concatAxis = 1)
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 7, 8))
    seq.forward(input)
  }

  "Merge dot" should "work properly" in {
    val input1 = Tensor[Float](2, 4).rand()
    val input2 = Tensor[Float](2, 4).rand()
    val input = T(1 -> input1, 2 -> input2)
    val seq = Sequential[Float]()
    val l1 = InputLayer[Float](inputShape = Shape(4))
    val l2 = InputLayer[Float](inputShape = Shape(4))
    val layer = Merge[Float](layers = List(l1, l2), mode = "dot")
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 1))
    seq.forward(input)
  }

  "Merge dense" should "work properly" in {
    val input1 = Tensor[Float](3, 8).rand()
    val input2 = Tensor[Float](3, 6).rand()
    val input = T(1 -> input1, 2 -> input2)
    val seq = Sequential[Float]()
    val l1 = Dense[Float](10, inputShape = Shape(8))
    val l2 = Dense[Float](10, inputShape = Shape(6))
    val layer = Merge[Float](layers = List(l1, l2), mode = "max")
    seq.add(layer)
    seq.add(Dense[Float](15))
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 15))
    seq.forward(input)
  }

}

class MergeSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val l1 = InputLayer[Float](inputShape = Shape(4, 8))
    val l2 = InputLayer[Float](inputShape = Shape(4, 8))
    val layer = Merge[Float](layers = List(l1, l2), mode = "sum")
    layer.build(Shape(List(Shape(2, 4, 8), Shape(2, 4, 8))))
    val input1 = Tensor[Float](2, 4, 8).rand()
    val input2 = Tensor[Float](2, 4, 8).rand()
    val input = T(1 -> input1, 2 -> input2)
    runSerializationTest(layer, input)
  }
}
