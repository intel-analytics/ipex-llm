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
import com.intel.analytics.zoo.pipeline.api.autograd.Parameter
import com.intel.analytics.zoo.pipeline.api.keras.models.{Model, Sequential}
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest

import scala.util.Random

class Expand_dimSpec extends KerasBaseSpec {
  //  seq.add(new Expand_dim[Float](dim = 0))

  "ExpandDim0" should "be test" in {
    val input = Parameter[Float](inputShape = Shape(3), name = "input1")
    val expand = new ExpandDim[Float](dim = 0).from(input)
    val seq = Model(input, expand)
    seq.getOutputShape().toSingle().toArray should be(Array(1, 3))
    val inputData = Tensor[Float](Array(3)).randn()
    val out = seq.forward(inputData).toTensor[Float]
    assert(out.size().deep == Array(1, 3).deep)
    out.toTensor[Float].almostEqual(inputData.addSingletonDimension(dim = 1), 1e-4)
  }

  "ExpandDim1" should "be test" in {
    val input = Parameter[Float](inputShape = Shape(3, 4), name = "input2")
    val expand = new ExpandDim[Float](dim = 1).from(input)
    val seq = Model(input, expand)
    seq.getOutputShape().toSingle().toArray should be(Array(3, 1, 4))
    val inputData = Tensor[Float](Array(2, 1, 6)).rand()
    val out = seq.forward(inputData).toTensor[Float]
    assert(out.size().deep == Array(3, 1, 4).deep)
    out.toTensor[Float].almostEqual(inputData.addSingletonDimension(dim = 1), 1e-4)
  }
}

class ExpandDimSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val ss = new ExpandDim[Float](inputShape = Shape(3, 2))
    ss.build(Shape(2, 3, 2))
    val input = Tensor[Float](2, 3, 2).apply1(_ => Random.nextFloat())
    runSerializationTest(ss, input)
  }
}
