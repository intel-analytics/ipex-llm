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

package com.intel.analytics.bigdl.keras.nn

import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.nn.keras.{Dropout, Sequential => KSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class DropoutSpec extends FlatSpec with Matchers {

  "Dropout forward and backward" should "work properly" in {
    val seq = KSequential[Float]()
    val layer = Dropout[Float](0.3, inputShape = Shape(3, 4))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 3, 4))
    val input = Tensor[Float](2, 3, 4).rand()
    val output = seq.forward(input)
    val gradInput = seq.backward(input, output)
  }

}

class DropoutSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = Dropout[Float](0.3, inputShape = Shape(3, 4))
    layer.build(Shape(2, 3, 4))
    val input = Tensor[Float](2, 3, 4).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}

