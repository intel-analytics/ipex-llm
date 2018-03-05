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

import com.intel.analytics.bigdl.keras.KerasBaseSpec
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.keras.{ZeroPadding1D, Sequential => KSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class ZeroPadding1DSpec extends KerasBaseSpec {

  "ZeroPadding1D" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[4, 5])
        |input = np.random.random([2, 4, 5])
        |output_tensor = ZeroPadding1D(padding=2)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = ZeroPadding1D[Float](padding = 2, inputShape = Shape(4, 5))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 8, 5))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "ZeroPadding1D asymmetric" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 3])
        |input = np.random.random([2, 3, 3])
        |output_tensor = ZeroPadding1D(padding=(2, 3))(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = new ZeroPadding1D[Float](padding = Array(2, 3), inputShape = Shape(3, 3))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 8, 3))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

}

class ZeroPadding1DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = ZeroPadding1D[Float](padding = 2, inputShape = Shape(4, 5))
    layer.build(Shape(2, 4, 5))
    val input = Tensor[Float](2, 4, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
