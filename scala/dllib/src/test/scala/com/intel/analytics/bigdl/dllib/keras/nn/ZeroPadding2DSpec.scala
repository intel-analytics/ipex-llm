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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat}
import com.intel.analytics.bigdl.nn.keras.{ZeroPadding2D, Sequential => KSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class ZeroPadding2DSpec extends KerasBaseSpec {

  "ZeroPadding2D NCHW" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[2, 8, 8])
        |input = np.random.random([3, 2, 8, 8])
        |output_tensor = ZeroPadding2D(padding=(2, 1), dim_ordering="th")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = ZeroPadding2D[Float](padding = (2, 1), inputShape = Shape(2, 8, 8))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 2, 12, 10))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "ZeroPadding2D NCHW asymmetric" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[2, 4, 5])
        |input = np.random.random([3, 2, 4, 5])
        |output_tensor = ZeroPadding2D(padding=(2, 1, 3, 2), dim_ordering="th")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = new ZeroPadding2D[Float](padding = Array(2, 1, 3, 2), inputShape = Shape(2, 4, 5))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 2, 7, 10))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "ZeroPadding2D NHWC" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[6, 8, 1])
        |input = np.random.random([3, 6, 8, 1])
        |output_tensor = ZeroPadding2D(dim_ordering="tf")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = ZeroPadding2D[Float](dimOrdering = "tf", inputShape = Shape(6, 8, 1))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 8, 10, 1))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "ZeroPadding2D NHWC asymmetric" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[5, 5, 2])
        |input = np.random.random([3, 5, 5, 2])
        |output_tensor = ZeroPadding2D(padding=(1, 2, 3, 4), dim_ordering="tf")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = new ZeroPadding2D[Float](padding = Array(1, 2, 3, 4), dimOrdering = DataFormat.NHWC,
      inputShape = Shape(5, 5, 2))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 8, 12, 2))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

}

class ZeroPadding2DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = ZeroPadding2D[Float](padding = (2, 1), inputShape = Shape(2, 8, 8))
    layer.build(Shape(2, 2, 8, 8))
    val input = Tensor[Float](2, 2, 8, 8).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
