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
import com.intel.analytics.bigdl.dllib.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.dllib.nn.internal.{Activation, SoftMax, Sequential => KSequential}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.utils.serializer.ModuleSerializationTest

import scala.util.Random

class ActivationSpec extends KerasBaseSpec{

  "tanh" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[4, 5])
        |input = np.random.random([2, 4, 5])
        |output_tensor = Activation('tanh')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = Activation[Float]("tanh", inputShape = Shape(4, 5))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "relu" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[4, 5])
        |input = np.random.random([2, 4, 5])
        |output_tensor = Activation('relu')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = Activation[Float]("relu", inputShape = Shape(4, 5))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "sigmoid" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[4, 5])
        |input = np.random.random([2, 4, 5])
        |output_tensor = Activation('sigmoid')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = Activation[Float]("sigmoid", inputShape = Shape(4, 5))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "hard_sigmoid" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[4, 5])
        |input = np.random.random([2, 4, 5])
        |output_tensor = Activation('hard_sigmoid')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = Activation[Float]("hard_sigmoid", inputShape = Shape(4, 5))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "softmax 2D input" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[8])
        |input = np.random.random([4, 8])
        |output_tensor = Activation('softmax')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = Activation[Float]("softmax", inputShape = Shape(8))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "softmax 3D input" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[4, 5])
        |input = np.random.random([2, 4, 5])
        |output_tensor = Activation('softmax')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = Activation[Float]("softmax", inputShape = Shape(4, 5))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "softmax 4D input" should "be the same as tf.keras" in {
    val seq = KSequential[Float]()
    val layer = Activation[Float]("softmax", inputShape = Shape(2, 2, 2))
    seq.add(layer)

    val input = Tensor[Float](Array[Float](0.24730608f, 0.4993295f, 0.65849156f, 0.01415449f,
      0.30830528f, 0.35399206f, 0.8642406f, 0.89646093f, 0.4380061f, 0.9498415f, 0.24368175f,
      0.27009865f, 0.3127402f, 0.19534959f, 0.27038921f, 0.20165165f),
      Array(2, 2, 2, 2))
    val output = seq.forward(input).toTensor[Float]

    val expectOutput = Tensor[Float](Array[Float](0.43732557f, 0.5626745f, 0.6557332f,
      0.3442668f, 0.48858032f, 0.5114197f, 0.4919456f, 0.5080544f, 0.37476337f,
      0.62523663f, 0.49339616f, 0.50660384f, 0.529314f, 0.470686f, 0.51717764f,
      0.4828224f), Array(2, 2, 2, 2))
    require(output.almostEqual(expectOutput, 7e-3) == true)
  }

  "softplus" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[4, 5])
        |input = np.random.random([2, 4, 5])
        |output_tensor = Activation('softplus')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = Activation[Float]("softplus", inputShape = Shape(4, 5))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "softsign" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[4, 5])
        |input = np.random.random([2, 4, 5])
        |output_tensor = Activation('softsign')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = Activation[Float]("softsign", inputShape = Shape(4, 5))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

}

class ActivationSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = Activation[Float]("tanh", inputShape = Shape(4, 5))
    layer.build(Shape(2, 4, 5))
    val input = Tensor[Float](2, 4, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}

class SoftMaxSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = SoftMax[Float](inputShape = Shape(4, 5))
    layer.build(Shape(3, 4, 5))
    val input = Tensor[Float](3, 4, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
