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
package com.intel.analytics.bigdl.utils.serializer

import com.intel.analytics.bigdl.nn.keras.{Sequential => KSequential}
import com.intel.analytics.bigdl.nn.keras._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.Shape

import scala.collection.mutable
import scala.util.Random

class KerasModuleSerializerSpec extends SerializerSpecHelper {

  override def getPackage(): String = "com.intel.analytics.bigdl.nn.keras"

  override def getExpected(): mutable.Set[String] = {
    super.getExpected().filter(_.contains(getPackage()))
  }

  "Input serializer" should "work properly" in {
    val input = InputLayer[Float](inputShape = Shape(20))
    val inputData = Tensor[Float](2, 20).apply1(_ => Random.nextFloat())
    runSerializationTest(input, inputData)
  }

  "Dense serializer" should "work properly" in {
    val dense = Dense[Float](10, inputShape = Shape(20))
    dense.build(Shape(2, 20))
    val input = Tensor[Float](2, 20).apply1(_ => Random.nextFloat())
    runSerializationTest(dense, input)
  }

  "Sequence serializer" should "work properly" in {
    val dense = Dense[Float](10, inputShape = Shape(20))
    val kseq = KSequential[Float]()
    kseq.add(dense)
    val input = Tensor[Float](2, 20).apply1(_ => Random.nextFloat())
    runSerializationTest(kseq, input)
  }

  "Model serializer" should "work properly" in {
    val input = Input[Float](inputShape = Shape(10))
    val d = Dense[Float](20).setName("dense1").inputs(input)
    val d2 = Dense[Float](5).setName("dense2").inputs(d)
    val model = Model[Float](input, d2)
    val inputData = Tensor[Float](Array(20, 10)).rand()
    runSerializationTest(model, inputData)
  }

  "Convolution2D serializer" should "work properly" in {
    val layer = Convolution2D[Float](64, 2, 5, inputShape = Shape(3, 24, 24))
    layer.build(Shape(2, 3, 24, 24))
    val input = Tensor[Float](2, 3, 24, 24).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "MaxPooling2D serializer" should "work properly" in {
    val layer = MaxPooling2D[Float](inputShape = Shape(3, 24, 24))
    layer.build(Shape(2, 3, 24, 24))
    val input = Tensor[Float](2, 3, 24, 24).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "Activation serializer" should "work properly" in {
    val layer = Activation[Float]("tanh", inputShape = Shape(4, 5))
    layer.build(Shape(2, 4, 5))
    val input = Tensor[Float](2, 4, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "Dropout serializer" should "work properly" in {
    val layer = Dropout[Float](0.3, inputShape = Shape(3, 4))
    layer.build(Shape(2, 3, 4))
    val input = Tensor[Float](2, 3, 4).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "Flatten serializer" should "work properly" in {
    val layer = Flatten[Float](inputShape = Shape(3, 4, 5))
    layer.build(Shape(2, 3, 4, 5))
    val input = Tensor[Float](2, 3, 4, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "Reshape serializer" should "work properly" in {
    val layer = Reshape[Float](Array(4, 15), inputShape = Shape(3, 4, 5))
    layer.build(Shape(2, 3, 4, 5))
    val input = Tensor[Float](2, 3, 4, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "SoftMax serializer" should "work properly" in {
    val layer = SoftMax[Float](inputShape = Shape(4, 5))
    layer.build(Shape(3, 4, 5))
    val input = Tensor[Float](3, 4, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "SimpleRNN serializer" should "work properly" in {
    val layer = SimpleRNN[Float](8, activation = "relu", inputShape = Shape(4, 5))
    layer.build(Shape(3, 4, 5))
    val input = Tensor[Float](3, 4, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "LSTM serializer" should "work properly" in {
    val layer = LSTM[Float](8, returnSequences = true,
      innerActivation = "sigmoid", inputShape = Shape(32, 32))
    layer.build(Shape(3, 32, 32))
    val input = Tensor[Float](3, 32, 32).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "GRU serializer" should "work properly" in {
    val layer = GRU[Float](16, returnSequences = true,
      goBackwards = true, inputShape = Shape(28, 32))
    layer.build(Shape(2, 28, 32))
    val input = Tensor[Float](2, 28, 32).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "Highway serializer" should "work properly" in {
    val layer = Highway[Float](activation = "tanh", bias = false, inputShape = Shape(4))
    layer.build(Shape(3, 4))
    val input = Tensor[Float](3, 4).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "Convolution1D serializer" should "work properly" in {
    val layer = Convolution1D[Float](64, 3, inputShape = Shape(12, 20))
    layer.build(Shape(2, 12, 20))
    val input = Tensor[Float](2, 12, 20).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "Convolution3D serializer" should "work properly" in {
    val layer = Convolution3D[Float](12, 2, 1, 3, inputShape = Shape(3, 32, 32, 32))
    layer.build(Shape(2, 3, 32, 32, 32))
    val input = Tensor[Float](2, 3, 32, 32, 32).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "MaxPooling1D serializer" should "work properly" in {
    val layer = MaxPooling1D[Float](inputShape = Shape(12, 12))
    layer.build(Shape(2, 12, 12))
    val input = Tensor[Float](2, 12, 12).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "MaxPooling3D serializer" should "work properly" in {
    val layer = MaxPooling3D[Float](inputShape = Shape(3, 20, 15, 35))
    layer.build(Shape(2, 3, 20, 15, 35))
    val input = Tensor[Float](2, 3, 20, 15, 35).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "AveragePooling1D serializer" should "work properly" in {
    val layer = AveragePooling1D[Float](inputShape = Shape(12, 16))
    layer.build(Shape(2, 12, 16))
    val input = Tensor[Float](2, 12, 16).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "AveragePooling2D serializer" should "work properly" in {
    val layer = AveragePooling2D[Float](inputShape = Shape(3, 24, 24))
    layer.build(Shape(2, 3, 24, 24))
    val input = Tensor[Float](2, 3, 24, 24).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "AveragePooling3D serializer" should "work properly" in {
    val layer = AveragePooling3D[Float](inputShape = Shape(3, 12, 12, 12))
    layer.build(Shape(2, 3, 12, 12, 12))
    val input = Tensor[Float](2, 3, 12, 12, 12).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "GlobalMaxPooling2D serializer" should "work properly" in {
    val layer = GlobalMaxPooling2D[Float](inputShape = Shape(4, 24, 32))
    layer.build(Shape(2, 4, 24, 32))
    val input = Tensor[Float](2, 4, 24, 32).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "GlobalAveragePooling2D serializer" should "work properly" in {
    val layer = GlobalAveragePooling2D[Float](inputShape = Shape(4, 24, 32))
    layer.build(Shape(2, 4, 24, 32))
    val input = Tensor[Float](2, 4, 24, 32).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "RepeatVector serializer" should "work properly" in {
    val layer = RepeatVector[Float](4, inputShape = Shape(12))
    layer.build(Shape(2, 12))
    val input = Tensor[Float](2, 12).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "Permute serializer" should "work properly" in {
    val layer = Permute[Float](Array(3, 1, 4, 2), inputShape = Shape(3, 4, 5, 6))
    layer.build(Shape(2, 3, 4, 5, 6))
    val input = Tensor[Float](2, 3, 4, 5, 6).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "GlobalAveragePooling1D serializer" should "work properly" in {
    val layer = GlobalAveragePooling1D[Float](inputShape = Shape(3, 24))
    layer.build(Shape(2, 3, 24))
    val input = Tensor[Float](2, 3, 24).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "GlobalAveragePooling3D serializer" should "work properly" in {
    val layer = GlobalAveragePooling3D[Float](inputShape = Shape(3, 4, 5, 6))
    layer.build(Shape(2, 3, 4, 5, 6))
    val input = Tensor[Float](2, 3, 4, 5, 6).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "Cropping1D serializer" should "work properly" in {
    val layer = Cropping1D[Float](inputShape = Shape(5, 6))
    layer.build(Shape(2, 5, 6))
    val input = Tensor[Float](2, 5, 6).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "Cropping2D serializer" should "work properly" in {
    val layer = Cropping2D[Float](inputShape = Shape(3, 8, 12))
    layer.build(Shape(2, 3, 8, 12))
    val input = Tensor[Float](2, 3, 8, 12).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "Cropping3D serializer" should "work properly" in {
    val layer = Cropping3D[Float](inputShape = Shape(4, 12, 16, 20))
    layer.build(Shape(2, 4, 12, 16, 20))
    val input = Tensor[Float](2, 4, 12, 16, 20).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "GlobalMaxPooling1D serializer" should "work properly" in {
    val layer = GlobalMaxPooling1D[Float](inputShape = Shape(12, 24))
    layer.build(Shape(2, 12, 24))
    val input = Tensor[Float](2, 12, 24).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "GlobalMaxPooling3D serializer" should "work properly" in {
    val layer = GlobalMaxPooling3D[Float](inputShape = Shape(12, 24, 3, 6))
    layer.build(Shape(2, 12, 24, 3, 6))
    val input = Tensor[Float](2, 12, 24, 3, 6).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "LocallyConnected2D serializer" should "work properly" in {
    val layer = LocallyConnected2D[Float](32, 2, 2, activation = "relu",
      inputShape = Shape(12, 24, 24))
    layer.build(Shape(2, 12, 24, 24))
    val input = Tensor[Float](2, 12, 24, 24).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "SeparableConvolution2D serializer" should "work properly" in {
    val layer = SeparableConvolution2D[Float](1, 2, 2, inputShape = Shape(3, 128, 128))
    layer.build(Shape(2, 3, 128, 128))
    val input = Tensor[Float](2, 3, 128, 128).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "ZeroPadding3D serializer" should "work properly" in {
    val layer = ZeroPadding3D[Float]((1, 1, 1), inputShape = Shape(5, 6, 7, 8))
    layer.build(Shape(2, 5, 6, 7, 8))
    val input = Tensor[Float](2, 5, 6, 7, 8).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "LocallyConnected1D serializer" should "work properly" in {
    val layer = LocallyConnected1D[Float](32, 3, inputShape = Shape(12, 24))
    layer.build(Shape(2, 12, 24))
    val input = Tensor[Float](2, 12, 24).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "ConvLSTM2D serializer" should "work properly" in {
    val layer = ConvLSTM2D[Float](32, 4, inputShape = Shape(8, 40, 40, 32))
    layer.build(Shape(2, 8, 40, 40, 32))
    val input = Tensor[Float](2, 8, 40, 40, 32).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "Deconvolution2D serializer" should "work properly" in {
    val layer = Deconvolution2D[Float](3, 3, 3, inputShape = Shape(3, 24, 24))
    layer.build(Shape(2, 12, 24, 24))
    val input = Tensor[Float](2, 12, 24, 24).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "AtrousConvolution1D serializer" should "work properly" in {
    val layer = AtrousConvolution1D[Float](64, 3, inputShape = Shape(8, 32))
    layer.build(Shape(2, 8, 32))
    val input = Tensor[Float](2, 8, 32).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

  "AtrousConvolution2D serializer" should "work properly" in {
    val layer = AtrousConvolution2D[Float](32, 2, 4, atrousRate = (2, 2),
      inputShape = Shape(3, 64, 64))
    layer.build(Shape(2, 3, 64, 64))
    val input = Tensor[Float](2, 3, 64, 64).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }

}
