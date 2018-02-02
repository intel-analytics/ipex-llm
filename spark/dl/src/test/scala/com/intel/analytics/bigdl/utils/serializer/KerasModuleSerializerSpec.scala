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

}

