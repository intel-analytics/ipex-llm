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

package com.intel.analytics.zoo.pipeline.api.keras.models

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.autograd.{AutoGrad, Parameter, Variable}
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}


class ModelSerialTest extends ModuleSerializationTest {
  private def testParameterSerialWithModel(): Unit = {
    val input = Variable[Float](Shape(3))
    val w = Parameter[Float](Shape(2, 3)) // outputSize * inputSize
    val bias = Parameter[Float](Shape(2))
    val cDense = AutoGrad.mm(input, w, axes = List(1, 1)) + bias
    val model = Model[Float](input = input, output = cDense)

    val inputData = Tensor[Float](8, 3).rand()
    runSerializationTest(model, inputData)
  }

  override def test(): Unit = {
    val input = Input[Float](inputShape = Shape(10))
    val model = Model(input, Dense[Float](8).inputs(input))
    val tmpFile = ZooSpecHelper.createTmpFile()
    model.saveModule(tmpFile.getAbsolutePath, overWrite = true)
    val reloadModel = Net.load[Float](tmpFile.getAbsolutePath)
    val inputData = Tensor[Float](2, 10).rand()
    ZooSpecHelper.compareOutputAndGradInput(
      model.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      reloadModel.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      inputData)

    testParameterSerialWithModel()
  }
}

class SequentialSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val model = Sequential[Float]()
    model.add(Dense[Float](8, inputShape = Shape(10)))
    val tmpFile = ZooSpecHelper.createTmpFile()
    model.saveModule(tmpFile.getAbsolutePath, overWrite = true)
    val reloadModel = Net.load[Float](tmpFile.getAbsolutePath)
    val inputData = Tensor[Float](2, 10).rand()
    ZooSpecHelper.compareOutputAndGradInput(
      model.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      reloadModel.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      inputData)
  }
}

class TopologySpec extends FlatSpec with Matchers with BeforeAndAfter {

  "Sequential to Model" should "work" in {
    val model = Sequential[Float]()
    model.add(Dense[Float](8, inputShape = Shape(10)))
    val newModel = model.toModel()
    val output = newModel.forward(Tensor[Float](Array(4, 10)).rand())
    output.toTensor[Float].size() should be (Array(4, 8))
  }

  "model.summary() for Model" should "work properly" in {
    val input = Input[Float](inputShape = Shape(10))
    val dense1 = Dense[Float](12).setName("dense1").inputs(input)
    val dense2 = Dense[Float](10).setName("dense2").inputs(dense1)
    val output = Activation[Float]("softmax").inputs(dense2)
    val model = Model(input, output)
    model.summary()
  }

  "model.summary() for Sequential" should "work properly" in {
    val model = Sequential[Float]()
    model.add(Embedding[Float](20000, 128, inputLength = 100).setName("embedding1"))
    model.add(Dropout[Float](0.25))
    model.add(Convolution1D[Float](nbFilter = 64, filterLength = 5, borderMode = "valid",
      activation = "relu", subsampleLength = 1).setName("conv1"))
    model.add(MaxPooling1D[Float](poolLength = 4))
    model.add(LSTM[Float](70))
    model.add(Dense[Float](1).setName("dense1"))
    model.add(Activation[Float]("sigmoid"))
    model.freeze("dense1")
    model.summary()
  }

  "model.summary() for nested Sequential" should "work properly" in {
    val model = Sequential[Float]()
    model.add(Convolution2D[Float](32, 3, 3, borderMode = "valid",
      inputShape = Shape(1, 28, 28)))
    model.add(Activation[Float]("relu"))
    model.add(Convolution2D[Float](32, 3, 3))
    model.add(Activation[Float]("relu"))
    model.add(MaxPooling2D[Float]())
    model.add(Dropout[Float](0.25))
    val seq1 = Sequential[Float]()
    seq1.add(Flatten[Float](inputShape = Shape(32, 12, 12)))
    seq1.add(Dense[Float](128).setName("dense1"))
    seq1.add(Dense[Float](32).setName("dense2"))
    model.add(seq1)
    val seq2 = Sequential[Float]()
    seq2.add(Activation[Float]("relu", inputShape = Shape(32)))
    seq2.add(Dropout[Float](0.5))
    seq2.add(Dense[Float](10))
    model.add(seq2)
    model.add(Activation[Float]("softmax"))
    model.freeze("dense1")
    model.summary()
  }

  "model.summary() for Model with merge" should "work properly" in {
    val input = Input[Float](inputShape = Shape(10))
    val dense1 = Dense[Float](12).setName("dense1").inputs(input)
    val dense2 = Dense[Float](10).setName("dense2").inputs(dense1)
    val branch1 = Activation[Float]("softmax").setName("softmax").inputs(dense2)
    val relu = Activation[Float]("relu").setName("relu").inputs(input)
    val branch2 = Dense[Float](10, activation = "softmax").setName("dense3").inputs(relu)
    val output = Merge.merge(List(branch1, branch2), mode = "concat")
    val model = Model(input, output)
    model.summary()
  }
}
