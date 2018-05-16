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
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Dense, Input}
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class ModelSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val input = Input[Float](inputShape = Shape(10))
    val model = Model(input, Dense[Float](8).inputs(input))
    val tmpFile = ZooSpecHelper.createTmpFile()
    model.saveModule(tmpFile.getAbsolutePath, overWrite = true)
    val reloadModel = Net.load[Float](tmpFile.getAbsolutePath)
    val inputData = Tensor[Float](2, 10).apply1(_ => Random.nextFloat())
    ZooSpecHelper.compareOutputAndGradInput(
      model.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      reloadModel.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      inputData)
  }
}

class SequentialSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val model = Sequential[Float]()
    model.add(Dense[Float](8, inputShape = Shape(10)))
    val tmpFile = ZooSpecHelper.createTmpFile()
    model.saveModule(tmpFile.getAbsolutePath, overWrite = true)
    val reloadModel = Net.load[Float](tmpFile.getAbsolutePath)
    val inputData = Tensor[Float](2, 10).apply1(_ => Random.nextFloat())
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
    model.toModel()
    val output = model.forward(Tensor[Float](Array(4, 10)).rand())
    output.toTensor[Float].size() should be (Array(4, 8))
  }
}
