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

package com.intel.analytics.bigdl.dllib.keras

import com.intel.analytics.bigdl.dllib.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.autograd.{AutoGrad, Parameter, Variable}
import com.intel.analytics.bigdl.dllib.keras.ZooSpecHelper
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.serializer.ModuleSerializationTest
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

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
