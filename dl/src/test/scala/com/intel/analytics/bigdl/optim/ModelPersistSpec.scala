/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.models.imagenet.{AlexNet, AlexNet_OWT}
import com.intel.analytics.bigdl.nn.abstractnn.Module
import com.intel.analytics.bigdl.nn.{Linear, Sequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{File, T, Table}
import org.scalatest.{FlatSpec, Matchers}

class ModelPersistSpec extends FlatSpec with Matchers {
  "save model without iteration number for small model" should "be correct" in {
    val filePath = java.io.File.createTempFile("ModelPersistSpec", ".model").getAbsolutePath
    val mp = new ModelPersistTest[Float]
    mp.setPath(filePath)
    val model = new Sequential[Float]()
    model.add(new Linear[Float](3, 10))
    model.add(new Linear[Float](10, 5))

    val input = Tensor[Float](10, 3).rand()
    val grad = Tensor[Float](10, 5).rand()
    val (weight1, gradweight1) = model.getParameters()
    model.forward(input)
    model.backward(input, grad)
    model.clearState()
    mp.saveModel(model)
    val loadedModel = Module.load[Tensor[Float], Tensor[Float], Float](filePath)
    loadedModel.forward(input)
    loadedModel.backward(input, grad)
    val (weight2, gradweight2) = loadedModel.getParameters()
    weight2 should be(weight1)
    model.evaluate()
    loadedModel.evaluate()
    val output1 = model.forward(input)
    val output2 = loadedModel.forward(input)
    output1 should be(output2)
    model.clearState()
    loadedModel.clearState()
    loadedModel.zeroGradParameters()
    model.zeroGradParameters()
    loadedModel should be(model)
  }

  "save model without iteration number" should "be correct" in {
    val filePath = java.io.File.createTempFile("ModelPersistSpec", ".model").getAbsolutePath
    val mp = new ModelPersistTest[Float]
    mp.setPath(filePath)
    val model = AlexNet_OWT(1000, false, true)

    val input = Tensor[Float](4, 3, 227, 227).rand()
    val grad = Tensor[Float](4, 1000).rand()
    val (weight1, gradweight1) = model.getParameters()
    model.forward(input)
    model.backward(input, grad)
    model.clearState()
    mp.saveModel(model)
    val loadedModel = Module.load[Tensor[Float], Tensor[Float], Float](filePath)
    loadedModel.forward(input)
    loadedModel.backward(input, grad)
    val (weight2, gradweight2) = loadedModel.getParameters()
    weight2 should be(weight1)
    model.evaluate()
    loadedModel.evaluate()
    val output1 = model.forward(input)
    val output2 = loadedModel.forward(input)
    output1 should be(output2)
    loadedModel.clearState()
    model.clearState()
    model.zeroGradParameters()
    loadedModel.zeroGradParameters()
    loadedModel should be(model)
  }

  "save model with iteration number for small model" should "be correct" in {
    val filePath = java.io.File.createTempFile("ModelPersistSpec", ".model").getAbsolutePath
    val mp = new ModelPersistTest[Float]
    mp.setModelSaveInterval(10)
    mp.setPath(filePath)
    val model = new Sequential[Float]()
    model.add(new Linear[Float](3, 10))
    model.add(new Linear[Float](10, 5))
    mp.saveModel(model, 10, true)
    val loadedModel = Module.load[Tensor[Float], Tensor[Float], Float](filePath + ".10")
    loadedModel should be(model)
  }

  "save model with iteration number" should "be correct" in {
    val filePath = java.io.File.createTempFile("ModelPersistSpec", ".model").getAbsolutePath
    val mp = new ModelPersistTest[Float]
    mp.setModelSaveInterval(10)
    mp.setPath(filePath)
    val model = AlexNet(1000)
    mp.saveModel(model, 10, true)
    val loadedModel = Module.load[Tensor[Float], Tensor[Float], Float](filePath + ".10")
    loadedModel should be(model)
  }

  "save state without iteration number" should "be correct" in {
    val filePath = java.io.File.createTempFile("ModelPersistSpec", ".model").getAbsolutePath
    val mp = new ModelPersistTest[Float]
    mp.setPath(filePath)
    val state = T("test" -> 123)
    mp.saveState(state)
    val loadedState = File.load[Table](filePath + ".state")
    loadedState should be(state)
  }

  "save state with iteration number" should "be correct" in {
    val filePath = java.io.File.createTempFile("ModelPersistSpec", ".model").getAbsolutePath
    val mp = new ModelPersistTest[Float]
    mp.setModelSaveInterval(10)
    mp.setPath(filePath)
    val state = T("test" -> 123)
    mp.saveState(state, 10, true)
    val loadedState = File.load[Table](filePath + ".state.10")
    loadedState should be(state)
  }
}

class ModelPersistTest[T] extends ModelPersist[T]
