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

package com.intel.analytics.sparkdl.optim

import com.intel.analytics.sparkdl.models.AlexNet
import com.intel.analytics.sparkdl.nn.Module
import com.intel.analytics.sparkdl.tensor.torch
import com.intel.analytics.sparkdl.utils.{T, Table}
import org.scalatest.{FlatSpec, Matchers}

class ModelPersistSpec extends FlatSpec with Matchers {
  "save model without iteration number" should "be correct" in {
    val filePath = java.io.File.createTempFile("ModelPersistSpec", ".model").getAbsolutePath
    val mp = new ModelPersistTest[Double]
    mp.setPath(filePath)
    val model = AlexNet[Double](1000)
    mp.saveModel(model)
    val loadedModel = torch.loadObj[Module[Double]](filePath)
    loadedModel should be(model)
  }

  "save model with iteration number" should "be correct" in {
    val filePath = java.io.File.createTempFile("ModelPersistSpec", ".model").getAbsolutePath
    val mp = new ModelPersistTest[Double]
    mp.setModelSaveInterval(10)
    mp.setPath(filePath)
    val model = AlexNet[Double](1000)
    mp.saveModel(model, 10, true)
    val loadedModel = torch.loadObj[Module[Double]](filePath + ".10")
    loadedModel should be(model)
  }

  "save state without iteration number" should "be correct" in {
    val filePath = java.io.File.createTempFile("ModelPersistSpec", ".model").getAbsolutePath
    val mp = new ModelPersistTest[Double]
    mp.setPath(filePath)
    val state = T("test" -> 123)
    mp.saveState(state)
    val loadedState = torch.loadObj[Table](filePath + ".state")
    loadedState should be(state)
  }

  "save state with iteration number" should "be correct" in {
    val filePath = java.io.File.createTempFile("ModelPersistSpec", ".model").getAbsolutePath
    val mp = new ModelPersistTest[Double]
    mp.setModelSaveInterval(10)
    mp.setPath(filePath)
    val state = T("test" -> 123)
    mp.saveState(state, 10, true)
    val loadedState = torch.loadObj[Table](filePath + ".state.10")
    loadedState should be(state)
  }
}

class ModelPersistTest[T] extends ModelPersist[T]
