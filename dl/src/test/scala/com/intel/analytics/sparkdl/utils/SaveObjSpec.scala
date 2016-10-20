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

package com.intel.analytics.sparkdl.utils

import com.intel.analytics.sparkdl.models.imagenet.{AlexNet, GoogleNet_v1}
import com.intel.analytics.sparkdl.nn.Module
import com.intel.analytics.sparkdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

class SaveObjSpec extends FlatSpec with Matchers {
  "A tensor load from saved file" should "be same with original tensor" in {
    val originTensor = Tensor[Double](3, 2, 4).rand()
    val filePath = java.io.File.createTempFile("SaveObjSpecTensor", ".obj").getAbsolutePath
    File.save(originTensor, filePath, true)
    val loadedTensor = File.loadObj[Tensor[Double]](filePath)
    loadedTensor should be(originTensor)
  }

  "Alexnet load from saved file" should "be same with the original one" in {
    val model = AlexNet[Double](1000)
    val filePath = java.io.File.createTempFile("SaveObjSpecAlexnet", ".obj").getAbsolutePath
    model.forward(Tensor[Double](4, 3, 227, 227))
    File.save(model, filePath, true)
    val loadedModel = File.loadObj[Module[Double]](filePath)
    loadedModel should be(model)
    loadedModel.forward(Tensor[Double](4, 3, 227, 227))
  }

  "GoogleNet load from saved file" should "be same with the original one" in {
    val model = GoogleNet_v1[Double](1000)
    val filePath = java.io.File.createTempFile("SaveObjSpecGoogleNet", ".obj").getAbsolutePath
    model.forward(Tensor[Double](4, 3, 224, 224))
    File.save(model, filePath, true)
    val loadedModel = File.loadObj[Module[Double]](filePath)
    loadedModel should be(model)
    loadedModel.forward(Tensor[Double](4, 3, 224, 224))
  }

  "A table load from saved file" should "be same with original table" in {
    val table = T("test" -> "test2", "test3" -> 4)
    val filePath = java.io.File.createTempFile("SaveObjSpecTable", ".obj").getAbsolutePath
    File.save(table, filePath, true)
    val loadedTable = File.loadObj[Table](filePath)
    loadedTable should be(table)
  }
}
