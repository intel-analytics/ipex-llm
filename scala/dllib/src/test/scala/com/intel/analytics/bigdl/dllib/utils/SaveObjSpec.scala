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

package com.intel.analytics.bigdl.utils

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.models.inception.Inception_v1
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class SaveObjSpec extends FlatSpec with Matchers {
  "A tensor load from saved file" should "be same with original tensor" in {
    val originTensor = Tensor[Float](3, 2, 4).rand()
    val filePath = java.io.File.createTempFile("SaveObjSpecTensor", ".obj").getAbsolutePath
    File.save(originTensor, filePath, true)
    val loadedTensor = File.load[Tensor[Float]](filePath)
    loadedTensor should be(originTensor)
  }

  "Alexnet load from saved file" should "be same with the original one" in {
    val model = AlexNet(1000)
    val filePath = java.io.File.createTempFile("SaveObjSpecAlexnet", ".obj").getAbsolutePath
    model.forward(Tensor[Float](4, 3, 227, 227))
    File.save(model, filePath, true)
    val loadedModel = File.load[Module[Float]](filePath)
    loadedModel should be(model)
    loadedModel.forward(Tensor[Float](4, 3, 227, 227))
  }

  "Inception load from saved file" should "be same with the original one" in {
    val model = Inception_v1(1000)

    val filePath = java.io.File.createTempFile("SaveObjSpecInception", ".obj").getAbsolutePath
    model.forward(Tensor[Float](4, 3, 224, 224))
    File.save(model, filePath, true)
    val loadedModel = File.load[Module[Float]](filePath)
    loadedModel should be(model)
    loadedModel.forward(Tensor[Float](4, 3, 224, 224))
  }

  "A table load from saved file" should "be same with original table" in {
    val table = T("test" -> "test2", "test3" -> 4)
    val filePath = java.io.File.createTempFile("SaveObjSpecTable", ".obj").getAbsolutePath
    File.save(table, filePath, true)
    val loadedTable = File.load[Table](filePath)
    loadedTable should be(table)
  }
}
