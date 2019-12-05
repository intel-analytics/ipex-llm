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

package com.intel.analytics.zoo.pipeline.api.net

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}


class GraphNetSpec extends FlatSpec with Matchers with BeforeAndAfter {

  "GraphNet " should "return correct parameters" in {
    val resource = getClass().getClassLoader().getResource("models")
    val path = resource.getPath + "/" + "bigdl"
    val model = Net.loadBigDL[Float](s"$path/bigdl_lenet.model")

    model.parameters()._1.length should be (8)
  }

  "GraphNet" should "return correct submodules" in {
    val resource = getClass().getClassLoader().getResource("models")
    val path = resource.getPath + "/" + "bigdl"
    val model = Net.loadBigDL[Float](s"$path/bigdl_lenet.model")

    model.getSubModules().length should be (12)
  }

}

class GraphNetSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val resource = getClass().getClassLoader().getResource("models")
    val path = resource.getPath + "/" + "bigdl"
    val model = Net.loadBigDL[Float](s"$path/bigdl_lenet.model")

    val tmpFile = ZooSpecHelper.createTmpFile()
    model.saveModule(tmpFile.getAbsolutePath, overWrite = true)
    val reloadModel = Net.loadBigDL[Float](tmpFile.getAbsolutePath)
    val inputData = Tensor[Float](2, 1, 28, 28).rand()
    ZooSpecHelper.compareOutputAndGradInput(
      model.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      reloadModel.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      inputData)
  }
}
