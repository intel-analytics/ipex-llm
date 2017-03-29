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
package com.intel.analytics.bigdl.integration

import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.utils.File
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}


@com.intel.analytics.bigdl.tags.Integration
class HdfsSpec extends FlatSpec with Matchers with BeforeAndAfter{

    val hdfs = System.getProperty("hdfsMaster")
    val mnistFolder = System.getProperty("mnist")

  private def processPath(path: String): String = {
    if (path.contains(":")) {
      path.substring(1)
    } else {
      path
    }
  }

  "save and load model from hdfs" should "be correct" in {
    val model = LeNet5(10)
    val hdfsPath = hdfs + "/lenet.obj"
    File.saveToHdfs(model, hdfsPath, true)
    val hdfsModel = Module.load(hdfsPath)

    val localPath = java.io.File.createTempFile("lenet", ".obj").getAbsolutePath
    File.save(model, localPath, true)
    val localModel = Module.load(localPath)

    hdfsModel should be (model)
    hdfsModel should be (localModel)
  }

  "load minist from hdfs" should "be correct" in {
    val folder = mnistFolder + "/t10k-images-idx3-ubyte"
    val resource = getClass().getClassLoader().getResource("mnist")

    val hdfsData = File.readHdfsByte(folder)
    val localData = Files.readAllBytes(
      Paths.get(processPath(resource.getPath()), "/t10k-images.idx3-ubyte"))

    hdfsData should be (localData)
  }
}
