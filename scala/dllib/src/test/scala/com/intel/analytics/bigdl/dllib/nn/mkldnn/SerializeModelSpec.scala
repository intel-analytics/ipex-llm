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

package com.intel.analytics.bigdl.nn.mkldnn

import java.io.File
import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.nn.mkldnn.ResNet.DatasetType.ImageNet
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

class SerializeModelSpec extends FlatSpec with Matchers {

  "Save a model" should "work correctly" in {
    val identity = System.identityHashCode(this).toString
    val name = "resnet_50." + identity
    val tmpdir = System.getProperty("java.io.tmpdir")
    val path = Paths.get(tmpdir, name).toAbsolutePath

    // do not use vgg16 model, the vgg16 model will set Xavier to average
    // mode, which will influence other test cases because of Xavier is a
    // case object.
    val model = ResNet(32, 1000, T("depth" -> 50, "dataSet" -> ImageNet))
    println(s"generate the model file ${path.toString}")
    model.save(path.toString, true)
    val loaded = Module.load[Float](path.toString)

    val length = Files.size(path) / 1024.0 / 1024.0
    length should be < 300.0

    println(s"delete the model file ${path.toString}")
    Files.deleteIfExists(path)
  }

}
