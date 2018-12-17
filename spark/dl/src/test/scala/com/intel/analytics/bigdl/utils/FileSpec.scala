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
import com.intel.analytics.bigdl.example.loadmodel.AlexNet_OWT
import com.intel.analytics.bigdl.nn._
import org.scalatest.{FlatSpec, Matchers}


@com.intel.analytics.bigdl.tags.Serial
class FileSpec extends FlatSpec with Matchers {

  "read/write alexnet" should "work properly" in {
    val tmpFile = java.io.File.createTempFile("module", ".t7")
    val absolutePath = tmpFile.getAbsolutePath
    val alexnet = AlexNet_OWT(1000)
    alexnet.saveTorch(absolutePath, true)
    val model = Module.loadTorch[Float](absolutePath).asInstanceOf[Sequential[Float]]
    model.getParameters() should be (alexnet.getParameters())
    for (i <- 0 until model.modules.size) {
      println(s"check the $i th layer in the model...")
      // torch will discard the name
      model.modules(i).setName(alexnet.asInstanceOf[Sequential[Float]].modules(i).getName())
      model.modules(i) should be (alexnet.asInstanceOf[Sequential[Float]].modules(i))
    }
    model should be (alexnet)
  }

  "save/load Java object file" should "work properly" in {
    val tmpFile = java.io.File.createTempFile("module", "obj")
    val absolutePath = tmpFile.getAbsolutePath


    val module = new Sequential[Double]

    module.add(new SpatialConvolution(1, 6, 5, 5))
    module.add(new Tanh())
    module.add(new SpatialMaxPooling(2, 2, 2, 2))
    // stage 2 : filter bank -> squashing -> max pooling
    module.add(new SpatialConvolution(6, 12, 5, 5))
    module.add(new Tanh())
    module.add(new SpatialMaxPooling(2, 2, 2, 2))
    // stage 3 : standard 2-layer neural network
    module.add(new Reshape(Array(12 * 5 * 5)))
    module.add(new Linear(12 * 5 * 5, 100))
    module.add(new Tanh())
    module.add(new Linear(100, 6))
    module.add(new LogSoftMax[Double]())

    File.save(module, absolutePath, true)
    val testModule: Module[Double] = File.load(absolutePath)

    testModule should be(module)
  }


  "save/load big size model" should "work properly" in {
    val tmpFile = java.io.File.createTempFile("module", "obj")
    val absolutePath = tmpFile.getAbsolutePath


    val module = Linear[Float](40000, 8000)

    File.save(module, absolutePath, true)
    val testModule: Module[Double] = File.load(absolutePath)

    testModule should be(module)

    if (tmpFile.exists()) {
      tmpFile.delete()
    }
  }

}
