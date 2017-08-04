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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl._
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericFloat

class AbstractModuleSpec extends FlatSpec with Matchers {
  "Get name" should "find the module if it exists" in {
    val m = Linear(4, 3).setName("module")
    m("module").get should be(m)
  }

  "Get name" should "find the module if it exists in container" in {
    val m = Linear(4, 3).setName("module")
    val s = Sequential()
    s.add(m)

    s("module").get should be(m)
  }

  "Get name" should "find the module if it exists in deeper container" in {
    val m = Linear(4, 3).setName("module")
    val s = Sequential()
    s.add(m)
    val k = Sequential()
    k.add(s)

    k("module").get should be(m)
  }

  "Get name" should "get the container if it is the container" in {
    val m = Linear(4, 3).setName("module")
    val s = Sequential()
    s.setName("container")
    s.add(m)

    s("container").get should be(s)
  }

  "Get name" should "not find if there is no such module" in {
    val m = Linear(4, 3)
    m("module") should be(None)
    val s = Sequential()
    s.add(m)
    s("container") should be(None)
  }

  "Get name" should "throw exception if there are two modules with same name" in {
    val m1 = Linear(4, 3)
    val m2 = Linear(4, 3)
    m1.setName("module")
    m2.setName("module")
    val s = Sequential()
    s.add(m1).add(m2)

    intercept[IllegalArgumentException] {
      s("module").get
    }
  }

  "weights save and load" should "work properly" in {
    val tmpFile = java.io.File.createTempFile("module", "obj")
    val absolutePath = tmpFile.getAbsolutePath
    val module = Sequential()

    module.add(SpatialConvolution(1, 6, 5, 5).setName("conv1"))
    module.add(Tanh())
    module.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 2 : filter bank -> squashing -> max pooling
    module.add(SpatialConvolution(6, 12, 5, 5).setName("conv2"))
    module.add(Tanh())
    module.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 3 : standard 2-layer neural network
    module.add(Reshape(Array(12 * 5 * 5)))
    module.add(Linear(12 * 5 * 5, 100).setName("l1"))
    module.add(Tanh())
    module.add(Linear(100, 6).setName("l2"))
    module.add(LogSoftMax())

    val module2 = Sequential()

    module2.add(SpatialConvolution(1, 6, 5, 5).setName("conv1"))
    module2.add(Tanh())
    module2.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 2 : filter bank -> squashing -> max pooling
    module2.add(SpatialConvolution(6, 12, 5, 5).setName("conv2"))
    module2.add(Tanh())
    module2.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 3 : standard 2-layer neural network
    module2.add(Reshape(Array(12 * 5 * 5)))
    module2.add(Linear(12 * 5 * 5, 100).setName("l1"))
    module2.add(Tanh())
    module2.add(Linear(100, 6).setName("l2"))
    module2.add(LogSoftMax())

    module.saveWeights(absolutePath, true)

    module2.loadWeights(absolutePath)

    module.parameters()._1 should be(module2.parameters()._1)
  }

  "weights save and load with different model definition" should "work properly" in {
    val tmpFile = java.io.File.createTempFile("module", "obj")
    val absolutePath = tmpFile.getAbsolutePath
    val module = Sequential()

    module.add(SpatialConvolution(1, 6, 5, 5).setName("conv1"))
    module.add(Tanh())
    module.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 2 : filter bank -> squashing -> max pooling
    module.add(SpatialConvolution(6, 12, 5, 5).setName("conv2"))
    module.add(Tanh())
    module.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 3 : standard 2-layer neural network
    module.add(Reshape(Array(12 * 5 * 5)))
    module.add(Linear(12 * 5 * 5, 100).setName("l1"))
    module.add(Tanh())
    module.add(Linear(100, 6).setName("l2"))
    module.add(LogSoftMax())

    val module2 = Sequential()

    module2.add(SpatialConvolution(1, 6, 5, 5).setName("conv1"))
    module2.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 2 : filter bank -> squashing -> max pooling
    module2.add(SpatialConvolution(6, 12, 5, 5).setName("conv2"))
    module2.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 3 : standard 2-layer neural network
    module2.add(Reshape(Array(12 * 5 * 5)))
    module2.add(Linear(12 * 5 * 5, 100).setName("l1"))
    module2.add(Linear(100, 6).setName("l2"))

    module.saveWeights(absolutePath, true)

    module2.loadWeights(absolutePath)

    module.parameters()._1 should be(module2.parameters()._1)
  }

  "weights save and load with only weight or bias" should "work properly" in {
    val tmpFile = java.io.File.createTempFile("module", "obj")
    val absolutePath = tmpFile.getAbsolutePath
    val module = Sequential()

    module.add(CMul(Array(1, 4, 1, 1)).setName("cmul"))
    module.add(CAdd(Array(1, 4, 1, 1)).setName("cadd"))

    val module2 = Sequential()

    module2.add(CMul(Array(1, 4, 1, 1)).setName("cmul"))
    module2.add(CAdd(Array(1, 4, 1, 1)).setName("cadd"))

    module.saveWeights(absolutePath, true)

    module2.loadWeights(absolutePath)

    module.parameters()._1 should be(module2.parameters()._1)
  }

  "loadModelWeights" should "work properly" in {
    val module = Sequential()

    module.add(SpatialConvolution(1, 6, 5, 5).setName("conv1"))
    module.add(Tanh())
    module.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 2 : filter bank -> squashing -> max pooling
    module.add(SpatialConvolution(6, 12, 5, 5).setName("conv2"))
    module.add(Tanh())
    module.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 3 : standard 2-layer neural network
    module.add(Reshape(Array(12 * 5 * 5)))
    module.add(Linear(12 * 5 * 5, 100).setName("l1"))
    module.add(Tanh())
    module.add(Linear(100, 6).setName("l2"))
    module.add(LogSoftMax())

    val module2 = Sequential()

    module2.add(SpatialConvolution(1, 6, 5, 5).setName("conv1"))
    module2.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 2 : filter bank -> squashing -> max pooling
    module2.add(SpatialConvolution(6, 12, 5, 5).setName("conv2"))
    module2.add(SpatialMaxPooling(2, 2, 2, 2))
    // stage 3 : standard 2-layer neural network
    module2.add(Reshape(Array(12 * 5 * 5)))
    module2.add(Linear(12 * 5 * 5, 100).setName("l1"))
    module2.add(Linear(100, 6).setName("l2"))
    module.loadModelWeights(module2)

    module.parameters()._1 should be(module2.parameters()._1)
  }
}
