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

import java.nio.file.Paths

import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

/**
 * AlexNet and GoogleNet pre-trained models are large in size, please copy both prototxt
 * and binary model files to you local caffe directory under classpath for testing
 * For AlexNet please check https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
 * For GoogleNet please check https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
*/

class CaffeDynamicLoaderSpec extends FlatSpec with Matchers {

  val resource = getClass().getClassLoader().getResource("caffe")

  val alexNetProtoTxt = Paths.get(resource.getPath(), "alexnet_deploy.prototxt").toString
  val alexNetModelPath = Paths.get(resource.getPath(), "bvlc_alexnet.caffemodel").toString

  val googleNetProtoTxt = Paths.get(resource.getPath(), "googlenet_deploy.prototxt").toString
  val googleNetModelPath = Paths.get(resource.getPath(), "bvlc_googlenet.caffemodel").toString

  "Load caffe model dynamically" should "match the static module result" in {
    RandomGenerator.RNG.setSeed(1000)
    val staticLoadedModule = CaffeLoader.load[Float](AlexNet(1000),
      alexNetProtoTxt, alexNetModelPath)
    val input1 = Tensor[Float](10, 3, 227, 227).apply1(e => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    val staticResult = staticLoadedModule.forward(input1)
    RandomGenerator.RNG.setSeed(1000)
    val dynamicLoadedModule = CaffeLoader.loadDynamic(alexNetProtoTxt, alexNetModelPath)
    val dynamicResult = dynamicLoadedModule.forward(input2)

    val staticModuleDim = staticResult.toTensor.dim()
    val dynamicModuleDim = dynamicResult.toTensor.dim()

    assert(staticModuleDim == dynamicModuleDim)

    val staticModuleSize = staticResult.toTensor.size()
    val dynamicModuleSize = dynamicResult.toTensor.size()

    assert(staticModuleSize.length == dynamicModuleSize.length)

    for (i <- 1 to staticModuleSize.length) {
      assert(staticModuleSize(i - 1) == dynamicModuleSize(i - 1))
    }

    val staticNelements = staticResult.toTensor.nElement()
    val dynamicNelements = dynamicResult.toTensor.nElement()

    assert(staticNelements == dynamicNelements)

    val staticResultData = staticResult.toTensor.storage().toArray
    val dynamicResultData = dynamicResult.toTensor.storage().toArray

    for (i <- 1 to staticNelements) {
      assert(staticResultData(i - 1) == dynamicResultData(i - 1))
    }

  }

  "Dynamic Loading Cost" should "be inline with static module result" in {
    val input1 = Tensor[Float](10, 3, 227, 227).apply1(e => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    var staticOutput = Tensor[Float]()
    var staticLoadedModule = CaffeLoader.load[Float](AlexNet(1000),
      alexNetProtoTxt, alexNetModelPath)
    var staticStart = System.nanoTime()
    for (i <- 1 to 20) {
      staticOutput = staticLoadedModule.forward(input1).toTensor
    }
    var staticEnd = System.nanoTime()
    val staticAvgWithouLoading = (staticEnd - staticStart)/(1e6 * 20)
    var dynamicLoadedModule = CaffeLoader.loadDynamic[Float](alexNetProtoTxt, alexNetModelPath)
    var dynamicStart = System.nanoTime()
    var dynamicOutput = Tensor[Float]()
    for (i <- 1 to 20) {
      dynamicOutput = dynamicLoadedModule.forward(input2).toTensor
    }
    var dynamicEnd = System.nanoTime()
    val dynamicAvgWithoutLoading = (dynamicEnd - dynamicStart)/(1e6 * 20)

    staticStart = System.nanoTime()
    for (i <- 1 to 20) {
      RandomGenerator.RNG.setSeed(1000)
      staticLoadedModule = CaffeLoader.load[Float](AlexNet(1000),
        alexNetProtoTxt, alexNetModelPath)
      staticOutput = staticLoadedModule.forward(input1).toTensor
    }
    staticEnd = System.nanoTime()
    val staticAvgWithLoading = (staticEnd - staticStart)/(1e6 * 20)
    dynamicStart = System.nanoTime()
    for (i <- 1 to 20) {
      RandomGenerator.RNG.setSeed(1000)
      dynamicLoadedModule = CaffeLoader.loadDynamic[Float](alexNetProtoTxt, alexNetModelPath)
      dynamicOutput = dynamicLoadedModule.forward(input2).toTensor
    }
    dynamicEnd = System.nanoTime()
    val dynamicAvgWithLoading = (dynamicEnd - dynamicStart)/(1e6 * 20)
    println(s"static process without loading average cost per iteration is " +
      s"$staticAvgWithouLoading ms")
    println(s"dynamic process without loading average cost per iteration is " +
      s"$dynamicAvgWithoutLoading ms")
    println(s"static process with loading average cost per iteration is $staticAvgWithLoading ms")
    println(s"dynamic process with loading average cost per iteration is $dynamicAvgWithLoading ms")

    staticOutput should be (dynamicOutput)
  }

  "Load caffe inception model" should "work properly" in {
    val input = Tensor[Float](10, 3, 224, 224).apply1(e => Random.nextFloat())
    val dynamicLoadedModule = CaffeLoader.loadDynamic(googleNetProtoTxt, googleNetModelPath)
    val dynamicResult = dynamicLoadedModule.forward(input).asInstanceOf[Table]
    dynamicResult.length() should be (3)
  }
}
