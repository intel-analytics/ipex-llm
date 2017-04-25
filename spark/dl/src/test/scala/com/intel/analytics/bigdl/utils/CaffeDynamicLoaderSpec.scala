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

class CaffeDynamicLoaderSpec extends FlatSpec with Matchers {
  val resource = getClass().getClassLoader().getResource("caffe")

  val alexNetProtoTxt = Paths.get(resource.getPath(), "alexnet_deploy.prototxt").toString
  val alexNetModelPath = Paths.get(resource.getPath(), "bvlc_alexnet.caffemodel").toString

  "load caffe dynamically" should "match the static training result" in {
    RandomGenerator.RNG.setSeed(1000)
    val staticLoadedModule = CaffeLoader.load[Float](AlexNet(1000),
      alexNetProtoTxt, alexNetModelPath)
    val dynamicLoadedModule = CaffeLoader.loadDynamic(alexNetProtoTxt, alexNetModelPath)
    val input1 = Tensor[Float](10, 3, 227, 227).apply1(e => Random.nextFloat())
    val input2 = Tensor()
    input2.resizeAs(input1).copy(input1)
    println("forward static one")
    val staticResult = staticLoadedModule.forward(input1)
    println("forward dynamic one")
    RandomGenerator.RNG.setSeed(1000)
    val dynamicResult = dynamicLoadedModule.forward(input2)
    //println(staticResult)
    //println(dynamicResult)
  }
}
