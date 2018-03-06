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

import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class SpatialConvolutionMapSpec extends FlatSpec with Matchers {

  "A SpatialConvolutionMap 3D input" should "generate correct output" in {
    val seed = 100
    RNG.setSeed(seed)

    val nInputPlane = 3
    val nOutputPlane = 16
    val kW = 5
    val kH = 5

    val layer1 = new SpatialConvolutionMap[Double](
      SpatialConvolutionMap.random[Double](nInputPlane, nOutputPlane, 1), kW, kH)
    val layer2 = layer1.cloneModule().asInstanceOf[SpatialConvolutionMap[Double]]

    Random.setSeed(seed)
    val input = Tensor[Double](3, 32, 32).apply1(e => Random.nextDouble())

    layer2.setScaleW(2.0).setScaleB(0.5)

    val output1 = layer1.forward(input)
    val output2 = layer2.forward(input)
    output1 should be (output2)

    val gradOutput = Tensor(output1)
    val gradInput1 = layer1.backward(input, gradOutput)
    val gradInput2 = layer2.backward(input, gradOutput)


    gradInput1 should be (gradInput2)

    layer2.gradWeight should be (layer1.gradWeight.mul(2))
    layer2.gradBias should be (layer1.gradBias.mul(0.5))
  }
}

class SpatialConvolutionMapSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val spatialConvolutionMap = SpatialConvolutionMap[Float](
      SpatialConvolutionMap.random(1, 1, 1), 2, 2).setName("spatialConvolutionMap")
    val input = Tensor[Float](1, 3, 3).apply1( e => Random.nextFloat())
    runSerializationTest(spatialConvolutionMap, input)
  }
}
