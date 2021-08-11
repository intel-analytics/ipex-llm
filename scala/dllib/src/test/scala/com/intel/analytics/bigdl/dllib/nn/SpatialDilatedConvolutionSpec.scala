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
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import com.intel.analytics.bigdl.utils.{Shape, TestUtils}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class SpatialDilatedConvolutionSpec extends FlatSpec with Matchers {

  "A SpatialDilatedConvolution" should "generate correct output and grad" in {
    val seed = 100
    RNG.setSeed(seed)
    val nInputPlane = 3
    val nOutputPlane = 6
    val kW = 3
    val kH = 3
    val dW = 1
    val dH = 1
    val padW = 2
    val padH = 2

    val layer1 = new SpatialDilatedConvolution[Double](nInputPlane, nOutputPlane,
      kW, kH, dW, dH, padW, padH)
    val layer2 = layer1.cloneModule().asInstanceOf[SpatialDilatedConvolution[Double]]
    layer2.setScaleW(2).setScaleB(0.5)

    val input = Tensor[Double](3, 3, 6, 6).apply1(e => Random.nextDouble())

    val output1 = layer1.forward(input)
    val output2 = layer2.forward(input)
    output1 should be (output2)

    val gradOutput = Tensor[Double]().resizeAs(output1).apply1(e => Random.nextDouble())
    val gradInput1 = layer1.backward(input, gradOutput)
    val gradInput2 = layer2.backward(input, gradOutput)
    gradInput1 should be (gradInput2)

    layer2.gradWeight should be (layer1.gradWeight.mul(2))
    layer2.gradBias should be (layer1.gradBias.mul(0.5))

  }

  "SpatialDilatedConvolution computeOutputShape" should "work properly" in {
    val layer = SpatialDilatedConvolution[Float](4, 8, 2, 3, 1, 2, 0, 0, 2, 2)
    TestUtils.compareOutputShape(layer, Shape(4, 256, 256)) should be (true)
  }

}

class SpatialDilatedConvolutionSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val spatialDilatedConvolution = SpatialDilatedConvolution[Float](1, 1,
      2, 2, 1, 1, 0, 0).setName("spatialDilatedConvolution")
    val input = Tensor[Float](1, 3, 3).apply1( e => Random.nextFloat())
    runSerializationTest(spatialDilatedConvolution, input)
  }
}
