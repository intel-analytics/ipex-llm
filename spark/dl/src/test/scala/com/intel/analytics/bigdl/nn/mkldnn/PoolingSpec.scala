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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class PoolingSpec extends FlatSpec with Matchers {

  "PoolingDnn with format=nchw" should "work correctly" in {
    // todo: no ceil mode
      val batchSize = 2
      val input = Tensor[Float](batchSize, 1, 4, 4).apply1(e => Random.nextFloat())
      val gradOutput = Tensor[Float](batchSize, 1, 3, 2).apply1(e => Random.nextFloat())

      RNG.setSeed(100)
      val conv = PoolingDnn[Float](3, 2, 2, 2, 1, 1)
      RNG.setSeed(100)
      val layer = SpatialMaxPooling[Float](3, 2, 2, 2, 1, 1)

      val output2 = layer.forward(input)
      val grad2 = layer.updateGradInput(input, gradOutput)

      val output = conv.forward(input)
      val grad1 = conv.updateGradInput(input, gradOutput)

      DnnUtils.nearequals(output, output2) should be(true)
      DnnUtils.nearequals(grad1, grad2) should be(true)
  }


  "PoolingDnn with format=nchw 1111" should "work correctly" in {
    val batchSize = 2
    val input = Tensor[Float](batchSize, 96, 55, 55).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](2, 96, 27, 27).apply1(e => Random.nextFloat())

    RNG.setSeed(100)
    val conv = PoolingDnn[Float](3, 3, 2, 2, 0, 0)
    RNG.setSeed(100)
    val layer = new SpatialMaxPooling[Float](3, 3, 2, 2, 0, 0)

    val output2 = layer.forward(input)
    val grad2 = layer.updateGradInput(input, gradOutput)

    val output = conv.forward(input)
    val grad1 = conv.updateGradInput(input, gradOutput)

    DnnUtils.nearequals(output, output2) should be(true)
    DnnUtils.nearequals(grad1, grad2) should be(true)
  }


  "PoolingDnn with ceil test1" should "work correctly" in {
    val batchSize = 2
    val input = Tensor[Float](batchSize, 480, 28, 28).apply1(e => Random.nextFloat())

    RNG.setSeed(100)
    val conv = PoolingDnn[Float](3, 3, 2, 2).ceil()
    RNG.setSeed(100)
    val layer = SpatialMaxPooling[Float](3, 3, 2, 2).ceil()

    val output2 = layer.forward(input).toTensor[Float]
    val output1 = conv.forward(input)
    output1.storage()
    DnnUtils.nearequals(output1, output2) should be(true)

    val grad2 = layer.updateGradInput(input, output2).toTensor[Float]
    val grad1 = conv.updateGradInput(input, output2)
    grad1.storage()
    DnnUtils.nearequals(grad1, grad2) should be(true)

    println("done")
  }

  "PoolingDnn with ceil test2" should "work correctly" in {
    val batchSize = 2
    val input = Tensor[Float](batchSize, 64, 112, 112).apply1(e => Random.nextFloat())

    RNG.setSeed(100)
    val conv = PoolingDnn[Float](3, 3, 2, 2).ceil()
    RNG.setSeed(100)
    val layer = SpatialMaxPooling[Float](3, 3, 2, 2).ceil()

    val output2 = layer.forward(input).toTensor[Float]
    val output1 = conv.forward(input)
    output1.storage()
    DnnUtils.nearequals(output1, output2) should be(true)

    val grad2 = layer.updateGradInput(input, output2).toTensor[Float]
    val grad1 = conv.updateGradInput(input, output2)
    grad1.storage()
    DnnUtils.nearequals(grad1, grad2) should be(true)

    println("done")
  }
}
