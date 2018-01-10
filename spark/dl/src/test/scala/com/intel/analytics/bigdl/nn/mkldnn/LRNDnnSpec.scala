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
import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class LRNDnnSpec extends FlatSpec with Matchers {

  "LRNDnn with format=nchw" should "work correctly" in {
      val batchSize = 2
      val input = Tensor[Float](batchSize, 7, 3, 3).apply1(e => Random.nextFloat())
      val gradOutput = Tensor[Float](batchSize, 7, 3, 3).apply1(e => Random.nextFloat())

      RNG.setSeed(100)
      val conv = LRNDnn[Float](5, 0.0001, 0.75, 1.0)
      RNG.setSeed(100)
      val layer = SpatialCrossMapLRN[Float](5, 0.0001, 0.75, 1.0)

      val output2 = layer.forward(input)
      val grad2 = layer.updateGradInput(input, gradOutput)

      val output = conv.forward(input)
      val grad1 = conv.updateGradInput(input, gradOutput)

      DnnUtils.nearequals(output, output2) should be(true)
      DnnUtils.nearequals(grad1, grad2) should be(true)
  }


  "LRNDnn time test" should "work correctly" in {
    val batchSize = 4

    val input = Tensor[Float](batchSize, 96, 55, 55).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, 96, 55, 55).apply1(e => Random.nextFloat())

     // val layer = SpatialMaxPooling[Float](3, 3, 2, 2, 0, 0)
    val layer = LRNDnn[Float](5, 0.0001, 0.75, 1.0)

    // warm up
    for (i <- 1 to 30) {
      val output = layer.forward(input)
      val grad1 = layer.backward(input, gradOutput)
      val t = 1
    }

    val s1 = System.nanoTime()
    for (i <- 1 to 50) {
       val output = layer.forward(input)
      val grad1 = layer.backward(input, gradOutput)
    }
    val end1 = System.nanoTime() - s1
    println(s"lrn dnn time ${end1/1e9} s")
    println("done")
  }
}
