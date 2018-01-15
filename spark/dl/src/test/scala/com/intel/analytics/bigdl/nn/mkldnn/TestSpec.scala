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
import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class TestSpec extends FlatSpec with Matchers {

  def bigdlModel(classNum: Int): Module[Float] = {
    val model = Sequential[Float]()
      .add(SpatialConvolution[Float](3, 96, 11, 11, 4, 4, propagateBack = false))
      .add(ReLU[Float](false))
      .add(SpatialCrossMapLRN[Float](5, 0.0001, 0.75, 1.0))
      .add(SpatialMaxPooling[Float](3, 3, 2, 2, 0, 0))
    model
  }

  def dnnModel(classNum: Int): Module[Float] = {
    val model = Sequential[Float]()
      .add(ConvolutionDnn(3, 96, 11, 11, 4, 4, propagateBack = false))
      .add(ReLUDnn[Float](false))
      .add(LRNDnn[Float](5, 0.0001, 0.75, 1.0))
      .add(PoolingDnn[Float](3, 3, 2, 2, 0, 0))
    model
  }

  "MklDnn alexnet time test" should "work correctly" in {
    val batchSize = 4
    val input = Tensor[Float](batchSize, 3, 227, 227).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, 1000).apply1(e => Random.nextFloat())

    val layer = DnnUtils.dnnAlexNet(1000)// dnnAlexNet(1000)
    // warm up
    for (i <- 1 to 30) {
      val output = layer.forward(input)
      val grad1 = layer.backward(input, gradOutput)
    }

    val s1 = System.nanoTime()
    for (i <- 1 to 50) {
      val output = layer.forward(input)
      val grad1 = layer.backward(input, gradOutput)
      //      val tmp = layer.getTimes()
      //      DnnUtils.getTopTimes(tmp)
      //      layer.resetTimes()
      //      println("111111111111")
    }
    val end1 = System.nanoTime() - s1
    println(s"mkldnn model time ${end1/1e9} s")
    println("done")
  }

  "BigDL model time test" should "work correctly" in {
    val batchSize = 4
    val input = Tensor[Float](batchSize, 3, 227, 227).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, 96, 27, 27).apply1(e => Random.nextFloat())

    val layer = bigdlModel(10)
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
    println(s"bigdl model time ${end1/1e9} s")
    println("done")
  }

  "MklDnn model time test" should "work correctly" in {
    val batchSize = 4
    val input = Tensor[Float](batchSize, 3, 227, 227).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, 96, 27, 27).apply1(e => Random.nextFloat())

    val layer = dnnModel(10)
    // warm up
    for (i <- 1 to 30) {
      val output = layer.forward(input)
      val grad1 = layer.backward(input, gradOutput)
    }

    val s1 = System.nanoTime()
    for (i <- 1 to 50) {
      val output = layer.forward(input)
      val grad1 = layer.backward(input, gradOutput)
    }
    val end1 = System.nanoTime() - s1
    println(s"mkldnn model time ${end1/1e9} s")
    println("done")
  }
}
