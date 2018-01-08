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
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class PoolingSpec extends FlatSpec with Matchers {

  "PoolingDnn with format=nchw" should "work correctly" in {
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

  "ConvolutionDnn time test" should "work correctly" in {
    val batchSize = 4

    val input = Tensor[Float](batchSize, 96, 55, 55).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, 96, 27, 27).apply1(e => Random.nextFloat())

    val layer = SpatialMaxPooling[Float](3, 3, 2, 2, 0, 0)
    // val layer = PoolingDnn[Float](3, 3, 2, 2, 0, 0)

    // for lenet
//    val batchSize = 8
//    val input = Tensor[Float](batchSize, 6, 24, 24).apply1(e => Random.nextFloat())
//    val gradOutput = Tensor[Float](batchSize, 6, 12, 12).apply1(e => Random.nextFloat())
//
//    RNG.setSeed(100)
//    val layer = SpatialMaxPooling[Float](2, 2, 2, 2)
//    val layer = PoolingDnn[Float](2, 2, 2, 2)


    // for inception
//    val batchSize = 4
//    val input = Tensor[Float](batchSize, 64, 112, 112).apply1(e => Random.nextFloat())
//    val gradOutput = Tensor[Float](batchSize, 64, 55, 55).apply1(e => Random.nextFloat())
//
//    RNG.setSeed(100)
//    val layer = SpatialMaxPooling[Float](3, 3, 2, 2)
//    // val layer = PoolingDnn[Float](3, 3, 2, 2)

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
    println(s"conv time ${end1/1e9} s")
    println("done")
  }

  def lenet(classNum: Int): Module[Float] = {
    val model = Sequential[Float]()
    model.add(Reshape(Array(1, 28, 28)))
      .add(SpatialConvolution[Float](1, 6, 5, 5).setName("conv1_5x5"))
      .add(Tanh[Float]())
      .add(SpatialMaxPooling[Float](2, 2, 2, 2))
      .add(Tanh[Float]())
      .add(SpatialConvolution[Float](6, 12, 5, 5).setName("conv2_5x5"))
      .add(SpatialMaxPooling[Float](2, 2, 2, 2))
      .add(Reshape[Float](Array(12 * 4 * 4)))
      .add(Linear[Float](12 * 4 * 4, 100).setName("fc1"))
      .add(Tanh[Float]())
      .add(Linear[Float](100, classNum).setName("fc2"))
      .add(LogSoftMax[Float]())
  }

  import com.intel.analytics.bigdl._
  import com.intel.analytics.bigdl.nn._
  import com.intel.analytics.bigdl.numeric.NumericFloat

  def lenetDnn(classNum: Int): Module[Float] = {
//    val model = Sequential[Float]()
//    model.add(Reshape(Array(1, 28, 28)))
//      .add(ConvolutionDnn(1, 6, 5, 5).setName("conv1_5x5"))
//      .add(MemoryReOrder())
//      .add(Tanh[Float]())
//      .add(SpatialMaxPooling[Float](2, 2, 2, 2))
//      .add(Tanh[Float]())
//      .add(ConvolutionDnn(6, 12, 5, 5).setName("conv2_5x5"))
//       .add(MemoryReOrder())
//      .add(SpatialMaxPooling[Float](2, 2, 2, 2))
//      .add(Reshape[Float](Array(12 * 4 * 4)))
//      .add(Linear[Float](12 * 4 * 4, 100).setName("fc1"))
//      .add(Tanh[Float]())
//      .add(Linear[Float](100, classNum).setName("fc2"))
//      .add(LogSoftMax[Float]())

    val input = Reshape(Array(1, 28, 28)).inputs()
    val conv1 = ConvolutionDnn(1, 6, 5, 5).setName("conv1_5x5").inputs(input)
    val reorder1 = MemoryReOrder().inputs(conv1)
    val tanh1 = Tanh().inputs(reorder1)
    val pool1 = SpatialMaxPooling(2, 2, 2, 2).inputs(tanh1)
    val tanh2 = Tanh().inputs(pool1)
    val conv2 = ConvolutionDnn(6, 12, 5, 5).setName("conv2_5x5").inputs(tanh2)
    val reorder2 = MemoryReOrder().inputs(conv2)
    val pool2 = SpatialMaxPooling(2, 2, 2, 2).inputs(reorder2)
    val reshape = Reshape(Array(12 * 4 * 4)).inputs(pool2)
    val fc1 = Linear(12 * 4 * 4, 100).setName("fc1").inputs(reshape)
    val tanh3 = Tanh().inputs(fc1)
    val fc2 = Linear(100, classNum).setName("fc2").inputs(tanh3)
    val output = LogSoftMax().inputs(fc2)

    Graph(input, output)
  }

  def getTopTimes(times: Array[(AbstractModule[_ <: Activity, _ <: Activity, Float],
    Long, Long)]): Unit = {
    var forwardSum = 0L
    var backwardSum = 0L
    times.foreach(x => {
      forwardSum += x._2
      backwardSum += x._3
    })
    println(s"forwardSum = ${forwardSum}", s"backwardSum = ${backwardSum}")

    val timeBuffer = new ArrayBuffer[(AbstractModule[_ <: Activity,
      _ <: Activity, Float], Long, Long, Long, Double)]
    var i = 0
    while (i < times.length) {
      val all = times(i)._2 + times(i)._3
      val rate = times(i)._3.toDouble/ times(i)._2
      timeBuffer.append((times(i)._1, times(i)._2, times(i)._3, all, rate))
      i += 1
    }
    val sortData = timeBuffer.sortBy(a => a._4)
    sortData.foreach(println)
  }

  "Lenet test" should "good" in {
    var t = 1
    val batchSize = 32
    var input = Tensor[Float](batchSize, 1, 28, 28).apply1(e => Random.nextFloat())
    var inputTemp = input.clone()
    var gradOutput = Tensor[Float](batchSize, 10).apply1(e => Random.nextFloat())
    var gradOutputTemp = gradOutput.clone()

    RNG.setSeed(100)
    val m1 = lenet(10)
    RNG.setSeed(100)
    val m2 = lenetDnn(10)

//    val output = m1.forward(input)
//    val grad1 = m1.backward(input, gradOutput)
//    val output2 = m2.forward(input)
//    val grad2 = m2.backward(input, gradOutput)
//    DnnUtils.nearequals(output.toTensor[Float], output2.toTensor[Float]) should be(true)

    var lastOutput : Tensor[Float] = null
    // warm up
    for (i <- 1 to 20) {
      println(t)
      input = Tensor[Float](batchSize, 1, 28, 28).apply1(e => Random.nextFloat())
      gradOutput = Tensor[Float](batchSize, 10).apply1(e => Random.nextFloat())
      val output = m1.forward(input).toTensor[Float]
      val grad1 = m1.updateGradInput(input, gradOutput).toTensor[Float]
      m1.accGradParameters(input, gradOutput)
      val p1 = m1.parameters()
      val (w1, b1) = (Module.flatten[Float](p1._1), Module.flatten[Float](p1._2))

      val output2 = m2.forward(input).toTensor[Float]
      val grad2 = m2.updateGradInput(input, gradOutput).toTensor[Float]
      m2.accGradParameters(input, gradOutput)
      val p2 = m2.parameters()
      val (w2, b2) = (Module.flatten[Float](p2._1), Module.flatten[Float](p2._2))


      DnnUtils.nearequals(w1, w2) should be(true)
      DnnUtils.getunequals(b1, b2) should be(true)

      if (lastOutput != null) {
        DnnUtils.nearequals(lastOutput, output2) should be(true)
        DnnUtils.nearequals(lastOutput, output) should be(true)
      }
      DnnUtils.nearequals(output, output2) should be(true)
      lastOutput = output
      // DnnUtils.getunequals(grad1, grad2) should be(true)

      // DnnUtils.nearequals(p1._1, p2._1) should be(true)
      // DnnUtils.nearequals(p1._2, p2._2) should be(true)

      t = t + 1
    }

    val s1 = System.nanoTime()
//    for (i <- 1 to 50) {
//      val output = m1.forward(input)
//      val grad1 = m1.backward(input, gradOutput)
//      val tmp = m1.getTimes()
//      getTopTimes(tmp)
//      m1.resetTimes()
//      println("111111111111111111111111111111")
//    }
    val end1 = System.nanoTime() - s1
    println(s"conv time ${end1/1e9} s")
    println("done")

  }

  "ConvolutionDnn time test 1111" should "work correctly" in {
    val batchSize = 32
    val input = Tensor[Float](batchSize, 1, 28, 28).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, 96, 24, 24).apply1(e => Random.nextFloat())

     val layer = new SpatialConvolution[Float](1, 96, 5, 5)
    // val layer = ConvTest(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    // val layer = ConvolutionDnn(1, 96, 5, 5)
    // warm up
    for (i <- 1 to 20) {
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
    println(s"conv time ${end1/1e9} s")
    println("done")
  }
}
