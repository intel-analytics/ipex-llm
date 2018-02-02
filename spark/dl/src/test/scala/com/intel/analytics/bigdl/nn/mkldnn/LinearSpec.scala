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

import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.mkl.MKL
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.nn.mkldnn.Utils.{manyTimes, speedup}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Optimizer, Trigger}
import com.intel.analytics.bigdl.utils.Engine
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.concurrent.Future

class LinearSpec extends FlatSpec with Matchers {
  "linear updateOutput" should "work correctly" in {
    System.setProperty("bigdl.localMode", "true")
    Engine.init
    var tasks: ArrayBuffer[Future[_]] = new ArrayBuffer()
    tasks += Engine.default.invoke ( () => {
      val inputSize = 2
      val outputSize = 2
      val batchSize = 2

      val initWeight = Tensor[Float](outputSize, inputSize).rand()
      val initBias = Tensor[Float](outputSize).rand()

      val linear = Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
      val input = Tensor[Float](batchSize, inputSize).rand()

      val output = linear.forward(input)
      println(output)

      val nnLinear = nn.Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
      val nnOutput = nnLinear.forward(input)
      println(nnOutput)

      output should be (nnOutput)
      1
    })

    Engine.default.sync(tasks)
  }

  "linear updateOutput multi times" should "work correctly" in {
    val inputSize = 2
    val outputSize = 2
    val batchSize = 2

    val initWeight = Tensor[Float](outputSize, inputSize).rand()
    val initBias = Tensor[Float](outputSize).rand()

    val inputs = new Array[Tensor[Float]](100)
    for (i <- inputs.indices) {
      inputs(i) = Tensor[Float](batchSize, inputSize).rand()
    }

    val linear = Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)

    for (in <- inputs) {
      linear.forward(in)
    }
    println(linear.output)

    val nnLinear = nn.Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    for (in <- inputs) {
      nnLinear.forward(in)
    }
    println(nnLinear.output)

    linear.output should be (nnLinear.output)
  }

  "linear updateOutput" should "work much faster than blas" in {
    val inputSize = 64
    val outputSize = 64
    val batchSize = 32

    val initWeight = Tensor[Float](outputSize, inputSize).rand()
    val initBias = Tensor[Float](outputSize).rand()

    val input = Tensor[Float](batchSize, inputSize)

    val linear = Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)

    val warm = 10
    val iters = 100
    manyTimes[Tensor[Float]] {
      linear.forward(input)
    }(warm)

    val (costs, _) = manyTimes[Tensor[Float]] {
      linear.forward(input)
    }(iters)

    println(linear.output)

    val nnLinear = nn.Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    manyTimes[Tensor[Float]] {
      nnLinear.forward(input)
    }(warm)

    val (nnCosts, _) = manyTimes[Tensor[Float]] {
      nnLinear.forward(input)
    }(iters)
    println(nnLinear.output)

    println(costs)
    println(nnCosts)
    println(speedup(nnCosts, costs))

    linear.output should be (nnLinear.output)
    costs should be < nnCosts
  }

  "linear updateGradInput" should "work correctly" in {
    val inputSize = 2
    val outputSize = 2
    val batchSize = 2

    val initWeight = Tensor[Float](outputSize, inputSize).rand()
    val initBias = Tensor[Float](outputSize).rand()

    val linear = Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    val input = Tensor[Float](batchSize, inputSize).rand()
    val output = linear.forward(input)

    val gradOutput = Tensor[Float]().resizeAs(output).rand()
    val gradInput = linear.updateGradInput(input, gradOutput)

    val nnLinear = nn.Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    val nnOutput = nnLinear.forward(input)
    val nnGradInput = nnLinear.updateGradInput(input, gradOutput)

    println(gradInput)
    println("-" * 80)
    println(nnGradInput)

    gradInput should be (nnGradInput)
  }

  "linear updateGradInput multi times" should "work correctly" in {
    val inputSize = 2
    val outputSize = 2
    val batchSize = 2

    val initWeight = Tensor[Float](outputSize, inputSize).rand()
    val initBias = Tensor[Float](outputSize).rand()

    val linear = Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)

    val length = 100
    val inputs = new Array[Tensor[Float]](length)
    for (i <- inputs.indices) {
      inputs(i) = Tensor[Float](batchSize, inputSize).rand()
    }

    val gradOutputs = new Array[Tensor[Float]](length)
    for (i <- gradOutputs.indices) {
      gradOutputs(i) = Tensor[Float](batchSize, outputSize).rand()
    }

    linear.forward(inputs.last)

    for (i <- inputs.indices) {
      linear.updateGradInput(inputs(i), gradOutputs(i))
    }

    val nnLinear = nn.Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    val nnOutput = nnLinear.forward(inputs.last)

    for (i <- inputs.indices) {
      nnLinear.updateGradInput(inputs(i), gradOutputs(i))
    }

    linear.gradInput should be (nnLinear.gradInput)
  }

  "linear updateGradInput" should "work much faster than blas" in {
    val inputSize = 64
    val outputSize = 64
    val batchSize = 32

    val initWeight = Tensor[Float](outputSize, inputSize).rand()
    val initBias = Tensor[Float](outputSize).rand()

    val linear = Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)

    val warm = 10
    val iters = 100

    val input = Tensor[Float](batchSize, inputSize)
    val gradOutput = Tensor[Float](batchSize, outputSize)

    linear.forward(input)

    manyTimes[Tensor[Float]]{
      linear.updateGradInput(input, gradOutput)
    }(warm)

    val (costs, _) = manyTimes[Tensor[Float]]{
      linear.updateGradInput(input, gradOutput)
    }(iters)

    val nnLinear = nn.Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    val nnOutput = nnLinear.forward(input)

    manyTimes[Tensor[Float]]{
      nnLinear.updateGradInput(input, gradOutput)
    }(warm)

    val (nnCosts, _) = manyTimes[Tensor[Float]]{
      nnLinear.updateGradInput(input, gradOutput)
    }(iters)

    println(costs)
    println(nnCosts)
    println(speedup(nnCosts, costs))

    linear.gradInput should be (nnLinear.gradInput)
    costs should be < nnCosts
  }

  "linear accGradParameters" should "work correctly" in {
    val inputSize = 2
    val outputSize = 2
    val batchSize = 2

    val initWeight = Tensor[Float](outputSize, inputSize).rand()
    val initBias = Tensor[Float](outputSize).rand()

    val linear = Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    val input = Tensor[Float](batchSize, inputSize).rand()
    val output = linear.forward(input)

    val gradOutput = Tensor[Float]().resizeAs(output).rand()
    val gradInput = linear.updateGradInput(input, gradOutput)

    val nnLinear = nn.Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    val nnOutput = nnLinear.forward(input)
    val nnGradInput = nnLinear.updateGradInput(input, gradOutput)

    linear.accGradParameters(input, gradOutput)
    nnLinear.accGradParameters(input, gradOutput)

    println(linear.gradWeight)
    println(linear.gradBias)
    println("-" * 80)
    println(nnLinear.gradWeight)
    println(nnLinear.gradBias)

    linear.gradWeight should be (nnLinear.gradWeight)
    linear.gradBias should be (nnLinear.gradBias)
  }

  "linear accGradParameters multi times" should "work correctly" in {
    val inputSize = 2
    val outputSize = 2
    val batchSize = 2

    val initWeight = Tensor[Float](outputSize, inputSize).rand()
    val initBias = Tensor[Float](outputSize).rand()

    val length = 100
    val inputs = new Array[Tensor[Float]](length)
    for (i <- inputs.indices) {
      inputs(i) = Tensor[Float](batchSize, inputSize).rand()
    }

    val gradOutputs = new Array[Tensor[Float]](length)
    for (i <- 0 until length) {
      gradOutputs(i) = Tensor[Float](batchSize, outputSize).rand()
    }

    val linear = Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    linear.forward(inputs.last)
    linear.updateGradInput(inputs.last, gradOutputs.last)

    val nnLinear = nn.Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    nnLinear.forward(inputs.last)
    nnLinear.updateGradInput(inputs.last, gradOutputs.last)

    linear.gradWeight should be (nnLinear.gradWeight)
    linear.gradBias should be (nnLinear.gradBias)

    for (i <- 0 until length) {
      linear.accGradParameters(inputs(i), gradOutputs(i))
      nnLinear.accGradParameters(inputs(i), gradOutputs(i))
      linear.gradWeight should be (nnLinear.gradWeight)
    }

    linear.gradWeight should be (nnLinear.gradWeight)
    linear.gradBias should be (nnLinear.gradBias)
  }

  "linear accGradParameters multi times" should "work much faster than blas" in {
    val inputSize = 64
    val outputSize = 64
    val batchSize = 4

    val initWeight = Tensor[Float](outputSize, inputSize).rand()
    val initBias = Tensor[Float](Array(outputSize)).rand()

    val warm = 10
    val iters = 100

    val input = Tensor[Float](batchSize, inputSize).rand()
    val gradOutput = Tensor[Float](batchSize, outputSize).rand()

    val linear = Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    linear.forward(input)
    linear.updateGradInput(input, gradOutput)

    val nnLinear = nn.Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    nnLinear.forward(input)
    nnLinear.updateGradInput(input, gradOutput)

    val time = manyTimes {
      linear.accGradParameters(input, gradOutput)
    } _

    val nnTime = manyTimes {
      nnLinear.accGradParameters(input, gradOutput)
    } _

    time(warm)
    nnTime(warm)

    val (costs, _) = time(iters)
    val (nnCosts, _) = nnTime(iters)

    println(costs)
    println(nnCosts)
    println(speedup(nnCosts, costs))
    println("-" * 80)
    println(linear.computing / 1e9)
    println(linear.aggregating / 1e9)

    linear.gradWeight should be (nnLinear.gradWeight)
    linear.gradBias should be (nnLinear.gradBias)

    costs should be < nnCosts
  }

  "linear perf with blas" should "work correctly" in {
    val inputSize = 4096
    val outputSize = 1000
    val batchSize = 32

    val initWeight = Tensor[Float](outputSize, inputSize).rand(-1, 1)
    val initBias = Tensor[Float](outputSize).rand(-1, 1)

    val linear = Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    val input = Tensor[Float](batchSize, inputSize).rand(-1, 1)
    val output = linear.forward(input)
    val gradOutput = Tensor[Float]().resizeAs(output).rand(-1, 1)

    val nnLinear = nn.Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)

    val warm = 10
    val iters = 100
    val time = manyTimes {
      linear.forward(input)
      linear.updateGradInput(input, gradOutput)
      linear.accGradParameters(input, gradOutput)
    } _

    val nnTime = manyTimes {
      nnLinear.forward(input)
      nnLinear.updateGradInput(input, gradOutput)
      nnLinear.accGradParameters(input, gradOutput)
    } _

    nnTime(warm)
    linear.computing = 0.0
    linear.aggregating = 0.0
    val (nnCosts, _) = nnTime(iters)
    time(warm)
    val (costs, _) = time(iters)

    println(costs)
    println(nnCosts)
    println((nnCosts - costs) / nnCosts)
    println(speedup(nnCosts, costs))

    println("-" * 80)
    println(linear.computing / 1e9)
    println(linear.aggregating / 1e9)

    nnCosts should be > costs
  }

  "linear with maxpooling" should "work correctly" in {
    val model = nn.Sequential[Float]()
    model.add(ConvolutionDnn(3, 64, 11, 11, 4, 4, 2, 2, 1)
      .setName("conv1"))
    model.add(ReLUDnn(true).setName("relu1"))
    model.add(PoolingDnn(3, 3, 2, 2).setName("pool1"))
    model.add(ConvolutionDnn(64, 192, 5, 5, 1, 1, 2, 2).setName("conv2"))
    model.add(ReLUDnn(true).setName("relu2"))
    model.add(PoolingDnn(3, 3, 2, 2).setName("pool2"))
    model.add(ConvolutionDnn(192, 384, 3, 3, 1, 1, 1, 1).setName("conv3"))
    model.add(ReLUDnn(true).setName("relu3"))
    model.add(ConvolutionDnn(384, 256, 3, 3, 1, 1, 1, 1).setName("conv4"))
    model.add(ReLUDnn(true).setName("relu4"))
    model.add(ConvolutionDnn(256, 256, 3, 3, 1, 1, 1, 1).setName("conv5"))
    model.add(ReLUDnn(true).setName("relu5"))
    model.add(PoolingDnn(3, 3, 2, 2).setName("poo5"))
    model.add(nn.View(256 * 6 * 6).setName("view"))
    model.add(Linear(256 * 6 * 6, 4096).setName("fc6"))
    model.add(ReLUDnn(true).setName("relu6"))
    model.add(nn.Dropout(0.5).setName("drop6"))
    model.add(Linear(4096, 4096).setName("fc7"))
    model.add(ReLUDnn(true).setName("relu7"))
    model.add(nn.Dropout(0.5).setName("drop7"))
    model.add(Linear(4096, 10).setName("fc8"))
    model.add(nn.SoftMax().setName("logsoftmax"))

    val input = Tensor[Float](4, 3, 227, 227).rand()
    model.forward(input)

    val gradOutput = Tensor[Float]().resizeAs(model.output.toTensor).rand()
    model.backward(input, gradOutput)
  }

  "linear + relu" should "work correctly" in {
    val model = nn.Sequential[Float]().add(Linear(10, 20)).add(ReLUDnn())
    model.getParameters()._2.zero()
    val clone = model.cloneModule()
    System.setProperty("bigdl.localMode", "true")
    System.setProperty("bigdl.coreNumber", "3")
    Engine.init

    val input = Tensor(4, 10).rand()
    val gradOutput = Tensor(4, 20).rand()

    Engine.default.invokeAndWait((0 until 2).map(i =>
      () => {
        val clone = model.cloneModule()
        clone.forward(input)
        clone.backward(input, gradOutput)
    }))
  }

  "relu + linear" should "work correctly" in {
    val model = nn.Sequential().add(ReLUDnn(false)).add(Linear(20, 10))
    val clone = model.cloneModule()
    val input = Tensor(20).rand()
    model.forward(input)
    println("=" * 80)
    clone.forward(input)

    val gradOutput = Tensor().resizeAs(model.output.toTensor)
    model.backward(input, gradOutput)
    clone.backward(input, gradOutput)
  }

  "test clone module" should "work correctly" in {
    val model = Linear(20, 10)
    val clone = model.cloneModule()

    val input = Tensor(4, 20).rand()

    model.forward(input)
    clone.forward(input)

    val gradOutput = Tensor().resizeAs(model.output).rand

    model.backward(input, gradOutput)
    clone.backward(input, gradOutput)
  }

  "alexnet clone" should "work correctly" in {
    val model = AlexNet(1000)
    val clone = model.cloneModule()
    model.training()
    clone.training()
    val input = Tensor(1, 3, 227, 227)

    model.forward(input)
    clone.forward(input)

    val gradOutput = Tensor().resizeAs(model.output.toTensor)
    model.backward(input, gradOutput)
    clone.backward(input, gradOutput)
  }

  "AlexNet perf" should "work correctly" in {
    val blas = AlexNet(1000, hasDropout = false)
    val dnn = AlexNet.dnn(1000, hasDropout = false)

    blas.training()
    dnn.training()

    val input = Tensor(4, 3, 227, 227).rand()
    blas.forward(input)
    dnn.forward(input)

    val gradOutput = Tensor().resizeAs(blas.output.toTensor)
    blas.backward(input, gradOutput)
    dnn.backward(input, gradOutput)

    blas.resetTimes()
    dnn.resetTimes()

    val warmup = 20
    val iters = 50

    var i = 0
    while (i < warmup) {
      blas.forward(input)
      blas.backward(input, gradOutput)

      dnn.forward(input)
      dnn.backward(input, gradOutput)
      i += 1
    }

    blas.resetTimes()
    dnn.resetTimes()

    i = 0
    while (i < iters) {
      blas.forward(input)
      blas.backward(input, gradOutput)

      dnn.forward(input)
      dnn.backward(input, gradOutput)
      i += 1
    }

    def format(v: Double): Double = {
      (v / 1e6 / iters).formatted("%2.4f").toDouble
    }
    val names = blas.getTimes().map(_._1.getName())
    val blasForwardTime = blas.getTimes().map(x => format(x._2))
    val blasBackwardTime = blas.getTimes().map(x => format(x._3))

    val dnnForwardTime = dnn.getTimes().map(x => format(x._2))
    val dnnBackwardTime = dnn.getTimes().map(x => format(x._3))

    val forwardUpgrade = blasForwardTime.zip(dnnForwardTime).map { t =>
      ((t._1 - t._2) / t._2.toDouble).formatted("%2.2f")
    }
    val backwardUpgrade = blasBackwardTime.zip(dnnBackwardTime).map { t =>
      ((t._1 - t._2) / t._2.toDouble).formatted("%2.2f")
    }

    val header = List("MODULE NAME", "MKL-BLAS", "MKL-DNN", "UPGRADE")

    def rows4(input: List[Array[_]]): List[List[_]] = {
      input(0).toList zip input(1).toList zip input(2) zip input(3) map {
        case (((a, b), c), d) => List(a, b, c, d)
      }
    }

    val forwardTime = rows4(List(names, blasForwardTime, dnnForwardTime, forwardUpgrade))

    val backwardTime = rows4(List(names, blasBackwardTime, dnnBackwardTime, backwardUpgrade))

    println(Tabulator.format(header:: forwardTime))
    println("=" * 80)
    println(Tabulator.format(header:: backwardTime))
  }

  "the num of omp threads" should "be 1" in {
    val threads = MKL.getNumThreads
    println(threads)
  }

  "1-D input" should "work correctly" in {
    val model = Linear(20, 10)
    val input = Tensor(20).rand()

    model.forward(input)

    val gradOutput = Tensor().resizeAs(model.output).rand()
    model.updateGradInput(input, gradOutput)
  }

  "AlexNet test" should "work correctly" in {
    System.setProperty("bigdl.localMode", "true")
    System.setProperty("bigdl.coreNumber", "4")
    Engine.init

    val batchSize = 16
    val model = AlexNet(1000)
    println(model)
    val criterion = ClassNLLCriterion()
    val miniBatch = MiniBatch[Float](Tensor(batchSize, 3, 227, 227), Tensor(batchSize).fill(1))

    val dummyDataSet = new LocalDataSet[MiniBatch[Float]] {
      override def data(train : Boolean): Iterator[MiniBatch[Float]] = {
        new Iterator[MiniBatch[Float]] {
          private val index = new AtomicInteger()
          override def hasNext: Boolean = {
            if (train) {
              true
            } else {
              index.get() < 100000
            }
          }

          override def next(): MiniBatch[Float] = {
            index.getAndIncrement()
            miniBatch
          }
        }
      }
      override def size(): Long = 100000
      override def shuffle(): Unit = {}
    }

    model.training()
    model.resetTimes()
    val optimizer = Optimizer(model, dummyDataSet, criterion)
    val optimizedModel = optimizer.setEndWhen(Trigger.maxIteration(50)).optimize()
  }
}
