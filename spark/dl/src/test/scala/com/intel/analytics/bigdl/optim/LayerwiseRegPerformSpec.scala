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
package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class LayerwiseRegPerformSpec extends FlatSpec with Matchers {
  "perform No L2 regularizer" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val seed = 100

    RNG.setSeed(seed)
    val state1 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "momentum" -> 0.002)

    val inputN = 28
    val outputN = 10
    val batchSize = 5
    val criterion = new MSECriterion[Double]

    val input = Tensor[Double](batchSize, inputN, inputN).rand().mul(100)
    val labels = Tensor[Double](batchSize, outputN).rand().mul(10)

    val model1 = Sequential()
      .add(Reshape(Array(1, 28, 28)))
      .add(SpatialConvolution(1, 6, 5, 5).setName("conv1_5x5"))
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Tanh())
      .add(SpatialConvolution(6, 12, 5, 5).setName("conv2_5x5"))
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Reshape(Array(12 * 4 * 4)))
      .add(Linear(12 * 4 * 4, 100).setName("fc1"))
      .add(Tanh())
      .add(Linear(100, outputN).setName("fc2"))
      .add(LogSoftMax())

    val (weights1, grad1) = model1.getParameters()

    val sgd = new SGD[Double]

    def feval1(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model1.forward(input).toTensor[Double]
      val _loss = criterion.forward(output, labels)
      model1.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model1.backward(input, gradInput)
      (_loss, grad1)
    }

    var loss1: Array[Double] = null

    // Warm up
    for (i <- 1 to 500) {
      loss1 = sgd
        .optimize(feval1, weights1, state1)._2
    }

    val iteration = 10000
    var start: Long = 0
    var end: Long = 0

    // Global L2
    start = System.nanoTime()
    for (i <- 1 to iteration) {
      loss1 = sgd.optimize(feval1, weights1, state1)._2
    }
    end = System.nanoTime()
    println(s"NO L2 time cost: ${(end - start) / 1e6}")
  }

  "perform Global L2 regularizer" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val seed = 100

    RNG.setSeed(seed)
    val state1 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.1, "momentum" -> 0.002)

    val inputN = 28
    val outputN = 10
    val batchSize = 5
    val criterion = new MSECriterion[Double]

    val input = Tensor[Double](batchSize, inputN, inputN).rand().mul(100)
    val labels = Tensor[Double](batchSize, outputN).rand().mul(10)

    val model1 = Sequential()
      .add(Reshape(Array(1, 28, 28)))
      .add(SpatialConvolution(1, 6, 5, 5).setName("conv1_5x5"))
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Tanh())
      .add(SpatialConvolution(6, 12, 5, 5).setName("conv2_5x5"))
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Reshape(Array(12 * 4 * 4)))
      .add(Linear(12 * 4 * 4, 100).setName("fc1"))
      .add(Tanh())
      .add(Linear(100, outputN).setName("fc2"))
      .add(LogSoftMax())

    val (weights1, grad1) = model1.getParameters()

    val sgd = new SGD[Double]

    def feval1(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model1.forward(input).toTensor[Double]
      val _loss = criterion.forward(output, labels)
      model1.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model1.backward(input, gradInput)
      (_loss, grad1)
    }

    var loss1: Array[Double] = null

    // Warm up
    for (i <- 1 to 500) {
      loss1 = sgd
        .optimize(feval1, weights1, state1)._2
    }

    val iteration = 10000
    var start: Long = 0
    var end: Long = 0

    // Global L2
    start = System.nanoTime()
    for (i <- 1 to iteration) {
      loss1 = sgd.optimize(feval1, weights1, state1)._2
    }
    end = System.nanoTime()
    println(s"Global L2 time cost: ${(end - start) / 1e6}")
  }

  "perform layerwise L2 regularizer" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val seed = 100

    RNG.setSeed(seed)
    val state2 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      ,"momentum" -> 0.002)

    val inputN = 28
    val outputN = 10
    val batchSize = 5
    val criterion = new MSECriterion[Double]

    val input = Tensor[Double](batchSize, inputN, inputN).rand().mul(100)
    val labels = Tensor[Double](batchSize, outputN).rand().mul(10)

    val model2 = Sequential()
      .add(Reshape(Array(1, 28, 28)))
      .add(SpatialConvolution(1, 6, 5, 5,
        wRegularizer = L2Regularizer(0.1), bRegularizer = L2Regularizer(0.1)
      ).setName("conv1_5x5"))
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Tanh())
      .add(SpatialConvolution(6, 12, 5, 5,
        wRegularizer = L2Regularizer(0.1), bRegularizer = L2Regularizer(0.1)
      ).setName("conv2_5x5"))
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Reshape(Array(12 * 4 * 4)))
      .add(Linear(12 * 4 * 4, 100,
        wRegularizer = L2Regularizer(0.1), bRegularizer = L2Regularizer(0.1)
      ).setName("fc1"))
      .add(Tanh())
      .add(Linear(100, outputN,
        wRegularizer = L2Regularizer(0.1), bRegularizer = L2Regularizer(0.1)
      ).setName("fc2"))
      .add(LogSoftMax())
    val (weights2, grad2) = model2.getParameters()

    val sgd = new SGD[Double]

    def feval2(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model2.forward(input).toTensor[Double]
      val _loss = criterion.forward(output, labels)
      model2.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model2.backward(input, gradInput)
      (_loss, grad2)
    }

    var loss2: Array[Double] = null

    // Warm up
    for (i <- 1 to 500) {
      loss2 = sgd.optimize(feval2, weights2, state2)._2
    }

    val iteration = 10000
    var start: Long = 0
    var end: Long = 0

    // layer-wise L2
    start = System.nanoTime()
    for (i <- 1 to iteration) {
      loss2 = sgd.optimize(feval2, weights2, state2)._2
    }
    end = System.nanoTime()
    println(s"Layer-wise L2 time cost: ${(end - start) / 1e6}")
  }

  "perform layerwise L1 regularizer" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble

    val state3 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.0, "momentum" -> 0.002)
    val seed = 100

    RNG.setSeed(seed)
    val inputN = 28
    val outputN = 10
    val batchSize = 5
    val criterion = new MSECriterion[Double]

    val input = Tensor[Double](batchSize, inputN, inputN).rand().mul(100)
    val labels = Tensor[Double](batchSize, outputN).rand().mul(10)

    val model3 = Sequential()
      .add(Reshape(Array(1, 28, 28)))
      .add(SpatialConvolution(1, 6, 5, 5,
        wRegularizer = L1Regularizer(0.1), bRegularizer = L1Regularizer(0.1)
      ).setName("conv1_5x5"))
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Tanh())
      .add(SpatialConvolution(6, 12, 5, 5,
        wRegularizer = L1Regularizer(0.1), bRegularizer = L1Regularizer(0.1)
      ).setName("conv2_5x5"))
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Reshape(Array(12 * 4 * 4)))
      .add(Linear(12 * 4 * 4, 100,
        wRegularizer = L1Regularizer(0.1), bRegularizer = L1Regularizer(0.1)
      ).setName("fc1"))
      .add(Tanh())
      .add(Linear(100, outputN,
        wRegularizer = L1Regularizer(0.1), bRegularizer = L1Regularizer(0.1)
      ).setName("fc2"))
      .add(LogSoftMax())

    val (weights3, grad3) = model3.getParameters()
    val sgd = new SGD[Double]

    def feval3(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model3.forward(input).toTensor[Double]
      val _loss = criterion.forward(output, labels)
      model3.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model3.backward(input, gradInput)
      (_loss, grad3)
    }

    var loss1: Array[Double] = Array(100)

    // Warm up
    for (i <- 1 to 500) {
      loss1 = loss1.map(_ * 10 - 10 + 10)
    }

    val iteration = 10000
    var start: Long = 0
    var end: Long = 0

    var loss3: Array[Double] = null
    // layer-wise L1
    start = System.nanoTime()
    for (i <- 1 to iteration) {
      loss3 = sgd.optimize(feval3, weights3, state3)._2
    }
    end = System.nanoTime()
    println(s"Layer-wise L1 time cost: ${(end - start) / 1e6}")
  }
}
