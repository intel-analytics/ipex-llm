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

package com.intel.analytics.bigdl.integration.torch

import java.io.PrintWriter

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericDouble
import com.intel.analytics.bigdl.optim.{L2Regularizer, SGD}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{Engine, T}

import scala.collection.mutable.ArrayBuffer
import scala.sys.process._

@com.intel.analytics.bigdl.tags.Serial
class GRUSpec  extends TorchRNNSpec {
  System.setProperty("bigdl.disableCheckSysEnv", "true")
  Engine.init(1, 1, true)

  "A GRU" should " be fast" in {
    val inputSize = 1000
    val hiddenSize = 1000
    val batchSize = 12
    val time = 200
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Float](batchSize, time, inputSize).rand
    val gradOutput = Tensor[Float](batchSize, time, hiddenSize).rand

    val model = Recurrent[Float]()
      .add(GRU[Float](inputSize, hiddenSize))

    var startTime = System.nanoTime()
    var duration = (System.nanoTime() - startTime) / 1e9
    var sum = 0.0

    println("warmup ..")
    for (i <- 1 to 5) {
      model.forward(input)
      model.backward(input, gradOutput)
    }

    val n = 5
    for (i <- 1 to n) {
      startTime = System.nanoTime()
      model.forward(input)
      model.backward(input, gradOutput)
      duration = (System.nanoTime() - startTime) / 1e9
      sum += duration
      println(s"iteration-${i} = ${duration}")
    }
    println(s"average = ${sum / n}")
  }

  "A LSTM L2 regularizer" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val hiddenSize = 4
    val inputSize = 6
    val outputSize = 5
    val seqLength = 5
    val seed = 100

    RNG.setSeed(seed)
    val input = Tensor[Double](Array(1, seqLength, inputSize))
    val labels = Tensor[Double](Array(1, seqLength))
    for (i <- 1 to seqLength) {
      val rdmLabel = Math.ceil(RNG.uniform(0, 1) * outputSize).toInt
      val rdmInput = Math.ceil(RNG.uniform(0, 1) * inputSize).toInt
      input.setValue(1, i, rdmInput, 1.0)
      labels.setValue(1, i, rdmLabel)
    }

    println(input)
    val rec1 = Recurrent[Double]()
    val rec2 = Recurrent[Double]()

    val model1 = Sequential[Double]()
      .add(rec1
        .add(GRU[Double](inputSize, hiddenSize)))
      .add(TimeDistributed[Double](Linear[Double](hiddenSize, outputSize)))

    val model2 = Sequential[Double]()
      .add(rec2
        .add(GRU[Double](inputSize, hiddenSize, uRegularizer = L2Regularizer(0.1),
          wRegularizer = L2Regularizer(0.1), bRegularizer = L2Regularizer(0.1))))
      .add(TimeDistributed[Double](Linear[Double](hiddenSize, outputSize,
        wRegularizer = L2Regularizer(0.1), bRegularizer = L2Regularizer(0.1))))

    val criterion = TimeDistributedCriterion[Double](
      CrossEntropyCriterion[Double](), false)
    val state1 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.1, "momentum" -> 0.002)
    val state2 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.0, "momentum" -> 0.002)

    val (weights1, grad1) = model1.getParameters()
    val (weights2, grad2) = model2.getParameters()
    weights2.copy(weights1.clone())
    grad2.copy(grad1.clone())


    val sgd = new SGD[Double]

    def feval1(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model1.forward(input).toTensor[Double]
      val _loss = criterion.forward(output, labels)
      model1.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model1.backward(input, gradInput)
      (_loss, grad1)
    }

    def feval2(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model2.forward(input).toTensor[Double]
      val _loss = criterion.forward(output, labels)
      model2.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model2.backward(input, gradInput)
      (_loss, grad2)
    }

    var loss1: Array[Double] = null
    for (i <- 1 to 100) {
      loss1 = sgd.optimize(feval1, weights1, state1)._2
      println(s"${i}-th loss = ${loss1(0)}")
    }

    var loss2: Array[Double] = null
    for (i <- 1 to 100) {
      loss2 = sgd.optimize(feval2, weights2, state2)._2
      println(s"${i}-th loss = ${loss2(0)}")
    }


    weights1 should be(weights2)
    loss1 should be(loss2)
  }

  "A GRU " should "has same loss as torch rnn" in {
    torchCheck()

    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 3
    val seqLength = 5
    val seed = 100

    RNG.setSeed(seed)
    val input = Tensor[Double](Array(1, seqLength, inputSize))
    val labels = Tensor[Double](Array(1, seqLength))
    for (i <- 1 to seqLength) {
      val rdmLabel = Math.ceil(RNG.uniform(0, 1) * outputSize).toInt
      val rdmInput = Math.ceil(RNG.uniform(0, 1) * inputSize).toInt
      input.setValue(1, i, rdmInput, 1.0)
      labels.setValue(1, i, rdmLabel)
    }

    println(input)
    val rec = Recurrent[Double]()

    val model = Sequential[Double]()
      .add(rec
        .add(GRU[Double](inputSize, hiddenSize)))
      //      .add(GRU[Double](inputSize, hiddenSize)))
      //      .add(RnnCell[Double](inputSize, hiddenSize, Sigmoid[Double]())))
      //            .add(FastGRUCell[Double](inputSize, hiddenSize)))
      .add(TimeDistributed[Double](Linear[Double](hiddenSize, outputSize)))

    val criterion = TimeDistributedCriterion[Double](
      CrossEntropyCriterion[Double]())
    val logSoftMax = TimeDistributed[Double](LogSoftMax[Double]())

    val (weights, grad) = model.getParameters()


    /*
    * Since we changed the structure of GRU, we have to rearrange the parameters.
    */

    val (weightsArray1, gradArray) = model.parameters()
    val weightsArray = weightsArray1.clone
    val weightsTorch = new ArrayBuffer[Tensor[Double]]()

    val i2g12 = weightsArray(0).narrow(1, 1, 2 * hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g12).copy(i2g12)
    val i2g12bias = weightsArray(1).narrow(1, 1, 2 * hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g12bias).copy(i2g12bias)
    val h2g12 = weightsArray(2)
    weightsTorch += Tensor().resizeAs(h2g12).copy(h2g12)
    val i2g3 = weightsArray(0).narrow(1, 1 + 2 * hiddenSize, hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g3).copy(i2g3)
    val i2g3bias = weightsArray(1).narrow(1, 1 + 2 * hiddenSize, hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g3bias).copy(i2g3bias)
    val h2g3 = weightsArray(3)
    weightsTorch += Tensor().resizeAs(h2g3).copy(h2g3)

    weightsTorch += Tensor().resizeAs(weightsArray(4)).copy(weightsArray(4))
    weightsTorch += Tensor().resizeAs(weightsArray(5)).copy(weightsArray(5))

    val (weights2Torch, grad2Torch) =
      (Module.flatten[Double](weightsTorch.toArray), Module.flatten[Double](gradArray))

    val code =
      s"""
         |
      |-- 1.4. Combine 1.1 and 1.3 to produce final model
         |require 'rnn'
         |torch.manualSeed($seed)
         |
      |local rm =  nn.Sequential() -- input is {x[t], h[t-1]}
         |         :add(nn.ParallelTable()
         |            :add(nn.Linear($inputSize, $hiddenSize)) -- input layer
         |            :add(nn.Linear($hiddenSize, $hiddenSize))) -- recurrent layer
         |         :add(nn.CAddTable()) -- merge
         |         :add(nn.Sigmoid()) -- transfer
         |     --    :add(nn.Tanh()) -- transfer
         |
      | -- local rm1 =  nn.Sequential() -- input is {x[t], h[t-1]}
         | --        :add(nn.ParallelTable()
         | --           :add(nn.Linear($inputSize, $hiddenSize)) -- input layer
         | --           :add(nn.Identity())) -- recurrent layer
         | --        :add(nn.CAddTable()) -- merge
         | --        :add(nn.Sigmoid()) -- transfer
         |      rnn = nn.Recurrence(rm, $hiddenSize, 1)
         | --     rnn.userPrevOutput = torch.Tensor(1, $hiddenSize):zero()
         |
      |model = nn.Sequential()
         |:add(nn.SplitTable(1))
         |:add(nn.Sequencer(
         | nn.Sequential()
         |   :add(nn.GRU($inputSize, $hiddenSize, 1))
         |   :add(nn.Linear($hiddenSize, $outputSize))
         |   ))
         |
         |
         |local parameters, gradParameters = model:getParameters()
         |model:zeroGradParameters()
         |parameters:copy(weights)
         |
         |parameters_initial = parameters : clone()
         |gradParameters_initial = gradParameters : clone()
         |
         |local criterion =  nn.SequencerCriterion(nn.CrossEntropyCriterion())
         |
      |
      |state = {
         |  learningRate = 0.5,
         |  momentum = 0.0,
         |  dampening = 0.0,
         |  weightDecay = 0.0
         |}
         |
      |feval = function(x)
         |model:zeroGradParameters()
         |model_initial = model : clone()
         |
      |local output1 = model:forward(input)
         |local err1 = criterion:forward(output1, labels)
         |local gradOutput1 = criterion:backward(output1, labels)
         |model:backward(input, gradOutput1)
         |return err1, gradParameters
         |end
         |
      |for i = 1,50,1 do
         |   optim.sgd(feval, parameters, state)
         |end
         |
      |output=model.output
         |err=criterion.output
         |err2=criterion.gradInput
         |gradOutput=criterion.gradInput
         |gradInput = model.gradInput
    """.stripMargin

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input.transpose(1, 2), "weights" -> weights2Torch,
        "labels" -> SplitTable[Double](1).forward(labels.t())),
      Array("output", "err", "parameters", "gradParameters", "output2", "gradInput", "err2"))

    //    println("Element forward: " + output1)
    //    println("BigDL forward: " + model.forward(input).toTensor[Double].clone())
    //
    val luaOutput2 = torchResult("err").asInstanceOf[Double]
    val luaweight = torchResult("parameters").asInstanceOf[Tensor[Double]]

    val state = T("learningRate" -> 0.5, "momentum" -> 0.0,
      "weightDecay" -> 0.0, "dampening" -> 0.0)
    val sgd = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model.forward(input).asInstanceOf[Tensor[Double]]
      val _loss = criterion.forward(output, labels)
      model.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model.backward(input, gradInput)
      (_loss, grad)
    }

    val start = System.nanoTime()
    var loss: Array[Double] = null
    for (i <- 1 to 50) {
      loss = sgd.optimize(feval, weights, state)._2
      println(s"${i}-th loss = ${loss(0)}")
    }
    val end = System.nanoTime()
    println("Time: " + (end - start) / 1E6)

    //    println("Element weight: " + luaweight)
    //    println("BigDL weight: " + weights)
    //
    //    println("Element forward: " + output1)
    //    println("BigDL forward: " + model.output)
    //
    //    println("BigDL labels: " + labels)
    //
    //    val crtnGradInput = criterion.backward(model.output, labels)
    //    println(s"element: criterion gradInput: $err2")
    //    println("BigDL criterion gradInput: " + crtnGradInput)
    //
    //    println(s"element: gradInput: $gradInput1")
    //    println("BigDL: " + model.gradInput.toTensor[Double])
    //
    //    println(s"element: gradWeight: $gradParameters")
    //    println("BigDL: " + model.getParameters()._2)


    val output = model.forward(input).toTensor
    val logOutput = logSoftMax.forward(output)
    val prediction = logOutput.max(3)._2

    luaOutput2 should be(loss(0) +- 1e-5)
  }


  "A GRU " should "converge" in {
    torchCheck()

    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 3
    val seqLength = 5
    val seed = 100
    RNG.setSeed(seed)

    val input = Tensor[Double](Array(1, seqLength, inputSize))
    val labels = Tensor[Double](Array(1, seqLength))
    for (i <- 1 to seqLength) {
      val rdmLabel = Math.ceil(RNG.uniform(0, 1) * outputSize).toInt
      val rdmInput = Math.ceil(RNG.uniform(0, 1) * inputSize).toInt
      input.setValue(1, i, rdmInput, 1.0)
      labels.setValue(1, i, rdmLabel)
    }

    println(input)
    // RNG.setSeed(seed)
    val rec = Recurrent[Double]()

    val model = Sequential[Double]()
      .add(rec
        .add(GRU[Double](inputSize, hiddenSize)))
      .add(TimeDistributed[Double](Linear[Double](hiddenSize, outputSize)))

    val criterion = TimeDistributedCriterion[Double](
      CrossEntropyCriterion[Double]())
    val logSoftMax = TimeDistributed[Double](LogSoftMax[Double]())

    val (weights, grad) = model.getParameters()

    val state = T("learningRate" -> 0.5, "momentum" -> 0.0,
      "weightDecay" -> 0.0, "dampening" -> 0.0)
    val sgd = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model.forward(input).asInstanceOf[Tensor[Double]]
      val _loss = criterion.forward(output, labels)
      model.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model.backward(input, gradInput)
      (_loss, grad)
    }

    val start = System.nanoTime()
    var loss: Array[Double] = null
    for (i <- 1 to 100) {
      loss = sgd.optimize(feval, weights, state)._2
      println(s"${i}-th loss = ${loss(0)}")
    }
    val end = System.nanoTime()
    println("Time: " + (end - start) / 1E6)

    val output = model.forward(input).toTensor
    val logOutput = logSoftMax.forward(output)
    val prediction = logOutput.max(3)._2

    labels.squeeze() should be (prediction.squeeze())
  }


  "A GRU " should "has same loss as torch rnn in batch mode" in {
    torchCheck()

    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 3
    val seqLength = 5
    val seed = 100
    val batchSize = 5

    RNG.setSeed(seed)
    val input = Tensor[Double](Array(batchSize, seqLength, inputSize))
    val labels = Tensor[Double](Array(batchSize, seqLength))
    for (b <- 1 to batchSize) {
      for (i <- 1 to seqLength) {
        val rdmInput = Math.ceil(RNG.uniform(0, 1)   * inputSize).toInt
        input.setValue(b, i, rdmInput, 1.0)
        val rdmLabel = Math.ceil(RNG.uniform(0, 1)  * outputSize).toInt
        labels.setValue(b, i, rdmLabel)
      }
    }

    println(input)
    val rec = Recurrent[Double]()

    val model = Sequential[Double]()
      .add(rec
        .add(GRU[Double](inputSize, hiddenSize)))
      .add(TimeDistributed[Double](Linear[Double](hiddenSize, outputSize)))

    val criterion = TimeDistributedCriterion[Double](
      CrossEntropyCriterion[Double]())
    val logSoftMax = TimeDistributed[Double](LogSoftMax[Double]())

    val (weights, grad) = model.getParameters()


    /*
    * Since we changed the structure of GRU, we have to rearrange the parameters.
    */

    val (weightsArray1, gradArray) = model.parameters()
    val weightsArray = weightsArray1.clone
    val weightsTorch = new ArrayBuffer[Tensor[Double]]()

    val i2g12 = weightsArray(0).narrow(1, 1, 2 * hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g12).copy(i2g12)
    val i2g12bias = weightsArray(1).narrow(1, 1, 2 * hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g12bias).copy(i2g12bias)
    val h2g12 = weightsArray(2)
    weightsTorch += Tensor().resizeAs(h2g12).copy(h2g12)
    val i2g3 = weightsArray(0).narrow(1, 1 + 2 * hiddenSize, hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g3).copy(i2g3)
    val i2g3bias = weightsArray(1).narrow(1, 1 + 2 * hiddenSize, hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g3bias).copy(i2g3bias)
    val h2g3 = weightsArray(3)
    weightsTorch += Tensor().resizeAs(h2g3).copy(h2g3)

    weightsTorch += Tensor().resizeAs(weightsArray(4)).copy(weightsArray(4))
    weightsTorch += Tensor().resizeAs(weightsArray(5)).copy(weightsArray(5))

    val (weights2Torch, grad2Torch) =
      (Module.flatten[Double](weightsTorch.toArray), Module.flatten[Double](gradArray))

    val code =
      s"""
         |
      |-- 1.4. Combine 1.1 and 1.3 to produce final model
         |require 'rnn'
         |torch.manualSeed($seed)
         |
      |local rm =  nn.Sequential() -- input is {x[t], h[t-1]}
         |         :add(nn.ParallelTable()
         |            :add(nn.Linear($inputSize, $hiddenSize)) -- input layer
         |            :add(nn.Linear($hiddenSize, $hiddenSize))) -- recurrent layer
         |         :add(nn.CAddTable()) -- merge
         |         :add(nn.Sigmoid()) -- transfer
         |     --    :add(nn.Tanh()) -- transfer
         |
      |model = nn.Sequential()
         |:add(nn.SplitTable(1))
         |:add(nn.Sequencer(
         | nn.Sequential()
         |   :add(nn.GRU($inputSize, $hiddenSize, 1))
         |   :add(nn.Linear($hiddenSize, $outputSize))
         |   ))
         |
         |
         |local parameters, gradParameters = model:getParameters()
         |model:zeroGradParameters()
         |parameters:copy(weights)
         |
         |parameters_initial = parameters : clone()
         |gradParameters_initial = gradParameters : clone()
         |
         |local criterion =  nn.SequencerCriterion(nn.CrossEntropyCriterion())
         |
      |
      |state = {
         |  learningRate = 0.5,
         |  momentum = 0.0,
         |  dampening = 0.0,
         |  weightDecay = 0.0
         |}
         |
      |feval = function(x)
         |model:zeroGradParameters()
         |model_initial = model : clone()
         |
      |local output1 = model:forward(input)
         |local err1 = criterion:forward(output1, labels)
         |local gradOutput1 = criterion:backward(output1, labels)
         |model:backward(input, gradOutput1)
         |return err1, gradParameters
         |end
         |
      |for i = 1,50,1 do
         |   optim.sgd(feval, parameters, state)
         |end
         |
      |output=model.output
         |err=criterion.output
         |err2=criterion.gradInput
         |gradOutput=criterion.gradInput
         |gradInput = model.gradInput
    """.stripMargin

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input.transpose(1, 2), "weights" -> weights2Torch,
        "labels" -> SplitTable[Double](1).forward(labels.t())),
      Array("output", "err", "parameters", "gradParameters", "output2", "gradInput", "err2"))

    //    println("Element forward: " + output1)
    //    println("BigDL forward: " + model.forward(input).toTensor[Double].clone())
    //
    val luaOutput2 = torchResult("err").asInstanceOf[Double]
    val luaweight = torchResult("parameters").asInstanceOf[Tensor[Double]]

    val state = T("learningRate" -> 0.5, "momentum" -> 0.0,
      "weightDecay" -> 0.0, "dampening" -> 0.0)
    val sgd = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model.forward(input).asInstanceOf[Tensor[Double]]
      val _loss = criterion.forward(output, labels)
      model.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model.backward(input, gradInput)
      (_loss, grad)
    }

    val start = System.nanoTime()
    var loss: Array[Double] = null
    for (i <- 1 to 50) {
      loss = sgd.optimize(feval, weights, state)._2
      println(s"${i}-th loss = ${loss(0)}")
    }
    val end = System.nanoTime()
    println("Time: " + (end - start) / 1E6)

    //    println("Element weight: " + luaweight)
    //    println("BigDL weight: " + weights)
    //
    //    println("Element forward: " + output1)
    //    println("BigDL forward: " + model.output)
    //
    //    println("BigDL labels: " + labels)
    //
    //    val crtnGradInput = criterion.backward(model.output, labels)
    //    println(s"element: criterion gradInput: $err2")
    //    println("BigDL criterion gradInput: " + crtnGradInput)
    //
    //    println(s"element: gradInput: $gradInput1")
    //    println("BigDL: " + model.gradInput.toTensor[Double])
    //
    //    println(s"element: gradWeight: $gradParameters")
    //    println("BigDL: " + model.getParameters()._2)


    val output = model.forward(input).toTensor
    val logOutput = logSoftMax.forward(output)

        luaOutput2 should be(loss(0) +- 1e-5)
  }


  "A GRU " should "converge in batch mode" in {
    torchCheck()

    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 3
    val batchSize = 3
    val seqLength = 5
    val seed = 100

    RNG.setSeed(seed)
    val input = Tensor[Double](Array(batchSize, seqLength, inputSize))
    val labels = Tensor[Double](Array(batchSize, seqLength))
    for (b <- 1 to batchSize) {
      for (i <- 1 to seqLength) {
        val rdmInput = Math.ceil(RNG.uniform(0.0, 1.0) * inputSize).toInt
        input.setValue(b, i, rdmInput, 1.0)
        val rdmLabel = Math.ceil(RNG.uniform(0.0, 1.0) * outputSize).toInt
        labels.setValue(b, i, rdmLabel)
      }
    }

    println(input)
    val rec = Recurrent[Double]()

    val model = Sequential[Double]()
      .add(rec
        .add(GRU[Double](inputSize, hiddenSize)))
      .add(TimeDistributed[Double](Linear[Double](hiddenSize, outputSize)))

    val criterion = TimeDistributedCriterion[Double](
      CrossEntropyCriterion[Double]())
    val logSoftMax = TimeDistributed[Double](LogSoftMax[Double]())

    val (weights, grad) = model.getParameters()

    val state = T("learningRate" -> 0.5, "momentum" -> 0.0,
      "weightDecay" -> 0.0, "dampening" -> 0.0)
    val sgd = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model.forward(input).asInstanceOf[Tensor[Double]]
      val _loss = criterion.forward(output, labels)
      model.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model.backward(input, gradInput)
      (_loss, grad)
    }

    val start = System.nanoTime()
    var loss: Array[Double] = null
    for (i <- 1 to 200) {
      loss = sgd.optimize(feval, weights, state)._2
      println(s"${i}-th loss = ${loss(0)}")
    }
    val end = System.nanoTime()
    println("Time: " + (end - start) / 1E6)

    val output = model.forward(input).toTensor
    val logOutput = logSoftMax.forward(output)
    val prediction = logOutput.max(3)._2

    labels.squeeze() should be (prediction.squeeze())
  }

  "A GRU " should "perform correct gradient check" in {
    torchCheck()

    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 10
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double]()
        .add(GRU[Double](inputSize, hiddenSize)))
      .add(Select(1, 1))
      .add(Linear[Double](hiddenSize, outputSize))

    model.reset()

    val input = Tensor[Double](Array(1, 5, inputSize))
    val labels = Tensor[Double](Array(1, 5))
    for (i <- 1 to 5) {
      val rdmLabel = Math.ceil(RNG.uniform(0, 1)*inputSize).toInt
      val rdmInput = Math.ceil(RNG.uniform(0, 1)*inputSize).toInt
      input.setValue(1, i, rdmInput, 1.0)
      labels.setValue(1, i, rdmLabel)
    }

    println("gradient check for input")
    val gradCheckerInput = new GradientChecker(1e-2, 1)
    val checkFlagInput = gradCheckerInput.checkLayer[Double](model, input)
    println("gradient check for weights")
    val gradCheck = new GradientCheckerRNN(1e-2, 1)
    val checkFlag = gradCheck.checkLayer(model, input, labels)
  }
}
