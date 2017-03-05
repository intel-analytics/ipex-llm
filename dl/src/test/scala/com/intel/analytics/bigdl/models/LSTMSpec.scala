/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.models

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.torch.TH
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class LSTMSpec  extends FlatSpec with BeforeAndAfter with Matchers {
  "A LSTM " should "converge" in {

    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 3
    val seqLength = 5
    val seed = 100

    val input = Tensor[Double](Array(1, seqLength, inputSize))
    val labels = Tensor[Double](Array(1, seqLength))
    for (i <- 1 to seqLength) {
      val rdmLabel = Math.ceil(math.random * outputSize).toInt
      val rdmInput = Math.ceil(math.random * inputSize).toInt
      input.setValue(1, i, rdmInput, 1.0)
      labels.setValue(1, i, rdmLabel)
    }

//    input.zero()
    println(input)
    RNG.setSeed(seed)
    val model = Sequential[Double]()
      .add(Recurrent[Double](hiddenSize, bpttTruncate)
      .add(RnnCell[Double](inputSize, hiddenSize, Identity[Double]())))
//      .add(RnnCell[Double](inputSize, hiddenSize, Tanh[Double]())))
//            .add(FastLSTMCell[Double](inputSize, hiddenSize)))
      .add(TimeDistributed[Double]().add(Linear[Double](hiddenSize, outputSize)))

//    model.reset()

    val criterion = TimeDistributedCriterion[Double](
      CrossEntropyCriterion[Double](sizeAverage = false))
    val logSoftMax = TimeDistributed[Double]().add(LogSoftMax[Double]())

    val (weights, grad) = model.getParameters()
//    weights.apply1(_ => 1.0)


    val facebookCode =
      s"""
        |torch.manualSeed($seed)
        |
        |require 'nn'
        |
        |local rnnlib = require 'rnnlib'
        |
        |local lstm = nn.LSTM{ inputsize = $inputSize,
        | hidsize = $hiddenSize, nlayer = 1, usecudnn = false  }
        |local model = nn.Sequential()
        | :add(lstm)
        | :add(nn.SelectTable(2))
        |    -- Select the output of the last layer, since the output
        |    -- of all layers are returned.
        |    :add(nn.SelectTable(-1))
        |    -- Flatten the output from bptt x bsz x ntoken to bptt * bsz x ntoken.
        |    -- Note that the first dimension is actually a table, so there is
        |    -- copying involved during the flattening.
        |    :add(nn.JoinTable(1))
        |    :add(nn.Linear($hiddenSize, $outputSize))
        |
        |-- Create and initialize hiddens to zero.
        |lstm:initializeHidden(1)
        |
        |
        |-- Perform the forward pass.
        |local output = model:forward{ lstm.hiddenbuffer, input  }
        |-- This is just an example, normally you would not use the output as the gradOutput.
        |-- But the gradOutput should have the same shape as the model output.
        |model:backward({ lstm.hiddenbuffer, input  }, output)
        |model:updateParameters(0.1)
        |
        |local parameters, gradParameters = model:getParameters()
        |model:zeroGradParameters()
        |--parameters:copy(weights)
        |
        |output2 = model:forward({ lstm.hiddenbuffer, input  })
        |gradInput2= model.gradInput
        |
        |parameters_initial = parameters : clone()
        |gradParameters_initial = gradParameters : clone()
        |
        |local criterion =  nn.CrossEntropyCriterion()
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
        |local output1 = model:forward({ lstm.hiddenbuffer, input  })
        |local err1 = criterion:forward(output1, labels)
        |local gradOutput1 = criterion:backward(output1, labels)
        |model:backward({ lstm.hiddenbuffer, input  }, gradOutput1)
        |return err1, gradParameters
        |end
        |
        |for i = 1,100,1 do
        |   optim.sgd(feval, parameters, state)
        |end
        |
        |output=model.output
        |err=criterion.output
        |gradOutput=criterion.gradInput
        |gradInput = model.gradInput
        |
      """.stripMargin

    val code = s"""
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
      |         :add(nn.Identity()) -- transfer
      |     --    :add(nn.Tanh()) -- transfer
      |
      | -- local rm1 =  nn.Sequential() -- input is {x[t], h[t-1]}
      | --        :add(nn.ParallelTable()
      | --           :add(nn.Linear($inputSize, $hiddenSize)) -- input layer
      | --           :add(nn.Identity())) -- recurrent layer
      | --        :add(nn.CAddTable()) -- merge
      | --        :add(nn.Sigmoid()) -- transfer
      |      rnn = nn.Recurrence(rm, $hiddenSize, 1, 1)
      | --     rnn.userPrevOutput = torch.Tensor(1, $hiddenSize):zero()
      |
      |model = nn.Sequential()
      |:add(nn.SplitTable(1))
      |:add(nn.Sequencer(
      | nn.Sequential()
      |   :add(rnn)
      |   :add(nn.Linear($hiddenSize, $outputSize))
      |   ))
      |
      |--   model:reset()
      |
      |
 |local parameters, gradParameters = model:getParameters()
      |model:zeroGradParameters()
 |parameters:copy(weights)
 |
      |output2 = model:forward(input)
      |gradInput2= model.gradInput
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
      |for i = 1,100,1 do
      |   optim.sgd(feval, parameters, state)
      |end
      |
      |output=model.output
      |err=criterion.output
      |err2=criterion.gradInput
      |gradOutput=criterion.gradInput
      |gradInput = model.gradInput
    """.stripMargin

//    val (luaTime, torchResult) = TH.run(facebookCode,
//      Map("input" -> SplitTable[Double](1).forward(input.transpose(1, 2)), "weights" -> weights,
//        "labels" -> SplitTable[Double](1).forward(labels.t())),
//      Array("output", "err", "parameters", "gradweight", "output2", "gradInput2", "gradInput"))

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input.transpose(1, 2), "weights" -> weights,
        "labels" -> SplitTable[Double](1).forward(labels.t())),
      Array("output", "err", "parameters", "gradParameters", "output2", "gradInput", "err2"))

//    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val output1 = torchResult("output2")
    val gradInput1 = torchResult("gradInput")
    val gradParameters = torchResult("gradParameters")
    val err2 = torchResult("err2")

    println("Element forward: " + output1)
    println("BigDL forward: " + model.forward(input).toTensor[Double].clone())

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
    for (i <- 1 to 100) {
      loss = sgd.optimize(feval, weights, state)._2
      println(s"${i}-th loss = ${loss(0)}")
    }


    println("Element weight: " + luaweight)
    println("BigDL weight: " + weights)

    println("Element forward: " + output1)
    println("BigDL forward: " + model.output)

    println("BigDL labels: " + labels)

    val crtnGradInput = criterion.backward(model.output, labels)
    println(s"element: criterion gradInput: $err2")
    println("BigDL criterion gradInput: " + crtnGradInput)

    println(s"element: gradInput: $gradInput1")
    println("BigDL: " + model.gradInput.toTensor[Double])

    println(s"element: gradWeight: $gradParameters")
    println("BigDL: " + model.getParameters()._2)

    val end = System.nanoTime()
    println("Time: " + (end - start) / 1E6)

    val output = model.forward(input).asInstanceOf[Tensor[Double]]
    val logOutput = logSoftMax.forward(output)
    val prediction = logOutput.max(2)._2

//    labels.squeeze() should be (prediction.squeeze())

//    luaOutput1 should be(output)
    luaOutput2 should be(loss(0))
  }

  "A LSTM " should "converge in batch mode" in {

    val batchSize = 10
    val nWords = 5
    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 3
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double](hiddenSize, bpttTruncate)
        .add(LSTMCell[Double](inputSize, hiddenSize)))
      .add(Select(2, nWords))
      .add(Linear[Double](hiddenSize, outputSize))

    model.reset()

    val criterion = CrossEntropyCriterion[Double]()
    val logSoftMax = LogSoftMax[Double]()

    val (weights, grad) = model.getParameters()

    val input = Tensor[Double](Array(batchSize, nWords, inputSize))
    val labels = Tensor[Double](batchSize)
    for (b <- 1 to batchSize) {
      for (i <- 1 to nWords) {
        val rdmInput = Math.ceil(RNG.uniform(0.0, 1.0) * inputSize).toInt
        input.setValue(b, i, rdmInput, 1.0)
      }
      val rdmLabel = Math.ceil(RNG.uniform(0.0, 1.0) * outputSize).toInt
      labels.setValue(b, rdmLabel)
    }

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

    for (i <- 1 to 200) {
      val (_, loss) = sgd.optimize(feval, weights, state)
      println(s"${i}-th loss = ${loss(0)}")
    }

    val output = model.forward(input).asInstanceOf[Tensor[Double]]
    val logOutput = logSoftMax.forward(output)
    val prediction = logOutput.max(2)._2

    labels.squeeze() should be (prediction.squeeze())
  }

  "A LSTM " should "perform correct gradient check" in {

    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 10
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double](hiddenSize, bpttTruncate)
        .add(LSTMCell[Double](inputSize, hiddenSize)))
      .add(Select(1, 1))
      .add(Linear[Double](hiddenSize, outputSize))

    model.reset()

    val input = Tensor[Double](Array(1, 5, inputSize))
    val labels = Tensor[Double](Array(1, 5))
    for (i <- 1 to 5) {
      val rdmLabel = Math.ceil(Math.random()*inputSize).toInt
      val rdmInput = Math.ceil(Math.random()*inputSize).toInt
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
