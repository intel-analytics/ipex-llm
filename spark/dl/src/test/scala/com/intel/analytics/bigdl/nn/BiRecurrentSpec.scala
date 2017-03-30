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

package com.intel.analytics.bigdl.torch

import java.io.PrintWriter

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.numeric.NumericDouble
import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{T, Table}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.sys.process._

@com.intel.analytics.bigdl.tags.Serial
class BiRecurrentSpec  extends FlatSpec with BeforeAndAfter with Matchers {
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
    val tmpFile = java.io.File.createTempFile("checkRNN", ".lua")
    val writer = new PrintWriter(tmpFile)
    writer.write("exist = (pcall(require, 'rnn'))\n print(exist)")
    writer.close()

    val existsRNN =
      Seq(System.getProperty("torch_location", "th"), tmpFile.getAbsolutePath).!!.trim
    if (!existsRNN.contains("true")) {
      cancel("Torch rnn is not installed")
    }
  }

  "A BiRecurrent " should "has same loss as torch rnn" in {

    val hiddenSize = 4
    val linearHidden = 8
    val inputSize = 6
    val outputSize = 5
    val bpttTruncate = 3
    val seqLength = 5
    val seed = 100
    val depth = 2

    val input = Tensor[Double](Array(1, seqLength, inputSize))
    val labels = Tensor[Double](Array(1, seqLength))
    for (i <- 1 to seqLength) {
      val rdmLabel = Math.ceil(math.random * outputSize).toInt
      val rdmInput = Math.ceil(math.random * inputSize).toInt
      input.setValue(1, i, rdmInput, 1.0)
      labels.setValue(1, i, rdmLabel)
    }

    RNG.setSeed(seed)

    def basicBlock(inputSize: Int, hiddenSize: Int): Module[Double] = {
      Sequential()
        .add(BiRecurrent[Double](CAddTable[Double]())
          .add(RnnCell[Double](inputSize, hiddenSize, Sigmoid[Double]())))
    }

    val model = Sequential[Double]()
    for (i <- 1 to depth) {
      if (i == 1) {
        model.add(basicBlock(inputSize, hiddenSize))
      } else {
        model.add(basicBlock(hiddenSize, hiddenSize))
      }
    }
      model.add(TimeDistributed[Double](Linear[Double](hiddenSize, outputSize)))
    val criterion = TimeDistributedCriterion[Double](
      CrossEntropyCriterion[Double]())
    val logSoftMax = TimeDistributed[Double](LogSoftMax[Double]())

    val (weights, grad) = model.getParameters()
    val code =
      s"""
         |
      |-- 1.4. Combine 1.1 and 1.3 to produce final model
         |require 'rnn'
         |torch.manualSeed($seed)
         |
         |local function basicblock(inputSize, hiddenSize)
         |      local rm =  nn.Sequential() -- input is {x[t], h[t-1]}
         |         :add(nn.ParallelTable()
         |            :add(nn.Linear(inputSize, hiddenSize)) -- input layer
         |            :add(nn.Linear(hiddenSize, hiddenSize))) -- recurrent layer
         |         :add(nn.CAddTable()) -- merge
         |         :add(nn.Sigmoid()) -- transfer
         |      local rm1 =  nn.Sequential() -- input is {x[t], h[t-1]}
         |         :add(nn.ParallelTable()
         |            :add(nn.Linear(inputSize, hiddenSize)) -- input layer
         |            :add(nn.Linear(hiddenSize, hiddenSize))) -- recurrent layer
         |         :add(nn.CAddTable()) -- merge
         |         :add(nn.Sigmoid()) -- transfer
         |
         |      local rnn = nn.Recurrence(rm, hiddenSize, 1)
         |      local rnn1 = nn.Recurrence(rm1, hiddenSize, 1)
         |  return nn.Sequential()
         |          :add(nn.BiSequencer(rnn, rnn1, nn.CAddTable()))
         |end
         |
         |
      |model = nn.Sequential()
         |:add(nn.SplitTable(1))
         |
         |  for i=1,$depth do
         |    if i == 1 then
         |    model:add(basicblock($inputSize, $hiddenSize))
         |    else
         |    model:add(basicblock($hiddenSize, $hiddenSize))
         |    end
         |  end
         |
         |  model:add(nn.JoinTable(1, 5))
         |--:add(nn.Sequencer(
         |-- nn.Sequential()
         |--   --:add(nn.LSTM($inputSize, $hiddenSize, 1, true))
         |--   :add(nn.FastLSTM($inputSize, $hiddenSize))
         |   :add(nn.Linear($hiddenSize, $outputSize))
         |--   ))
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
      |for i = 1,10,1 do
         |   optim.sgd(feval, parameters, state)
         |end
         |
         |labels = labels
         |err=criterion.output
         |err2=criterion.gradInput
         |output = model.output
         |gradInput = model.gradInput
    """.stripMargin

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input.transpose(1, 2), "weights" -> weights,
        "labels" -> labels(1)),
      Array("err", "parameters", "gradParameters", "output", "gradInput", "err2", "labels"))

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
    for (i <- 1 to 10) {
      loss = sgd.optimize(feval, weights, state)._2
      println(s"${i}-th loss = ${loss(0)}")
    }
    val end = System.nanoTime()
    println("Time: " + (end - start) / 1E6)

    val output = model.output.toTensor[Double]
    val logOutput = logSoftMax.forward(output)
    val prediction = logOutput.max(3)._2

    luaOutput2 should be(loss(0) +- 1e-5)
  }

}
