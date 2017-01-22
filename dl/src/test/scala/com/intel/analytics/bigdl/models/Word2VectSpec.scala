/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.torch.TH
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.immutable
import scala.math._
import scala.util.Random

class Word2VectSpec extends FlatSpec with BeforeAndAfter with Matchers {

  "Word2Vec Float" should "generate correct output" in {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }

    val seed = 10
    RNG.setSeed(seed)
    val vocabSize = 10
    val embeddingSize = 10
    val numNegSamples = 5
    val batchSize = 2

    def getModel: Module[Float] = {
      val wordVectors = LookupTable(vocabSize, embeddingSize)
 //      wordVectors.weight.apply1(x => 1)
      new Sequential()
        .add(wordVectors)
         .add(ConcatTable()
           .add(Narrow(2, 1, 1))
           .add(Narrow(2, 2, numNegSamples + 1)))
         .add(MM(transA = false, transB = true))
        .add(Sigmoid())
    }

    val input = Tensor(batchSize, 2 + numNegSamples).apply1( e => Random.nextInt(vocabSize) + 1)
//    val labels = Tensor(batchSize, 2 + numNegSamples, embeddingSize).zero()
    val labels = Tensor(batchSize, 1 + numNegSamples).zero()

//    for (i <- 1 to batchSize) {
//      for (j <- 1 to 2 + numNegSamples) {
//        labels.setValue(i, j, 1, 1)
//      }
//    }

     for (i <- 1 to batchSize) {
       labels.setValue(i, 1, 1)
     }

    val model = getModel


    val code =
      "torch.setdefaulttensortype('torch.FloatTensor')" +
        "torch.manualSeed(" + seed + ")\n" +
        "word_vecs = nn.LookupTable(" + vocabSize + ", " +embeddingSize + ")" +
      s"""
        |   -- word_vecs.weight:apply(function() return 1 end)
        |   w2v = nn.Sequential()
        |   w2v:add(word_vecs)
        |   w2v:add(nn.ConcatTable()
        |                   :add(nn.Narrow(2,1,1))
        |                   :add(nn.Narrow(2,2, ${numNegSamples + 1})))
        |    w2v:add(nn.MM(false, true)) -- dot prod and sigmoid to get probabilities
        |    w2v:add(nn.Sigmoid())
        |
        |    model = w2v
        |
        |     local parameters, gradParameters = model:getParameters()
        |                parameters_initial = parameters : clone()
        |                gradParameters_initial = gradParameters : clone()
        |
        |                local criterion = nn.BCECriterion()
        |                state = {
        |                  learningRate = 1e-2,
        |                }
        |
        |         feval = function(x)
        |              model:forward(input)
        |              criterion:forward(model.output, labels)
        |              model:zeroGradParameters()
        |              criterion:backward(model.output, labels)
        |              model:backward(input, criterion.gradInput)
        |              return criterion.output, gradParameters
        |           end
        |
        |             for i = 1, 5, 1 do
        |              w, err = optim.sgd(feval, parameters, state)
        |             end
        | --print(model.modules[2].output)
        |                output=model.output
        |                gradOutput=criterion.gradInput
        |                err = criterion.output
        |                gradInput = model.gradInput
      """.stripMargin

    TH.runNM(code, immutable.Map(
      "input" -> input,
      "labels" -> labels),
      Array("output", "gradOutput", "err", "parameters_initial",
        "gradParameters_initial", "gradInput", "model"))

    val parameterTorch = TH.map("parameters_initial").asInstanceOf[Tensor[Float]]
    val parameters = model.getParameters()._1

    for (i <- 0 until parameters.nElement()) {
      if (abs(parameters.storage().array()(i) - parameterTorch.storage().array()(i)) > 1e-8) {
        println(s"${parameters.storage().array()(i)} ${parameterTorch.storage().array()(i)}")
      }
    }

    val (weights, grad) = model.getParameters()
    val criterion = BCECriterion[Float]()

    val state = T("learningRate" -> 1e-2)

    val sgd = new SGD[Float]

    def feval(x: Tensor[Float]): (Float, Tensor[Float]) = {
      model.forward(input)
      criterion.forward(model.output.asInstanceOf[Tensor[Float]], labels)
      model.zeroGradParameters()
      val gradOutputTest = criterion.backward(model.output.asInstanceOf[Tensor[Float]], labels)
      model.backward(input, gradOutputTest)
      (criterion.output, grad)
    }
    for (i <- 1 to 5) {
      sgd.optimize(feval, weights, state)
    }

    val output = TH.map("output").asInstanceOf[Tensor[Float]]
    val outputTest = model.output.toTensor[Float]
    var abss = 0.0
    for (i <- 0 until outputTest.nElement()) {
      val tmp = abs(outputTest.storage().array()(i) - output.storage().array()(i))
      abss += tmp
    }
    println(s"outputAbs:$abss")
    output should be (outputTest)
    assert(abss < 1e-2)


    val errTest = criterion.output
    val err = TH.map("err").asInstanceOf[Double]
    println(s"${abs(errTest - err)}")
//    assert(abs(errTest - err) < 1.5e-6)

    val gradOutputTest = criterion.backward(outputTest, labels)
    val gradOutput = TH.map("gradOutput").asInstanceOf[Tensor[Float]]
    abss = 0.0
    for (i <- 0 until gradOutputTest.nElement()) {
      val tmp = abs(gradOutputTest.storage().array()(i) - gradOutput.storage().array()(i))
      abss += tmp
    }

    gradOutput should be (gradOutputTest)
    assert(abss < 2e-6)
    println(s"gradOutputTestAbs:$abss")

    val gradInput = model.gradInput.asInstanceOf[Tensor[Float]]
    val gradInputTorch = TH.map("gradInput").asInstanceOf[Tensor[Float]]

    abss = 0.0
    for (i <- 0 until gradInputTorch.nElement()) {
      val tmp = abs(gradInputTorch.storage().array()(i) - gradInput.storage().array()(i))
      abss += tmp
    }
    println(s"gradInputTestAbs:$abss")
  }
}
