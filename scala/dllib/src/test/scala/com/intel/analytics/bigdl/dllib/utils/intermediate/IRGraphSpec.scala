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

package com.intel.analytics.bigdl.utils.intermediate

import com.intel.analytics.bigdl.example.languagemodel.PTBModel
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, DataFormat}
import com.intel.analytics.bigdl.nn.mkldnn._
import com.intel.analytics.bigdl.{Module, nn, utils}
import com.intel.analytics.bigdl.nn.{Graph, Reshape, StaticGraph, TimeDistributed}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils._

class IRGraphSpec extends BigDLSpecHelper {
  def modelIR(inputFormats: Int = Memory.Format.nchw,
              outputFormats: Int = Memory.Format.nchw): IRGraph[Float] = {
    val conv1 = Node(IRElement[Float]("", IRSpatialConvolution[Float](1, 20, 5, 5)))
    val bn1 = Node(IRElement[Float]("", IRSpatialBatchNormalization[Float](20)))
    val weightsAndBias = Tensor[Float](2 * 20).rand()
    bn1.element.setWeights(weightsAndBias)

    val pool1 = Node(IRElement[Float]("", IRSpatialMaxPooling[Float](2, 2, 2, 2)))
    val conv2 = Node(IRElement[Float]("", IRSpatialConvolution[Float](20, 50, 5, 5)))
    val pool2 = Node(IRElement[Float]("", IRSpatialMaxPooling[Float](2, 2, 2, 2)))

    conv1 -> pool1 -> conv2 -> pool2
    val output = pool2
    IRGraph(Array(conv1), Array(output), inputFormats = inputFormats, outputFormats = outputFormats)
  }

  def modelIR2(inputFormats: Int = Memory.Format.nchw,
               outputFormats: Int = Memory.Format.nc): IRGraph[Float] = {
    val conv1 = Node(IRElement[Float]("", IRSpatialConvolution[Float](1, 20, 5, 5)))
    val pool1 = Node(IRElement[Float]("", IRSpatialMaxPooling[Float](2, 2, 2, 2)))
    val conv2 = Node(IRElement[Float]("", IRSpatialConvolution[Float](20, 50, 5, 5)))
    val pool2 = Node(IRElement[Float]("", IRSpatialMaxPooling[Float](2, 2, 2, 2)))
    val reshape = Node(IRElement("", IRGeneralModule(Reshape[Float](Array(50*4*4)))))
    val linear = Node(IRElement("", IRLinear[Float](50 * 4 * 4, 500)))
    val relu = Node(IRElement("", IRReLU[Float]()))
    val fc2 = Node(IRElement("", IRLinear[Float](500, 10)))

    conv1 -> pool1 -> conv2 -> pool2 ->
      reshape -> linear -> relu -> fc2
    val output = fc2

    IRGraph(Array(conv1), Array(output), inputFormats = inputFormats, outputFormats = outputFormats)
  }

  def modelIR3(inputFormats: Int = Memory.Format.nchw,
               outputFormats: Int = Memory.Format.nc): IRGraph[Float] = {
    val conv1 = Node(IRElement[Float]("", IRSpatialConvolution[Float](1, 20, 5, 5)))
    val pool1 = Node(IRElement[Float]("", IRSpatialMaxPooling[Float](2, 2, 2, 2)))
    val conv2 = Node(IRElement[Float]("", IRSpatialConvolution[Float](20, 50, 5, 5)))
    val pool2 = Node(IRElement[Float]("", IRSpatialMaxPooling[Float](2, 2, 2, 2)))
    val reshape = Node(IRElement("", IRGeneralModule(Reshape[Float](Array(50*4*4)))))
    val linear = Node(IRElement("", IRLinear[Float](50 * 4 * 4, 500)))
    val relu = Node(IRElement("", IRReLU[Float]()))
    val fc2 = Node(IRElement("", IRLinear[Float](500, 10)))

    val identity = Node(IRElement("", IRIdentity[Float]()))
    val join = Node(IRElement("", IRJoinTable[Float](2)))

    conv1 -> pool1 -> conv2 -> pool2 ->
      reshape -> linear -> relu -> fc2

    fc2 -> join
    identity -> join

    new IRGraph(Array(conv1, identity), Array(join),
      inputFormats = Seq(Memory.Format.nchw, Memory.Format.nc),
      outputFormats = Seq(Memory.Format.nc))
  }

  "Convert IRgraph to Dnn or Blas Graph" should "be correct" in {
    RandomGenerator.RNG.setSeed(1000)
    val input = Tensor[Float](2, 1, 28, 28).rand(-1, 1)
    val gradOutput = Tensor[Float](2, 50, 4, 4).rand(-1, 1)

    RandomGenerator.RNG.setSeed(1000)
    utils.Engine.setEngineType(MklBlas)
    val irBlas = modelIR()
    irBlas.build()
    val outBlas = irBlas.forward(input)
    val gradInputBlas = irBlas.backward(input, gradOutput)

    RandomGenerator.RNG.setSeed(1000)
    utils.Engine.setEngineType(MklDnn)
    val irDnn = modelIR()
    irDnn.build()
    val outDnn = irDnn.forward(input)
    val gradInputDnn = irDnn.backward(input, gradOutput).toTensor[Float]

    Equivalent.nearequals(outDnn.toTensor, outBlas.toTensor, 1e-4) should be (true)
    Equivalent.nearequals(gradInputDnn.toTensor, gradInputBlas.toTensor, 1e-4) should be (true)
  }

  "Convert IRgraph to Dnn or Blas Graph with 2 dimentions output" should "be correct" in {
    RandomGenerator.RNG.setSeed(1000)
    val input = Tensor[Float](2, 1, 28, 28).rand(-1, 1)
    val gradOutput = Tensor[Float](2, 10).rand(-1, 1)
    RandomGenerator.RNG.setSeed(1000)
    utils.Engine.setEngineType(MklBlas)
    val irBlas = modelIR2()
    irBlas.build()
    val outBlas = irBlas.forward(input)
    val gradInputBlas = irBlas.backward(input, gradOutput)

    RandomGenerator.RNG.setSeed(1000)
    utils.Engine.setEngineType(MklDnn)
    val irDnn = modelIR2()
    irDnn.build()
    val outDnn = irDnn.forward(input)
    val gradInputDnn = irDnn.backward(input, gradOutput).toTensor[Float]

    Equivalent.nearequals(outDnn.toTensor, outBlas.toTensor, 1e-4) should be (true)
    Equivalent.nearequals(gradInputDnn.toTensor, gradInputBlas.toTensor, 1e-4) should be (true)
  }

  "Convert IRgraph with two inputs to Dnn or Blas Graph" should "be correct" in {
    RandomGenerator.RNG.setSeed(1000)
    val input = T(Tensor[Float](2, 1, 28, 28).rand(-1, 1), Tensor[Float](2, 4)
      .rand(-1, 1))
    val gradOutput = Tensor[Float](2, 14).rand(-1, 1)

    RandomGenerator.RNG.setSeed(1000)
    utils.Engine.setEngineType(MklBlas)
    val irBlas = modelIR3()
    irBlas.build()
    val outBlas = irBlas.forward(input)
    val gradInputBlas = irBlas.backward(input, gradOutput).asInstanceOf[Table]

    RandomGenerator.RNG.setSeed(1000)
    utils.Engine.setEngineType(MklDnn)
    val irDnn = modelIR3()
    irDnn.build()
    val outDnn = irDnn.forward(input)
    val gradInputDnn = irDnn.backward(input, gradOutput).toTable

    Equivalent.nearequals(outDnn.toTensor, outBlas.toTensor, 1e-4) should be (true)
    Equivalent.nearequals(gradInputDnn.get[Tensor[Float]](1).get,
      gradInputBlas.get[Tensor[Float]](1).get, 1e-4) should be (true)
    Equivalent.nearequals(gradInputDnn.get[Tensor[Float]](2).get,
      gradInputBlas.get[Tensor[Float]](2).get, 1e-4) should be (true)
  }

  "PTB LSTM model running with mkldnn" should "work correctly" in {
    Engine.init(1, 1, true)
    RandomGenerator.RNG.setSeed(1000)

    val vocabSize = 10001
    val hiddenSize = 256
    val numLayers = 1
    val batchSize = 8
    val seqLength = 16
    var i = 2

    Engine.setEngineType(MklBlas)
    val blas = PTBModel.lstm(
      inputSize = vocabSize,
      hiddenSize = hiddenSize,
      outputSize = vocabSize,
      numLayers = numLayers,
      keepProb = 1.0F)

    Engine.setEngineType(MklDnn)
    val dnn = blas.cloneModule().asInstanceOf[StaticGraph[Float]].toIRgraph()

    val input = Tensor[Float](batchSize, seqLength).apply1(n => {
      i += 1
      i
    })

    val outBlas = blas.forward(input).toTensor[Float]
    val outDnn = dnn.forward(input).toTensor[Float]

    Equivalent.nearequals(outBlas, outDnn, 1e-6) should be(true)


    val gradOutput = Tensor[Float](outBlas.size()).rand()

    val grad1 = blas.backward(input, gradOutput).toTensor[Float]
    val grad2 = dnn.backward(input, gradOutput).toTensor[Float]

    Equivalent.nearequals(grad1, grad2, 1e-6) should be(true)
  }

  "timedistributed with softmax" should "work correctly" in {
    Engine.setEngineType(MklBlas)
    val input = nn.Input[Float]()
    val softMax = nn.SoftMax[Float]()
    val timeDistri = nn.TimeDistributed[Float](softMax).inputs(input)
    val blas = nn.Graph(input, timeDistri).evaluate()

    Engine.setEngineType(MklDnn)
    val dnn = blas.cloneModule()
      .asInstanceOf[StaticGraph[Float]]
      .setInputFormats(Seq(Memory.Format.ntc))
      .setOutputFormats(Seq(Memory.Format.ntc))
      .toIRgraph()
      .evaluate()

    val data = Tensor[Float](2, 255, 21)

    val outBlas = blas.forward(data).toTensor[Float]
    val outDnn = dnn.forward(data).toTensor[Float]

    Equivalent.nearequals(outBlas, outDnn, 1e-6) should be (true)
  }

  "convert softmax" should "work correctly" in {
    Engine.setEngineType(MklBlas)
    val input = nn.Input[Float]()
    val softMax = nn.SoftMax[Float]().inputs(input)
    val blas = nn.Graph(input, softMax).evaluate()

    Engine.setEngineType(MklDnn)
    val dnn = blas.cloneModule()
      .asInstanceOf[StaticGraph[Float]]
      .setInputFormats(Seq(Memory.Format.nc))
      .setOutputFormats(Seq(Memory.Format.nc))
      .toIRgraph()
      .evaluate()

    val data = Tensor[Float](255, 21)

    val outBlas = blas.forward(data).toTensor[Float]
    val outDnn = dnn.forward(data).toTensor[Float]

    Equivalent.nearequals(outBlas, outDnn, 1e-6) should be (true)
  }
}
