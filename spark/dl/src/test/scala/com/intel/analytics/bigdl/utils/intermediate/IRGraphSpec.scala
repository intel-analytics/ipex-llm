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

import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.nn.mkldnn.HeapData
import com.intel.analytics.bigdl.{Module, nn, utils}
import com.intel.analytics.bigdl.nn.{Graph, Reshape, StaticGraph}
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

    outDnn should be(outBlas)
    gradInputDnn should be(gradInputBlas)
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

    outDnn should be(outBlas)
    gradInputDnn should be(gradInputBlas)
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

    outDnn should be(outBlas)
    gradInputDnn.get[Tensor[Float]](1) should be(gradInputBlas.get[Tensor[Float]](1))
    gradInputDnn.get[Tensor[Float]](2) should be(gradInputBlas.get[Tensor[Float]](2))
  }
}
