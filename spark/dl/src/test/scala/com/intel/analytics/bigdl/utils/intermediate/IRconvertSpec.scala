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
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
import com.intel.analytics.bigdl.nn.mkldnn.{DnnGraph, Equivalent, Input, Output}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.{Module, nn}

class IRconvertSpec extends BigDLSpecHelper {

  def modelIR(): Array[Node[IRElement[Float]]] = {
    val conv1 = Node(IRElement[Float]("input", IRSpatialConvolution[Float](1, 20, 5, 5)))
    val pool1 = Node(IRElement[Float]("", IRSpatialMaxPooling[Float](2, 2, 2, 2)))
    val conv2 = Node(IRElement[Float]("", IRSpatialConvolution[Float](20, 50, 5, 5)))
    val pool2 = Node(IRElement[Float]("output", IRSpatialMaxPooling[Float](2, 2, 2, 2)))

    conv1 -> pool1 -> conv2 -> pool2
    Array(conv1, pool1, conv2, pool2)
  }

  def modelIR2() : Array[Node[IRElement[Float]]] = {
    val conv1 = Node(IRElement[Float]("input", IRSpatialConvolution[Float](1, 20, 5, 5)))
    val pool1 = Node(IRElement[Float]("", IRSpatialMaxPooling[Float](2, 2, 2, 2)))
    val conv2 = Node(IRElement[Float]("", IRSpatialConvolution[Float](20, 50, 5, 5)))
    val pool2 = Node(IRElement[Float]("", IRSpatialMaxPooling[Float](2, 2, 2, 2)))
    val reshape = Node(IRElement("", IRGeneralModule[Float](Reshape[Float](Array(50*4*4)))))
    val linear = Node(IRElement("", IRLinear[Float](50 * 4 * 4, 500)))
    val relu = Node(IRElement("", IRReLU[Float]()))
    val fc2 = Node(IRElement("output", IRLinear[Float](500, 10)))

    conv1 -> pool1 -> conv2 -> pool2 ->
    reshape -> linear -> relu -> fc2
    Array(conv1, pool1, conv2, pool2, reshape, linear, relu, fc2)
  }

  def modelBlas(format: DataFormat = DataFormat("NCHW")) : Module[Float] = {
    val conv1 = nn.SpatialConvolution(1, 20, 5, 5, format = format).setName("input").inputs()
    val pool1 = nn.SpatialMaxPooling(2, 2, 2, 2, format = format).setName("pool").inputs(conv1)
    val conv2 = nn.SpatialConvolution(20, 50, 5, 5, format = format).inputs(pool1)
    val pool2 = nn.SpatialMaxPooling(2, 2, 2, 2, format = format).inputs(conv2)
    val reshape = nn.Reshape(Array(50 * 4 * 4)).inputs(pool2)
    val fc = nn.Linear(50 * 4 * 4, 500).inputs(reshape)
    val relu = nn.ReLU().setName("relu1").inputs(fc)
    val fc2 = nn.Linear(500, 10).setName("output").inputs(relu)
    val output = fc2
    Graph(conv1, output)
  }

  "Convert Blas with NCHW to Dnn" should "be correct" in {
    System.setProperty("bigdl.engineType", "mkldnn")
    val input = Tensor[Float](2, 1, 28, 28).rand()
    val gradOutput = Tensor[Float](2, 10).rand()

    val blas = modelBlas().asInstanceOf[StaticGraph[Float]]
    val allNodes = blas.getSortedForwardExecutions()
    require(BlasToIR[Float].convertingCheck(allNodes))
    val irNodes = BlasToIR[Float].convert(allNodes).map(_._2).toArray
    require(IRToDnn[Float].convertingCheck(irNodes))
    val dnnNodes = IRToDnn[Float].convert(irNodes).map(_._2).toArray

    val inputsNodes = dnnNodes.filter(_.element.getName() == "input")(0)
    val outputsNodes = dnnNodes.filter(_.element.getName() == "output")(0)

    val inputs = Input(Array(2, 1, 28, 28), Memory.Format.nchw).inputs()
    inputsNodes.from(inputs)
    val outputs = Output(Memory.Format.nc).inputs(outputsNodes)
    val dnn = DnnGraph(Array(inputs), Array(outputs))
    dnn.compile(TrainingPhase)

    val outBlas = blas.forward(input)
    val gradInputBlas = blas.backward(input, gradOutput)
    val outDnn = dnn.forward(input)
    val gradInputDnn = dnn.backward(input, gradOutput).toTensor[Float]

    Equivalent.nearequals(outDnn.toTensor, outBlas.toTensor, 1e-4) should be (true)
    Equivalent.nearequals(gradInputDnn.toTensor, gradInputBlas.toTensor, 1e-4) should be (true)

    val p1 = dnn.getParameters()
    val p2 = blas.getParameters()
    Equivalent.nearequals(p1._1, p1._1, 1e-4) should be (true)
    Equivalent.nearequals(p1._2, p1._2, 1e-4) should be (true)
  }

  "Convert IRgraph to Dnn or Blas Graph" should "be correct" in {
    val input = Tensor[Float](2, 1, 28, 28).rand()
    val gradOutput = Tensor[Float](2, 50, 4, 4).rand()

    val allNodes = modelIR()
    RandomGenerator.RNG.setSeed(1000)
    require(IRToBlas[Float].convertingCheck(allNodes))
    val blasNodes = IRToBlas[Float].convert(allNodes).map(_._2).toArray
    RandomGenerator.RNG.setSeed(1000)
    require(IRToDnn[Float].convertingCheck(allNodes))
    val dnnNodes = IRToDnn[Float].convert(allNodes).map(_._2).toArray

    val blas = Graph(blasNodes.filter(_.element.getName() == "input"),
      blasNodes.filter(_.element.getName() == "output"))

    val inputsNodes = dnnNodes.filter(_.element.getName() == "input")(0)
    val outputsNodes = dnnNodes.filter(_.element.getName() == "output")(0)

    val inputs = Input(Array(2, 1, 28, 28), Memory.Format.nchw).inputs()
    inputsNodes.from(inputs)
    val outputs = Output(Memory.Format.nchw).inputs(outputsNodes)
    val dnn = DnnGraph(Array(inputs), Array(outputs))
    dnn.compile(TrainingPhase)

    val outBlas = blas.forward(input)
    val gradInputBlas = blas.backward(input, gradOutput)
    val outDnn = dnn.forward(input)
    val gradInputDnn = dnn.backward(input, gradOutput).toTensor[Float]

    Equivalent.nearequals(outDnn.toTensor, outBlas.toTensor, 1e-4) should be (true)
    Equivalent.nearequals(gradInputDnn.toTensor, gradInputBlas.toTensor, 1e-4) should be (true)

    val p1 = dnn.getParameters()
    val p2 = blas.getParameters()

    Equivalent.nearequals(p1._1, p1._1, 1e-4) should be (true)
    Equivalent.nearequals(p1._2, p1._2, 1e-4) should be (true)
  }

  "Convert IRgraph to Dnn or Blas Graph with 2 dimentions output" should "be correct" in {
    val input = Tensor[Float](2, 1, 28, 28).rand()
    val gradOutput = Tensor[Float](2, 10).rand()

    val allNodes = modelIR2()
    RandomGenerator.RNG.setSeed(1000)
    require(IRToBlas[Float].convertingCheck(allNodes))
    val blasNodes = IRToBlas[Float].convert(allNodes).map(_._2).toArray
    RandomGenerator.RNG.setSeed(1000)
    require(IRToDnn[Float].convertingCheck(allNodes))
    val dnnNodes = IRToDnn[Float].convert(allNodes).map(_._2).toArray

    val blas = Graph(blasNodes.filter(_.element.getName() == "input"),
      blasNodes.filter(_.element.getName() == "output"))

    val inputsNodes = dnnNodes.filter(_.element.getName() == "input")(0)
    val outputsNodes = dnnNodes.filter(_.element.getName() == "output")(0)

    val inputs = Input(Array(2, 1, 28, 28), Memory.Format.nchw).inputs()
    inputsNodes.from(inputs)
    val outputs = Output(Memory.Format.nc).inputs(outputsNodes)
    val dnn = DnnGraph(Array(inputs), Array(outputs))
    dnn.compile(TrainingPhase)

    val outBlas = blas.forward(input)
    val gradInputBlas = blas.backward(input, gradOutput)
    val outDnn = dnn.forward(input)
    val gradInputDnn = dnn.backward(input, gradOutput).toTensor[Float]

    Equivalent.nearequals(outDnn.toTensor, outBlas.toTensor, 1e-4) should be (true)
    Equivalent.nearequals(gradInputDnn.toTensor, gradInputBlas.toTensor, 1e-4) should be (true)

    val p1 = dnn.getParameters()
    val p2 = blas.getParameters()

    Equivalent.nearequals(p1._1, p1._1, 1e-4) should be (true)
    Equivalent.nearequals(p1._2, p1._2, 1e-4) should be (true)
  }
}
