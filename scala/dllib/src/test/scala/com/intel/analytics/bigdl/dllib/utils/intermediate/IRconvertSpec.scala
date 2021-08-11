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
import com.intel.analytics.bigdl.nn.keras._
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
    val reshape = Node(IRElement("", IRGeneralModule[Float](nn.Reshape[Float](Array(50*4*4)))))
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

  def modelWithScale(format: DataFormat = DataFormat("NCHW")) : Module[Float] = {
    val convElement = nn.SpatialConvolution(1, 20, 5, 5, format = format)
    convElement.setInputDimMask(1, true)
    convElement.setWeightDimMask(2, true)
    convElement.setOutputDimMask(3, true)
    convElement.setInputScales(Array(Array(1, 2, 3)))
    convElement.setWeightScales(Array(Array(4, 5, 6)))
    val conv1 = convElement.setName("input").inputs()
    val pool1 = nn.SpatialMaxPooling(2, 2, 2, 2, format = format).setName("pool").inputs(conv1)
    val conv2 = nn.SpatialConvolution(20, 50, 5, 5, format = format).inputs(pool1)
    val pool2 = nn.SpatialMaxPooling(2, 2, 2, 2, format = format).inputs(conv2)
    val reshape = nn.Reshape(Array(50 * 4 * 4)).inputs(pool2)
    val fc = nn.Linear(50 * 4 * 4, 500).inputs(reshape)
    val relu = nn.ReLU().setName("relu1").inputs(fc)

    val linearElement = nn.Linear(500, 10)
    linearElement.setInputDimMask(1, true)
    linearElement.setOutputDimMask(2, true)
    linearElement.setInputScales(Array(Array(0, 1, 2)))
    linearElement.setOutputScales(Array(Array(7, 8, 9)))
    val fc2 = linearElement.setName("output").inputs(relu)
    val output = fc2
    Graph(conv1, output)
  }

  def keras(classNum: Int, shape: Shape = Shape(28, 28, 3)): nn.keras.Sequential[Float] = {
    import com.intel.analytics.bigdl.nn.keras._
    import com.intel.analytics.bigdl.utils.Shape

    val model = Sequential()
    model.add(Convolution2D(6, 5, 5, activation = "tanh",
      dimOrdering = "tf", inputShape = shape).setName("conv1_5x5"))
    model.add(BatchNormalization(dimOrdering = "tf")).setName("bnbn")
    model.add(MaxPooling2D(dimOrdering = "tf"))
    model.add(Convolution2D(12, 5, 5, activation = "tanh", dimOrdering = "tf")
      .setName("conv2_5x5"))
    model.add(BatchNormalization(dimOrdering = "tf")).setName("bnbn22")
    model.add(MaxPooling2D(dimOrdering = "tf"))
    model.add(Flatten())
    model.add(Dense(100, activation = "tanh").setName("fc1"))
    model.add(Dense(classNum, activation = "softmax").setName("fc2"))
    model
  }

  "Convert Blas with NHWC" should "be correct" in {
    System.setProperty("bigdl.engineType", "mkldnn")
    val input = Tensor[Float](2, 28, 28, 1).rand()
    val gradOutput = Tensor[Float](2, 10).rand()

    val blas = modelBlas(format = DataFormat("NHWC")).asInstanceOf[StaticGraph[Float]]
    blas.setInputFormats(Seq(Memory.Format.nhwc))
    blas.setOutputFormats(Seq(Memory.Format.nc))
    val dnn = blas.cloneModule().toIRgraph()

    val outBlas = blas.forward(input)
    val outDnn = dnn.forward(input)
    Equivalent.nearequals(outDnn.toTensor, outBlas.toTensor, 1e-4) should be (true)

    val gradInputBlas = blas.backward(input, gradOutput)
    val gradInputDnn = dnn.backward(input, gradOutput).toTensor[Float]
    Equivalent.nearequals(gradInputDnn.toTensor, gradInputBlas.toTensor, 1e-4) should be (true)
    System.clearProperty("bigdl.engineType")
  }

  // todo: use those unit tests after supporting keras conversion

//  "Running keras with mkldnn" should "be correct"in {
//    System.setProperty("bigdl.engineType", "mkldnn")
//    val model = keras(classNum = 10)
//
//    val blas = model.toGraph().asInstanceOf[StaticGraph[Float]]
//    blas.setInputFormats(Seq(Memory.Format.nhwc))
//    blas.setOutputFormats(Seq(Memory.Format.nc))
//
//    val dnn = blas.cloneModule().asInstanceOf[StaticGraph[Float]].toIRgraph()
//
//    val input = Tensor[Float](2, 28, 28, 3).rand()
//
//    val out1 = blas.forward(input)
//    val out2 = dnn.forward(input)
//    Equivalent.nearequals(out1.toTensor[Float], out2.toTensor[Float], 1e-5) should be (true)
//
//    val gradOutput = Tensor[Float]().resizeAs(out1.toTensor[Float])
//
//    val gradInputBlas = blas.backward(input, gradOutput)
//    val gradInputDnn = dnn.backward(input, gradOutput)
//    Equivalent.nearequals(gradInputDnn.toTensor, gradInputBlas.toTensor, 1e-4) should be (true)
//    System.clearProperty("bigdl.engineType")
//  }
//
//  "KSequential to IRGraph" should "work" in {
//    System.setProperty("bigdl.engineType", "mkldnn")
//
//    RandomGenerator.RNG.setSeed(10)
//    import com.intel.analytics.bigdl.mkl.Memory
//
//    val seq = nn.keras.Sequential[Float]()
//    seq.add(InputLayer(inputShape = Shape(20, 100)))
//    seq.add(Convolution1D(10, 5, activation = "relu"))
//    seq.add(GlobalMaxPooling1D())
//    seq.add(Dense(128))
//    // seq.add(KDropout(0.2))
//    seq.add(Activation("relu"))
//    seq.add(Dense(10, activation = "softmax"))
//
//    // For such cases, toSingleGraph() is unnecessary
//    val graph = seq.toGraph().asInstanceOf[StaticGraph[Float]]
//    graph.asInstanceOf[StaticGraph[Float]].setInputFormats(Seq(Memory.Format.ntc))
//    graph.asInstanceOf[StaticGraph[Float]].setOutputFormats(Seq(Memory.Format.nc))
//    // set gradWeight
//    graph.getParameters()._2.rand()
//
//    val ir = graph.asInstanceOf[StaticGraph[Float]].cloneModule().toIRgraph()
//
//    val tensor = Tensor[Float](Array(3, 20, 100)).rand()
//    val outputBlas = graph.forward(tensor)
//    val output = ir.forward(tensor)
//    outputBlas should be(output)
//
//    val gradOutput = Tensor[Float]().resizeAs(outputBlas.toTensor[Float]).rand()
//    val gradInputBlas = graph.backward(tensor, gradOutput)
//    val gradInput = ir.backward(tensor, gradOutput)
//
//    Equivalent.nearequals(gradInput.toTensor[Float],
//      gradInputBlas.toTensor[Float], 1e-5) should be(true)
//
//    System.clearProperty("bigdl.engineType")
//  }
//
//  "KGraph to IRGraph" should "work" in {
//    System.setProperty("bigdl.engineType", "mkldnn")
//    RandomGenerator.RNG.setSeed(10)
//
//    import com.intel.analytics.bigdl.mkl.Memory
//    val input = nn.keras.Input[Float](inputShape = Shape(10))
//    val d = nn.keras.Dense[Float](20, activation = "relu").setName("dense1").inputs(input)
//    val d2 = nn.keras.Dense[Float](5).setName("dense2").inputs(d)
//    val model = nn.keras.Model[Float](input, d2)
//
//    val graph = model.toGraph().asInstanceOf[StaticGraph[Float]]
//    graph.asInstanceOf[StaticGraph[Float]].setInputFormats(Seq(Memory.Format.nc))
//    graph.asInstanceOf[StaticGraph[Float]].setOutputFormats(Seq(Memory.Format.nc))
//
//    // set gradWeight
//    graph.getParameters()._2.rand()
//
//    // graph.evaluate()
//    val ir = graph.asInstanceOf[StaticGraph[Float]].cloneModule().toIRgraph()
//    val tensor = Tensor[Float](Array(3, 10)).rand()
//
//    val outputBlas = graph.forward(tensor)
//    val output = ir.forward(tensor)
//
//    val gradOutput = Tensor[Float]().resizeAs(outputBlas.toTensor[Float]).rand()
//    val gradInputBlas = graph.backward(tensor, gradOutput)
//    val gradInput = ir.backward(tensor, gradOutput)
//
//    outputBlas should be(output)
//    gradInputBlas should be(gradInput)
//
//    System.clearProperty("bigdl.engineType")
//  }

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

  "Convert Blas with scale to Dnn" should "be correct" in {
    System.setProperty("bigdl.engineType", "mkldnn")
    val input = Tensor[Float](2, 1, 28, 28).rand()
    val gradOutput = Tensor[Float](2, 10).rand()

    val blasModel = modelWithScale().asInstanceOf[StaticGraph[Float]]
    val irModel = blasModel.cloneModule().toIRgraph()

    val blasExecutions = blasModel.getSortedForwardExecutions()
    val irExecutions = irModel.graph.getSortedForwardExecutions()

    val blasInputs = blasExecutions.filter(_.element.getName() == "input")(0)
      .element.asInstanceOf[MklInt8Convertible]
    val blasOutputs = blasExecutions.filter(_.element.getName() == "output")(0)
      .element.asInstanceOf[MklInt8Convertible]

    val inputs = irExecutions.filter(_.element.getName() == "input")(0)
      .element.asInstanceOf[MklInt8Convertible]
    val outputs = irExecutions.filter(_.element.getName() == "output")(0)
      .element.asInstanceOf[MklInt8Convertible]

    blasInputs.getWeightDimMask() should be(inputs.getWeightDimMask())
    blasInputs.getInputDimMask() should be(inputs.getInputDimMask())
    blasInputs.getOutputDimMask() should be(inputs.getOutputDimMask())

    blasInputs.getWeightScales() should be(inputs.getWeightScales())
    blasInputs.getInputScales() should be(inputs.getInputScales())
    blasInputs.getOutputScales() should be(inputs.getOutputScales())

    val outBlas = blasModel.forward(input).toTensor[Float]
    val gradInputBlas = blasModel.backward(input, gradOutput)
    val outDnn = irModel.forward(input).toTensor[Float]
    val gradInputDnn = irModel.backward(input, gradOutput).toTensor[Float]

    Equivalent.nearequals(outDnn, outBlas, 1e-4) should be (true)
    Equivalent.nearequals(gradInputDnn.toTensor, gradInputBlas.toTensor, 1e-4) should be (true)

    System.clearProperty("bigdl.engineType")
  }

  "convert blas gap to dnn" should "work correctly" in {
    System.setProperty("bigdl.engineType", "mkldnn")
    val graph = nn.Sequential()
      .add(SpatialAveragePooling[Float](2, 2, globalPooling = true))
      .toGraph()

    graph.asInstanceOf[StaticGraph[Float]].setOutputFormats(Seq(Memory.Format.nchw))
    val dnn = ConversionUtils.convert(graph.cloneModule())

    graph.evaluate()
    dnn.evaluate()

    val input = Tensor[Float](4, 2, 3, 3).rand(-1, 1)

    graph.forward(input)
    dnn.forward(input)

    graph.output should be (dnn.output)

    dnn.release()
    System.clearProperty("bigdl.engineType")
  }

  "dnn resnet50" should "work correctly with dnn computing thread" in {
    System.setProperty("bigdl.engineType", "mkldnn")
    val batchSize = 2

    RandomGenerator.RNG.setSeed(1)
    val dnnThread = mkldnn.ResNet.graph(batchSize, classNum = 1000,
      T("depth" -> 50, "optnet" -> false, "dataSet" -> mkldnn.ResNet.DatasetType.ImageNet))
    Engine.dnnComputing.invokeAndWait2(Array(0).map(_ => () => {
      dnnThread.compile(mkldnn.Phase.TrainingPhase)
    }))

    RandomGenerator.RNG.setSeed(1)
    val dnn = mkldnn.ResNet.graph(batchSize, classNum = 1000,
      T("depth" -> 50, "optnet" -> false, "dataSet" -> mkldnn.ResNet.DatasetType.ImageNet))
    dnn.compile(mkldnn.Phase.TrainingPhase)

    RandomGenerator.RNG.setSeed(100)
    val in = Tensor[Float](batchSize, 3, 224, 224).rand(-1, 1)

    val out = dnn.forward(in).toTensor[Float]
    Engine.dnnComputing.invokeAndWait2(Array(0).map(_ => () => {
      dnnThread.forward(in).toTensor[Float]
    }))
    val outThread = dnnThread.output.toTensor[Float]

    val gradOutput = out.clone()
    val grad = dnn.backward(in, gradOutput).toTensor[Float]
    Engine.dnnComputing.invokeAndWait2(Array(0).map(_ => () => {
      dnnThread.backward(in, gradOutput).toTensor[Float]
    }))
    val gradThread = dnnThread.gradInput.toTensor[Float]

    Equivalent.nearequals(out, outThread) should be(true)
    Equivalent.nearequals(grad, gradThread) should be(true)
  }
}
