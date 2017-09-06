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

package com.intel.analytics.bigdl.nn.quantized

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.bigquant.BigQuant
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.nn.quantized.Utils.ANode
import com.intel.analytics.bigdl.nn.{Linear => NNLinear, SpatialConvolution => NNConv, SpatialDilatedConvolution => NNDilatedConv, _}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.utils.T
import org.apache.log4j.Logger
import org.scalatest.{FlatSpec, Matchers}

class QuantizableSpec extends FlatSpec with Matchers {
  val logger: Logger = Logger.getLogger(getClass)

  "Sequential LeNet5" should "work correctly" in {
    val seq = LeNet5(10)
    seq.getParameters()._1.fill(1)

    val input = Tensor(4, 28, 28).fill(1)
    val output = seq.forward(input).toTensor

    val quantSeq = seq.quantize()
    val quantOutput = quantSeq.forward(input).toTensor

    output should be (quantOutput)
  }

  "Quantize sequential LeNet5 twice" should "work correctly" in {
    val seq = LeNet5(10)

    val input = Tensor(4, 28, 28)

    val quantSeq1 = seq.quantize()
    val quantOutput1 = quantSeq1.forward(input).toTensor

    val quantSeq2 = seq.quantize()
    val quantOutput2 = quantSeq2.forward(input).toTensor

    quantOutput1 should be (quantOutput2)
  }

  "Graph LeNet5" should "work correctly" in {
    val graph = LeNet5.graph(10)
    graph.getParameters()._1.fill(1)

    val input = Tensor(4, 28, 28).fill(1)
    val output = graph.forward(input).toTensor

    val quantGraph = graph.quantize()
    val quantOutput = quantGraph.forward(input).toTensor

    output should be (quantOutput)
  }

  "Quantize graph LeNet5 twice" should "work correctly" in {
    val graph = LeNet5.graph(10)

    val input = Tensor(4, 28, 28)

    val quantGraph1 = graph.quantize()
    val quantOutput1 = quantGraph1.forward(input).toTensor

    val quantGraph2 = graph.quantize()
    val quantOutput2 = quantGraph2.forward(input).toTensor

    quantOutput1 should be (quantOutput2)
  }

  "load quantized graph model" should "work correctly" in {
    val input = Input()
    val linear1 = NNLinear(3, 4).inputs(input)
    val linear2 = NNLinear(3, 4).inputs(input)
    val output = CAddTable().inputs(linear1, linear2)
    val graph = Graph(Array(input), Array(output))
    graph.getParameters()._1.fill(1)

    val in = Tensor(3, 3).fill(1)
    val out = graph.forward(in).toTensor

    val quantModel = graph.quantize()
    val quantOut = quantModel.forward(in).toTensor

    out should be (quantOut)
  }

  "quantize a quantized linear" should "work correctly" in {
    case class TestCase(batchSize: Int, inputSize: Int, outputSize: Int)
    val test = TestCase(1, 1, 1)

    val weight = Tensor(test.outputSize, test.inputSize).fill(1.0f)
    val bias = Tensor(test.outputSize).fill(0f)
    val input = Tensor(test.batchSize, test.inputSize).fill(1.0f)

    val linear = Linear[Float](test.inputSize, test.outputSize, initWeight = weight,
      initBias = bias)

    val linear2 = linear.quantize()

    linear.updateOutput(input)
    linear2.updateOutput(input)

    linear.output shouldEqual linear2.output
  }

  "quantize a quantized SpatialConvolution" should "work correctly" in {
    case class TestCase(batchSize: Int, inputChannel: Int, inputHeight: Int, inputWidth: Int,
      group: Int, outputChannel: Int, kernelHeight: Int, kernelWidth: Int,
      strideHeight: Int, strideWidth: Int, padHeight: Int, padWidth: Int)
    val test = TestCase(1, 1, 3, 3, 1, 1, 2, 2, 1, 1, 0, 0)

    val weight = Tensor(test.group, test.outputChannel / test.group,
      test.inputChannel / test.group, test.kernelHeight, test.kernelWidth).fill(1.0f)
    val bias = Tensor(test.outputChannel).fill(0f)
    val input = Tensor().resize(Array(test.batchSize, test.inputChannel,
      test.inputHeight, test.inputWidth)).fill(1.0f)

    val conv = SpatialConvolution(test.inputChannel, test.outputChannel,
      test.kernelHeight, test.kernelWidth, test.strideHeight, test.strideWidth,
      test.padHeight, test.padWidth, test.group, weight, bias)

    val conv2 = conv.quantize()

    conv.updateOutput(input)
    conv2.updateOutput(input)

    conv.output shouldEqual conv2.output
  }

  "Quantize a cell" should "work correctly" in {
    val batchSize = 2
    val inputSize = 4
    val hiddenSize = 4

    val initWeight = Tensor(hiddenSize, hiddenSize).fill(1)
    val initBias = Tensor(hiddenSize).fill(0)

    val cell = RnnCell(inputSize, hiddenSize, Tanh())
    val cellGraph = cell.cell.asInstanceOf[Graph[Float]]
    cellGraph.inputs(1).element.isInstanceOf[NNLinear[Float]] should be (true)
    val linear = cellGraph.inputs(1).element.asInstanceOf[NNLinear[Float]]
    linear.weight.copy(initWeight)
    linear.bias.copy(initBias)

    val input = Tensor(batchSize, inputSize).rand()
    val hidden = Tensor(batchSize, hiddenSize).fill(1)

    cell.forward(T(input, hidden))

    val quantCell = cell.quantize().asInstanceOf[RnnCell[Float]]
    val quantCellGraph = quantCell.cell.asInstanceOf[Graph[Float]]
    quantCellGraph.inputs(1).element.isInstanceOf[Linear[Float]] should be (true)
    quantCell.forward(T(input, hidden))

    equalWithPrecision(cell.output(1), quantCell.output(1), 10) should be (true)
    equalWithPrecision(cell.output(2), quantCell.output(2), 10) should be (true)
  }

  "Quantize a TimeDistributed" should "work correctly" in {
    val outputSize = 4
    val inputSize = 8
    val batchSize = 4
    val times = 1

    val initWeight = Tensor(4, 8).fill(1)
    val initBias = Tensor(4).fill(0)

    val tmd = TimeDistributed(NNLinear(inputSize, outputSize, initWeight = initWeight,
      initBias = initBias))
    tmd.layer.isInstanceOf[NNLinear[Float]] should be (true)
    val input = Tensor(batchSize, times, inputSize).fill(1)

    val output = tmd.forward(input).toTensor

    val quantTmd = tmd.quantize().asInstanceOf[TimeDistributed[Float]]
    quantTmd.layer.isInstanceOf[Linear[Float]] should be (true)
    val quantOutput = quantTmd.forward(input).toTensor

    output should be (quantOutput)
  }

  "Quantize a Recurrent" should "work correctly" in {
    val inputSize = 4
    val hiddenSize = 4
    val batchSize = 2
    val times = 4

    RNG.setSeed(100)
    val input = Tensor(batchSize, times, inputSize).fill(1)

    val cell = RnnCell(inputSize, hiddenSize, Tanh())
    val model = Sequential().add(Recurrent().add(cell))
    model.getParameters()._1.fill(1)
    val output = model.forward(input).toTensor

    val quantModel = model.quantize()
    val quantOut = quantModel.forward(input).toTensor

    val quantCell = findCell(findModule(quantModel, 0))
    isQuantizedLinear(quantCell.cell.asInstanceOf[Graph[Float]].inputs(1).element) should be (true)

    output should be (quantOut)
  }

  "Quantize a BiRecurrent" should "work correctly" in {
    val inputSize = 4
    val hiddenSize = 4
    val batchSize = 2
    val times = 2

    val input = Tensor(batchSize, times, inputSize).fill(1)

    val cell = RnnCell(inputSize, hiddenSize, Sigmoid())
    val model = Sequential().add(BiRecurrent().add(cell))
    model.getParameters()._1.fill(1)

    val output = model.forward(input).toTensor

    val quantModel = model.quantize()
    val quantOut = quantModel.forward(input).toTensor

    output should be (quantOut)

    val bireccurent = findModule(quantModel, 0).asInstanceOf[BiRecurrent[Float]]
    val layer = bireccurent.layer
    val revLayer = bireccurent.revLayer

    def check(recurrent: Recurrent[Float]): Unit = {
      val cell = findCell(recurrent)
      isQuantizedLinear(cell.cell.asInstanceOf[Graph[Float]].inputs(1).element) should be (true)
    }

    check(layer)
    check(revLayer)

    equalWithPrecision(output, quantOut, 3) should be (true)
  }

  "Quantize a ConvLSTMPeephole" should "work correctly" in {
    val hiddenSize = 5
    val inputSize = 3
    val seqLength = 2
    val batchSize = 2
    val kernalW = 3
    val kernalH = 3

    val model = Sequential()
      .add(Recurrent()
        .add(ConvLSTMPeephole(
          inputSize,
          hiddenSize,
          kernalW, kernalH,
          withPeephole = false)))
    model.getParameters()._1.fill(1)

    val input = Tensor(batchSize, seqLength, inputSize, kernalW, kernalH).fill(1)
    val output = model.forward(input).toTensor

    val quantModel = model.quantize()
    val quantOut = quantModel.forward(input).toTensor

    // check modules
    val recurrent = findModule(quantModel, 0)
    val quantCell = findCell(recurrent).asInstanceOf[ConvLSTMPeephole[Float]]
    val inputGate = quantCell.inputGate
    val hiddenLayer = quantCell.hiddenLayer

    {
      // for input gate
      val index = if (quantCell.withPeephole) {
        0
      } else {
        1
      }
      val parallelTable = inputGate.modules(index).asInstanceOf[ParallelTable[Float]]
      val i2g = parallelTable.modules(0)
      val h2g = parallelTable.modules(1)

      val first = findModule(i2g, 1)
      val second = findModule(h2g, 1)

      isQuantizedConv(first) should be (true)
      isQuantizedConv(second) should be (true)
    }

    {
      // for hidden layer
      val parallelTable = hiddenLayer.modules(1).asInstanceOf[ParallelTable[Float]]
      val i2h = parallelTable.modules(0)
      val h2h = parallelTable.modules(1)

      val first = findModule(i2h, 1)
      val second = findModule(h2h, 1)

      isQuantizedConv(first) should be (true)
      isQuantizedConv(second) should be (true)
    }

    output should be (quantOut)
  }

  "Quantize a GRU" should "work correctly" in {
    val inputSize = 10
    val hiddenSize = 10
    val batchSize = 4
    val time = 20
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Float](batchSize, time, inputSize).fill(1)

    val model = Recurrent[Float]()
      .add(GRU[Float](inputSize, hiddenSize))
    model.getParameters()._1.fill(1)

    val output = model.forward(input).toTensor

    val quantModel = model.quantize()
    val quantOutput = quantModel.forward(input).toTensor

    val gru = findCell(quantModel).asInstanceOf[GRU[Float]]

    isQuantizedLinear(gru.h2g.element) should be (true)

    output should be (quantOutput)
  }

  "Quantize a LSTMPeephole" should "work correctly" in {
    val inputSize = 4
    val hiddenSize = 4
    val batchSize = 2
    val time = 10

    RNG.setSeed(100)
    val input = Tensor(batchSize, time, inputSize).fill(1f)
    val input2 = input.clone()

    val model = Recurrent()
      .add(LSTMPeephole(inputSize, hiddenSize))
    model.getParameters()._1.fill(1f)
    val output = model.forward(input)

    val quantModel = model.quantize()
    val quantOut = quantModel.forward(input2)

    def executions(model: Module[Float]): Array[ANode[Float]] = {
      val lstm = findCell(model).asInstanceOf[LSTMPeephole[Float]]
      findModule(lstm.cell, 1).asInstanceOf[Graph[Float]].getForwardExecutions
    }

    val numLinears = executions(model).count(_.element.isInstanceOf[NNLinear[Float]])
    val numQuantLinears = executions(quantModel).count(_.element.isInstanceOf[Linear[Float]])

    numLinears should be (numQuantLinears)

    output should be (quantOut)
  }

  "Quantize a LSTM" should "work correctly" in {
    val inputSize = 4
    val hiddenSize = 4
    val batchSize = 2
    val time = 10

    RNG.setSeed(100)
    val input = Tensor(batchSize, time, inputSize).fill(1f)
    val input2 = input.clone()

    val model = Recurrent()
      .add(LSTM(inputSize, hiddenSize))
    model.getParameters()._1.fill(1f)
    val output = model.forward(input)

    val quantModel = model.quantize()
    val quantOut = quantModel.forward(input2)

    def executions(model: Module[Float]): Array[ANode[Float]] = {
      val lstm = findCell(model).asInstanceOf[LSTM[Float]]
      findModule(lstm.cell, 1).asInstanceOf[Graph[Float]].getForwardExecutions
    }

    val numLinears = executions(model).count(_.element.isInstanceOf[NNLinear[Float]])
    val numQuantLinears = executions(quantModel).count(_.element.isInstanceOf[Linear[Float]])

    numLinears should be (numQuantLinears)
    output should be (quantOut)
  }

  "Linear perf" should "speed up" in {
    val inputSize = 1152
    val hiddenSize = 1152
    val time = 100
    val batchSize = 2

    val model = Sequential[Float]()
    for (i <- 1 to 9) {
      val flag = if (i == 1) false else true
      model.add(BiRecurrent[Float](isSplitInput = flag, merge = JoinTable[Float](2, 2))
        .add(RnnCell[Float](inputSize, hiddenSize, HardTanh[Float](0, 20, inplace = true))))
    }
    logger.info(model)
    val quantize = model.quantize()
    logger.info(quantize)

    val input = Tensor[Float](batchSize, time, inputSize).rand()
    (0 until 5).foreach { _ =>
      model.forward(input)
    }
    val start1 = System.nanoTime()
    (0 until 5).foreach { _ =>
      val s = System.nanoTime()
      model.forward(input)
      val e = System.nanoTime()
      logger.info((e - s) / 1e6)
    }
    val end1 = System.nanoTime()
    val eta1 = (end1 - start1) / 1e6

    (0 until 5).foreach { _ =>
      quantize.forward(input)
    }
    logger.info("*" * 80)
    val start2 = System.nanoTime()
    (0 until 5).foreach { _ =>
      val s = System.nanoTime()
      quantize.forward(input)
      val e = System.nanoTime()
      logger.info((e - s) / 1e6)
    }
    val end2 = System.nanoTime()
    val eta2 = (end2 - start2) / 1e6

    (eta1 > eta2) should be (true)
  }

  "JNI test" should "work correctly" in {
    BigQuant.printHello()
  }

  "Multi inputs" should "work correctly" in {
    val input1 = Input()
    val input2 = Input()
    val initWeight = Tensor(800, 200).fill(1)
    val initBias = Tensor(800).fill(0)
    val linear = NNLinear(200, 800, initWeight = initWeight, initBias = initBias).inputs(input1)
    val cadd = CAddTable().inputs(linear, input2)
    val out = cadd
    val graph = Graph(Array(input1, input2), Array(out))

    val t1 = Tensor(4, 200).fill(1)
    val t2 = Tensor(4, 800).rand()
    val input = T(t1, t2)

    graph.forward(input)

    val quantizedGraph = graph.quantize()
    logger.info(quantizedGraph)
    quantizedGraph.forward(input)

    graph.output.toTensor should be (quantizedGraph.output.toTensor)
  }

  private def equalWithPrecision(t1: Tensor[Float], t2: Tensor[Float], precision: Int): Boolean = {
    t1.nElement() should be (t2.nElement())

    var ret = true

    val t1Offset = t1.storageOffset() - 1
    val t2Offset = t2.storageOffset() - 1
    for (i <- 0 until t1.nElement()) {
      val a1 = trunc(t1.storage().array()(t1Offset + i), precision)
      val a2 = trunc(t2.storage().array()(t2Offset + i), precision)

      if (a1 != a2) {
        logger.info(a1 + "\t" + a2)
        ret = false
      }
    }

    ret
  }

  private def loss(t1: Tensor[Float], t2: Tensor[Float], precision: Double): Boolean = {
    t1.nElement() should be (t2.nElement())

    var ret = true

    val t1Offset = t1.storageOffset() - 1
    val t2Offset = t2.storageOffset() - 1
    for (i <- 0 until t1.nElement()) {
      val a1 = t1.storage().array()(t1Offset + i)
      val a2 = t2.storage().array()(t2Offset + i)

      val percent = Math.abs((a1 - a2) / a1)
      logger.info(a1 + "\t" + a2 + "\t" + percent)
      if (percent > precision) ret = false
    }

    ret
  }

  private def trunc(num: Float, precision: Int): Float = {
    val value = Math.pow(10, precision)
    ((num * value).toInt / value).toFloat
  }

  private def findCell(module: Module[Float]): Cell[Float] = {
    module.asInstanceOf[Recurrent[Float]].modules.last.asInstanceOf[Cell[Float]]
  }

  private def findModule(module: Module[Float], index: Int): Module[Float] = {
    module.asInstanceOf[Sequential[Float]].modules(index)
  }

  private def isQuantizedLinear(module: Module[Float]): Boolean = {
    module.isInstanceOf[Linear[Float]]
  }

  private def isQuantizedConv(module: Module[Float]): Boolean = {
    module.isInstanceOf[SpatialConvolution[Float]]
  }
}
