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
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.quantized.Utils.ANode
import com.intel.analytics.bigdl.nn.{Linear => NNLinear, SpatialConvolution => NNConv, SpatialDilatedConvolution => NNDilatedConv, _}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.utils.{Engine, MklBlas, T, Table}
import org.apache.log4j.Logger
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class QuantizableSpec extends FlatSpec with Matchers with BeforeAndAfter {
  before {
    System.setProperty("bigdl.engineType", "mklblas")
    Engine.setEngineType(MklBlas)
  }

  after {
    System.clearProperty("bigdl.engineType")
  }

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
