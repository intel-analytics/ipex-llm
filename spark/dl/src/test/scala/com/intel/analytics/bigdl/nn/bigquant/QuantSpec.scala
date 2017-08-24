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

package com.intel.analytics.bigdl.nn.bigquant

import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.bigquant.Quant._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

class QuantSpec extends FlatSpec with Matchers {
  "Quantize a number with same sign max and min" should "generate correct output" in {
    val src = 0.6f
    val (max, min) = (0.7f, 0.2f)

    val dst = quantize(src, max, min)
    dst should be (109)

    val result = dequantize(dst, max, min)
    result should be (0.6f +- 0.01f)

    val los = Math.abs(result - src).toDouble
    los should be (0.toDouble +- 0.01) // here is an experiment result
  }

  "Quantize a number with different sign max and min" should "generate correct output" in {
    val src = -0.6f
    val (max, min) = (0.7f, -0.2f)

    val dst = quantize(src, max, min)
    dst should be (-109)

    val result = dequantize(dst, max, min)
    result should be (-0.6f +- 0.01f)

    val los = Math.abs(result - src).toDouble
    los should be (0.toDouble +- 0.01) // here is an experiment result
  }

  "Quantize a array" should "generate correct output" in {
    val src = Array[Float](0.6f, 0.4f, -0.3f, 0.2f, 0.1f)

    val dst = new Array[Byte](src.length)

    val (max, min) = quantize(src, 0, src.length, dst, 0)

    dst(0) should be (127)
    dst(1) should be (85)
    dst(2) should be (-63)
    dst(3) should be (42)
    dst(4) should be (21)

    val before = src.clone()
    for (i <- src.indices) {
      src(i) = 0f
    }

    dequantize(src, 0, src.length, dst, 0, max, min)

    src(0) should be (0.6f +- 0.01f)
    src(1) should be (0.4f +- 0.01f)
    src(2) should be (-0.3f +- 0.01f)
    src(3) should be (0.2f +- 0.01f)
    src(4) should be (0.1f +- 0.01f)

    val after = src.clone()

    val los = loss(before, after, 0, src.length)

    los should be (0.toDouble +- 0.01) // here is an experiment result
  }

  "Quantize a matrix" should "generate correct output" in {
    val src = Array(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.4f, 0.3f, 0.2f, 0.1f)
    val dst = new Array[Byte](src.length)

    val (max, min) = quantize(src, 0, src.length, dst, 0, Array(2, 5))

    for (i <- src.indices) {
      println(dst(i))
    }

    dst(0) should be (25)
    dst(1) should be (51)
    dst(2) should be (76)
    dst(3) should be (102)
    dst(4) should be (127)
    dst(5) should be (127)
    dst(6) should be (85)
    dst(7) should be (64)
    dst(8) should be (42)
    dst(9) should be (21)

    val before = src.clone()
    for (i <- src.indices) {
      src(i) = 0f
    }

    dequantize(src, 0, src.length, dst, 0, max, min, Array(2, 5))
    for (i <- src.indices) {
      println(src(i))
    }

    src(0) should be (0.1f +- 0.01f)
    src(1) should be (0.2f +- 0.01f)
    src(2) should be (0.3f +- 0.01f)
    src(3) should be (0.4f +- 0.01f)
    src(4) should be (0.5f +- 0.01f)
    src(5) should be (0.6f +- 0.01f)
    src(6) should be (0.4f +- 0.01f)
    src(7) should be (0.3f +- 0.01f)
    src(8) should be (0.2f +- 0.01f)
    src(9) should be (0.1f +- 0.01f)

    val after = src.clone()

    val los = loss(before, after, 0, src.length)
    los should be (0.toDouble +- 0.02) // here is an experiment result
  }

  "Quantize a 1-d tensor" should "generate correct output" in {
    val array = Array[Float](0.6f, 0.4f, -0.3f, 0.2f, 0.1f)
    val src = Tensor[Float](array, Array(5))

    val dst = new Array[Byte](src.nElement())

    val (max, min) = quantize(src, dst, 0)

    dst(0) should be (127)
    dst(1) should be (85)
    dst(2) should be (-63)
    dst(3) should be (42)
    dst(4) should be (21)

    val before = src.clone()
    src.apply1(_ => 0f)

    dequantize(src, dst, 0, max, min)

    src.valueAt(1) should be (0.6f +- 0.01f)
    src.valueAt(2) should be (0.4f +- 0.01f)
    src.valueAt(3) should be (-0.3f +- 0.01f)
    src.valueAt(4) should be (0.2f +- 0.01f)
    src.valueAt(5) should be (0.1f +- 0.01f)

    val after = src.clone()

    val los = loss(before.storage().array(), after.storage().array(), 0, src.nElement())
    los should be (0.toDouble +- 0.01) // here is an experiment result
  }

  "Quantize a 2-d tensor" should "generate correct output" in {
    val array = Array(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.4f, 0.3f, 0.2f, 0.1f)
    val src = Tensor[Float](array, Array(2, 5))

    val dst = new Array[Byte](src.nElement())

    val (max, min) = quantize(src, dst, 0)

    dst(0) should be (25)
    dst(1) should be (51)
    dst(2) should be (76)
    dst(3) should be (102)
    dst(4) should be (127)
    dst(5) should be (127)
    dst(6) should be (85)
    dst(7) should be (64)
    dst(8) should be (42)
    dst(9) should be (21)

    val before = src.clone()
    src.apply1(_ => 0f)

    dequantize(src, dst, 0, max, min)

    src.valueAt(1, 1) should be (0.1f +- 0.01f)
    src.valueAt(1, 2) should be (0.2f +- 0.01f)
    src.valueAt(1, 3) should be (0.3f +- 0.01f)
    src.valueAt(1, 4) should be (0.4f +- 0.01f)
    src.valueAt(1, 5) should be (0.5f +- 0.01f)
    src.valueAt(2, 1) should be (0.6f +- 0.01f)
    src.valueAt(2, 2) should be (0.4f +- 0.01f)
    src.valueAt(2, 3) should be (0.3f +- 0.01f)
    src.valueAt(2, 4) should be (0.2f +- 0.01f)
    src.valueAt(2, 5) should be (0.1f +- 0.01f)

    val after = src.clone()

    val los = loss(before.storage().array(), after.storage().array(), 0, src.nElement())
    los should be (0.toDouble +- 0.02) // here is an experiment result
  }

  "Quantize a 3-d tensor" should "generate correct output" in {
    val array = Array(
      0.1f, 0.2f, 0.3f,
      0.4f, 0.5f, 0.6f,

      -0.5f, 0.4f, 0.3f,
      0.2f, 0.1f, 0f
    )
    val src = Tensor[Float](array, Array(2, 2, 3))

    val dst = new Array[Byte](src.nElement())

    val (max, min) = quantize(src, dst, 0)

    dst(0) should be (21)
    dst(1) should be (42)
    dst(2) should be (64)
    dst(3) should be (85)
    dst(4) should be (106)
    dst(5) should be (127)
    dst(6) should be (-127)
    dst(7) should be (102)
    dst(8) should be (76)
    dst(9) should be (51)
    dst(10) should be (25)
    dst(11) should be (0)

    val before = src.clone()
    src.apply1(_ => 0f)

    dequantize(src, dst, 0, max, min)

    src.valueAt(1, 1, 1) should be (0.1f +- 0.01f)
    src.valueAt(1, 1, 2) should be (0.2f +- 0.01f)
    src.valueAt(1, 1, 3) should be (0.3f +- 0.01f)
    src.valueAt(1, 2, 1) should be (0.4f +- 0.01f)
    src.valueAt(1, 2, 2) should be (0.5f +- 0.01f)
    src.valueAt(1, 2, 3) should be (0.6f +- 0.01f)
    src.valueAt(2, 1, 1) should be (-0.5f +- 0.01f)
    src.valueAt(2, 1, 2) should be (0.4f +- 0.01f)
    src.valueAt(2, 1, 3) should be (0.3f +- 0.01f)
    src.valueAt(2, 2, 1) should be (0.2f +- 0.01f)
    src.valueAt(2, 2, 2) should be (0.1f +- 0.01f)
    src.valueAt(2, 2, 3) should be (0.0f +- 0.01f)

    val after = src.clone()

    val los = loss(before.storage().array(), after.storage().array(), 0, src.nElement())
    los should be (0.toDouble +- 0.02) // here is an experiment result
  }

  "load quantized graph model" should "work correctly" in {
    val input = Input()
    val linear1 = nn.Linear(3, 4).inputs(input)
    val linear2 = nn.Linear(3, 4).inputs(input)
    val output = nn.CAddTable().inputs(linear1, linear2)
    val graph = Graph(Array(input), Array(output))

    val in = Tensor(3, 3).randn()
    val out = graph.forward(in)

    val quantizedModel = Module.quantize(graph)
  }

  "quantize a quantized linear" should "work correctly" in {
    case class TestCase(batchSize: Int, inputSize: Int, outputSize: Int)
    val test = TestCase(1, 1, 1)

    val weight = Tensor(test.outputSize, test.inputSize).fill(1.0f)
    val bias = Tensor(test.outputSize).fill(0f)
    val input = Tensor(test.batchSize, test.inputSize).fill(1.0f)

    val linear = Linear[Float](test.inputSize, test.outputSize)
    linear.initWeightAndBias(weight, bias)

    val linear2 = Module.quantize(linear)

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
      test.padHeight, test.padWidth, test.group)
    conv.initWeightAndBias(weight, bias)

    val conv2 = Module.quantize(conv)

    conv.updateOutput(input)
    conv2.updateOutput(input)

    conv.output shouldEqual conv.output
  }

  "replace cell" should "work correctly" in {
    val cell = RnnCell[Float](4, 3, ReLU[Float]())
    cell.cell.asInstanceOf[Graph[Float]].executions(0).element
            .isInstanceOf[nn.Linear[Float]] should be (true)

    cell.cell = Module.quantize(cell.cell)

    cell.cell.asInstanceOf[Graph[Float]].executions(0).element
            .isInstanceOf[Linear[Float]] should be (true)
  }

  "linear perf" should "speed up" in {

    val inputSize = 1152
    val outputSize = 1152
    val hiddenSize = 1152
    val time = 100
    val batchSize = 4

    val model = Sequential[Float]()
    for (i <- 1 to 3) {
      model.add(TimeDistributed[Float](BatchNormalization[Float](outputSize)))
      model.add(BiRecurrent[Float]()
              .add(RnnCell[Float](inputSize, hiddenSize, ReLU[Float]())))
    }
    println(model)
    val quantize = Module.quantize(Module.quantize(model))
    println(quantize)

    val input = Tensor[Float](batchSize, time, inputSize).rand

    for (i <- 0 until 5) {
      model.updateOutput(input)
    }
    val start1 = System.nanoTime()
    for (i <- 0 until 10) {
      model.updateOutput(input)
    }
    val end1 = System.nanoTime()
    val eta1 = (end1 - start1) / 1e6

    for (i <- 0 until 5) {
      quantize.forward(input)
    }
    println("*" * 80)
    val start2 = System.nanoTime()
    for (i <- 0 until 10) {
      quantize.forward(input)
    }
    val end2 = System.nanoTime()
    val eta2 = (end2 - start2) / 1e6

    (eta1 > eta2) should be (true)
  }
}
