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

package com.intel.analytics.bigdl.nn.fixpoint

import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

@SerialVersionUID(- 8572055756810843156L)
class SpatialDilatedConvolution[T: ClassTag](
  nInputPlane: Int, // The number of expected input planes in the image given into forward()
  nOutputPlane: Int, // The number of output planes the convolution layer will produce.
  kernelW: Int, // The kernel width of the convolution
  kernelH: Int, // The kernel height of the convolution
  strideW: Int = 1, // The step of the convolution in the width dimension.
  strideH: Int = 1, // The step of the convolution in the height dimension
  padW: Int = 0, // The additional zeros added per width to the input planes.
  padH: Int = 0, // The additional zeros added per height to the input planes.
  val dilationW: Int = 1,
  val dilationH: Int = 1,
  nGroup: Int = 1 // Kernel group number
)(implicit ev: TensorNumeric[T]) extends SpatialConvolution[T](
  nInputPlane,
  nOutputPlane,
  kernelW,
  kernelH,
  strideW,
  strideH,
  padW,
  padH,
  nGroup
) {
  override val DILATION_WIDTH: Int = dilationW
  override val DILATION_HEIGHT: Int = dilationH

  override def toString(): String = {
    s"fixpoint.SpatialDilatedConvolution($nInputPlane -> $nOutputPlane, $kernelW x" +
      s" $kernelH, $strideW, $strideH, $padW, $padH, $dilationW, $dilationH)"
  }
}

object SpatialDilatedConvolution {
  def apply[@specialized(Float) T: ClassTag](
    nInputPlane: Int,
    nOutputPlane: Int,
    kW: Int,
    kH: Int,
    dW: Int = 1,
    dH: Int = 1,
    padW: Int = 0,
    padH: Int = 0,
    dilationW: Int = 1,
    dilationH: Int = 1
  )(implicit ev: TensorNumeric[T]) : SpatialDilatedConvolution[T] = {
    new SpatialDilatedConvolution[T](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH, dilationW, dilationH)
  }
}

object TestConv{

  def main(args: Array[String]): Unit = {
    import com.intel.analytics.bigdl.nn.{SpatialDilatedConvolution => NNSpatialConvolution}
    val test = TestCase(2, 1024, 19, 19, 1, 1024, 1, 1, 1, 1, 0, 0)

    val weight = Tensor[Float](test.group, test.outputChannel / test.group,
      test.inputChannel / test.group, test.kernelHeight, test.kernelWidth).fill(1.0f)
    for (i <- 0 until weight.nElement()) {
      weight.storage().array()(i) = i % 32
    }
    val bias = Tensor[Float](test.outputChannel).fill(0f)

    val nnConv = new NNSpatialConvolution[Float](test.inputChannel, test.outputChannel,
      test.kernelHeight, test.kernelWidth, test.strideHeight, test.strideWidth,
      test.padHeight, test.padWidth, 2, 2)
    nnConv.weight.copy(weight)
    nnConv.bias.copy(bias)

    val input = Tensor[Float]().resize(Array(test.batchSize, test.inputChannel,
      test.inputHeight, test.inputWidth)).fill(1.0f)
    for (i <- 0 until input.nElement()) {
      input.storage().array()(i) = i % 32
    }

    val quantizedConv = new SpatialDilatedConvolution[Float](test.inputChannel, test.outputChannel,
      test.kernelHeight, test.kernelWidth, test.strideHeight, test.strideWidth,
      test.padHeight, test.padWidth, 2, 2)

    nnConv.updateOutput(input)
    quantizedConv.initWeightAndBias(nnConv.weight, nnConv.bias)
    quantizedConv.updateOutput(input)

    Files.deleteIfExists(Paths.get("/tmp/quantizedConv"))
    quantizedConv.save("/tmp/quantizedConv")

    val tmp = Module.load("/tmp/quantizedConv").asInstanceOf[SpatialConvolution[Float]]
    println(tmp)
    tmp.updateOutput(input)

    Files.deleteIfExists(Paths.get("/tmp/nnConv"))
    nnConv.save("/tmp/nnConv")

    val newInput = Tensor[Float]().resize(Array(test.batchSize, test.inputChannel,
      test.inputHeight, test.inputWidth))
    for (i <- 0 until input.nElement()) {
      newInput.storage().array()(i) = i % 32
    }

    quantizedConv.updateOutput(newInput)
    nnConv.updateOutput(newInput)
    tmp.updateOutput(newInput)

    println(tmp.output.nElement())
    println(quantizedConv.output.nElement())
    println(nnConv.output.nElement())

    require(tmp.output.nElement() == quantizedConv.output.nElement(),
      s"elements number should be the same")

    for (i <- 0 until tmp.output.nElement()) {
      val ori = quantizedConv.output.storage().array()(i)
      val ser = tmp.output.storage().array()(i)

      require(Math.abs(ori - ser) < 0.1, s"values should be the same.")
    }

    quantizedConv.release()

  }

}
