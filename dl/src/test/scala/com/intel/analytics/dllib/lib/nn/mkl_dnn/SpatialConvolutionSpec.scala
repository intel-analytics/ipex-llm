package com.intel.webscaleml.nn.nn.mkl_dnn

import com.intel.webscaleml.nn.tensor.{Tensor, torch}
import org.scalatest.{FlatSpec, Matchers}

import scala.math._

class SpatialConvolutionSpec extends FlatSpec with Matchers {

  //TODO: need mkl-dnn library to run this unit test
//  "mkl-dnn SpatialConvolution" should "generate correct output" in {
//    val nInputPlane = 1
//    val nOutputPlane = 1
//    val kW = 2
//    val kH = 2
//    val dW = 1
//    val dH = 1
//    val padW = 0
//    val padH = 0
//    val layer = new SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
//
//    val inputData = Array(
//      1.0f, 2f, 3f,
//      4f, 5f, 6f,
//      7f, 8f, 9f
//    )
//
//    val kernelData = Array(
//      2.0f, 3f,
//      4f, 5f
//    )
//
//    val biasData = Array(0.5f)
//
//    layer.weight.copy(torch.Tensor[Float](torch.storage(kernelData), 1, Array(nOutputPlane, nInputPlane, kH, kW)))
//    layer.bias.copy(torch.Tensor[Float](torch.storage(biasData), 1, Array(nOutputPlane)))
//    val input = torch.Tensor[Float](torch.storage(inputData), 1, Array(1, 1, 3, 3))
//    val output = layer.updateOutput(input)
//    output(Array(1, 1, 1, 1)) should be(49.5)
//    output(Array(1, 1, 1, 2)) should be(63.5)
//    output(Array(1, 1, 2, 1)) should be(91.5)
//    output(Array(1, 1, 2, 2)) should be(105.5)
//  }
//
//  "A SpatialConvolution" should "generate correct gradInput" in {
//    val nInputPlane = 1
//    val nOutputPlane = 1
//    val kW = 2
//    val kH = 2
//    val dW = 1
//    val dH = 1
//    val padW = 0
//    val padH = 0
//    val layer = new SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
//
//    val inputData = Array(
//      1.0f, 2, 3,
//      4, 5, 6,
//      7, 8, 9
//    )
//
//    val kernelData = Array(
//      2.0f, 3,
//      4, 5
//    )
//
//    val gradOutputData = Array(
//      1.0f, 2.0f,
//      3.0f, 4.0f
//    )
//
//    val biasData = Array(0.0f)
//
//    layer.weight.copy(torch.Tensor[Float](torch.storage(kernelData), 1, Array(nOutputPlane, nInputPlane, kH, kW)))
//    layer.bias.copy(torch.Tensor[Float](torch.storage(biasData), 1, Array(nOutputPlane)))
//    val input = torch.Tensor[Float](torch.storage(inputData), 1, Array(1, 1, 3, 3))
//    val output = layer.updateOutput(input)
//    println(output)
//    val gradOutput = torch.Tensor[Float](torch.storage(gradOutputData), 1, Array(1, 1, 2, 2))
//    val gradInput = layer.backward(input, gradOutput)
//    gradInput(Array(1, 1, 1, 1)) should be(2)
//    gradInput(Array(1, 1, 1, 2)) should be(7)
//    gradInput(Array(1, 1, 1, 3)) should be(6)
//    gradInput(Array(1, 1, 2, 1)) should be(10)
//    gradInput(Array(1, 1, 2, 2)) should be(30)
//    gradInput(Array(1, 1, 2, 3)) should be(22)
//    gradInput(Array(1, 1, 3, 1)) should be(12)
//    gradInput(Array(1, 1, 3, 2)) should be(31)
//    gradInput(Array(1, 1, 3, 3)) should be(20)
//  }
}
