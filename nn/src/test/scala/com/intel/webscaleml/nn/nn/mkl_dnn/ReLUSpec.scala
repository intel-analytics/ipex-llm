package com.intel.webscaleml.nn.nn.mkl_dnn

import com.intel.webscaleml.nn.tensor.torch
import org.scalatest.FlatSpec

import scala.math.abs


class ReLUSpec extends FlatSpec{
//TODO: need mkl-dnn library to run this unit test
//  "A ReLU Module " should "generate correct output and grad" in {
//    val module = new ReLU[Float]()
//    val input = torch.Tensor[Float](1,2,2,2)
//    input(Array(1,1,1,1)) = -0.97008799016476f
//    input(Array(1,1,1,2)) = -0.89318234380335f
//    input(Array(1,1,2,1)) = -0.65073125436902f
//    input(Array(1,1,2,2)) = -0.35406025126576f
//    input(Array(1,2,1,1)) = -1.0360766677186f
//    input(Array(1,2,1,2)) = 1.173689913936f
//    input(Array(1,2,2,1)) = 1.6776262558997f
//    input(Array(1,2,2,2)) = -0.64814318157732f
//    val gradOutput = torch.Tensor[Float](1,2,2,2)
//    gradOutput(Array(1,1,1,1)) = 0.43442418193445f
//    gradOutput(Array(1,1,1,2)) = 0.97614445211366f
//    gradOutput(Array(1,1,2,1)) = 0.081252868985757f
//    gradOutput(Array(1,1,2,2)) = 0.24688877537847f
//    gradOutput(Array(1,2,1,1)) = 0.027903598966077f
//    gradOutput(Array(1,2,1,2)) = 0.0086153273005038f
//    gradOutput(Array(1,2,2,1)) = 0.053113180678338f
//    gradOutput(Array(1,2,2,2)) = 0.74842141871341f
//    val expectedOutput = torch.Tensor[Float](1,2,2,2)
//    expectedOutput(Array(1,1,1,1)) = 0f
//    expectedOutput(Array(1,1,1,2)) = 0f
//    expectedOutput(Array(1,1,2,1)) = 0f
//    expectedOutput(Array(1,1,2,2)) = 0f
//    expectedOutput(Array(1,2,1,1)) = 0f
//    expectedOutput(Array(1,2,1,2)) = 1.173689913936f
//    expectedOutput(Array(1,2,2,1)) = 1.6776262558997f
//    expectedOutput(Array(1,2,2,2)) = 0f
//    val expectedGrad = torch.Tensor[Float](1,2,2,2)
//    expectedGrad(Array(1,1,1,1)) = 0f
//    expectedGrad(Array(1,1,1,2)) = 0f
//    expectedGrad(Array(1,1,2,1)) = 0f
//    expectedGrad(Array(1,1,2,2)) = 0f
//    expectedGrad(Array(1,2,1,1)) = 0f
//    expectedGrad(Array(1,2,1,2)) = 0.0086153273005038f
//    expectedGrad(Array(1,2,2,1)) = 0.053113180678338f
//    expectedGrad(Array(1,2,2,2)) = 0f
//    val inputOrg = input.clone()
//    val gradOutputOrg = gradOutput.clone()
//    val output = module.forward(input)
//    val gradInput = module.backward(input,gradOutput)
//    expectedOutput.map(output,(v1,v2)=>{assert(abs(v1-v2)<1e-6);v1})
//    expectedGrad.map(gradInput,(v1,v2)=>{assert(abs(v1-v2)<1e-6);v1})
//    assert(input == inputOrg)
//    assert(gradOutput == gradOutputOrg)
//
//  }

}
