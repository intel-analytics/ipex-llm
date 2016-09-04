package com.intel.analytics.dllib.lib.nn

import org.scalatest.FlatSpec
import scala.math.abs
import com.intel.analytics.dllib.lib.tensor.{torch, DenseTensor$}

class ReLUSpec extends FlatSpec{
  "A ReLU Module " should "generate correct output and grad" in {
    val module = new ReLU[Double]()
    val input = torch.Tensor[Double](2,2,2)
    input(Array(1,1,1)) = -0.97008799016476
    input(Array(1,1,2)) = -0.89318234380335
    input(Array(1,2,1)) = -0.65073125436902
    input(Array(1,2,2)) = -0.35406025126576
    input(Array(2,1,1)) = -1.0360766677186
    input(Array(2,1,2)) = 1.173689913936
    input(Array(2,2,1)) = 1.6776262558997
    input(Array(2,2,2)) = -0.64814318157732
    val gradOutput = torch.Tensor[Double](2,2,2)
    gradOutput(Array(1,1,1)) = 0.43442418193445
    gradOutput(Array(1,1,2)) = 0.97614445211366
    gradOutput(Array(1,2,1)) = 0.081252868985757
    gradOutput(Array(1,2,2)) = 0.24688877537847
    gradOutput(Array(2,1,1)) = 0.027903598966077
    gradOutput(Array(2,1,2)) = 0.0086153273005038
    gradOutput(Array(2,2,1)) = 0.053113180678338
    gradOutput(Array(2,2,2)) = 0.74842141871341
    val expectedOutput = torch.Tensor[Double](2,2,2)
    expectedOutput(Array(1,1,1)) = 0
    expectedOutput(Array(1,1,2)) = 0
    expectedOutput(Array(1,2,1)) = 0
    expectedOutput(Array(1,2,2)) = 0
    expectedOutput(Array(2,1,1)) = 0
    expectedOutput(Array(2,1,2)) = 1.173689913936
    expectedOutput(Array(2,2,1)) = 1.6776262558997
    expectedOutput(Array(2,2,2)) = 0
    val expectedGrad = torch.Tensor[Double](2,2,2)
    expectedGrad(Array(1,1,1)) = 0
    expectedGrad(Array(1,1,2)) = 0
    expectedGrad(Array(1,2,1)) = 0
    expectedGrad(Array(1,2,2)) = 0
    expectedGrad(Array(2,1,1)) = 0
    expectedGrad(Array(2,1,2)) = 0.0086153273005038
    expectedGrad(Array(2,2,1)) = 0.053113180678338
    expectedGrad(Array(2,2,2)) = 0
    val inputOrg = input.clone()
    val gradOutputOrg = gradOutput.clone()
    val output = module.forward(input)
    val gradInput = module.backward(input,gradOutput)
    expectedOutput.map(output,(v1,v2)=>{assert(abs(v1-v2)<1e-6);v1})
    expectedGrad.map(gradInput,(v1,v2)=>{assert(abs(v1-v2)<1e-6);v1})
    assert(input == inputOrg)
    assert(gradOutput == gradOutputOrg)

  }

  "A ReLU Module with inPlace = true" should "generate correct output and grad" in {
    val module = new ReLU[Double](true)
    val input = torch.Tensor[Double](2,2,2)
    input(Array(1,1,1)) = -0.97008799016476
    input(Array(1,1,2)) = -0.89318234380335
    input(Array(1,2,1)) = -0.65073125436902
    input(Array(1,2,2)) = -0.35406025126576
    input(Array(2,1,1)) = -1.0360766677186
    input(Array(2,1,2)) = 1.173689913936
    input(Array(2,2,1)) = 1.6776262558997
    input(Array(2,2,2)) = -0.64814318157732
    val gradOutput = torch.Tensor[Double](2,2,2)
    gradOutput(Array(1,1,1)) = 0.43442418193445
    gradOutput(Array(1,1,2)) = 0.97614445211366
    gradOutput(Array(1,2,1)) = 0.081252868985757
    gradOutput(Array(1,2,2)) = 0.24688877537847
    gradOutput(Array(2,1,1)) = 0.027903598966077
    gradOutput(Array(2,1,2)) = 0.0086153273005038
    gradOutput(Array(2,2,1)) = 0.053113180678338
    gradOutput(Array(2,2,2)) = 0.74842141871341
    val expectedOutput = torch.Tensor[Double](2,2,2)
    expectedOutput(Array(1,1,1)) = 0
    expectedOutput(Array(1,1,2)) = 0
    expectedOutput(Array(1,2,1)) = 0
    expectedOutput(Array(1,2,2)) = 0
    expectedOutput(Array(2,1,1)) = 0
    expectedOutput(Array(2,1,2)) = 1.173689913936
    expectedOutput(Array(2,2,1)) = 1.6776262558997
    expectedOutput(Array(2,2,2)) = 0
    val expectedGrad = torch.Tensor[Double](2,2,2)
    expectedGrad(Array(1,1,1)) = 0
    expectedGrad(Array(1,1,2)) = 0
    expectedGrad(Array(1,2,1)) = 0
    expectedGrad(Array(1,2,2)) = 0
    expectedGrad(Array(2,1,1)) = 0
    expectedGrad(Array(2,1,2)) = 0.0086153273005038
    expectedGrad(Array(2,2,1)) = 0.053113180678338
    expectedGrad(Array(2,2,2)) = 0
    val output = module.forward(input)
    val gradInput = module.backward(input,gradOutput)
    expectedOutput.map(output,(v1,v2)=>{assert(abs(v1-v2)<1e-6);v1})
    expectedGrad.map(gradInput,(v1,v2)=>{assert(abs(v1-v2)<1e-6);v1})
    //inPlace test
    assert(output == input)
    assert(gradInput == gradOutput)
  }
}
