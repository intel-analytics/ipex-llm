package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.Tensor
import org.scalatest.FlatSpec

/**
  * Created by yao on 9/21/16.
  */
class MulConstantSpec extends FlatSpec {
  "MulConstant" should "generate correct output and grad" in {
    val input = Tensor[Double](2, 2, 2).randn()
    val scalar = 25.0
    val expectedOutput = input.clone().apply1(_ * scalar)
    val gradOutput = Tensor[Double](2, 2, 2).rand()
    val expectedGrad = gradOutput.clone().apply1(_ * scalar)

    val module = new MulConstant[Double](scalar)
    val output = module.forward(input)
    assert(expectedOutput equals output)

    val gradInput = module.backward(input, gradOutput)
    assert(gradInput equals expectedGrad )
  }

  "MulConstant with inPlace = true" should "generate correct output and grad" in {
    var input = Tensor[Double](2, 2, 2).randn()
    val scalar = 25.0
    val expectedOutput = input.clone().apply1(_ * scalar)
    val gradOutput = Tensor[Double](2, 2, 2).rand()
    val expectedGrad = gradOutput.clone().apply1(_ * scalar)

    // Test forward
    val module = new MulConstant[Double](scalar, true)
    val output = module.forward(input)
    assert(expectedOutput equals output)

    // Test backward
    input = Tensor[Double](2, 2, 2).randn()
    val expctedInput = input.clone().apply1(_ / scalar)
    val gradInput = module.backward(input, gradOutput)
    assert(gradInput equals expectedGrad)
    assert(input equals expctedInput)
  }
}
