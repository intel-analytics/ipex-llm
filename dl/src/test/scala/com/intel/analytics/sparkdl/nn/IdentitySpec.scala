package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

/**
  * Created by yao on 9/20/16.
  */
class IdentitySpec extends FlatSpec with Matchers {
  "Identity" should "generate correct output and grad" in {
    val batchN = 3
    val inputN = 5
    val outputN = inputN

    val input = Tensor[Double](batchN, inputN)
    input.rand()
    val gradOutput = Tensor[Double](batchN, outputN)
    gradOutput.rand()

    val module = new Identity[Double]()
    val output = module.forward(input)
    assert(input equals output)

    val gradInput = module.backward(input, gradOutput)
    assert(gradInput equals gradOutput)
  }
}
