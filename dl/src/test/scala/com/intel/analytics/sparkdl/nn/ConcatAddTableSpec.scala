package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

/**
  * Created by yao on 9/25/16.
  */
class ConcatAddTableSpec extends FlatSpec with Matchers {
  "A ConcatAddTable" should "generate correct output" in {
    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )
    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    val model = new Sequential[Double]()
      .add(new ConcatAddTable[Double](true)
             .add(new Identity[Double]())
             .add(new MulConstant[Double](5)))

    val output = model.forward(input)

    val outputData = Array(
      6.0, 12, 18,
      24, 30, 36,
      42, 48, 54
    )
    val expectedOutput = Tensor[Double](Storage(outputData), 1, Array(1, 3, 3))
    assert(output equals expectedOutput)

    val gradOutputData = Array(
      11.0, 12, 13,
      14, 15, 16,
      17, 18, 19
    )
    val gradOutput = Tensor[Double](Storage(gradOutputData), 1, Array(1, 3, 3))
    val gradInput = model.backward(input, gradOutput)

    val expectedGradIntputData = Array(
      66.0, 72, 78,
      84, 90, 96,
      102, 108, 114
    )
    val expectedGradInput = Tensor[Double](Storage(expectedGradIntputData), 1, Array(1, 3, 3))
    assert(gradInput equals expectedGradInput)
  }
}
