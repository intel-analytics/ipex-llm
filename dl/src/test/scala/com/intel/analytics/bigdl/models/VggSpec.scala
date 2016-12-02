package com.intel.analytics.bigdl.models

import com.intel.analytics.bigdl.nn.GradientChecker
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class VggSpec extends FlatSpec with BeforeAndAfter with Matchers{
  "VggLike model in batch mode" should "be good in gradient check" in {
    val input = Tensor[Double](8, 3, 32, 32).apply1(e => Random.nextDouble())
    val model = VggLike[Double](10)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-5)
    checker.checkLayer(model, input, 1e-2) should be(true)
    checker.checkWeight(model, input, 1e-2) should be(true)
  }
}
