package com.intel.analytics.dllib.lib.nn.performTest

import com.intel.analytics.dllib.lib.nn.BCECriterion
import com.intel.analytics.dllib.lib.tensor.{torch}
import org.scalatest.{BeforeAndAfter, FlatSpec}
import com.intel.analytics.dllib.lib.tensor.RandomGenerator._

/**
  * Created by yao on 5/19/16.
  */
class BCECriterionPerform extends FlatSpec with BeforeAndAfter{
  val ForwardIterations = PerformTestData.forwardIterations
  val BackwardIterations = PerformTestData.backwardIterations
  val ForwardTimeoutMillis = 20
  val BackwardTimeoutMillis = 20
  val InputNum = 100
  val FeatureDim = 512
  val Seed = 100

  RNG.setSeed(Seed)
  val input = torch.Tensor[Float](InputNum, FeatureDim).rand()
  val target = torch.Tensor[Float](InputNum, FeatureDim).rand()
  val criterion = new BCECriterion[Float]()

  "BCECriterion " should "forward with a good performance" in {
    if (!TestUtils.isRun())
      cancel("Performance regression tests are forbidden")
    val timeMillis = TestUtils.testCriterionForwardPerform(input, target, criterion, ForwardIterations)
    //assert(timeMillis < ForwardTimeoutMillis)
  }

  it should " backward with a good performance" in {
    if (!TestUtils.isRun())
      cancel("Performance regression tests are forbidden")
    val timeMillis = TestUtils.testCriterionBackwardPerform(input, target, criterion, BackwardIterations)
    //assert(timeMillis < BackwardTimeoutMillis)
  }
}
