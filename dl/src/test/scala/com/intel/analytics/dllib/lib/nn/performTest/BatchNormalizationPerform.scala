package com.intel.analytics.dllib.lib.nn.performTest

import com.intel.analytics.dllib.lib.nn.BatchNormalization
import com.intel.analytics.dllib.lib.tensor.torch
import org.scalatest.{BeforeAndAfter, FlatSpec}
import com.intel.analytics.dllib.lib.tensor.RandomGenerator._


/**
  * Created by yao on 5/19/16.
  */
class BatchNormalizationPerform extends FlatSpec with BeforeAndAfter{
  val ForwardIterations = PerformTestData.forwardIterations
  val BackwardIterations = PerformTestData.backwardIterations
  val ForwardTimeoutMillis = 5
  val BackwardTimeoutMillis = 6
  val InputNum = 100
  val FeatureDim = 512
  val Seed = 100
  val bn = new BatchNormalization[Double](FeatureDim)

  for (i <- 1 until FeatureDim) {
    bn.weight(i) = 0.1 * i
    bn.bias(i) = 0.1 * i
  }

  RNG.setSeed(Seed)
  val input = torch.Tensor[Double](InputNum, FeatureDim).rand()

  //Warm up
  for (j <- 0 until ForwardIterations){
    bn.forward(input)
  }

  "A BatchNormalization" should " forward with a good performance" in {
    if (!TestUtils.isRun())
      cancel("Performance regression tests are forbidden")
    val timeMillis = TestUtils.testModuleForwardPerform(input, bn, ForwardIterations)
    //assert(timeMillis < ForwardTimeoutMillis)
  }

  val grads = torch.Tensor[Double](InputNum, FeatureDim).rand()

  it should(" backward with a good performance") in {
    if (!TestUtils.isRun())
      cancel("Performance regression tests are forbidden")
    val timeMillis = TestUtils.testModuleBackwardPerform(input, grads, bn, BackwardIterations)
    assert(timeMillis < BackwardTimeoutMillis)
  }
}
