package com.intel.webscaleml.nn.nn.performTest

import com.intel.webscaleml.nn.nn.ClassNLLCriterion
import com.intel.webscaleml.nn.tensor.{RandomGenerator, torch}
import org.scalatest.{BeforeAndAfter, FlatSpec}

/**
  * Created by yao on 5/19/16.
  */
class ClassNLLCriterionPerform extends FlatSpec with BeforeAndAfter{
  val ForwardIterations = PerformTestData.forwardIterations
  val BackwardIterations = PerformTestData.backwardIterations
  val ForwardTimeoutMillis = 1.0
  val BackwardTimeoutMillis = 1.0
  val InputNum = 512
  val FeatureDim = 512
  val Seed = 100
  val Generator = new RandomGenerator(Seed)
  val isRun = TestUtils.isRun()

  val criterion = new ClassNLLCriterion[Float]()
  val input = torch.Tensor[Float](InputNum, FeatureDim).rand()
  val target = torch.Tensor[Float](InputNum)

  //Initialize output and targets
  target.apply1(_ => Generator.uniform(1, 10).asInstanceOf[Float].ceil)

  //Warm up JVM
  for (j <- 0 until ForwardIterations) {
    criterion.forward(input, target)
  }

  "ClassNLLCriterion " should "forward with a good performance" in {
    if (!TestUtils.isRun())
      cancel("Performance regression tests are forbidden")
    val timeMillis = TestUtils.testCriterionForwardPerform(input, target, criterion, ForwardIterations)
    assert(timeMillis < ForwardTimeoutMillis)
  }

  it should " backward with a good performance" in {
    if (!TestUtils.isRun())
      cancel("Performance regression tests are forbidden")
    val timeMillis = TestUtils.testCriterionBackwardPerform(input, target, criterion, BackwardIterations)
    assert(timeMillis < BackwardTimeoutMillis)
  }

}
