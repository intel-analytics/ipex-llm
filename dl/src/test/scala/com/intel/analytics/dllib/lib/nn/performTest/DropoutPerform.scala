package com.intel.webscaleml.nn.nn.performTest

import java.io.{BufferedWriter, File, FileWriter}

import com.intel.webscaleml.nn.nn.Dropout
import com.intel.webscaleml.nn.tensor.{torch}
import org.scalatest.{BeforeAndAfter, FlatSpec}
import com.intel.webscaleml.nn.tensor.RandomGenerator._

/**
  * Created by yao on 5/19/16.
  */
class DropoutPerform extends FlatSpec with BeforeAndAfter{
  val ForwardIterations = PerformTestData.forwardIterations
  val BackwardIterations = PerformTestData.backwardIterations
  val InputNum = 1000
  val FeatureDim = 512
  val Seed = 100
  val allTestCases = List(
    TestCase(0.3, 42, 25),
    TestCase(0.4, 45, 25),
    TestCase(0.5, 43, 25)
  )

  RNG.setSeed(Seed);
  val input = torch.Tensor[Float](InputNum, FeatureDim).rand()

  for (testCase <- allTestCases){
    val model = new Dropout[Float](testCase.p)
    "A Dropout" should s" forward with a good performance with parameters: ${testCase.p}" in {
      if (!TestUtils.isRun())
        cancel("Performance regression tests are forbidden")

      val timeMillis = TestUtils.testModuleForwardPerform(input, model, ForwardIterations)
      //assert(timeMillis < testCase.ForwardTimeoutMillis)
    }
  }

  for (testCase <- allTestCases) {
    val model = new Dropout[Float](testCase.p)
    val output = model.forward(input)
    val grads = output.clone().rand()
    it should s" backward with a good performance with parameters: ${testCase.p}" in {
      if (!TestUtils.isRun())
        cancel("Performance regression tests are forbidden")

      val timeMillis = TestUtils.testModuleBackwardPerform(input, grads, model, BackwardIterations)
      //assert(timeMillis < testCase.BackwardTimeoutMillis)
    }
  }

  case class TestCase(p: Double, ForwardTimeoutMillis: Int, BackwardTimeoutMillis: Int)
}
