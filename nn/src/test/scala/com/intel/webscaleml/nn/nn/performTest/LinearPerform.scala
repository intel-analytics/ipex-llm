package com.intel.webscaleml.nn.nn.performTest

import java.io.{BufferedWriter, File, FileWriter}

import com.intel.webscaleml.nn.nn.Linear
import com.intel.webscaleml.nn.tensor.{RandomGenerator, torch}
import org.scalatest.{BeforeAndAfter, FlatSpec}
import org.scalatest.concurrent.Timeouts._
import org.scalatest.time.{Millis, Span}

/**
  * Created by yao on 5/19/16.
  */
class LinearPerform extends FlatSpec with BeforeAndAfter{
  val ForwardIterations = PerformTestData.forwardIterations
  val BackwardIterations = PerformTestData.backwardIterations
  val InputNum = 100
  val Seed = 100
  val allTestCases = List(
    TestCase(256 * 6 * 6, 4096, 55, 90),
    TestCase(4096, 4096,25, 55),
    TestCase(256 * 5 * 5, 128, 1.5, 2),
    TestCase(512, 512, 1.2, 2.5),
    TestCase(512, 10, 0.5, 0.5),
    TestCase(28 * 4 * 4, 768, 1, 2.5)
  )

  for (testCase <- allTestCases) {
    "A Linear" should s"forward with a good performance with parameters: ${testCase.inputSize}," +
      s" ${testCase.outputSize}" in {
      if (!TestUtils.isRun())
        cancel("Performance regression tests are forbidden")
      val input = torch.Tensor[Float](InputNum, testCase.inputSize).rand()
      val model = new Linear[Float](testCase.inputSize, testCase.outputSize)

      val timeMillis = TestUtils.testModuleForwardPerform(input, model, ForwardIterations)
      assert(timeMillis < testCase.ForwardTimeoutMillis)
    }
  }

  for (testCase <- allTestCases) {
    val model = new Linear[Float](testCase.inputSize, testCase.outputSize)
    val input = torch.Tensor[Float](InputNum, testCase.inputSize)
    val output = model.forward(input)
    val grad = output.clone().rand()

    it should s" backward with a good performance with parameters: ${testCase.inputSize}," +
      s" ${testCase.outputSize}" in {
      if (!TestUtils.isRun())
        cancel("Performance regression tests are forbidden")
      val timeMillis = TestUtils.testModuleBackwardPerform(input, grad, model, BackwardIterations)
      assert(timeMillis < testCase.BackwardTimeoutMillis)
    }

  }

  case class TestCase(inputSize: Int, outputSize: Int, ForwardTimeoutMillis: Double, BackwardTimeoutMillis: Double)
}
