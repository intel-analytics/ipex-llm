package com.intel.webscaleml.nn.nn.performTest

import java.io.{BufferedWriter, File, FileWriter}

import com.intel.webscaleml.nn.nn.SpatialMaxPooling
import com.intel.webscaleml.nn.tensor.{RandomGenerator, torch}
import org.scalatest.{BeforeAndAfter, FlatSpec}

/**
  * Created by yao on 5/19/16.
  */
class SpatialMaxPoolingPerform extends FlatSpec with BeforeAndAfter{
  val ForwardIterations = PerformTestData.forwardIterations
  val BackwardIterations = PerformTestData.backwardIterations
  val BatchSize = 100
  val NInputPlane = 3
  val Height = 100
  val Width = 512
  val Seed = 100
  val allTestCases = List(
    TestCase(3, 3, 2, 2, 40, 20), // AlexNet, GoogleNet
    TestCase(2, 2, 2, 2, 30, 25), // Cifar, CifarLocal
    TestCase(3, 3, 1, 1, 130, 25), //GoogleNet
    TestCase(3, 3, 3, 3, 22, 20) // MNIST
  )

  val input = torch.Tensor[Float](BatchSize, NInputPlane, Height, Width).rand()

  for (testCase <- allTestCases) {
    val model = new SpatialMaxPooling[Float](testCase.kW, testCase.kH, testCase.dW, testCase.dH)

    "A SpatialMaxPoolingPerform" should s" forward with a good performance with parameters" +
      s": ${testCase.kW}, ${testCase.kH}, ${testCase.dW}, ${testCase.dH}," in {
      if (!TestUtils.isRun())
        cancel("Performance regression tests are forbidden")
      val timeMillis = TestUtils.testModuleForwardPerform(input, model, ForwardIterations)
      assert(timeMillis < testCase.ForwardTimeoutMillis)
    }
  }

  for (testCase <- allTestCases) {
    val model = new SpatialMaxPooling[Float](testCase.kW, testCase.kH, testCase.dW, testCase.dH)
    val output = model.forward(input)
    val grad = output.clone().rand()

    "A SpatialMaxPoolingPerform" should s" backward with a good performance with parameters" +
      s": ${testCase.kW}, ${testCase.kH}, ${testCase.dW}, ${testCase.dH}," in {
      if (!TestUtils.isRun())
        cancel("Performance regression tests are forbidden")
      val timeMillis = TestUtils.testModuleBackwardPerform(input, grad, model, BackwardIterations)
      assert(timeMillis < testCase.BackwardTimeoutMillis)
    }
  }

  case class TestCase(kW: Int, kH: Int, dW: Int, dH: Int, ForwardTimeoutMillis: Int, BackwardTimeoutMillis: Int)
}


