package com.intel.analytics.dllib.lib.nn.performTest

import com.intel.analytics.dllib.lib.nn._
import com.intel.analytics.dllib.lib.tensor.torch
import org.scalatest.{BeforeAndAfter, FlatSpec}

/**
  * Created by yao on 5/24/16.
  */
class SpatialConvolutionPerform extends FlatSpec with BeforeAndAfter{
  val ForwardIterations = PerformTestData.forwardIterations
  val BackwardIterations = PerformTestData.backwardIterations
  val Seed = 100
  val batchSize = 10

  val allTestCases = List(
    // AlexNet
    TestCase(3, 64, 11, 11, 4, 4, 2, 2 , 224, 224, 16, 35),
    TestCase(64, 192, 5, 5, 1, 1, 2, 2, 25, 25, 22, 55),
    TestCase(191, 384, 3, 3, 1, 1, 1, 1, 12, 12, 10, 35),
    TestCase(384, 256, 3, 3, 1, 1, 1, 1, 6, 6, 10, 25),
    TestCase(256, 256, 3, 3, 1, 1, 1, 1, 3, 3, 6, 10),
    //Cifar
    TestCase(3, 64, 3, 3, 1, 1, 1, 1, 224, 224, 70, 65),
    TestCase(64, 64, 3, 3, 1, 1, 1, 1, 110, 110, 90, 180),
    TestCase(64, 128, 3, 3, 1, 1, 1, 1, 54, 54, 40, 70),
    TestCase(128, 128, 3, 3, 1, 1, 1 ,1 ,26, 26, 19, 30),
    TestCase(128, 256, 3, 3, 1, 1, 1, 1, 13, 13, 7, 17),
    TestCase(256, 256, 3, 3, 1, 1, 1, 1, 6, 6, 7, 15),
    TestCase(256, 512, 3, 3, 1, 1, 1, 1, 3, 3, 7, 20),
    TestCase(512, 512, 3, 3, 1, 1, 1, 1, 2, 2, 12, 40),

    //GoogleNet
    TestCase(3, 64, 7, 7, 2, 2, 3, 3, 224, 224, 40, 55),
    TestCase(64, 64, 1, 1, 1, 1, 0, 0, 54, 54, 5, 8),
    TestCase(64, 192, 3, 3, 1, 1, 1, 1, 27, 27, 10, 25),
    TestCase(192, 576, 3, 3, 1, 1, 1, 1, 12, 12, 16, 42),
    TestCase(576, 576, 2, 2, 2, 2, 0, 0, 4, 4, 10, 25)
  )

  for (testCase <- allTestCases) {
    val input = torch.Tensor[Float](batchSize, testCase.nInputPlane, testCase.inputWidth, testCase.inputHeight).rand()
    val model = new SpatialConvolution[Float](testCase.nInputPlane, testCase.nOutputPlane,
      testCase.kW, testCase.kH, testCase.dW, testCase.dh, testCase.padW, testCase.padH)

    "A SpatialConvolutionPerform" should s" forward with a good performance with parameters" +
      s": ${testCase.nInputPlane}, ${testCase.nOutputPlane}, ${testCase.kW}, ${testCase.kH}, ${testCase.dW}, ${testCase.dh}," +
      s" ${testCase.padW}, ${testCase.padH}, ${testCase.inputHeight}, ${testCase.inputWidth}" in {
      if (!TestUtils.isRun())
        cancel("Performance regression tests are forbidden")
      val timeMillis = TestUtils.testModuleForwardPerform(input, model, ForwardIterations)
      assert(timeMillis < testCase.ForwardTimeoutMillis)
    }
  }

  for (testCase <- allTestCases) {
    val input = torch.Tensor[Float](batchSize, testCase.nInputPlane, testCase.inputWidth, testCase.inputHeight).rand()
    val model = new SpatialConvolution[Float](testCase.nInputPlane, testCase.nOutputPlane,
      testCase.kW, testCase.kH, testCase.dW, testCase.dh, testCase.padW, testCase.padH)
    val output = model.forward(input)
    val grad = output.clone().rand()

    "A SpatialConvolutionPerform" should s" backward with a good performance with parameters" +
      s": ${testCase.nInputPlane}, ${testCase.nOutputPlane}, ${testCase.kW}, ${testCase.kH}, ${testCase.dW}, ${testCase.dh}" +
      s", ${testCase.padW}, ${testCase.padH}, ${testCase.inputHeight}, ${testCase.inputWidth}" in {
      if (!TestUtils.isRun())
        cancel("Performance regression tests are forbidden")
      val timeMillis = TestUtils.testModuleBackwardPerform(input, grad, model, BackwardIterations)
      assert(timeMillis < testCase.BackwardTimeoutMillis)
    }
  }

  case class TestCase(nInputPlane : Int, nOutputPlane : Int, kW : Int, kH : Int, dW : Int, dh : Int,
                      padW: Int, padH: Int, inputWidth: Int, inputHeight: Int, ForwardTimeoutMillis: Int, BackwardTimeoutMillis: Int)
}

