package com.intel.webscaleml.nn.nn.performTest

import com.intel.webscaleml.nn.tensor.torch
import org.scalatest.{BeforeAndAfter, FlatSpec}
/**
  * Created by yansh on 16-7-7.
  */
class SpatialConvolutionNNPackPerform extends FlatSpec with BeforeAndAfter{
    /*val ForwardIterations = PerformTestData.forwardIterations
    val BackwardIterations = PerformTestData.backwardIterations
    val Seed = 100
    val batchSize = 10

    val allTestCases = List(
      // AlexNet
      //TestCase(3, 64, 11, 11, 4, 4, 2, 2 , 224, 224, 450, 960),
      TestCase(64, 192, 5, 5, 1, 1, 2, 2, 25, 25, 1050, 2700),
      TestCase(191, 384, 3, 3, 1, 1, 1, 1, 12, 12, 550, 1370),
      TestCase(384, 256, 3, 3, 1, 1, 1, 1, 6, 6, 200, 393),
      TestCase(256, 256, 3, 3, 1, 1, 1, 1, 3, 3, 40, 150),
      //Cifar
      TestCase(3, 64, 3, 3, 1, 1, 1, 1, 224, 224, 570, 1260),
      TestCase(64, 64, 3, 3, 1, 1, 1, 1, 110, 110, 2500, 6000),
      TestCase(64, 128, 3, 3, 1, 1, 1, 1, 54, 54, 1150, 2900),
      TestCase(128, 128, 3, 3, 1, 1, 1 ,1 ,26, 26, 550, 1400),
      TestCase(128, 256, 3, 3, 1, 1, 1, 1, 13, 13, 300, 670),
      TestCase(256, 256, 3, 3, 1, 1, 1, 1, 6, 6, 225, 290),
      TestCase(256, 512, 3, 3, 1, 1, 1, 1, 3, 3, 75, 190),
      TestCase(512, 512, 3, 3, 1, 1, 1, 1, 2, 2, 80, 230),

      //GoogleNet
      //TestCase(3, 64, 7, 7, 2, 2, 3, 3, 224, 224, 650, 1700),
      TestCase(64, 64, 1, 1, 1, 1, 0, 0, 54, 54, 80, 175),
      TestCase(64, 192, 3, 3, 1, 1, 1, 1, 27, 27, 450, 1100),
      TestCase(192, 576, 3, 3, 1, 1, 1, 1, 12, 12, 800, 2000)
      //TestCase(576, 576, 2, 2, 2, 2, 0, 0, 4, 4, 45, 150)
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
        val timeMillis = TestUtils.testModuleNNPackForwardPerform(input, model, ForwardIterations)
        //assert(timeMillis < testCase.ForwardTimeoutMillis)
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
        val timeMillis = TestUtils.testModuleNNPackBackwardPerform(input, grad, model, BackwardIterations)
        //assert(timeMillis < testCase.BackwardTimeoutMillis)
      }
    }

    case class TestCase(nInputPlane : Int, nOutputPlane : Int, kW : Int, kH : Int, dW : Int, dh : Int,
                        padW: Int, padH: Int, inputWidth: Int, inputHeight: Int, ForwardTimeoutMillis: Int, BackwardTimeoutMillis: Int)

*/
}
