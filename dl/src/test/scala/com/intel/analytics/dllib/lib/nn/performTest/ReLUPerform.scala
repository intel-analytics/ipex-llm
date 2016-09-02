package com.intel.analytics.dllib.lib.nn.performTest

import java.io.{BufferedWriter, File, FileWriter}

import com.intel.analytics.dllib.lib.nn.ReLU
import com.intel.analytics.dllib.lib.tensor.{RandomGenerator, torch}
import org.scalatest.{BeforeAndAfter, FlatSpec}


/**
  * Created by yao on 5/19/16.
  */
class ReLUPerform extends FlatSpec with BeforeAndAfter{
  val ForwardIterations = PerformTestData.forwardIterations
  val BackwardIterations = PerformTestData.backwardIterations
  val ForwardTimeoutMillisIP = 11
  val BackwardTimeoutMillisIP = 17
  val ForwardTimeoutMillisNIP = 20
  val BackwardTimeoutMillisNIP = 290
  val InputNum = 100
  val FeatureNum = 512
  val Seed = 100
  val bw = new BufferedWriter(new FileWriter(new File(".").getPath + "/run_time.txt", true))

  val input = torch.Tensor[Float](InputNum, FeatureNum, 512).rand()

  //AlexNet, Cifar, GoogleNet
  val ipModel = new ReLU[Float](true)
  //GoogleNet
  val nipModel = new ReLU[Float]()

  //Warm up
  for (i <- 0 until ForwardIterations) {
    ipModel.forward(input)
  }

  "ReLU(in place)" should "forward with a good performance" in {
    if (!TestUtils.isRun())
      cancel("Performance regression tests are forbidden")
    val timeMillis = TestUtils.testModuleForwardPerform(input, ipModel, ForwardIterations)
    assert(timeMillis < ForwardTimeoutMillisIP)
  }


  val grads = torch.Tensor[Float](InputNum, FeatureNum, 512).rand()

  "ReLU(in place)" should "backward with a good performance" in {
    if (!TestUtils.isRun())
      cancel("Performance regression tests are forbidden")
    val timeMillis = TestUtils.testModuleBackwardPerform(input, grads, ipModel, BackwardIterations)
    assert(timeMillis < BackwardTimeoutMillisIP)
  }

  "ReLU(not in place)" should "forward with a good performance" in {
    if (!TestUtils.isRun())
      cancel("Performance regression tests are forbidden")
    val timeMillis = TestUtils.testModuleForwardPerform(input, nipModel, ForwardIterations)
    assert(timeMillis < ForwardTimeoutMillisNIP)
  }

  "ReLU(not in place)" should "backward with a good performance" in {
    if (!TestUtils.isRun())
      cancel("Performance regression tests are forbidden")
    val timeMillis = TestUtils.testModuleBackwardPerform(input, grads, nipModel, BackwardIterations)
    assert(timeMillis < BackwardTimeoutMillisNIP)
  }
}
