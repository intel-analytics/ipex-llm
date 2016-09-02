package com.intel.webscaleml.nn.nn

import com.intel.webscaleml.nn.tensor.{torch, DenseTensor$}
import org.scalatest.{Matchers, FlatSpec}

import scala.math._

class LinearSpec  extends FlatSpec with Matchers {
  "Linear module" should "converate to correct weight and bias" in {
    val inputN = 5
    val outputN = 2

    val linear = new Linear[Double](inputN,outputN)
    val mse = new MSECriterion[Double]

    val input = torch.Tensor[Double](inputN)
    val res = torch.Tensor[Double](outputN)
    var err = 0.0
    for(i <-1 to 10000) {
      input.rand()
      for(y<-1 to outputN){
        res(Array(y)) = 1.0*y
        for(x<-1 to inputN){
          res(Array(y)) += 0.1*y*x*input(Array(x))
        }
      }
      val output = linear.forward(input)
      err = mse.forward(output, res)
      val grad = mse.backward(output, res)
      linear.zeroGradParameters()
      linear.backward(input,grad)
      linear.updateParameters(0.5/log(i+3))
    }
    val params = linear.parameters()
    val weight = params._1(0)
    val bias = params._1(1)

    val expectedWeight = torch.Tensor[Double](outputN, inputN)
    val expectedBias = torch.Tensor[Double](outputN)
    for(y<-1 to outputN){
      expectedBias(Array(y)) = 1.0*y
      for(x<-1 to inputN){
        expectedWeight(Array(y,x)) = 0.1*y*x
      }
    }

    expectedBias.map(bias,(v1,v2)=>{assert(abs(v1-v2)<1e-6);v1})
    expectedWeight.map(weight,(v1,v2)=>{assert(abs(v1-v2)<1e-6);v1})
    assert(err < 1e-6)
  }

  "Linear module in batch mode" should "converate to correct weight and bias" in {
    val inputN = 5
    val outputN = 2
    val batchN = 3

    val linear = new Linear[Double](inputN,outputN)
    val mse = new MSECriterion[Double]

    val input = torch.Tensor[Double](batchN,inputN)
    val res = torch.Tensor[Double](batchN, outputN)
    var err = 0.0
    for(i <-1 to 10000) {
      input.rand()
      for(k<-1 to batchN) {
        for(y<-1 to outputN){
          res(Array(k,y)) = 1.0*y
          for(x<-1 to inputN){
            res(Array(k,y)) += 0.1*y*x*input(Array(k,x))
          }
        }
      }
      val output = linear.forward(input)
      err = mse.forward(output, res)
      val grad = mse.backward(output, res)
      linear.zeroGradParameters()
      linear.backward(input,grad)
      linear.updateParameters(0.5/log(i+3))
    }
    val params = linear.parameters()
    val weight = params._1(0)
    val bias = params._1(1)

    val expectedWeight = torch.Tensor[Double](outputN, inputN)
    val expectedBias = torch.Tensor[Double](outputN)
    for(y<-1 to outputN){
      expectedBias(Array(y)) = 1.0*y
      for(x<-1 to inputN){
        expectedWeight(Array(y,x)) = 0.1*y*x
      }
    }

    expectedBias.map(bias,(v1,v2)=>{assert(abs(v1-v2)<1e-6);v1})
    expectedWeight.map(weight,(v1,v2)=>{assert(abs(v1-v2)<1e-6);v1})
    assert(err < 1e-6)
  }

  "Linear module in batch mode" should "be good in gradient check" in {
    val linear = new Linear[Double](5,2)
    linear.reset()
    val input = torch.Tensor[Double](3,5).rand()

    val checker = new GradientChecker(1e-2, 1e-2)
    checker.checkLayer(linear, input) should be(true)
  }
}
