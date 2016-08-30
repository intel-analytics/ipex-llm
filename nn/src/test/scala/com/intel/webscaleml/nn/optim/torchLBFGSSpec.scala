package com.intel.webscaleml.nn.optim

import com.intel.webscaleml.nn.tensor.{T, Table, torch}
import org.scalatest.{Matchers, FlatSpec}

import scala.collection.mutable.ArrayBuffer

class torchLBFGSSpec extends FlatSpec with Matchers {
  "torchLBFGS in regular batch test" should "perform well on rosenbrock function" in {
    val x = torch.Tensor[Double](2).fill(0)
    val optm = new LBFGS[Double]
    val result = optm.optimize(TestUtils.rosenBrock, x, T("maxIter" -> 100, "learningRate" -> 1e-1))
    val fx = result._2

    println()
    println("Rosenbrock test")
    println()

    println(s"x = $x")
    println("fx = ")
    for(i <- 1 to fx.length) {
      println(s"$i ${fx(i - 1)}")
    }
    println()
    println()

    fx.last < 1e-6 should be(true)
  }

  "torchLBFGS in stochastic test" should "perform well on rosenbrock function" in {
    val x = torch.Tensor[Double](2).fill(0)
    val optm = new LBFGS[Double]
    val fx = new ArrayBuffer[Double]()

    val config = T("maxIter" -> 1, "learningRate" -> 1e-1)
    for(i <- 1 to 100) {
      val result = optm.optimize(TestUtils.rosenBrock, x, config)
      fx.append(result._2(0))
    }

    println()
    println("Rosenbrock test")
    println()

    println(s"x = $x")
    println("fx = ")
    for(i <- 1 to fx.length) {
      println(s"$i ${fx(i - 1)}")
    }
    println()
    println()

    fx.last < 1e-6 should be(true)
  }
}