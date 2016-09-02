package com.intel.analytics.dllib.lib.optim

import com.intel.analytics.dllib.lib.tensor.{T, Table, torch}
import org.scalatest.{Matchers, FlatSpec}

import scala.collection.mutable.ArrayBuffer

class AdagradSpec extends FlatSpec with Matchers {
  "adagrad" should "perform well on rosenbrock function" in {
    val x = torch.Tensor[Double](2).fill(0)
    val config = T("learningRate" -> 1e-1)
    val optm = new Adagrad[Double]
    var fx = new ArrayBuffer[Double]
    for(i <- 1 to 10001) {
      val result = optm.optimize(TestUtils.rosenBrock, x, config)
      if((i - 1) % 1000 == 0) {
        fx += (result._2(0))
      }
    }

    println(s"x is \n$x")
    println("fx is")
    for(i <- 1 to fx.length) {
      println(s"${(i - 1) * 1000 + 1}, ${fx(i - 1)}")
    }

    (fx.last < 1e-9) should be(true)
    x(Array(1)) should be (1.0 +- 0.01)
    x(Array(2)) should be (1.0 +- 0.01)
  }
}
