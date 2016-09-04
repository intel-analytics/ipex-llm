package com.intel.analytics.dllib.lib.tensor.th

import com.intel.analytics.dllib.lib.tensor.{RandomGenerator, Tensor, torch, DenseTensor}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}


class RandomGeneratorSpec extends FlatSpec with BeforeAndAfter with Matchers {
  "uniform" should "return correct value" in{
    val a = new RandomGenerator(100)
    a.uniform(0,1) should be (0.543404 +- 1e-6)
    a.uniform(0,1) should be (0.671155 +- 1e-6)
    a.uniform(0,1) should be (0.278369 +- 1e-6)
  }

  "normal" should "return correct value" in{
    val a = new RandomGenerator(100)
    a.normal(0,1) should be (-1.436301 +- 1e-6)
    a.normal(0,1) should be (-0.401719 +- 1e-6)
    a.normal(0,1) should be (-0.182739 +- 1e-6)
  }

  "exponential" should "return correct value" in{
    val a = new RandomGenerator(100)
    a.exponential(1) should be (0.783958 +- 1e-6)
    a.exponential(1) should be (1.112170 +- 1e-6)
    a.exponential(1) should be (0.326241 +- 1e-6)
  }

  "cauchy" should "return correct value" in{
    val a = new RandomGenerator(100)
    a.cauchy(1,1) should be (1.137212 +- 1e-6)
    a.cauchy(1,1) should be (1.596309 +- 1e-6)
    a.cauchy(1,1) should be (0.164062 +- 1e-6)
  }

  "logNormal" should "return correct value" in{
    val a = new RandomGenerator(100)
    a.logNormal(1,1) should be (0.213872 +- 1e-6)
    a.logNormal(1,1) should be (0.506097 +- 1e-6)
    a.logNormal(1,1) should be (0.607310 +- 1e-6)
  }

  "geometric" should "return correct value" in{
    val a = new RandomGenerator(100)
    a.geometric(0.5) should be (2)
    a.geometric(0.5) should be (2)
    a.geometric(0.5) should be (1)
  }

  "bernoulli" should "return correct value" in{
    val a = new RandomGenerator(100)
    a.bernoulli(0.5) should be (false)
    a.bernoulli(0.5) should be (false)
    a.bernoulli(0.5) should be (true)
  }
}
