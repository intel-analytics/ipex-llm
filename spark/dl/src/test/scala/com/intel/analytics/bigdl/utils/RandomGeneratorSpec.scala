/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.utils

import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class RandomGeneratorSpec extends FlatSpec with BeforeAndAfter with Matchers {
  "uniform" should "return correct value" in {
    val a = new RandomGenerator(100)
    a.uniform(0, 1) should be(0.543404 +- 1e-6)
    a.uniform(0, 1) should be(0.671155 +- 1e-6)
    a.uniform(0, 1) should be(0.278369 +- 1e-6)
  }

  "normal" should "return correct value" in {
    val a = new RandomGenerator(100)
    a.normal(0, 1) should be(-1.436301 +- 1e-6)
    a.normal(0, 1) should be(-0.401719 +- 1e-6)
    a.normal(0, 1) should be(-0.182739 +- 1e-6)
  }

  "exponential" should "return correct value" in {
    val a = new RandomGenerator(100)
    a.exponential(1) should be(0.783958 +- 1e-6)
    a.exponential(1) should be(1.112170 +- 1e-6)
    a.exponential(1) should be(0.326241 +- 1e-6)
  }

  "cauchy" should "return correct value" in {
    val a = new RandomGenerator(100)
    a.cauchy(1, 1) should be(1.137212 +- 1e-6)
    a.cauchy(1, 1) should be(1.596309 +- 1e-6)
    a.cauchy(1, 1) should be(0.164062 +- 1e-6)
  }

  "logNormal" should "return correct value" in {
    val a = new RandomGenerator(100)
    a.logNormal(1, 1) should be(0.213872 +- 1e-6)
    a.logNormal(1, 1) should be(0.506097 +- 1e-6)
    a.logNormal(1, 1) should be(0.607310 +- 1e-6)
  }

  "geometric" should "return correct value" in {
    val a = new RandomGenerator(100)
    a.geometric(0.5) should be(2)
    a.geometric(0.5) should be(2)
    a.geometric(0.5) should be(1)
  }

  "bernoulli" should "return correct value" in {
    val a = new RandomGenerator(100)
    a.bernoulli(0.5) should be(false)
    a.bernoulli(0.5) should be(false)
    a.bernoulli(0.5) should be(true)
  }
}
