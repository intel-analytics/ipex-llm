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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

class LogSpec extends FlatSpec with Matchers {
  "An Log" should "generate correct output" in {
    val input = Tensor(Storage[Double](Array(1.0, 2, Math.E, 4, 5, 6)), 1, Array(2, 3))

    val output = Tensor(Storage(Array(
      0.0, 0.6931471805599453, 1.0,
      1.3862943611198906, 1.6094379124341003, 1.791759469228055)), 1, Array(2, 3))

    val log = Log[Double, Double]()

    val powerOutput = log.forward(input)

    powerOutput should equal (output)
  }

  "An Log1p" should "generate correct output" in {
    val input = Tensor(Storage[Double](Array(0.0, 1.0, Math.E-1, 3, 4, 5)), 1, Array(2, 3))

    val output = Tensor(Storage(Array(
      0.0, 0.6931471805599453, 1.0,
      1.3862943611198906, 1.6094379124341003, 1.791759469228055)), 1, Array(2, 3))

    val log1p = Log1p[Double, Double]()

    val powerOutput = log1p.forward(input)

    powerOutput should equal (output)
  }

  "An LogN" should "generate correct output" in {
    val input = Tensor(Storage[Double](Array(1.0, 2, 3, 4, 5, 6)), 1, Array(2, 3))

    val output = Tensor(Storage(Array(
      0.0, 1.0, 1.5849625007211563,
      2.0, 2.321928094887362, 2.584962500721156)), 1, Array(2, 3))

    val logN = LogN[Double, Double](2)

    val powerOutput = logN.forward(input)

    powerOutput should equal (output)
  }

  "An LogN1p" should "generate correct output" in {
    val input = Tensor(Storage[Double](Array(0.0, 1, 2, 3, 4, 5)), 1, Array(2, 3))

    val output = Tensor(Storage(Array(
      0.0, 1.0, 1.5849625007211563,
      2.0, 2.321928094887362, 2.584962500721156)), 1, Array(2, 3))

    val logN1p = LogN1p[Double, Double](2)

    val powerOutput = logN1p.forward(input)

    powerOutput should equal (output)
  }
}
