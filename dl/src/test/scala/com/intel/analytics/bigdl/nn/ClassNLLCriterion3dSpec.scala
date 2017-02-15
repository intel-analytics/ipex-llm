/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn

import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

import scala.math._

class ClassNLLCriterion3dSpec extends FlatSpec with Matchers {
  "A ClassNLL Criterion for 3d input" should "generate correct output and grad" in {
    val criterion = new ClassNLLCriterion3d[Double]()
    val input = Tensor[Double](3, 2, 3)
    input(Array(1, 1, 1)) = -1.0262627674932
    input(Array(1, 1, 2)) = -1.2412600935171
    input(Array(1, 1, 3)) = -1.0423174168648
    input(Array(1, 2, 1)) = -1.0262627674932
    input(Array(1, 2, 2)) = -1.2412600935171
    input(Array(1, 2, 3)) = -1.0423174168648
    input(Array(2, 1, 1)) = -0.90330565804228
    input(Array(2, 1, 2)) = -1.3686840144413
    input(Array(2, 1, 3)) = -1.0778380454479
    input(Array(2, 2, 1)) = -0.90330565804228
    input(Array(2, 2, 2)) = -1.3686840144413
    input(Array(2, 2, 3)) = -1.0778380454479
    input(Array(3, 1, 1)) = -0.99131220658219
    input(Array(3, 1, 2)) = -1.0559142847536
    input(Array(3, 1, 3)) = -1.2692712660404
    input(Array(3, 2, 1)) = -0.99131220658219
    input(Array(3, 2, 2)) = -1.0559142847536
    input(Array(3, 2, 3)) = -1.2692712660404
    val target = Tensor[Double](3, 2)
    target(Array(1, 1)) = 1
    target(Array(1, 2)) = 1
    target(Array(2, 1)) = 2
    target(Array(2, 2)) = 2
    target(Array(3, 1)) = 3
    target(Array(3, 2)) = 3

    val output = criterion.forward(input, target)
    val gradInput = criterion.backward(input, target)

    println(s"Scala output, $output")
    println(s"Scala gradInput, $gradInput")

    val expectedOutput = 1.2214060159916
    val expectedGrad = Tensor[Double](3, 2, 3)
    expectedGrad(Array(1, 1, 1)) = -0.16666666666666666
    expectedGrad(Array(1, 1, 2)) = 0
    expectedGrad(Array(1, 1, 3)) = 0
    expectedGrad(Array(1, 2, 1)) = -0.16666666666666666
    expectedGrad(Array(1, 2, 2)) = 0
    expectedGrad(Array(1, 2, 3)) = 0
    expectedGrad(Array(2, 1, 1)) = 0
    expectedGrad(Array(2, 1, 2)) = -0.16666666666666666
    expectedGrad(Array(2, 1, 3)) = 0
    expectedGrad(Array(2, 2, 1)) = 0
    expectedGrad(Array(2, 2, 2)) = -0.16666666666666666
    expectedGrad(Array(2, 2, 3)) = 0
    expectedGrad(Array(3, 1, 1)) = 0
    expectedGrad(Array(3, 1, 2)) = 0
    expectedGrad(Array(3, 1, 3)) = -0.16666666666666666
    expectedGrad(Array(3, 2, 1)) = 0
    expectedGrad(Array(3, 2, 2)) = 0
    expectedGrad(Array(3, 2, 3)) = -0.16666666666666666
    assert(abs(expectedOutput - output) < 1e-6)
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
  }
}

