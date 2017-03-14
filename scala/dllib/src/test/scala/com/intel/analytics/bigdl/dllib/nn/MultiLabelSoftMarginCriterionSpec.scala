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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class MultiLabelSoftMarginCriterionSpec  extends FlatSpec with Matchers {
  "hashcode()" should "behave correctly" in {
    val m1 = new MultiLabelSoftMarginCriterion[Double]()
    val m2 = new MultiLabelSoftMarginCriterion[Double]()
    val m3 = new MultiLabelSoftMarginCriterion[Double](Tensor[Double](3).randn())
    val m4 = new MultiLabelSoftMarginCriterion[Double]()
    val bce = new BCECriterion[Double]()
    val input = Tensor[Double](3, 3).randn()
    val target = Tensor[Double](3, 3).randn()
    m4.forward(input, target)


    m1.hashCode() should equal (m2.hashCode())
    m1.hashCode() should not equal null
    m1.hashCode() should not equal bce.hashCode()
    m1.hashCode() should not equal m3.hashCode()
    m1.hashCode() should not equal m4.hashCode()
  }

  "equals()" should "behave correctly" in {
    val m1 = new MultiLabelSoftMarginCriterion[Double]()
    val m2 = new MultiLabelSoftMarginCriterion[Double]()
    val m3 = new MultiLabelSoftMarginCriterion[Double](Tensor[Double](3).randn())
    val m4 = new MultiLabelSoftMarginCriterion[Double]()
    val bce = new BCECriterion[Double]()
    val input = Tensor[Double](3, 3).randn()
    val target = Tensor[Double](3, 3).randn()
    m4.forward(input, target)


    m1 should equal (m2)
    m1 should not equal null
    m1 should not equal null.isInstanceOf[MultiLabelSoftMarginCriterion[Double]]
    m1 should not equal bce
    m1 should not equal m3
    m1 should not equal m4
  }

  "MultiLabelSoftMarginCriterion " should "return return right output and gradInput" in {
    val criterion = new MultiLabelSoftMarginCriterion[Double]()
    val output = Tensor[Double](3)
    output(Array(1)) = 0.4
    output(Array(2)) = 0.5
    output(Array(3)) = 0.6
    val target = Tensor[Double](3)
    target(Array(1)) = 0
    target(Array(2)) = 1
    target(Array(3)) = 1

    val loss = criterion.forward(output, target)
    loss should be(0.608193395686766 +- 1e-8)
    val gradInput = criterion.backward(output, target)
    gradInput(Array(1)) should be(0.19956255336948944 +- 0.0001)
    gradInput(Array(2)) should be(-0.12584688959851295 +- 0.0001)
    gradInput(Array(3)) should be(-0.11811456459055192 +- 0.0001)
  }
}
