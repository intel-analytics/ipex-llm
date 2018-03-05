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

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.keras.Dense
import com.intel.analytics.bigdl.nn.ops.{BucketizedCol, CrossEntropy}
import com.intel.analytics.bigdl.optim.L2Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

class ModuleInitializerSpec extends FlatSpec with Matchers {

  "ModuleInitializer" should "init a Module correctly" in {
    val state = T("name" -> "Linear", "inputSize" -> 10, "outputSize" -> 5)
    var linear = Module[Float](state).asInstanceOf[Linear[Float]]
    linear.withBias shouldEqual true
    linear.forward(Tensor[Float](2, 10).rand()).size() shouldEqual Array(2, 5)

    state.update("withBias", false)
    state.update("wRegularizer", new L2Regularizer[Float](1e-3))
    state.update("initWeight", Tensor.ones[Float](5, 10))
    linear = Module[Float](state).asInstanceOf[Linear[Float]]
    linear.withBias shouldEqual false
    linear.wRegularizer should not be null
    linear.weight.storage().array().forall(_ == 1.0f) shouldEqual true
  }

  "ModuleInitializer" should "init a Module by indices-style Table correctly" in {
    val linear = Module[Float](
      T("Linear", 10, 5, false)).asInstanceOf[Linear[Float]]
    linear.withBias shouldEqual false
    linear.inputSize shouldEqual 10
    linear.outputSize shouldEqual 5
    linear.forward(Tensor[Float](2, 10).rand()).size() shouldEqual Array(2, 5)
  }

  "ModuleInitializer" should "init a KerasLayer correctly" in {
    val state = T(
      "name" -> "keras.Dense", "outputDim" -> 5,
      "inputShape" -> SingleShape(List(100, 10)),
      "init" -> Ones,
      "activation" -> ReLU[Double]()
    )
    val dense = Module[Double](state).asInstanceOf[Dense[Double]]
    dense.outputDim shouldEqual 5
    dense.activation.getClass.getName.contains("ReLU") shouldEqual true
    dense.inputShape.toSingle() shouldEqual List(100, 10)

    dense.build(dense.inputShape)
    val seq = dense.modules.head.asInstanceOf[Sequential[Double]]
    val linear = seq.modules(1).asInstanceOf[Linear[Double]]
    seq.modules.last.asInstanceOf[ReLU[Double]]
    linear.weight.storage().array().forall(_ == 1.0f) shouldEqual true
    seq.forward(Tensor[Double](5, 10)).toTensor[Double].size() shouldEqual Array(5, 5)
  }

  "ModuleInitializer" should "init a Operation correctly" in {
    val crossEntropy = Module[Float](
      T("name" -> "ops.CrossEntropy")).asInstanceOf[CrossEntropy[Float]]
    crossEntropy.updateOutput(
      T(Tensor[Float](3, 5).rand(), Tensor[Float](3, 5).rand()))

    val bucketizedCol = Module[Double](T(
      "name" -> "ops.BucketizedCol",
      "boundaries" -> Array(0.0, 10.0, 100.0)
    )).asInstanceOf[BucketizedCol[Double]]
    val input = Tensor[Double](T(T(-1, 1), T(101, 10), T(5, 100)))
    val expectOutput = Tensor[Int](T(T(0, 1), T(3, 2), T(1, 3)))
    bucketizedCol.forward(input) shouldEqual expectOutput
  }

  "ModuleInitializer" should "init a Module with multiple type params correctly" in {
    val rdUniform = Module[Float](T(
      "name" -> "ops.RandomUniform",
      "D" -> "Double",
      "minVal" -> 10.0,
      "maxVal" -> 20.0
    )).asInstanceOf[ops.RandomUniform[Float, Double]]
    val input = Tensor[Int](T(1, 2, 3))
    rdUniform.forward(input)
  }

}
