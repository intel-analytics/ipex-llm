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
package com.intel.analytics.bigdl.keras
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, AbstractModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.BigDLSpecHelper

import scala.sys.process._

abstract class KerasBaseSpec extends BigDLSpecHelper {

  protected def ifskipTest(): Unit = {
    // Skip unitest if environment is not ready
    try {
    Seq("python", "-c", "import keras; import tensorflow").!!
    } catch {
      case e: Throwable => cancel("python or keras or tensorflow is not installed", e)
    }
  }

  private def defaultWeightConverter(in: Array[Tensor[Float]]) = in

  // weightConverter: convert keras weight to BigDL format,
  // do nothing for the default converter
  def checkOutputAndGrad(bmodel: AbstractModule[Tensor[Float], Tensor[Float], Float],
                         kerasCode: String,
                         weightConverter: (Array[Tensor[Float]]) => Array[Tensor[Float]]
                         = defaultWeightConverter,
                         precision: Double = 1e-5): Unit = {
    ifskipTest()
    val (gradInput, gradWeight, weights, input, target, output) = KerasRunner.run(kerasCode)
    // Ensure they share the same weights
    if (weights != null) {
      bmodel.setWeightsBias(weightConverter(weights))
    }

    val boutput = bmodel.forward(input)
    boutput.almostEqual(output, precision) should be(true)

    val bgradInput = bmodel.backward(input, boutput.clone())
    bgradInput.almostEqual(gradInput, precision) should be(true)

    val parameters = bmodel.parameters()
    if (gradWeight != null) {
      val bgradWeights = parameters._2
      (bgradWeights, weightConverter(gradWeight)).zipped.foreach { (bgrad, kgrad) =>
        bgrad.almostEqual(kgrad, precision) should be(true)
      }
    }
  }

  def checkOutputAndGradForLoss(bmodel: AbstractCriterion[Tensor[Float], Tensor[Float], Float],
                                kerasCode: String,
                                precision: Double = 1e-5): Unit = {
    ifskipTest()

    val (gradInput, gradWeight, weights, input, target, output) =
      KerasRunner.run(kerasCode, Loss)

    val boutput = bmodel.forward(input, target)
    val koutput = output.mean()  // the return value from keras is not always averaged.
    NumericFloat.nearlyEqual(boutput, koutput, precision) should be(true)
    val kgradInput = gradInput.div(output.nElement())  // div is an in-place operation.
    val bgradInput = bmodel.backward(input, target.clone())
    bgradInput.almostEqual(kgradInput, precision) should be(true)
  }
}

