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
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class KerasBaseSpec extends FlatSpec with BeforeAndAfter with Matchers {

  private def defaultWeightConverter(in: Array[Tensor[Float]]) = in

  private def getFieldByReflect(obj: Object, name: String): Object = {
    val fieldDefinition = obj.getClass().getDeclaredField(name)
    fieldDefinition.setAccessible(true)
    return fieldDefinition.get(obj)
  }
  // weightConverter: convert keras weight to BigDL format,
  // do nothing for the default converter
  def checkOutputAndGrad(bmodel: AbstractModule[Tensor[Float], Tensor[Float], Float],
                         kerasCode: String,
                         weightConverter: (Array[Tensor[Float]]) => Array[Tensor[Float]]
                         = defaultWeightConverter,
                         precision: Double = 1e-5): Unit = {
    val (gradInput, gradWeight, weights, input, output) = KerasRunner.run(kerasCode)
    // Ensure they share the same weights
    if (weights != null) {
      bmodel.setWeightsBias(weightConverter(weights))
    }

    val boutput = bmodel.forward(input)
    boutput.almostEqual(output, precision) should be(true)

    val bgradInput = bmodel.backward(input, boutput.clone().fill(1))
    bgradInput.almostEqual(gradInput, precision) should be(true)

    // assuming the first one is weight, the second one is bias
    if (gradWeight != null) {
      val bgradWeight = getFieldByReflect(bmodel, "gradWeight").asInstanceOf[Tensor[Float]]
      bgradWeight.almostEqual(weightConverter(gradWeight)(0), precision) should be(true)

      if (gradWeight.length > 1) {
        val bgradBias = getFieldByReflect(bmodel, "gradBias").asInstanceOf[Tensor[Float]]
        bgradBias.almostEqual(weightConverter(gradWeight)(1), precision) should be(true)
      }
    }
  }
}

