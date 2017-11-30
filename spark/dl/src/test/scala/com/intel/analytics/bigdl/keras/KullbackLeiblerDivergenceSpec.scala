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

import com.intel.analytics.bigdl.nn.KullbackLeiblerDivergenceCriterion
import com.intel.analytics.bigdl.tensor.Tensor

class KullbackLeiblerDivergenceSpec extends KerasBaseSpec {

  "KullbackLeiblerDivergenceCriterion" should "be ok" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3])
        |target_tensor = Input(shape=[3])
        |loss = kullback_leibler_divergence(target_tensor, input_tensor)
        |input = np.random.uniform(0, 1, [2, 3])
        |Y = np.random.uniform(0, 1, [2, 3])
      """.stripMargin
    val kld = new KullbackLeiblerDivergenceCriterion[Float]()
    checkOutputAndGradForLoss(kld, kerasCode)
  }

  "KullbackLeiblerDivergenceCriterion" should "be ok with epsilon" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3])
        |target_tensor = Input(shape=[3])
        |loss = mean_squared_logarithmic_error(input_tensor, target_tensor)
        |input = np.array([[1e-07, 1e-06, 1e-08]])
        |Y = np.array([[1, 2, 3]])
      """.stripMargin
    val criterion = KullbackLeiblerDivergenceCriterion[Float]()
    checkOutputAndGradForLoss(criterion, kerasCode)
  }

  "KullbackLeiblerDivergenceCriterion" should "be ok with 1" in {
    val criterion = KullbackLeiblerDivergenceCriterion[Float]()
    val input = Tensor[Float](Array(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f), Array(2, 3))
    val target = Tensor[Float](Array(0.2f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f), Array(2, 3))
    val loss = criterion.forward(input, target)
    println(loss)

    val gradient = criterion.updateGradInput(input, target)
    println(gradient)
  }

}