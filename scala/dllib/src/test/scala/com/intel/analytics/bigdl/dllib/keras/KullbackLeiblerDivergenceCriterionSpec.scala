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

class KullbackLeiblerDivergenceCriterionSpec extends KerasBaseSpec {

  "KullbackLeiblerDivergenceCriterion" should "match Keras for batch input" in {
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

  "KullbackLeiblerDivergenceCriterion" should "match Keras for values out of clip boundary" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3])
        |target_tensor = Input(shape=[3])
        |loss = kullback_leibler_divergence(target_tensor, input_tensor)
        |input = np.random.uniform(-1, 2, [2, 3])
        |Y = np.random.uniform(-1, 2, [2, 3])
      """.stripMargin
    val kld = new KullbackLeiblerDivergenceCriterion[Float]()
    checkOutputAndGradForLoss(kld, kerasCode)
  }

  "KullbackLeiblerDivergenceCriterion" should "be ok with input close to epsilon" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3])
        |target_tensor = Input(shape=[3])
        |loss = kullback_leibler_divergence(target_tensor, input_tensor)
        |input = np.array([[1e-8, 1e-7, 1e-6]])
        |Y = np.array([[1.0, 1.0, 1.0]])
      """.stripMargin
    val criterion = KullbackLeiblerDivergenceCriterion[Float]()
    checkOutputAndGradForLoss(criterion, kerasCode)
  }

}
