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

import com.intel.analytics.bigdl.nn.MarginCriterion

class MarginCriterionSpec extends KerasBaseSpec {

  "MarginCriterion" should "be the same as Keras hinge" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 4])
        |target_tensor = Input(shape=[3, 4])
        |loss = hinge(target_tensor, input_tensor)
        |input = np.random.random([2, 3, 4])
        |Y = np.random.random([2, 3, 4])
      """.stripMargin
    val loss = MarginCriterion[Float]()
    checkOutputAndGradForLoss(loss, kerasCode)
  }

  "MarginCriterion squared" should "be the same as Keras squared_hinge" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 4])
        |target_tensor = Input(shape=[3, 4])
        |loss = squared_hinge(target_tensor, input_tensor)
        |input = np.random.random([2, 3, 4])
        |Y = np.random.random([2, 3, 4])
      """.stripMargin
    val loss = MarginCriterion[Float](squared = true)
    checkOutputAndGradForLoss(loss, kerasCode)
  }

}
