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

import com.intel.analytics.bigdl.nn.BCECriterion

class BCECriterionSpec extends KerasBaseSpec {

  "BCECriterion" should "be the same as Keras binary_crossentropy" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 4])
        |target_tensor = Input(shape=[3, 4])
        |loss = binary_crossentropy(target_tensor, input_tensor)
        |input = np.random.random([2, 3, 4])
        |Y = np.random.random([2, 3, 4])
      """.stripMargin
    val mse = new BCECriterion[Float]()
    checkOutputAndGradForLoss(mse, kerasCode)
  }

}
