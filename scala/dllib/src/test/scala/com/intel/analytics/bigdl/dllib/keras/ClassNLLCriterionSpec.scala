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

import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

class ClassNLLCriterionSpec extends KerasBaseSpec {

  "ClassNLLCriterion log" should "be the same as Keras sparse_categorical_crossentropy" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, ])
        |target_tensor = Input(batch_shape=[3, ])
        |loss = sparse_categorical_crossentropy(target_tensor, input_tensor)
        |input = input = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])
        |Y = np.array([0.0, 1.0, 2.0])
      """.stripMargin
    val loss = ClassNLLCriterion[Float](logProbAsInput = false)
    val (gradInput, gradWeight, weights, input, target, output) =
      KerasRunner.run(kerasCode, Loss)
    val boutput = loss.forward(input, target + 1) // index in BigDL starts from 1
    val koutput = output.mean()
    NumericFloat.nearlyEqual(boutput, koutput, 1e-5) should be (true)
  }

}
