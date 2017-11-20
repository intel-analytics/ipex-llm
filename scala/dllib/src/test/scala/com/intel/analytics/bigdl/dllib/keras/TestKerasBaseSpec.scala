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

import com.intel.analytics.bigdl.nn.{Linear, MSECriterion, ReLU}
import com.intel.analytics.bigdl.tensor.Tensor

class TestKerasBaseSpec extends KerasBaseSpec{

  "Layer with weights" should "be ok" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3])
        |input = np.random.uniform(0, 1, [1, 3])
        |output_tensor = Dense(2, init='uniform')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val linear = new Linear[Float](3, 2)
    def weightConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] = Array(in(0).t(), in(1))
    checkOutputAndGrad(linear, kerasCode, weightConverter)

  }

  "Layer without weights" should "be ok" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[2])
        |input = np.random.uniform(0, 1, [1, 2])
        |output_tensor = Activation(activation="relu")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val relu = new ReLU[Float]()
    checkOutputAndGrad(relu, kerasCode)
  }

  "MSE loss" should "be ok" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3])
        |target_tensor = Input(shape=[3])
        |loss = mean_squared_error(input_tensor, target_tensor)
        |input = np.random.uniform(0, 1, [2, 3])
        |Y = np.random.uniform(0, 1, [2, 3])
      """.stripMargin
    val mse = new MSECriterion[Float]()
    checkOutputAndGradForLoss(mse, kerasCode)
  }
}
