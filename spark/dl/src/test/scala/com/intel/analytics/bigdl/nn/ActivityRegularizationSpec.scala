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

import com.intel.analytics.bigdl.keras.{KerasBaseSpec, KerasRunner, Regularizer}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class ActivityRegularizationSpec extends KerasBaseSpec {
  "ActivityRegularization" should "same as keras" in {
    ifskipTest()

    val keras =
      """
        |act_reg = core.ActivityRegularization(l1=0.01, l2=0.01)
        |
        |input_tensor = Input(shape=(2,))
        |output_tensor = act_reg(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
        |
        |input = np.random.random((2, 2))
        |loss = model.losses
        |
        |Y = []
      """.stripMargin

    val ar = ActivityRegularization[Float](0.01, 0.01)

    val (gradInput, gradWeight, weights, input, target, output) = KerasRunner.run(keras,
      Regularizer)

    val boutput = ar.forward(input)
    boutput.almostEqual(output, 1e-5) should be(true)

    ar.loss.toDouble should be (target.value().toDouble +- 1e-5)

    val bgradInput = ar.backward(input, boutput.clone())
    bgradInput.almostEqual(gradInput, 1e-5) should be(true)
  }
}

class ActivityRegularizationSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val activityRegularization = ActivityRegularization[Float](l1 = 0.01, l2 = 0.01).
      setName("activityRegularization")
    val input = Tensor[Float](5, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(activityRegularization, input)
  }
}
