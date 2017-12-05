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

import com.intel.analytics.bigdl.keras.KerasBaseSpec
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.serializer.{ModuleLoader, ModulePersister}

class SReLUSpec extends KerasBaseSpec {
  "SReLU without share axes" should "same as keras" in {
    val keras =
      """
        |input_tensor = Input(shape=[3, 4])
        |input = np.random.uniform(-1, 1, [2, 3, 4])
        |output_tensor = SReLU()(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin

    val srelu = SReLU[Float]()
    checkOutputAndGrad(srelu, keras)
  }

  "SReLU with share axes" should "same as keras" in {

    val keras =
      """
        |input_tensor = Input(shape=[2, 3, 4])
        |input = np.random.uniform(-1, 1, [5, 2, 3, 4])
        |share_axes = [1, 2]
        |output_tensor = SReLU(shared_axes = share_axes)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin

    val srelu = SReLU[Float](Array(1, 2))
    checkOutputAndGrad(srelu, keras)
  }

  "SReLU with share axes not contiguous" should "same as keras" in {

    val keras =
      """
        |input_tensor = Input(shape=[2, 3, 4, 5])
        |input = np.random.uniform(-1, 1, [6, 2, 3, 4, 5])
        |share_axes = [2, 4]
        |output_tensor = SReLU(shared_axes = share_axes)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin

    val srelu = SReLU[Float](Array(2, 4))
    checkOutputAndGrad(srelu, keras)
  }

  // do not delete this, it's for testing the initialization of SReLU
  "SReLU init" should "same as keras" in {
    val srelu = SReLU[Float]()
    val input = Tensor[Float](5, 2, 3, 4).randn()
    srelu.forward(input)
    println(srelu.output)
  }
}
