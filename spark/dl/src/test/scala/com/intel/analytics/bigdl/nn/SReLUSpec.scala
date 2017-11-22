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

class SReLUSpec extends KerasBaseSpec {

  def multiWeightsConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] = {
    val shape = Array(in.length) ++ in(0).size()
    val len = in(0).nElement()
    val weight = Tensor[Float]().resize(shape)

    var i = 0
    while (i < in.length) {
      require(len == in(i).nElement(), s"the shape must be same as the first.")

      System.arraycopy(in(i).storage().array(), in(i).storageOffset() - 1,
        weight.storage().array(), weight.storageOffset() - 1 + i * len, len)
      i += 1
    }

    Array(weight)
  }

  "SReLU without share axes" should "same as keras" in {
    val keras =
      """
        |input_tensor = Input(shape=[3, 4])
        |input = np.random.uniform(-1, 1, [2, 3, 4])
        |output_tensor = SReLU()(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin

    val srelu = SReLU[Float]()
    checkOutputAndGrad(srelu, keras, multiWeightsConverter)
  }

  "SReLU with share axes" should "same as keras" in {

    val keras =
      """
        |input_tensor = Input(shape=[2, 3, 4])
        |input = np.random.uniform(-1, 1, [5, 2, 3, 4])
        |share_axes = [1, 2]
        |output_tensor = SReLU()(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin

    val srelu = SReLU[Float](Array(1, 2))
    checkOutputAndGrad(srelu, keras, multiWeightsConverter)
  }
}
