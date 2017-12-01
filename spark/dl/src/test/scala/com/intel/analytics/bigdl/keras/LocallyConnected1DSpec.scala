// scalastyle:off
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
// * limitations under the License.
 */

package com.intel.analytics.bigdl.keras

import com.intel.analytics.bigdl.nn.{HardSigmoid, LocallyConnected1D}
import com.intel.analytics.bigdl.tensor.Tensor


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


class LocallyConnected1DSpec extends KerasBaseSpec {

  "LocallyConnected1D" should "be ok" in {
    ifskipTest()
    val kerasCode =
      """
        |input_tensor = Input(shape=[100,10])
        |input = np.random.rand(1,100,10)
        |output_tensor = LocallyConnected1D(8,5,subsample_length=2,input_shape=(100,10))(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val locallyConnected1d = LocallyConnected1D[Float](100, 10, 8, 5, 2)
    checkOutputAndGrad(locallyConnected1d, kerasCode)


    val inputSize = 10
    val maxoutNumber = 2
    val outputSize = 2

    val wc = (data: Array[Tensor[Float]]) => {

      val output = Tensor[Float](data.length,data(0).size(0),data(0).size(1))

      val out = new Array[Tensor[Float]](data.length)
      out(0) = Tensor(inputSize, maxoutNumber * outputSize)
      val weight = out.head.storage().array()
      var index = 0
      for (i <- 1 to maxoutNumber) {
        val sliceW = data(0).select(1, i).t.clone().storage().array()
        System.arraycopy(sliceW, 0, weight, index, sliceW.size)
        index += sliceW.size
      }

      if (data.length > 1) {
        out(1) = data(1)
      }
      out
    }

  }

}