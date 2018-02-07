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

import com.intel.analytics.bigdl.nn.LocallyConnected2D
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Shape, TestUtils}


class LocallyConnected2DSpec extends KerasBaseSpec {
  "LocallyConnected2D NHWC Float" should "be ok" in {
    ifskipTest()
    val kerasCode =
      """
        |input_tensor = Input(shape=[3,6,2])
        |input = np.array([[[[1,2], [2,3], [3,4],[4,5],[5,6],[6,7]],
        | [[2,3], [3,4],[4,5],[5,6],[6,7], [1,2]],
        | [[1,2], [2,3], [3,4],[4,5],[6,7],[5,6]]]])
        |output_tensor = LocallyConnected2D(3, 2, 1, dim_ordering="tf",
        |input_shape=(3,6,2))(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val locallyConnected2d =
      LocallyConnected2D[Float](2, 6, 3, 3, 1, 2, format = DataFormat.NHWC)
    val a = locallyConnected2d.parameters()


    val wc = (data: Array[Tensor[Float]]) => {

      val out = new Array[Tensor[Float]](data.length)
      val d1l: Int = data(0).size(1)
      val d2l: Int = data(0).size(2)
      val d3l: Int = data(0).size(3)

      out(0) = Tensor(d1l, d3l, d2l)

      val page: Int = d2l * d3l
      for (i <- 0 to d1l * d2l * d3l - 1) {
        val d1 = i / page + 1
        val d2 = (i % page) / (d3l) + 1
        val d3 = (i % page) % d3l + 1
        val v = data(0).valueAt(d1, d2, d3)
        out(0).setValue(d1, d3, d2, v)
      }

      if (data.length > 1) {
        out(1) = data(1)
      }
      out
    }

    checkOutputAndGrad(locallyConnected2d, kerasCode, wc)

  }

  "LocallyConnected1D computeOutputShape NCHW" should "work properly" in {
    val layer = LocallyConnected2D[Float](3, 12, 12, 3, 2, 2, 2, 1)
    TestUtils.compareOutputShape(layer, Shape(3, 12, 12)) should be (true)
  }

  "LocallyConnected2D computeOutputShape NHWC" should "work properly" in {
    val layer = LocallyConnected2D[Float](2, 16, 12, 4, 1, 2, format = DataFormat.NHWC)
    TestUtils.compareOutputShape(layer, Shape(12, 16, 2)) should be (true)
  }
}
