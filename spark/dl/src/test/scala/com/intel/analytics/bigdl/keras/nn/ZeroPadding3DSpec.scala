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

package com.intel.analytics.bigdl.keras.nn

import com.intel.analytics.bigdl.keras.KerasBaseSpec
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.keras.ZeroPadding3D
import com.intel.analytics.bigdl.nn.keras.{Sequential => KSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape

class ZeroPadding3DSpec extends KerasBaseSpec {

  "ZeroPadding3D" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 4, 5, 6])
        |input = np.random.random([2, 3, 4, 5, 6])
        |output_tensor = ZeroPadding3D(padding=(1,1,1), dim_ordering='th')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = ZeroPadding3D[Float]((1, 1, 1), "CHANNEL_FIRST", inputShape = Shape(3, 4, 5, 6))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "ZeroPadding3D padding" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 4, 5, 6])
        |input = np.random.random([2, 3, 4, 5, 6])
        |output_tensor = ZeroPadding3D(padding=(2,1,3), dim_ordering='th')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = ZeroPadding3D[Float]((2, 1, 3), "CHANNEL_FIRST", inputShape = Shape(3, 4, 5, 6))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "ZeroPadding3D channel_last" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 4, 5, 6])
        |input = np.random.random([2, 3, 4, 5, 6])
        |output_tensor = ZeroPadding3D(padding=(1,1,1), dim_ordering='tf')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = ZeroPadding3D[Float]((1, 1, 1), "CHANNEL_LAST", inputShape = Shape(3, 4, 5, 6))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }
}
