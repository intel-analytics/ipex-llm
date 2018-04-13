/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.layers.extra

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.layers.Activation

class ActivationSpec extends ZooSpecHelper {

  "ReLU6 Zoo" should "be the same as BigDL" in {
    val blayer = ReLU6[Float]()
    val zlayer = Activation[Float]("relu6", inputShape = Shape(4, 5))
    zlayer.build(Shape(-1, 4, 5))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 4, 5))
    val input = Tensor[Float](Array(2, 4, 5)).rand()
    compareOutputAndGradInput(blayer, zlayer, input)
  }

  "TanhShrink Zoo" should "be the same as BigDL" in {
    val blayer = TanhShrink[Float]()
    val zlayer = Activation[Float]("tanh_shrink", inputShape = Shape(4, 5))
    zlayer.build(Shape(-1, 4, 5))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 4, 5))
    val input = Tensor[Float](Array(2, 4, 5)).rand()
    compareOutputAndGradInput(blayer, zlayer, input)
  }

  "SoftMin Zoo" should "be the same as BigDL" in {
    val blayer = SoftMin[Float]()
    val zlayer = Activation[Float]("softmin", inputShape = Shape(4, 5))
    zlayer.build(Shape(-1, 4, 5))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 4, 5))
    val input = Tensor[Float](Array(2, 4, 5)).rand()
    compareOutputAndGradInput(blayer, zlayer, input)
  }

  "LogSigmoid Zoo" should "be the same as BigDL" in {
    val blayer = LogSigmoid[Float]()
    val zlayer = Activation[Float]("log_sigmoid", inputShape = Shape(4, 5))
    zlayer.build(Shape(-1, 4, 5))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 4, 5))
    val input = Tensor[Float](Array(2, 4, 5)).rand()
    compareOutputAndGradInput(blayer, zlayer, input)
  }

  "LogSoftMax Zoo" should "be the same as BigDL" in {
    val blayer = LogSoftMax[Float]()
    val zlayer = Activation[Float]("log_softmax", inputShape = Shape(10))
    zlayer.build(Shape(-1, 10))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 10))
    val input = Tensor[Float](Array(2, 10)).rand()
    compareOutputAndGradInput(blayer, zlayer, input)
  }

}
