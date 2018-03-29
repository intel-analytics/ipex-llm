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

import com.intel.analytics.bigdl.models.inception._
import com.intel.analytics.bigdl.tensor.Tensor

class Inception_v1Spec extends KerasBaseSpec {

  "Inception_v1_NoAuxClassifier" should "generate the correct outputShape" in {
    val inception = Inception_v1_NoAuxClassifier.keras(classNum = 1000)
    inception.getOutputShape().toSingle().toArray should be (Array(-1, 1000))
  }

  "Inception_v1_NoAuxClassifier forward and backward" should "work properly" in {
    val inception = Inception_v1_NoAuxClassifier.keras(classNum = 1000)
    val input = Tensor[Float](Array(10, 3, 224, 224)).rand()
    val output = inception.forward(input)
    val gradInput = inception.backward(input, output)
  }

}
