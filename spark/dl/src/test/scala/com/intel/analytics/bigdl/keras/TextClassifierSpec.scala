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

import com.intel.analytics.bigdl.example.utils._
import com.intel.analytics.bigdl.tensor.Tensor

class TextClassifierSpec extends KerasBaseSpec {

  "TextClassifier model" should "generate the correct outputShape" in {
    val textclassifier = new TextClassifier(TextClassificationParams()).buildKerasModel(20)
    textclassifier.getOutputShape().toSingle().toArray should be (Array(-1, 20))
  }

  "TextClassifier model forward and backward" should "work properly" in {
    val input = Tensor[Float](Array(32, 500, 200)).rand()
    val textclassifier = new TextClassifier(TextClassificationParams()).buildKerasModel(20)
    val output = textclassifier.forward(input)
    val gradInput = textclassifier.backward(input, output)
  }

}
