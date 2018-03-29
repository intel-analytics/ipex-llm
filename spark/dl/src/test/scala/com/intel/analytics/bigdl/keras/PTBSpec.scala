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

import com.intel.analytics.bigdl.example.languagemodel.PTBModel
import com.intel.analytics.bigdl.tensor.Tensor

class PTBSpec extends KerasBaseSpec {

  "PTB model" should "generate the correct outputShape" in {
    val ptb = PTBModel.keras(10001, 650, 10001, 2)
    ptb.getOutputShape().toSingle().toArray should be (Array(-1, 35, 10001))
  }

  "PTB Keras-Style definition" should "be the same as Torch-Style definition" in {
    val kmodel = PTBModel.keras(10001, 650, 10001, 2)
    val tmodel = PTBModel(10001, 650, 10001, 2)
    val input = Tensor[Float](Array(10, 35)).fill(1f)
    compareModels(kmodel, tmodel, input)
  }

}
