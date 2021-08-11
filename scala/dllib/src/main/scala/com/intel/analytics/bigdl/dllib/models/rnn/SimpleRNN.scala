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

package com.intel.analytics.bigdl.models.rnn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn._

object SimpleRNN {
  def apply(
  inputSize: Int,
  hiddenSize: Int,
  outputSize: Int)
  : Module[Float] = {
    val model = Sequential[Float]()
    model.add(Recurrent[Float]()
      .add(RnnCell[Float](inputSize, hiddenSize, Tanh[Float]())))
      .add(TimeDistributed[Float](Linear[Float](hiddenSize, outputSize)))
    model
  }
}
