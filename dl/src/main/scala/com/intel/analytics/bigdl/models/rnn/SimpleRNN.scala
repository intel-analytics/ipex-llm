/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
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
  outputSize: Int,
  bpttTruncate: Int = 4)
  : Module[Float] = {
    val model = Sequential[Float]()
    model.add(Recurrent[Float](hiddenSize, bpttTruncate)
      .add(RnnCell[Float](inputSize, hiddenSize))
      .add(Tanh[Float]()))
      .add(Select(1, 1))
      .add(Linear[Float](hiddenSize, outputSize))
    model
  }
}
