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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.utils.RandomGenerator._

class FPNPredictor(val numClass: Int, val inChannels: Int)
  (implicit ev: TensorNumeric[Float]) extends AbstractModule[Tensor[Float], Table, Float] {
  val cls_score = Linear[Float](inChannels, numClass)
  // todo: check with torch
  cls_score.weight.apply1(_ => RNG.normal(0, 0.01).toFloat)
  cls_score.bias.fill(0.0f)

  val bbox_pred = Linear[Float](inChannels, numClass * 4)
  bbox_pred.weight.apply1(_ => RNG.normal(0, 0.001).toFloat)
  bbox_pred.bias.fill(0.0f)

  override def updateOutput(input: Tensor[Float]): Table = {
    val scores = cls_score.forward(input)
    val bbox_deltas = bbox_pred.forward(input)
    output = T(scores, bbox_deltas)
    output
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Table): Tensor[Float] = {
    gradInput
  }
}
