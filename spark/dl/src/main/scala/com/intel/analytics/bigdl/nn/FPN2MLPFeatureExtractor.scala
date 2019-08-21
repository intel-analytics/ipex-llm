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

import breeze.linalg.dim
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}
import org.dmg.pmml.{False, True}

class FPN2MLPFeatureExtractor(inChannels: Int, resolution: Int,
  scales: Array[Float], samplingRatio: Int, representationSize: Int)
  (implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Tensor[Float], Float] {

  val pooler = new Pooler(resolution, scales, samplingRatio)
  val input_size = inChannels * math.pow(resolution, 2)
  val fc6 = Linear[Float](input_size.toInt, representationSize, withBias = true)
  val fc7 = Linear[Float](representationSize, representationSize, withBias = true)
  fc6.setInitMethod(Xavier)
  fc6.bias.fill(ev.zero)
  fc7.setInitMethod(Xavier)
  fc7.bias.fill(ev.zero)

  // todo:
  fc6.weight.fill(0.001f)
  fc7.weight.fill(0.001f)

  val model = Sequential[Float]()
            .add(fc6)
            .add(ReLU[Float]())
            .add(fc7)
            .add(ReLU[Float]())

  // input contains:
  // 1th features, and 2th proposals
  override def updateOutput(input: Table): Tensor[Float] = {
    val features = input[Table](1)
    val proposals = input[Tensor[Float]](2)

    val x = this.pooler.forward(T(features, proposals))
    val y = x.view(Array(x.size(1), x.nElement() / x.size(1)))

    output = model.forward(y).toTensor[Float]
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[Float]): Table = {
    throw new UnsupportedOperationException("FPN2MLPFeatureExtractor only support inference")
  }
}
