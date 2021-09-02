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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.{Reverse, Select => TSelect, Sequential => TSequential}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.{Recurrent => BKerasRecurrent}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.layers.internal.InternalRecurrent

import scala.reflect.ClassTag

abstract class Recurrent[T: ClassTag](
    override val outputDim: Int,
    override val returnSequences: Boolean = false,
    override val goBackwards: Boolean = false,
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BKerasRecurrent[T](outputDim, returnSequences, goBackwards, inputShape) {

  val rec = new InternalRecurrent[T]()

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val model = TSequential[T]()
    if (goBackwards) model.add(Reverse(2))
    rec.add(buildCell(input))
    model.add(rec)
    if (!returnSequences) model.add(TSelect(2, -1))
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }

  def getHiddenShape(): Array[Int] = {
    require(this.isBuilt(), "Cannot getHiddenShape before call doBuild!")
    rec.getHiddenShape()
  }

  def getHiddenState(): Activity = {
    rec.getHiddenState()
  }

  def setHiddenState(hiddenState: Activity): Unit = {
    rec.setHiddenState(hiddenState)
  }

  def getGradHiddenState(): Activity = {
    rec.getGradHiddenState()
  }

  def setGradHiddenState(gradHiddenState: Activity): Unit = {
    rec.setGradHiddenState(gradHiddenState)
  }
}
