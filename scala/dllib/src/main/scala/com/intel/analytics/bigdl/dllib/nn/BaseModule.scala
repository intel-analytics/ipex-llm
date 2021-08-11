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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag


private[nn] abstract class BaseModule[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends AbstractModule[Activity, Activity, T] {

  private[bigdl] var model : Module[T] = buildModel()

  def buildModel(): Module[T]

  override def updateOutput(input: Activity): Activity = {
    output = model.updateOutput(input)
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput = model.updateGradInput(input, gradOutput)
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    model.accGradParameters(input, gradOutput)
  }

  override def training(): this.type = {
    train = true
    model.training()
    this
  }

  override def evaluate(): this.type = {
    train = false
    model.evaluate()
    this
  }

  override def getExtraParameter(): Array[Tensor[T]] = {
    model.getExtraParameter()
  }

  override def getTimes(): Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)] = {
    model.getTimes()
  }

  override def resetTimes(): Unit = {
    model.resetTimes()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    model.parameters()
  }

  override def getParametersTable(): Table = {
    model.getParametersTable()
  }

  override def clearState(): this.type = {
    model.clearState()
    this
  }
}
