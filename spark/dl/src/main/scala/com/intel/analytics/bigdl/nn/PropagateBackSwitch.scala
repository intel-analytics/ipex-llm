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
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Support disable backpropagation, a gradInput of null is returned if isPropagateBack = false
 * Usage: PropagateBackSwitch(module)
 * @param module the module you want to control the back-propagation
 * @tparam T Numeric type. Only support float/double now
 */
@SerialVersionUID(-7754615386732745708L)
class PropagateBackSwitch[T: ClassTag](module: Module[T])
  (implicit ev: TensorNumeric[T]) extends Container[Activity, Activity, T] {

  add(module)

  var isPropagateBack: Boolean = false

  def setPropagateBack(back: Boolean): this.type = {
    isPropagateBack = back
    this
  }

  override def updateOutput(input: Activity): Activity = {
    output = module.updateOutput(input)
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    if (isPropagateBack) {
      gradInput = module.updateGradInput(input, gradOutput)
    } else {
      gradInput = null
    }
    gradInput
  }

  override def toString: String = s"${ getPrintName }($isPropagateBack)"
}

object PropagateBackSwitch {
  def apply[@specialized(Float, Double) T: ClassTag](module: Module[T])
    (implicit ev: TensorNumeric[T]): PropagateBackSwitch[T] = new PropagateBackSwitch(module)
}


