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

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Shape, T, Table}

import scala.reflect.ClassTag

/**
 * NormalizeScale is conposed of normalize and scale, this is equal to caffe Normalize layer
 * @param p L_p norm
 * @param eps smoothing parameter
 * @param scale scale parameter
 * @param size size of scale input
 * @param wRegularizer weight regularizer
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
@SerialVersionUID(8394549762420197622L)
class NormalizeScale[T: ClassTag](val p: Double, val eps: Double = 1e-10,
  val scale: Double, val size: Array[Int],
  var wRegularizer: Regularizer[T] = null)(implicit ev: TensorNumeric[T])
  extends TensorModule[T] {
  val normalize = Normalize[T](p, eps)
  val cmul = CMul[T](size, wRegularizer = wRegularizer)
  cmul.weight.fill(ev.fromType(scale))

  override def setScaleW(w: Double): this.type = {
    cmul.setScaleW(w)
    this
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    normalize.forward(input)
    output = cmul.forward(normalize.output)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = cmul.updateGradInput(output, normalize.updateGradInput(input, gradOutput))
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    cmul.accGradParameters(input, gradOutput)
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(cmul.weight), Array(cmul.gradWeight))
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val outShape = normalize.computeOutputShape(inputShape)
    cmul.computeOutputShape(outShape)
  }
}

object NormalizeScale {
  def apply[@specialized(Float, Double) T: ClassTag]
  (p: Double, eps: Double = 1e-10, scale: Double, size: Array[Int],
    wRegularizer: Regularizer[T] = null)
    (implicit ev: TensorNumeric[T]): NormalizeScale[T] =
    new NormalizeScale[T](p, eps, scale, size, wRegularizer)
}
