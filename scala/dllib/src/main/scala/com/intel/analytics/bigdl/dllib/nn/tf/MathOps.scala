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
package com.intel.analytics.bigdl.nn.tf

import com.intel.analytics.bigdl.nn.ops.Operation
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class SqrtGrad[T: ClassTag, D: ClassTag](implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Table, Tensor[D], T] {

  output = Tensor[D]()

  override def updateOutput(inputs: Table): Tensor[D] = {
    val grads = inputs[Tensor[D]](2)
    val y = inputs[Tensor[D]](1)

    output.resizeAs(grads).copy(grads).mul(ev2.fromType(0.5)).div(y)

    output
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

object SqrtGrad {
  def apply[T: ClassTag, D: ClassTag]()
    (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]): SqrtGrad[T, D] = new SqrtGrad[T, D]()
}

private[bigdl] class RsqrtGrad[T: ClassTag, D: ClassTag]
(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]) extends Operation[Table, Tensor[D], T] {

  output = Tensor[D]()

  override def updateOutput(inputs: Table): Tensor[D] = {
    val grads = inputs[Tensor[D]](2)
    val y = inputs[Tensor[D]](1)

    output.resizeAs(y).copy(y).pow(ev2.fromType(3.0)).mul(ev2.fromType(-0.5f)).cmul(grads)

    output
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

private[bigdl] object RsqrtGrad {
  def apply[T: ClassTag, D: ClassTag]()
    (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]): RsqrtGrad[T, D] = new RsqrtGrad[T, D]()
}
