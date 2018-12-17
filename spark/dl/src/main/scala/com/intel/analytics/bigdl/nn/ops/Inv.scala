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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildCard, TensorNumeric}
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class Inv[T: ClassTag, D: ClassTag]()(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Tensor[D], Tensor[D], T] {
  output = Tensor[D]()

  override def updateOutput(input: Tensor[D]): Tensor[D] = {
    output.resizeAs(input).copy(input).inv()
    output
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

object Inv {
  def apply[T: ClassTag, D: ClassTag]()(
    implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]): Inv[T, D] = new Inv()
}

private[bigdl] class InvGrad[T: ClassTag, D: ClassTag]()
  (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]) extends Operation[Table, Tensor[D], T] {
  output = Tensor[D]()

  override def updateOutput(input: Table): Tensor[D] = {
    require(input.length() == 2, "InvGrad requires two tensors as input")
    val x = input[Tensor[D]](1)
    val d = input[Tensor[D]](2)

    if (d.getType() != output.getType()) {
      output = d.emptyInstance()
    }
    output.resizeAs(x)
    output.copy(x).pow(ev2.fromType(2)).cmul(d).mul(ev2.fromType(-1))
    output
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

private[bigdl] object InvGrad {
  def apply[T: ClassTag, D: ClassTag]()(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  : InvGrad[T, D] = new InvGrad()
}
