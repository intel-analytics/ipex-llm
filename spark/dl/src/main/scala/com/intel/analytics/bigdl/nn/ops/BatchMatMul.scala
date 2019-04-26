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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * Multiplies all slices of `Tensor` `x` and `y` (each slice can be
 * viewed as an element of a batch), and arranges the individual results
 * in a single output tensor of the same batch size. Each of the
 * individual slices can optionally be adjointed (to adjoint a matrix
 * means to transpose and conjugate it) before multiplication by setting
 * the `adj_x` or `adj_y` flag to `True`, which are by default `False`.
 *
 */

class BatchMatMul[T: ClassTag, D: ClassTag](
                                             val adjX: Boolean = false,
                                             val adjY: Boolean = false)
         (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Table, Tensor[D], T] {
  gradInput = T(Tensor[D], Tensor[D]())
  output = Tensor[D]()

  override def updateOutput(input: Table): Tensor[D] = {
    var x: Tensor[D] = input(1)
    var y: Tensor[D] = input(2)

    require(x.dim() == y.dim(), "tensor x and tensor y must have the same number of dims")
    require(x.dim() >= 2, "tensor dim num must be at least 2")

    if (x.dim() == 2) {
      require(y.dim() == 2, "second input tensor must be 2D" +
        s"second input dim ${y.dim()}")

      if (adjX) {
        x = x.t()
      }
      if (adjY) {
        y = y.t()
      }
      require(x.size(2) == y.size(1), "matrix sizes do not match" +
        s"The sizes are ${x.size(2)} and ${y.size(1)}")

      output.resize(x.size(1), y.size(2))
      output.mm(x, y)
    } else {

      require(x.size(1) == y.size(1), "inputs must contain the same number of minibatches" +
        s"The minibatces of each are ${x.size(1)} and ${y.size(1)}")

      val dimNum = x.dim()

      val batchSize = x.size().slice(0, dimNum - 2).product

      var reshapedX = x.view(Array(batchSize, x.size(dimNum - 1), x.size(dimNum)))
      var reshapedY = y.view(Array(batchSize, y.size(dimNum - 1), y.size(dimNum)))

      if (adjX) {
        reshapedX = reshapedX.transpose(2, 3)
      }
      if (adjY) {
        reshapedY = reshapedY.transpose(2, 3)
      }
      require(reshapedX.size(3) == reshapedY.size(2), "matrix sizes do not match" +
        s"the matrix sizes are ${reshapedX.size(2)} and ${reshapedY.size(3)}")

      output.resize(batchSize, reshapedX.size(2), reshapedY.size(3))
      output.bmm(reshapedX, reshapedY)
      val outputSize = x.size().slice(0, dimNum - 2) ++ Array(reshapedX.size(2), reshapedY.size(3))
      output.resize(outputSize)
    }

    output
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

object BatchMatMul {
  def apply[@specialized(Float, Double) T: ClassTag, D: ClassTag](
        adjX: Boolean = false,
        adjY: Boolean = false)
        (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]): BatchMatMul[T, D] = {
    new BatchMatMul[T, D](adjX, adjY)
  }
}
