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

import com.intel.analytics.bigdl.nn.abstractnn.TensorCriterion
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * The negative of the mean cosine proximity between predictions and targets.
 * The cosine proximity is defined as below:
 * x'(i) = x(i) / sqrt(max(sum(x(i)^2), 1e-12))
 * y'(i) = y(i) / sqrt(max(sum(x(i)^2), 1e-12))
 * cosine_proximity(x, y) = mean(-1 * x'(i) * y'(i))
 *
 * Both batch and un-batched inputs are supported
 */
class CosineProximityCriterion[@specialized(Float, Double) T: ClassTag]
(implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    val x = l2Norm(input)
    val y = l2Norm(target)
    val mul = x.cmul(y)
    var loss = ev.fromType[Double](0.0)
    val func = new TensorFunc2[T] {
      override def apply(v1: Array[T], v2: Int): Unit = {
        loss = ev.plus(ev.negative(v1(v2)), loss)
      }
    }
    DenseTensorApply.apply1(mul, func)
    loss = ev.divide(loss, ev.fromType(mul.nElement()))
    loss
  }

  private def l2Norm(input : Tensor[T]) : Tensor[T] = {

    val tiled = reduceSum(input)
    tiled.apply1((t) => ev.divide(ev.fromType[Double](1.0),
      ev.sqrt(ev.max(t, ev.fromType[Double](1e-12)))))
    val result = Tensor[T]()
    result.resizeAs(input)
    result.cmul(input, tiled)
    result
  }

  private def reduceSum(input : Tensor[T]): Tensor[T] = {
    val square = Tensor[T]()
    val dim = input.dim()
    square.resizeAs(input).copy(input)
    square.apply1((t) => ev.pow(t, ev.fromType[Double](2.0)))
    // apply sum to last dim
    val squareSum = square.sum(dim)
    tileLastDim(squareSum, input.size())
  }

  private def tileLastDim(input : Tensor[T], sizeRef : Array[Int]): Tensor[T] = {
    val sizes = new Array[Int](sizeRef.length)
    var index = 0
    sizeRef.foreach(size => {
      index += 1
      sizes(index - 1) = if (index == sizeRef.length) sizeRef(sizeRef.length - 1) else 1
    })
    input.repeatTensor(sizes)
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {

    gradInput.resizeAs(input)

    // Calculate target norm (yi/Sqrt(y1^2 + y2^2 .... + yn^2))
    val targetNorm = l2Norm(target)

    // expand as the input sizes and tile each element `dim` size
    val tiled = reduceSum(input)

    // Calculate xi * norm(yi)

    val inputTargetNom = Tensor[T]().resizeAs(input)
    inputTargetNom.cmul(input, targetNorm)

    val inputTargetNomSum = inputTargetNom.sum(inputTargetNom.dim())

    val inputTargetNomSumTiled = tileLastDim(inputTargetNomSum, input.size())

    // First calculate xi * (x1 * norm(y1)... xn * norm(yn))
    val func1 = new TensorFunc6[T] {
      override def apply(data1: Array[T], offset1: Int, data2: Array[T],
                         offset2: Int, data3: Array[T], offset3: Int): Unit = {
        data1(offset1) = ev.times(data2(offset2), data3(offset3))

      }
    }

    DenseTensorApply.apply3[T](gradInput, input, inputTargetNomSumTiled, func1)

    // Then appy -1/sqrt((x1^2 + ... xn^2)^(3/2)) * ((x1^2 + ... xn^2) - grad(xi))

    val total = input.nElement()

    val func2 = new TensorFunc6[T] {
      override def apply(data1: Array[T], offset1: Int, data2: Array[T],
                         offset2: Int, data3: Array[T], offset3: Int): Unit = {
        if (ev.isGreater(ev.fromType[Double](1e-12f), data2(offset2))) {
          data1(offset1) = ev.negative(ev.fromType[Double](1.0/Math.sqrt(1e-12)))
        } else {
          val f1 = ev.divide(ev.fromType[Double](1.0),
            ev.times(data3(offset3), ev.sqrt(data3(offset3))))
          val s1 = ev.times(data3(offset3), data2(offset2))
          data1(offset1) = ev.divide(ev.negative(ev.times(f1, ev.minus(s1, data1(offset1))))
            , ev.fromType(total))
        }
      }
    }
    DenseTensorApply.apply3[T](gradInput, targetNorm, tiled, func2)
    gradInput
  }
}

object CosineProximityCriterion {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]):
  CosineProximityCriterion[T] = new CosineProximityCriterion[T]
}
