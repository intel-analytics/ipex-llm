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

import com.intel.analytics.bigdl.tensor.{DenseTensorApply, DoubleType, FloatType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import java.util.Arrays.binarySearch

import scala.reflect.ClassTag

/**
 * BucketizedCol operation represents discretized dense input.
 *
 * The Operation can handle single or multi feature column,
 * as long as the boundaries is same between feature columns.
 *
 * Buckets include the left boundary, and exclude the right boundary.
 * Namely, boundaries=Array(0, 1, 10) generates buckets (-inf,0),[0,1),[1,10),[10,+inf)
 *
 * For example, boundaries = Array(0, 10, 100) and input tensor is an 2D 3x2 DenseTensor:
 *  -1, 1
 *  101, 10
 *  5, 100
 *
 *  the output tensor should be an 2D 3x2 DenseTensor
 *  0, 1
 *  3, 2
 *  1, 3
 *
 * @param boundaries The bound Array of each bucket.
 * @tparam T Numeric type. Parameter tensor numeric type. Only support float/double now
 */

class BucketizedCol[T: ClassTag](
  private val boundaries: Array[Double])(implicit ev: TensorNumeric[T])
  extends Operation[Tensor[T], Tensor[Int], T] {

  require(boundaries.length >= 1,
    "the length of boundaries must be more than or equal to 1")

  private val boundariesImpl = boundaries.map(ev.fromType[Double])

  output = Tensor[Int]()

  override def updateOutput(input: Tensor[T]): Tensor[Int] = {

    val resTensor = Tensor[Int](input.size())

    ev.getType() match {
      case FloatType =>
        resTensor.applyFun[Float](
          input.asInstanceOf[Tensor[Float]],
          x => math.abs(binarySearch(boundariesImpl.asInstanceOf[Array[Float]], x) + 1))
      case DoubleType =>
        resTensor.applyFun[Double](
          input.asInstanceOf[Tensor[Double]],
          x => math.abs(binarySearch(boundariesImpl.asInstanceOf[Array[Double]], x) + 1))
      case _ =>
        throw new RuntimeException("Unsupported tensor type")
    }

    output = resTensor
    output
  }
}

object BucketizedCol {
  def apply[T: ClassTag](
    boundaries: Array[Double])
    (implicit ev: TensorNumeric[T]): BucketizedCol[T]
  = new BucketizedCol[T](
    boundaries = boundaries
  )
}
