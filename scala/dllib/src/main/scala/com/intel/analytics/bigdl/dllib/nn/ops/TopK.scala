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

class TopK[T: ClassTag, D: ClassTag](
  val k: Int,
  val sorted: Boolean = true,
  val startIndex: Int = 1,
  val dim: Int = -1,
  val increase: Boolean = false)
(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Tensor[D], Table, T] {

  private val indices = Tensor[Int]()
  private val values = Tensor[D]()
  private val indicesD = Tensor[D]()

  output = T(values, indices)

  override def updateOutput(input: Tensor[D]): Table = {
    input.topk(
      k = k,
      dim = dim,
      increase = increase,
      result = values,
      indices = indicesD,
      sortedResult = sorted)
    indices.resizeAs(indicesD)
    indices.zipWith[Int, D](indices, indicesD, (a, b) => {
      ev2.toType[Int](b) + startIndex - 1
    })
    output
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

object TopK {
  def apply[T: ClassTag, D: ClassTag](
  k: Int,
  sorted: Boolean = true,
  startIndex : Int = 1,
  dim: Int = -1,
  increase: Boolean = false)
  (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]):
  TopK[T, D] = new TopK(k, sorted, startIndex, dim, increase)
}
