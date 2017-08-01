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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

abstract class TreeLSTM[T: ClassTag](
  val inputSize: Int,
  val hiddenSize: Int = 150
)(implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Tensor[T], T] {
  protected val memZero: Tensor[T] = Tensor[T](hiddenSize).zero()

  def shareParams(
    cell: AbstractModule[Activity, Activity, T],
    src: AbstractModule[Activity, Activity, T]): Unit = {
    var i = 0
    val cellParams = cell.parameters()
    val srcParams = src.parameters()
    while (i < cellParams._1.length) {
      cellParams._1(i).set(srcParams._1(i))
      i += 1
    }
    i = 0
    while (i < cellParams._2.length) {
      cellParams._2(i).set(srcParams._2(i))
      i += 1
    }
  }
}
