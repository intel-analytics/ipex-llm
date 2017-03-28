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

package com.intel.analytics.bigdl.nn.abstractnn

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect._

/**
 * [[Activity]] is a trait which represents
 * the concept of neural input within neural
 * networks. For now, two type of input are
 * supported and extending this trait, which
 * are [[Tensor]] and [[Table]].
 */
trait Activity {
  def toTensor[D](implicit ev: TensorNumeric[D]): Tensor[D]

  def toTable: Table
}

object Activity {
  def apply[A <: Activity: ClassTag, T : ClassTag]()(
    implicit ev: TensorNumeric[T]): A = {
    val result = if (classTag[A] == classTag[Tensor[T]]) {
      Tensor[T]()
    } else if (classTag[A] == classTag[Table]) {
      T()
    } else {
      null
    }

    result.asInstanceOf[A]
  }
}
