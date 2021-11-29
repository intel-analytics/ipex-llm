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
package com.intel.analytics.bigdl.dllib.feature.common

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * a Preprocessing that converts an Array[Double] or Array[Float] to a Tensor.
 * @param size dimensions of target Tensor.
 */
class ArrayToTensor[T: ClassTag](size: Array[Int])(implicit ev: TensorNumeric[T])
  extends Preprocessing[Seq[AnyVal], Tensor[T]] {

  override def apply(prev: Iterator[Seq[AnyVal]]): Iterator[Tensor[T]] = {
    prev.map { f =>
      val feature = f.head match {
        case dd: Double => f.asInstanceOf[Seq[Double]].map(ev.fromType(_))
        case ff: Float => f.asInstanceOf[Seq[Float]].map(ev.fromType(_))
        case _ => throw new IllegalArgumentException("SeqToTensor only supports Float and Double")
      }
      Tensor(feature.toArray, size)
    }
  }
}

object ArrayToTensor {
  def apply[T: ClassTag](size: Array[Int])(implicit ev: TensorNumeric[T]): ArrayToTensor[T] =
    new ArrayToTensor[T](size)
}
