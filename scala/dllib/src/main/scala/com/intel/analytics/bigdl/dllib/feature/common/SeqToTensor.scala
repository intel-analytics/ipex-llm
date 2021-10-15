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
 * a Preprocessing that converts Float, Double, Array[Float], Array[Double] or MLlib Vector to a
 * Tensor.
 * @param size dimensions of target Tensor.
 */
class SeqToTensor[T: ClassTag](size: Array[Int])(implicit ev: TensorNumeric[T])
  extends Preprocessing[Any, Tensor[T]] {

  override def apply(prev: Iterator[Any]): Iterator[Tensor[T]] = {
    prev.map { f =>
      val feature = f match {
        case ff: Float => Array(ff).map(ev.fromType(_))
        case dd: Double => Array(dd).map(ev.fromType(_))
        case ii: Int => Array(ii).map(ev.fromType(_))
        case sd: Seq[Any] => matchSeq(sd)
        case mllibVec: org.apache.spark.mllib.linalg.Vector =>
          f.asInstanceOf[org.apache.spark.mllib.linalg.Vector ].toArray.map(ev.fromType(_))
        case _ => throw new IllegalArgumentException("SeqToTensor only supports Float, Double, " +
          s"Array[Float], Array[Double] or MLlib Vector but got $f")
      }
      Tensor(feature, if (size.isEmpty) Array(feature.length) else size).contiguous()
    }
  }

  def matchSeq(list: Seq[Any]): Array[T] = list.head match {
    case dd: Double => list.asInstanceOf[Seq[Double]].map(ev.fromType(_)).toArray
    case ff: Float => list.asInstanceOf[Seq[Float]].map(ev.fromType(_)).toArray
    case ii: Int => list.asInstanceOf[Seq[Int]].map(ev.fromType(_)).toArray
    case _ => throw new IllegalArgumentException(s"SeqToTensor only supports Array[Int], " +
      s"Array[Float] and Array[Double] for ArrayType, but got $list")
  }
}


object SeqToTensor {
  def apply[T: ClassTag](
      size: Array[Int] = Array()
    )(implicit ev: TensorNumeric[T]): SeqToTensor[T] = new SeqToTensor[T](size)
}
