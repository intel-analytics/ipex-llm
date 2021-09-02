/*
 * Copyright 2018 Analytics Zoo Authors.
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
package com.intel.analytics.zoo.feature.common

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * a Preprocessing that converts Array[Float], Array[Double] or MLlib Vector to multiple tensors
 * for multi-input models
 * @param multiSizes dimensions of target Tensors.
 */
class SeqToMultipleTensors[T: ClassTag](multiSizes: Array[Array[Int]])
    (implicit ev: TensorNumeric[T]) extends Preprocessing[Any, Array[Tensor[T]]] {

  override def apply(prev: Iterator[Any]): Iterator[Array[Tensor[T]]] = {
    prev.map { f =>
      val tensors = f match {
        case sd: Seq[Any] => matchSeq(sd)
        case _ => throw new IllegalArgumentException("SeqToTensor only supports Float, Double, " +
          s"Array[Float], Array[Double] or MLlib Vector but got $f")
      }
      tensors
    }
  }

  def matchSeq(list: Seq[Any]): Array[Tensor[T]] = {
    val rawData = list.head match {
      case dd: Double => list.asInstanceOf[Seq[Double]].map(ev.fromType(_)).toArray
      case ff: Float => list.asInstanceOf[Seq[Float]].map(ev.fromType(_)).toArray
      case ii: Int => list.asInstanceOf[Seq[Int]].map(ev.fromType(_)).toArray
      case _ => throw new IllegalArgumentException(s"SeqToTensor only supports Array[Int], " +
        s"Array[Float] and Array[Double] for ArrayType, but got $list")
    }

    require(multiSizes.map(s => s.product).sum == rawData.length, s"feature columns length " +
      s"${rawData.length} does not match with the sum of tensors" +
      s" ${multiSizes.map(a => a.mkString(",")).mkString("\n")}")

    var cur = 0
    val tensors = multiSizes.map { size =>
      val rawLength = size.product
      val t = Tensor(rawData.slice(cur, cur + rawLength), size)
      cur += rawLength
      t
    }
    tensors
  }
}


object SeqToMultipleTensors {
  def apply[T: ClassTag](
      multiSizes: Array[Array[Int]]
    )(implicit ev: TensorNumeric[T]): SeqToMultipleTensors[T] =
    new SeqToMultipleTensors[T](multiSizes)
}
