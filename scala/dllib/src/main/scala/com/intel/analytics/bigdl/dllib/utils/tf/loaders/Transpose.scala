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
package com.intel.analytics.bigdl.utils.tf.loaders

import java.nio.ByteOrder

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.{Contiguous, Sequential, Transpose}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.tensorflow.framework.NodeDef

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class Transpose extends TensorflowOpsLoader {

  import Utils._

  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder)
                                 (implicit ev: TensorNumeric[T]): Module[T] = {
    Adapter[T](Array(2), tensorArrays => {
      val perm = tensorArrays(0).asInstanceOf[Tensor[Int]].storage().array()
      val paris = permToPair(perm)
      val layer = Sequential()
      layer.add(Transpose[T](paris))
      layer.add(Contiguous())
      layer
    })
  }

  private def permToPair(perm: Array[Int]): Array[(Int, Int)] = {

    val numToRank = perm.zipWithIndex.toMap
    val arr = perm.indices.toArray
    val pairs = ArrayBuffer[(Int, Int)]()

    def sort(arr: Array[Int], low: Int, high: Int): Unit = {
      var i = low
      var j = high
      val pivot = arr(low + (high - low)/2)

      while (i <= j) {
        while (arr(i) < pivot) i += 1
        while (arr(j) > pivot) j -= 1

        if (i <= j) {
          exchangeNumbers(arr, i, j)
          i += 1
          j -= 1
        }
      }

      if (low < j) sort(arr, low, j)
      if (i < high) sort(arr, i, high)
    }

    def exchangeNumbers(arr: Array[Int], i: Int, j: Int): Unit = {
      val temp = arr(i)
      arr(i) = arr(j)
      arr(j) = temp
      pairs += ((i, j))
    }

    sort(arr.map(numToRank), 0, arr.length-1)

    pairs.filter(pair => pair._1 != pair._2).toArray
  }
}
