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

package com.intel.analytics.bigdl.nn.keras

import com.intel.analytics.bigdl.nn.Transpose
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * Permutes the dimensions of the input according to a given pattern.
 * Useful for connecting RNNs and convnets together.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param dims Int array. Permutation pattern, does not include the batch dimension.
 *             Indexing starts at 1.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Permute[T: ClassTag](
   val dims: Array[Int],
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

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

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    val outputShape = input.clone()
    var i = 0
    while (i < dims.length) {
      outputShape(i + 1) = input(dims(i))
      i += 1
    }
    Shape(outputShape)
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val swaps = permToPair(dims.map(x => x - 1)).map(pair => (pair._1 + 2, pair._2 + 2))
    val layer = Transpose(swaps)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Permute {
  def apply[@specialized(Float, Double) T: ClassTag](
    dims: Array[Int],
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Permute[T] = {
    new Permute[T](dims, inputShape)
  }
}
