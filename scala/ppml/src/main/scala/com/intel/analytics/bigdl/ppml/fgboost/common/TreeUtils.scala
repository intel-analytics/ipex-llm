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

package com.intel.analytics.bigdl.ppml.fgboost.common



import com.intel.analytics.bigdl.dllib.tensor.Tensor


object TreeUtils {

  // TODO sort dataset by feature, sort index of sorted result
  def sortByFeature(inputData: Array[Tensor[Float]]): Array[Array[Int]] = {
    Array.tabulate(inputData(0).size(1)){featureI =>
      Array.tabulate(inputData.length)(i => (i, inputData(i).valueAt(featureI + 1))).sortBy(_._2).map(_._1)
    }
  }

  protected val method = classOf[java.util.BitSet].getDeclaredField("words")
  method.setAccessible(true)

  def getArrayLongFromBitSet(bs: java.util.BitSet): Array[Long] = {
    method.get(bs).asInstanceOf[Array[Long]]
  }

  def getBitSetFromArray(array: Array[Long]): java.util.BitSet = {
    // TODO array to bitset for candidate split
    new java.util.BitSet()
  }


  def expandGrads(grads: Array[Array[Float]], dataSize: Int, nLabel: Int): Array[Array[Array[Float]]] = {
    val groupedGrad = grads(0).grouped(nLabel).toArray
    val groupedHess = grads(1).grouped(nLabel).toArray
    val nGrads =  (0 until nLabel).map { gID =>
      val currGrad = (0 until dataSize).map { rowID =>
        groupedGrad(rowID)(gID)
      }.toArray
      val currHess = (0 until dataSize).map { rowID =>
        groupedHess(rowID)(gID)
      }.toArray
      Array(currGrad, currHess)
    }
    nGrads.toArray
  }

  def computeScore(indices: Array[Int],
                   grads: Array[Array[Float]],
                   lambda: Float): Float = {
    val gradSum = indices.map(grads(0)(_)).sum
    val hessSum = indices.map(grads(1)(_)).sum
    math.pow(gradSum, 2).toFloat / (hessSum + lambda)
  }

  def computeScoreWithSum(gradSum: Float,
                          hessSum: Float,
                          lambda: Float): Float = {
    math.pow(gradSum, 2).toFloat / (hessSum + lambda)
  }

  def computeScoreWithSum(gradSum: Double,
                          hessSum: Double,
                          lambda: Float): Float = {
    (math.pow(gradSum, 2) / (hessSum + lambda)).toFloat
  }

  def computeOutput(indices: Array[Int],
                    grads: Array[Array[Float]],
                    lambda: Float = 1): Float = {
    -1 * indices.map(grads(0)(_)).sum /
      (indices.map(grads(1)(_)).sum + lambda)
  }

  def sigmoidFloat(input: Float): Float = {
    (1 / (1 + Math.exp(-input))).toFloat
  }
}
