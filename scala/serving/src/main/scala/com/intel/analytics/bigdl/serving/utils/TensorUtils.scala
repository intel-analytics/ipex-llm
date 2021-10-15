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

package com.intel.analytics.bigdl.serving.utils

import com.intel.analytics.bigdl.dllib.tensor.Tensor

/**
 * Tensor operations used in Cluster Serving
 */
object TensorUtils {
  def getTotalSize(t: Tensor[Float]): Int = {
    var res: Int = 1
    (0 until t.size().length).foreach(i => res *= t.size()(i))
    res
  }
  def getTopN(n: Int, t: Tensor[Float]): List[(Int, Float)] = {
    val arr = t.toArray().toList
    val idx = (0 until arr.size)
    val l = idx.zip(arr).toList

    def update(l: List[(Int, Float)], e: (Int, Float)): List[(Int, Float)] = {
      if (e._2 > l.head._2) (e :: l.tail).sortWith(_._2 < _._2) else l
    }

    l.drop(n).foldLeft(l.take(n).sortWith(_._2 < _._2))(update).
      sortWith(_._2 > _._2)
  }
}
