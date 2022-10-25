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

package com.intel.analytics.bigdl.friesian

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.{Log4Error, T, Table}
import org.apache.spark.ml.linalg.DenseVector

import scala.collection.mutable
import scala.util.Random

object TestUtils {
  def conditionFailTest(condition: Boolean, msg: String = null): Unit = {
    // scalastyle:off
    assert(condition, msg)
    // scalastyle:on
  }

  def generateRankingInput(length: Int, dim: Int): Table = {
    T.array(
      Array.fill(length)(
        Tensor[Float](Array.fill(dim)(Random.nextFloat()), Array(dim))))
  }

  def toFloatArray(obj: Any): Array[Float] = {
    obj match {
      case vector: DenseVector =>
        vector.toArray.map(_.toFloat)
      case array: mutable.WrappedArray[Any] =>
        array.array.map(_.toString.toFloat)
      case _ =>
        Log4Error.invalidInputError(condition = false,
          "Recall's initialData parquet files contain unsupported Dtype. ",
          "Make sure ID's type is int and ebd's type is DenseVector or array.")
        new Array[Float](0)
    }
  }
}
