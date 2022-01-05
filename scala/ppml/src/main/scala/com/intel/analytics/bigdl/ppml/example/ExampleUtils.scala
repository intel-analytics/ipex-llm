/*
 * Copyright 2021 The BigDL Authors.
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

package com.intel.analytics.bigdl.ppml.example

import org.apache.spark.sql.DataFrame


object ExampleUtils {
  /**
   * Split a DataFrame into 2 parts for training and evaluation, the size is controlled by ratio
   * @param df DataFrame to split
   * @param ratio Float, default 0.8, means training data size is 0.8 * totalSize
   * @return 2-tuple of DataFrame
   */
  def splitDataFrameToTrainVal(df: DataFrame, ratio: Float = 0.8f): (DataFrame, DataFrame) = {
    val size = df.count()
    val trainSize = (size * ratio).toInt
    val trainDf = df.limit(trainSize)
    val valDf = df.except(trainDf)
    (trainDf, valDf)
  }
  def minMaxNormalize(data: Array[Array[Float]], col: Int): Array[Array[Float]] = {
    val min = data.map(_ (col)).min
    val max = data.map(_ (col)).max
    data.foreach { d =>
      d(col) = (d(col) - min) / (max - min)
    }
    data
  }
}
