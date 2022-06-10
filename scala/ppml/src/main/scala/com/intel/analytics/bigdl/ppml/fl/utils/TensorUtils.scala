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

package com.intel.analytics.bigdl.ppml.fl.utils

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import org.apache.spark.sql.DataFrame

object TensorUtils {
  def fromDataFrame(df: DataFrame,
                    columns: Array[String]): Tensor[Float] = {
    if (columns == null) {
      null
    } else {
      val localDf = df.collect()
      val rowNum = localDf.length
      val dataArray = localDf.map(row => {
        val rowArray = new Array[Float](columns.length)
        columns.indices.foreach(i => {
          rowArray(i) = row.getAs[String](columns(i)).toFloat
        })
        rowArray
      })
      Tensor[Float](dataArray.flatten, Array(rowNum, columns.length))
    }
  }
}
