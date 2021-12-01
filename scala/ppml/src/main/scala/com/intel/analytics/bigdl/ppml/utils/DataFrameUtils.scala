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

package com.intel.analytics.bigdl.ppml.utils


import com.intel.analytics.bigdl.dllib.feature.dataset.{DataSet, MiniBatch, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.ppml.vfl.VflContext
import com.intel.analytics.bigdl
import org.apache.spark.sql.types.FloatType
import org.apache.spark.sql.{DataFrame, SparkSession}

object DataFrameUtils {
  def dataFrameToSample(df: DataFrame, batchSize: Int = 4): bigdl.DataSet[MiniBatch[Float]] = {
    val spark = VflContext.getSparkSession()
    import spark.implicits._
    var fDf: DataFrame = df
    df.columns.foreach(colName => {
      fDf = fDf.withColumn(colName, df.col(colName).cast(FloatType))
    })
    val samples = fDf.rdd.map(r => {
      val arr = (0 until r.size).map(i => r.getAs[Float](i)).toArray
      val featureNum = r.size - 1
      val features = Tensor[Float](arr.slice(0, featureNum), Array(featureNum))
      val target = Tensor[Float](Array(arr(featureNum)), Array(1))
      Sample(features, target)
    })
    DataSet.array(samples.collect()) ->
      SampleToMiniBatch(batchSize, parallelizing = false)
  }
}
