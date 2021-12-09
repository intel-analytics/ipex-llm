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

package com.intel.analytics.bigdl.ppml

import com.intel.analytics.bigdl.dllib.nn.Sequential
import com.intel.analytics.bigdl.ppml.base.Estimator
import com.intel.analytics.bigdl.ppml.utils.DataFrameUtils
import com.intel.analytics.bigdl.ppml.vfl.nn.VflNNEstimator
import org.apache.spark.sql.DataFrame

abstract class FLModel() {
  val model: Sequential[Float]
  val estimator: Estimator

  /**
   *
   * @param trainData DataFrame of training data
   * @param epoch training epoch
   * @param batchSize training batchsize
   * @param featureColumn Array of String, specifying feature columns
   * @param labelColumn Array of String, specifying label columns
   * @param valData DataFrame of validation data
   * @return
   */
  def fit(trainData: DataFrame,
          epoch: Int = 1,
          batchSize: Int = 4,
          featureColumn: Array[String] = null,
          labelColumn: Array[String] = null,
          valData: DataFrame = null) = {
    val _trainData = DataFrameUtils.dataFrameToMiniBatch(trainData, featureColumn, labelColumn,
      isTrain = true, batchSize = batchSize)
    val _valData = DataFrameUtils.dataFrameToMiniBatch(valData)
    estimator.train(epoch, _trainData.toLocal(), _valData.toLocal())
  }
  def evaluate(data: DataFrame = null,
               batchSize: Int = 4,
               featureColumn: Array[String] = null) = {
    if (data == null) {
      estimator.getEvaluateResults().foreach{r =>
        println(r._1 + ":" + r._2.mkString(","))
      }
    } else {
      throw new Error("Not implemented.")
    }

  }
  def predict(data: DataFrame,
              batchSize: Int = 4,
              featureColumn: Array[String] = null) = {
    val _data = DataFrameUtils.dataFrameToSampleRDD(
      data, featureColumn, hasLabel = false, batchSize = batchSize)
    model.predict(_data).collect()
  }
}

