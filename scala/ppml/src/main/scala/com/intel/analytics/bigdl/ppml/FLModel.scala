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
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.optim.LocalPredictor
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.ppml.base.Estimator
import com.intel.analytics.bigdl.ppml.utils.DataFrameUtils
import org.apache.spark.sql.DataFrame

abstract class FLModel() {
  val model: Sequential[Float]
  val estimator: Estimator

  /**
   *
   * @param trainData DataFrame of training data
   * @param epoch training epoch
   * @param batchSize training batch size
   * @param featureColumn Array of String, specifying feature columns
   * @param labelColumn Array of String, specifying label columns
   * @param valData DataFrame of validation data
   * @param hasLabel whether dataset has label, dataset always has label in common machine learning
   *                 and HFL cases, while dataset of some parties in VFL cases does not has label
   * @return
   */
  def fit(trainData: DataFrame,
          epoch: Int = 1,
          batchSize: Int = 4,
          featureColumn: Array[String] = null,
          labelColumn: Array[String] = null,
          valData: DataFrame = null,
          hasLabel: Boolean = true) = {
    val _trainData = DataFrameUtils.dataFrameToArrayRDD(trainData, featureColumn, labelColumn,
      hasLabel = hasLabel)
    val _valData = if (valData != null) {
      DataFrameUtils.dataFrameToArrayRDD(valData, featureColumn, labelColumn,
        hasLabel = hasLabel)
    } else null

    estimator.train(epoch, _trainData, _valData)
  }

  /**
   *
   * @param data DataFrame of evaluation data
   * @param batchSize evaluation batch size
   * @param featureColumn Array of String, specifying feature columns
   * @param labelColumn Array of String, specifying label columns
   * @param hasLabel whether dataset has label, dataset always has label in common machine learning
   *                 and HFL cases, while dataset of some parties in VFL cases does not has label
   */
  def evaluate(data: DataFrame = null,
               batchSize: Int = 4,
               featureColumn: Array[String] = null,
               labelColumn: Array[String] = null,
               hasLabel: Boolean = true) = {
    if (data == null) {
      estimator.getEvaluateResults().foreach{r =>
        println(r._1 + ":" + r._2.mkString(","))
      }
    } else {
      val _data = DataFrameUtils.dataFrameToArrayRDD(data, featureColumn, labelColumn,
        hasLabel = hasLabel)
      estimator.evaluate(_data)
    }
  }

  /**
   *
   * @param data DataFrame of prediction data
   * @param batchSize prediction batch size
   * @param featureColumn Array of String, specifying feature columns
   * @return
   */
  def predict(data: DataFrame,
              batchSize: Int = 4,
              featureColumn: Array[String] = null): Array[Activity] = {
    val _data = DataFrameUtils.dataFrameToArrayRDD(data, featureColumn, hasLabel = false)
    estimator.predict(_data)
  }
}

