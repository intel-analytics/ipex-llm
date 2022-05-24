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

package com.intel.analytics.bigdl.ppml.fl

import com.intel.analytics.bigdl.dllib.nn.Sequential
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.ppml.fl.base.Estimator
import com.intel.analytics.bigdl.ppml.fl.utils.{DataFrameUtils, VFLTensorUtils}
import org.apache.spark.sql.DataFrame

abstract class NNModel() {
  val model: Sequential[Float]
  val estimator: Estimator

  /**
   * Fit API for Tensor
   * @param xTrain
   * @param yTrain
   * @param epoch
   * @param batchSize
   * @param xValidate
   * @param yValidate
   */
  def fit(xTrain: Tensor[Float],
          yTrain: Tensor[Float],
          epoch: Int = 1,
          batchSize: Int = 4,
          xValidate: Tensor[Float] = null,
          yValidate: Tensor[Float] = null): Any = {
    estimator.train(epoch,
      VFLTensorUtils.featureLabelToMiniBatch(xTrain, yTrain, batchSize),
      VFLTensorUtils.featureLabelToMiniBatch(xValidate, yValidate, batchSize))
  }
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
  def fitDataFrame(trainData: DataFrame,
                   epoch: Int = 1,
                   batchSize: Int = 4,
                   featureColumn: Array[String] = null,
                   labelColumn: Array[String] = null,
                   valData: DataFrame = null,
                   hasLabel: Boolean = true): Any = {
    val _trainData = DataFrameUtils.dataFrameToMiniBatch(trainData, featureColumn, labelColumn,
      hasLabel = hasLabel, batchSize = batchSize)
    val _valData = DataFrameUtils.dataFrameToMiniBatch(valData, featureColumn, labelColumn,
      hasLabel = hasLabel, batchSize = batchSize)
    estimator.train(epoch, _trainData.toLocal(), _valData.toLocal())
  }

  /**
   * Evaluate API for Tensor
   * @param x
   * @param y
   * @param batchSize
   */
  def evaluate(x: Tensor[Float],
               y: Tensor[Float] = null,
               batchSize: Int = 4): Unit = {
    estimator.evaluate(VFLTensorUtils.featureLabelToMiniBatch(x, y, batchSize))
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
  def evaluateDataFrame(data: DataFrame = null,
                        batchSize: Int = 4,
                        featureColumn: Array[String] = null,
                        labelColumn: Array[String] = null,
                        hasLabel: Boolean = true): Unit = {
    if (data == null) {
      estimator.getEvaluateResults().foreach{r =>
        println(r._1 + ":" + r._2.mkString(","))
      }
    } else {
      val _data = DataFrameUtils.dataFrameToMiniBatch(data, featureColumn, labelColumn,
        hasLabel = hasLabel, batchSize = batchSize)
      estimator.evaluate(_data.toLocal())
    }
  }

  /**
   * Predict API for Tensor
   * @param x
   * @param batchSize
   * @return
   */
  def predict(x: Tensor[Float], batchSize: Int = 4): Array[Activity] = {
    estimator.predict(VFLTensorUtils.featureLabelToMiniBatch(x, null, batchSize))
  }
  /**
   *
   * @param data DataFrame of prediction data
   * @param batchSize prediction batch size
   * @param featureColumn Array of String, specifying feature columns
   * @return
   */
  def predictDataFrame(data: DataFrame,
                       batchSize: Int = 4,
                       featureColumn: Array[String] = null): Array[Activity] = {
    val _data = DataFrameUtils.dataFrameToMiniBatch(data, featureColumn,
      hasLabel = false, batchSize = batchSize)
    estimator.predict(_data.toLocal())
  }
}

