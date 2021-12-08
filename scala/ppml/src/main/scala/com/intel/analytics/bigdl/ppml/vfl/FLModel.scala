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

package com.intel.analytics.bigdl.ppml.vfl

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dllib.feature.dataset.MiniBatch
import com.intel.analytics.bigdl.ppml.FLClient
import com.intel.analytics.bigdl.ppml.utils.DataFrameUtils
import com.intel.analytics.bigdl.ppml.vfl.nn.VflNNEstimator
import org.apache.spark.sql.DataFrame

class FLModel() {
  val estimator: VflNNEstimator = null
  def fit(trainData: DataFrame,
          valData: DataFrame,
          featureColumn: Array[String] = null,
          epoch : Int = 1) = {
    val _trainData = DataFrameUtils.dataFrameToSample(trainData)
    val _valData = DataFrameUtils.dataFrameToSample(valData)
    estimator.train(epoch, _trainData.toLocal(), _valData.toLocal())
  }
  def evaluate() = {
    estimator.getEvaluateResults().foreach{r =>
      println(r._1 + ":" + r._2.mkString(","))
    }
  }
  def predict(data: DataFrame,
              featureColumn: Array[String] = null) = {

  }
}

