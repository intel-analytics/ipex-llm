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

package com.intel.analytics.bigdl.ppml.model.lightgbm

import com.intel.analytics.bigdl.ppml.crypto.{CryptoMode, PLAIN_TEXT}
import com.microsoft.azure.synapse.ml.lightgbm.booster.LightGBMBooster
import org.apache.hadoop.fs.Path
import org.apache.spark.sql.SparkSession

object LightGBMModel {
  val CLASSIFICATION = "LightGBMClassificationModel"
  val REGRESSION = "LightGBMRegressionModel"
  val RANKER = "LightGBMRankerModel"
  def loadLightGBMBooster(modelPath: String): LightGBMBooster = {
      val lightGBMBooster = new LightGBMBooster(
        SparkSession.builder().getOrCreate()
          .read.text(modelPath)
          .collect().map { row => row.getString(0) }.mkString("\n")
      )
      lightGBMBooster
  }
}
