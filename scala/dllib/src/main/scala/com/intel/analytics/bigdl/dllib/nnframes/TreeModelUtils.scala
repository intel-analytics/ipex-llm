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

package com.intel.analytics.bigdl.dllib.nnframes
import com.intel.analytics.bigdl.dllib.utils.Log4Error
import com.microsoft.azure.synapse.ml.lightgbm.{LightGBMClassifier => MLightGBMClassifier}
import com.microsoft.azure.synapse.ml.lightgbm.{LightGBMRegressor => MLightGBMRegressor}


object TreeModelUtils {

  def convert2CamelCase(pythonStyle: String): String = {
    val data = pythonStyle.split("_")
    data(0) + data.slice(1, data.size)
      .map(x => x.substring(0, 1).toUpperCase() + x.substring(1)).mkString("")
  }

  def setParams(lgbmEstimator: Any, lgbmParams: Map[String, Any]): Unit = {

    val estimator = if (lgbmEstimator.isInstanceOf[MLightGBMClassifier]) {
      lgbmEstimator.asInstanceOf[MLightGBMClassifier]
    }
    else if (lgbmEstimator.isInstanceOf[MLightGBMRegressor]) {
      lgbmEstimator.asInstanceOf[MLightGBMRegressor]
    }
    else {
      throw new Exception(s"LightGBM setParams:  ${lgbmEstimator} is not supported right now")
    }

    lgbmParams.foreach(kv => kv._1 match {
      case "boostingType" => estimator.setBoostingType(kv._2.asInstanceOf[String])
      case "numLeaves" => estimator.setNumLeaves(kv._2.asInstanceOf[Int])
      case "maxDepth" => estimator.setMaxDepth(kv._2.asInstanceOf[Int])
      case "learningRate" => estimator.setLearningRate(kv._2.asInstanceOf[Double])
      case "numIterations" => estimator.setNumIterations(kv._2.asInstanceOf[Int])
      case "binConstructSampleCnt" => estimator.setBinSampleCount(kv._2.asInstanceOf[Int])
      case "objective" => estimator.setObjective(kv._2.asInstanceOf[String])
      case "minSplitGain" => estimator.setMinGainToSplit(kv._2.asInstanceOf[Double])
      case "minSumHessianInLeaf" => estimator.setMinSumHessianInLeaf(kv._2.asInstanceOf[Double])
      case "minDataInLeaf" => estimator.setMinDataInLeaf(kv._2.asInstanceOf[Int])
      case "baggingFraction" => estimator.setBaggingFraction(kv._2.asInstanceOf[Double])
      case "baggingFreq" => estimator.setBaggingFreq(kv._2.asInstanceOf[Int])
      case "featureFraction" => estimator.setFeatureFraction(kv._2.asInstanceOf[Double])
      case "lambdaL1" => estimator.setLambdaL1(kv._2.asInstanceOf[Double])
      case "lambdaL2" => estimator.setLambdaL2(kv._2.asInstanceOf[Double])
      case "numThreads" => estimator.setNumThreads(kv._2.asInstanceOf[Int])
      case "earlyStoppingRound" => estimator.setEarlyStoppingRound(kv._2.asInstanceOf[Int])
      case "maxBin" => estimator.setMaxBin(kv._2.asInstanceOf[Int])
      case _ =>
        Log4Error.invalidInputError(false,
          s"LightGBM setParams: key ${kv._1} is not supported by lgbmParams map",
          s"try to set this parameter by calling .set${kv._1}")
    })
  }
}
