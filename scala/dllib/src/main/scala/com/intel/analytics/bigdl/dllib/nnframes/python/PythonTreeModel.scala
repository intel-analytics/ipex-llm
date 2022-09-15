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

package com.intel.analytics.bigdl.dllib.nnframes.python

import com.intel.analytics.bigdl.dllib.common.PythonZoo
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

import java.util.{ArrayList => JArrayList, Map => JMap}
import com.intel.analytics.bigdl.dllib.nnframes._
import com.microsoft.azure.synapse.ml.lightgbm.LightGBMUtils
import org.apache.spark.sql.DataFrame

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

object PythonTreeModel {

  def ofFloat(): PythonTreeModel[Float] = new PythonTreeModel[Float]()

  def ofDouble(): PythonTreeModel[Double] = new PythonTreeModel[Double]()
}


class PythonTreeModel[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {

  def getXGBClassifier(xgbparamsin: JMap[String, Any]): XGBClassifier = {
    val xgbparams = if (xgbparamsin == null) Map[String, Any]() else xgbparamsin.asScala.toMap
    new XGBClassifier(xgbparams)
  }

  def setXGBClassifierNthread(model: XGBClassifier, value: Int): Unit = {
    model.setNthread(value)
  }

  def setXGBClassifierNumRound(model: XGBClassifier, value: Int): Unit = {
    model.setNumRound(value)
  }

  def fitXGBClassifier(model: XGBClassifier, df : DataFrame): XGBClassifierModel = {
    model.fit(df)
  }

  def setXGBClassifierNumWorkers(model: XGBClassifier, value: Int): Unit = {
    model.setNumWorkers(value)
  }

  def setXGBClassifierMissing(model: XGBClassifier, value: Int): Unit = {
    model.setMissing(value)
  }

  def setXGBClassifierMaxDepth(model: XGBClassifier, value: Int): Unit = {
    model.setMaxDepth(value)
  }

  def setXGBClassifierEta(model: XGBClassifier, value: Double): Unit = {
    model.setEta(value)
  }

  def setXGBClassifierGamma(model: XGBClassifier, value: Int): Unit = {
    model.setGamma(value)
  }

  def setXGBClassifierTreeMethod(model: XGBClassifier, value: String): Unit = {
    model.setTreeMethod(value)
  }

  def setXGBClassifierObjective(model: XGBClassifier, value: String): Unit = {
    model.setObjective(value)
  }

  def setXGBClassifierNumClass(model: XGBClassifier, value: Int): Unit = {
    model.setNumClass(value)
  }

  def setXGBClassifierFeaturesCol(model: XGBClassifier, value: String): Unit = {
    model.setFeaturesCol(value)
  }

  def loadXGBClassifierModel(path: String, numClasses: Int): XGBClassifierModel = {
    XGBClassifierModel.load(path, numClasses)
  }

  def saveXGBClassifierModel(model: XGBClassifierModel, path: String): Unit = {
    model.model.nativeBooster.saveModel(path)
  }

  def setFeaturesXGBClassifierModel(model: XGBClassifierModel, features: String): Unit = {
    model.setFeaturesCol(features)
  }

  def setPredictionXGBClassifierModel(model: XGBClassifierModel,
                                            prediction: String): Unit = {
    model.setPredictionCol(prediction)
  }

  def setInferBatchSizeXGBClassifierModel(model: XGBClassifierModel,
                                          batchSize: Int): Unit = {
    model.setInferBatchSize(batchSize)
  }

  def transformXGBClassifierModel(model: XGBClassifierModel,
                                        dataset: DataFrame): DataFrame = {
    model.transform(dataset)
  }

  def getXGBRegressor(): XGBRegressor = {
    val model = new XGBRegressor()
    model
  }

  def setXGBRegressorNthread(model: XGBRegressor, value: Int): Unit = {
    model.setNthread(value)
  }

  def setXGBRegressorNumRound(model: XGBRegressor, value: Int): Unit = {
    model.setNumRound(value)
  }

  def setXGBRegressorNumWorkers(model: XGBRegressor, value: Int): Unit = {
    model.setNumWorkers(value)
  }

  def fitXGBRegressor(model: XGBRegressor, df : DataFrame): XGBRegressorModel = {
    model.fit(df)
  }

  def loadXGBRegressorModel(path: String) : XGBRegressorModel = {
    XGBRegressorModel.load(path)
  }

  def setPredictionXGBRegressorModel(model: XGBRegressorModel, prediction : String): Unit = {
    model.setPredictionCol(prediction)
  }

  def setInferBatchSizeXGBRegressorModel(model: XGBRegressorModel, value : Int): Unit = {
    model.setInferBatchSize(value)
  }

  def setFeaturesXGBRegressorModel(model: XGBRegressorModel, features: String): Unit = {
    model.setFeaturesCol(features)
  }

  def transformXGBRegressorModel(model: XGBRegressorModel,
                                 dataset: DataFrame): DataFrame = {
    model.transform(dataset)
  }

  def saveXGBRegressorModel(model: XGBRegressorModel, path: String): Unit = {
    model.save(path)
  }

  // lightGBM support
  def getLightGBMClassifier(lgbmParamsin: JMap[String, Any]): LightGBMClassifier = {
    val lgbmParams = if (lgbmParamsin == null) Map[String, Any]() else lgbmParamsin.asScala.toMap
      .map(x => (TreeModelUtils.convert2CamelCase(x._1), x._2))
    new LightGBMClassifier(lgbmParams)
  }

  def getLightGBMRegressor(lgbmParamsin: JMap[String, Any]): LightGBMRegressor = {
    val lgbmParams = if (lgbmParamsin == null) Map[String, Any]() else lgbmParamsin.asScala.toMap
      .map(x => (TreeModelUtils.convert2CamelCase(x._1), x._2))
    new LightGBMRegressor(lgbmParams)
  }

  def setLGBMFeaturesCol(estimator: LightGBMClassifier, value: String): Unit = {
    estimator.setFeaturesCol(value)
  }

  def setLGBMFeaturesCol(estimator: LightGBMRegressor, value: String): Unit = {
    estimator.setFeaturesCol(value)
  }

  def setLGBMLabelCol(estimator: LightGBMClassifier, value: String): Unit = {
    estimator.setLabelCol(value)
  }

  def setLGBMLabelCol(estimator: LightGBMRegressor, value: String): Unit = {
    estimator.setLabelCol(value)
  }

  def setLGBMBoostType(estimator: LightGBMClassifier, value: String): Unit = {
    estimator.setBoostingType(value)
  }

  def setLGBMBoostType(estimator: LightGBMRegressor, value: String): Unit = {
    estimator.setBoostingType(value)
  }

  def setLGBMMaxBin(estimator: LightGBMClassifier, value: Int): Unit = {
    estimator.setMaxBin(value)
  }

  def setLGBMMaxBin(estimator: LightGBMRegressor, value: Int): Unit = {
    estimator.setMaxBin(value)
  }

  def setLGBMNumLeaves(estimator: LightGBMClassifier, value: Int): Unit = {
    estimator.setNumLeaves(value)
  }

  def setLGBMNumLeaves(estimator: LightGBMRegressor, value: Int): Unit = {
    estimator.setNumLeaves(value)
  }

  def setLGBMMinDataInLeaf(estimator: LightGBMClassifier, value: Int): Unit = {
    estimator.setMinDataInLeaf(value)
  }

  def setLGBMMinDataInLeaf(estimator: LightGBMRegressor, value: Int): Unit = {
    estimator.setMinDataInLeaf(value)
  }

  def setLGBMMinSumHessainInLeaf(estimator: LightGBMClassifier, value: Int): Unit = {
    estimator.setMinSumHessianInLeaf(value)
  }

  def setLGBMMinSumHessainInLeaf(estimator: LightGBMRegressor, value: Int): Unit = {
    estimator.setMinSumHessianInLeaf(value)
  }

  def setLGBMBaggingFraction(estimator: LightGBMClassifier, value: Int): Unit = {
    estimator.setBaggingFraction(value)
  }

  def setLGBMBaggingFraction(estimator: LightGBMRegressor, value: Int): Unit = {
    estimator.setBaggingFraction(value)
  }

  def setLGBMBaggingFreq(estimator: LightGBMClassifier, value: Int): Unit = {
    estimator.setBaggingFreq(value)
  }

  def setLGBMBaggingFreq(estimator: LightGBMRegressor, value: Int): Unit = {
    estimator.setBaggingFreq(value)
  }

  def setLGBMFeatureFraction(estimator: LightGBMClassifier, value: Double): Unit = {
    estimator.setFeatureFraction(value)
  }

  def setLGBMFeatureFraction(estimator: LightGBMRegressor, value: Double): Unit = {
    estimator.setFeatureFraction(value)
  }

  def setLGBMLambdaL1(estimator: LightGBMClassifier, value: Double): Unit = {
    estimator.setLambdaL1(value)
  }

  def setLGBMLambdaL1(estimator: LightGBMRegressor, value: Double): Unit = {
    estimator.setLambdaL1(value)
  }

  def setLGBMLambdaL2(estimator: LightGBMClassifier, value: Double): Unit = {
    estimator.setLambdaL2(value)
  }

  def setLGBMLambdaL2(estimator: LightGBMRegressor, value: Double): Unit = {
    estimator.setLambdaL2(value)
  }

  def setLGBMMaxDepth(estimator: LightGBMClassifier, value: Int): Unit = {
    estimator.setMaxDepth(value)
  }

  def setLGBMMaxDepth(estimator: LightGBMRegressor, value: Int): Unit = {
    estimator.setMaxDepth(value)
  }

  def setLGBMMinGainToSplit(estimator: LightGBMClassifier, value: Double): Unit = {
    estimator.setMinGainToSplit(value)
  }

  def setLGBMMinGainToSplit(estimator: LightGBMRegressor, value: Double): Unit = {
    estimator.setMinGainToSplit(value)
  }

  def setLGBMMaxDeltaStep(estimator: LightGBMClassifier, value: Double): Unit = {
    estimator.setMaxDeltaStep(value)
  }

  def setLGBMMaxDeltaStep(estimator: LightGBMRegressor, value: Double): Unit = {
    estimator.setMaxDeltaStep(value)
  }

  def setLGBMNumThreads(estimator: LightGBMRegressor, value: Int): Unit = {
    estimator.setNumThreads(value)
  }
  def setLGBMNumThreads(estimator: LightGBMClassifier, value: Int): Unit = {
    estimator.setNumThreads(value)
  }

  def setLGBMSkipDrop(estimator: LightGBMRegressor, value: Double): Unit = {
    estimator.setSkipDrop(value)
  }

  def setLGBMSkipDrop(estimator: LightGBMClassifier, value: Double): Unit = {
    estimator.setSkipDrop(value)
  }

  def setLGBMNumIterations(estimator: LightGBMRegressor, value: Int): Unit = {
    estimator.setNumIterations(value)
  }

  def setLGBMNumIterations(estimator: LightGBMClassifier, value: Int): Unit = {
    estimator.setNumIterations(value)
  }

  def setLGBMLearningRate(estimator: LightGBMRegressor, value: Double): Unit = {
    estimator.setLearningRate(value)
  }

  def setLGBMLearningRate(estimator: LightGBMClassifier, value: Double): Unit = {
    estimator.setLearningRate(value)
  }

  def setLGBMEarlyStopRound(estimator: LightGBMRegressor, value: Int): Unit = {
    estimator.setEarlyStoppingRound(value)
  }

  def setLGBMEarlyStopRound(estimator: LightGBMClassifier, value: Int): Unit = {
    estimator.setEarlyStoppingRound(value)
  }

  def setLGBMCategoricalSlotNames(estimator: LightGBMRegressor,
                                  value: JArrayList[String]): Unit = {
    estimator.setCategoricalSlotNames(value.asScala.toArray)
  }

  def setLGBMCategoricalSlotNames(estimator: LightGBMClassifier,
  value: JArrayList[String]): Unit = {
    estimator.setCategoricalSlotNames(value.asScala.toArray)
  }


  def setLGBMCategoricalSlotIndexes(estimator: LightGBMRegressor,
                                    value: JArrayList[Int]): Unit = {
    estimator.setCategoricalSlotIndexes(value.asScala.toArray)
  }

  def setLGBMCategoricalSlotIndexes(estimator: LightGBMClassifier,
                                    value: JArrayList[Int]): Unit = {
    estimator.setCategoricalSlotIndexes(value.asScala.toArray)
  }

  def setLGBMObjective(estimator: LightGBMRegressor, value: String): Unit = {
    estimator.setObjective(value)
  }

  def setLGBMObjective(estimator: LightGBMClassifier, value: String): Unit = {
    estimator.setObjective(value)
  }

  def fitLGBM(estimator: LightGBMClassifier, df : DataFrame): LightGBMClassifierModel = {
    estimator.fit(df)
  }

  def fitLGBM(estimator: LightGBMRegressor, df : DataFrame): LightGBMRegressorModel = {
    estimator.fit(df)
  }

  def setFeaturesLGBMModel(model: LightGBMClassifierModel, features: String): Unit = {
    model.setFeaturesCol(features)
  }

  def setFeaturesLGBMModel(model: LightGBMRegressorModel, features: String): Unit = {
    model.setFeaturesCol(features)
  }

  def setPredictionLGBMModel(model: LightGBMClassifierModel,
  prediction: String): Unit = {
    model.setPredictionCol(prediction)
  }

  def setPredictionLGBMModel(model: LightGBMRegressorModel,
                             prediction: String): Unit = {
    model.setPredictionCol(prediction)
  }

  def transformLGBMModel(model: LightGBMClassifierModel,
  dataset: DataFrame): DataFrame = {
    model.transform(dataset)
  }

  def transformLGBMModel(model: LightGBMRegressorModel,
                         dataset: DataFrame): DataFrame = {
    model.transform(dataset)
  }

  def loadLGBMClassifierModel(path: String): LightGBMClassifierModel = {
    LightGBMClassifierModel.loadNativeModel(path)
  }

  def loadLGBMRegressorModel(path: String): LightGBMRegressorModel = {
    LightGBMRegressorModel.loadNativeModel(path)
  }

  def saveLGBMModel(model: LightGBMClassifierModel, path: String): Unit = {
    model.saveNativeModel(path)
  }

  def saveLGBMModel(model: LightGBMRegressorModel, path: String): Unit = {
    model.saveNativeModel(path)
  }

  def setLGBMClassifierIsUnbalance(estimator: LightGBMClassifier, value: Boolean): Unit = {
    estimator.setIsUnbalance(value)
  }

  def setLGBMRegressorAlpha(estimator: LightGBMRegressor, value: Double): Unit = {
    estimator.setAlpha(value)
  }
}
