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
import java.util.{ArrayList => JArrayList, List => JList, Map => JMap}
// import com.intel.analytics.bigdl.dllib.feature.pmem._
import com.intel.analytics.bigdl.dllib.nnframes._
import org.apache.spark.api.java.JavaSparkContext
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

  def getLightGBMClassifier(): LightGBMClassifier = {
    new LightGBMClassifier()
  }

  def setLGBMClassifierFeaturesCol(estimator: LightGBMClassifier, value: String): Unit = {
    estimator.setFeaturesCol(value)
  }

  def setLGBMClassifierLabelCol(estimator: LightGBMClassifier, value: String): Unit = {
    estimator.setLabelCol(value)
  }

  def setLGBMClassifierBoostType(estimator: LightGBMClassifier, value: String): Unit = {
    estimator.setBoostingType(value)
  }

  def setLGBMClassifierMaxBin(estimator: LightGBMClassifier, value: Int): Unit = {
    estimator.setMaxBin(value)
  }

  def setLGBMClassifierNumLeaves(estimator: LightGBMClassifier, value: Int): Unit = {
    estimator.setNumLeaves(value)
  }

  def setLGBMClassifierMinDataInLeaf(estimator: LightGBMClassifier, value: Int): Unit = {
    estimator.setMinDataInLeaf(value)
  }

  def setLGBMClassifierMinSumHessainInLeaf(estimator: LightGBMClassifier, value: Int): Unit = {
    estimator.setMinSumHessianInLeaf(value)
  }

  def setLGBMClassifierBaggingFraction(estimator: LightGBMClassifier, value: Int): Unit = {
    estimator.setBaggingFraction(value)
  }

  def setLGBMClassifierBaggingFreq(estimator: LightGBMClassifier, value: Int): Unit = {
    estimator.setBaggingFreq(value)
  }

  def setLGBMClassifierFeatureFraction(estimator: LightGBMClassifier, value: Double): Unit = {
    estimator.setFeatureFraction(value)
  }

  def setLGBMClassifierLambdaL1(estimator: LightGBMClassifier, value: Double): Unit = {
    estimator.setLambdaL1(value)
  }

  def setLGBMClassifierLambdaL2(estimator: LightGBMClassifier, value: Double): Unit = {
    estimator.setLambdaL2(value)
  }

  def setLGBMClassifierMaxDepth(estimator: LightGBMClassifier, value: Int): Unit = {
    estimator.setMaxDepth(value)
  }

  def setLGBMClassifierMinGainToSplit(estimator: LightGBMClassifier, value: Double): Unit = {
    estimator.setMinGainToSplit(value)
  }

  def setLGBMClassifierMaxDeltaStep(estimator: LightGBMClassifier, value: Double): Unit = {
    estimator.setMaxDeltaStep(value)
  }

  def setLGBMClassifierSkipDrop(estimator: LightGBMClassifier, value: Double): Unit = {
    estimator.setSkipDrop(value)
  }

  def setLGBMClassifierNumInterations(estimator: LightGBMClassifier, value: Int): Unit = {
    estimator.setNumIterations(value)
  }

  def setLGBMClassifierLearningRate(estimator: LightGBMClassifier, value: Double): Unit = {
    estimator.setLearningRate(value)
  }

  def setLGBMClassifierEarlyStopRound(estimator: LightGBMClassifier, value: Int): Unit = {
    estimator.setEarlyStoppingRound(value)
  }

  def setLGBMClassifierCategoricalSlotNames(estimator: LightGBMClassifier,
                                            value: Array[String]): Unit = {
    estimator.setCategoricalSlotNames(value)
  }

  def setLGBMClassifierCategoricalSlotIndexes(estimator: LightGBMClassifier,
                                            value: Array[Int]): Unit = {
    estimator.setCategoricalSlotIndexes(value)
  }

  def setLGBMClassifierObjective(estimator: LightGBMClassifier, value: String): Unit = {
    estimator.setObjective(value)
  }

  def setLGBMClassifierIsUnbalance(estimator: LightGBMClassifier, value: Boolean): Unit = {
    estimator.setIsUnbalance(value)
  }

  def setLGBMClassifierNumThreads(estimator: LightGBMClassifier, value: Int): Unit = {
    estimator.setNumThreads(value)
  }

  def fitLGBMClassifier(estimator: LightGBMClassifier, df : DataFrame): LightGBMClassifierModel = {
    estimator.fit(df)
  }

  def loadLGBMClassifierModel(path: String): LightGBMClassifierModel = {
    LightGBMClassifierModel.loadNativeModel(path)
  }

  def saveLGBMClassifierModel(model: LightGBMClassifierModel, path: String): Unit = {
    model.saveNativeModel(path)
  }

  def setFeaturesLGBMClassifierModel(model: LightGBMClassifierModel, features: String): Unit = {
    model.setFeaturesCol(features)
  }

  def setPredictionLGBMClassifierModel(model: LightGBMClassifierModel,
                                      prediction: String): Unit = {
    model.setPredictionCol(prediction)
  }

  def transformLGBMClassifierModel(model: LightGBMClassifierModel,
                                  dataset: DataFrame): DataFrame = {
    model.transform(dataset)
  }

  def getLightGBMRegressor(): LightGBMRegressor = {
    new LightGBMRegressor()
  }

  def setLGBMRegressorFeaturesCol(estimator: LightGBMRegressor, value: String): Unit = {
    estimator.setFeaturesCol(value)
  }

  def setLGBMRegressorLabelCol(estimator: LightGBMRegressor, value: String): Unit = {
    estimator.setLabelCol(value)
  }

  def setLGBMRegressorBoostType(estimator: LightGBMRegressor, value: String): Unit = {
    estimator.setBoostingType(value)
  }

  def setLGBMRegressorMaxBin(estimator: LightGBMRegressor, value: Int): Unit = {
    estimator.setMaxBin(value)
  }

  def setLGBMRegressorNumLeaves(estimator: LightGBMRegressor, value: Int): Unit = {
    estimator.setNumLeaves(value)
  }

  def setLGBMRegressorMinDataInLeaf(estimator: LightGBMRegressor, value: Int): Unit = {
    estimator.setMinDataInLeaf(value)
  }

  def setLGBMRegressorMinSumHessainInLeaf(estimator: LightGBMRegressor, value: Int): Unit = {
    estimator.setMinSumHessianInLeaf(value)
  }

  def setLGBMRegressorBaggingFraction(estimator: LightGBMRegressor, value: Int): Unit = {
    estimator.setBaggingFraction(value)
  }

  def setLGBMRegressorBaggingFreq(estimator: LightGBMRegressor, value: Int): Unit = {
    estimator.setBaggingFreq(value)
  }

  def setLGBMClassifierFeatureFraction(estimator: LightGBMRegressor, value: Double): Unit = {
    estimator.setFeatureFraction(value)
  }

  def setLGBMClassifierLambdaL1(estimator: LightGBMRegressor, value: Double): Unit = {
    estimator.setLambdaL1(value)
  }

  def setLGBMRegressorLambdaL2(estimator: LightGBMRegressor, value: Double): Unit = {
    estimator.setLambdaL2(value)
  }

  def setLGBMClassifierMaxDepth(estimator: LightGBMRegressor, value: Int): Unit = {
    estimator.setMaxDepth(value)
  }

  def setLGBMRegressorMinGainToSplit(estimator: LightGBMRegressor, value: Double): Unit = {
    estimator.setMinGainToSplit(value)
  }

  def setLGBMRegressorMaxDeltaStep(estimator: LightGBMRegressor, value: Double): Unit = {
    estimator.setMaxDeltaStep(value)
  }

  def setLGBMRegressorSkipDrop(estimator: LightGBMRegressor, value: Double): Unit = {
    estimator.setSkipDrop(value)
  }

  def setLGBMRegressorNumInterations(estimator: LightGBMRegressor, value: Int): Unit = {
    estimator.setNumIterations(value)
  }

  def setLGBMRegressorLearningRate(estimator: LightGBMRegressor, value: Double): Unit = {
    estimator.setLearningRate(value)
  }

  def setLGBMRegressorEarlyStopRound(estimator: LightGBMRegressor, value: Int): Unit = {
    estimator.setEarlyStoppingRound(value)
  }

  def setLGBMRegressorCategoricalSlotNames(estimator: LightGBMRegressor,
                                            value: Array[String]): Unit = {
    estimator.setCategoricalSlotNames(value)
  }

  def setLGBMRegressorCategoricalSlotIndexes(estimator: LightGBMRegressor,
                                              value: Array[Int]): Unit = {
    estimator.setCategoricalSlotIndexes(value)
  }

  def setLGBMRegressorObjective(estimator: LightGBMRegressor, value: String): Unit = {
    estimator.setObjective(value)
  }


  def setLGBMRegressorNumThreads(estimator: LightGBMRegressor, value: Int): Unit = {
    estimator.setNumThreads(value)
  }

  def fitLGBMRegressor(estimator: LightGBMRegressor, df : DataFrame): LightGBMRegressorModel = {
    estimator.fit(df)
  }

  def loadLGBMRegressorModel(path: String): LightGBMRegressorModel = {
    LightGBMRegressorModel.loadNativeModel(path)
  }

  def saveLGBMRegressorModel(model: LightGBMRegressorModel, path: String): Unit = {
    model.saveNativeModel(path)
  }

  def setFeaturesLGBMRegressorModel(model: LightGBMRegressorModel, features: String): Unit = {
    model.setFeaturesCol(features)
  }

  def setPredictionLGBMRegressorModel(model: LightGBMRegressorModel,
                                       prediction: String): Unit = {
    model.setPredictionCol(prediction)
  }

  def transformLGBMRegressorModel(model: LightGBMRegressorModel,
                                  dataset: DataFrame): DataFrame = {
    model.transform(dataset)
  }
}
