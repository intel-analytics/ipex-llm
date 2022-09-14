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

import com.intel.analytics.bigdl.dllib.utils.{Engine, Log4Error}
import org.apache.spark.sql.SparkSession
import ml.dmlc.xgboost4j.scala.spark._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import com.microsoft.azure.synapse.ml.lightgbm.{LightGBMClassificationModel => MLightGBMClassificationModel}
import com.microsoft.azure.synapse.ml.lightgbm.{LightGBMClassifier => MLightGBMClassifier}
import com.microsoft.azure.synapse.ml.lightgbm.{LightGBMRegressionModel => MLightGBMRegressionModel}
import com.microsoft.azure.synapse.ml.lightgbm.{LightGBMRegressor => MLightGBMRegressor}
import com.microsoft.azure.synapse.ml.lightgbm.{LightGBMRankerModel => MLightGBMRankerModel}
import com.microsoft.azure.synapse.ml.lightgbm.{LightGBMRanker => MLightGBMRanker}
import com.microsoft.azure.synapse.ml.lightgbm.{LightGBMBase => MLightGBMBase}
import com.microsoft.azure.synapse.ml.lightgbm.params.{LightGBMParams => MLightGBMParams}
import org.apache.spark.ml.Model
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.util.Identifiable

/**
 * [[lightGBMClassifier wrapper]]
 */
class LightGBMClassifier {

  val sc = SparkSession.active.sparkContext
  sc.getConf.set("spark.task.cpus", Engine.coreNumber().toString)

  val estimator = new MLightGBMClassifier()
  estimator.setNumThreads(Engine.coreNumber())

  def setLabelCol(labelColName : String) : this.type = {
    estimator.setLabelCol(labelColName)
    this
  }

  def setFeaturesCol(featuresColName: String): this.type = {
    estimator.setFeaturesCol(featuresColName)
    this
  }

  def setBoostingType(value: String): this.type = {
    estimator.setBoostingType(value)
    this
  }
  // for regularization
  def setMaxBin(value: Int): this.type = {
    estimator.setMaxBin(value)
    this
  }

  def setNumLeaves(value: Int): this.type = {
    estimator.setNumLeaves(value)
    this
  }

  def setMinDataInLeaf(value: Int): this.type = {
    estimator.setMinDataInLeaf(value)
    this
  }

  def setMinSumHessianInLeaf(value: Int): this.type = {
    estimator.setMinSumHessianInLeaf(value)
    this
  }

  def setBaggingFraction(value: Double): this.type = {
    estimator.setBaggingFraction(value)
    this
  }

  def setBaggingFreq(value: Int): this.type = {
    estimator.setBaggingFreq(value)
    this
  }

  def setFeatureFraction(value: Double): this.type = {
    estimator.setFeatureFraction(value)
    this
  }

  def setLambdaL1(value: Double): this.type = {
    estimator.setLambdaL1(value)
    this
  }

  def setLambdaL2(value: Double): this.type = {
    estimator.setLambdaL2(value)
    this
  }

  def setMaxDepth(value: Int): this.type = {
    estimator.setMaxDepth(value)
    this
  }

  def setMinGainToSplit(value: Double): this.type = {
    estimator.setMinGainToSplit(value)
    this
  }

  def setMaxDeltaStep(value: Double): this.type = {
    estimator.setMaxDeltaStep(value)
    this
  }

  def setSkipDrop(value: Double): this.type = {
    estimator.setSkipDrop(value)
    this
  }

  // training
  def setNumIterations(value: Int): this.type = {
    estimator.setNumIterations(value)
    this
  }
  def getNumIterations: Int = {
    estimator.getNumIterations
  }

  def setLearningRate(value: Double): this.type = {
    estimator.setLearningRate(value)
    this
  }

  def setEarlyStoppingRound(value: Int): this.type = {
    estimator.setEarlyStoppingRound(value)
    this
  }

  def setCategoricalSlotNames(value: Array[String]): this.type = {
    estimator.setCategoricalSlotNames(value)
    this
  }
  def setCategoricalSlotIndexes(value: Array[Int]): this.type = {
    estimator.setCategoricalSlotIndexes(value)
    this
  }

  def setObjective(value: String): this.type = {
    estimator.setObjective(value)
    this
  }


  def setIsUnbalance(value: Boolean): this.type = {
    estimator.setIsUnbalance(value)
    this
  }
  def setNumThreads(value: Int): this.type = {
    estimator.setNumThreads(value)
    this
  }
  def fit(df: DataFrame): LightGBMClassifierModel = {
    df.repartition(Engine.nodeNumber())
    val lightgbmmodel = estimator.fit(df)
    new LightGBMClassifierModel(lightgbmmodel)
  }
}

/**
 * [[LightGBMClassifierModel]] is a trained LightGBM classification model.
 * The prediction column will have the prediction results.
 *
 * @param model trained MLightGBMClassificationModel to use in prediction.
 */
class LightGBMClassifierModel private[bigdl](val model: MLightGBMClassificationModel) {
  private var featuresCols: String = "features"
  private var predictionCol: String = "prediction"

  def setFeaturesCol(featuresColName: String): this.type = {
    featuresCols = featuresColName
    this
  }

  def setPredictionCol(value: String): this.type = {
    predictionCol = value
    this
  }

  def transform(dataset: DataFrame): DataFrame = {
    Log4Error.invalidInputError(featuresCols!=None, "Please set feature columns before transform")
    model.setFeaturesCol(featuresCols)
    var output = model.transform(dataset)
    if(predictionCol != null) {
      output = output.withColumnRenamed("prediction", predictionCol)
    }
    output
  }

  def saveNativeModel(path: String): Unit = {
    model.saveNativeModel(path, overwrite = true)
  }
}

object LightGBMClassifierModel {
  def loadNativeModel(path: String): LightGBMClassifierModel = {
    new LightGBMClassifierModel(MLightGBMClassificationModel.loadNativeModelFromFile(path))
  }
}

/**
 * [[LightGBMRegressor]] lightGBM wrapper of LightGBMRegressor.
 */
class LightGBMRegressor {

  val sc = SparkSession.active.sparkContext
  sc.getConf.set("spark.task.cpus", Engine.coreNumber().toString)

  private val estimator = new MLightGBMRegressor()
  estimator.setNumThreads(Engine.coreNumber())

  def setAlpha(value: Double): this.type = {
    estimator.setAlpha(value)
    this
  }

  def setLabelCol(labelColName : String) : this.type = {
    estimator.setLabelCol(labelColName)
    this
  }

  def setFeaturesCol(featuresColName: String): this.type = {
    estimator.setFeaturesCol(featuresColName)
    this
  }

  def setBoostingType(value: String): this.type = {
    estimator.setBoostingType(value)
    this
  }
  // for regularization
  def setMaxBin(value: Int): this.type = {
    estimator.setMaxBin(value)
    this
  }

  def setNumLeaves(value: Int): this.type = {
    estimator.setNumLeaves(value)
    this
  }

  def setMinDataInLeaf(value: Int): this.type = {
    estimator.setMinDataInLeaf(value)
    this
  }

  def setMinSumHessianInLeaf(value: Int): this.type = {
    estimator.setMinSumHessianInLeaf(value)
    this
  }

  def setBaggingFraction(value: Double): this.type = {
    estimator.setBaggingFraction(value)
    this
  }

  def setBaggingFreq(value: Int): this.type = {
    estimator.setBaggingFreq(value)
    this
  }

  def setFeatureFraction(value: Double): this.type = {
    estimator.setFeatureFraction(value)
    this
  }

  def setLambdaL1(value: Double): this.type = {
    estimator.setLambdaL1(value)
    this
  }

  def setLambdaL2(value: Double): this.type = {
    estimator.setLambdaL2(value)
    this
  }

  def setMaxDepth(value: Int): this.type = {
    estimator.setMaxDepth(value)
    this
  }

  def setMinGainToSplit(value: Double): this.type = {
    estimator.setMinGainToSplit(value)
    this
  }

  def setMaxDeltaStep(value: Double): this.type = {
    estimator.setMaxDeltaStep(value)
    this
  }

  def setSkipDrop(value: Double): this.type = {
    estimator.setSkipDrop(value)
    this
  }

  // training
  def setNumIterations(value: Int): this.type = {
    estimator.setNumIterations(value)
    this
  }


  def setLearningRate(value: Double): this.type = {
    estimator.setLearningRate(value)
    this
  }

  def setEarlyStoppingRound(value: Int): this.type = {
    estimator.setEarlyStoppingRound(value)
    this
  }

  def setCategoricalSlotNames(value: Array[String]): this.type = {
    estimator.setCategoricalSlotNames(value)
    this
  }
  def setCategoricalSlotIndexes(value: Array[Int]): this.type = {
    estimator.setCategoricalSlotIndexes(value)
    this
  }

  def setObjective(value: String): this.type = {
    estimator.setObjective(value)
    this
  }

  def setNumThreads(value: Int): this.type = {
    estimator.setNumThreads(value)
    this
  }

  def fit(df: DataFrame): LightGBMRegressorModel = {
    df.repartition(Engine.nodeNumber())
    val lightgbmmodel = estimator.fit(df)
    new LightGBMRegressorModel(lightgbmmodel)
  }
}

/**
 * [[LightGBMRegressorModel]] lightGBM wrapper of LightGBMRegressorModel.
 */
class LightGBMRegressorModel private[bigdl](val model: MLightGBMRegressionModel) {
  var predictionCol: String = null
  var featuresCol: String = "features"
  var featurearray: Array[String] = Array("features")
  def setPredictionCol(value: String): this.type = {
    predictionCol = value
    this
  }

  def setFeaturesCol(value: String): this.type = {
    model.setFeaturesCol(value)
    featuresCol = value
    this
  }

  def transform(dataset: DataFrame): DataFrame = {
    val featureVectorAssembler = new VectorAssembler()
      .setInputCols(featurearray)
      .setOutputCol("featureAssembledVector")
    val assembledDF = featureVectorAssembler.transform(dataset)
    import org.apache.spark.ml.linalg.Vector
    import org.apache.spark.sql.functions.{col, udf}
    val asDense = udf((v: Vector) => v.toDense)
    val xgbInput = assembledDF.withColumn("DenseFeatures", asDense(col("featureAssembledVector")))
    model.setFeaturesCol("DenseFeatures")
    var output = model.transform(xgbInput).drop("DenseFeatures", "featureAssembledVector")
    if(predictionCol != null) {
      output = output.withColumnRenamed("prediction", predictionCol)
    }
    output
  }

  def saveNativeModel(path: String): Unit = {
    model.saveNativeModel(path, overwrite = true)
  }
}

object LightGBMRegressorModel {
  /**
   * Load pretrained Zoo XGBRegressorModel.
   */
  def loadNativeModel(path: String): LightGBMRegressorModel = {
    new LightGBMRegressorModel(MLightGBMRegressionModel.loadNativeModelFromFile(path))
  }
}

