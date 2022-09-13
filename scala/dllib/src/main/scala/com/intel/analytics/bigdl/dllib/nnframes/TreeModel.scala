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


class XGBClassifier (val xgboostParams: Map[String, Any] = Map()) {
  val sc = SparkSession.active.sparkContext
  sc.getConf.set("spark.task.cpus", Engine.coreNumber().toString)
  private val estimator = new XGBoostClassifier(xgboostParams)
  estimator.setNthread(Engine.coreNumber())
  estimator.setNumWorkers(Engine.nodeNumber())
  estimator.setMaxBins(256)

  def setFeaturesCol(featuresColName: String): this.type = {
    estimator.setFeaturesCol(featuresColName)
    this
  }

  def fit(df: DataFrame): XGBClassifierModel = {
    df.repartition(Engine.nodeNumber())
    val xgbmodel = estimator.fit(df)
    new XGBClassifierModel(xgbmodel)
  }

  def setNthread(value: Int): this.type = {
    estimator.setNthread(value)
    this
  }

  def setNumRound(value: Int): this.type = {
    estimator.setNumRound(value)
    this
  }

  def setNumWorkers(value: Int): this.type = {
    estimator.setNumWorkers(value)
    this
  }

  def setEta(value: Double): this.type = {
    estimator.setEta(value)
    this
  }

  def setGamma(value: Int): this.type = {
    estimator.setGamma(value)
    this
  }

  def setMaxDepth(value: Int): this.type = {
    estimator.setMaxDepth(value)
    this
  }

  def setMissing(value: Float): this.type = {
    estimator.setMissing(value)
    this
  }

  def setLabelCol(labelColName: String): this.type = {
    estimator.setLabelCol(labelColName)
    this
  }
  def setTreeMethod(value: String): this.type = {
    estimator.setTreeMethod(value)
    this
  }

  def setObjective(value: String): this.type = {
    estimator.setObjective(value)
    this
  }

  def setNumClass(value: Int): this.type = {
    estimator.setNumClass(value)
    this
  }

  def setTimeoutRequestWorkers(value: Long): this.type = {
    estimator.setTimeoutRequestWorkers(value)
    this
  }
}
/**
 * [[XGBClassifierModel]] is a trained XGBoost classification model.
 * The prediction column will have the prediction results.
 *
 * @param model trained XGBoostClassificationModel to use in prediction.
 */
class XGBClassifierModel private[bigdl](
   val model: XGBoostClassificationModel) {
  private var featuresCols: String = null
  private var predictionCol: String = null

  def setFeaturesCol(featuresColName: String): this.type = {
    featuresCols = featuresColName
    this
  }

  def setPredictionCol(value: String): this.type = {
    predictionCol = value
    this
  }

  def setInferBatchSize(value: Int): this.type = {
    model.setInferBatchSize(value)
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

  def save(path: String): Unit = {
    model.write.overwrite().save(path)
  }

}

object XGBClassifierModel {
  def load(path: String, numClass: Int): XGBClassifierModel = {
    new XGBClassifierModel(XGBoostHelper.load(path, numClass))
  }

  def load(path: String): XGBClassifierModel = {
    new XGBClassifierModel(XGBoostClassificationModel.load(path))
  }

}

/**
 * [[XGBRegressor]] xgboost wrapper of XGBRegressor.
 */
class XGBRegressor () {

  private val estimator = new XGBoostRegressor()
  estimator.setNthread(Engine.coreNumber())
  estimator.setMaxBins(256)

  def setLabelCol(labelColName : String) : this.type = {
    estimator.setLabelCol(labelColName)
    this
  }

  def setFeaturesCol(featuresColName: String): this.type = {
    estimator.setFeaturesCol(featuresColName)
    this
  }

  def fit(df: DataFrame): XGBRegressorModel = {
    df.repartition(Engine.nodeNumber())
    val xgbModel = estimator.fit(df)
    new XGBRegressorModel(xgbModel)
  }

  def setNumRound(value: Int): this.type = {
    estimator.setNumRound(value)
    this
  }

  def setNumWorkers(value: Int): this.type = {
    estimator.setNumWorkers(value)
    this
  }

  def setNthread(value: Int): this.type = {
    estimator.setNthread(value)
    this
  }

  def setSilent(value: Int): this.type = {
    estimator.setSilent(value)
    this
  }

  def setMissing(value: Float): this.type = {
    estimator.setMissing(value)
    this
  }

  def setCheckpointPath(value: String): this.type = {
    estimator.setCheckpointPath(value)
    this
  }

  def setCheckpointInterval(value: Int): this.type = {
    estimator.setCheckpointInterval(value)
    this
  }

  def setSeed(value: Long): this.type = {
    estimator.setSeed(value)
    this
  }

  def setEta(value: Double): this.type = {
    estimator.setEta(value)
    this
  }

  def setGamma(value: Double): this.type = {
    estimator.setGamma(value)
    this
  }

  def setMaxDepth(value: Int): this.type = {
    estimator.setMaxDepth(value)
    this
  }

  def setMinChildWeight(value: Double): this.type = {
    estimator.setMinChildWeight(value)
    this
  }

  def setMaxDeltaStep(value: Double): this.type = {
    estimator.setMaxDeltaStep(value)
    this
  }

  def setColsampleBytree(value: Double): this.type = {
    estimator.setColsampleBytree(value)
    this
  }

  def setColsampleBylevel(value: Double): this.type = {
    estimator.setColsampleBylevel(value)
    this
  }

  def setLambda(value: Double): this.type = {
    estimator.setLambda(value)
    this
  }

  def setAlpha(value: Double): this.type = {
    estimator.setAlpha(value)
    this
  }

  def setTreeMethod(value: String): this.type = {
    estimator.setTreeMethod(value)
    this
  }

  def setGrowPolicy(value: String): this.type = {
    estimator.setGrowPolicy(value)
    this
  }

  def setMaxBins(value: Int): this.type = {
    estimator.setMaxBins(value)
    this
  }

  def setMaxLeaves(value: Int): this.type = {
    estimator.setMaxLeaves(value)
    this
  }

  def setSketchEps(value: Double): this.type = {
    estimator.setSketchEps(value)
    this
  }

  def setScalePosWeight(value: Double): this.type = {
    estimator.setScalePosWeight(value)
    this
  }

  def setSampleType(value: String): this.type = {
    estimator.setSampleType(value)
    this
  }

  def setNormalizeType(value: String): this.type = {
    estimator.setNormalizeType(value)
    this
  }

  def setRateDrop(value: Double): this.type = {
    estimator.setRateDrop(value)
    this
  }

  def setSkipDrop(value: Double): this.type = {
    estimator.setSkipDrop(value)
    this
  }

  def setLambdaBias(value: Double): this.type = {
    estimator.setLambdaBias(value)
    this
  }

  def setObjective(value: String): this.type = {
    estimator.setObjective(value)
    this
  }

  def setObjectiveType(value: String): this.type = {
    estimator.setObjectiveType(value)
    this
  }

  def setSubsample(value: Double): this.type = {
    estimator.setSubsample(value)
    this
  }

  def setBaseScore(value: Double): this.type = {
    estimator.setBaseScore(value)
    this
  }

  def setEvalMetric(value: String): this.type = {
    estimator.setEvalMetric(value)
    this
  }

  def setNumEarlyStoppingRounds(value: Int): this.type = {
    estimator.setNumEarlyStoppingRounds(value)
    this
  }

  def setMaximizeEvaluationMetrics(value: Boolean): this.type = {
    estimator.setMaximizeEvaluationMetrics(value)
    this
  }
}

/**
 * [[XGBRegressorModel]] xgboost wrapper of XGBRegressorModel.
 */
class XGBRegressorModel private[bigdl](val model: XGBoostRegressionModel) {
  var predictionCol: String = null
  var featuresCol: String = "features"
  var featurearray: Array[String] = Array("features")
  def setPredictionCol(value: String): this.type = {
    predictionCol = value
    this
  }

  def setInferBatchSize(value: Int): this.type = {
    model.setInferBatchSize(value)
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

  def save(path: String): Unit = {
    model.write.overwrite().save(path)
  }
}

object XGBRegressorModel {
  /**
   * Load pretrained Zoo XGBRegressorModel.
   */
  def load(path: String): XGBRegressorModel = {
    new XGBRegressorModel(XGBoostRegressionModel.load(path))
  }

  /**
   * Load pretrained xgboost XGBoostRegressionModel.
   */
  def loadFromXGB(path: String): XGBRegressorModel = {
    new XGBRegressorModel(XGBoostHelper.load(path))
  }
}

/**
 * [[lightGBMClassifier wrapper]]
 * @param lgbmParams, a map of parameters, currently supported 18 params to be set from lgbmParams:
        "boostingType", "numLeaves", "maxDepth", "learningRate", "numIterations",
        "binConstructSampleCnt", "objective", "minSplitGain", "minSumHessianInLeaf",
        "minDataInLeaf", "baggingFraction", "baggingFreq", "featureFraction",
        "lambdaL1", "lambdaL2", "numThreads", "earlyStoppingRound", "maxBin".
 */
class LightGBMClassifier (val lgbmParams: Map[String, Any] = Map()) {

  val sc = SparkSession.active.sparkContext
  sc.getConf.set("spark.task.cpus", Engine.coreNumber().toString)

  private val estimator = new MLightGBMClassifier()
  estimator.setNumThreads(Engine.coreNumber())
  TreeModelUtils.setParams(estimator, lgbmParams)

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
  private var featuresCol: String = "features"
  private var predictionCol: String = "prediction"

  def setFeaturesCol(featuresColName: String): this.type = {
    featuresCol = featuresColName
    this
  }

  def setPredictionCol(value: String): this.type = {
    predictionCol = value
    this
  }

  def transform(dataset: DataFrame): DataFrame = {
    Log4Error.invalidInputError(featuresCol!=None, "Please set feature columns before transform")
    model.setFeaturesCol(featuresCol)
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
 * @param lgbmParams, a map of parameters, currently supported 18 params to be set from lgbmParams:
        "boostingType", "numLeaves", "maxDepth", "learningRate", "numIterations",
        "binConstructSampleCnt", "objective", "minSplitGain", "minSumHessianInLeaf",
        "minDataInLeaf", "baggingFraction", "baggingFreq", "featureFraction",
        "lambdaL1", "lambdaL2", "numThreads", "earlyStoppingRound", "maxBin".
 */
class LightGBMRegressor (val lgbmParams: Map[String, Any] = Map()) {

  val sc = SparkSession.active.sparkContext
  sc.getConf.set("spark.task.cpus", Engine.coreNumber().toString)

  private val estimator = new MLightGBMRegressor()
  estimator.setNumThreads(Engine.coreNumber())
  TreeModelUtils.setParams(estimator, lgbmParams)

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
    Log4Error.invalidInputError(featuresCol!=None, "Please set feature columns before transform")
    model.setFeaturesCol(featuresCol)
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

object LightGBMRegressorModel {
  /**
   * Load pretrained Zoo XGBRegressorModel.
   */
  def loadNativeModel(path: String): LightGBMRegressorModel = {
    new LightGBMRegressorModel(MLightGBMRegressionModel.loadNativeModelFromFile(path))
  }
}

