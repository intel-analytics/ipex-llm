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

class XGBClassifier (val xgboostParams: Map[String, Any] = Map()) {
  val sc = SparkSession.active.sparkContext
  sc.getConf.set("spark.task.cpus", Engine.coreNumber().toString)
  private val model = new XGBoostClassifier(xgboostParams)
  model.setNthread(Engine.coreNumber())
  model.setNumWorkers(Engine.nodeNumber())
  model.setMaxBins(256)

  def setFeaturesCol(featuresColName: String): this.type = {
    model.setFeaturesCol(featuresColName)
    this
  }

  def fit(df: DataFrame): XGBClassifierModel = {
    df.repartition(Engine.nodeNumber())
    val xgbmodel = model.fit(df)
    new XGBClassifierModel(xgbmodel)
  }

  def setNthread(value: Int): this.type = {
    model.setNthread(value)
    this
  }

  def setNumRound(value: Int): this.type = {
    model.setNumRound(value)
    this
  }

  def setNumWorkers(value: Int): this.type = {
    model.setNumWorkers(value)
    this
  }

  def setEta(value: Double): this.type = {
    model.setEta(value)
    this
  }

  def setGamma(value: Int): this.type = {
    model.setGamma(value)
    this
  }

  def setMaxDepth(value: Int): this.type = {
    model.setMaxDepth(value)
    this
  }

  def setMissing(value: Float): this.type = {
    model.setMissing(value)
    this
  }

  def setLabelCol(labelColName: String): this.type = {
    model.setLabelCol(labelColName)
    this
  }
  def setTreeMethod(value: String): this.type = {
    model.setTreeMethod(value)
    this
  }

  def setObjective(value: String): this.type = {
    model.setObjective(value)
    this
  }

  def setNumClass(value: Int): this.type = {
    model.setNumClass(value)
    this
  }

  def setTimeoutRequestWorkers(value: Long): this.type = {
    model.setTimeoutRequestWorkers(value)
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

  private val model = new XGBoostRegressor()
  model.setNthread(Engine.coreNumber())
  model.setMaxBins(256)

  def setLabelCol(labelColName : String) : this.type = {
    model.setLabelCol(labelColName)
    this
  }

  def setFeaturesCol(featuresColName: String): this.type = {
    model.setFeaturesCol(featuresColName)
    this
  }

  def fit(df: DataFrame): XGBRegressorModel = {
    df.repartition(Engine.nodeNumber())
    val xgbModel = model.fit(df)
    new XGBRegressorModel(xgbModel)
  }

  def setNumRound(value: Int): this.type = {
    model.setNumRound(value)
    this
  }

  def setNumWorkers(value: Int): this.type = {
    model.setNumWorkers(value)
    this
  }

  def setNthread(value: Int): this.type = {
    model.setNthread(value)
    this
  }

  def setSilent(value: Int): this.type = {
    model.setSilent(value)
    this
  }

  def setMissing(value: Float): this.type = {
    model.setMissing(value)
    this
  }

  def setCheckpointPath(value: String): this.type = {
    model.setCheckpointPath(value)
    this
  }

  def setCheckpointInterval(value: Int): this.type = {
    model.setCheckpointInterval(value)
    this
  }

  def setSeed(value: Long): this.type = {
    model.setSeed(value)
    this
  }

  def setEta(value: Double): this.type = {
    model.setEta(value)
    this
  }

  def setGamma(value: Double): this.type = {
    model.setGamma(value)
    this
  }

  def setMaxDepth(value: Int): this.type = {
    model.setMaxDepth(value)
    this
  }

  def setMinChildWeight(value: Double): this.type = {
    model.setMinChildWeight(value)
    this
  }

  def setMaxDeltaStep(value: Double): this.type = {
    model.setMaxDeltaStep(value)
    this
  }

  def setColsampleBytree(value: Double): this.type = {
    model.setColsampleBytree(value)
    this
  }

  def setColsampleBylevel(value: Double): this.type = {
    model.setColsampleBylevel(value)
    this
  }

  def setLambda(value: Double): this.type = {
    model.setLambda(value)
    this
  }

  def setAlpha(value: Double): this.type = {
    model.setAlpha(value)
    this
  }

  def setTreeMethod(value: String): this.type = {
    model.setTreeMethod(value)
    this
  }

  def setGrowPolicy(value: String): this.type = {
    model.setGrowPolicy(value)
    this
  }

  def setMaxBins(value: Int): this.type = {
    model.setMaxBins(value)
    this
  }

  def setMaxLeaves(value: Int): this.type = {
    model.setMaxLeaves(value)
    this
  }

  def setSketchEps(value: Double): this.type = {
    model.setSketchEps(value)
    this
  }

  def setScalePosWeight(value: Double): this.type = {
    model.setScalePosWeight(value)
    this
  }

  def setSampleType(value: String): this.type = {
    model.setSampleType(value)
    this
  }

  def setNormalizeType(value: String): this.type = {
    model.setNormalizeType(value)
    this
  }

  def setRateDrop(value: Double): this.type = {
    model.setRateDrop(value)
    this
  }

  def setSkipDrop(value: Double): this.type = {
    model.setSkipDrop(value)
    this
  }

  def setLambdaBias(value: Double): this.type = {
    model.setLambdaBias(value)
    this
  }

  def setObjective(value: String): this.type = {
    model.setObjective(value)
    this
  }

  def setObjectiveType(value: String): this.type = {
    model.setObjectiveType(value)
    this
  }

  def setSubsample(value: Double): this.type = {
    model.setSubsample(value)
    this
  }

  def setBaseScore(value: Double): this.type = {
    model.setBaseScore(value)
    this
  }

  def setEvalMetric(value: String): this.type = {
    model.setEvalMetric(value)
    this
  }

  def setNumEarlyStoppingRounds(value: Int): this.type = {
    model.setNumEarlyStoppingRounds(value)
    this
  }

  def setMaximizeEvaluationMetrics(value: Boolean): this.type = {
    model.setMaximizeEvaluationMetrics(value)
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
 */
class LightGBMClassifier {
  System.setProperty("KMP_DUPLICATE_LIB_OK", "true")

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

  def save(path: String): Unit = {
    model.write.overwrite().save(path)
  }
}

object LightGBMClassifierModel {
  def load(path: String): LightGBMClassifierModel = {
    new LightGBMClassifierModel(MLightGBMClassificationModel.load(path))
  }
}

/**
 * [[XGBRegressor]] xgboost wrapper of XGBRegressor.
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

  def save(path: String): Unit = {
    model.write.overwrite().save(path)
  }
}

object LightGBMRegressorModel {
  /**
   * Load pretrained Zoo XGBRegressorModel.
   */
  def load(path: String): LightGBMRegressorModel = {
    new LightGBMRegressorModel(MLightGBMRegressionModel.load(path))
  }
}

