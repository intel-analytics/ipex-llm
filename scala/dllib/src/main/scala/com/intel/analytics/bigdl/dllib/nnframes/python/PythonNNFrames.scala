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

import java.util.{ArrayList => JArrayList, List => JList, Map => JMap}

import com.intel.analytics.bigdl.dllib.feature.dataset.{Sample, Transformer}
import com.intel.analytics.bigdl.dllib.optim.{OptimMethod, Trigger, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.dllib.utils.python.api.EvaluatedResult
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.dllib.common.PythonZoo
import com.intel.analytics.bigdl.dllib.feature.common._
import com.intel.analytics.bigdl.dllib.feature.image.RowToImageFeature

import scala.collection.mutable
// import com.intel.analytics.bigdl.dllib.feature.pmem._
import com.intel.analytics.bigdl.dllib.nnframes._
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.sql.DataFrame

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

object PythonNNFrames {

  def ofFloat(): PythonNNFrames[Float] = new PythonNNFrames[Float]()

  def ofDouble(): PythonNNFrames[Double] = new PythonNNFrames[Double]()
}

class PythonNNFrames[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {

  def nnReadImage(path: String, sc: JavaSparkContext, minParitions: Int,
                  resizeH: Int, resizeW: Int, imageCodec: Int): DataFrame = {
    NNImageReader.readImages(path, sc.sc, minParitions, resizeH, resizeW, imageCodec)
  }

  def createNNEstimator(
      model: Module[T],
      criterion: Criterion[T],
      sampleTransformer: Preprocessing[(Any, Option[Any]), Sample[T]]
    ): NNEstimator[T] = {
    NNEstimator(model, criterion).setSamplePreprocessing(sampleTransformer)
  }

  def createNNClassifier(
      model: Module[T],
      criterion: Criterion[T],
      samplePreprocessing: Preprocessing[(Any, Option[AnyVal]), Sample[T]]
    ): NNClassifier[T] = {
    NNClassifier(model, criterion).setSamplePreprocessing(samplePreprocessing)
  }

  def createNNModel(
      model: Module[T],
      samplePreprocessing: Preprocessing[Any, Sample[T]]): NNModel[T] = {
    new NNModel(model).setSamplePreprocessing(samplePreprocessing)
  }

  def createNNClassifierModel(
      model: Module[T],
      samplePreprocessing: Preprocessing[Any, Sample[T]]): NNClassifierModel[T] = {
    NNClassifierModel(model).setSamplePreprocessing(samplePreprocessing)
  }

  def setOptimMethod(
      estimator: NNEstimator[T],
      optimMethod: OptimMethod[T]): NNEstimator[T] = {
    estimator.setOptimMethod(optimMethod)
  }

  def setSamplePreprocessing(
      estimator: NNEstimator[T],
      samplePreprocessing: Preprocessing[(Any, Option[AnyVal]), Sample[T]]): NNEstimator[T] = {
    estimator.setSamplePreprocessing(samplePreprocessing)
  }

  def setSamplePreprocessing(
      model: NNModel[T],
      samplePreprocessing: Preprocessing[Any, Sample[T]]): NNModel[T] = {
    model.setSamplePreprocessing(samplePreprocessing)
  }

  def withOriginColumn(imageDF: DataFrame, imageColumn: String, originColumn: String): DataFrame = {
    NNImageSchema.withOriginColumn(imageDF, imageColumn, originColumn)
  }

  def createScalarToTensor(): ScalarToTensor[T] = {
    new ScalarToTensor()
  }

  def createSeqToTensor(size: JArrayList[Int]): SeqToTensor[T] = {
    SeqToTensor(size.asScala.toArray)
  }

  def createSeqToMultipleTensors(size: JArrayList[JArrayList[Int]]): SeqToMultipleTensors[T] = {
    SeqToMultipleTensors(size.asScala.map(x => x.asScala.toArray).toArray)
  }

  def createArrayToTensor(size: JArrayList[Int]): ArrayToTensor[T] = {
    ArrayToTensor(size.asScala.toArray)
  }

  def createMLlibVectorToTensor(size: JArrayList[Int]): MLlibVectorToTensor[T] = {
    MLlibVectorToTensor(size.asScala.toArray)
  }

  def createRowToImageFeature(): RowToImageFeature[T] = {
    RowToImageFeature()
  }

  def createFeatureLabelPreprocessing(
      featureTransfomer: Preprocessing[Any, Tensor[T]],
      labelTransformer: Preprocessing[Any, Tensor[T]]
    ): FeatureLabelPreprocessing[Any, Any, Any, Sample[T]] = {
    FeatureLabelPreprocessing(featureTransfomer, labelTransformer)
      .asInstanceOf[FeatureLabelPreprocessing[Any, Any, Any, Sample[T]]]
  }

  def createChainedPreprocessing(list: JList[Preprocessing[Any, Any]]): Preprocessing[Any, Any] = {
    var cur = list.get(0)
    (1 until list.size()).foreach(t => cur = cur -> list.get(t))
    cur
  }

  def createTensorToSample(): TensorToSample[T] = {
    TensorToSample()
  }

  def createToTuple(): ToTuple = {
    ToTuple()
  }

  def createBigDLAdapter(bt: Transformer[Any, Any]): BigDLAdapter[Any, Any] = {
    BigDLAdapter(bt)
  }

  def setTrainSummary(
      estimator: NNEstimator[T],
      summary: TrainSummary
    ): NNEstimator[T] = {
    estimator.setTrainSummary(summary)
  }

  def setValidation(
      estimator: NNEstimator[T],
      trigger: Trigger,
      validationDF: DataFrame,
      vMethods : JList[ValidationMethod[T]],
      batchSize: Int): NNEstimator[T] = {
    estimator.setValidation(trigger, validationDF, vMethods.asScala.toArray, batchSize)
  }

  def setEndWhen(estimator: NNEstimator[T], trigger: Trigger): NNEstimator[T] = {
    estimator.setEndWhen(trigger)
  }

  def setDataCacheLevel(
      estimator: NNEstimator[T],
      level: String,
      numSlice: Int = 4): NNEstimator[T] = {
     import com.intel.analytics.bigdl.dllib.feature._
    val memType = level.trim.toUpperCase match {
      case "DRAM" => DRAM
      case "DISK_AND_DRAM" => DISK_AND_DRAM(numSlice)
      case _ => throw new IllegalArgumentException(s"$level is not supported.")
    }
    estimator.setDataCacheLevel(memType)
  }

  def setCheckpoint(
      estimator: NNEstimator[T],
      path: String,
      trigger: Trigger,
      isOverWrite: Boolean): NNEstimator[T] = {
    estimator.setCheckpoint(path, trigger, isOverWrite)
  }

  def setValidationSummary(
      estimator: NNEstimator[T],
      value: ValidationSummary): NNEstimator[T] = {
    estimator.setValidationSummary(value)
  }

  def setNNModelPreprocessing(
      model: NNModel[T],
      sampleTransformer: Preprocessing[Any, Sample[T]]): NNModel[T] = {
    model.setSamplePreprocessing(sampleTransformer)
  }

  def nnEstimatorClearGradientClipping(estimator: NNEstimator[T]): Unit = {
    estimator.clearGradientClipping()
  }

  def nnEstimatorSetConstantGradientClipping(
      estimator: NNEstimator[T],
      min: Float,
      max: Float): Unit = {
    estimator.setConstantGradientClipping(min, max)
  }

  def nnEstimatorSetGradientClippingByL2Norm(
      estimator: NNEstimator[T],
      clipNorm: Float): Unit = {
    estimator.setGradientClippingByL2Norm(clipNorm)
  }

  def saveNNModel(model: NNModel[T], path: String): Unit = {
    model.save(path)
  }

  def loadNNModel(path: String): NNModel[_] = {
    val loaded = NNModel.load(path)
    loaded
  }

  def loadNNClassifierModel(path: String): NNClassifierModel[_] = {
    NNClassifierModel.load(path)
  }

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
    model.save(path)
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

  def internalEval(estimator: NNEstimator[T],
                   dataFrame: DataFrame): JList[EvaluatedResult] = {
    estimator.internalEval(dataFrame)
  }

  def setNNFeaturesCol(estimator: NNEstimator[T],
                       featuresColName: String): NNEstimator[T] = {
    estimator.setFeaturesCol(featuresColName)
  }

  def setNNLabelCol(estimator: NNEstimator[T],
                    labelColName: String): NNEstimator[T] = {
    estimator.setLabelCol(labelColName)
  }

  def setNNBatchSize(estimator: NNEstimator[T],
                     value: Int): NNEstimator[T] = {
    estimator.setBatchSize(value)
  }
}
