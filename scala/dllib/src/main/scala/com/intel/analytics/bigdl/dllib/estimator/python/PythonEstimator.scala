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

package com.intel.analytics.bigdl.dllib.estimator.python

import java.util.{List => JList, Map => JMap}

import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.dllib.feature.dataset.{MiniBatch, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.dllib.optim.{OptimMethod, Trigger, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.dllib.utils.python.api.EvaluatedResult
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.{ImageFeature, ImageFeatureToMiniBatch}
import com.intel.analytics.bigdl.dllib.utils.Table
import com.intel.analytics.bigdl.dllib.common.PythonZoo
import com.intel.analytics.bigdl.dllib.feature.FeatureSet
import com.intel.analytics.bigdl.dllib.estimator.Estimator

import scala.reflect.ClassTag
import scala.collection.JavaConverters._

object PythonEstimator {
  def ofFloat(): PythonEstimator[Float] = new PythonEstimator[Float]()

  def ofDouble(): PythonEstimator[Double] = new PythonEstimator[Double]()
}

class PythonEstimator[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T]{
  def createEstimator(model: Module[T],
                      optimMethod: OptimMethod[T],
                      modelDir: String): Estimator[T] = {
    Estimator(model, optimMethod, modelDir)
  }

  def createEstimator(model: Module[T],
                      optimMethods: JMap[String, OptimMethod[T]],
                      modelDir: String): Estimator[T] = {
    require(optimMethods != null, "optimMethods cannot be null")
    Estimator(model, optimMethods.asScala.toMap, modelDir)
  }

  def estimatorEvaluate(estimator: Estimator[T], validationSet: FeatureSet[Sample[T]],
                        validationMethod: JList[ValidationMethod[T]], batchSize: Int
                       ): JList[EvaluatedResult] = {
    val sample2batch = SampleToMiniBatch(batchSize)
    val validationMiniBatch = validationSet -> sample2batch
    val evalResult = estimator.evaluate(validationMiniBatch, validationMethod.asScala.toArray)
    toEvaluatedResult(evalResult)
  }

  def estimatorEvaluateImageFeature(estimator: Estimator[T],
                                    validationSet: FeatureSet[ImageFeature],
                                    validationMethod: JList[ValidationMethod[T]],
                                    batchSize: Int
                                   ): JList[EvaluatedResult] = {
    val imageFeature2batch = ImageFeatureToMiniBatch(batchSize)
    val validationMiniBatch = validationSet -> imageFeature2batch
    val evalResult = estimator.evaluate(validationMiniBatch, validationMethod.asScala.toArray)
    toEvaluatedResult(evalResult)
  }

  def estimatorEvaluateMiniBatch(
      estimator: Estimator[T],
      validationSet: FeatureSet[MiniBatch[T]],
      validationMethod: JList[ValidationMethod[T]]): JList[EvaluatedResult] = {
    val evalResult = estimator.evaluate(validationSet, validationMethod.asScala.toArray)
    toEvaluatedResult(evalResult)
  }

  def estimatorTrain(estimator: Estimator[T], trainSet: FeatureSet[Sample[T]],
                     criterion: Criterion[T],
                     endTrigger: Trigger = null,
                     checkPointTrigger: Trigger = null,
                     validationSet: FeatureSet[Sample[T]] = null,
                     validationMethod: JList[ValidationMethod[T]] = null,
                     batchSize: Int)
  : estimator.type = {
    val sample2batch = SampleToMiniBatch(batchSize)
    val trainMiniBatch = trainSet -> sample2batch
    val validationMiniBatch = if (validationSet != null) {
      validationSet -> sample2batch
    } else {
      null
    }

    estimator.train(trainMiniBatch, criterion,
      Option(endTrigger), Option(checkPointTrigger),
      validationMiniBatch, Option(validationMethod).map(_.asScala.toArray).orNull)
  }

  def createSampleToMiniBatch(batchSize: Int): SampleToMiniBatch[T] = {
    SampleToMiniBatch(batchSize)
  }

  def estimatorTrainMiniBatch(
      estimator: Estimator[T],
      trainSet: FeatureSet[MiniBatch[T]],
      criterion: Criterion[T],
      endTrigger: Trigger = null,
      checkPointTrigger: Trigger = null,
      validationSet: FeatureSet[MiniBatch[T]] = null,
      validationMethod: JList[ValidationMethod[T]] = null) : estimator.type = {
    estimator.train(trainSet, criterion,
      Option(endTrigger), Option(checkPointTrigger),
      validationSet, Option(validationMethod).map(_.asScala.toArray).orNull)
  }

  def estimatorTrainImageFeature(estimator: Estimator[T],
                                 trainSet: FeatureSet[ImageFeature],
                                 criterion: Criterion[T],
                                 endTrigger: Trigger = null,
                                 checkPointTrigger: Trigger = null,
                                 validationSet: FeatureSet[ImageFeature] = null,
                                 validationMethod: JList[ValidationMethod[T]] = null,
                                 batchSize: Int)
  : estimator.type = {
    val imageFeature2batch = ImageFeatureToMiniBatch(batchSize)
    val trainMiniBatch = trainSet -> imageFeature2batch
    val validationMiniBatch = if (validationSet != null) {
      validationSet -> imageFeature2batch
    } else {
      null
    }
    val valMethods = if (validationMethod != null) {
      validationMethod.asScala.toArray
    } else {
      null
    }

    estimator.train(trainMiniBatch, criterion,
      Option(endTrigger), Option(checkPointTrigger),
      validationMiniBatch, valMethods)
  }

  def clearGradientClipping(estimator: Estimator[T]): Unit = {
    estimator.clearGradientClipping()
  }

  def setConstantGradientClipping(estimator: Estimator[T], min: Double, max: Double): Unit = {
    estimator.setConstantGradientClipping(min, max)
  }

  def setGradientClippingByL2Norm(estimator: Estimator[T], clipNorm: Double): Unit = {
    estimator.setGradientClippingByL2Norm(clipNorm)
  }

  def estimatorSetTensorBoard(
    estimator: Estimator[T],
    logDir: String,
    appName: String): Unit = {
    estimator.setTensorBoard(logDir, appName)
  }

  def estimatorGetScalarFromSummary(estimator: Estimator[T], tag: String,
                               target: String): JList[JList[Any]] = {
    require(target == "Train" || target == "Validation",
      "Invalid target, must be Train or Validation.")
    val scalarArray = if (target == "Train") estimator.getTrainSummary(tag)
    else estimator.getValidationSummary(tag)

    if (scalarArray != null) {
      scalarArray.toList.map { tuple =>
        List(tuple._1, tuple._2, tuple._3).asJava.asInstanceOf[JList[Any]]
      }.asJava
    } else {
      null
    }
  }

  protected def toEvaluatedResult(evalResult: Map[ValidationMethod[T], ValidationResult]
                                ): JList[EvaluatedResult] = {
    val evalResultArray = evalResult.map(result =>
      EvaluatedResult(result._2.result()._1, result._2.result()._2, result._1.toString()))
    evalResultArray.toList.asJava
  }
}
