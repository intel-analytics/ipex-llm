/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.estimator.python

import java.util.{ArrayList => JArrayList, HashMap => JHashMap, List => JList, Map => JMap}

import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.dataset.{ByteRecord, MiniBatch, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.optim.{OptimMethod, Trigger, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFeatureToMiniBatch}
import com.intel.analytics.zoo.common.PythonZoo
import com.intel.analytics.zoo.feature.{DistributedFeatureSet, FeatureSet}
import com.intel.analytics.zoo.pipeline.estimator.Estimator

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
                      optimMethods: Map[String, OptimMethod[T]],
                      modelDir: String): Estimator[T] = {
    Estimator(model, optimMethods, modelDir)
  }

  def estimatorEvaluate(estimator: Estimator[T], validationSet: FeatureSet[Sample[T]],
                        validationMethod: Array[ValidationMethod[T]], batchSize: Int
                       ): Map[ValidationMethod[T], ValidationResult] = {
    val sample2batch = SampleToMiniBatch(batchSize)
    val validationMiniBatch = validationSet -> sample2batch
    estimator.evaluate(validationMiniBatch, validationMethod)
  }

  def estimatorEvaluateImageFeature(estimator: Estimator[T],
                                    validationSet: FeatureSet[ImageFeature],
                                    validationMethod: Array[ValidationMethod[T]],
                                    batchSize: Int
                                   ): Map[ValidationMethod[T], ValidationResult] = {
    val imageFeature2batch = ImageFeatureToMiniBatch(batchSize)
    val validationMiniBatch = validationSet -> imageFeature2batch
    estimator.evaluate(validationMiniBatch, validationMethod)
  }

  def estimatorTrain(estimator: Estimator[T], trainSet: FeatureSet[Sample[T]],
                     criterion: Criterion[T],
                     endTrigger: Trigger = null,
                     checkPointTrigger: Trigger = null,
                     validationSet: FeatureSet[Sample[T]] = null,
                     validationMethod: Array[ValidationMethod[T]] = null, batchSize: Int)
  : estimator.type = {
    val sample2batch = SampleToMiniBatch(batchSize)
    val trainMiniBatch = trainSet -> sample2batch
    val validationMiniBatch = if (validationSet != null) {
      validationSet -> sample2batch
    } else {
      null
    }

    estimator.train(trainMiniBatch, criterion,
      Some(endTrigger), Some(checkPointTrigger),
      validationMiniBatch, validationMethod)
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
      Some(endTrigger), Some(checkPointTrigger),
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
}
