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
package com.intel.analytics.zoo.pipeline.estimator

import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.dataset.MiniBatch
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.feature.{DistributedFeatureSet, FeatureSet}
import com.intel.analytics.zoo.pipeline.api.keras.models.InternalDistriOptimizer

import scala.reflect.ClassTag

/**
 * Abstract interface for Estimator.
 * Estimator is a class to train and evaluate BigDL models.
 */
trait AbstractEstimator[T]{
  def train(trainSet: FeatureSet[MiniBatch[T]],
            criterion: Criterion[T] = null,
            endTrigger: Option[Trigger] = None,
            checkPointTrigger: Option[Trigger] = None,
            validationSet: FeatureSet[MiniBatch[T]] = null,
            validationMethod: Array[ValidationMethod[T]] = null): this.type

  def evaluate(validationSet: FeatureSet[MiniBatch[T]],
               validationMethod: Array[ValidationMethod[T]]
              ): Map[ValidationMethod[T], ValidationResult]

}

/**
 * Estimator class for training and evaluation BigDL models.
 *
 * Estimator wraps a model, and provide an uniform training, evaluation or prediction operation
 * on both local host and distributed spark environment.
 *
 * @param model model
 * @param optimMethods optimMethods to optimize model
 * @param modelDir model checkpoint directory, and related summary directory.
 * @tparam T tensor numeric type
 */
class Estimator[T: ClassTag] private[zoo](
      model: Module[T],
      optimMethods: Map[String, OptimMethod[T]] = Map(),
      modelDir: Option[String] = None)(implicit ev: TensorNumeric[T]) extends AbstractEstimator[T] {
  protected var internalEstimator: AbstractEstimator[T] = null

  /**
   * Train model with provided trainSet and criterion.
   * The training will end until the endTrigger is triggered.
   * During the training, if checkPointTrigger is defined and triggered,
   * the model will be saved to modelDir. And if validationSet and validationMethod
   * is defined, the model will be evaluated at the checkpoint.
   *
   * @param trainSet training FeatureSet
   * @param criterion Loss function
   * @param endTrigger When to finish the training
   * @param checkPointTrigger When to save a checkpoint and evaluate model.
   * @param validationSet Validation FeatureSet.
   * @param validationMethod Validation Methods.
   * @return self
   */
  override def train(trainSet: FeatureSet[MiniBatch[T]],
            criterion: Criterion[T],
            endTrigger: Option[Trigger] = None,
            checkPointTrigger: Option[Trigger] = None,
            validationSet: FeatureSet[MiniBatch[T]] = null,
            validationMethod: Array[ValidationMethod[T]] = null): this.type = {
    if (internalEstimator == null) {
      internalEstimator = trainSet match {
        case d: DistributedFeatureSet[MiniBatch[T]] =>
          new InternalDistriOptimizer[T](model, null, criterion)
            .setCheckpointDir(modelDir)
            .setOptimMethods(optimMethods)
        case _ => throw new IllegalArgumentException("Unsupported FeatureSet type.")
      }
    }
    internalEstimator.train(trainSet, criterion, endTrigger, checkPointTrigger,
      validationSet, validationMethod)
    this
  }

  /**
   * Evaluate the model on the validationSet with the validationMethods.
   * @param validationSet validation FeatureSet
   * @param validationMethod validation methods
   * @return validation results
   */
  override def evaluate(validationSet: FeatureSet[MiniBatch[T]],
                        validationMethod: Array[ValidationMethod[T]]
              ): Map[ValidationMethod[T], ValidationResult] = {
    if (internalEstimator == null) {
      internalEstimator = validationSet match {
        case d: DistributedFeatureSet[MiniBatch[T]] =>
          new InternalDistriOptimizer[T](model, null, null)
            .setCheckpointDir(modelDir)
            .setOptimMethods(optimMethods)
        case _ => throw new IllegalArgumentException("Unsupported FeatureSet type.")
      }
    }
    internalEstimator.evaluate(validationSet, validationMethod)
  }
}

object Estimator {
  /**
   * Create an estimator
   * @param model model
   * @param optimMethods optimMethods to optimize model, submodule names and optimMethod pairs.
   * @param modelDir model checkpoint directory, and related summary directory.
   * @tparam T tensor numeric type
   * @return a new estimator
   */
  def apply[T: ClassTag](
        model: Module[T],
        optimMethods: Map[String, OptimMethod[T]],
        modelDir: String)(implicit ev: TensorNumeric[T]): AbstractEstimator[T] = {
    if (null != modelDir && "" != modelDir) {
      new Estimator[T](model, optimMethods, Some(modelDir))
    } else {
      new Estimator[T](model, optimMethods)
    }
  }

  /**
   * Create an estimator
   * @param model model
   * @param optimMethods optimMethods to optimize model, submodule names and optimMethod pairs.
   * @tparam T tensor numeric type
   * @return a new estimator
   */
  def apply[T: ClassTag](
       model: Module[T],
       optimMethods: Map[String, OptimMethod[T]]
      )(implicit ev: TensorNumeric[T]): AbstractEstimator[T] = {
    apply(model, optimMethods, "")
  }

  /**
   * Create an estimator
   * @param model model
   * @param optimMethods optimMethod to optimize model
   * @param modelDir model checkpoint directory, and related summary directory.
   * @tparam T tensor numeric type
   * @return a new estimator
   */
  def apply[T: ClassTag](
        model: Module[T],
        optimMethod: OptimMethod[T],
        modelDir: String)(implicit ev: TensorNumeric[T]): AbstractEstimator[T] = {
    if (null != modelDir && "" != modelDir) {
      new Estimator[T](model, Map(model.getName() -> optimMethod), Some(modelDir))
    } else {
      new Estimator[T](model, Map(model.getName() -> optimMethod))
    }
  }

  /**
   * Create an estimator
   * @param model model
   * @param optimMethods optimMethod to optimize model
   * @tparam T tensor numeric type
   * @return a new estimator
   */
  def apply[T: ClassTag](
        model: Module[T],
        optimMethod: OptimMethod[T])(implicit ev: TensorNumeric[T]): AbstractEstimator[T] = {
    apply(model, optimMethod, "")
  }

  /**
   * Create an estimator
   * @param model model
   * @tparam T tensor numeric type
   * @return a new estimator
   */
  def apply[T: ClassTag](
        model: Module[T])(implicit ev: TensorNumeric[T]): AbstractEstimator[T] = {
    new Estimator[T](model, Map())
  }

}
