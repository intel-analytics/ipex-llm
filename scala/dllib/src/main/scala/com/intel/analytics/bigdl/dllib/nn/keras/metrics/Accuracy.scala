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

package com.intel.analytics.zoo.pipeline.api.keras.metrics

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.optim.{AccuracyResult, Top1Accuracy, ValidationResult, Top5Accuracy => BigDLTop5Accuracy}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Measures top1 accuracy for multi-class classification
 * or accuracy for binary classification.
 *
 * @param zeroBasedLabel Boolean. Whether target labels start from 0. Default is true.
 *                       If false, labels start from 1.
 *                       Note that this only takes effect for multi-class classification.
 *                       For binary classification, labels ought to be 0 or 1.
 */
@deprecated("use SparseCategoricalAccuracy, CategoricalAccuracy or BinaryAccuracy instead", "0.5.0")
class Accuracy[T: ClassTag](
    val zeroBasedLabel: Boolean = true)(implicit ev: TensorNumeric[T])
  extends Top1Accuracy[T] {

  override def apply(output: Activity, target: Activity):
  ValidationResult = {
    val _output = output.toTensor[T]
    val binaryClassification = (_output.dim() == 2 && _output.size(2) == 1) ||
      (_output.dim() == 1 && _output.size(1) == 1)
    if (zeroBasedLabel && ! binaryClassification) {
      super.apply(output, target.toTensor[T].clone().add(ev.fromType(1.0f)))
    }
    else {
      super.apply(output, target)
    }
  }
}

/**
 * Measures top1 accuracy for multi-class classification with sparse target and zero-base index.
 *
 */
class SparseCategoricalAccuracy[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends
  Top1Accuracy[T] {
  override def apply(output: Activity, target: Activity):
  ValidationResult = {
    super.apply(output, target.toTensor[T].clone().add(ev.fromType(1.0f)))
  }
}


/**
 * Measures top1 accuracy for binary classification with zero-base index.
 *
 */
class BinaryAccuracy[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends Top1Accuracy[T]


/**
 * Measures top1 accuracy for multi-class with "one-hot" target.
 *
 */
class CategoricalAccuracy[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends
  Top1Accuracy[T] {
  override def apply(output: Activity, target: Activity):
  ValidationResult = {
    val _target = target.toTensor[T]
    val _output = output.toTensor[T]

    require(_target.dim() == 2, "Target should have 2 dims with one-hot encoding")
    require(_target.size().deep == _output.size().deep,
      s"${_target.size()} == ${_output.size()}")

    val bigdlTarget = if (_target.dim() == 2) {
      _target.max(2)._2
    } else {
      _target.max(1)._2
    }
    super.apply(output, bigdlTarget)
  }
}

/**
 * Measures top5 accuracy for multi-class classification.
 *
 * @param zeroBasedLabel Boolean. Whether target labels start from 0. Default is true.
 *                       If false, labels start from 1.
 */
class Top5Accuracy[T: ClassTag](
    val zeroBasedLabel: Boolean = true)(implicit ev: TensorNumeric[T])
  extends BigDLTop5Accuracy[T] {

  override def apply(output: Activity, target: Activity):
  AccuracyResult = {
    if (zeroBasedLabel) {
      super.apply(output, target.toTensor[T].clone().add(ev.fromType(1.0f)))
    }
    else {
      super.apply(output, target)
    }
  }
}
