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

/**
 * Measures top1 accuracy for classification problems.
 *
 * @param zeroBasedLabel Boolean. Whether target labels start from 0. Default is true.
 *                       If false, labels start from 1.
 */
class Accuracy[T](
    val zeroBasedLabel: Boolean = true)(implicit ev: TensorNumeric[T])
  extends Top1Accuracy[T] {

  override def apply(output: Activity, target: Activity):
  ValidationResult = {
    if (zeroBasedLabel) {
      super.apply(output, target.toTensor[T].clone().add(ev.fromType(1.0f)))
    }
    else {
      super.apply(output, target)
    }
  }
}

/**
 * Measures top5 accuracy for classification problems.
 *
 * @param zeroBasedLabel Boolean. Whether target labels start from 0. Default is true.
 *                       If false, labels start from 1.
 */
class Top5Accuracy[T](
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
