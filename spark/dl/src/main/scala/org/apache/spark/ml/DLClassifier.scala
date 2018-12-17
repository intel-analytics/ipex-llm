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
package org.apache.spark.ml

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.{Criterion, Module}
import org.apache.spark.ml.util.Identifiable

import scala.reflect.ClassTag

/**
 * Deprecated. Please refer to package com.intel.analytics.bigdl.dlframes.
 *
 * [[DLClassifier]] is a specialized [[DLEstimator]] that simplifies the data format for
 * classification tasks. It only supports label column of DoubleType.
 * and the fitted [[DLClassifierModel]] will have the prediction column of DoubleType.
 *
 * @param model BigDL module to be optimized
 * @param criterion  BigDL criterion method
 * @param featureSize The size (Tensor dimensions) of the feature data.
 */
@deprecated("`DLClassifier` has been migrated to package `com.intel.analytics.bigdl.dlframes`." +
  "This will be removed in BigDL 0.6.", "0.5.0")
class DLClassifier[T: ClassTag](
    @transient override val model: Module[T],
    override val criterion : Criterion[T],
    override val featureSize : Array[Int],
    override val uid: String = Identifiable.randomUID("dlClassifier")
  )(implicit ev: TensorNumeric[T])
  extends com.intel.analytics.bigdl.dlframes.DLClassifier[T](model, criterion, featureSize) {

  override protected def wrapBigDLModel(
      m: Module[T], featureSize: Array[Int]): DLClassifierModel[T] = {
    val dlModel = new DLClassifierModel[T](m, featureSize)
    copyValues(dlModel.setParent(this)).asInstanceOf[DLClassifierModel[T]]
  }
}

/**
 * Deprecated. Please refer to package com.intel.analytics.bigdl.dlframes.
 *
 * [[DLClassifierModel]] is a specialized [[DLModel]] for classification tasks.
 * The prediction column will have the datatype of Double.
 *
 * @param model BigDL module to be optimized
 * @param featureSize The size (Tensor dimensions) of the feature data.
 */
@deprecated("`DLClassifierModel` is migrated to package `com.intel.analytics.bigdl.dlframes`." +
  "This will be removed in BigDL 0.6.", "0.5.0")
class DLClassifierModel[T: ClassTag](
    @transient override val model: Module[T],
    featureSize : Array[Int],
    override val uid: String = "DLClassifierModel"
  )(implicit ev: TensorNumeric[T])
  extends com.intel.analytics.bigdl.dlframes.DLClassifierModel[T](model, featureSize)
