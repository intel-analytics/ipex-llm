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

import scala.reflect.ClassTag


/**
 * Deprecated. Please refer to package com.intel.analytics.bigdl.dlframes.
 *
 * [[DLEstimator]] helps to train a BigDL Model with the Spark ML Estimator/Transfomer pattern,
 * thus Spark users can conveniently fit BigDL into Spark ML pipeline.
 *
 * [[DLEstimator]] supports feature and label data in the format of
 * Array[Double], Array[Float], org.apache.spark.mllib.linalg.{Vector, VectorUDT},
 * org.apache.spark.ml.linalg.{Vector, VectorUDT}, Double and Float.
 *
 * User should specify the feature data dimensions and label data dimensions via the constructor
 * parameters featureSize and labelSize respectively. Internally the feature and label data are
 * converted to BigDL tensors, to further train a BigDL model efficiently.
 *
 * For details usage, please refer to examples in package
 * com.intel.analytics.bigdl.example.MLPipeline
 *
 * @param model BigDL module to be optimized
 * @param criterion  BigDL criterion method
 * @param featureSize The size (Tensor dimensions) of the feature data. e.g. an image may be with
 *                    width * height = 28 * 28, featureSize = Array(28, 28).
 * @param labelSize The size (Tensor dimensions) of the label data.
 */
@deprecated("`DLEstimator` has been migrated to package `com.intel.analytics.bigdl.dlframes`." +
  "org.apache.spark.ml.DLEstimator will be removed in BigDL 0.6.", "0.5.0")
class DLEstimator[T: ClassTag](
    @transient override val model: Module[T],
    override val criterion : Criterion[T],
    featureSize : Array[Int],
    override val labelSize : Array[Int],
    override val uid: String = "DLEstimator")(implicit ev: TensorNumeric[T])
  extends com.intel.analytics.bigdl.dlframes.DLEstimator[T](
    model, criterion, featureSize, labelSize) {

  override protected def wrapBigDLModel(m: Module[T], featureSize: Array[Int]): DLModel[T] = {
    val dlModel = new DLModel[T](m, featureSize)
    copyValues(dlModel.setParent(this)).asInstanceOf[DLModel[T]]
  }
}


/**
 * Deprecated. Please refer to package com.intel.analytics.bigdl.dlframes.
 *
 * [[DLModel]] helps embed a BigDL model into a Spark Transformer, thus Spark users can
 * conveniently merge BigDL into Spark ML pipeline.
 * [[DLModel]] supports feature data in the format of
 * Array[Double], Array[Float], org.apache.spark.mllib.linalg.{Vector, VectorUDT},
 * org.apache.spark.ml.linalg.{Vector, VectorUDT}, Double and Float.
 * Internally [[DLModel]] use features column as storage of the feature data, and create
 * Tensors according to the constructor parameter featureSize.
 *
 * [[DLModel]] is compatible with both spark 1.5-plus and 2.0 by extending ML Transformer.
 * @param model trainned BigDL models to use in prediction.
 * @param featureSize The size (Tensor dimensions) of the feature data. (e.g. an image may be with
 * featureSize = 28 * 28).
 */
@deprecated("`DLModel` has been migrated to package `com.intel.analytics.bigdl.dlframes`." +
  "This will be removed in BigDL 0.6.", "0.5.0")
class DLModel[T: ClassTag](
    @transient override val model: Module[T],
    featureSize : Array[Int],
    override val uid: String = "DLModel"
    )(implicit ev: TensorNumeric[T])
  extends com.intel.analytics.bigdl.dlframes.DLModel[T](model, featureSize)


// TODO, add save/load
object DLModel {

}

