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
package com.intel.analytics.bigdl.dlframes

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.{Criterion, Module}
import org.apache.spark.ml.adapter.SchemaUtils
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types._

import scala.reflect.ClassTag

/**
 * [[DLClassifier]] is a specialized [[DLEstimator]] that simplifies the data format for
 * classification tasks. It only supports label column of DoubleType.
 * and the fitted [[DLClassifierModel]] will have the prediction column of DoubleType.
 *
 * @param model BigDL module to be optimized
 * @param criterion  BigDL criterion method
 * @param featureSize The size (Tensor dimensions) of the feature data.
 */
@deprecated("`DLClassifier` is deprecated." +
  "com.intel.analytics.bigdl.dlframes is deprecated in BigDL 0.11, " +
  "and will be removed in future releases", "0.10.0")
class DLClassifier[T: ClassTag](
    @transient override val model: Module[T],
    override val criterion : Criterion[T],
    override val featureSize : Array[Int],
    override val uid: String = Identifiable.randomUID("dlClassifier")
  )(implicit ev: TensorNumeric[T])
  extends DLEstimator[T](model, criterion, featureSize, Array(1)) {

  override protected def wrapBigDLModel(
      m: Module[T], featureSize: Array[Int]): DLClassifierModel[T] = {
    val dlModel = new DLClassifierModel[T](m, featureSize)
    copyValues(dlModel.setParent(this)).asInstanceOf[DLClassifierModel[T]]
  }

  override def transformSchema(schema : StructType): StructType = {
    validateParams(schema)
    SchemaUtils.appendColumn(schema, $(predictionCol), DoubleType)
  }

  override def copy(extra: ParamMap): DLClassifier[T] = {
    copyValues(new DLClassifier(model, criterion, featureSize), extra)
  }
}

/**
 * [[DLClassifierModel]] is a specialized [[DLModel]] for classification tasks.
 * The prediction column will have the datatype of Double.
 *
 * @param model BigDL module to be optimized
 * @param featureSize The size (Tensor dimensions) of the feature data.
 */
@deprecated("`DLClassifierModel` is deprecated." +
  "com.intel.analytics.bigdl.dlframes is deprecated in BigDL 0.11, " +
  "and will be removed in future releases", "0.10.0")
class DLClassifierModel[T: ClassTag](
    @transient override val model: Module[T],
    featureSize : Array[Int],
    override val uid: String = "DLClassifierModel"
  )(implicit ev: TensorNumeric[T]) extends DLModel[T](model, featureSize) {

  protected override def outputToPrediction(output: Tensor[T]): Any = {
    if (output.size().deep == Array(1).deep) {
      val raw = ev.toType[Double](output.toArray().head)
      if (raw > 0.5) 1.0 else 0.0
    } else {
      ev.toType[Double](output.max(1)._2.valueAt(1))
    }
  }

  override def transformSchema(schema : StructType): StructType = {
    validateDataType(schema, $(featuresCol))
    SchemaUtils.appendColumn(schema, $(predictionCol), DoubleType)
  }
}

