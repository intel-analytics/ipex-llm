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

package com.intel.analytics.zoo.pipeline.nnframes

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.{Criterion, Module}
import org.apache.spark.ml.adapter.SchemaUtils
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types._

import scala.reflect.ClassTag

/**
 * [[NNClassifier]] is a specialized [[NNEstimator]] that simplifies the data format for
 * classification tasks. It only supports label column of DoubleType.
 * and the fitted [[NNClassifierModel]] will have the prediction column of DoubleType.
 *
 * @param model BigDL module to be optimized
 * @param criterion  BigDL criterion method
 * @param featureSize The size (Tensor dimensions) of the feature data.
 */
class NNClassifier[T: ClassTag](
    @transient override val model: Module[T],
    override val criterion : Criterion[T],
    override val featureSize : Array[Int],
    override val uid: String = Identifiable.randomUID("dlClassifier")
  )(implicit ev: TensorNumeric[T])
  extends NNEstimator[T](model, criterion, featureSize, Array(1)) {

  override protected def wrapBigDLModel(
      m: Module[T], featureSize: Array[Int]): NNClassifierModel[T] = {
    val dlModel = new NNClassifierModel[T](m, featureSize)
    copyValues(dlModel.setParent(this)).asInstanceOf[NNClassifierModel[T]]
  }

  override def transformSchema(schema : StructType): StructType = {
    validateParams(schema)
    SchemaUtils.appendColumn(schema, $(predictionCol), DoubleType)
  }

  override def copy(extra: ParamMap): NNClassifier[T] = {
    copyValues(new NNClassifier(model, criterion, featureSize), extra)
  }
}

/**
 * [[NNClassifierModel]] is a specialized [[NNModel]] for classification tasks.
 * The prediction column will have the datatype of Double.
 *
 * @param model BigDL module to be optimized
 * @param featureSize The size (Tensor dimensions) of the feature data.
 */
class NNClassifierModel[T: ClassTag](
    @transient override val model: Module[T],
    featureSize : Array[Int],
    override val uid: String = "DLClassifierModel"
  )(implicit ev: TensorNumeric[T]) extends NNModel[T](model, featureSize) {

  protected override def outputToPrediction(output: Tensor[T]): Any = {
    ev.toType[Double](output.max(1)._2.valueAt(1))
  }

  override def transformSchema(schema : StructType): StructType = {
    validateDataType(schema, $(featuresCol))
    SchemaUtils.appendColumn(schema, $(predictionCol), DoubleType)
  }
}
