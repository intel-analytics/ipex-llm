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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.{Criterion, Module}
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.sql.DataFrame
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
class DLClassifier[@specialized(Float, Double) T: ClassTag](
    override val model: Module[T],
    override val criterion : Criterion[T],
    override val featureSize : Array[Int],
    override val uid: String = "DLClassifier"
  )(implicit ev: TensorNumeric[T])
  extends DLEstimator[T](model, criterion, featureSize, Array(1)) {

  override protected def wrapBigDLModel(
      m: Module[T], featureSize: Array[Int]): DLClassifierModel[T] = {
    val dlModel = new DLClassifierModel[T](m, featureSize)
    copyValues(dlModel.setParent(this)).asInstanceOf[DLClassifierModel[T]]
  }

  override def transformSchema(schema : StructType): StructType = {
    validateSchema(schema)
    SchemaUtils.appendColumn(schema, $(predictionCol), DoubleType)
  }
}

/**
 * [[DLClassifierModel]] is a specialized [[DLModel]] for classification tasks.
 * The prediction column will have the datatype of Double.
 *
 * @param model BigDL module to be optimized
 * @param featureSize The size (Tensor dimensions) of the feature data.
 */
class DLClassifierModel[@specialized(Float, Double) T: ClassTag](
    override val model: Module[T],
    featureSize : Array[Int],
    override val uid: String = "DLClassifierModel"
  )(implicit ev: TensorNumeric[T]) extends DLModel[T](model, featureSize) {

  override protected def batchOutputToPrediction(output: Tensor[T]): Iterable[_] = {
    output.split(1)
    val result = if (output.dim == 2) {
      output.split(1).map(t => t.max(1)._2.storage().head)
    } else {
      throw new IllegalArgumentException
    }
    result.map(ev.toType[Double])
  }

  override def transformSchema(schema : StructType): StructType = {
    validateSchema(schema)
    SchemaUtils.appendColumn(schema, $(predictionCol), DoubleType)
  }
}

