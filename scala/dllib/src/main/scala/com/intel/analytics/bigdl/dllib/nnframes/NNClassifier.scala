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
import com.intel.analytics.zoo.pipeline.nnframes.NNModel.NNModelWriter
import org.apache.spark.ml.DefaultParamsWriterWrapper
import org.apache.spark.ml.adapter.SchemaUtils
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{Identifiable, MLReadable, MLReader}
import org.apache.spark.sql.types._
import org.json4s.DefaultFormats

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

object NNClassifierModel extends MLReadable[NNClassifierModel[_]] {
  private[nnframes] class NNClassifierModelReader() extends MLReader[NNClassifierModel[_]] {
    import scala.language.existentials
    implicit val format: DefaultFormats.type = DefaultFormats
    override def load(path: String): NNClassifierModel[_] = {
      val (meta, model, typeTag) = NNModel.getMetaAndModel(path, sc)
      val featureSize = (meta.metadata \ "featureSize").extract[Seq[Int]].toArray
      val nnModel = typeTag match {
        case "TensorDouble" =>
          new NNClassifierModel[Double](model.asInstanceOf[Module[Double]], featureSize)
        case "TensorFloat" =>
          new NNClassifierModel[Float](model.asInstanceOf[Module[Float]], featureSize)
        case _ =>
          throw new Exception("Only support float and double for now")
      }

      DefaultParamsWriterWrapper.getAndSetParams(nnModel, meta)
      nnModel
    }
  }

  class NNClassifierModelWriter[T: ClassTag]
  (instance: NNClassifierModel[T])(implicit ev: TensorNumeric[T]) extends NNModelWriter[T](instance)

  override def read: MLReader[NNClassifierModel[_]] = {
    new NNClassifierModel.NNClassifierModelReader
  }
}
