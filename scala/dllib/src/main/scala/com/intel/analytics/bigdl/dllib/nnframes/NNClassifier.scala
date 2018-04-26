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

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.zoo.pipeline.nnframes.NNModel.NNModelWriter
import com.intel.analytics.zoo.pipeline.nnframes.transformers.NumToTensor
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
 */
class NNClassifier[F, T: ClassTag](
    @transient override val model: Module[T],
    override val criterion : Criterion[T],
    override val featureTransformers: Transformer[F, Tensor[T]],
    override val uid: String = Identifiable.randomUID("nnClassifier")
  )(implicit ev: TensorNumeric[T])
  extends NNEstimator[F, AnyVal, T](model, criterion, featureTransformers, NumToTensor()) {

  override protected def wrapBigDLModel(m: Module[T]): NNClassifierModel[F, T] = {
    val dlModel = new NNClassifierModel[F, T](m, featureTransformers)
    copyValues(dlModel.setParent(this)).asInstanceOf[NNClassifierModel[F, T]]
  }

  override def transformSchema(schema : StructType): StructType = {
    validateParams(schema)
    SchemaUtils.appendColumn(schema, $(predictionCol), DoubleType)
  }

  override def copy(extra: ParamMap): NNClassifier[F, T] = {
    val copied = copyValues(
      new NNClassifier[F, T](
        model.cloneModule(),
        criterion.cloneCriterion(),
        featureTransformers.cloneTransformer(),
        this.uid
      ),
      extra)

    if (this.validationTrigger.isDefined) {
      copied.setValidation(
        validationTrigger.get, validationDF, validationMethods.clone(), validationBatchSize)
    }
    copied
  }
}

/**
 * [[NNClassifierModel]] is a specialized [[NNModel]] for classification tasks.
 * The prediction column will have the datatype of Double.
 *
 * @param model BigDL module to be optimized
 */
class NNClassifierModel[F, T: ClassTag](
    @transient override val model: Module[T],
    featureTransformers: Transformer[F, Tensor[T]],
    override val uid: String = "DLClassifierModel"
  )(implicit ev: TensorNumeric[T]) extends NNModel[F, T](model, featureTransformers) {

  protected override def outputToPrediction(output: Tensor[T]): Any = {
    ev.toType[Double](output.max(1)._2.valueAt(1))
  }

  override def transformSchema(schema : StructType): StructType = {
    SchemaUtils.appendColumn(schema, $(predictionCol), DoubleType)
  }

  override def copy(extra: ParamMap): NNClassifierModel[F, T] = {
    val copied = new NNClassifierModel(
      model.cloneModule(), featureTransformers.cloneTransformer(), uid)
      .setParent(parent)
    copyValues(copied, extra).asInstanceOf[NNClassifierModel[F, T]]
  }
}

object NNClassifierModel extends MLReadable[NNClassifierModel[_, _]] {
  private[nnframes] class NNClassifierModelReader() extends MLReader[NNClassifierModel[_, _]] {
    import scala.language.existentials
    implicit val format: DefaultFormats.type = DefaultFormats
    override def load(path: String): NNClassifierModel[_, _] = {
      val (meta, model, typeTag) = NNModel.getMetaAndModel(path, sc)
      val nnModel = null
//        typeTag match {
//        case "TensorDouble" =>
//          new NNClassifierModel[_, Double](model.asInstanceOf[Module[Double]], null)
//        case "TensorFloat" =>
//          new NNClassifierModel[_, Float](model.asInstanceOf[Module[Float]], null)
//        case _ =>
//          throw new Exception("Only support float and double for now")
//      }

      DefaultParamsWriterWrapper.getAndSetParams(nnModel, meta)
      nnModel
    }
  }

  class NNClassifierModelWriter[T: ClassTag](
      instance: NNClassifierModel[_, T])(implicit ev: TensorNumeric[T])
    extends NNModelWriter[T](instance)

  override def read: MLReader[NNClassifierModel[_, _]] = {
    new NNClassifierModel.NNClassifierModelReader
  }
}
