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

import com.intel.analytics.bigdl.dataset.{Sample, Transformer}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.zoo.pipeline.nnframes.NNModel.NNModelWriter
import com.intel.analytics.zoo.pipeline.nnframes.transformers.{ArrayToTensor,
  MLlibVectorToTensor, NumToTensor, TensorToSample}
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
 * @param featureTransformer A transformer that transforms the feature data to a Tensor[T].
 *        Some pre-defined transformers are provided in package
 *        [[com.intel.analytics.zoo.pipeline.nnframes.transformers]]. E.g.
 *        [[ArrayToTensor]] is used to transform Array[_] in DataFrame to Tensor. For a feature
 *        column that contains 576 floats in an Array, Users can set ArrayToTensor(Array(28, 28))
 *        as featureTransformer, which will convert the feature data into Tensors with dimension
 *        28 * 28 to be processed by a convolution Model. For a simple linear model, user may
 *        just use ArrayToTensor(Array(576)), which will convert the data into Tensors with
 *        single dimension (576).
 *        [[MLlibVectorToTensor]] is used to transform [[org.apache.spark.mllib.linalg.Vector]]
 *        to a Tensor.
 *        [[NumToTensor]] transform a number to a Tensor with singel dimension of length 1.
 *        Multiple transformer can be combined as a ChainedTransformer.
 */
class NNClassifier[F, T: ClassTag](
    @transient override val model: Module[T],
    override val criterion : Criterion[T],
    val featureTransformer: Transformer[F, Tensor[T]],
    override val uid: String = Identifiable.randomUID("nnClassifier")
  )(implicit ev: TensorNumeric[T])
  extends NNEstimator[F, AnyVal, T](model, criterion, featureTransformer, NumToTensor()) {

  override protected def wrapBigDLModel(m: Module[T]): NNClassifierModel[F, T] = {
    val dlModel = new NNClassifierModel[F, T](m, featureTransformer)
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
        featureTransformer.cloneTransformer(),
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
 * The prediction column will have the data type of Double.
 *
 * @param model trained BigDL models to use in prediction.
 * @param sampleTransformer A transformer that transforms the feature data to a Tensor[T].
 */
class NNClassifierModel[F, T: ClassTag](
    @transient override val model: Module[T],
    sampleTransformer: Transformer[F, Sample[T]],
    override val uid: String = "DLClassifierModel"
  )(implicit ev: TensorNumeric[T]) extends NNModel[F, T](model, sampleTransformer) {

  /**
   * @param model trainned BigDL models to use in prediction.
   * @param featureTransformer A transformer that transforms the feature data to a Tensor[T].
   *        Some pre-defined transformers are provided in package
   *        [[com.intel.analytics.zoo.pipeline.nnframes.transformers]]. E.g.
   *        [[ArrayToTensor]] is used to transform Array[_] in DataFrame to Tensor. For a feature
   *        column that contains 576 floats in an Array, Users can set ArrayToTensor(Array(28, 28))
   *        as featureTransformer, which will convert the feature data into Tensors with dimension
   *        28 * 28 to be processed by a convolution Model. For a simple linear model, user may
   *        just use ArrayToTensor(Array(576)), which will convert the data into Tensors with
   *        single dimension (576).
   *        [[MLlibVectorToTensor]] is used to transform [[org.apache.spark.mllib.linalg.Vector]]
   *        to a Tensor.
   *        [[NumToTensor]] transform a number to a Tensor with singel dimension of length 1.
   *        Multiple transformer can be combined as a ChainedTransformer.
   */
  def this(
      model: Module[T],
      featureTransformer: Transformer[F, Tensor[T]]
    )(implicit ev: TensorNumeric[T]) =
    this(model, featureTransformer -> TensorToSample())

  protected override def outputToPrediction(output: Tensor[T]): Any = {
    ev.toType[Double](output.max(1)._2.valueAt(1))
  }

  override def transformSchema(schema : StructType): StructType = {
    SchemaUtils.appendColumn(schema, $(predictionCol), DoubleType)
  }

  override def copy(extra: ParamMap): NNClassifierModel[F, T] = {
    val copied = new NNClassifierModel(
      model.cloneModule(), sampleTransformer.cloneTransformer(), uid)
      .setParent(parent)
    copyValues(copied, extra).asInstanceOf[NNClassifierModel[F, T]]
  }
}

object NNClassifierModel extends MLReadable[NNClassifierModel[_, _]] {
  private[nnframes] class NNClassifierModelReader() extends MLReader[NNClassifierModel[_, _]] {
    import scala.language.existentials
    implicit val format: DefaultFormats.type = DefaultFormats
    override def load(path: String): NNClassifierModel[_, _] = {
      val (meta, model, typeTag, feaTran) = NNModel.getMetaAndModel(path, sc)
      val nnModel = typeTag match {
        case "TensorDouble" =>
          new NNClassifierModel[Any, Double](model.asInstanceOf[Module[Double]],
            feaTran.asInstanceOf[Transformer[Any, Sample[Double]]])
        case "TensorFloat" =>
          new NNClassifierModel[Any, Float](model.asInstanceOf[Module[Float]],
            feaTran.asInstanceOf[Transformer[Any, Sample[Float]]])
        case _ =>
          throw new Exception("Only support float and double for now")
      }

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
