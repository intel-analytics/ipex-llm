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
import com.intel.analytics.zoo.feature.common._
import com.intel.analytics.zoo.pipeline.nnframes.NNModel.NNModelWriter
import org.apache.spark.ml.DefaultParamsWriterWrapper
import org.apache.spark.ml.adapter.SchemaUtils
import org.apache.spark.ml.param.{DoubleParam, ParamMap}
import org.apache.spark.ml.util.{Identifiable, MLReadable, MLReader}
import org.apache.spark.sql.types._
import org.json4s.DefaultFormats

import scala.reflect.ClassTag

/**
 * [[NNClassifier]] is a specialized [[NNEstimator]] that simplifies the data format for
 * classification tasks. It explicitly supports label column of DoubleType.
 * and the fitted [[NNClassifierModel]] will have the prediction column of DoubleType.
 *
 * @param model BigDL module to be optimized
 * @param criterion  BigDL criterion method
 */
class NNClassifier[T: ClassTag] private[zoo]  (
    @transient override val model: Module[T],
    override val criterion : Criterion[T],
    override val uid: String = Identifiable.randomUID("nnClassifier")
  )(implicit ev: TensorNumeric[T])
  extends NNEstimator[T](model, criterion) {

  override protected def wrapBigDLModel(m: Module[T]): NNClassifierModel[T] = {
    val classifierModel = new NNClassifierModel[T](m)
    val originBatchsize = classifierModel.getBatchSize
    copyValues(classifierModel.setParent(this)).setBatchSize(originBatchsize)
    val clonedTransformer = ToTuple() -> $(samplePreprocessing)
      .asInstanceOf[Preprocessing[(Any, Option[Any]), Sample[T]]].clonePreprocessing()
    classifierModel.setSamplePreprocessing(clonedTransformer)
  }

  override def transformSchema(schema : StructType): StructType = {
    validateParams(schema)
    SchemaUtils.appendColumn(schema, $(predictionCol), DoubleType)
  }

  override def copy(extra: ParamMap): NNClassifier[T] = {
    val copied = copyValues(
      new NNClassifier[T](
        model.cloneModule(),
        criterion.cloneCriterion(),
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

object NNClassifier {

  /**
   * Construct a [[NNClassifier]] with default Preprocessing, SeqToTensor
   *
   * @param model BigDL module to be optimized
   * @param criterion  BigDL criterion method
   */
  def apply[T: ClassTag](
      model: Module[T],
      criterion: Criterion[T]
    )(implicit ev: TensorNumeric[T]): NNClassifier[T] = {
    new NNClassifier(model, criterion)
        .setSamplePreprocessing(FeatureLabelPreprocessing(SeqToTensor(), ScalarToTensor()))
  }

  /**
   * Construct a [[NNClassifier]] with a feature size. The constructor is useful
   * when the feature column contains the following data types:
   * Float, Double, Int, Array[Float], Array[Double], Array[Int] and MLlib Vector. The feature
   * data are converted to Tensors with the specified sizes before sending to the model.
   *
   * @param model BigDL module to be optimized
   * @param criterion  BigDL criterion method
   * @param featureSize The size (Tensor dimensions) of the feature data. e.g. an image may be with
   *                    width * height = 28 * 28, featureSize = Array(28, 28).
   */
  def apply[T: ClassTag](
      model: Module[T],
      criterion: Criterion[T],
      featureSize: Array[Int]
    )(implicit ev: TensorNumeric[T]): NNClassifier[T] = {
    new NNClassifier(model, criterion)
        .setSamplePreprocessing(
          FeatureLabelPreprocessing(SeqToTensor(featureSize), ScalarToTensor()))
  }


  /**
   * Construct a [[NNClassifier]] with multiple input sizes. The constructor is useful
   * when the feature column and label column contains the following data types:
   * Float, Double, Int, Array[Float], Array[Double], Array[Int] and MLlib Vector. The feature
   * data are converted to Tensors with the specified sizes before sending to the model.
   *
   * This API is used for multi-input model, where user need to specify the tensor sizes for
   * each of the model input.
   *
   * @param model module to be optimized
   * @param criterion  criterion method
   * @param featureSize The sizes (Tensor dimensions) of the feature data.
   */
  def apply[T: ClassTag](
      model: Module[T],
      criterion: Criterion[T],
      featureSize : Array[Array[Int]]
    )(implicit ev: TensorNumeric[T]): NNClassifier[T] = {
    new NNClassifier(model, criterion)
      .setSamplePreprocessing(FeatureLabelPreprocessing(
        SeqToMultipleTensors(featureSize), ScalarToTensor()
      )
    )
  }

  /**
   * Construct a [[NNClassifier]] with a feature Preprocessing.
   *
   * @param model BigDL module to be optimized
   * @param criterion  BigDL criterion method
   * @param featurePreprocessing Preprocessing[F, Tensor[T] ].
   */
  def apply[F, T: ClassTag](
      model: Module[T],
      criterion: Criterion[T],
      featurePreprocessing: Preprocessing[F, Tensor[T]]
    )(implicit ev: TensorNumeric[T]): NNClassifier[T] = {
    new NNClassifier(model, criterion)
        .setSamplePreprocessing(
          FeatureLabelPreprocessing(featurePreprocessing, ScalarToTensor()))
  }
}

/**
 * [[NNClassifierModel]] is a specialized [[NNModel]] for classification tasks.
 * The prediction column will have the data type of Double.
 *
 * @param model trained BigDL models to use in prediction.
 */
class NNClassifierModel[T: ClassTag] private[zoo] (
    @transient override val model: Module[T],
    override val uid: String = Identifiable.randomUID("nnClassifierModel")
  )(implicit ev: TensorNumeric[T]) extends NNModel[T](model) {

  /**
   * Param for threshold in binary classification prediction.
   *
   * The threshold applies to the raw output of the model. If the output is greater than
   * threshold, then predict 1, else 0. A high threshold encourages the model to predict 0
   * more often; a low threshold encourages the model to predict 1 more often.
   *
   * Note: the param is different from the one in Spark ProbabilisticClassifier which is compared
   * against estimated probability.
   *
   * Default is 0.5.
   */
  final val threshold = new DoubleParam(this, "threshold", "threshold in binary" +
    " classification prediction")

  def getThreshold: Double = $(threshold)

  def setThreshold(value: Double): this.type = {
    set(threshold, value)
  }
  setDefault(threshold, 0.5)

  protected override def outputToPrediction(output: Tensor[T]): Any = {
    if (output.size().deep == Array(1).deep) {
      val raw = ev.toType[Double](output.toArray().head)
      if (raw > 0.5) 1.0 else 0.0
    } else {
      ev.toType[Double](output.max(1)._2.valueAt(1))
    }
  }

  override def transformSchema(schema : StructType): StructType = {
    SchemaUtils.appendColumn(schema, $(predictionCol), DoubleType)
  }

  override def copy(extra: ParamMap): NNClassifierModel[T] = {
    val copied = new NNClassifierModel(model.cloneModule(), uid).setParent(parent)
    copyValues(copied, extra).asInstanceOf[NNClassifierModel[T]]
  }
}

object NNClassifierModel extends MLReadable[NNClassifierModel[_]] {

  /**
   * Construct a [[NNClassifierModel]] with default Preprocessing, SeqToTensor
   *
   * @param model BigDL module to be optimized
   */
  def apply[T: ClassTag](
      model: Module[T]
    )(implicit ev: TensorNumeric[T]): NNClassifierModel[T] = {
    new NNClassifierModel(model)
      .setSamplePreprocessing(SeqToTensor() -> TensorToSample())
  }

  /**
   * Construct a [[NNClassifierModel]] with a feature size. The constructor is useful
   * when the feature column contains the following data types:
   * Float, Double, Int, Array[Float], Array[Double], Array[Int] and MLlib Vector. The feature
   * data are converted to Tensors with the specified sizes before sending to the model.
   *
   * @param model BigDL module to be optimized
   * @param featureSize The size (Tensor dimensions) of the feature data. e.g. an image may be with
   *                    width * height = 28 * 28, featureSize = Array(28, 28).
   */
  def apply[T: ClassTag](
      model: Module[T],
      featureSize : Array[Int]
    )(implicit ev: TensorNumeric[T]): NNClassifierModel[T] = {
    new NNClassifierModel(model)
      .setSamplePreprocessing(SeqToTensor(featureSize) -> TensorToSample())
  }

  /**
   * Construct a [[NNClassifierModel]] with sizes of multiple model inputs. The constructor is
   * useful when the feature column contains the following data types:
   * Float, Double, Int, Array[Float], Array[Double], Array[Int] and MLlib Vector. The feature
   * data are converted to Tensors with the specified sizes before sending to the model.
   *
   * This API is used for multi-input model, where user need to specify the tensor sizes for
   * each of the model input.
   *
   * @param model model to be used, which should be a multi-input model.
   * @param featureSize The sizes (Tensor dimensions) of the feature data.
   */
  def apply[T: ClassTag](
      model: Module[T],
      featureSize : Array[Array[Int]]
    )(implicit ev: TensorNumeric[T]): NNClassifierModel[T] = {
    new NNClassifierModel(model)
      .setSamplePreprocessing(SeqToMultipleTensors(featureSize) -> MultiTensorsToSample())
  }

  /**
   * Construct a [[NNClassifierModel]] with a feature Preprocessing.
   *
   * @param model BigDL module to be optimized
   * @param featurePreprocessing Preprocessing[F, Tensor[T] ].
   */
  def apply[F, T: ClassTag](
      model: Module[T],
      featurePreprocessing: Preprocessing[F, Tensor[T]]
    )(implicit ev: TensorNumeric[T]): NNClassifierModel[T] = {
    new NNClassifierModel(model).setSamplePreprocessing(featurePreprocessing -> TensorToSample())
  }

  private[nnframes] class NNClassifierModelReader() extends MLReader[NNClassifierModel[_]] {
    import scala.language.existentials
    implicit val format: DefaultFormats.type = DefaultFormats
    override def load(path: String): NNClassifierModel[_] = {
      val (meta, model, typeTag, feaTran) = NNModel.getMetaAndModel(path, sc)
      val nnModel = typeTag match {
        case "TensorDouble" =>
          new NNClassifierModel[Double](model.asInstanceOf[Module[Double]], meta.uid)
            .setSamplePreprocessing(feaTran.asInstanceOf[Preprocessing[Any, Sample[Double]]])
        case "TensorFloat" =>
          new NNClassifierModel[Float](model.asInstanceOf[Module[Float]], meta.uid)
            .setSamplePreprocessing(feaTran.asInstanceOf[Preprocessing[Any, Sample[Float]]])
        case _ =>
          throw new Exception("Only support float and double for now")
      }

      DefaultParamsWriterWrapper.getAndSetParams(nnModel, meta)
      nnModel
    }
  }

  class NNClassifierModelWriter[T: ClassTag](
      instance: NNClassifierModel[T])(implicit ev: TensorNumeric[T])
    extends NNModelWriter[T](instance)

  override def read: MLReader[NNClassifierModel[_]] = {
    new NNClassifierModel.NNClassifierModelReader
  }
}
