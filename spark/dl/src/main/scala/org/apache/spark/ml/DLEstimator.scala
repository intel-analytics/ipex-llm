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

import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.T
import org.apache.spark.ml.param.{IntParam, ParamMap, ParamValidators, _}
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}

import scala.reflect.ClassTag

/**
 * [[DLEstimator]] helps to train a BigDL Model with the Spark ML Estimator/Transfomer pattern,
 * thus Spark users can conveniently fit BigDL into Spark ML pipeline.
 *
 * [[DLEstimator]] supports feature and label data in the format of Array[Double], Array[Float],
 * org.apache.spark.mllib.linalg.{Vector, VectorUDT} for Spark 1.5, 1.6 and
 * org.apache.spark.ml.linalg.{Vector, VectorUDT} for Spark 2.0+. Also label data can be of
 * DoubleType.
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
class DLEstimator[@specialized(Float, Double) T: ClassTag](
    val model: Module[T],
    val criterion : Criterion[T],
    val featureSize : Array[Int],
    val labelSize : Array[Int],
    override val uid: String = "DLEstimator")(implicit ev: TensorNumeric[T])
  extends DLEstimatorBase[DLEstimator[T], DLModel[T]] with DLParams with HasBatchSize {

  def setFeaturesCol(featuresColName: String): this.type = set(featuresCol, featuresColName)

  def setLabelCol(labelColName : String) : this.type = set(labelCol, labelColName)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setBatchSize(value: Int): this.type = set(batchSize, value)

  /**
   * optimization method to be used. BigDL supports many optimization methods like Adam,
   * SGD and LBFGS. Refer to package com.intel.analytics.bigdl.optim for all the options.
   * Default: SGD
   */
  val optimMethod = new Param[OptimMethod[_]](this, "optimMethod", "optimMethod")

  def getOptimMethod: OptimMethod[_] = $(optimMethod)

  def setOptimMethod(value: OptimMethod[_]): this.type = set(optimMethod, value)

  /**
   * number of max Epoch for the training, an epoch refers to a traverse over the training data
   * Default: 100
   */
  val maxEpoch = new IntParam(this, "maxEpoch", "number of max Epoch", ParamValidators.gt(0))
  setDefault(maxEpoch -> 100)

  def getMaxEpoch: Int = $(maxEpoch)

  def setMaxEpoch(value: Int): this.type = set(maxEpoch, value)

  /**
   * learning rate for the optimizer in the DLEstimator.
   * Default: 0.001
   */
  val learningRate = new DoubleParam(this, "learningRate", "learningRate", ParamValidators.gt(0))

  setDefault(learningRate -> 1e-3)


  def getLearningRate: Double = $(learningRate)

  def setLearningRate(value: Double): this.type = set(learningRate, value)

    /**
   * learning rate decay.
   * Default: 0
   */
  val learningRateDecay = new DoubleParam(this, "learningRateDecay", "learningRateDecay")
  setDefault(learningRateDecay -> 0.0)

  def getLearningRateDecay: Double = $(learningRateDecay)

  def setLearningRateDecay(value: Double): this.type = set(learningRateDecay, value)

  override def transformSchema(schema : StructType): StructType = {
    validateSchema(schema)
    SchemaUtils.appendColumn(schema, $(predictionCol), ArrayType(DoubleType, false))
  }

  protected override def internalFit(
      featureAndLabel: RDD[(Seq[AnyVal], Seq[AnyVal])]): DLModel[T] = {
    val batches = toMiniBatch(featureAndLabel)

    if(!isDefined(optimMethod)) {
      set(optimMethod, new SGD[T])
    }
    val state = T("learningRate" -> $(learningRate), "learningRateDecay" -> $(learningRateDecay))
    val optimizer = Optimizer(model, batches, criterion)
      .setState(state)
      .setOptimMethod($(optimMethod).asInstanceOf[OptimMethod[T]])
      .setEndWhen(Trigger.maxEpoch($(maxEpoch)))
    val optimizedModel = optimizer.optimize()

    wrapBigDLModel(optimizedModel, featureSize)
  }

  /**
   * sub classes can extend the method and return required model for different transform tasks
   */
  protected def wrapBigDLModel(m: Module[T], featureSize: Array[Int]): DLModel[T] = {
    val dlModel = new DLModel[T](m, featureSize)
    copyValues(dlModel.setParent(this))
  }

  /**
   * Extract and reassemble data according to batchSize
   */
  private def toMiniBatch(
      featureAndLabel: RDD[(Seq[AnyVal], Seq[AnyVal])]): DistributedDataSet[MiniBatch[T]] = {

    val samples = featureAndLabel.map { case (f, l) =>
      // convert feature and label data type to the same type with model
      val feature = f.head match {
        case dd: Double => f.asInstanceOf[Seq[Double]].map(ev.fromType(_))
        case ff: Float => f.asInstanceOf[Seq[Float]].map(ev.fromType(_))
      }
      val label = l.head match {
        case dd: Double => l.asInstanceOf[Seq[Double]].map(ev.fromType(_))
        case ff: Float => l.asInstanceOf[Seq[Float]].map(ev.fromType(_))
      }
      (feature, label)
    }.map { case (feature, label) =>
      Sample(Tensor(feature.toArray, featureSize), Tensor(label.toArray, labelSize))
    }
    (DataSet.rdd(samples) -> SampleToMiniBatch(${batchSize}))
      .asInstanceOf[DistributedDataSet[MiniBatch[T]]]
  }

  override def copy(extra: ParamMap): DLEstimator[T] = {
    copyValues(new DLEstimator(model, criterion, featureSize, labelSize), extra)
  }
}


/**
 * [[DLModel]] helps embed a BigDL model into a Spark Transformer, thus Spark users can
 * conveniently merge BigDL into Spark ML pipeline.
 * [[DLModel]] supports feature data in the format of Array[Double], Array[Float],
 * org.apache.spark.mllib.linalg.{Vector, VectorUDT} for Spark 1.5, 1.6 and
 * org.apache.spark.ml.linalg.{Vector, VectorUDT} for Spark 2.0+.
 * Internally [[DLModel]] use features column as storage of the feature data, and create
 * Tensors according to the constructor parameter featureSize.
 *
 * [[DLModel]] is compatible with both spark 1.5-plus and 2.0 by extending ML Transformer.
 * @param model trainned BigDL models to use in prediction.
 * @param featureSize The size (Tensor dimensions) of the feature data. (e.g. an image may be with
 * featureSize = 28 * 28).
 */
class DLModel[@specialized(Float, Double) T: ClassTag](
    val model: Module[T],
    var featureSize : Array[Int],
    override val uid: String = "DLModel"
    )(implicit ev: TensorNumeric[T])
  extends DLTransformerBase[DLModel[T]] with DLParams with HasBatchSize {

  def setFeaturesCol(featuresColName: String): this.type = set(featuresCol, featuresColName)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setFeatureSize(value: Array[Int]): this.type = {
    this.featureSize = value
    this
  }

  def setBatchSize(value: Int): this.type = set(batchSize, value)

  def getFeatureSize: Array[Int] = this.featureSize

  /**
   * Perform a prediction on featureCol, and write result to the predictionCol.
   * @param featureData featureData in the format of Seq
   * @return output DataFrame
   */
  protected override def internalTransform(
      featureData: RDD[Seq[AnyVal]], dataset: DataFrame): DataFrame = {

    model.evaluate()
    val modelBroadCast = ModelBroadcast[T]().broadcast(featureData.sparkContext, model)
    val predictRdd = featureData.map { f =>
      // convert feature data type to the same type with model
      f.head match {
        case dd: Double => f.asInstanceOf[Seq[Double]].map(ev.fromType(_))
        case ff: Float => f.asInstanceOf[Seq[Float]].map(ev.fromType(_))
      }
    }.mapPartitions { feature =>
      val localModel = modelBroadCast.value()
      val tensorBuffer = Tensor[T](Array($(batchSize)) ++ featureSize)
      val batches = feature.grouped($(batchSize))
      batches.flatMap { batch =>
        var i = 1
        // Notice: if the last batch is smaller than the batchSize, we still continue
        // to use this tensorBuffer, but only add the meaningful parts to the result Array.
        batch.foreach { row =>
          tensorBuffer.select(1, i).copy(Tensor(Storage(row.toArray)))
          i += 1
        }
        val output = localModel.forward(tensorBuffer).toTensor[T]
        val predict = batchOutputToPrediction(output)
        predict.take(batch.length)
      }
    }

    val resultRDD = dataset.rdd.zip(predictRdd).map { case (row, predict) =>
      Row.fromSeq(row.toSeq ++ Seq(predict))
    }
    val resultSchema = transformSchema(dataset.schema)
    dataset.sqlContext.createDataFrame(resultRDD, resultSchema)
  }

  protected def batchOutputToPrediction(output: Tensor[T]): Iterable[_] = {
    val predict = output.split(1)
    predict.map(p => p.clone().storage().toArray.map(ev.toType[Double]))
  }

  override def transformSchema(schema : StructType): StructType = {
    validateSchema(schema)
    SchemaUtils.appendColumn(schema, $(predictionCol), ArrayType(DoubleType, false))
  }

  override def copy(extra: ParamMap): DLModel[T] = {
    val copied = new DLModel(model, featureSize, uid).setParent(parent)
    copyValues(copied, extra)
  }
}

// TODO, add save/load
object DLModel {


}

trait HasBatchSize extends Params {

  final val batchSize: Param[Int] = new Param[Int](this, "batchSize", "batchSize")
  setDefault(batchSize -> 1)

  final def getBatchSize: Int = $(batchSize)
}
