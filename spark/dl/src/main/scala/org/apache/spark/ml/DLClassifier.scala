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

import scala.reflect.ClassTag

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.apache.spark.ml.param.{ParamMap, _}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}

// TODO: override transformSchema to change label and prediction type
class DLClassifier[@specialized(Float, Double) T: ClassTag] (
    override val model: Module[T],
    featureSize : Array[Int],
    override val uid: String = "DLClassifier"
  ) (implicit ev: TensorNumeric[T]) extends DLModel[T](model, featureSize) {

}

/**
 * [[DLModel]] helps embedding a BigDL model into a Spark Transformer, thus Spark users can
 * conveniently merge BigDL into Spark ML pipeline. The features column holds the storage
 * (Vector, float array or double array) of the feature data, and user should specify the
 * tensor size (dimensions) via featureSize. (e.g. an image may be with featureSize = 28 * 28).
 *
 * Internally the feature data are converted to BigDL tensors with batch acceleration, and
 * further predict with a BigDL model.
 *
 * [[DLModel]] is compatible with both spark 1.5-plus and 2.0 by extending ML Transformer.
 */
class DLModel[@specialized(Float, Double) T: ClassTag](
    val model: Module[T],
    var featureSize : Array[Int],
    override val uid: String = "DLModel"
  )(implicit ev: TensorNumeric[T]) extends DLTransformerBase with DLParams with HasBatchSize {

  def setFeaturesCol(featuresColName: String): this.type = set(featuresCol, featuresColName)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setFeatureSize(value: Array[Int]): this.type = {
    this.featureSize = value
    this
  }

  def setBatchSize(value: Int): this.type = set(batchSize, value)

  def getFeatureSize: Array[Int] = this.featureSize

  /**
   * Perform a prediction on inputCol, and write result to the outputCol.
   * @param dataset input DataFrame
   * @return output DataFrame
   */
  protected override def internalTransform(dataset: DataFrame): DataFrame = {
    val featureArrayCol = if (dataset.schema($(featuresCol)).dataType.isInstanceOf[ArrayType]) {
      $(featuresCol)
    } else {
      getFeatureArrayCol
    }

    process[T](featureSize, model, featureArrayCol, $(predictionCol), dataset)
  }

  private[DLModel] def process[@specialized(Float, Double) T: ClassTag](
      featureSize: Array[Int],
      modelTrain: Module[T],
      featuresArrayCol: String,
      predictionCol: String,
      dataset: DataFrame)(implicit ev: TensorNumeric[T]): DataFrame = {

    val model = modelTrain.evaluate()
    val modelBroadCast = ModelBroadcast[T].broadcast(dataset.sqlContext.sparkContext, model)
    val featureColIndex = dataset.schema.fieldIndex(featuresArrayCol)

    val featureType = dataset.schema(featuresArrayCol).dataType.asInstanceOf[ArrayType].elementType
    def featureToTensor = featureType match {
      case DoubleType =>
        (row: Row, index: Int) =>
          Tensor(Storage(row.getSeq[Double](featureColIndex).toArray.map(ev.fromType(_))))
      case FloatType =>
        (row: Row, index: Int) =>
          Tensor(Storage(row.getSeq[Float](featureColIndex).toArray.map(ev.fromType(_))))
    }

    val predictRdd = dataset.rdd.mapPartitions { rows =>
      val localModel = modelBroadCast.value()
      val tensorBuffer = Tensor[T](Array($(batchSize)) ++ featureSize)
      val batches = rows.grouped($(batchSize))

      val results = batches.flatMap { batch =>
        val batchResult = new Array[Row](batch.length)
        var i = 1
        // Notice: if the last batch is smaller than the batchSize, we still continue
        // to use this tensorBuffer, but only add the meaningful parts to the result Array.
        batch.foreach { row =>
          tensorBuffer.select(1, i).copy(featureToTensor(row, featureColIndex))
          i += 1
        }
        val output = localModel.forward(tensorBuffer).toTensor[T]
        val predict = if (output.dim == 2) {
          output.max(2)._2.squeeze().storage().array()
        } else if (output.dim == 1) {
          output.max(1)._2.squeeze().storage().array()
        } else {
          throw new IllegalArgumentException
        }

        i = 0
        batch.foreach { row =>
          batchResult(i) = Row.fromSeq(
            row.toSeq ++ Seq(Array[Double](ev.toType[Double](predict(i)))))
          i += 1
        }

        batchResult.toIterator
      }

      results
    }
    val predictSchema = dataset.schema.add(predictionCol, new ArrayType(DoubleType, false))
    dataset.sqlContext.createDataFrame(predictRdd, predictSchema)
  }

  override def copy(extra: ParamMap): DLModel[T] = {
    val copied = new DLModel(model, featureSize, uid).setParent(parent)
    copyValues(copied, extra).asInstanceOf[DLModel[T]]
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
