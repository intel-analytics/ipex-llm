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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.param.{ParamMap, _}
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}

import scala.reflect.ClassTag

/**
 * A general Classifier to classify the input data in inputCol, and write the results to outputCol.
 * Use setInputCol to set inputCol name, and use setOutputCol to set outputCol name.
 *
 * DLClassifier is compatible with both spark 1.5-plus and 2.0 by extending MLTransform.
 */
class DLClassifier[@specialized(Float, Double) T: ClassTag]
  (override val uid: String = "DLClassifier")(implicit ev: TensorNumeric[T]) extends MlTransformer
  with HasInputCol with HasOutputCol with DataParams[T] {

  /**
   * Set the input column name
   * @param inputColName the name of the input column
   * @return this.
   */
  def setInputCol(inputColName: String): this.type = set(inputCol, inputColName)

  /**
   * Set the output column name, should not existed in the DataFrame.
   * @param outputColName the name of the output column
   * @return this.
   */
  def setOutputCol(outputColName: String): this.type = set(outputCol, outputColName)

  def validateParameters(): Unit = {
    val params = this.extractParamMap()
    require(null != params.getOrElse(modelTrain, null),
      "DLClassifier: model for predict must not be null")
    require(null != params.getOrElse(batchShape, null),
      "DLClassifier: batchSize for predict must not be null")
    require(null != params.getOrElse(inputCol, null),
      "DLClassifier: inputCol must not be null")
    require(null != params.getOrElse(outputCol, null),
      "DLClassifier: inputCol must not be null")
  }

  /**
   * Perform a prediction on inputCol, and write result to the outputCol.
   * @param dataset input DataFrame
   * @return output DataFrame
   */
  override def process(dataset: DataFrame): DataFrame = {
    this.validateParameters()
    DLClassifier.process[T]($(batchShape), $(modelTrain), $(inputCol), $(outputCol), dataset)
  }

  override def copy(extra: ParamMap): DLClassifier[T] = {
    copyValues(new DLClassifier(uid), extra)
  }
}

object DLClassifier{
  private[DLClassifier] def process[@specialized(Float, Double) T: ClassTag](
    batchSize: Array[Int],
    modelTrain: Module[T],
    inputCol: String,
    outputCol: String,
    dataset: DataFrame)(implicit ev: TensorNumeric[T]) : DataFrame = {
    val model = modelTrain.evaluate()

    val modelBroadCast = ModelBroadcast[T].broadcast(dataset.sqlContext.sparkContext, model)

    val predictRdd = dataset.rdd.mapPartitions{ rows =>
      val localModel = modelBroadCast.value()
      val tensorBuffer = Tensor[T](batchSize)
      val batches = rows.grouped(batchSize(0))

      val results = batches.flatMap{ batch =>
        val batchResult = new Array[Row](batch.length)
        var i = 1
        // Notice: if the last batch is smaller than the batchSize(0), we still continue
        // to use this tensorBuffer, but only add the meaningful parts to the result Array.
        batch.foreach{ row =>
          tensorBuffer.select(1, i).copy(
            Tensor(Storage(row.getAs[DenseVector](inputCol).toArray.map(ev.fromType(_)))))
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
        batch.foreach{ row =>
          batchResult(i) = Row.fromSeq(row.toSeq ++ Array[Int](ev.toType[Int](predict(i))))
          i += 1
        }

        batchResult.toIterator
      }

      results
    }
    val predictSchema = dataset.schema.add(outputCol, IntegerType)
    dataset.sqlContext.createDataFrame(predictRdd, predictSchema)
  }
}

/**
 * parameters passed to DLClassifier
 * @tparam T data type
 */
trait DataParams[@specialized(Float, Double) T] extends Params {
  final val modelTrain = new Param[Module[T]](this, "module factory", "network model")
  final val batchShape = new Param[Array[Int]](this, "batch size", "batch size for input")

  /**
   * get the model
   * @return modelTrain
   */
  final def getModel: Module[T] = $(modelTrain)

  /**
   * get the batch shape
   * @return batchShape
   */
  final def getBatchSize: Array[Int] = $(batchShape)
}

