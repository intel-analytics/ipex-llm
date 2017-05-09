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
import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch}
import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.optim.Optimizer
import com.intel.analytics.bigdl.tensor.{Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.ml.param.shared.HasInputCols
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.sql.{DataFrame}
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.reflect.ClassTag

/**
 * A wrapper of Optimizer to support fit() in ML Pipelines as an Estimator
 */
class DLEstimator[@specialized(Float, Double) T: ClassTag]
(override val uid: String = "DLEstimator")(implicit ev: TensorNumeric[T])
  extends MLEstimator with HasInputCols with DLDataParams[T]{

  def setInputCols(inputColName: Array[String]): this.type = set(inputCols, inputColName)

  def validateParameters(): Unit = {
    val params = this.extractParamMap()
    require(null != params.getOrElse(modelTrain, null),
      "DLEstimator: model for optimization must not be null")
    require(null != params.getOrElse(batchShape, null),
      "DLEstimator: batchShape for optimization must not be null")
    require(null != params.getOrElse(inputCols, null),
      "DLEstimator: inputCols must not be null")
    require(null != params.getOrElse(criterion, null),
      "DLEstimator: criterion must not be null")
  }

  override def process(dataFrame: DataFrame): MlTransformer = {

    validateParameters()

    val batches = toMiniBatch(dataFrame)

    val dataset = DataSet.rdd(batches)

    val optimizer = Optimizer($(modelTrain), dataset, $(criterion))

    var optimizedModule = $(modelTrain)

    optimizedModule = optimizer.optimize()

    var classifier = new DLClassifier[T]()
      .setInputCol("features")
      .setOutputCol("predict")

    val paramsTrans = ParamMap(
      classifier.modelTrain -> optimizedModule,
      classifier.batchShape -> ${batchShape})

    classifier = classifier.copy(paramsTrans)

    classifier
  }

  private def toMiniBatch(dataFrame: DataFrame) : RDD[MiniBatch[T]] = {
    val inputs = $(inputCols)
    require(inputs.length == 4, "Input columns size " +
      s" ${inputs.length} != 4 ,which stands for feature data,feature size," +
      s" label data and lable size respectively")
    val data = dataFrame.select(inputs.map(input => dataFrame(input)) : _*).rdd
    val batchs = data.map(row => {
      val featureData = row.getAs[mutable.WrappedArray[T]](0).toArray
      val featureSize = row.getAs[mutable.WrappedArray[Int]](1).toArray
      val labelData = row.getAs[mutable.WrappedArray[T]](2).toArray
      val labelSize = row.getAs[mutable.WrappedArray[Int]](3).toArray
      MiniBatch[T](Tensor(featureData, featureSize), Tensor(labelData, labelSize))
    })
    batchs
  }

  override def copy(extra: ParamMap): DLEstimator[T] = {
    copyValues(new DLEstimator(uid), extra)
  }
}

trait DLDataParams[@specialized(Float, Double) T] extends Params {

  final val modelTrain = new Param[Module[T]](this, "module factory", "network model")

  final val criterion = new Param[Criterion[T]](this, "criterion factory", "criterion for optimize")

  final val batchShape = new Param[Array[Int]](this, "batch size", "batch size for input")

  final def getModel: Module[T] = $(modelTrain)

}



