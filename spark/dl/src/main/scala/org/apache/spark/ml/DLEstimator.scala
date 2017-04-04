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
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.{Criterion, DataSet, Module}
import com.intel.analytics.bigdl.optim.Optimizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.ml.param.Param
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.types.UserDefinedType

import scala.reflect.ClassTag

/**
 * A wrapper of Optimizer to support fit() in ML Pipelines as an Estimator
 */

class DLEstimator[M <: org.apache.spark.ml.Model[M], T : ClassTag, D : ClassTag]
  (module : Module[T], criterion : Criterion[T], batchSize : Array[Int])
  (override val uid: String = "DLEstimator")
  extends MLEstimator[M]{

  final val tensorNumericPara: Param[TensorNumeric[T]] =
    new Param[TensorNumeric[T]](this, "tensorNumeric", "tensor numeric")

  final val dataSetPara: Param[DataSet[D]] =
    new Param[DataSet[D]](this, "dataSet", "data set")

  def setTensorNumric(ev: TensorNumeric[T]) : this.type = set(tensorNumericPara, ev)

  def getTensorNumric() : TensorNumeric[T] =
    this.extractParamMap().getOrElse(tensorNumericPara, null)

  def setDataSet(dataset : DataSet[D]) : this.type = set(dataSetPara, dataset)

  def getDataSet() : DataSet[D] =
    this.extractParamMap().getOrElse(dataSetPara, null)

  def validateParameters(): Unit = {

    require(null != module,
      "DLEstimator: module for estimator must not be null")

    require(null != criterion,
      "DLEstimator: criterion must not be null")

    require(null != getDataSet(),
      "DLEstimator: dataset for estimator must not be null")

    require(null != getTensorNumric(),
      "DLEstimator: TensorNumric for estimator must not be null")

  }


  override def process(dataFrame: DataFrame): M = {

    val rdd : RDD[D] = dataFrame.rdd.map( row => row.get(0).asInstanceOf[D])

    val dataset = DataSet.rdd(rdd)

    this.validateParameters()

    implicit val tensorNumric : TensorNumeric[T] = getTensorNumric()

    val optimizer = Optimizer(module, dataset, criterion)

    val estimatedModule = optimizer.optimize()

    val classifier = new DLClassifier[T]()
      .setInputCol("features")
      .setOutputCol("predict")

    classifier.modelTrain -> estimatedModule

    classifier.batchShape -> batchSize

    val model = new DLModel(classifier)("DLEstimator")

    model.asInstanceOf[M]
  }

}


