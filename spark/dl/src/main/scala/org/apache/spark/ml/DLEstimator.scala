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
import org.apache.spark.sql.{DataFrame}
import org.apache.spark.ml.param.Param
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * A wrapper of Optimizer to support fit() in ML Pipelines as an Estimator
 */

class DLEstimator[T : ClassTag, D : ClassTag]
  (module : Module[T], criterion : Criterion[T], batchSize : Array[Int])
  (override val uid: String = "DLEstimator")(implicit ev: TensorNumeric[T])
  extends MLEstimator{

  def validateParameters(): Unit = {

    require(null != module,
      "DLEstimator: module for estimator must not be null")

    require(null != criterion,
      "DLEstimator: criterion must not be null")

  }


  override def process(dataFrame: DataFrame): MlTransformer = {

    this.validateParameters()

    val rdd : RDD[D] = dataFrame.rdd.map( row => row.get(0).asInstanceOf[D])

    val dataset = DataSet.rdd(rdd)


    val optimizer = Optimizer(module, dataset, criterion)

    val estimatedModule = optimizer.optimize()

    val classifier = new DLClassifier[T]()
      .setInputCol("features")
      .setOutputCol("predict")

    classifier.modelTrain -> estimatedModule

    classifier.batchShape -> batchSize

    classifier
  }

}


