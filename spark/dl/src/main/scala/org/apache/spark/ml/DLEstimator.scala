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
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.MSECriterion
import com.intel.analytics.bigdl.{Criterion, DataSet, Module}
import com.intel.analytics.bigdl.optim.Optimizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

import scala.reflect.ClassTag

/**
 * A wrapper of Optimizer to support fit() in ML Pipelines as an Estimator
 */

class DLEstimator[T : ClassTag, D: ClassTag] extends MLEstimator[T]{
  def setModule(m:Module[T]) : this.type  = set("module",m)
  def getModule() : Module[T]  = getParam("module")
  def setDataSet(dataset: RDD[Sample[T]]) : this.type  = set("dataSet",dataset)
  def getDataSet() : RDD[Sample[T]]  = getParam("dataSet")
  def setCriterion(criterion: Criterion[T]):this.type  = set("criterion",criterion)
  def getCriterion():Criterion[T]  = getParam("criterion")
  def setBatchSize(batchSize:Int) : this.type  = set("batchSize",batchSize)
  def getBatchSize() : Int = getParam("batchSize")
  override def process(dataset: DataFrame): M{
      val optimizer=Optimizer(
        (
          getModule(),
          getDataSet(),
          getCriterion(),
          getBatchSize(),
        )
      )
  }
}

