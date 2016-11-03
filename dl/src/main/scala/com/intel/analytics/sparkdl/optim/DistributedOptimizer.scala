/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.sparkdl.optim

import com.intel.analytics.sparkdl.nn.{Criterion, Module}
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.{File, T, Table}
import org.apache.spark.Logging

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * Train a neural network model on a distributed data set
 *
 * @param module    module to be optimized
 * @param criterion cost function
 * @param dataSet   distributed data set
 * @tparam T numeric type of model
 */
abstract class DistributedOptimizer[T](
  val module: Module[Tensor[T], Tensor[T], T],
  val criterion: Criterion[Tensor[T], T],
  val dataSet: DataSet[_, T]) extends Serializable with Logging
  with HasCrossValidation[T] with ModelPersist[T] {

  import DistributedOptimizer._

  def optimize(): Module[Tensor[T], Tensor[T], T]

  // We pre-create models on each partition of the data set
  private def init() = {
    val broadcast = dataSet.getSparkContext().broadcast((module, criterion))
    val models = dataSet.partitions().mapPartitions(_ => {
      val (broadcastModule, broadcastCriterion) = broadcast.value
      val localModule = broadcastModule.cloneModule()
      val localCriterion = broadcastCriterion.cloneCriterion()
      val (weights, grads) = localModule.getParameters()
      Iterator.single(CachedModel(localModule, localCriterion, weights, grads, T()))
    }).persist()
    models.setName("modelRDD")
    logInfo("Cache models...")
    models.count()
    logInfo("Cache models... done")
    models
  }

  val models = init()
}

object DistributedOptimizer {

  /**
   * Represent a cached module and its cost function
   *
   * @param model     module instance
   * @param criterion cost function instance
   * @param weight    a single tensor storing all parameters of the module
   * @param gradient  a single tensor storing all gradient of the parameters of the module
   * @param state     contains train state
   * @tparam T
   */
  case class CachedModel[T](model: Module[Tensor[T], Tensor[T], T],
    criterion: Criterion[Tensor[T], T], weight: Tensor[T],
    gradient: Tensor[T], state: Table)

}
