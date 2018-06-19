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
package com.intel.analytics.bigdl.parameters

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch}
import org.apache.spark.rdd.RDD
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.optim.DistriOptimizer.Cache
import com.intel.analytics.bigdl.optim.Metrics
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import org.apache.spark.broadcast.Broadcast

import scala.collection.mutable

/**
 * Process parameters trait, subclass must be independent of each other
 */
private[bigdl] trait ParameterProcessor
  extends Serializable {
  /**
   * Collect global data according to operations list, usually executed in driver
   *
   * @param models cached models
   * @param parameters [[AllReduceParameter]]
   * @param metrics metrics
   * @param state A table contained needed information
   */
  def collectGlobalData[T](models: RDD[Cache[T]],
    parameters: AllReduceParameter[T],
    metrics: Metrics,
    state: Table)(implicit ev: TensorNumeric[T]) : Unit = {}

  /**
   * Advance operations to process parameters, usually executed in worker
   *
   * @param parameters [[AllReduceParameter]]
   * @param state A table contained needed information
   */
  def processParameters[T](parameters: AllReduceParameter[T],
    modelCache: Cache[T],
    state: Table)(implicit ev: TensorNumeric[T]): Unit = {}

  /**
   * Advance operations to process parameters, usually executed in local optimer
   *
   * @param model the model to be trained
   * @param state A table contained needed information
   */
  def processParameters[T](model: Module[T],
    state: Table)(implicit ev: TensorNumeric[T]): Unit = {}
}

/**
 * Process constant clipping
 */
private[bigdl] class ConstantClippingProcessor(min: Double, max: Double)
  extends ParameterProcessor {
  override def processParameters[T](parameters: AllReduceParameter[T],
    modelCache: Cache[T],
    state: Table)(implicit ev: TensorNumeric[T]): Unit = {
    parameters.gradientPartition.clamp(min, max)
  }

  override def processParameters[T](model: Module[T],
    state: Table)(implicit ev: TensorNumeric[T]): Unit = {
    val gradients = model.getParameters()._2
    gradients.clamp(min, max)
  }
}

/**
 * Process l2 norm clipping
 */
private[bigdl] class L2NormClippingProcessor(l2NormThreshold: Double)
  extends ParameterProcessor {
  override def collectGlobalData[T](models: RDD[Cache[T]],
    parameters: AllReduceParameter[T],
    metrics: Metrics,
    state: Table)(implicit ev: TensorNumeric[T]) : Unit = {
    val numFinishedModel = state.get[Int]("numFinishedModel").get
    val parallelism = state.get[Int]("parallelism").get
    val isGradientUpdated = state.get[Boolean]("isGradientUpdated").get

    val sumSquare = models.mapPartitions(modelIter => {
      if (!isGradientUpdated) {
        val getG = System.nanoTime()
        parameters.aggregateGradientPartition(numFinishedModel)
        metrics.add("aggregrateGradientParition average executor",
          System.nanoTime() - getG)
      }
      val sum = Util.getSumsquareInParallel(parameters.gradientPartition, parallelism)
      Iterator.single(sum)
    }).reduce(_ + _)

    state("isGradientUpdated") = true
    state("l2Norm") = math.sqrt(sumSquare)
  }

  override def processParameters[T](parameters: AllReduceParameter[T],
    modelCache: Cache[T],
    state: Table)(implicit ev: TensorNumeric[T]): Unit = {
    val l2Norm = state.get[Double]("l2Norm").get
    if (l2Norm > l2NormThreshold) {
      val scale = ev.fromType[Double](l2Norm / l2NormThreshold)
      parameters.gradientPartition.div(scale)
    }
  }

  override def processParameters[T](model: Module[T],
    state: Table)(implicit ev: TensorNumeric[T]): Unit = {
    val parallelism = state.get[Int]("parallelism").get
    val gradients = model.getParameters()._2
    val l2Norm = math.sqrt(Util.getSumsquareInParallel(gradients, parallelism))

    if (l2Norm > l2NormThreshold) {
      val scale = ev.fromType[Double](l2Norm / l2NormThreshold)
      gradients.div(scale)
    }
  }
}
