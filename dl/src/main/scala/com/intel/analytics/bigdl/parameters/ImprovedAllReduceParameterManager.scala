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

package com.intel.analytics.bigdl.parameters

import java.util.concurrent.{Callable, Executors, TimeUnit}

import com.intel.analytics.bigdl.optim.DistributedOptimizer.CachedModel

import scala.collection.JavaConverters._
import com.intel.analytics.bigdl.optim.{DropSlowModuleGradAggEpochOptimizer, Metrics}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.{StorageLevel, TaskResultBlockId}
import org.apache.spark.{SparkContext, SparkEnv, TaskContext}

import scala.collection.mutable
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.reflect._
import org.apache.log4j.Logger

object ImprovedAllReduceParameterManager {
  var task1InnerTime: Long = 0L

  private val logger = Logger.getLogger(getClass)
}

class ImprovedAllReduceParameterManager[T: ClassTag](
  parameter: Tensor[T], dataset: RDD[_], metrics: Metrics = new Metrics()
)(implicit ev: TensorNumeric[T]) extends ParameterManager[T]{

  override def sync(parameters: RDD[Tensor[T]]): RDD[Tensor[T]] = {
    parameters.mapPartitions(paramIter => {
      Iterator.single(paramIter.next())
    })
  }

  override def sumAndUpdate(parameters: RDD[Tensor[T]],
    update: (Tensor[T], Tensor[T], Table) => Unit): Unit = {
  }

  override def getParameter(): Tensor[T] = {
    null
  }

  override def getState(): Table = T()
}
