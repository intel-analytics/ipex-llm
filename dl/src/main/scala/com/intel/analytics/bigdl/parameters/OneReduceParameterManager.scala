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

import com.intel.analytics.bigdl.optim.Metrics
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T, Table}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.{StorageLevel, TaskResultBlockId}
import org.apache.spark.{SparkContext, SparkEnv, TaskContext}

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect._

class OneReduceParameterManager[T: ClassTag](
  parameter: Tensor[T],
  dataset: RDD[_],
  metrics: Metrics = new Metrics()
) extends ParameterManager[T] {

  private val partitionNum = dataset.partitions.length

  private val state = T()

  private val parameterForReduce = parameter.clone()

  @transient
  private val buffers: RDD[CompressedTensor[T]] = {
    val parameterLength = parameter.nElement()
    val driverClassTag = classTag[T]
    dataset.mapPartitions(iter => {
      Iterator.single(new FP16CompressedTensor[T](new Array[Byte](parameterLength * 2),
        0, parameterLength * 2)(driverClassTag).asInstanceOf[CompressedTensor[T]])
    }).persist()
  }

  @transient
  private val sc: SparkContext = dataset.sparkContext

  @transient
  private val globalBuffer: CompressedTensor[T] = SerializerInstance.serialize(parameter)

  override def sync(parameters: RDD[Tensor[T]]): RDD[Tensor[T]] = {
    var before = System.nanoTime()
    globalBuffer.compress(parameter)
    metrics.set("driver prepare parameter", System.nanoTime() - before)
    before = System.nanoTime()
    val broadcastParameter = sc.broadcast(globalBuffer)
    metrics.set("driver serialize broadcast value", System.nanoTime() - before)

    metrics.set("worker fetch and deserialize broadcast value", 0.0, sc, partitionNum)
    metrics.set("worker extract parameter", 0.0, sc, partitionNum)
    val driverMetrics = metrics
    parameters.mapPartitions(paramIter => {
      var before = System.nanoTime()
      val localBuffer = broadcastParameter.value
      driverMetrics.add("worker fetch and deserialize broadcast value", System.nanoTime() - before)
      require(paramIter.hasNext)
      val localParameter = paramIter.next()
      require(!paramIter.hasNext)

      before = System.nanoTime()
      localBuffer.deCompress(localParameter)
      driverMetrics.add("worker extract parameter", System.nanoTime() - before)

      Iterator.single(localParameter)
    })
  }

  override def sumAndUpdate(parameters: RDD[Tensor[T]],
    update: (Tensor[T], Tensor[T], Table) => Unit): Unit = {
    var before = System.nanoTime()
    metrics.set("worker prepare parameter", 0.0, sc, partitionNum)
    metrics.set("worker serialization", 0.0, sc, partitionNum)
    val driverMetrics = metrics
    val blockids = parameters.zipPartitions(buffers)((paramStatusIter, bufferIter) => {
      require(paramStatusIter.hasNext)
      val localParam = paramStatusIter.next()
      require(!paramStatusIter.hasNext)
      val localBuffer = bufferIter.next()

      var before = System.nanoTime()
      localBuffer.compress(localParam)
      driverMetrics.add("worker prepare parameter", System.nanoTime() - before)

      before = System.nanoTime()
      val blockId = TaskResultBlockId(TaskContext.get().taskAttemptId())
      SparkEnv.get.blockManager.putBytes(
        blockId, localBuffer.bytes(), StorageLevel.MEMORY_AND_DISK_SER)
      driverMetrics.add("worker serialization", System.nanoTime() - before)

      Iterator.single(blockId)
    }).collect()
    metrics.set("rdd collect time", System.nanoTime() - before)

    val sparkEnv = SparkEnv.get
    before = System.nanoTime()
    val collectedParameters = blockids.map(blockId => Future[CompressedTensor[T]] {
      val result = SerializerInstance.serialize(sparkEnv.blockManager.getRemoteBytes(blockId).get)
      sparkEnv.blockManager.master.removeBlock(blockId)
      result
    }(Engine.getInstance())).map(Await.result(_, Duration.Inf))
    metrics.set("driver fetch parameter", System.nanoTime() - before)

    before = System.nanoTime()

    val taskSize = parameter.nElement() / Engine.coresNum()
    val extraSize = parameter.nElement() % Engine.coresNum()
    val availableTask = if (taskSize == 0) extraSize else Engine.coresNum
    (0 until availableTask).map(tid => Future {
      val start = tid * taskSize + math.min(extraSize, tid)
      val length = taskSize + (if (tid < extraSize) 1 else 0)
      collectedParameters.reduce((l, r) => l.add(r.bytes(start, length), start, length))
    }(Engine.getInstance())).map(Await.result(_, Duration.Inf))

    val reducedParameter = collectedParameters.head
    metrics.set("driver reduce parameter", System.nanoTime() - before)

    before = System.nanoTime()
    reducedParameter.deCompress(parameterForReduce)
    metrics.set("extract reduce parameter", System.nanoTime() - before)

    before = System.nanoTime()
    update(parameter, parameterForReduce, state)
    metrics.set("driver update time", System.nanoTime() - before)
  }

  override def getParameter(): Tensor[T] = parameter

  override def getState(): Table = state
}

