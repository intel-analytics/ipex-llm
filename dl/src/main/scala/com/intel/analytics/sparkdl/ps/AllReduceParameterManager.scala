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

package com.intel.analytics.sparkdl.ps

import com.intel.analytics.sparkdl.optim.Metrics
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.{Engine, T, Table}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.{BlockId, StorageLevel, TaskResultBlockId}
import org.apache.spark.{Logging, SparkContext, SparkEnv, TaskContext}

import scala.collection.mutable
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect._

class AllReduceParameterManager[T: ClassTag](
  parameter: Tensor[T], dataset: RDD[_], metrics: Metrics = new Metrics()
)(implicit ev: TensorNumeric[T]) extends ParameterManager[T] with Logging {

  @transient
  private val sc: SparkContext = dataset.sparkContext

  @transient
  private var splits: Array[ParamSplit] = null

  @transient
  private var buffers: RDD[(Parameter[T], Tensor[T], Tensor[T], Table)] = null

  private def init() = {
    val partitionIds = dataset.mapPartitions(iter =>
      Iterator.single(TaskContext.getPartitionId())).collect()
    require(partitionIds.distinct.length == partitionIds.length)
    val parameterLength = parameter.nElement()
    val partitionNum: Int = dataset.partitions.length
    val taskSize = parameterLength / partitionNum
    require(taskSize != 0, "parameter length should not less than partition number")
    val extraSize = parameterLength % partitionNum
    val broadcastParameter = dataset.sparkContext.broadcast(parameter)

    splits = {
      val pid2Splits = dataset.mapPartitions(iter => {
        val localParameter = broadcastParameter.value
        val pid = TaskContext.getPartitionId()
        var splitId = -1
        var i = 0
        while (i < partitionIds.length) {
          if (partitionIds(i) == pid) {
            splitId = i
          }
          i += 1
        }
        require(splitId != -1)
        val offset = splitId * taskSize + math.min(extraSize, splitId)
        val length = taskSize + (if (splitId < extraSize) 1 else 0)
        val blockIds = partitionIds.map(i => {
          val fp16param = new FP16Parameter[T](length)
          fp16param.copyFrom(0, localParameter, offset, length)
          // assume the cluster size is smaller than 10000
          val blockId = TaskResultBlockId(pid * 10000 + i)
          SparkEnv.get.blockManager.putBytes(
            blockId, fp16param.bytes(), StorageLevel.MEMORY_ONLY_SER)
          i -> blockId
        }).toMap
        Iterator.single(Map(pid -> ParamSplit(pid, offset, length, blockIds)))
      }).reduce(_ ++ _)
      partitionIds.map(pid2Splits(_))
    }

    val driverClassTag = classTag[T]
    val driverEV = ev
    val broadcastSplit = sc.broadcast(splits)
    buffers = dataset.mapPartitions(iter => {
      val localParameter = broadcastParameter.value
      val paramBuffer =
        new FP16SplitsParameter[T](parameterLength, partitionNum)(driverClassTag).
          asInstanceOf[Parameter[T]]
      val pid = TaskContext.getPartitionId()
      val localSplits = broadcastSplit.value
      var k = 0
      var localWeight: Tensor[T] = null
      var localGradient: Tensor[T] = null
      var localState: Table = null
      while (k < localSplits.length) {
        if (localSplits(k).partitionId == pid) {
          localWeight = Tensor[T](localSplits(k).length)(driverClassTag, driverEV).
            copy(localParameter.narrow(1, localSplits(k).start + 1, localSplits(k).length))
          localGradient = Tensor[T](localSplits(k).length)(driverClassTag, driverEV)
          localState = T()
        }
        k += 1
      }
      Iterator.single((paramBuffer, localWeight, localGradient, localState))
    }).setName("Parameter Manager Buffers").persist()
  }

  init()

  override def sync(parameters: RDD[Tensor[T]]): RDD[Tensor[T]] = {
    val before = System.nanoTime()
    val broadcastSplit = buffers.context.broadcast(splits)
    metrics.set("driver broadcast split table", System.nanoTime() - before)

    val partitionNum: Int = dataset.partitions.length
    metrics.set("worker fetch split table", 0.0, sc, partitionNum)
    metrics.set("worker sync weight", 0.0, sc, partitionNum)
    val driverMetrics = metrics
    val driverClassTag = classTag[T]
    parameters.mapPartitions(paramIter => {
      var before = System.nanoTime()
      val localSplits = broadcastSplit.value
      driverMetrics.add("worker fetch split table", System.nanoTime() - before)
      require(paramIter.hasNext)
      val localParameter = paramIter.next()
      require(!paramIter.hasNext)

      before = System.nanoTime()
      val bm = SparkEnv.get.blockManager
      val pid = TaskContext.getPartitionId()
      localSplits.map(s => {
        Future {
          val localBuffer = bm.getRemoteBytes(s.blockIds(pid)).get
          require(localBuffer.array().length == s.length * 2)
          Parameter(localBuffer)(driverClassTag).copyTo(0, localParameter, s.start, s.length)
        }(Engine.getInstance())
      }).map(Await.result(_, Duration.Inf))

      driverMetrics.add("worker sync weight", System.nanoTime() - before)
      Iterator.single(localParameter)
    })
  }

  override def sumAndUpdate(parameters: RDD[Tensor[T]],
    update: (Tensor[T], Tensor[T], Table) => Unit): Unit = {
    require(splits != null)
    var before = System.nanoTime()
    val driverMetrics = metrics
    val driverClassTag = classTag[T]
    val broadcastSplits = buffers.context.broadcast(splits)
    val partitionNum: Int = dataset.partitions.length
    metrics.set("driver broadcast splits", System.nanoTime() - before)

    metrics.set("worker prepare parameter", 0.0, sc, partitionNum)
    metrics.set("worker put result", 0.0, sc, partitionNum)
    val paramTable = parameters.zipPartitions(buffers)((paramStatusIter, bufferIter) => {
      require(paramStatusIter.hasNext)
      val localParameter = paramStatusIter.next()
      require(!paramStatusIter.hasNext)
      val localBuffer = bufferIter.next()._1

      var before = System.nanoTime()
      localBuffer.copyFrom(localParameter)
      driverMetrics.add("worker prepare parameter", System.nanoTime() - before)

      val localSplits = broadcastSplits.value
      before = System.nanoTime()
      val env = SparkEnv.get
      val pid = TaskContext.getPartitionId()
      var i = 0
      val results = mutable.Map[(Int, Int), BlockId]()
      while (i < localSplits.length) {
        // we assume the cluster is not larger than 10000
        val blockId = TaskResultBlockId(TaskContext.get().taskAttemptId() * 10000 + i)
        env.blockManager.putBytes(
          blockId, localBuffer.bytes(localSplits(i).start, localSplits(i).length),
          StorageLevel.MEMORY_ONLY_SER)
        results.put((i, pid), blockId)
        i += 1
      }
      driverMetrics.add("worker put result", System.nanoTime() - before)

      Iterator.single(results)
    }).reduce(_ ++= _)

    metrics.set("gradient sync", 0.0, sc, partitionNum)
    metrics.set("gradient reduce", 0.0, sc, partitionNum)
    metrics.set("worker gradient extract", 0.0, sc, partitionNum)
    metrics.set("worker fetch broadcast values", 0.0, sc, partitionNum)
    metrics.set("worker update", 0.0, sc, partitionNum)
    metrics.set("worker serialize weight", 0.0, sc, partitionNum)
    before = System.nanoTime()
    val broadcastSplitBlockids = buffers.context.broadcast(paramTable)
    val broadcastUpdate = buffers.context.broadcast(update)
    metrics.set("driver broadcast split blocks and udpate", System.nanoTime() - before)
    buffers.mapPartitions(iter => {
      var before = System.nanoTime()
      val localSplitBlockids = broadcastSplitBlockids.value
      val localSplits = broadcastSplits.value
      val localUpdate = broadcastUpdate.value
      driverMetrics.add("worker fetch broadcast values", System.nanoTime() - before)
      before = System.nanoTime()
      val pid = TaskContext.getPartitionId()
      var k = 0
      var splitId = -1
      while (k < localSplits.length) {
        if (pid == localSplits(k).partitionId) {
          splitId = k
        }
        k += 1
      }
      require(splitId != -1, s"can't find split for current partition ${pid} (${
        localSplits.map(_.partitionId).mkString(",")
      })")
      val bm = SparkEnv.get.blockManager
      val newParams = localSplits.map(s => localSplitBlockids(splitId, s.partitionId)).map(bid => {
        Future {
          val r = Parameter[T](bm.getRemoteBytes(bid).getOrElse(
            throw new IllegalArgumentException(s"Can't get the block(${bid})")
          ))(driverClassTag)
          bm.removeBlock(bid)
          r
        }(Engine.getInstance())
      }).map(Await.result(_, Duration.Inf))
      driverMetrics.add("gradient sync", System.nanoTime() - before)

      before = System.nanoTime()
      val taskSize = localSplits(splitId).length / Engine.coresNum()
      val extraSize = localSplits(splitId).length % Engine.coresNum()
      val availableTask = if (taskSize == 0) extraSize else Engine.coresNum
      (0 until availableTask).map(tid => Future {
        val start = tid * taskSize + math.min(extraSize, tid)
        val length = taskSize + (if (tid < extraSize) 1 else 0)
        newParams.reduce((l, r) => l.add(r.bytes(start, length), start, length))
      }(Engine.getInstance())).map(Await.result(_, Duration.Inf))
      driverMetrics.add("gradient reduce", System.nanoTime() - before)

      before = System.nanoTime()
      val (localParameter, localParam, localGradient, localState) = iter.next()
      newParams.head.copyTo(localGradient)
      driverMetrics.add("worker gradient extract", System.nanoTime() - before)

      before = System.nanoTime()
      val paramBlockIds = localSplits(splitId).blockIds
      localUpdate(localParam, localGradient, localState)
      driverMetrics.add("worker update", System.nanoTime() - before)

      before = System.nanoTime()
      newParams.head.copyFrom(localParam)
      paramBlockIds.map { case (pid, paramBlockId) =>
        SparkEnv.get.blockManager.removeBlock(paramBlockId)
        SparkEnv.get.blockManager.putBytes(
          paramBlockId, newParams.head.bytes(), StorageLevel.MEMORY_ONLY_SER)
      }
      driverMetrics.add("worker serialize weight", System.nanoTime() - before)
      Iterator.single(1)
    }).count()
  }

  override def getParameter(): Tensor[T] = {
    val pidToWeightSplit = buffers.mapPartitions(iter => {
      val localWeights = iter.next()._2
      val curPartitionId = TaskContext.getPartitionId()
      Iterator.single(Map(curPartitionId -> localWeights))
    }).reduce(_ ++ _)

    splits.map(split => {
      parameter.narrow(1, split.start + 1, split.length).copy(pidToWeightSplit(split.partitionId))
    })

    parameter
  }

  override def getState(): Table = T()
}

case class ParamSplit(partitionId: Int, start: Int, length: Int, blockIds: Map[Int, BlockId])
