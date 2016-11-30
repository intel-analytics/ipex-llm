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

package com.intel.analytics.bigdl.ps

import java.util.concurrent.{Callable, Executors, TimeUnit}

import scala.collection.JavaConverters._
import com.intel.analytics.bigdl.optim.{DropSlowModuleGradAggEpochOptimizer, Metrics}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.{StorageLevel, TaskResultBlockId}
import org.apache.spark.{Logging, SparkContext, SparkEnv, TaskContext}

import scala.collection.mutable
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.reflect._
import org.apache.log4j.Logger

object ImprovedAllReduceParameterManager {
  var task1InnerTime: Long = 0L

  private val logger = Logger.getLogger(getClass);

  private val poolSize: Int = System.getProperty(
    "com.intel.analytics.bigdl.ps.AllReduceParameterManager.poolSize",
    (Runtime.getRuntime().availableProcessors() / 2).toString()).toInt

  private val syncPoolSize: Int = System.getProperty(
    "com.intel.analytics.bigdl.ps.AllReduceParameterManager.syncPoolSize",
    4.toString()).toInt

  private val maxClusterSize = System.getProperty(
    "com.intel.analytics.bigdl.ps.AllReduceParameterManager.maxClusterSize", "10000").toInt

  val syncPool = Executors.newFixedThreadPool(syncPoolSize)
  private val context = new ExecutionContext {
    val threadPool = Executors.newFixedThreadPool(poolSize)

    def execute(runnable: Runnable) {
      threadPool.submit(runnable)
    }

    def reportFailure(t: Throwable) {}
  }

  private def getWeightBlockId(pid : Int): TaskResultBlockId = {
    TaskResultBlockId(maxClusterSize + pid)
  }

  private def getGradientBlockId(pidFrom : Int, pidTo : Int): TaskResultBlockId = {
    TaskResultBlockId(pidTo + pidFrom * maxClusterSize * 10)
  }
}

class ImprovedAllReduceParameterManager[T: ClassTag](
  parameter: Tensor[T], dataset: RDD[_], metrics: Metrics = new Metrics()
)(implicit ev: TensorNumeric[T]) extends ParameterManager[T] with Logging {

  import ImprovedAllReduceParameterManager._

  @transient
  private val sc: SparkContext = dataset.sparkContext

  @transient
  private var buffers: RDD[(Parameter[T], Tensor[T], Tensor[T], Table)] = null

  val parameterLength = parameter.nElement()

  val partitionNum: Int = dataset.partitions.length

  val idealModuleNum: Int = partitionNum * DropSlowModuleGradAggEpochOptimizer.subModuleNumber

  val taskSize = parameterLength / partitionNum
  require(taskSize != 0, "parameter length should not less than partition number")

  val extraSize = parameterLength % partitionNum

  private def init() = {
    val broadcastParameter = dataset.sparkContext.broadcast(parameter)
    val _classTag = classTag[T]
    val _ev = ev
    val _parameterLength = parameterLength
    val _partitionNum = partitionNum
    val _taskSize = taskSize
    val _extraSize = extraSize
    buffers = dataset.mapPartitions(iter => {
      val taskParameter = broadcastParameter.value
      val paramBuffer = new FP16SplitsParameter[T](_parameterLength,
        _partitionNum)(_classTag).asInstanceOf[Parameter[T]]
      val pid = TaskContext.getPartitionId()
      val start = pid * _taskSize + math.min(pid, _extraSize)
      val length = _taskSize + (if (pid < _extraSize) 1 else 0)
      val localWeight = Tensor[T](length)(_classTag, _ev).copy(taskParameter.narrow(1,
        start + 1, length))
      val localGradient = Tensor[T](length)(_classTag, _ev)
      val localState = T()
      val fp16param = new FP16Parameter[T](length)(_classTag)
      fp16param.copyFrom(0, taskParameter, start, length)
      val blockId = getWeightBlockId(pid)
      SparkEnv.get.blockManager.putBytes(blockId, fp16param.bytes(), StorageLevel.MEMORY_ONLY_SER)
      Iterator.single((paramBuffer, localWeight, localGradient, localState))
    }).setName("Parameter Manager Buffers").persist()
    buffers.count()
  }

  init()

  override def sync(parameters: RDD[Tensor[T]]): RDD[Tensor[T]] = {
    val _classTag = classTag[T]
    val _partitionNum = partitionNum
    val _taskSize = taskSize
    val _extraSize = extraSize
    val _metrics = metrics

    metrics.set("worker fetch split table", 0.0, sc, partitionNum)
    metrics.set("worker sync weight average", 0.0, sc, partitionNum)
    metrics.set("sync weight for each node", mutable.ArrayBuffer[Double](), sc)
    parameters.mapPartitions(paramIter => {
      var before = System.nanoTime()
      ImprovedAllReduceParameterManager.task1InnerTime = System.nanoTime()
      _metrics.add("worker fetch split table", System.nanoTime() - before)
      require(paramIter.hasNext)
      val localParameter = paramIter.next()
      require(!paramIter.hasNext)

      before = System.nanoTime()
      val bm = SparkEnv.get.blockManager

      val swThreads = (0 until _partitionNum).map(pid => {
        new Callable[Int] {
          override def call(): Int = {
            val blockId = getWeightBlockId(pid)
            val localBuffer = bm.getLocalBytes(blockId).getOrElse(bm.getRemoteBytes(blockId).get)
            val start = pid * _taskSize + math.min(pid, _extraSize)
            val length = _taskSize + (if (pid < _extraSize) 1 else 0)
            require(localBuffer.array().length == length * 2)
            Parameter(localBuffer)(_classTag).copyTo(0, localParameter, start, length)
            pid
          }
        }})
      syncPool.invokeAll(swThreads.asJava)

      val weightSync = System.nanoTime() - before
      _metrics.add("worker sync weight average", weightSync)
      _metrics.add("sync weight for each node", weightSync)
      DropSlowModuleGradAggEpochOptimizer.weightSyncTime = weightSync
      Iterator.single(localParameter)
    })
  }

  override def sumAndUpdate(parameters: RDD[Tensor[T]],
    update: (Tensor[T], Tensor[T], Table) => Unit): Unit = {
    val before = System.nanoTime()
    val _classTag = classTag[T]
    val _partitionNum = partitionNum
    val _taskSize = taskSize
    val _extraSize = extraSize
    val _metrics = metrics

    metrics.set("driver broadcast splits", System.nanoTime() - before)
    metrics.set("worker prepare parameter", 0.0, sc, partitionNum)
    metrics.set("worker put result", 0.0, sc, partitionNum)
    metrics.set("task1 avg time", 0.0, sc, partitionNum)
    metrics.set("task1 time from worker", mutable.ArrayBuffer[Double](), sc)

    val task1Before = System.nanoTime()
    parameters.zipPartitions(buffers)((paramStatusIter, bufferIter) => {
      require(paramStatusIter.hasNext)
      val localParameter = paramStatusIter.next()
      require(!paramStatusIter.hasNext)
      val localBuffer = bufferIter.next()._1

      var before = System.nanoTime()
      localBuffer.copyFrom(localParameter)
      _metrics.add("worker prepare parameter", System.nanoTime() - before)

      before = System.nanoTime()
      val env = SparkEnv.get
      val curPid = TaskContext.getPartitionId()
      var pid = 0
      while (pid < _partitionNum) {
        val start = pid * _taskSize + math.min(pid, _extraSize)
        val length = _taskSize + (if (pid < _extraSize) 1 else 0)
        val blockId = getGradientBlockId(curPid, pid)
        env.blockManager.removeBlock(blockId)
        env.blockManager.putBytes(
          blockId, localBuffer.bytes(start, length),
          StorageLevel.MEMORY_ONLY_SER)
        pid += 1
      }
      _metrics.add("worker put result", System.nanoTime() - before)
      _metrics.add("task1 avg time", System.nanoTime() - task1InnerTime)
      _metrics.add("task1 time from worker", System.nanoTime() -
        ImprovedAllReduceParameterManager.task1InnerTime)
      Iterator.empty
    }).count()
    metrics.set("task1 time from driver", System.nanoTime() - task1Before)

    val droppedModule = metrics.get("dropped modules")._1
    if (droppedModule >= 0.5 * DropSlowModuleGradAggEpochOptimizer.subModuleNumber*partitionNum) {
      logger.info("Warning!!! Ignore this iteration as more than half " +
        "module is dropped!! Dropped module: " + droppedModule)
    } else {
      val task2Before = System.nanoTime()
      metrics.set("gradient sync average", 0.0, sc, partitionNum)
      metrics.set("gradient sync for each node", mutable.ArrayBuffer[Double](), sc)
      metrics.set("gradient reduce", 0.0, sc, partitionNum)
      metrics.set("worker gradient extract", 0.0, sc, partitionNum)
      metrics.set("worker fetch broadcast values", 0.0, sc, partitionNum)
      metrics.set("worker update", 0.0, sc, partitionNum)
      metrics.set("worker serialize weight", 0.0, sc, partitionNum)
      metrics.set("task2 time from worker", mutable.ArrayBuffer[Double](), sc)
      val broadcastUpdate = sc.broadcast(update)

      val _remainedValue = ev.fromType[Int](idealModuleNum-droppedModule.toInt)
      buffers.mapPartitions(iter => {
        val task2InnerBefore = System.nanoTime()
        var before = System.nanoTime()
        _metrics.add("worker fetch broadcast values", System.nanoTime() - before)
        before = System.nanoTime()
        val curPid = TaskContext.getPartitionId()
        val bm = SparkEnv.get.blockManager
        val newParams = new Array[Parameter[T]](_partitionNum)
        val sgThreads = (0 until _partitionNum).map(pid => {
          new Callable[Int] {
            override def call(): Int = {
              val blockId = getGradientBlockId(pid, curPid)
              newParams(pid) = Parameter[T](bm.getLocalBytes(blockId)
                .getOrElse(bm.getRemoteBytes(blockId).getOrElse(
                  throw new IllegalArgumentException(s"Can't get the block(${blockId})")
                )))(_classTag)
              pid
            }
          }
        })
        syncPool.invokeAll(sgThreads.asJava)

        val syncGradient = System.nanoTime() - before
        _metrics.add("gradient sync average", syncGradient)
        _metrics.add("gradient sync for each node", syncGradient)

        before = System.nanoTime()
        val length = _taskSize + (if (curPid < _extraSize) 1 else 0)
        val innerTaskSize = length / poolSize
        val innerExtraSize = length % poolSize
        val availableTask = if (innerTaskSize == 0) innerExtraSize else poolSize
        (0 until availableTask).map(tid => Future {
          val innerStart = tid * innerTaskSize + math.min(innerExtraSize, tid)
          val innerLength = innerTaskSize + (if (tid < innerExtraSize) 1 else 0)
          newParams.reduce((l, r) => l.add(r.bytes(innerStart, innerLength), innerStart,
            innerLength))
        }(context)).map(Await.result(_, Duration.Inf))
        _metrics.add("gradient reduce", System.nanoTime() - before)

        before = System.nanoTime()
        val (localParameter, localParam, localGradient, localState) = iter.next()
        newParams.head.copyTo(localGradient)
        _metrics.add("worker gradient extract", System.nanoTime() - before)

        before = System.nanoTime()
        val workerUpdate = broadcastUpdate.value
        localGradient.div(_remainedValue)
        workerUpdate(localParam, localGradient, localState)
        _metrics.add("worker update", System.nanoTime() - before)

        before = System.nanoTime()

        val blockId = getWeightBlockId(curPid)
        SparkEnv.get.blockManager.removeBlock(blockId)
        SparkEnv.get.blockManager.putBytes(
          blockId,
          Parameter[T](localParam)(_classTag).bytes(), StorageLevel.MEMORY_ONLY_SER)
        _metrics.add("worker serialize weight", System.nanoTime() - before)
        _metrics.add("task2 time from worker", System.nanoTime() - task2InnerBefore)
        Iterator.empty
      }).count()

      metrics.set("task2 time from driver", System.nanoTime() - task2Before)
    }
  }

  override def getParameter(): Tensor[T] = {
    val pidToWeightSplit = buffers.mapPartitions(iter => {
      val localWeights = iter.next()._2
      val curPartitionId = TaskContext.getPartitionId()
      Iterator.single(Map(curPartitionId -> localWeights))
    }).reduce(_ ++ _)


    val partitionNum: Int = dataset.partitions.length
    val parameterLength = parameter.nElement()
    val taskSize = parameterLength / partitionNum
    require(taskSize != 0, "parameter length should not less than partition number")
    val extraSize = parameterLength % partitionNum

    (0 until partitionNum).map(pid => {
      val start = pid * taskSize + math.min(pid, extraSize)
      val length = taskSize + (if (pid < extraSize) 1 else 0)
      parameter.narrow(1, start + 1, length).copy(pidToWeightSplit(pid))
    })

    parameter
  }

  override def getState(): Table = T()
}
