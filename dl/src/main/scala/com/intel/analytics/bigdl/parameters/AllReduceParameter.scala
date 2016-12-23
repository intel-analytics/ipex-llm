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

import java.util.concurrent.{Callable, Executors, Future}

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}
import org.apache.spark.sparkExtension.SparkExtension
import org.apache.spark.{SparkEnv, TaskContext}
import org.apache.spark.storage.{BlockManagerWrapper, StorageLevel, TaskResultBlockId}

import scala.collection.JavaConverters._
import scala.reflect._

object AllReduceParameter {
  private val syncPoolSize: Int = System.getProperty(
    "bigdl.Parameter.syncPoolSize", "4").toInt

  private val computePoolSize: Int = System.getProperty(
    "bigdl.Parameter.computePoolSize",
    (Runtime.getRuntime().availableProcessors() / 2).toString()).toInt

  private val maxClusterSize = System.getProperty(
    "bigdl.Parameter.maxClusterSize", "10000").toInt

  val syncPool = Executors.newFixedThreadPool(syncPoolSize)
  // TODO: Replace it with Engine.threadpool
  val computePool = Executors.newFixedThreadPool(computePoolSize)
}

class AllReduceParameter[T: ClassTag](id: Int) extends Serializable {
  import AllReduceParameter._
  
  var partitionNum = 0
  var tlength = 0

  @transient lazy private val taskSize = tlength / partitionNum
  @transient lazy private val extraSize = tlength % partitionNum
  @transient lazy private val partitionId: Int = TaskContext.getPartitionId()
  
  @transient lazy val parameterBuffer: CompressedTensor[T] = readParameterBuffer()
  @transient lazy val weightPartition: Tensor[T] = readWeights()
  @transient lazy val gradientPartition: Tensor[T] = readGradients()
  @transient lazy val state = readState()
  
  def readParameterBuffer(): CompressedTensor[T] = {
    new FP16SplitsCompressedTensor[T](tlength,
      partitionNum).asInstanceOf[CompressedTensor[T]]
  }

  def readWeights(): Tensor[T] = {
    val blockId = SparkExtension.getLocalBlockId(id + "weights" + partitionId)
    BlockManagerWrapper.getLocal(blockId).map(_.data.next()) match {
      case Some(x) =>
        x.asInstanceOf[Tensor[T]]

      case None =>
        throw new Exception("Please initialize AllReduceParameter first!!")
    }
  }

  def readGradients(): Tensor[T] = {
    val blockId = SparkExtension.getLocalBlockId(id + "gradients" + partitionId)
    BlockManagerWrapper.getLocal(blockId).map(_.data.next()) match {
      case Some(x) =>
        x.asInstanceOf[Tensor[T]]

      case None =>
        throw new Exception("Please initialize AllReduceParameter first!!")
    }
  }

  def readState(): Table = {
    val blockId = SparkExtension.getLocalBlockId(id + "state" + partitionId)
    BlockManagerWrapper.getLocal(blockId).map(_.data.next()) match {
      case Some(x) =>
        x.asInstanceOf[Table]

      case None =>
        throw new Exception("Please initialize AllReduceParameter first!!")
    }
  }

  def init(parameter: Tensor[T])(implicit ev: TensorNumeric[T]): Unit = {
    val _classTag = classTag[T]
    val start = partitionId * taskSize + math.min(partitionId, extraSize)
    val length = taskSize + (if (partitionId < extraSize) 1 else 0)

    val _weights = Tensor[T](length)(_classTag, ev).copy(parameter.narrow(1,
      start + 1, length))
    val _gradients = Tensor[T](length)(_classTag, ev)

    SparkEnv.get.blockManager.putSingle(
      SparkExtension.getLocalBlockId(id + "weights" + partitionId),
      _weights, StorageLevel.MEMORY_AND_DISK, tellMaster = false)
    SparkEnv.get.blockManager.putSingle(
      SparkExtension.getLocalBlockId(id + "gradients" + partitionId),
      _gradients, StorageLevel.MEMORY_AND_DISK, tellMaster = false)
    SparkEnv.get.blockManager.putSingle(
      SparkExtension.getLocalBlockId(id + "state" + partitionId),
      T(), StorageLevel.MEMORY_AND_DISK, tellMaster = false)
    val blockId = getWeightBlockId(partitionId)
    val fp16param = new FP16CompressedTensor[T](length)(_classTag)
    fp16param.compress(0, parameter, start, length)
    BlockManagerWrapper.putBytes(blockId, fp16param.bytes(), StorageLevel.MEMORY_ONLY_SER)
  }

  def getWeightBlockId(pid : Int): TaskResultBlockId = {
    TaskResultBlockId(maxClusterSize + pid)
  }

  def getGradientBlockId(pidFrom : Int, pidTo : Int): TaskResultBlockId = {
    TaskResultBlockId(pidTo + pidFrom * maxClusterSize * 10)
  }

  def getWeights(localParameter: Tensor[T]): FutureResult[Int] = {
  val bm = SparkEnv.get.blockManager
    val tasks = (0 until partitionNum).map(pid => {
      syncPool.submit(new Callable[Int] {
        override def call(): Int = {
          val blockId = getWeightBlockId(pid)
          val localBuffer = BlockManagerWrapper.byteBufferConvert(
            bm.getLocalBytes(blockId).getOrElse(bm.getRemoteBytes(blockId).get))
          val start = pid * taskSize + math.min(pid, extraSize)
          val length = taskSize + (if (pid < extraSize) 1 else 0)
          require(localBuffer.array().length == length * 2)
          SerializerInstance.serialize(localBuffer).deCompress(0, localParameter, start, length)
          BlockManagerWrapper.unlock(blockId)
          pid
        }
      })
    })
    new FutureResult(tasks)
  }

  def aggregrateGradientParition(): Unit = {
    val bm = SparkEnv.get.blockManager
    require(partitionId < partitionNum)
    val params = new Array[CompressedTensor[T]](partitionNum)
    val sgThreads = (0 until partitionNum).map(pid => {
      new Callable[Int] {
        override def call(): Int = {
          val blockId = getGradientBlockId(pid, partitionId)
          val tmp = BlockManagerWrapper.byteBufferConvert(bm.getLocalBytes(blockId)
            .getOrElse(bm.getRemoteBytes(blockId).getOrElse(
              throw new IllegalArgumentException(s"Can't get the block(${blockId})")
            )))
          params(pid) = SerializerInstance.serialize(tmp)
          BlockManagerWrapper.unlock(blockId)
          pid
        }
      }
    })
    syncPool.invokeAll(sgThreads.asJava)

    val length = taskSize + (if (partitionId < extraSize) 1 else 0)
    val innerTaskSize = length / computePoolSize
    val innerExtraSize = length % computePoolSize
    val availableTask = if (innerTaskSize == 0) innerExtraSize else computePoolSize
    val tasks = (0 until availableTask).map(tid => new Callable[Int] {
        override def call(): Int = {
          val innerStart = tid * innerTaskSize + math.min(innerExtraSize, tid)
          val innerLength = innerTaskSize + (if (tid < innerExtraSize) 1 else 0)
          params.reduce((l, r) => l.add(r.bytes(innerStart, innerLength), innerStart,
            innerLength))
          tid
        }
    })
    computePool.invokeAll(tasks.asJava)
    params.head.deCompress(gradientPartition)
  }

  def putGradients(parameter: Tensor[T]): Unit = {
    var pid = 0
    val bm = SparkEnv.get.blockManager
    require(parameterBuffer != null)
    parameterBuffer.compress(parameter)
    while (pid < partitionNum) {
      val start = pid * taskSize + math.min(pid, extraSize)
      val length = taskSize + (if (pid < extraSize) 1 else 0)
      val blockId = getGradientBlockId(partitionId, pid)
      BlockManagerWrapper.putBytes(
        blockId, parameterBuffer.bytes(start, length),
        StorageLevel.MEMORY_ONLY_SER)
      pid += 1
    }
  }

  def sendWeightPartition(): Unit = {
    val blockId = getWeightBlockId(partitionId)
    val weightsId = SparkExtension.getLocalBlockId(id + "weights" + partitionId)
    val stateId = SparkExtension.getLocalBlockId(id + "state" + partitionId)
    require(weightPartition != null)
    require(state != null)
    val bm = SparkEnv.get.blockManager
    bm.removeBlock(blockId)
    BlockManagerWrapper.unlock(weightsId)
    bm.removeBlock(weightsId)
    bm.putSingle((weightsId),
      weightPartition, StorageLevel.MEMORY_AND_DISK, tellMaster = false)
    BlockManagerWrapper.unlock(stateId)
    bm.removeBlock(stateId)
    bm.putSingle((stateId),
      state, StorageLevel.MEMORY_AND_DISK, tellMaster = false)
    BlockManagerWrapper.putBytes(blockId,
      SerializerInstance.serialize(weightPartition).bytes(), StorageLevel.MEMORY_ONLY_SER)
  }
}

class FutureResult[T](private val futures: Seq[Future[T]]) {
  def waits(): Seq[T] = {
    futures.map(_.get())
  }
}