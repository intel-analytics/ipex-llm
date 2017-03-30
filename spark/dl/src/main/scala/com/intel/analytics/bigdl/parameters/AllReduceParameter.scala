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

import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.{Callable, Executors, Future, ThreadFactory}

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Engine
import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.log4j.Logger
import org.apache.spark.sparkExtension.SparkExtension
import org.apache.spark.storage.{BlockId, BlockManagerWrapper, StorageLevel}
import org.apache.spark.{SparkEnv, TaskContext}

import scala.collection.JavaConverters._
import scala.reflect._

object AllReduceParameter {
  private val syncPoolSize: Int = System.getProperty(
    "bigdl.Parameter.syncPoolSize", "4").toInt

  val logger = Logger.getLogger(getClass)
  val syncPool = Executors.newFixedThreadPool(syncPoolSize, new ThreadFactory {
    override def newThread(r: Runnable): Thread = {
      val t = Executors.defaultThreadFactory().newThread(r)
      t.setDaemon(true)
      t
    }
  })

  private val nextId = new AtomicLong(0)

  def newParameter[T: ClassTag](partitionNum: Int, size: Int): AllReduceParameter[T] = {
    new AllReduceParameter(nextId.getAndIncrement(), partitionNum, size)
  }
}

class AllReduceParameter[T: ClassTag](id: Long, partitionNum: Int,
  size: Int) extends Serializable {
  import AllReduceParameter._

  @transient private var taskSize = 0
  @transient private var extraSize = 0
  @transient private var partitionId: Int = 0

  @transient lazy val parameterBuffer: CompressedTensor[T] = readParameterBuffer()
  @transient lazy val weightPartition: Tensor[T] = readWeightParititon()
  @transient lazy val gradientPartition: Tensor[T] = readGradientPartition()

  private def readObject(in: java.io.ObjectInputStream) = {
    in.defaultReadObject()
    taskSize = size / partitionNum
    extraSize = size % partitionNum
    partitionId = TaskContext.getPartitionId()
  }

  def readParameterBuffer(): CompressedTensor[T] = {
    new FP16SplitsCompressedTensor[T](size,
      partitionNum).asInstanceOf[CompressedTensor[T]]
  }

  def readWeightParititon(): Tensor[T] = {
    val blockId = getWeightPartitionId()
    BlockManagerWrapper.getLocal(blockId).map(_.data.next()) match {
      case Some(x) =>
        x.asInstanceOf[Tensor[T]]

      case None =>
        throw new Exception("Please initialize AllReduceParameter first!!")
    }
  }

  def readGradientPartition(): Tensor[T] = {
    val blockId = getGradientPartitionId()
    BlockManagerWrapper.getLocal(blockId).map(_.data.next()) match {
      case Some(x) =>
        x.asInstanceOf[Tensor[T]]

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

    BlockManagerWrapper.removeBlock(getWeightPartitionId())
    BlockManagerWrapper.putSingle(getWeightPartitionId(),
      _weights, StorageLevel.MEMORY_AND_DISK, tellMaster = false)
    BlockManagerWrapper.removeBlock(getGradientPartitionId())
    BlockManagerWrapper.putSingle(getGradientPartitionId(),
      _gradients, StorageLevel.MEMORY_AND_DISK, tellMaster = false)
    val blockId = getWeightBlockId(partitionId)
    val fp16param = new FP16CompressedTensor[T](length)(_classTag)
    fp16param.compress(0, parameter, start, length)
    BlockManagerWrapper.putBytes(blockId, fp16param.bytes(), StorageLevel.MEMORY_ONLY_SER)
  }

  def getWeightBlockId(pid : Int): BlockId = {
    SparkExtension.getLocalBlockId(id + "weightBytes" + pid)
  }

  def getWeightPartitionId(): BlockId = {
    SparkExtension.getLocalBlockId(id + "weights" + partitionId)
  }

  def getGradientPartitionId(): BlockId = {
    SparkExtension.getLocalBlockId(id + "gradients" + partitionId)
  }

  def getGradientBlockId(pidFrom : Int, pidTo : Int): BlockId = {
    SparkExtension.getLocalBlockId(id.toString + pidTo + "gradientBytes" + pidFrom)
  }

  def getWeights(localParameter: Tensor[T]): FutureResult[Int] = {
  val bm = SparkEnv.get.blockManager
    val tasks = (0 until partitionNum).map(pid => {
      syncPool.submit(new Callable[Int] {
        override def call(): Int = {
          try {
            val blockId = getWeightBlockId(pid)
            val localBuffer = BlockManagerWrapper.byteBufferConvert(
              bm.getLocalBytes(blockId).getOrElse(bm.getRemoteBytes(blockId)
                .get))
            val start = pid * taskSize + math.min(pid, extraSize)
            val length = taskSize + (if (pid < extraSize) 1 else 0)
            require(localBuffer.array().length == length * 2)
            SerializerInstance.serialize(localBuffer).deCompress(0, localParameter, start, length)
            BlockManagerWrapper.unlock(blockId)
            pid
          } catch {
            case t : Throwable =>
              logger.error("Error: " + ExceptionUtils.getStackTrace(t))
              throw t
          }
        }
      })
    })
    new FutureResult(tasks)
  }

  def aggregrateGradientPartition(): Unit = {
    val bm = SparkEnv.get.blockManager
    require(partitionId < partitionNum)
    val params = new Array[CompressedTensor[T]](partitionNum)
    val sgThreads = (0 until partitionNum).map(pid => {
      new Callable[Int] {
        override def call(): Int = {
          try {
            val blockId = getGradientBlockId(pid, partitionId)
            val tmp = BlockManagerWrapper.byteBufferConvert(bm.getLocalBytes(blockId)
              .getOrElse(bm.getRemoteBytes(blockId).get))
            params(pid) = SerializerInstance.serialize(tmp)
            BlockManagerWrapper.unlock(blockId)
            pid
          } catch {
            case t : Throwable =>
              logger.error("Error: " + ExceptionUtils.getStackTrace(t))
              throw t
          }
        }
      }
    })
    syncPool.invokeAll(sgThreads.asJava)

    val length = taskSize + (if (partitionId < extraSize) 1 else 0)
    val poolSize = Engine.default.getPoolSize
    val innerTaskSize = length / poolSize
    val innerExtraSize = length % poolSize
    val availableTask = if (innerTaskSize == 0) innerExtraSize else poolSize
    Engine.default.invokeAndWait2((0 until availableTask).map(tid => () => {
      val innerStart = tid * innerTaskSize + math.min(innerExtraSize, tid)
      val innerLength = innerTaskSize + (if (tid < innerExtraSize) 1 else 0)
      params.reduce((l, r) => l.add(r.bytes(innerStart, innerLength), innerStart,
        innerLength))
      tid
    }))

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
    val weightsId = getWeightPartitionId()
    require(weightPartition != null)
    BlockManagerWrapper.removeBlock(blockId)
    BlockManagerWrapper.unlock(weightsId)
    BlockManagerWrapper.removeBlock(weightsId)
    BlockManagerWrapper.putSingle((weightsId),
      weightPartition, StorageLevel.MEMORY_AND_DISK, tellMaster = false)
    BlockManagerWrapper.putBytes(blockId,
      SerializerInstance.serialize(weightPartition).bytes(), StorageLevel.MEMORY_ONLY_SER)
  }
}

class FutureResult[T](private val futures: Seq[Future[T]]) {
  def waitResult(): Seq[T] = {
    futures.map(_.get())
  }
}
