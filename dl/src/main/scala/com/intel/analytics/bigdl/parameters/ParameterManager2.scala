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

import java.util.concurrent.atomic.{AtomicInteger}
import java.util.concurrent.{Callable, Executors, ThreadFactory}

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, T, Table}
import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.log4j.Logger

import org.apache.spark.sparkExtension.{ParameterManagerMaster, SparkExtension}
import org.apache.spark.{SparkEnv}
import org.apache.spark.storage.{BlockId, BlockManagerWrapper, StorageLevel}

import scala.collection.JavaConverters._
import scala.collection.mutable.HashMap
import scala.reflect._

object ParameterManager2 {
  val logger = Logger.getLogger(getClass)

  private val nextId = new AtomicInteger(0)

  private val pm = new HashMap[Int, ParameterManager2]()

  private var executorIdMap: HashMap[String, Int] = null

  def setExecutorMap(map: HashMap[String, Int]): Unit = {
    executorIdMap = map
  }

  def get(executorId: String): ParameterManager2 = {
    val id = executorIdMap(executorId)
    if (pm.contains(id)) pm(id)
    else null
  }

  def createParameterManager[T: ClassTag](executorId: Int, executorNum: Int, partitionNum: Int,
    size: Int, isDriver: Boolean): ParameterManager2 = {
    val id = nextId.getAndIncrement()
    val conf = SparkEnv.get.conf
    val master = ParameterManagerMaster.createEnv(conf, isDriver)
    val p = new ParameterManager2(id, executorId, executorNum, partitionNum, size, master)
    pm.put(executorId, p)
    p
  }
}

class ParameterManager2(val id: Int, val executorId: Int,
  executorNum: Int, partitionNum: Int, size: Int, master: ParameterManagerMaster) {
  import ParameterManager2._

  private val syncPoolSize: Int = System.getProperty(
    "bigdl.Parameter.syncPoolSize", "4").toInt

  val syncPool = Executors.newFixedThreadPool(syncPoolSize, new ThreadFactory {
    override def newThread(r: Runnable): Thread = {
      val t = Executors.defaultThreadFactory().newThread(r)
      t.setDaemon(true)
      t
    }
  })

  var job1Start: Boolean = false
  var initFinished = false

  val taskSize = size / executorNum
  val extraSize = size % executorNum

  def init[T: ClassTag](parameter: Tensor[T], state: Table)
    (implicit ev: TensorNumeric[T]): Unit = {
    val _classTag = classTag[T]
    val start = executorId * taskSize + math.min(executorId, extraSize)
    val length = taskSize + (if (executorId < extraSize) 1 else 0)
    val _weightsExecutor = Tensor[T](length)(_classTag, ev).copy(parameter.narrow(1,
      start + 1, length))
    BlockManagerWrapper.putSingle(getWeightExecutorId(),
      _weightsExecutor, StorageLevel.MEMORY_AND_DISK, tellMaster = false)

    BlockManagerWrapper.putSingle(getWeightId(),
      parameter, StorageLevel.MEMORY_AND_DISK, tellMaster = false)

    BlockManagerWrapper.putSingle(getStateId(),
      state, StorageLevel.MEMORY_AND_DISK, tellMaster = false)

    val _gradientsExecutor = Tensor[T](length)(_classTag, ev)
    BlockManagerWrapper.putSingle(getGradientExecutorId(),
      _gradientsExecutor, StorageLevel.MEMORY_AND_DISK, tellMaster = false)

    val blockId = getWeightBlockId(executorId)
    val fp16param = new FP16CompressedTensor[T](length)(_classTag)
    fp16param.compress(0, parameter, start, length)
    BlockManagerWrapper.putBytes(blockId, fp16param.bytes(), StorageLevel.MEMORY_ONLY_SER)
  }

  def aggregateLocalGradient[T: ClassTag]() : Tensor[T] = {
    val blockIds = master.getBlockId(executorId)
    val gradientBuffer = new Array[Tensor[T]](blockIds.size)
    Engine.compute.invokeAndWait((0 until blockIds.size).map(tid => () => {
      gradientBuffer(tid) = getLocalParameter(blockIds(tid))
    }))

    val poolSize = Engine.compute.getPoolSize
    val innerTaskSize = size / poolSize
    val innerExtraSize = size % poolSize
    val availableTask = if (innerTaskSize == 0) innerExtraSize else poolSize

    Engine.compute.invokeAndWait((0 until availableTask).map(tid => () => {
      val innerStart = tid * innerTaskSize + math.min(tid, innerExtraSize)
      val innerLength = innerTaskSize + (if (tid < innerExtraSize) 1 else 0)
      var i = 1
      while (i < gradientBuffer.length) {
        gradientBuffer(0).narrow(1, innerStart + 1, innerLength)
          .add(gradientBuffer(i).narrow(1, innerStart + 1, innerLength))
        i += 1
      }
    }))
    master.clearBlockId(executorId)
    gradientBuffer(0)
  }

  def putGradients[T: ClassTag](parameter: Tensor[T]): Unit = {
    val _classTag = classTag[T]
    var pid = 0
    while (pid < executorNum) {
      val start = pid * taskSize + math.min(pid, extraSize)
      val length = taskSize + (if (pid < extraSize) 1 else 0)
      val blockId = getGradientBlockId(executorId, pid)
      val fp16param = new FP16CompressedTensor[T](length)(_classTag)
      fp16param.compress(0, parameter, start, length)
      BlockManagerWrapper.putBytes(blockId, fp16param.bytes(), StorageLevel.MEMORY_ONLY_SER)
      pid += 1
    }
  }

  def aggregrateGradientParition[T: ClassTag](params: Array[CompressedTensor[T]]): Unit = {
    val bm = SparkEnv.get.blockManager
    val sgThreads = (0 until executorNum).map(pid => {
      new Callable[Int] {
        override def call(): Int = {
          try {
            val blockId = getGradientBlockId(pid, executorId)
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

    val length = taskSize + (if (executorId < extraSize) 1 else 0)
    val poolSize = Engine.compute.getPoolSize
    val innerTaskSize = length / poolSize
    val innerExtraSize = length % poolSize
    val availableTask = if (innerTaskSize == 0) innerExtraSize else poolSize

    Engine.compute.invokeAndWait((0 until availableTask).map(tid => () => {
      val innerStart = tid * innerTaskSize + math.min(innerExtraSize, tid)
      val innerLength = innerTaskSize + (if (tid < innerExtraSize) 1 else 0)
      params.reduce((l, r) => l.add(r.bytes(innerStart, innerLength), innerStart,
        innerLength))
        }))
    val gradientExecutorId = getGradientExecutorId()
    val gradientExecutor = getLocalParameter[T](gradientExecutorId)
    params.head.deCompress(gradientExecutor)
  }

  def syncWeights[T: ClassTag](localParameter: Tensor[T]): Unit = {
    val bm = SparkEnv.get.blockManager
    val tasks = (0 until executorNum).map(pid => {
      new Callable[Int] {
        override def call(): Int = {
          try {
            val blockId = getWeightBlockId(pid)
            val localBuffer = BlockManagerWrapper.byteBufferConvert(
              bm.getLocalBytes(blockId).getOrElse(bm.getRemoteBytes(blockId)
                .get))
            val start = pid * taskSize + math.min(pid, extraSize)
            val length = taskSize + (if (pid < extraSize) 1 else 0)
            require(localBuffer.array().length == length * 2)
            SerializerInstance.serialize[T](localBuffer)
              .deCompress(0, localParameter, start, length)
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
    syncPool.invokeAll(tasks.asJava)
  }

  def sendWeightExecutor[T: ClassTag]() : Unit = {
    val weightExecutorId = getWeightExecutorId()
    val weightExecutor = getLocalParameter(weightExecutorId)
    val blockId = getWeightBlockId(executorId)
    BlockManagerWrapper.removeBlock(blockId)
    BlockManagerWrapper.putBytes(blockId,
      SerializerInstance.serialize(weightExecutor).bytes(), StorageLevel.MEMORY_ONLY_SER)
  }

  def getLocalParameter[T: ClassTag](blockId: BlockId): Tensor[T] = {
    BlockManagerWrapper.getLocal(blockId).map(_.data.next()) match {
      case Some(x) =>
        x.asInstanceOf[Tensor[T]]

      case None =>
        throw new Exception("Please initialize AllReduceParameter first!!")
    }
  }

  def getState(): Table = {
    BlockManagerWrapper.getLocal(getStateId()).map(_.data.next()) match {
      case Some(x) =>
        x.asInstanceOf[Table]

      case None =>
        throw new Exception("Please initialize AllReduceParameter first!!")
    }
  }

  def sendGradientPartition[T: ClassTag](gradient: Tensor[T], pid: Int): Unit = {
    val gradientsId = getGradientPartitionId(pid)

    BlockManagerWrapper.getLocal(gradientsId).map(_.data.next()) match {
      case Some(x) =>
        val t = x.asInstanceOf[Tensor[T]]
        t.copy(gradient)

      case None =>
        BlockManagerWrapper.putSingle(gradientsId, gradient,
          StorageLevel.MEMORY_AND_DISK, tellMaster = false)
    }
    master.updateBlockId(executorId, gradientsId)
  }

  def getGradientBlockId(pidFrom : Int, pidTo : Int): BlockId = {
    SparkExtension.getLocalBlockId("pm" + pidTo + "gBytes" + pidFrom)
  }

  def getWeightBlockId(pid : Int): BlockId = {
    SparkExtension.getLocalBlockId("pm_wBytes" + pid)
  }

  def getWeightExecutorId(): BlockId = {
    SparkExtension.getLocalBlockId("pm_w" + executorId)
  }

  def getWeightId(): BlockId = {
    SparkExtension.getLocalBlockId("weight" + executorId)
  }

  def getStateId(): BlockId = {
    SparkExtension.getLocalBlockId("pm_s" + executorId)
  }

  def getGradientPartitionId(pid: Int): BlockId = {
    SparkExtension.getLocalBlockId("pm_g_p" + pid)
  }

  def getGradientExecutorId(): BlockId = {
    SparkExtension.getLocalBlockId("pm_g" + executorId)
  }
}

