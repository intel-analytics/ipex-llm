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

import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.{Callable, Executors, Future, ThreadFactory}

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, T, Table}
import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.log4j.Logger
import org.apache.spark.sparkExtension.{ParameterManagerMaster, SparkExtension}
import org.apache.spark.SparkEnv
import org.apache.spark.storage.{BlockId, BlockManagerWrapper, StorageLevel}

import scala.collection.JavaConverters._
import scala.collection.mutable.HashMap
import scala.reflect._

private[bigdl] object AllReduceParameterManager {
  val logger = Logger.getLogger(getClass)

  private val nextId = new AtomicInteger(0)

  private val pm = new HashMap[String, AllReduceParameterManager]()

  private var executorIdMap: HashMap[String, Int] = null

  def setExecutorMap(map: HashMap[String, Int]): Unit = {
    executorIdMap = map
  }

  def get(id: Int, executorId: String): Option[AllReduceParameterManager] = {
    val eid = id + "exe" + executorIdMap(executorId)
    pm.get(eid)
  }

  def createParameterManager[T: ClassTag](executorId: Int, executorNum: Int, partitionNum: Int,
    size: Int, port: Option[Int] = None, pid: Option[Int] = None): AllReduceParameterManager = {
    val id = pid.getOrElse(nextId.getAndIncrement())
    val conf = SparkEnv.get.conf
    val master = ParameterManagerMaster.createEnv(conf, port)
    val p = new AllReduceParameterManager(id, executorId, executorNum, partitionNum, size, master)
    pm.put(id + "exe" + executorId, p)
    p
  }
}

/**
 * Represent parameters stored on the block manager. In distributed optimization, we put parameters
 * on block manager of spark. Each worker syncs parameters through the block manager. Block manager
 * here serves as a parameter server.
 *
 * A Tensor is sliced into `executorNum` chunks and each chunk is assigned to a particular node
 * (Spark executor). Likewise, gradients for each chunk are also assigned and stored on separate
 * nodes. In this way, gradient aggregation and parameter updates can be performed independently for
 * each chunk on separate nodes.
 *
 * @param id distinguish from other parameters
 * @param executorId executorId of the node
 * @param executorNum how many executors will use this parameter
 * @param partitionNum partition numbers
 * @param size size of the parameter (1D vector)
 * @param master ParameterManagerMaster endpoint
 */
private[bigdl] class AllReduceParameterManager(val id: Int, val executorId: Int, executorNum: Int,
  partitionNum: Int, size: Int, val master: ParameterManagerMaster) {
  import AllReduceParameterManager._

  private val syncPoolSize: Int = System.getProperty(
    "bigdl.Parameter.syncPoolSize", "4").toInt

  private val syncPool = Executors.newFixedThreadPool(syncPoolSize, new ThreadFactory {
    override def newThread(r: Runnable): Thread = {
      val t = Executors.defaultThreadFactory().newThread(r)
      t.setDaemon(true)
      t
    }
  })

  // Used to make sure in one iteration, in each executor, get weight is only executed once
  var syncWeight: Boolean = false
  // Used to make sure in each executor, init is only executed once
  var initFinished = false

  private val taskSize = size / executorNum
  private val extraSize = size % executorNum

  /**
   * This method should be called on each RDD partition before parameter synchronization begins.
   * An empty gradient tensor is placed in the block manager that can be used to store gradients.
   * A 1 / executorNum fraction of the `parameter` tensor is copied to the block manager as a
   * compressed tensor.
   *
   * @param parameter A tensor representing the initial underlying weights of this
   *                  `AllReduceParameterManager`
   */
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

    val _gradientsExecutor = Tensor[T](length)(_classTag, ev)
    BlockManagerWrapper.putSingle(getGradientExecutorId(),
      _gradientsExecutor, StorageLevel.MEMORY_AND_DISK, tellMaster = false)

    val blockId = getWeightBlockId(executorId)
    val fp16param = new FP16CompressedTensor[T](length)(_classTag)
    fp16param.compress(0, parameter, start, length)
    BlockManagerWrapper.putBytes(blockId, fp16param.bytes(), StorageLevel.MEMORY_ONLY_SER)
  }

  /** Aggregate gradients hosted in one executor */
  def aggregateLocalGradient[T: ClassTag]() : Tensor[T] = {
    val blockIds = master.getBlockId(executorId)
    val gradientBuffer = new Array[Tensor[T]](blockIds.size)
    Engine.default.invokeAndWait((0 until blockIds.size).map(tid => () => {
      gradientBuffer(tid) = getLocalParameter(blockIds(tid))
    }))

    val poolSize = Engine.default.getPoolSize
    val innerTaskSize = size / poolSize
    val innerExtraSize = size % poolSize
    val availableTask = if (innerTaskSize == 0) innerExtraSize else poolSize

    Engine.default.invokeAndWait((0 until availableTask).map(tid => () => {
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

  /**
   * Slice aggregated gradients into chunks, and mark each chunk to be sent
   * to the appropriate parameter node, and put it in the block manager.
   *
   * @param parameter A Tensor that contains gradients computed on the entire model on a single
   *                  node.
   */
  def putGradientsExecutor[T: ClassTag](parameter: Tensor[T]): Unit = {
    val _classTag = classTag[T]
    var pid = 0
    while (pid < executorNum) {
      val start = pid * taskSize + math.min(pid, extraSize)
      val length = taskSize + (if (pid < extraSize) 1 else 0)
      val blockId = getGradientBlockId(executorId, pid)
      val block = BlockManagerWrapper.getLocalBytes(blockId)
      if (block.isDefined) {
        val fp16param = new FP16CompressedTensor[T](block.get)(_classTag)
        fp16param.compress(0, parameter, start, length)
      } else {
        val fp16param = new FP16CompressedTensor[T](length)(_classTag)
        fp16param.compress(0, parameter, start, length)
        BlockManagerWrapper.putBytes(blockId, fp16param.bytes(), StorageLevel.MEMORY_ONLY_SER)
      }

      pid += 1
    }
  }

  /**
   * Retrieve gradients for the slice of the model that this node is responsible for from all the
   * other nodes. A new thread is created for each separate node. The gradients are then summed
   * and then stored in decompressed form in blockmanager.
   */
  def aggregrateGradientParition[T: ClassTag](): Unit = {
    val params = new Array[CompressedTensor[T]](executorNum)
    val sgThreads = (0 until executorNum).map(pid => {
      new Callable[Int] {
        override def call(): Int = {
          try {
            val blockId = getGradientBlockId(pid, executorId)
            val tmp = BlockManagerWrapper.getLocalOrRemoteBytes(blockId).get
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
    val poolSize = Engine.default.getPoolSize
    val innerTaskSize = length / poolSize
    val innerExtraSize = length % poolSize
    val availableTask = if (innerTaskSize == 0) innerExtraSize else poolSize

    Engine.default.invokeAndWait((0 until availableTask).map(tid => () => {
      val innerStart = tid * innerTaskSize + math.min(innerExtraSize, tid)
      val innerLength = innerTaskSize + (if (tid < innerExtraSize) 1 else 0)
      params.reduce((l, r) => l.add(r.bytes(innerStart, innerLength), innerStart,
        innerLength))
        }))
    val gradientExecutorId = getGradientExecutorId()
    val gradientExecutor = getLocalParameter[T](gradientExecutorId)
    params.head.deCompress(gradientExecutor)
  }

  /**
   * Use a fixed thread pool to launch a thread for each node of the weights. Each thread
   * requests a node of the weights from the Spark block manager.
   *
   * @param localParameter The Tensor that will hold the retrieved weights.
   * @return A [[FutureResult]] which contains a [[Future]] for each thread.
   */
  def getWeights[T: ClassTag](localParameter: Tensor[T]): FutureResult[Int] = {
    val tasks = (0 until executorNum).map { pid =>
      syncPool.submit {
        new Callable[Int] {
          override def call(): Int = {
            try {
              val blockId = getWeightBlockId(pid)
              val localBuffer = BlockManagerWrapper.getLocalOrRemoteBytes(blockId).getOrElse {
                throw new RuntimeException(s"Didn't find weight block $blockId in the block " +
                  s"manager. Did you initialize this AllReduceParameter on every executor?")
              }
              val start = pid * taskSize + math.min(pid, extraSize)
              val length = taskSize + (if (pid < extraSize) 1 else 0)
              require(localBuffer.array().length == length * 2)
              SerializerInstance.serialize(localBuffer).deCompress(0, localParameter, start, length)
              BlockManagerWrapper.unlock(blockId)
              pid
            } catch {
              case t: Throwable =>
                logger.error("Error: " + ExceptionUtils.getStackTrace(t))
                throw t
            }
          }
        }
      }
    }
    new FutureResult(tasks)
  }

  /**
   * Put the portion of the weights that this node is responsible for to the block manager.
   * Weights are placed locally, then pulled when needed by other nodes.
   */
  def sendWeightPartition[T: ClassTag]() : Unit = {
    val weightExecutorId = getWeightExecutorId()
    val weightExecutor = getLocalParameter(weightExecutorId)
    val blockId = getWeightBlockId(executorId)
    BlockManagerWrapper.removeBlock(blockId)
    val data = SerializerInstance.serialize(weightExecutor)
    val block = BlockManagerWrapper.getLocalBytes(blockId)
    if (block.isDefined) {
      block.get.put(data.bytes())
    } else {
        BlockManagerWrapper.putBytes(blockId,
          data.bytes(), StorageLevel.MEMORY_ONLY_SER)
    }
  }

  /** Get a block from local blockmanager */
  def getLocalParameter[T: ClassTag](blockId: BlockId): Tensor[T] = {
    BlockManagerWrapper.getLocal(blockId)
      .map(_.data.next().asInstanceOf[Tensor[T]])
      .getOrElse(throw new IllegalStateException("Please initialize AllReduceParameter first!"))
  }

  /** Put a complete gradient in local blockmanager, waiting for another job to aggregate */
  def sendCompleteGradients[T: ClassTag](gradient: Tensor[T], pid: Int): Unit = {
    val gradientsId = getGradientPartitionId(pid)

    val gradients = BlockManagerWrapper.getLocal(gradientsId)
      .map(_.data.next().asInstanceOf[Tensor[T]])
      if (gradients.isDefined) {
        gradients.get.copy(gradient)
      } else {
        BlockManagerWrapper.putSingle(gradientsId, gradient, StorageLevel.MEMORY_AND_DISK,
          tellMaster = false)
      }
    master.updateBlockId(executorId, gradientsId)
  }

  def getWeightExecutorId(): BlockId = {
    SparkExtension.getLocalBlockId(id + "pm_w" + executorId)
  }

  def getWeightId(): BlockId = {
    SparkExtension.getLocalBlockId(id + "weight" + executorId)
  }

  def getGradientExecutorId(): BlockId = {
    SparkExtension.getLocalBlockId(id + "pm_g" + executorId)
  }

  private def getGradientBlockId(pidFrom : Int, pidTo : Int): BlockId = {
    SparkExtension.getLocalBlockId(id + "pm" + pidTo + "gBytes" + pidFrom)
  }

  private def getWeightBlockId(pid : Int): BlockId = {
    SparkExtension.getLocalBlockId(id + "pm_wBytes" + pid)
  }

  private def getGradientPartitionId(pid: Int): BlockId = {
    SparkExtension.getLocalBlockId(id + "pm_g_p" + pid)
  }
}
