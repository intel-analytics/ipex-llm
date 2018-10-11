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
package com.intel.analytics.bigdl.utils

import java.nio.ByteBuffer
import java.util
import java.util.concurrent._
import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.mkl.hardware.{Affinity, CpuInfo}
import com.intel.analytics.bigdl.parameters.{CompressedTensor, FP16CompressedTensor, SerializerInstance}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.log4j.Logger
import org.apache.spark.sparkExtension.SparkExtension
import org.apache.spark.storage.{BlockId, BlockManagerWrapper, StorageLevel}

import scala.collection.mutable
import scala.reflect.{ClassTag, classTag}
import scala.collection.JavaConverters._

trait DistriParameterSynchronizer[T] {

  /**
   * Init synchronization context for new parameter
   * @param name  identifier for parameter
   * @param globalSize total size of parameter
   * @param priority priority for this parameter
   */
  def init(name: String, globalSize: Int, priority: Int = 1, weights: Tensor[T],
           grads: Tensor[T]): Unit

  /**
   * put parameter to global
   * @param name identifier for parameter
   */
  def put(name: String): Unit

  /**
   * get parameter for specific identifier
   * @param name  identifier for parameter
   * @return  parameter
   */
  def get(name: String): (Tensor[T], Tensor[T])

  /**
   * clear the synchronizer
   */
  def clear(): Unit
}

class BlockManagerParameterSynchronizer[T: ClassTag](val partitionID: Int, val totalPartition: Int)
  (implicit ev: TensorNumeric[T]) extends DistriParameterSynchronizer[T]{

  import com.intel.analytics.bigdl.utils.BlockManagerParameterSynchronizer.logger

  @volatile private var shutdown = false

  private val communicationStartCore = CpuInfo.getPhysicalProcessorCount - 4

  private val syncResults: mutable.HashMap[String, FutureTask[Tensor[T]]]
  = new mutable.HashMap[String, FutureTask[Tensor[T]]]()

  private val taskSize: Int = System.getProperty("bigdl.ParameterSynchronier." +
    "asyncTaskSize", "500").toInt

  private val clearPoolSize: Int = System.getProperty("bigdl.ParameterSynchronier." +
    "clearPoolSize", "2").toInt

  private val workerPoolSize: Int = System.getProperty("bigdl.ParameterSynchronier" +
    ".workerPoolSize", "4").toInt

  private val syncPoolSize: Int = Math.max(System.getProperty("bigdl.ParameterSynchronier" +
    ".syncPoolSize",
    (Runtime.getRuntime().availableProcessors() / 2).toString).toInt, 2)

  private val fetchCompletionPoolSize: Int = System.getProperty("bigdl.ParameterSynchronier" +
    ".fetchCompletionPoolSize", "2").toInt

  private val blockFetchRequestQueue: PriorityBlockingQueue[BlockFetchRequest] =
    new PriorityBlockingQueue[BlockFetchRequest](taskSize)

  private val asyncTaskWaitingQueue : PriorityBlockingQueue[SyncRequest] =
    new PriorityBlockingQueue[SyncRequest](taskSize)

  private val longRunningThreads = new util.ArrayList[Thread]()

  private val syncMetaMap = new ConcurrentHashMap[String, SyncMeta[T]]

  val threadCount = new AtomicInteger(0)

  // thread pool to update sow on fetching completion
  private val fetchCompletionPool: ExecutorService = initThreadPool(fetchCompletionPoolSize)
  // to fetch all remote blocks

  private lazy val fetchPool: ExecutorService = initThreadPool(syncPoolSize)
  // to process request
  private val workerPool: ExecutorService = initThreadPool(workerPoolSize)
  // to do local sync threads
  private lazy val syncPool: ExecutorService = initThreadPool(syncPoolSize)
  private lazy val clearPool: ExecutorService = initThreadPool(clearPoolSize)

  private def initThreadPool(capacity: Int): ExecutorService =
    Executors.newFixedThreadPool(capacity, new ThreadFactory {
      override def newThread(r: Runnable): Thread = {
        val t = Executors.defaultThreadFactory().newThread(r)
        t.setDaemon(true)
        t
      }
    })

  initAffinityThreads

  private def initAffinityThreads(): Unit = {
    initAffinityThreadsForThreadPool(fetchPool, syncPoolSize)
    initAffinityThreadsForThreadPool(workerPool, workerPoolSize)
    initAffinityThreadsForThreadPool(fetchCompletionPool, fetchCompletionPoolSize)
    initAffinityThreadsForThreadPool(clearPool, clearPoolSize)
  }

  private def initAffinityThreadsForThreadPool(threadPool: ExecutorService, capacity: Int): Unit = {
    (0 until capacity).map(wp => {
      threadPool.submit(new Runnable {
        override def run(): Unit = {
          val v = threadCount.incrementAndGet()
          Affinity.setAffinity(communicationStartCore + (v) % 4)
        }
      })
    })
  }

  (0 until syncPoolSize).foreach(th => {
    fetchPool.submit(new Runnable {
      override def run(): Unit = {
        val v = threadCount.incrementAndGet()
        longRunningThreads.add(Thread.currentThread)
        while (!shutdown) {
          try {
            val fetchRequest = blockFetchRequestQueue.take
            val syncMeta = fetchRequest.syncMeta
            val pid = fetchRequest.blockFetchFutureTask.fetchOnCompletion.fromPartition
            val aggregated = fetchRequest.aggregated
            val parameterBlockId = if (aggregated) {
              getParameterBlockId(s"${syncMeta.name}_aggregated", syncMeta.counter, pid, -1)
            }
            else {
              getParameterBlockId(syncMeta.name, syncMeta.counter, pid, partitionID)
            }
            val block = BlockManagerWrapper.getLocalOrRemoteBytes(parameterBlockId)
            if (block == None) {
              blockFetchRequestQueue.add(fetchRequest)
            } else {
              val fetchOnCompletion = fetchRequest.blockFetchFutureTask.fetchOnCompletion
              fetchOnCompletion.setFetched(block.get)
              fetchCompletionPool.submit(fetchRequest.blockFetchFutureTask.task)
            }
          } catch {
            case e : InterruptedException =>
              logger.info("exit thread gracefully")
          }
        }
      }
    })
  })


  private class BlockFetchRequest(val syncMeta: SyncMeta[T],
                                  var priority: Int,
                                  val blockFetchFutureTask: BlockFetchFutureTask,
                                  val aggregated: Boolean = false)
    extends Comparable[BlockFetchRequest] {
    override def compareTo(o: BlockFetchRequest): Int = {
      o.priority.compareTo(this.priority)
    }
  }

  private class BlockFetchOnCompletion(val syncRequest: SyncRequest, val fromPartition: Int,
                                       val aggregated: Boolean = false)
    extends Callable[Int] {
    val _classTag = classTag[T]
    private var _fetched: ByteBuffer = null
    def setFetched(fetched: ByteBuffer): Unit = {
      this._fetched = fetched
    }
    override def call(): Int = {
      val syncMeta = syncRequest.syncMeta
      if (aggregated) {
        val partitionParam = syncMeta.aggregatedStateOfWorld.get(fromPartition)
        SerializerInstance.create(_fetched)(_classTag).deCompress(partitionParam)
        val acc = syncRequest.state.addAndGet(1)
        if (acc == syncRequest.syncMeta.partitionToCount + SyncState.PUT_AGGREGATED.id) {
          asyncTaskWaitingQueue.add(syncRequest)
        }
      } else {
        syncMeta.stateOfWorld.put(fromPartition, SerializerInstance.create(_fetched)(_classTag))
        if (syncMeta.stateOfWorld.size == totalPartition) {
          val updated = syncRequest.state.compareAndSet(SyncState.FETCH_PARTITION.id,
            SyncState.AGGREGATION.id)
          if (updated) {
            asyncTaskWaitingQueue.add(syncRequest)
          }
        }
      }
      fromPartition
    }
  }

  (0 until workerPoolSize).foreach(_ => {
    workerPool.submit(new Runnable {
      override def run(): Unit = {
        while (!shutdown) {
          val asyncTaskReq = asyncTaskWaitingQueue.take
          val syncMeta = asyncTaskReq.syncMeta
          val partitonToCount = asyncTaskReq.syncMeta.partitionToCount
          val _classTag = classTag[T]
          val size = asyncTaskReq.syncMeta.globalSize
          val taskSize = size / partitonToCount
          val extraSize = size % partitonToCount
          val state = asyncTaskReq.state.get
          if (state == SyncState.INIT.id) {
            // step 1: clear last status
            syncMeta.stateOfWorld.clear

            val parameter = asyncTaskReq.asyncTask.parameter

            // step 2 : put all local partitioned parameter to global
            val putThreads = (0 until partitonToCount).map { pid =>
              new Callable[Int] {
                override def call(): Int = {
                  try {
                    val offset = 1 + pid * taskSize + math.min(pid, extraSize)
                    val length = taskSize + (if (pid < extraSize) 1 else 0)
                    if (!syncMeta.aggregatedStateOfWorld.contains(pid)) {
                      val partitionParam = parameter.narrow(1, offset, length)
                      syncMeta.aggregatedStateOfWorld.put(pid, partitionParam)
                    }
                    val parameterBlockId = getParameterBlockId(syncMeta.name,
                      syncMeta.counter, partitionID, pid)
                    val fp16param = new FP16CompressedTensor[T](length)(_classTag)
                    fp16param.compress(0, parameter, offset - 1, length)
                    BlockManagerWrapper.putBytes(parameterBlockId,
                      fp16param.bytes(), StorageLevel.MEMORY_ONLY_SER)
                    pid
                  } catch {
                    case t: Throwable =>
                      logger.error("Error: " + ExceptionUtils.getStackTrace(t))
                      throw t
                  }
                }
              }
            }
            putThreads.foreach(pth => syncPool.submit(pth))
            asyncTaskReq.state.set(SyncState.FETCH_PARTITION.id)
            asyncTaskWaitingQueue.add(asyncTaskReq)
          } else if (state == SyncState.FETCH_PARTITION.id) {
            // fetch aggregated partition
            if (partitionID < syncMeta.partitionToCount) {
              val syncThreads = (0 until totalPartition).map { pid =>
                new Callable[Int] {
                  override def call(): Int = {
                    try {
                      val fetchOnCompletion = new BlockFetchOnCompletion(asyncTaskReq, pid)
                      val futureTask = new FutureTask[Int](fetchOnCompletion)
                      val blockFetchFutureTask = new BlockFetchFutureTask(futureTask,
                        fetchOnCompletion)
                      val fetchRequest = new BlockFetchRequest(syncMeta, syncMeta.priority,
                        blockFetchFutureTask)
                      blockFetchRequestQueue.add(fetchRequest)
                      pid
                    } catch {
                      case t: Throwable =>
                        logger.error("Error in processing fetching request: "
                          + ExceptionUtils.getStackTrace(t))
                        throw t
                    }
                  }
                }
              }
              syncThreads.foreach(sth => syncPool.submit(sth))
            } else {
              asyncTaskReq.state.set(SyncState.PUT_AGGREGATED.id)
              asyncTaskWaitingQueue.add(asyncTaskReq)
            }
          } else if (state == SyncState.AGGREGATION.id) {
            // aggregated completed
            val length = taskSize + (if (partitionID < extraSize) 1 else 0)
            val poolSize = Engine.default.getPoolSize
            val innerTaskSize = length / poolSize
            val innerExtraSize = length % poolSize
            val availableTask = if (innerTaskSize == 0) innerExtraSize else poolSize
            val params = syncMeta.stateOfWorld.values().toArray().
              map(_.asInstanceOf[CompressedTensor[T]])
            syncPool.invokeAll((0 until availableTask).map(tid =>
              new Callable[Int] {
                override def call(): Int = {
                  val innerStart = tid * innerTaskSize + math.min(innerExtraSize, tid)
                  val innerLength = innerTaskSize + (if (tid < innerExtraSize) 1 else 0)
                  params.reduce { (l, r) =>
                    l.add(r.bytes(innerStart, innerLength), innerStart, innerLength)
                  }
                  tid
                }
              }
            ).asJava)
            val res = Tensor[T](length)
            params.head.deCompress(res)
            res.div(ev.fromType(totalPartition))
            // step 5: put aggregated to global
            val parameterBlockId = getParameterBlockId(s"${syncMeta.name}_aggregated",
              syncMeta.counter, partitionID, -1)
            val fp16paramAggregated = new FP16CompressedTensor[T](length)(_classTag)
            fp16paramAggregated.compress(0, res, 0, length)
            BlockManagerWrapper.putBytes(parameterBlockId,
              fp16paramAggregated.bytes(), StorageLevel.MEMORY_ONLY_SER)
            asyncTaskReq.state.set(SyncState.PUT_AGGREGATED.id)
            asyncTaskWaitingQueue.add(asyncTaskReq)
          } else if (state == SyncState.PUT_AGGREGATED.id) {
            val aggregatedSyncThreads = (0 until syncMeta.partitionToCount).map { pid =>
              new Callable[Int] {
                override def call(): Int = {
                  try {
                    val fetchOnCompletion = new BlockFetchOnCompletion(asyncTaskReq, pid, true)
                    val futureTask = new FutureTask[Int](fetchOnCompletion)
                    val priorityFutureTask = new BlockFetchFutureTask(futureTask,
                      fetchOnCompletion)
                    val fetchRequest = new BlockFetchRequest(syncMeta, syncMeta.priority,
                      priorityFutureTask, true)
                    blockFetchRequestQueue.add(fetchRequest)
                    pid
                  } catch {
                    case t: Throwable =>
                      logger.error("Error in processing request: "
                        + ExceptionUtils.getStackTrace(t))
                      throw t
                  }
                }
              }
            }
            aggregatedSyncThreads.foreach(aggr => syncPool.submit(aggr))
          } else if (state == SyncState.PUT_AGGREGATED.id + syncMeta.partitionToCount) {
            asyncTaskReq.futureTask.run
          }
        }
      }
    })
  })

  override def init(name: String, globalSize: Int, priority: Int = 1, weights: Tensor[T]
                    , grads: Tensor[T]): Unit = {
    val partitionToCount = if (globalSize < totalPartition) globalSize else totalPartition
    syncMetaMap.putIfAbsent(name, SyncMeta(name, 0, priority, globalSize, partitionToCount,
      new ConcurrentHashMap[Int, CompressedTensor[T]](),
      new ConcurrentHashMap[Int, Tensor[T]](), weights, grads))
  }

  override def put(name: String): Unit = {
    val syncMeta = syncMetaMap.get(name)
    syncMeta.counter += 1
    val asyncTask = new AsyncTask(syncMeta.grads)
    val futureTask = new FutureTask[Tensor[T]](asyncTask)
    val syncRequest = new SyncRequest(new AtomicInteger(0), syncMeta, futureTask, asyncTask)
    asyncTaskWaitingQueue.add(syncRequest)
    if (syncMeta.counter > 1) {
      val clearTask = new ClearTask(name, syncMeta.counter - 1,
        partitionID, syncMeta.partitionToCount)
      clearPool.execute(clearTask)
    }
    syncResults.put(name, futureTask)
  }

  override def get(name: String): (Tensor[T], Tensor[T]) = {
    val syncMeta = syncMetaMap.get(name)
    // no need to do aggregation for first forward
    if (syncMeta.counter == 0) {
      return (null, null)
    }
    require(syncResults.contains(name), "put must be done before get")
    val res = syncResults.get(name).get.get()
    (syncMeta.weights, res)
  }

  override def clear(): Unit = {
    shutdown = true
    longRunningThreads.asScala.foreach(th => if (th != null) {th.interrupt()})
    clearPool.shutdown
    workerPool.shutdown
    syncPool.shutdown
    fetchPool.shutdown
    fetchCompletionPool.shutdown
  }

  private class SyncRequest(var state: AtomicInteger, val syncMeta: SyncMeta[T],
                            val futureTask: FutureTask[Tensor[T]],
                            val asyncTask: AsyncTask)
    extends Comparable[SyncRequest] {
    override def compareTo(o: SyncRequest): Int = {
      o.syncMeta.priority.compareTo(this.syncMeta.priority)
    }
  }

  class AsyncTask(val parameter: Tensor[T]) extends Callable[Tensor[T]] {
    override def call(): Tensor[T] = {
      parameter
    }
  }


  private class ClearTask(name: String, counter: Int, partitionID: Int,
                          partitionToCount: Int)
    extends Runnable {
    override def run(): Unit = {
      try {
        (0 until partitionToCount).foreach(pid => {
          val parameterBlockId = getParameterBlockId(name,
            counter, partitionID, pid)
          BlockManagerWrapper.removeBlock(parameterBlockId)
        })
        // only local parititon < partitionToCount, there are aggregated blocks
        if (partitionID < partitionToCount) {
          val aggregatedParameterBlockId = getParameterBlockId(s"${name}_aggregated",
            counter, partitionID, -1)
          BlockManagerWrapper.removeBlock(aggregatedParameterBlockId)
        }
      } catch {
        case e: Exception =>
          logger.info("exit thread gracefully")

      }
    }
  }

  private class BlockFetchFutureTask(val task : FutureTask[_],
                                     val fetchOnCompletion: BlockFetchOnCompletion) {

  }

  private def getParameterBlockId(name: String, counter: Int, pidFrom: Int, pidTo: Int): BlockId = {
    SparkExtension.getLocalBlockId(name + counter +  pidFrom + "paraBytes" + pidTo)
  }
}

object BlockManagerParameterSynchronizer {
  val logger: Logger = Logger.getLogger(getClass)
  def apply[T: ClassTag](partitionID: Int, totalPartition: Int)
    (implicit ev: TensorNumeric[T]): BlockManagerParameterSynchronizer[T]
    = new BlockManagerParameterSynchronizer[T](partitionID, totalPartition)
}


case class SyncMeta[T](name: String, var counter: Int, priority: Int,
                       globalSize: Int, partitionToCount: Int,
                       stateOfWorld: ConcurrentHashMap[Int, CompressedTensor[T]],
                       aggregatedStateOfWorld: ConcurrentHashMap[Int, Tensor[T]],
                       weights: Tensor[T] = null,
                       grads: Tensor[T] = null)

object SyncState extends Enumeration{
  type State = Value
  val INIT = Value("0")
  val FETCH_PARTITION = Value("1")
  val AGGREGATION = Value("2")
  val PUT_AGGREGATED = Value("3")
}
