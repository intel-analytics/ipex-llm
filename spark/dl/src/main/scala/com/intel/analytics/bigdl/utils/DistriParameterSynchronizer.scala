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
  def init(name: String, globalSize: Int, priority: Int = 1): Unit

  /**
   * put parameter to global
   * @param name identifier for parameter
   * @param parameter  parameter to put
   */
  def put(name: String, parameter: Tensor[T]): Unit

  /**
   * get parameter for specific identifier
   * @param name  identifier for parameter
   * @return  parameter
   */
  def get(name: String): Tensor[T]

  /**
   * clear the synchronizer
   */
  def clear(): Unit
}

class BlockManagerParameterSynchronizer[T: ClassTag](partitionID: Int,
                                                     totalPartition: Int)
                                                    (implicit ev: TensorNumeric[T])
  extends DistriParameterSynchronizer[T] {

  import com.intel.analytics.bigdl.utils.BlockManagerParameterSynchronizer.logger

  @volatile private var shutdown = false

  private val syncResults: mutable.HashMap[String, FutureTask[Tensor[T]]]
    = new mutable.HashMap[String, FutureTask[Tensor[T]]]()

  private val taskSize: Int = System.getProperty("bigdl.ParameterSynchronier." +
    "asyncTaskSize", "100").toInt

  private val clearPoolSize: Int = System.getProperty("bigdl.ParameterSynchronier." +
    "clearPoolSize", "1").toInt

  private val workerPoolSize: Int = System.getProperty("bigdl.ParameterSynchronier" +
    ".syncPoolSize", "4").toInt

  private val syncPoolSize: Int = Math.max(System.getProperty("bigdl.ParameterSynchronier" +
    ".computePoolSize",
    (Runtime.getRuntime().availableProcessors() / 2).toString).toInt, 2)

  private val fetchCompletionPoolSize: Int = System.getProperty("bigdl.ParameterSynchronier" +
    ".fetchCompletionPoolSize", "1").toInt

  private val asyncTaskWaitingQueue : PriorityBlockingQueue[AsyncFutureTask] =
    new PriorityBlockingQueue[AsyncFutureTask](taskSize)

  private val blockFetchRequestQueue: PriorityBlockingQueue[BlockFetchRequest] =
    new PriorityBlockingQueue[BlockFetchRequest](taskSize)

  private val longRunningThreads = new util.ArrayList[Thread]()

  // thread pool to remove expired blocks
  private lazy val clearPool: ExecutorService =
    Executors.newFixedThreadPool(clearPoolSize, new ThreadFactory {
      override def newThread(r: Runnable): Thread = {
        val t = Executors.defaultThreadFactory().newThread(r)
        t.setDaemon(true)
        t
      }
    })

  // main thread pool to do put-get-aggregate
  private val workerPool: ExecutorService =
    Executors.newFixedThreadPool(workerPoolSize, new ThreadFactory {
    override def newThread(r: Runnable): Thread = {
      val t = Executors.defaultThreadFactory().newThread(r)
      t.setDaemon(true)
      t
    }
  })

  // long running thread to fetch the request
  workerPool.submit(new Runnable {
    override def run(): Unit = {
      longRunningThreads.add(Thread.currentThread)
      while (!shutdown) {
        try {
          val asyncFutureTask = asyncTaskWaitingQueue.take
          workerPool.submit(asyncFutureTask.task)
        } catch {
          case e : InterruptedException =>
            logger.info("exit thread gracefully")
        }
      }
    }
  })

  // thread pool for put and aggregate
  private lazy val syncPool: ExecutorService = Executors.newFixedThreadPool(syncPoolSize,
    new ThreadFactory {
      override def newThread(r: Runnable): Thread = {
        val t = Executors.defaultThreadFactory().newThread(r)
        t.setDaemon(true)
        t
      }
    })

  // thread pool for fetching blocks
  private lazy val fetchPool: ExecutorService = Executors.newFixedThreadPool(syncPoolSize,
    new ThreadFactory {
      override def newThread(r: Runnable): Thread = {
        val t = Executors.defaultThreadFactory().newThread(r)
        t.setDaemon(true)
        t
      }
    })

  // thread pool to update sow on fetching completion
  private val fetchCompletionPool: ExecutorService = Executors.
    newFixedThreadPool(fetchCompletionPoolSize,
    new ThreadFactory {
      override def newThread(r: Runnable): Thread = {
        val t = Executors.defaultThreadFactory().newThread(r)
        t.setDaemon(true)
        t
      }
    })

  (0 until syncPoolSize).foreach(th => {
    fetchPool.submit(new Runnable {
      override def run(): Unit = {
        longRunningThreads.add(Thread.currentThread)
        while (!shutdown) {
          try {
            val fetchRequest = blockFetchRequestQueue.take
            val syncMeta = fetchRequest.syncMeta
            val pid = fetchRequest.futureTask.fetchOnCompletion.fromPartition
            val aggregated = fetchRequest.aggregated
            val parameterBlockId = if (aggregated) {
              getParameterBlockId(s"${syncMeta.name}_aggregated", syncMeta.counter, pid, -1)
            }
            else {
              getParameterBlockId(syncMeta.name, syncMeta.counter, pid, partitionID)
            }
            val block = BlockManagerWrapper.getLocalOrRemoteBytes(parameterBlockId)
            if (block == None) {
              // promote the priporty in next fetch
              fetchRequest.priority += 1
              blockFetchRequestQueue.add(fetchRequest)
            } else {
              val fetchOnCompletion = fetchRequest.futureTask.fetchOnCompletion
              fetchOnCompletion.setFetched(block.get)
              fetchCompletionPool.submit(fetchRequest.futureTask.task)
            }
          } catch {
            case e : InterruptedException =>
              logger.info("exit thread gracefully")
          }
        }
      }
    })
  })

  private val syncMetaMap = new ConcurrentHashMap[String, SyncMeta[T]]

  override def init(name: String, globalSize: Int, priority: Int = 1): Unit = {
    val partitionToCount = if (globalSize < totalPartition) globalSize else totalPartition
    syncMetaMap.putIfAbsent(name, SyncMeta(name, 1, priority, globalSize, partitionToCount,
      new ConcurrentHashMap[Int, CompressedTensor[T]](),
      new ConcurrentHashMap[Int, Tensor[T]]()))
  }

  override def put(name: String, parameter: Tensor[T]): Unit = {
    val syncMeta = syncMetaMap.get(name)
    val asyncTask = new AsyncTask(syncMeta, parameter)
    val futureTask = new FutureTask[Tensor[T]](asyncTask)
    val futureAsyncTask = new AsyncFutureTask(futureTask, syncMeta.priority)
    asyncTaskWaitingQueue.add(futureAsyncTask)
    val clearTask = new ClearTask(name, syncMeta.counter - 1,
      partitionID, syncMeta.partitionToCount)
    clearPool.execute(clearTask)
    syncResults.put(name, futureTask)
  }

  override def get(name: String): Tensor[T] = {
    require(syncResults.contains(name), "put must be done before get")
    val res = syncResults.get(name).get.get()
    val syncMeta = syncMetaMap.get(name)
    syncMeta.counter += 1
    res
  }

  private class ClearTask(name: String, counter: Int, partitionID: Int,
                         partitionToCount: Int)
    extends Runnable {
    override def run(): Unit = {
      (0 until partitionToCount).foreach(pid => {
        val parameterBlockId = getParameterBlockId(name,
          counter, partitionID, pid)
        BlockManagerWrapper.removeBlock(parameterBlockId)
      })
      // only local parititon < partitionToCount, there are aggregated blocks
      if (partitionID < partitionToCount) {
        val aggregatedParameterBlockId = getParameterBlockId(s"${name}_aggregated",
          counter, partitionID, -1)
      }
    }
  }

  private class AsyncFutureTask(val task : FutureTask[_], val priority: Int)
    extends Comparable[AsyncFutureTask] {
    override def compareTo(o: AsyncFutureTask): Int = {
      o.priority.compareTo(this.priority)
    }
  }

  private class AsyncTask(val syncMeta: SyncMeta[T],
    parameter: Tensor[T]) extends Callable[Tensor[T]] {

    override def call(): Tensor[T] = {

      // step 1: clear last status

      syncMeta.stateOfWorld.clear
      syncMeta.aggregatedStateOfWorld.clear

      val partitonToCount = syncMeta.partitionToCount

      val _classTag = classTag[T]
      val size = syncMeta.globalSize
      val taskSize = size / partitonToCount
      val extraSize = size % partitonToCount

      // step 2 : put all local partitioned parameter to global
      val putThreads = (0 until partitonToCount).map { pid =>
        new Callable[Int] {
          override def call(): Int = {
            try {
              val offset = 1 + pid * taskSize + math.min(pid, extraSize)
              val length = taskSize + (if (pid < extraSize) 1 else 0)
              val partitionParam = parameter.narrow(1, offset, length)
              syncMeta.aggregatedStateOfWorld.put(pid, partitionParam)
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
      syncPool.invokeAll(putThreads.asJava)

      // step 3: get all remote paritioned parameter to local
      if (partitionID < partitonToCount) {
        val syncThreads = (0 until totalPartition).map { pid =>
          new Callable[Int] {
            override def call(): Int = {
              try {
                val fetchOnCompletion = new BlockFetchOnCompletion(syncMeta, pid)
                val futureTask = new FutureTask[Int](fetchOnCompletion)
                val priorityFutureTask = new PriorityFutureTask(futureTask, fetchOnCompletion)
                val fetchRequest = new BlockFetchRequest(syncMeta, syncMeta.priority,
                  priorityFutureTask)
                blockFetchRequestQueue.add(fetchRequest)
                futureTask.get
              } catch {
                case t: Throwable =>
                  logger.error("Error: " + ExceptionUtils.getStackTrace(t))
                  throw t
              }
            }
          }
        }
        syncPool.invokeAll(syncThreads.asJava)


        // step 4: aggregation

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
      }

      // step 6: get all other aggregated partitions

      val AggregatedSyncThreads = (0 until partitonToCount).map { pid =>
        new Callable[Int] {
          override def call(): Int = {
            try {
              val fetchOnCompletion = new BlockFetchOnCompletion(syncMeta, pid, true)
              val futureTask = new FutureTask[Int](fetchOnCompletion)
              val priorityFutureTask = new PriorityFutureTask(futureTask, fetchOnCompletion)
              val fetchRequest = new BlockFetchRequest(syncMeta, syncMeta.priority,
                priorityFutureTask, true)
              blockFetchRequestQueue.add(fetchRequest)
              futureTask.get
            } catch {
              case t: Throwable =>
                logger.error("Error: " + ExceptionUtils.getStackTrace(t))
                throw t
            }
          }
        }
      }
      syncPool.invokeAll(AggregatedSyncThreads.asJava)

      parameter
    }
  }

  private class BlockFetchRequest(val syncMeta: SyncMeta[T],
                                  var priority: Int,
                                 val futureTask: PriorityFutureTask,
                                 val aggregated: Boolean = false)
    extends Comparable[BlockFetchRequest] {
    override def compareTo(o: BlockFetchRequest): Int = {
      o.priority.compareTo(this.priority)
    }
  }

  private class BlockFetchOnCompletion(val syncMeta: SyncMeta[T], val fromPartition: Int,
                                      val aggregated: Boolean = false)
    extends Callable[Int] {
    val _classTag = classTag[T]
    private var _fetched: ByteBuffer = null
    def setFetched(fetched: ByteBuffer): Unit = {
      this._fetched = fetched
    }
    override def call(): Int = {
      if (aggregated) {
        val partitionParam = syncMeta.aggregatedStateOfWorld.get(fromPartition)
        SerializerInstance.create(_fetched)(_classTag).deCompress(partitionParam)
      } else {
        syncMeta.stateOfWorld.put(fromPartition, SerializerInstance.create(_fetched)(_classTag))
      }
      fromPartition
    }
  }

 private class PriorityFutureTask(val task : FutureTask[_],
                               val fetchOnCompletion: BlockFetchOnCompletion) {

 }

  private def getBlockId(name: String): BlockId = {
    SparkExtension.getLocalBlockId(name)
  }

  private def getParameterBlockId(name: String, counter: Int, pidFrom: Int, pidTo: Int): BlockId = {
    SparkExtension.getLocalBlockId(name + counter +  pidFrom + "paraBytes" + pidTo)
  }

  override def clear(): Unit = {
    shutdown = true
    longRunningThreads.asScala.foreach(_.interrupt())
    clearPool.shutdown
    syncPool.shutdown
    workerPool.shutdown
    fetchPool.shutdown
    fetchCompletionPool.shutdown
  }
}

object BlockManagerParameterSynchronizer {
  val logger: Logger = Logger.getLogger(getClass)
  def apply[T: ClassTag](partitionID: Int,
            totalPartition: Int)
           (implicit ev: TensorNumeric[T]): BlockManagerParameterSynchronizer[T]
  = new BlockManagerParameterSynchronizer[T](partitionID, totalPartition)
}

case class SyncMeta[T](name: String, var counter: Int, priority: Int,
                       globalSize: Int, partitionToCount: Int,
                       stateOfWorld: ConcurrentHashMap[Int, CompressedTensor[T]],
                      aggregatedStateOfWorld: ConcurrentHashMap[Int, Tensor[T]])
