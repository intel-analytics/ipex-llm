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
import java.util.concurrent._

import com.intel.analytics.bigdl.parameters.AllReduceParameter.{computePoolSize, logger, syncPool, syncPoolSize}
import com.intel.analytics.bigdl.parameters.{CompressedTensor, FP16CompressedTensor, SerializerInstance}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.spark.sparkExtension.SparkExtension
import org.apache.spark.storage.{BlockId, BlockManagerWrapper, StorageLevel}

import scala.collection.mutable
import scala.reflect.{ClassTag, classTag}
import scala.collection.JavaConverters._

trait DistriParameterSynchronizer[T] {

  /**
   * Init synchronization context for new layer
   * @param name  identifier for parameter
   * @param globalSize toal size of parameter
   */
  def init(name: String, globalSize: Int): Unit

  /**
   * put parameter to global
   * @param name identifier for parameter
   * @param parameter  paraemter to put
   */
  def put(name: String, parameter: Tensor[T]): Unit

  /**
   * get parameter for specific identifier
   * @param name  identifier for parameter
   * @return  parameter
   */
  def get(name: String): Tensor[T]
}

class BlockManagerParameterSynchronizer[T: ClassTag](partitionID: Int,
                                                     totalPartition: Int)
                                                    (implicit ev: TensorNumeric[T])
  extends DistriParameterSynchronizer[T] with Serializable {

  private val syncResults: mutable.HashMap[String, Future[Tensor[T]]]
    = new mutable.HashMap[String, Future[Tensor[T]]]()

  private val workerPoolSize: Int = System.getProperty("bigdl.Parameter.syncPoolSize", "4").toInt

  private val syncPoolSize: Int = Math.max(System.getProperty("bigdl.Parameter.computePoolSize",
    (Runtime.getRuntime().availableProcessors() / 2).toString).toInt, 1)

  private lazy val workerPool: ExecutorService =
    Executors.newFixedThreadPool(workerPoolSize, new ThreadFactory {
    override def newThread(r: Runnable): Thread = {
      val t = Executors.defaultThreadFactory().newThread(r)
      t.setDaemon(true)
      t
    }
  })

  private lazy val syncPool: ExecutorService = Executors.newFixedThreadPool(syncPoolSize,
    new ThreadFactory {
      override def newThread(r: Runnable): Thread = {
        val t = Executors.defaultThreadFactory().newThread(r)
        t.setDaemon(true)
        t
      }
    })


  private val syncMetaMap = new ConcurrentHashMap[String, SyncMeta[T]]

  override def init(name: String, globalSize: Int): Unit = {
    syncMetaMap.putIfAbsent(name, SyncMeta(name, 0, globalSize,
      new ConcurrentHashMap[Int, CompressedTensor[T]]()))
  }

  override def put(name: String, parameter: Tensor[T]): Unit = {
    val syncMeta = syncMetaMap.get(name)
    val ayncTask = new AyncTask(syncMeta, parameter)
    val future = workerPool.submit(ayncTask)
    syncResults.put(name, future)
    syncMeta.counter = syncMeta.counter + 1
  }

  override def get(name: String): Tensor[T] = {
    require(syncResults.contains(name), "put must be done before get")
    syncResults.get(name).get.get()
  }

  private class AyncTask(syncMeta: SyncMeta[T],
    parameter: Tensor[T]) extends Callable[Tensor[T]] {
    override def call(): Tensor[T] = {

      // clear last status

      syncMeta.stateOfWorld.clear
      val _classTag = classTag[T]
      val size = syncMeta.globalSize
      val taskSize = size / totalPartition
      val extraSize = size % totalPartition

      // step 1 : put all local partitioned parameter to global

      val putThreads = (0 until totalPartition).map { pid =>
        new Callable[Int] {
          override def call(): Int = {
            try {
              val offset = parameter.storageOffset() + pid * taskSize + math.min(pid, extraSize)
              val length = taskSize + (if (pid < extraSize) 1 else 0)
              val partitionParam = parameter.narrow(1, offset, length)
              val parameterBlockId = getParameterBlockId(syncMeta.name, partitionID, pid)
              println(s"${partitionID} put : ${syncMeta.name}paraBytes${partitionID}${pid}")
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

      // step 2: get all remote paritioned parameter to local

      val syncThreads = (0 until totalPartition).map { pid =>
        new Callable[Int] {
          override def call(): Int = {
            try {
              var block : Option[ByteBuffer] = None
              val parameterBlockId = getParameterBlockId(syncMeta.name, pid, partitionID)
              println(s"${partitionID} get : ${syncMeta.name}paraBytes${pid}${partitionID}")
              while (block == None) {
                block = BlockManagerWrapper.getLocalOrRemoteBytes(parameterBlockId)
              }
              syncMeta.stateOfWorld.put(pid, SerializerInstance.create(block.get)(_classTag))
              pid
            } catch {
              case t: Throwable =>
                logger.error("Error: " + ExceptionUtils.getStackTrace(t))
                throw t
            }
          }
        }
      }
      syncPool.invokeAll(syncThreads.asJava)

      // step 3: aggregation

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
      res
    }
  }

  private def getBlockId(name: String): BlockId = {
    SparkExtension.getLocalBlockId(name)
  }

  private def getParameterBlockId(name: String, pidFrom: Int, pidTo: Int): BlockId = {
    SparkExtension.getLocalBlockId(name + pidFrom + "paraBytes" + pidTo)
  }

}

case class SyncMeta[T](name: String, var counter: Int, globalSize: Int,
                       stateOfWorld: ConcurrentHashMap[Int, CompressedTensor[T]])
