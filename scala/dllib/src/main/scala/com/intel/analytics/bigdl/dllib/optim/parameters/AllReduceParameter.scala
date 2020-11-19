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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Engine
import java.util.concurrent._
import java.util.concurrent.atomic.AtomicLong
import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.log4j.Logger
import org.apache.spark.TaskContext
import org.apache.spark.sparkExtension.SparkExtension
import org.apache.spark.storage.{BlockId, BlockManagerWrapper, StorageLevel}
import scala.collection.JavaConverters._
import scala.reflect._

object AllReduceParameter {
  private val syncPoolSize: Int = System.getProperty("bigdl.Parameter.syncPoolSize", "4").toInt

  val logger: Logger = Logger.getLogger(getClass)
  val syncPool: ExecutorService = Executors.newFixedThreadPool(syncPoolSize, new ThreadFactory {
    override def newThread(r: Runnable): Thread = {
      val t = Executors.defaultThreadFactory().newThread(r)
      t.setDaemon(true)
      t
    }
  })

  private val computePoolSize: Int = Math.max(System.getProperty("bigdl.Parameter.computePoolSize",
    (Runtime.getRuntime().availableProcessors() / 2).toString).toInt, 1)
  val computePool: ExecutorService = Executors.newFixedThreadPool(computePoolSize,
    new ThreadFactory {
    override def newThread(r: Runnable): Thread = {
      val t = Executors.defaultThreadFactory().newThread(r)
      t.setDaemon(true)
      t
    }
  })

  private val nextId = new AtomicLong(0)

  def newParameter[T: ClassTag](
        partitionNum: Int,
        size: Int,
        offset: Int = 1,
        compress: String = "fp16")(implicit ev: TensorNumeric[T]): AllReduceParameter[T] = {
    new AllReduceParameter(nextId.getAndIncrement(), partitionNum, size, offset, compress)
  }
}

/**
 * Represent parameters stored on the block manager. In distributed optimization, we put parameters
 * on block manager of spark. Each worker syncs parameters through the block manager. Block manager
 * here serves as a parameter server.
 *
 * A Tensor is sliced into `partitionNum` chunks and each chunk is assigned to a particular node
 * (Spark executor). Likewise, gradients for each chunk are also assigned and stored on separate
 * nodes. In this way, gradient aggregation and parameter updates can be performed independently for
 * each chunk on separate nodes.
 *
 * @param id distinguish from other parameters
 * @param partitionNum how many partitions will use this parameter
 * @param size size of the parameter (1D vector)
 * @param paramOffset start index in the origin parameter.
 * @tparam T Tensor element type
 */
class AllReduceParameter[T: ClassTag](
      id: Long,
      partitionNum: Int,
      val size: Int,
      val paramOffset: Int = 1,
      val compress: String = "fp16")(implicit ev: TensorNumeric[T]) extends Serializable {
  import AllReduceParameter._

  @transient private var taskSize = 0
  @transient private var extraSize = 0
  @transient private var partitionId: Int = 0

  /** Tensor to hold a slice of the global weights. */
  @transient lazy val weightPartition: Tensor[T] = readWeightPartition()

  /** Tensor to hold a slice of the global gradients. */
  @transient lazy val gradientPartition: Tensor[T] = readGradientPartition()

  /**
   * This is used by [[java.io.Serializable]] to update some class members when the object is
   * deserialized.
   */
  private def readObject(in: java.io.ObjectInputStream): Unit = {
    in.defaultReadObject()
    taskSize = size / partitionNum
    extraSize = size % partitionNum
    partitionId = TaskContext.getPartitionId()
  }

  /**
   * Reads the portion of the weights assigned to this node from the local block manager. The
   * `init` method must be called before calling this method.
   *
   * @return Tensor containing a slice of the overall model weights.
   */
  private def readWeightPartition(): Tensor[T] = {
    val blockId = getWeightPartitionId()
    BlockManagerWrapper.getLocal(blockId)
      .map(_.data.next().asInstanceOf[Tensor[T]])
      .getOrElse(throw new IllegalStateException("Please initialize AllReduceParameter first!"))
  }

  /**
   * Reads the portion of the gradients assigned to this node from the local block manager. The
   * `init` method must be called before calling this method.
   *
   * @return Tensor containing a slice of the overall model gradients.
   */
  private def readGradientPartition(): Tensor[T] = {
    val blockId = getGradientPartitionId()
    BlockManagerWrapper.getLocal(blockId)
      .map(_.data.next().asInstanceOf[Tensor[T]])
      .getOrElse(throw new IllegalStateException("Please initialize AllReduceParameter first!"))
  }

  /**
   * Returns the start index (starting from 1, within the whole origin parameter)
   * and length of the current local partition
   */
  private[bigdl] def localPartitionRange: (Int, Int) = {
    // add paramOffset to the starting index
    (paramOffset + partitionId * taskSize + math.min(partitionId, extraSize),
      taskSize + (if (partitionId < extraSize) 1 else 0))
  }



  /**
   * This method should be called on each RDD partition before parameter synchronization begins.
   * An empty gradient tensor is placed in the block manager that can be used to store gradients.
   * A 1 / numPartition fraction of the `parameter` tensor is copied to the block manager as a
   * compressed tensor.
   *
   * @param parameter A tensor representing the initial underlying weights of this
   *                  `AllReduceParameter`
   */
  def init(parameter: Tensor[T])(implicit ev: TensorNumeric[T]):
    (Int, Int, Int) = {
    val _classTag = classTag[T]
    val start = partitionId * taskSize + math.min(partitionId, extraSize)
    val length = taskSize + (if (partitionId < extraSize) 1 else 0)

    val _weights = Tensor[T](length)(_classTag, ev).copy(parameter.narrow(1, start + 1, length))
    val _gradients = Tensor[T](length)(_classTag, ev)

    BlockManagerWrapper.removeBlock(getWeightPartitionId())
    BlockManagerWrapper.putSingle(getWeightPartitionId(),
      _weights, StorageLevel.MEMORY_AND_DISK, tellMaster = false)
    BlockManagerWrapper.removeBlock(getGradientPartitionId())
    BlockManagerWrapper.putSingle(getGradientPartitionId(),
      _gradients, StorageLevel.MEMORY_AND_DISK, tellMaster = false)
    val blockId = getWeightBlockId(partitionId)
    val compressed: CompressedTensor[T] = SerializerInstance.create(length, compress)
    compressed.compress(0, parameter, start, length)
    BlockManagerWrapper.putBytes(blockId, compressed.bytes(), StorageLevel.MEMORY_ONLY_SER)
    (partitionId, start, length)
  }

  private def getWeightBlockId(pid: Int): BlockId = {
    SparkExtension.getLocalBlockId(id + "weightBytes" + pid)
  }

  private def getWeightPartitionId(): BlockId = {
    SparkExtension.getLocalBlockId(id + "weights" + partitionId)
  }

  private def getGradientPartitionId(): BlockId = {
    SparkExtension.getLocalBlockId(id + "gradients" + partitionId)
  }

  private def getGradientBlockId(pidFrom: Int, pidTo: Int): BlockId = {
    SparkExtension.getLocalBlockId(id.toString + "_" + pidTo + "gradientBytes" + pidFrom)
  }

  /**
   * Use a fixed thread pool to launch a thread for each partition of the weights. Each thread
   * requests a partition of the weights from the Spark block manager and copies it into
   * `localParameter`.
   *
   * @param localParameter The Tensor that will hold the retrieved weights.
   * @return A [[FutureResult]] which contains a [[Future]] for each thread.
   */
  def getWeights(localParameter: Tensor[T]): FutureResult[Int] = {
    val tasks = (0 until partitionNum).map { pid =>
      syncPool.submit {
        new Callable[Int] {
          override def call(): Int = {
            try {
              val blockId = getWeightBlockId(pid)
              val localBuffer = BlockManagerWrapper.getLocalOrRemoteBytes(blockId).getOrElse {
                throw new RuntimeException(s"Didn't find weight block $blockId in the block " +
                  s"manager. This is usually because executor crashed. Please check your" +
                  s"executors' log see the error (usually an OutOfMemory error)")
              }
              val start = pid * taskSize + math.min(pid, extraSize)
              val length = taskSize + (if (pid < extraSize) 1 else 0)
              SerializerInstance.create(localBuffer, compress)
                .deCompress(0, localParameter, start, length)
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
   * Retrieve gradients for the slice of the model that this node is responsible for from all the
   * other nodes. A new thread is created for each separate node. The gradients are then summed
   * and then stored in decompressed form in `gradientPartition`.
   * @param avgNumbers average numbers.
   */
  def aggregateGradientPartition(avgNumbers: Int): Unit = {
    require(partitionId < partitionNum, s"This parameter was created with $partitionNum " +
      s"partitions. It cannot be used on RDDs with > $partitionNum partitions.")
    val params = new Array[CompressedTensor[T]](partitionNum)
    val sgThreads = (0 until partitionNum).map { pid =>
      new Callable[Int] {
        override def call(): Int = {
          try {
            val blockId = getGradientBlockId(pid, partitionId)
            val tmp = BlockManagerWrapper.getLocalOrRemoteBytes(blockId).get
            params(pid) = SerializerInstance.create(tmp, compress)
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
    syncPool.invokeAll(sgThreads.asJava)

    val length = taskSize + (if (partitionId < extraSize) 1 else 0)
    val poolSize = Engine.default.getPoolSize
    val innerTaskSize = length / poolSize
    val innerExtraSize = length % poolSize
    val availableTask = if (innerTaskSize == 0) innerExtraSize else poolSize
    computePool.invokeAll((0 until availableTask).map(tid =>
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
    params.head.deCompress(gradientPartition)
    gradientPartition.div(ev.fromType(avgNumbers))
  }

  /**
   * Slice gradients learned from this partition of data into chunks, and mark each chunk to be sent
   * to the appropriate parameter node, and put it in the block manager.
   *
   * @param parameter A Tensor that contains gradients computed on the entire model on a single
   *                  partition of data.
   */
  def putGradients(parameter: Tensor[T]): Unit = {
    val _classTag = classTag[T]
    computePool.invokeAll((0 until partitionNum).map(i =>
      new Callable[Int] {
        override def call(): Int = {
          val start = i * taskSize + math.min(i, extraSize)
          val length = taskSize + (if (i < extraSize) 1 else 0)
          val blockId = getGradientBlockId(partitionId, i)
          val block = BlockManagerWrapper.getLocalBytes(blockId)
          if (block.isDefined) {
            val compressed: CompressedTensor[T] = SerializerInstance.create(block.get, compress)
            compressed.compress(0, parameter, start, length)
            i
          } else {
            val compressd: CompressedTensor[T] = SerializerInstance.create(length, compress)
            compressd.compress(0, parameter, start, length)
            BlockManagerWrapper.putBytes(blockId, compressd.bytes(), StorageLevel.MEMORY_ONLY_SER)
            i
          }
        }
      }
    ).asJava)
  }

  /**
   * Put the portion of the weights that this partition is responsible for to the block manager.
   * Weights are placed locally, then pulled when needed by other partitions.
   */
  def sendWeightPartition(): Unit = {
    val blockId = getWeightBlockId(partitionId)
    val localBuffer = BlockManagerWrapper.getLocalBytes(blockId).getOrElse {
      throw new RuntimeException(s"Didn't find weight block $blockId in the block " +
        s"manager. Did you initialize this AllReduceParameter on every executor?")
    }
    SerializerInstance.create(localBuffer, compress).compress(weightPartition)

    val weightsId = getWeightPartitionId()
    val weights = BlockManagerWrapper.getLocal(weightsId)
      .map(_.data.next().asInstanceOf[Tensor[T]])
      .getOrElse(throw new IllegalStateException("Please initialize AllReduceParameter first!"))
    weights.copy(weightPartition)
  }
}

private[bigdl] class FutureResult[T](private val futures: Seq[Future[T]]) {
  def waitResult(): Seq[T] = futures.map(_.get())
}
