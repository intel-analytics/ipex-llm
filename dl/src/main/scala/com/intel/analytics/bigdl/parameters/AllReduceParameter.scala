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
import java.nio.ByteBuffer
import java.util.UUID

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}
import org.apache.spark.sparkExtension.SparkExtension
import org.apache.spark.{SparkEnv, TaskContext}
import org.apache.spark.storage.{StorageLevel, TaskResultBlockId, TestBlockId}

import scala.collection.JavaConverters._
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext}
import scala.reflect._

object AllReduceParameter {
  private val syncPoolSize: Int = System.getProperty(
    "bigdl.Parameter.syncPoolSize", "4").toInt

  private val computePoolSize: Int = System.getProperty(
    "bigdl.Parameter.computePoolSize",
    (Runtime.getRuntime().availableProcessors() * 2).toString()).toInt

  private val maxClusterSize = System.getProperty(
    "bigdl.Parameter.maxClusterSize", "10000").toInt

  val syncPool = Executors.newFixedThreadPool(syncPoolSize)
  val computePool = Executors.newFixedThreadPool(computePoolSize)
  var taskSize = 0
  var extraSize = 0
  var tlength: Int = 0  
}

class AllReduceParameter[T: ClassTag]() extends Serializable {
//class AllReduceParameter() extends Serializable {
  import AllReduceParameter._

  @transient lazy val parameterBuffer: CompressedTensor[T] = readParameterBuffer()
  @transient lazy val weights: Tensor[T] = readWeights()
  @transient lazy val gradients: Tensor[T] = readGradients()  
  @transient lazy val state = readState()

  def readParameterBuffer(): CompressedTensor[T] = {
    new FP16SplitsCompressedTensor[T](tlength,
      16).asInstanceOf[CompressedTensor[T]]
  }

  def readWeights(): Tensor[T] = {
    val blockId = SparkExtension.getLocalBlockId("weights0")
    SparkEnv.get.blockManager.getLocal(blockId).map(_.data.next()) match {
      case Some(x) =>
        x.asInstanceOf[Tensor[T]]

      case None =>
        throw new Exception("Please initialize AllReduceParameter first!!")
    }
  }

  def readGradients(): Tensor[T] = {
    val blockId = SparkExtension.getLocalBlockId("gradients0")
    SparkEnv.get.blockManager.getLocal(blockId).map(_.data.next()) match {
      case Some(x) =>
        x.asInstanceOf[Tensor[T]]

      case None =>
        throw new Exception("Please initialize AllReduceParameter first!!")
    }
  }

  def readState(): Table = {
    val blockId = SparkExtension.getLocalBlockId("state0")
    SparkEnv.get.blockManager.getLocal(blockId).map(_.data.next()) match {
      case Some(x) =>
        x.asInstanceOf[Table]

      case None =>
        throw new Exception("Please initialize AllReduceParameter first!!")
    }
  }  
  
  def init(parameter: Tensor[T])(implicit ev: TensorNumeric[T]) = {
    val _classTag = classTag[T]    
      val pid = TaskContext.getPartitionId()    
      val start = pid * taskSize + math.min(pid, extraSize)
      val length = taskSize + (if (pid < extraSize) 1 else 0)

//    val _parameterBuffer = new FP16SplitsCompressedTensor[T](tlength,
//      16).asInstanceOf[CompressedTensor[T]]

      val _weights = Tensor[T](length)(_classTag, ev).copy(parameter.narrow(1,
        start + 1, length))

      val _gradients = Tensor[T](length)(_classTag, ev)

    SparkEnv.get.blockManager.putSingle(SparkExtension.getLocalBlockId("weights0"),
      _weights, StorageLevel.MEMORY_AND_DISK, tellMaster = false)
    SparkEnv.get.blockManager.putSingle(SparkExtension.getLocalBlockId("gradients0"),
      _gradients, StorageLevel.MEMORY_AND_DISK, tellMaster = false)    
    SparkEnv.get.blockManager.putSingle(SparkExtension.getLocalBlockId("state0"),
      T(), StorageLevel.MEMORY_AND_DISK, tellMaster = false)
    val blockId = getWeightBlockId(TaskContext.getPartitionId())
    SparkEnv.get.blockManager.putBytes(blockId,
      SerializerInstance.serialize(_weights).bytes(), StorageLevel.MEMORY_ONLY_SER)    
  }

  def getWeightBlockId(pid : Int): TaskResultBlockId = {
    TaskResultBlockId(maxClusterSize + pid)
  }

  def getGradientBlockId(pidFrom : Int, pidTo : Int): TaskResultBlockId = {
    TaskResultBlockId(pidTo + pidFrom * maxClusterSize * 10)
  }

  def getWeights(localParameter: Tensor[T], partitionNum: Int):
    IndexedSeq[Future[Int]] = {
  val bm = SparkEnv.get.blockManager
    val tasks = (0 until partitionNum).map(pid => {
      syncPool.submit(new Callable[Int] {
        override def call(): Int = {
          val blockId = getWeightBlockId(pid)          
          val localBuffer = bm.getLocalBytes(blockId).getOrElse(bm.getRemoteBytes(blockId).get)
          val start = pid * taskSize + math.min(pid, extraSize)
          val length = taskSize + (if (pid < extraSize) 1 else 0)          
          require(localBuffer.array().length == length * 2)
          SerializerInstance.serialize(localBuffer).deCompress(0, localParameter, start, length)
          pid
        }
      })     
    })
    tasks
  }

  def getGradients(params: Array[CompressedTensor[T]], curPid: Int, partitionNum: Int):
  IndexedSeq[Future[Int]] = {
    val bm = SparkEnv.get.blockManager
    val sgThreads = (0 until partitionNum).map(pid => {
      new Callable[Int] {
        override def call(): Int = {
          val blockId = getGradientBlockId(pid, curPid)          
          val tmp = bm.getLocalBytes(blockId)
            .getOrElse(bm.getRemoteBytes(blockId).getOrElse(
              throw new IllegalArgumentException(s"Can't get the block(${blockId})")
            ))          
          params(pid) = SerializerInstance.serialize(tmp)          
          pid
        }
      }
    })
    syncPool.invokeAll(sgThreads.asJava)

    val length = taskSize + (if (curPid < extraSize) 1 else 0)
    val innerTaskSize = length / computePoolSize
    val innerExtraSize = length % computePoolSize
    val availableTask = if (innerTaskSize == 0) innerExtraSize else computePoolSize
    val tasks = (0 until availableTask).map(tid => computePool.submit(new Callable[Int] {
        override def call(): Int = {
          val innerStart = tid * innerTaskSize + math.min(innerExtraSize, tid)
          val innerLength = innerTaskSize + (if (tid < innerExtraSize) 1 else 0)          
          params.reduce((l, r) => l.add(r.bytes(innerStart, innerLength), innerStart,
            innerLength))
          tid
        }
    }))
    tasks
  }

  def putGradients(parameter: Tensor[T],
   curPid: Int, partitionNum: Int): Unit = {
    var pid = 0
    val bm = SparkEnv.get.blockManager

    require(parameterBuffer != null)
    parameterBuffer.compress(parameter)    
    while (pid < partitionNum) {
      val start = pid * taskSize + math.min(pid, extraSize)
      val length = taskSize + (if (pid < extraSize) 1 else 0)
      val blockId = getGradientBlockId(curPid, pid)
      bm.removeBlock(blockId)
      bm.putBytes(
        blockId, parameterBuffer.bytes(start, length),
        StorageLevel.MEMORY_ONLY_SER)      
      pid += 1
    }
  }

  def putWeights(curPid: Int): Unit = {
    val blockId = getWeightBlockId(curPid)
    val weightsId = SparkExtension.getLocalBlockId("weights0")
    val stateId = SparkExtension.getLocalBlockId("state0")
    require(weights != null)
    require(state != null)
    val bm = SparkEnv.get.blockManager    
    bm.removeBlock(blockId)
    bm.removeBlock(weightsId)
    bm.putSingle((weightsId),
      weights, StorageLevel.MEMORY_AND_DISK, tellMaster = false)
    bm.removeBlock(stateId)
    bm.putSingle((stateId),
      state, StorageLevel.MEMORY_AND_DISK, tellMaster = false)
    bm.putBytes(blockId,
      SerializerInstance.serialize(weights).bytes(), StorageLevel.MEMORY_ONLY_SER)
  }
}