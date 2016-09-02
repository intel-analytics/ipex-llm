package com.intel.analytics.dllib.lib.optim

import java.nio.ByteBuffer

import com.intel.analytics.dllib.lib.tensor.Tensor
import org.apache.spark.{SparkContext, Accumulator}
import org.apache.spark.rdd.RDD

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.reflect.ClassTag
import scala.reflect._
import ExecutionContext.Implicits.global

/**
  * Communicate parameters between master and workers
 *
  * @tparam T
  */
trait Communicator[@specialized(Float, Double) T] extends Serializable{
  /**
    * Broadcast parameter from master to workers
 *
    * @param parameters
    * @param newParameter
    * @return
    */
  def broadcast(parameters : RDD[Tensor[T]], newParameter : Tensor[T]) : RDD[Tensor[T]]

  /**
    * Aggregate(add) all parameters on workers and send to master
 *
    * @param parameterStatus
    * @param parameter
    * @return
    */
  def aggregate(parameterStatus : RDD[Tensor[T]], parameter : Tensor[T]) : Tensor[T]
}

/**
  * Has metrics to be collect
  */
trait HasMetrics {
  def getMetrics() : Map[String, Double]
}

/**
  * Communicate with a compressed format(smaller size and loss precision
  * )
 *
  * @param partitions rdd has the same partition with train data set
  * @param partitionNum data partition number
  * @param serverNum server number
  * @tparam T
  */
class CompressedCommunicator[T : ClassTag](
    @transient partitions: RDD[_],
    partitionNum : Int,
    serverNum : Option[Int] = None
  ) extends Communicator[T] with HasMetrics {
  @transient
  private var buffers : RDD[Array[Byte]] = null

  @transient
  private var sc = partitions.sparkContext

  private def initBuffer(weightSize : Int) = {
    buffers = partitions.mapPartitions(_ => {
      // Use 2 bytes to store the data
      Iterator.single(new Array[Byte](weightSize * 2))
    }).setName("Compressed Communicator Buffers").persist()
    globalBuffer = new Array[Byte](weightSize * 2)
  }

  @transient
  private var globalBuffer : Array[Byte] = null

  private val metrics = Map(
    "broadcast time" -> sc.accumulator(0.0, "broadcast time"),
    "uncompress time" -> sc.accumulator(0.0, "uncompress time"),
    "compress time" -> sc.accumulator(0.0, "compress time")
  )

  override def broadcast(parameters: RDD[Tensor[T]], newParameter: Tensor[T]): RDD[Tensor[T]] = {
    if(buffers == null) {
      initBuffer(parameters.first().nElement())
    }
    metrics("broadcast time").setValue(0.0)
    metrics("uncompress time").setValue(0.0)
    if(classTag[T] == classTag[Double]) {
      CompressedCommunicator.toFP16(newParameter.asInstanceOf[Tensor[Double]].storage().array(), newParameter.nElement(),
        newParameter.storageOffset() - 1, globalBuffer)
    } else if(classTag[T] == classTag[Float]) {
      CompressedCommunicator.toFP16(newParameter.asInstanceOf[Tensor[Float]].storage().array(), newParameter.nElement(),
        newParameter.storageOffset() - 1, globalBuffer)
    }

    val broadcastParameter = buffers.context.broadcast(globalBuffer)
    parameters.mapPartitions(paramIter => {
      var before = System.nanoTime()
      val localBuffer = broadcastParameter.value
      metrics("broadcast time") += System.nanoTime() - before
      require(paramIter.hasNext)
      val localParameter = paramIter.next()
      require(!paramIter.hasNext)

      before = System.nanoTime()
      if(classTag[T] == classTag[Double]) {
        CompressedCommunicator.fromFP16(localBuffer, localParameter.asInstanceOf[Tensor[Double]].storage().array(),
          localParameter.storageOffset() - 1)
      } else if(classTag[T] == classTag[Float]) {
        CompressedCommunicator.fromFP16(localBuffer, localParameter.asInstanceOf[Tensor[Float]].storage().array(),
          localParameter.asInstanceOf[Tensor[Double]].storageOffset() - 1)
      } else {
        throw new IllegalArgumentException("Only support float/double")
      }
      metrics("uncompress time") += System.nanoTime() - before

      Array(localParameter).iterator
    })
  }

  override def aggregate(parameters: RDD[Tensor[T]], parameter: Tensor[T]): Tensor[T] = {
    if(buffers == null) {
      initBuffer(parameters.first().nElement())
    }
    metrics("compress time").setValue(0.0)
    val updatedBuffers = parameters.zipPartitions(buffers)((paramStatusIter, bufferIter) => {
      require(paramStatusIter.hasNext)
      val localParamIter = paramStatusIter.next()
      require(!paramStatusIter.hasNext)
      val localBuffer = bufferIter.next()

      val before = System.nanoTime()
      if(classTag[T] == classTag[Double]) {
        CompressedCommunicator.toFP16(localParamIter.asInstanceOf[Tensor[Double]].storage().array(), localParamIter.nElement(),
          localParamIter.storageOffset() - 1, localBuffer)
      } else if(classTag[T] == classTag[Float]) {
        CompressedCommunicator.toFP16(localParamIter.asInstanceOf[Tensor[Float]].storage().array(), localParamIter.nElement(),
          localParamIter.storageOffset() - 1, localBuffer)
      }

      metrics("compress time") += System.nanoTime() - before
      Iterator.single(localBuffer)
    })

    globalBuffer = (if (serverNum.isDefined) {
      updatedBuffers.setName("cached result")
      updatedBuffers.persist()
      updatedBuffers.count()
      updatedBuffers.coalesce(serverNum.get)
    } else {
      updatedBuffers
    }).reduce(CompressedCommunicator.FP16Add(_, _))

    if(classTag[T] == classTag[Double]) {
      CompressedCommunicator.fromFP16(globalBuffer, parameter.asInstanceOf[Tensor[Double]].storage().array(),
        parameter.storageOffset() - 1)
    } else if(classTag[T] == classTag[Float]) {
      CompressedCommunicator.fromFP16(globalBuffer, parameter.asInstanceOf[Tensor[Float]].storage().array(),
        parameter.asInstanceOf[Tensor[Double]].storageOffset() - 1)
    } else {
      throw new IllegalArgumentException("Only support float/double")
    }
    parameter
  }

  override def getMetrics(): Map[String, Double] = {
    metrics.mapValues(_.value / partitionNum / 1e9)
  }
}

object CompressedCommunicator {
  val taskSize : Int = System.getProperty("cpu.task.size", "250000").toInt

  def FP16Add(l : Array[Byte], r : Array[Byte]) : Array[Byte] = {
    require(l.length == r.length && l.length % 2 == 0)

    val tasks = for(offset <- 0 until l.length / taskSize + 1) yield Future {
      val buffer = ByteBuffer.allocate(4)
      var i = offset * taskSize
      while (i < l.length && i < (offset + 1) * taskSize) {
        buffer.clear()
        buffer.array()(0) = l(i)
        buffer.array()(1) = l(i + 1)
        buffer.array()(2) = 0
        buffer.array()(3) = 0
        val lFloat = buffer.getFloat


        buffer.clear()
        buffer.array()(0) = r(i)
        buffer.array()(1) = r(i + 1)
        buffer.array()(2) = 0
        buffer.array()(3) = 0
        val rFloat = buffer.getFloat

        buffer.clear()
        buffer.putFloat(lFloat + rFloat)

        l(i) = buffer.array()(0)
        l(i + 1) = buffer.array()(1)

        i += 2
      }
    }

    for(t <- tasks) {
      Await.result(t, Duration.Inf)
    }

    l
  }

  def toFP16(src: Array[Float], nElement : Int, offset : Int, tgt: Array[Byte]): Array[Byte] = {
    require(tgt.length == nElement * 2)

    val tasks = for(taskOffset <- 0 until src.length / taskSize + 1) yield Future {
      val buffer = ByteBuffer.allocate(4)
      var i = taskOffset * taskSize
      while (i < nElement && i < (taskOffset + 1) * taskSize) {
        buffer.clear()
        buffer.putFloat(src(i + offset))
        tgt(i * 2) = buffer.array()(0)
        tgt(i * 2 + 1) = buffer.array()(1)
        i += 1
      }
    }

    for(t <- tasks) {
      Await.result(t, Duration.Inf)
    }

    tgt
  }


  def toFP16(src: Array[Double], nElement : Int, offset : Int, tgt: Array[Byte]): Array[Byte] = {
    require(tgt.length == nElement * 2)

    val tasks = for(taskOffset <- 0 until src.length / taskSize + 1) yield Future {
      val buffer = ByteBuffer.allocate(4)
      var i = taskOffset * taskSize
      while (i < nElement && i < (taskOffset + 1) * taskSize) {
        buffer.clear()
        buffer.putFloat(src(i + offset).toFloat)
        tgt(i * 2) = buffer.array()(0)
        tgt(i * 2 + 1) = buffer.array()(1)
        i += 1
      }
    }

    for(t <- tasks) {
      Await.result(t, Duration.Inf)
    }

    tgt
  }

  def fromFP16(fp16: Array[Byte], tgt: Array[Float], offset : Int) : Array[Float] = {
    require(fp16.length % 2 == 0)
    require(fp16.length / 2 + offset <= tgt.length)

    val tasks = for(taskOffset <- 0 until (fp16.length / 2 / taskSize + 1)) yield Future {
      val buffer = ByteBuffer.allocate(4)
      val bufferData = buffer.array()
      var i = taskOffset * taskSize
      while (i < (fp16.length / 2) && i < (taskOffset + 1) * taskSize) {
        buffer.clear()
        bufferData(0) = fp16(i * 2)
        bufferData(1) = fp16(i * 2 + 1)
        bufferData(2) = 0
        bufferData(3) = 0
        tgt(i + offset) = buffer.getFloat
        i += 1
      }
    }

    for(t <- tasks) {
      Await.result(t, Duration.Inf)
    }

    tgt
  }

  def fromFP16(fp16: Array[Byte], tgt: Array[Double], offset: Int): Array[Double] = {
    require(fp16.length % 2 == 0)
    require(fp16.length / 2 + offset <= tgt.length)

    val tasks = for(taskOffset <- 0 until (fp16.length / 2 / taskSize + 1)) yield Future {
      val buffer = ByteBuffer.allocate(4)
      val bufferData = buffer.array()
      var i = taskOffset * taskSize
      while (i < fp16.length / 2 && i < (taskOffset + 1) * taskSize) {
        buffer.clear()
        bufferData(0) = fp16(i * 2)
        bufferData(1) = fp16(i * 2 + 1)
        bufferData(2) = 0
        bufferData(3) = 0
        tgt(i + offset) = buffer.getFloat.toDouble
        i += 1
      }
    }

    for(t <- tasks) {
      Await.result(t, Duration.Inf)
    }

    tgt
  }
}