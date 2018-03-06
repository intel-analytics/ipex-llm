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

package com.intel.analytics.bigdl.nn.mkldnn

import breeze.numerics.abs
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.{DenseTensorMath, Tensor}

import scala.collection.mutable.ArrayBuffer

object DnnUtils {

  def nearlyEqual(a: Float, b: Float, epsilon: Double): Boolean = {
    val absA = math.abs(a)
    val absB = math.abs(b)
    val diff = math.abs(a - b)

    val result = if (a == b) {
      true
    } else {
      math.min(diff / (absA + absB), diff) < epsilon
    }

    result
  }

  def nearequals(t1: Tensor[Float], t2: Tensor[Float],
                 epsilon: Double = DenseTensorMath.floatEpsilon): Boolean = {
    var result = true
    t1.map(t2, (a, b) => {
      if (result) {
        result = nearlyEqual(a, b, epsilon)
        if (!result) {
          val diff = math.abs(a - b)
          println("epsilon " + a + "***" + b + "***" + diff / (abs(a) + abs(b)) + "***" + diff)
        }
      }
      a
    })
    return result
  }

  def getunequals(t1: Tensor[Float], t2: Tensor[Float],
                  epsilon: Double = DenseTensorMath.floatEpsilon): Boolean = {
    var result = true
    t1.map(t2, (a, b) => {
      if (true) {
        result = nearlyEqual(a, b, epsilon)
        if (!result) {
          val diff = math.abs(a - b)
          println("epsilon " + a + "***" + b + "***" + diff / (abs(a) + abs(b)) + "***" + diff)
        }
      }
      a
    })
    return true
  }

  def reorderToUser(input: Tensor[Float], output: Tensor[Float], outputFormat: Int): Unit = {
    val dataType = MklDnn.DataType.f32
    val engine = MklDnn.EngineCreate( MklDnn.EngineType.cpu, 0)
    val stream = MklDnn.StreamCreate(MklDnn.StreamType.eager)
    val stream_fwd = new ArrayBuffer[Long]
    val sizes = input.size()
    val dim = input.dim()
    output.resizeAs(input)

    val src_pd = input.getPrimitiveDesc()
    val dst_memory = MklDnnOps.initDataMemory(dim, sizes, outputFormat, dataType, engine)
    val res = MklDnnOps.prepareReorder(dst_memory, src_pd, false)
    // val reorder_primitive = res._1
    val src_memory = res._2

    stream_fwd.clear()
    stream_fwd.append(res._1)

    /* build a simple net */
    val memoryPrimitives = Array(src_memory, dst_memory)
    val buffer = Array(input, output)
    MklDnnOps.streamSubmit(stream, 1, stream_fwd.toArray, 1, memoryPrimitives, buffer)
  }


  def reorderTwoTensor(input: Tensor[Float], inputFormat: Int,
                       output: Tensor[Float], outputFormat: Int): Unit = {
    val dataType = MklDnn.DataType.f32
    val engine = MklDnn.EngineCreate( MklDnn.EngineType.cpu, 0)
    val stream = MklDnn.StreamCreate(MklDnn.StreamType.eager)
    val stream_fwd = new ArrayBuffer[Long]
    val sizes = input.size()
    val dim = input.dim()
    output.resizeAs(input)

    // val src_pd = input.getPrimitiveDesc()
    val src_md = MklDnnOps.memoryDescInit(dim, sizes, dataType, inputFormat)
    val src_pd = MklDnnOps.memoryPrimitiveDescCreate(src_md, engine)

    val dst_memory = MklDnnOps.initDataMemory(dim, sizes, outputFormat, dataType, engine)
    val res = MklDnnOps.prepareReorder(dst_memory, src_pd, false)
    // val reorder_primitive = res._1
    val src_memory = res._2

    stream_fwd.clear()
    stream_fwd.append(res._1)

    /* build a simple net */
    val memoryPrimitives = Array(src_memory, dst_memory)
    val buffer = Array(input, output)
    MklDnnOps.streamSubmit(stream, 1, stream_fwd.toArray, 1, memoryPrimitives, buffer)
  }

  def getTopTimes(times: Array[(AbstractModule[_ <: Activity, _ <: Activity, Float],
    Long, Long)]): Unit = {
    var forwardSum = 0L
    var backwardSum = 0L
    times.foreach(x => {
      forwardSum += x._2
      backwardSum += x._3
    })
    println(s"forwardSum = ${forwardSum}", s"backwardSum = ${backwardSum}")

    val timeBuffer = new ArrayBuffer[(AbstractModule[_ <: Activity,
      _ <: Activity, Float], Long, Long, Long, Double)]
    var i = 0
    while (i < times.length) {
      val all = times(i)._2 + times(i)._3
      val rate = times(i)._3.toDouble/ times(i)._2
      timeBuffer.append((times(i)._1, times(i)._2, times(i)._3, all, rate))
      i += 1
    }
    val sortData = timeBuffer.sortBy(a => a._4)
    sortData.foreach(println)
  }
  def dnnAlexNet(classNum: Int, hasDropout : Boolean = true): Module[Float] = {
    val model = Sequential[Float]()
    model.add(ConvolutionDnn(3, 96, 11, 11, 4, 4, 0, 0, 1, false).setName("conv1"))
    model.add(ReLUDnn[Float](true).setName("relu1")) // ????
    model.add(LRNDnn[Float](5, 0.0001, 0.75).setName("norm1"))
    model.add(PoolingDnn[Float](3, 3, 2, 2).setName("pool1"))
    model.add(ConvolutionDnn(96, 256, 5, 5, 1, 1, 2, 2, 2).setName("conv2"))
    model.add(ReLUDnn[Float](true).setName("relu2")) // ???
    model.add(LRNDnn[Float](5, 0.0001, 0.75).setName("norm2"))
    model.add(PoolingDnn[Float](3, 3, 2, 2).setName("pool2"))
    model.add(ConvolutionDnn(256, 384, 3, 3, 1, 1, 1, 1).setName("conv3"))
    model.add(ReLUDnn[Float](true).setName("relu3"))
    model.add(ConvolutionDnn(384, 384, 3, 3, 1, 1, 1, 1, 2).setName("conv4"))
    model.add(ReLUDnn[Float](true).setName("relu4"))
    model.add(ConvolutionDnn(384, 256, 3, 3, 1, 1, 1, 1, 2).setName("conv5"))
    model.add(ReLUDnn[Float](true).setName("relu5"))
    model.add(PoolingDnn[Float](3, 3, 2, 2).setName("pool5"))
    model.add(MemoryReOrder())
    model.add(View(256 * 6 * 6))
    model.add(nn.Linear[Float](256 * 6 * 6, 4096).setName("fc6"))
    model.add(ReLUDnn[Float](true).setName("relu6"))
    if (hasDropout) model.add(Dropout[Float](0.5).setName("drop6"))
    model.add(nn.Linear[Float](4096, 4096).setName("fc7"))
    model.add(ReLUDnn[Float](true).setName("relu7"))
    if (hasDropout) model.add(Dropout[Float](0.5).setName("drop7"))
    model.add(nn.Linear[Float](4096, classNum).setName("fc8"))
    model.add(LogSoftMax[Float]().setName("loss"))

    model.createDnnEngine(0)
    model.createStream()
    model
  }
}
