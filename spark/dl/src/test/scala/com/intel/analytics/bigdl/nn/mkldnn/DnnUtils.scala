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
    } else if (a == 0 || b == 0 || diff < java.lang.Float.MIN_NORMAL) {
      diff < (epsilon * java.lang.Float.MIN_NORMAL)
    } else {
      diff / (absA + absB) < epsilon
    }

    result
  }

  def nearequals(t1: Tensor[Float], t2: Tensor[Float]): Boolean = {
    var result = true
    t1.map(t2, (a, b) => {
      if (result) {
        result = nearlyEqual(a, b, DenseTensorMath.floatEpsilon)
      }
      a
    })
    return result
  }

  def getunequals(t1: Tensor[Float], t2: Tensor[Float]): Boolean = {
    var result = true
    t1.map(t2, (a, b) => {
      if (result) {
        // result = nearlyEqual(a, b, DenseTensorMath.floatEpsilon)
        result = nearlyEqual(a, b, 5e-4)
        if (result == false) {
          println(a + " " + b + " " + (abs(a-b)/abs(a)))
        }
      }
      a
    })
    return true
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
    model.add(Linear[Float](256 * 6 * 6, 4096).setName("fc6"))
    model.add(ReLU[Float](true).setName("relu6"))
    if (hasDropout) model.add(Dropout[Float](0.5).setName("drop6"))
    model.add(Linear[Float](4096, 4096).setName("fc7"))
    model.add(ReLU[Float](true).setName("relu7"))
    if (hasDropout) model.add(Dropout[Float](0.5).setName("drop7"))
    model.add(Linear[Float](4096, classNum).setName("fc8"))
    model.add(LogSoftMax[Float]().setName("loss"))
    model
  }
}
