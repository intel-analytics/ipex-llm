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
}
