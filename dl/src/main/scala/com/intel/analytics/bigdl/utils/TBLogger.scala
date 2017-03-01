/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.utils

import java.io.{File, FileOutputStream}
import java.net.InetAddress
import java.util
import java.util.concurrent.Executors

import com.google.common.primitives.{Doubles, Ints, Longs}
import com.intel.analytics.bigdl.tensor.Tensor
import netty.Crc32c
import org.tensorflow.framework.{HistogramProto, Summary}
import org.tensorflow.framework.{HistogramProto, Summary}
import org.tensorflow.util.Event

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

// Logger for tensor board.
object TBLogger {
  // tag - specifying the type of the event, e.g. "learning rate", "loss", "throughput"
  // scalar - a single float value, e.g. learning rate, loss, etc.
  // returns a Summary protocol buffer containing a single scalar value.
  // proto definition see  org.tensorflow/org.tensorflow/core/framework/summary.proto
  def scalar(tag: String, scalar : Float): Summary = {
    val v = Summary.Value.newBuilder().setTag(tag).setSimpleValue(scalar)
    Summary.newBuilder().addValue(v).build()
  }

  val limits = makeHistogramBuckets()
  val counts = new Array[Int](limits.length)
  def histogram(tag: String, values: Tensor[Float]): Summary = {
    util.Arrays.fill(counts, 0)

    var squares = 0.0
    values.apply1{v =>
      squares += v * v
      val index = bisectLeft(limits, v)
      counts(index) += 1
      v
    }

    val histogram = HistogramProto.newBuilder()
      .setMin(values.min())
      .setMax(values.max())
      .setNum(values.nElement())
      .setSum(values.sum())
      .setSumSquares(squares)

    var i = 0
    while (i < counts.length) {
      if (counts(i) != 0) {
        histogram.addBucket(counts(i))
        histogram.addBucketLimit(limits(i))
      }
      i += 1
    }
    val v = Summary.Value.newBuilder().setTag(tag).setHisto(histogram)
    Summary.newBuilder().addValue(v).build()
  }

  def bisectLeft(
      a: Array[Double],
      x: Double,
      lo: Int = 0,
      hi: Int = -1): Int = {
    require(lo >= 0)
    var high = if (hi == -1) {
      a.length
    } else {
      hi
    }
    var low = lo

    while (low < high) {
      val mid = (low + high) / 2
      if (a(mid) < x) {
        low = mid + 1
      } else {
        high = mid
      }
    }
    low
  }

  // TODO: optimize
  private def makeHistogramBuckets(): Array[Double] = {
    var v = 1e-12
    val buckets = ArrayBuffer[Double]()
    val negBuckets = ArrayBuffer[Double]()
    while (v < 1e20) {
      buckets.append(v)
      negBuckets.append(-v)
      v *= 1.1
    }
    (negBuckets.reverse ++ Array(0.0) ++ buckets).toArray
  }

}



