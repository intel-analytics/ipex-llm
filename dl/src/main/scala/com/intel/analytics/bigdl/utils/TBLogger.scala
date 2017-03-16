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

import java.util

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.tensorflow.framework.{HistogramProto, Summary}

import scala.reflect.ClassTag

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
  def histogram[T: ClassTag](
      tag: String,
      values: Tensor[T])(implicit ev: TensorNumeric[T]): Summary = {
    util.Arrays.fill(counts, 0)

    var squares = 0.0
    values.apply1{value =>
      val v = ev.toType[Double](value)
      squares += v * v
      val index = bisectLeft(limits, v)
      counts(index) += 1
      value
    }

    val histogram = HistogramProto.newBuilder()
      .setMin(ev.toType[Double](values.min()))
      .setMax(ev.toType[Double](values.max()))
      .setNum(values.nElement())
      .setSum(ev.toType[Double](values.sum()))
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

  private def makeHistogramBuckets(): Array[Double] = {
    var v = 1e-12
    val buckets = new Array[Double](1549)
    var i = 1
    buckets(774) = 0.0
    while (i <= 774) {
      buckets(774 + i) = v
      buckets(774 - i) = -v
      v *= 1.1
      i += 1
    }
    buckets
  }

}



