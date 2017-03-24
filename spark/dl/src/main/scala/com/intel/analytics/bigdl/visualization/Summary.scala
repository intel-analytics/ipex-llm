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

package com.intel.analytics.bigdl.visualization

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.visualization.tensorboard.{FileWriter}
import org.tensorflow

import scala.reflect.ClassTag

/**
 * Logger for tensorboard.
 * Support scalar and histogram now.
 * @param logDir
 * @param appName
 */
abstract class Summary(
                        logDir: String,
                        appName: String) {
  protected val writer: FileWriter

  /**
   * Add a scalar summary.
   * @param tag tag name.
   * @param value tag value.
   * @param step current step.
   * @return this
   */
  def addScalar(
                 tag: String,
                 value: Float,
                 step: Long): this.type = {
    writer.addSummary(
      Summary.scalar(tag, value), step
    )
    this
  }

  /**
   * Add a histogram summary.
   * @param tag tag name.
   * @param value a tensor.
   * @param step current step.
   * @return this
   */
  def addHistogram[T: ClassTag](
                                 tag: String,
                                 value: Tensor[T],
                                 step: Long)(implicit ev: TensorNumeric[T]): this.type = {
    writer.addSummary(
      Summary.histogram[T](tag, value), step
    )
    this
  }

  /**
   * Read scalar values to an array of triple by tag name.
   * First element of the triple is step, second is value, third is wallclocktime.
   * @param tag tag name.
   * @return an array of triple.
   */
  def readScalar(tag: String): Array[(Long, Float, Double)]

  /**
   * Close this logger.
   */
  def close(): Unit = {
    writer.close()
  }
}

object Summary {

  /**
   * Create a scalar summary.
   * @param tag tag name
   * @param scalar scalar value
   * @return
   */
  def scalar(tag: String, scalar : Float): tensorflow.framework.Summary = {
    val v = tensorflow.framework.Summary.Value.newBuilder().setTag(tag).setSimpleValue(scalar)
    tensorflow.framework.Summary.newBuilder().addValue(v).build()
  }

  private val limits = makeHistogramBuckets()

  /**
   * Create a histogram summary.
   * @param tag tag name.
   * @param values values.
   * @return
   */
  def histogram[T: ClassTag](
      tag: String,
      values: Tensor[T])(implicit ev: TensorNumeric[T]): tensorflow.framework.Summary = {
    val counts = new Array[Int](limits.length)

    var squares = 0.0
    values.apply1{value =>
      val v = ev.toType[Double](value)
      squares += v * v
      val index = bisectLeft(limits, v)
      counts(index) += 1
      value
    }

    val histogram = tensorflow.framework.HistogramProto.newBuilder()
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
    val v = tensorflow.framework.Summary.Value.newBuilder().setTag(tag).setHisto(histogram)
    tensorflow.framework.Summary.newBuilder().addValue(v).build()
  }

  /**
   * Find a bucket for x.
   */
  private def bisectLeft(
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

  /**
   * Create a histogram buckets.
   * @return
   */
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
