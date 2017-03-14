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

import com.intel.analytics.bigdl.optim.Trigger
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.visualization.tensorboard.{FileReader, FileWriter}

import scala.reflect.ClassTag

abstract class Summary(
      logDir: String,
      appName: String,
      trigger: Map[String, Trigger]) {
  protected val folder: String
  protected val writer: FileWriter

  def getTrigger(): Map[String, Trigger] = {
    trigger
  }

  def addScalar(
        tag: String,
        value: Float,
        step: Long): this.type = {
    writer.addSummary(
      com.intel.analytics.bigdl.utils.Summary.scalar(tag, value), step
    )
    this
  }

  def addHistogram[T: ClassTag](
        tag: String,
        values: Tensor[T],
        step: Long)(implicit ev: TensorNumeric[T]): this.type = {
    writer.addSummary(
      Summary.histogram[T](tag, values), step
    )
    this
  }

  def readScalar(tag: String): Array[(Long, Float, Double)] = {
    FileReader.readScalar(folder, tag)
  }

}

class TrainSummary(
      logDir: String,
      appName: String,
      trigger: Map[String, Trigger] = Map(
        "learningRate" -> Trigger.severalIteration(1),
        "loss" -> Trigger.severalIteration(1),
        "throughput" -> Trigger.severalIteration(1))) extends Summary(logDir, appName, trigger) {
  protected override val folder = s"$logDir/$appName/train"
  protected override val writer = new FileWriter(folder)
}

object TrainSummary{
  def apply(logDir: String,
      appName: String,
      trigger: Map[String, Trigger] = Map(
        "learningRate" -> Trigger.severalIteration(1),
        "loss" -> Trigger.severalIteration(1),
        "throughput" -> Trigger.severalIteration(1))): TrainSummary = {
    new TrainSummary(logDir, appName, trigger)
  }
}

class ValidationSummary(
      logDir: String,
      appName: String,
      trigger: Map[String, Trigger] = null) extends Summary(logDir, appName, trigger) {
  protected override val folder = s"$logDir/$appName/validation"
  protected override val writer = new FileWriter(folder)
}

object ValidationSummary{
  def apply(logDir: String,
      appName: String,
      trigger: Map[String, Trigger] = null): ValidationSummary = {
    new ValidationSummary(logDir, appName, trigger)
  }
}

// Logger for tensor board.
object Summary {
  // tag - specifying the type of the event, e.g. "learning rate", "loss", "throughput"
  // scalar - a single float value, e.g. learning rate, loss, etc.
  // returns a Summary protocol buffer containing a single scalar value.
  // proto definition see  org.tensorflow/org.tensorflow/core/framework/summary.proto
  def scalar(tag: String, scalar : Float): org.tensorflow.framework.Summary = {
    val v = org.tensorflow.framework.Summary.Value.newBuilder().setTag(tag).setSimpleValue(scalar)
    org.tensorflow.framework.Summary.newBuilder().addValue(v).build()
  }

  val limits = makeHistogramBuckets()
  val counts = new Array[Int](limits.length)
  def histogram[T: ClassTag](
      tag: String,
      values: Tensor[T])(implicit ev: TensorNumeric[T]): org.tensorflow.framework.Summary = {
    util.Arrays.fill(counts, 0)

    var squares = 0.0
    values.apply1{value =>
      val v = ev.toType[Double](value)
      squares += v * v
      val index = bisectLeft(limits, v)
      counts(index) += 1
      value
    }

    val histogram = org.tensorflow.framework.HistogramProto.newBuilder()
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
    val v = org.tensorflow.framework.Summary.Value.newBuilder().setTag(tag).setHisto(histogram)
    org.tensorflow.framework.Summary.newBuilder().addValue(v).build()
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



