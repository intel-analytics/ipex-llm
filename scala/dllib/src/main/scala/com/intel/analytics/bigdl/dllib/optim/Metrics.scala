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

package com.intel.analytics.bigdl.optim

import com.google.common.util.concurrent.AtomicDouble
import org.apache.spark.SparkContext
import org.apache.spark.util.{AccumulatorV2, DoubleAccumulator}

import scala.collection.mutable.{ArrayBuffer, Map}

/**
 * Performance metrics for the training process.
 * Beyond collect local metrics(e.g. throughput) in driver node,
 * it can also be used to collect distributed metrics
 * (e.g. time of some steps among the workers).
 * The is useful for performance analysis.
 */
class Metrics extends Serializable {
  private val localMetricsMap: Map[String, LocalMetricsEntry] = Map()
  private val aggregateDistributeMetricsMap: Map[String, AggregateDistributeMetricsEntry] = Map()
  private val distributeMetricsMap: Map[String, DistributeMetricsEntry] = Map()

  def add(name: String, value: Double): this.type = {
    require(localMetricsMap.contains(name) || aggregateDistributeMetricsMap.contains(name) ||
      distributeMetricsMap.contains(name))
    if (localMetricsMap.contains(name)) {
      localMetricsMap(name).value.addAndGet(value)
    }

    if (aggregateDistributeMetricsMap.contains(name)) {
      aggregateDistributeMetricsMap(name).value.add(value)
    }

    if (distributeMetricsMap.contains(name)) {
      var bufferValue = ArrayBuffer(value)
      distributeMetricsMap(name).value.add(bufferValue)
    }
    this
  }

  def set(name: String, value: Double, parallel: Int = 1): this.type = {
    require(!aggregateDistributeMetricsMap.contains(name), "duplicated distribute metric")
    require(!distributeMetricsMap.contains(name), "duplicated distribute metric2")
    if (localMetricsMap.contains(name)) {
      localMetricsMap(name).value.set(value)
      localMetricsMap(name).parallel = parallel
    } else {
      localMetricsMap(name) = LocalMetricsEntry(new AtomicDouble(value), parallel)
    }
    this
  }

  def set(name: String, value: Double, sc: SparkContext, parallel: Int): this.type = {
    require(!localMetricsMap.contains(name), "duplicated local metric")
    if (aggregateDistributeMetricsMap.contains(name)) {
      aggregateDistributeMetricsMap(name).value.reset()
      aggregateDistributeMetricsMap(name).value.add(value)
      aggregateDistributeMetricsMap(name).parallel = parallel
    } else {
      aggregateDistributeMetricsMap(name) =
        AggregateDistributeMetricsEntry(sc.doubleAccumulator(name), parallel)
      aggregateDistributeMetricsMap(name).value.add(value)
    }
    this
  }

  def set(name: String, value: ArrayBuffer[Double], sc: SparkContext): this.type = {
    require(!localMetricsMap.contains(name), "duplicated local metric")
    require(!aggregateDistributeMetricsMap.contains(name), "duplicated distribute metric")
    if (distributeMetricsMap.contains(name)) {
      distributeMetricsMap(name).value.reset()
      distributeMetricsMap(name).value.add(value)
    } else {
      val accumulableCollection = new ArrayBufferAccumulator
      sc.register(accumulableCollection)
      distributeMetricsMap(name) = DistributeMetricsEntry(accumulableCollection)
      distributeMetricsMap(name).value.add(value)
    }
    this
  }

  def get(name: String): (Double, Int) = {
    require(localMetricsMap.contains(name) || aggregateDistributeMetricsMap.contains(name))
    if (localMetricsMap.contains(name)) {
      (localMetricsMap(name).value.get(), localMetricsMap(name).parallel)
    } else {
      (aggregateDistributeMetricsMap(name).value.value,
        aggregateDistributeMetricsMap(name).parallel)
    }
  }

  def get(name: String, number: Int): Array[Double] = {
    require(distributeMetricsMap.contains(name))
    distributeMetricsMap(name).value.value.toArray.dropRight(number)
  }

  def summary(unit: String = "s", scale: Double = 1e9): String = {
    "========== Metrics Summary ==========\n" +
      localMetricsMap.map(
        entry => s"${entry._1} : ${entry._2.value.get() / entry._2.parallel / scale} $unit\n")
        .mkString("") +
      aggregateDistributeMetricsMap.map(
        entry => s"${entry._1} : ${entry._2.value.value / entry._2.parallel / scale} $unit\n")
        .mkString("") +
      distributeMetricsMap.map { entry =>
        s"${entry._1} : ${entry._2.value.value.map(_ / scale).mkString(" ")} \n"
      }.mkString("") +
      "====================================="
  }
}


private case class LocalMetricsEntry(value: AtomicDouble, var parallel: Int)

private case class AggregateDistributeMetricsEntry(value: DoubleAccumulator, var parallel: Int)

private case class DistributeMetricsEntry(value: ArrayBufferAccumulator)

class ArrayBufferAccumulator extends AccumulatorV2[ArrayBuffer[Double], ArrayBuffer[Double]] {
  private var values = new ArrayBuffer[Double]()

  def reset(): Unit = {
    values.clear()
  }

  def value: ArrayBuffer[Double] = values

  def add(v: ArrayBuffer[Double]): Unit = {
    values ++= v
  }

  def copy(): ArrayBufferAccumulator = {
    val newArrayBufferAccumulator = new ArrayBufferAccumulator
    newArrayBufferAccumulator.values = this.values
    newArrayBufferAccumulator
  }

  def isZero: Boolean = {values.isEmpty}

  def merge(other: AccumulatorV2[ArrayBuffer[Double], ArrayBuffer[Double]]): Unit = other match {
    case o: ArrayBufferAccumulator => values ++= o.values
    case _ => throw new UnsupportedOperationException(
      s"Cannot merge ${this.getClass.getName} with ${other.getClass.getName}"
    )
  }

}
