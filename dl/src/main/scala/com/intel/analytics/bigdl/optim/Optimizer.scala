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

package com.intel.analytics.bigdl.optim

import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DataSet, _}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, File, T, Table}
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.reflect.ClassTag
// TODO: remove D to be MiniBatch[T]
abstract class Optimizer[T: ClassTag, D](
    protected val model: Module[T],
  protected val dataset: DataSet[D],
    protected val criterion: Criterion[T])(implicit ev : TensorNumeric[T])
{
  protected var state: Table = T()
  protected var optimMethod: OptimMethod[T] = new SGD[T]()
  protected var endWhen: Trigger = Trigger.maxIteration(100)

  protected var checkpointTrigger: Option[Trigger] = None
  protected var checkpointPath: Option[String] = None
  protected var isOverWrite: Boolean = false

  protected var validationTrigger: Option[Trigger] = None
  protected var validationMethods: Option[Array[ValidationMethod[T]]] = None
  protected var validationDataSet: Option[DataSet[MiniBatch[T]]] = None

  // To save the metrics.
  protected var metricsTrigger: Option[mutable.HashMap[String, Trigger]] = None
  protected var metricsPath: Option[String] = None

  // To achieve better performance, please set dropPercentage as 0.04
  protected var dropPercentage: Double = 0.0
  protected var maxDropPercentage: Double = 0.0
  protected var computeThresholdbatchSize: Int = 100
  protected var warmupIterationNum: Int = 200

  def optimize(): Module[T]

  @deprecated("Use bigdl.check.singleton instead", "0.1.0")
  def disableCheckSingleton(): this.type = {
    this.checkSingleton = false
    println("disableCheckSingleton is deprecated. Please use bigdl.check.singleton instead")
    this
  }

  // TODO: Remove below code to DistriOptimizer after disableCheckSingleton is not supported
  protected var checkSingleton = System.getProperty("bigdl.check.singleton",
    true.toString).toBoolean

  def setValidation(trigger: Trigger, dataset: DataSet[MiniBatch[T]],
    vMethods : Array[ValidationMethod[T]])
  : this.type = {
    this.validationTrigger = Some(trigger)
    this.validationDataSet = Some(dataset)
    this.validationMethods = Some(vMethods)
    this
  }

  def setValidation(trigger: Trigger, sampleRDD: RDD[Sample[T]],
      vMethods : Array[ValidationMethod[T]], batchSize: Int)
  : this.type = {
    this.validationTrigger = Some(trigger)
    val dataSet =
      (DataSet.rdd(sampleRDD) -> SampleToBatch(batchSize))
        .asInstanceOf[DistributedDataSet[MiniBatch[T]]]
    this.validationDataSet = Some(dataSet)
    this.validationMethods = Some(vMethods)
    this
  }

  def setCheckpoint(path: String, trigger: Trigger): this.type = {
    if (!path.startsWith(File.hdfsPrefix)) {
      require(Files.isDirectory(Paths.get(path)), s"Optimizer.setCheckpoint: $path is not a folder")
    }
    this.checkpointPath = Some(path)
    this.checkpointTrigger = Some(trigger)
    this
  }

  /**
   * Enable metrics and save to a folder.
   *
   * Metrics table should be defined as metricsName -> trigger. We support two kinds of metrics now:
   * 1. Supported train metrics are learningRate, loss, throughput, parameters.
   *    Parameters contains weight, bias, gradWeight, gradBias, and some running status(eg.
   *    runningMean and runningVar in BatchNormalization).
   * 2. Validation metrics relay on ValidationMethods set in method setValidation().
   *
   * Notice: If parameter metrics is empty, we record learningRate, loss and throughput each
   * iteration. But recording parameters is disabled by default, due to get parameters from
   * workers is a heavy operation when the model is very big, like AlexNet and Inception.
   *
   * Usage, to save learningRate each iteration, and parameters each 20 iteration:
   *    enableMetrics("./metrics", mutable.HashMap("learningRate" -> Trigger.severalIteration(1),
   *      "parameters" -> Trigger.severalIteration(20)))
   *
   * @param path The Folder to save metrics.
   * @param metricsTrigger metrics and trigger.
   * @return
   */
  def enableMetrics(
        path: String,
        metricsTrigger: mutable.HashMap[String, Trigger] = null): this.type = {
    this.metricsTrigger = if (null == metricsTrigger) {
      Some(mutable.HashMap("learningRate" -> Trigger.severalIteration(1),
        "loss" -> Trigger.severalIteration(1),
        "throughput" -> Trigger.severalIteration(1)))
    } else {
      Some(metricsTrigger)
    }
    metricsPath = Some(path)
    this
  }

  /**
   * Add some metrics.
   * @param metricsTrigger
   * @return
   */
  def addMetricsTrigger(metricsTrigger: mutable.HashMap[String, Trigger]): this.type = {
    require(!metricsPath.isEmpty, "Optimizer.addMetrics: should enableMetrics first!")
    metricsTrigger.foreach{ v =>
      this.metricsTrigger.get(v._1) = v._2
    }
    this
  }

  /**
   * Get the metrics trigger.
   * @return
   */
  def getMetricsTrigger(): mutable.HashMap[String, Trigger] = {
    metricsTrigger.get
  }

  def overWriteCheckpoint() : this.type = {
    isOverWrite = true
    this
  }

  def setState(state: Table): this.type = {
    this.state = state
    this
  }

  def setOptimMethod(method : OptimMethod[T]): this.type = {
    this.optimMethod = method
    this
  }

  def setEndWhen(endWhen: Trigger): this.type = {
    this.endWhen = endWhen
    this
  }

  def setDropMoudleProperty(dropPercentage: Double, maxDropPercentage: Double,
    batchsize: Int = 100, warmupIteration: Int = 200): this.type = {
    this.dropPercentage = dropPercentage
    this.maxDropPercentage = maxDropPercentage
    require(dropPercentage >= 0 && dropPercentage <= maxDropPercentage)
    this.computeThresholdbatchSize = batchsize
    this.warmupIterationNum = warmupIteration
    this
  }

  def assertEngineInited(): Unit = {
    require(Engine.isInitialized, s"you may forget to initialize Engine object.")
  }
}

object Optimizer {
  private[bigdl] def header(epoch: Int, count: Int, total: Long, iter: Int, wallClockTime: Long)
  : String = {
    s"[Epoch $epoch $count/$total][Iteration $iter][Wall Clock ${wallClockTime / 1e9}s]"
  }

  private[bigdl] def saveModel[T](model: Module[T], checkpointPath : Option[String],
    overWrite : Boolean, postfix: String = ""): Unit = {
    if (checkpointPath.isDefined) {
      model.save(s"${checkpointPath.get}/model$postfix", overWrite)
    }
  }

  private[bigdl] def saveState(state: Table, checkpointPath : Option[String], overWrite : Boolean,
    postfix: String = ""): Unit = {
    if (checkpointPath.isDefined) {
      state.save(s"${checkpointPath.get}/state$postfix", overWrite)
    }
  }

  def apply[T: ClassTag](
      model: Module[T],
      sampleRDD: RDD[Sample[T]],
      criterion: Criterion[T],
      batchSize: Int
      )(implicit ev: TensorNumeric[T]): Optimizer[T, MiniBatch[T]] = {
    new DistriOptimizer[T](
      model = model,
      dataset = (DataSet.rdd(sampleRDD) -> SampleToBatch(batchSize))
        .asInstanceOf[DistributedDataSet[MiniBatch[T]]],
      criterion = criterion
    ).asInstanceOf[Optimizer[T, MiniBatch[T]]]
  }

  def apply[T: ClassTag, D](
    model: Module[T],
    dataset: DataSet[D],
    criterion: Criterion[T]
  )(implicit ev: TensorNumeric[T]): Optimizer[T, D] = {
    dataset match {
      case d: DistributedDataSet[_] =>
        new DistriOptimizer[T](
          model = model,
          dataset = d.asInstanceOf[DistributedDataSet[MiniBatch[T]]],
          criterion = criterion
        ).asInstanceOf[Optimizer[T, D]]
      case d: LocalDataSet[_] =>
        new LocalOptimizer[T](
          model = model,
          dataset = d.asInstanceOf[LocalDataSet[MiniBatch[T]]],
          criterion = criterion
        ).asInstanceOf[Optimizer[T, D]]
      case _ =>
        throw new UnsupportedOperationException
    }
  }
}
