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
import com.intel.analytics.bigdl.utils.{Engine, T, Table}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dataset.{DistributedDataSet, LocalDataSet, MiniBatch}

import scala.reflect.ClassTag

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
  protected var validationDataSet: Option[DataSet[D]] = None

  // To achieve better performance, please set dropPercentage as 0.04
  protected var dropPercentage: Double = 0.0
  protected var maxDropPercentage: Double = 0.0
  protected var comupteThresholdbatchSize: Int = 100
  protected var warmupIterationNum: Int = 200

  def disableCheckSingleton(): this.type = {
    this.checkSingleton = false
    this
  }

  protected var checkSingleton = true

  def optimize(): Module[T]

  def setValidation(trigger: Trigger, dataset: DataSet[D],
    vMethods : Array[ValidationMethod[T]])
  : this.type = {
    this.validationTrigger = Some(trigger)
    this.validationDataSet = Some(dataset)
    this.validationMethods = Some(vMethods)
    this
  }

  def setCheckpoint(path: String, trigger: Trigger): this.type = {
    require(Files.isDirectory(Paths.get(path)), s"$path is not a folder")
    this.checkpointPath = Some(path)
    this.checkpointTrigger = Some(trigger)
    this
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
    this.comupteThresholdbatchSize = batchsize
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
