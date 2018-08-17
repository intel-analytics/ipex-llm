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

import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.Utils
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.{HeapData, MemoryData, MklDnnContainer}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import org.apache.log4j.Logger

import scala.reflect.ClassTag

object LocalOptimizer {
  val logger = Logger.getLogger(getClass)
}

/**
 * Optimize a model on a single machine
 *
 * @param model model to be optimized
 * @param dataset data set
 * @param criterion criterion to be used
 */
class LocalOptimizer[T: ClassTag] (
  model: Module[T],
  dataset: LocalDataSet[MiniBatch[T]],
  criterion: Criterion[T]
)(implicit ev: TensorNumeric[T])
  extends Optimizer[T, MiniBatch[T]](
    model, dataset, criterion) {

  import LocalOptimizer._
  import Optimizer.{header, saveModel, saveState, checkSubModules, getHyperParameterLog}

  private val coreNumber = Engine.coreNumber()

  private val subModelNumber = Engine.getEngineType match {
    case MklBlas => coreNumber
    case MklDnn => 1
    case _ => throw new IllegalArgumentException
  }

  private val workingModels = {
    model.getParameters()
    val wb = Util.getAndClearWeightBias(model.parameters())

    val models = (1 to subModelNumber).map(i => {
      logger.info(s"Clone $i model...")
      val m = model.cloneModule()
      Util.putWeightBias(wb, m)
      Util.initGradWeightBias(wb, m)
      m match {
        case container: MklDnnContainer => container.compile(TrainingPhase)
        case _ =>
      }
      m
    }).toArray
    Util.putWeightBias(wb, model)
    Util.initGradWeightBias(wb, model)
    models
  }
  private val (weight, grad) = model.getParameters()
  private val gradLength = grad.nElement()
  private val syncGradTaskSize = gradLength / subModelNumber
  private val syncGradExtraTask = gradLength % subModelNumber
  private val syncGradParallelNum =
    if (syncGradTaskSize == 0) syncGradExtraTask else subModelNumber

  private val workingModelWAndG = workingModels.map(_.getParameters())

  private val workingCriterion =
    (1 to subModelNumber).map(_ => criterion.cloneCriterion()).toArray

  override def optimize(): Module[T] = {
    var wallClockTime = 0L
    var count = 0
    optimMethods.values.foreach{ optimMethod =>
      optimMethod.clearHistory()
    }

    // To be compatible with the old usage that user define hyperparameters in a table.
    if (optimMethods.size == 1) {
      optimMethods.head._2.loadFromTable(state)
    }

    checkSubModules(model, optimMethods.keys.toSeq)
    val currentOptimMethods = optimMethods.map{case (subModuleName, optimMethod) =>
      val subModule = model(subModuleName)
      (optimMethod, subModule.get.getParameters())
    }
    state("epoch") = state.get[Int]("epoch").getOrElse(1)
    state("neval") = state.get[Int]("neval").getOrElse(1)
    state("isLayerwiseScaled") = Utils.isLayerwiseScaled(model)

    dataset.shuffle()
    val numSamples = dataset.data(train = false).map(_.size()).reduce(_ + _)
    var iter = dataset.data(train = true)
    logger.info("model thread pool size is " + Engine.model.getPoolSize)
    while (!endWhen(state)) {
      val start = System.nanoTime()

      // Fetch data and prepare tensors
      val batch = iter.next()
      var b = 0
      val stackSize = batch.size() / subModelNumber
      val extraSize = batch.size() % subModelNumber
      val parallelism = if (stackSize == 0) extraSize else subModelNumber
      state("parallelism") = parallelism
      val miniBatchBuffer = new Array[MiniBatch[T]](parallelism)
      while (b < parallelism) {
        val offset = b * stackSize + math.min(b, extraSize) + 1
        val length = stackSize + (if (b < extraSize) 1 else 0)
        miniBatchBuffer(b) = batch.slice(offset, length)
        b += 1
      }
      val dataFetchTime = System.nanoTime()

      val lossSum = Engine.default.invokeAndWait(
        (0 until parallelism).map(i =>
          () => {
            val localModel = workingModels(i)
            localModel.zeroGradParameters()
            localModel.training()
            val localCriterion = workingCriterion(i)
            val input = miniBatchBuffer(i).getInput()
            val target = miniBatchBuffer(i).getTarget()
            val output = localModel.forward(input)
            val _loss = ev.toType[Double](localCriterion.forward(output, target))
            val errors = localCriterion.backward(output, target)
            localModel.backward(input, errors)
            _loss
          })
      ).sum

      // copy multi-model gradient to the buffer
      Engine.default.invokeAndWait(
        (0 until syncGradParallelNum).map(tid =>
          () => {
            val offset = tid * syncGradTaskSize + math.min(tid, syncGradExtraTask)
            val length = syncGradTaskSize + (if (tid < syncGradExtraTask) 1 else 0)
            var i = 0
            while (i < parallelism) {
              if (i == 0) {
                grad.narrow(1, offset + 1, length)
                  .copy(workingModelWAndG(i)._2.narrow(1, offset + 1, length))
              } else {
                grad.narrow(1, offset + 1, length)
                  .add(workingModelWAndG(i)._2.narrow(1, offset + 1, length))
              }
              i += 1
            }
          })
      )
      val loss = lossSum / parallelism
      grad.div(ev.fromType(parallelism))

      parameterProcessors.foreach(_.processParameters(model, state))

      currentOptimMethods.foreach { case (optimMethod, (weight, grad)) =>
        optimMethod.state.update("epoch", state.get("epoch"))
        optimMethod.state.update("neval", state.get("neval"))
        optimMethod.optimize(_ => (ev.fromType(loss), grad), weight)
      }
      val end = System.nanoTime()
      wallClockTime += end - start
      count += batch.size()
      val head = header(state[Int]("epoch"), count, numSamples, state[Int]("neval"), wallClockTime)
      logger.info(s"$head " +
        s"loss is $loss, iteration time is ${(end - start) / 1e9}s " +
        s"data fetch time is ${(dataFetchTime - start) / 1e9}s, " +
        s"train time ${(end - dataFetchTime) / 1e9}s. " +
        s"Throughput is ${batch.size().toDouble / (end - start) * 1e9} record / second. " +
        getHyperParameterLog(optimMethods)
        )
      state("neval") = state[Int]("neval") + 1

      if (count >= numSamples) {
        state("epoch") = state[Int]("epoch") + 1
        dataset.shuffle()
        iter = dataset.toLocal().data(train = true)
        count = 0
      }

      validate(head)
      checkpoint(wallClockTime)
    }

    // copy running status from workingModels to model
    model.setExtraParameter(workingModels.head.getExtraParameter())
    shutdown()

    model
  }

  private def checkpoint(wallClockTime: Long): Unit = {
    if (checkpointTrigger.isEmpty || checkpointPath.isEmpty) {
      return
    }

    val trigger = checkpointTrigger.get
    if (trigger(state) && checkpointPath.isDefined) {
      logger.info(s"[Wall Clock ${wallClockTime / 1e9}s] Save model to path")
      saveModel(workingModels.head, checkpointPath, isOverWrite, s".${state[Int]("neval")}")
      saveState(state, checkpointPath, isOverWrite, s".${state[Int]("neval")}")
    }
  }

  private def validate(header: String): Unit = {
    if (validationTrigger.isEmpty || validationDataSet.isEmpty) {
      return
    }
    val trigger = validationTrigger.get
    if (!trigger(state)) {
      return
    }
    val vMethods = validationMethods.get
    val vMethodsArr = (1 to subModelNumber).map(i => vMethods.map(_.clone())).toArray
    val dataIter = validationDataSet.get.toLocal().data(train = false)
    logger.info(s"$header Validate model...")

    workingModels.foreach(_.evaluate())
    val localWorkingModels = workingModels.map {
      case container: MklDnnContainer =>
        val _c = container.cloneModule()
        _c.compile(InferencePhase)
        _c
      case default => default
    }

    var count = 0
    dataIter.map(batch => {
      val stackSize = batch.size() / subModelNumber
      val extraSize = batch.size() % subModelNumber
      val parallelism = if (stackSize == 0) extraSize else subModelNumber
      val start = System.nanoTime()
      val result = Engine.default.invokeAndWait(
        (0 until parallelism).map(b =>
          () => {
            val offset = b * stackSize + math.min(b, extraSize) + 1
            val length = stackSize + (if (b < extraSize) 1 else 0)
            val currentMiniBatch = batch.slice(offset, length)
            val input = currentMiniBatch.getInput()
            val target = currentMiniBatch.getTarget()
            val output = localWorkingModels(b).forward(input)
            val validatMethods = vMethodsArr(b)
            validatMethods.map(validation => {
              validation(output, target)
            })
          }
        )
      ).reduce((left, right) => {
        left.zip(right).map { case (l, r) =>
          l + r
        }
      })
      count += batch.size()
      logger.info(s"$header Throughput is ${
        batch.size() / ((System.nanoTime() - start) / 1e9)
      } record / sec")
      result
    }).reduce((left, right) => {
      left.zip(right).map { case (l, r) =>
        l + r
      }
    }).zip(vMethods).foreach(r => {
      logger.info(s"$header ${r._2} is ${r._1}")
    })
  }

  private[optim] override def shutdown(): Unit = {
    workingModels.foreach(_.release())
  }
}

