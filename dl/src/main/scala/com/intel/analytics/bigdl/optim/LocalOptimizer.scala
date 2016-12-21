/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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


import java.util.concurrent.{Callable, TimeUnit}

import com.intel.analytics.bigdl.dataset.{Batch, LocalDataSet, DataSet => DataSource}
import com.intel.analytics.bigdl._
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
 * @param model
 * @param dataset
 * @param criterion
 * @param ev$1
 * @param ev
 * @tparam T
 */
class LocalOptimizer[T: ClassTag](
  model: Module[T],
  dataset: DataSource[Iterator[Batch[T]]],
  criterion: Criterion[T]
)(implicit ev: TensorNumeric[T])
  extends Optimizer[T, Iterator[Batch[T]], Iterator[Batch[T]]](
    model, dataset, criterion) {

  import LocalOptimizer._
  import Optimizer._

  private val coreNumber = Engine.coreNumber()

  private val subModelNumber = Engine.getEngineType match {
    case MklBlas => coreNumber
    case MklDnn => 1
  }

  private val (weight, grad) = model.getParameters()
  private val gradLength = grad.nElement()
  private val syncGradTaskSize = gradLength / subModelNumber
  private val syncGradExtraTask = gradLength % subModelNumber
  private val syncGradParallelNum =
    if (syncGradTaskSize == 0) syncGradExtraTask else subModelNumber

  private val workingModels = (1 to subModelNumber).map(i => {
    logger.info(s"Clone $i model...")
    model.cloneModule()
  }).toArray

  private val workingModelWAndG = workingModels.map(_.getParameters())

  workingModelWAndG.foreach(_._1.storage().set(weight.storage()))

  private val workingCriterion =
    (1 to subModelNumber).map(_ => criterion.cloneCriterion()).toArray

  override def optimize(): Module[T] = {
    var wallClockTime = 0L
    var count = 0
    optimMethod.clearHistory(state)
    state("epoch") = state.get[Int]("epoch").getOrElse(1)
    state("neval") = state.get[Int]("neval").getOrElse(1)
    dataset.shuffle()
    var iter = dataset.data(looped = true)
    var iteration = 0
    var moduleTimeList = new Array[Long](subModelNumber * comupteThresholdbatchSize)
    var threshold = Long.MaxValue
    val k = (dropPercentage * comupteThresholdbatchSize * subModelNumber).toInt
    var dropModuleNum = 0

    while (!endWhen(state)) {
      val start = System.nanoTime()

      // Fetch data and prepare tensors
      val batch = iter.next()
      var b = 0
      require(batch.data.size(1) == batch.labels.size(1))
      val stackSize = batch.data.size(1) / subModelNumber
      val extraSize = batch.data.size(1) % subModelNumber
      val parallelism = if (stackSize == 0) extraSize else subModelNumber
      val tensorBuffer = new Array[(Tensor[T], Tensor[T])](parallelism)

      while (b < parallelism) {
        val offset = b * stackSize + math.min(b, extraSize)
        val length = stackSize + (if (b < extraSize) 1 else 0)
        tensorBuffer(b) = (batch.data.narrow(1, offset + 1, length),
          batch.labels.narrow(1, offset + 1, length))
        b += 1
      }
      val dataFetchTime = System.nanoTime()

      // copy multi-model gradient to the buffer
      val losses = new Array[Double](parallelism)
      val records = new Array[Int](parallelism)
      var lossSum = 0.0
      var recordsNum = 0
      val pre = iteration % comupteThresholdbatchSize * subModelNumber

      val trainingTasks = Engine.default.invokeAndWait2(
        (0 until parallelism).map(i =>
          () => {
            val start = System.nanoTime()
            val localModel = workingModels(i)
            localModel.zeroGradParameters()
            localModel.training()
            val localCriterion = workingCriterion(i)
            val (input, target) = tensorBuffer(i)
            val output = localModel.forward(input)
            losses(i) = ev.toType[Double](localCriterion.forward(output, target))
            val errors = localCriterion.backward(output, target)
            localModel.backward(input, errors)
            moduleTimeList(i + pre) = System.nanoTime() - start
            records(i) = target.size(1)
            i
          }), threshold, TimeUnit.NANOSECONDS)
      val finishedTasks = trainingTasks.filter(!_.isCancelled).map(_.get())

      if(finishedTasks.size > parallelism * 0.5) {
        finishedTasks.foreach { index =>
          lossSum += losses(index)
          recordsNum += records(index)
        }
        model.zeroGradParameters()

        val finishedG = finishedTasks.map(index => workingModelWAndG(index)._2)
        dropModuleNum += (parallelism - finishedG.size)

        Engine.default.invokeAndWait2(
          (0 until syncGradParallelNum).map(tid =>
            () => {
              val offset = tid * syncGradTaskSize + math.min(tid, syncGradExtraTask)
              val length = syncGradTaskSize + (if (tid < syncGradExtraTask) 1 else 0)
              var i = 0
              while (i < finishedTasks.size) {
                grad.narrow(1, offset + 1, length)
                  .add(finishedG(i).narrow(1, offset + 1, length))
                i += 1
              }
              tid
            }
          ))
        val loss = lossSum / finishedTasks.size
        grad.div(ev.fromType(finishedTasks.size))

        optimMethod.optimize(_ => (ev.fromType(loss), grad), weight, state)
        val end = System.nanoTime()
        wallClockTime += end - start
        count += recordsNum
        val head =
          header(state[Int]("epoch"), count, dataset.size(), state[Int]("neval"), wallClockTime)
        logger.info(s"$head " +
          s"loss is $loss, iteration time is ${(end - start) / 1e9}s " +
          s"data fetch time is ${(dataFetchTime - start) / 1e9}s, " +
          s"train time ${(end - dataFetchTime) / 1e9}s. " +
          s"Throughput is ${recordsNum.toDouble / (end - start) * 1e9} img / second. " +
          s"Drop module is ${parallelism - finishedTasks.size}")
        state("neval") = state[Int]("neval") + 1

        if (count >= dataset.size()) {
          state("epoch") = state[Int]("epoch") + 1
          dataset.shuffle()
          iter = dataset.data(looped = true)
          count = 0
        }

        validate(wallClockTime)
        checkpoint(wallClockTime)
        iteration += 1

        if(iteration > ignoreIterationNum && iteration % comupteThresholdbatchSize == 0) {
          if (k - dropModuleNum > 0) {
            threshold = Util.kthLargest(moduleTimeList, 0, moduleTimeList.length-1,
              k - dropModuleNum)
          } else {
            threshold = (threshold * 1.01).toLong
          }
          moduleTimeList = new Array[Long](subModelNumber * comupteThresholdbatchSize)
          dropModuleNum = 0
          logger.info(s"threshold: $threshold")
        }
      }
    }

    model
  }

  private def checkpoint(wallClockTime: Long): Unit = {
    if (cacheTrigger.isEmpty || cachePath.isEmpty) {
      return
    }

    val trigger = cacheTrigger.get
    if (trigger(state) && cachePath.isDefined) {
      logger.info(s"[Wall Clock ${wallClockTime / 1e9}s] Save model to path")
      saveModel(workingModels.head, cachePath, isOverWrite, s".${state[Int]("neval")}")
      saveState(state, cachePath, isOverWrite, s".${state[Int]("neval")}")
    }
  }

  private def validate(wallClockTime: Long): Unit = {
    if (validationTrigger.isEmpty || validationDataSet.isEmpty) {
      return
    }
    val trigger = validationTrigger.get
    if (!trigger(state)) {
      return
    }
    val vMethods = validationMethods.get
    val dataIter = validationDataSet.get.asInstanceOf[LocalDataSet[Batch[T]]].data(looped = false)
    logger.info(s"[Wall Clock ${wallClockTime / 1e9}s] Validate model...")

    workingModels.foreach(_.evaluate())

    var count = 0
    dataIter.map(batch => {
      require(batch.data.size(1) == batch.labels.size(1))
      val stackSize = batch.data.size(1) / subModelNumber
      val extraSize = batch.data.size(1) % subModelNumber
      val parallelism = if (stackSize == 0) extraSize else subModelNumber
      val start = System.nanoTime()
      val result = Engine.default.invokeAndWait(
        (0 until parallelism).map(b =>
          () => {
            val offset = b * stackSize + math.min(b, extraSize)
            val length = stackSize + (if (b < extraSize) 1 else 0)
            val input = batch.data.narrow(1, offset + 1, length)
            val target = batch.labels.narrow(1, offset + 1, length)
            val output = workingModels(b).forward(input)
            vMethods.map(validation => {
              validation(output, target)
            })
          }
        )
      ).reduce((left, right) => {
        left.zip(right).map { case (l, r) =>
          l + r
        }
      })
      count += batch.data.size(1)
      logger.info(s"[Validation] $count/${validationDataSet.get.size()} Throughput is ${
        batch.data.size(1) / ((System.nanoTime() - start) / 1e9)
      } record / sec")
      result
    }).reduce((left, right) => {
      left.zip(right).map { case (l, r) =>
        l + r
      }
    }).zip(vMethods).foreach(r => {
      logger.info(s"${r._2} is ${r._1}")
    })
  }
}

