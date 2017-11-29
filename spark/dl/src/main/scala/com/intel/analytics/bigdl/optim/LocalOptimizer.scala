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
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.{Container, SpatialConvolution, Utils}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
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
  import Optimizer._

  private val coreNumber = Engine.coreNumber()

  private val subModelNumber = Engine.getEngineType match {
    case MklBlas => coreNumber
    case _ => throw new IllegalArgumentException
  }

  private val workingModels = {
    val modelBroadcast = ModelBroadcast()
    model.getParameters()
    val wb = modelBroadcast.getAndClearWeightBias(model.parameters())

    val models = (1 to subModelNumber).map(i => {
      logger.info(s"Clone $i model...")
      val m = model.cloneModule()
      modelBroadcast.putWeightBias(wb, m)
      modelBroadcast.initGradWeightBias(wb, m)
      m
    }).toArray
    modelBroadcast.putWeightBias(wb, model)
    modelBroadcast.initGradWeightBias(wb, model)
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
    optimMethod.clearHistory()
    optimMethod.loadFromTable(state)
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
      val miniBatchBuffer = new Array[MiniBatch[T]](parallelism)
      while (b < parallelism) {
        val offset = b * stackSize + math.min(b, extraSize) + 1
        val length = stackSize + (if (b < extraSize) 1 else 0)
        miniBatchBuffer(b) = batch.slice(offset, length)
        b += 1
      }
      val dataFetchTime = System.nanoTime()

      val results = Engine.default.invokeAndWait(
        (0 until parallelism).map(i =>
          () => {
            val trainingStart = System.nanoTime()
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
            val trainTime = System.nanoTime() - trainingStart
            (_loss, trainTime)
          })
      )
      val lossSum = results.map(_._1).sum
      val maxTrainTime = results.map(_._2).max
      val minTrainTime = results.map(_._2).min
      val meanTrainTime = results.map(_._2).sum.toDouble / results.length

      val aggGradientStart = System.nanoTime()

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
      val updateParameter = System.nanoTime()
      val loss = lossSum / parallelism
      grad.div(ev.fromType(parallelism))

      optimMethod.state.update("epoch", state.get("epoch"))
      optimMethod.state.update("neval", state.get("neval"))
      optimMethod.optimize(_ => (ev.fromType(loss), grad), weight)
      val end = System.nanoTime()
      wallClockTime += end - start
      count += batch.size()
      val head = header(state[Int]("epoch"), count, numSamples, state[Int]("neval"), wallClockTime)
      logger.info(s"$head " +
        s"loss is $loss, iteration time is ${(end - start) / 1e9}s, " +
        s"data fetch time is ${(dataFetchTime - start) / 1e9}s, " +
        s"training time ${(aggGradientStart - dataFetchTime) / 1e9}s, " +
        s"agg gradient time ${(updateParameter - aggGradientStart) / 1e9}s, " +
        s"update gradient time ${(end - updateParameter) / 1e9}s. " +
        s"Throughput is ${batch.size().toDouble / (end - start) * 1e9} record / second. " +
        optimMethod.getHyperParameter()
        )
      logger.info(s"maxTrainTime is ${maxTrainTime / 1e9}, " +
        s"minTrainTime is ${minTrainTime / 1e9}, " +
        s"averageTrainTime is ${meanTrainTime / 1e9}")
      state("neval") = state[Int]("neval") + 1

      if (count >= numSamples) {
        state("epoch") = state[Int]("epoch") + 1
        dataset.shuffle()
        iter = dataset.toLocal().data(train = true)
        count = 0
      }

      validate(wallClockTime)
      checkpoint(wallClockTime)
    }

    logger.info(workingModels.head.getTimes()
      .filter(!_._1.isInstanceOf[Container[Activity, Activity, Float]])
      .map(v => s"${v._1}, ${v._2}, ${v._3}").mkString("\n"))
    logger.info("imgToCol&colToImg time is" + workingModels.head.getTimes()
      .filter(_._1.isInstanceOf[SpatialConvolution[Float]])
      .map(_._1.asInstanceOf[SpatialConvolution[Float]])
      .map(v => v.getCol2ImgTime() + v.getIm2ColTime())
      .sum)
    logger.info(workingModels.head.getTimes()
      .filter(!_._1.isInstanceOf[Container[Activity, Activity, Float]])
      .groupBy(_._1.getClass()).map(v => (v._1, v._2.map(a => a._2).sum,
      v._2.map(b => b._3).sum, v._2.map(c => c._2 + c._3).sum))
      .toArray.sortBy(_._2)
      .map(v => s"${v._1}, ${v._2.toDouble / 1e9}," +
        s" ${v._3.toDouble / 1e9}, ${v._4.toDouble / 1e9}").mkString("\n"))

    // copy running status from workingModels to model
    model.copyStatus(workingModels.head)

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

  private def validate(wallClockTime: Long): Unit = {
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
    logger.info(s"[Wall Clock ${wallClockTime / 1e9}s] Validate model...")

    workingModels.foreach(_.evaluate())

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
            val output = workingModels(b).forward(input)
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
      logger.info(s"[Validation] $count/${validationDataSet.get.size()} Throughput is ${
        batch.size() / ((System.nanoTime() - start) / 1e9)
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

