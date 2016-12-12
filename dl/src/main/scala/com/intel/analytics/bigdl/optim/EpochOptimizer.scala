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

import com.intel.analytics.bigdl.nn.{Criterion, Module}
import com.intel.analytics.bigdl.optim.DistributedOptimizer.CachedModel
import com.intel.analytics.bigdl.parameters.ParameterManager
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

abstract class EpochOptimizer[T: ClassTag](
  @transient module: Module[Tensor[T], Tensor[T], T],
  criterion: Criterion[Tensor[T], T],
  optm: OptimMethod[T],
  pm: ParameterManager[T],
  dataSets: DataSet[_, T] with HasEpoch,
  metrics: Metrics,
  config: Table = T()) extends DistributedOptimizer[T](module, criterion, dataSets) {

  protected var maxEpoch: Option[Int] = None

  def setMaxEpoch(maxEpoch: Int): this.type = {
    if (maxEpoch > 0) {
      this.maxEpoch = Some(maxEpoch)
    }
    this
  }
}

object GradAggEpochOptimizer{
  val logger = Logger.getLogger(getClass)
}

class GradAggEpochOptimizer[T: ClassTag](
  @transient module: Module[Tensor[T], Tensor[T], T],
  criterion: Criterion[Tensor[T], T],
  optm: OptimMethod[T],
  pm: ParameterManager[T],
  dataSets: DataSet[_, T] with HasEpoch,
  metrics: Metrics,
  config: Table = T())
  (implicit ev: TensorNumeric[T])
  extends EpochOptimizer[T](module, criterion, optm, pm, dataSets, metrics, config) {

  import GradAggEpochOptimizer._

  override def optimize(): Module[Tensor[T], Tensor[T], T] = {
    // don't send whole Optimizer in closure
    val broadcastEV = dataSets.getSparkContext().broadcast(ev)

    val sc = dataSets.getSparkContext()
    val partitionNum = dataSets.getPartitionNum()
    var wallClockTime = 0L
    val epochNum = maxEpoch.getOrElse(20)
    val state = T()
    for (i <- 1 to epochNum) {
      logger.info(s"[Epoch $i/$epochNum] Train start")
      val epochStart = System.nanoTime()

      logger.info("config" + config)

      logger.info(s"[Epoch $i/$epochNum] Shuffle data")
      dataSets.reset()
      val shuffleEnd = System.nanoTime()
      var accumulateCount = 0
      logger.info(s"[Epoch $i/$epochNum] Shuffle data complete. Takes ${
        (shuffleEnd -
          epochStart) / 1e9
      }s")
      config("epoch") = i
      while (!dataSets.epochFinished()) {
        val lossSum = sc.accumulator(0.0, "loss sum")
        val recordsNum = sc.accumulator(0, "record number")
        val stackCount = sc.accumulator(0, "stack count")
        metrics.set("init gradient time", 0.0, sc, partitionNum)
        metrics.set("construct tensor time", 0.0, sc, partitionNum)
        metrics.set("computing time", 0.0, sc, partitionNum)
        val driverMetrics = metrics
        val start = System.nanoTime()
        val resultRDD = dataSets.fetch().zipPartitions(
          models,
          pm.sync(models.mapPartitions(iter => Iterator.single(iter.next().weight))), true)(
          (data, modelIter, weights) => {
            weights.next() // Update local weights

            val localEV = broadcastEV.value
            val localCache = modelIter.next()
            val localModule = localCache.model
            val localCriterion = localCache.criterion
            val localGradient = localCache.gradient

            var tmp = System.nanoTime()
            localModule.training()
            localModule.zeroGradParameters()
            driverMetrics.add("init gradient time", System.nanoTime() - tmp)
            require(data.hasNext)
            val batch = data.next()
            require(!data.hasNext)
            while (batch.hasNext) {
              tmp = System.nanoTime()
              val (input, target) = batch.next()
              driverMetrics.add("construct tensor time", System.nanoTime() - tmp)
              tmp = System.nanoTime()
              val output = localModule.forward(input)
              lossSum += (localEV.toType[Double](localCriterion.forward(output, target)))
              val errors = localCriterion.backward(output, target)
              localModule.backward(input, errors)
              driverMetrics.add("computing time", System.nanoTime() - tmp)
              recordsNum += target.size(1)
              stackCount += 1
            }
            Iterator.single(localGradient)
          })
        val reduceBefore = System.nanoTime()
        val driverEV = ev
        val optM = optm
        val configDriver = config
        pm.sumAndUpdate(resultRDD, (weights, gradients, state) => {
          gradients.div(driverEV.fromType[Int](stackCount.value))
          optM.optimize(_ => (driverEV.fromType(lossSum.value / stackCount.value), gradients),
            weights, configDriver, state)
        })
        val reduceAfter = System.nanoTime()

        accumulateCount += recordsNum.value
        val end = System.nanoTime()
        logger.info(s"[Epoch $i/$epochNum $accumulateCount/${dataSets.total()}] Train ${
          recordsNum.value
        } in ${(end - start) / 1e9}seconds. " +
          s"Throughput is ${recordsNum.value / ((end - start) / 1e9)} records/second. Loss is ${
            lossSum.value / stackCount.value
          }. " +
          s"Calculate time is ${(reduceBefore - start) / 1e9}seconds. ")
        logger.info("\n" + metrics.summary())
      }
      val epochEnd = System.nanoTime()
      wallClockTime = wallClockTime + epochEnd - epochStart
      logger.info(s"[Epoch $i/$epochNum] Epoch finished. " +
        s"Wall clock time is ${wallClockTime / 1e6}ms")
      saveModel(module, i)
      saveState(pm.getState(), i)
      test(module, i)
    }

    saveModel(module)
    saveState(pm.getState())
    module
  }

}

object WeightAvgEpochOptimizer{
  val logger = Logger.getLogger(getClass)
}
class WeightAvgEpochOptimizer[T: ClassTag](
  @transient module: Module[Tensor[T], Tensor[T], T],
  criterion: Criterion[Tensor[T], T], optm: OptimMethod[T],
  pm: ParameterManager[T], dataSets: DataSet[_, T] with HasEpoch,
  metrics: Metrics, config: Table = T())(implicit ev: TensorNumeric[T])
  extends EpochOptimizer[T](module, criterion, optm, pm, dataSets, metrics, config) {

  import WeightAvgEpochOptimizer._

  override def optimize(): Module[Tensor[T], Tensor[T], T] = {
    // don't send whole Optimizer in closure
    val broadcast = dataSets.getSparkContext().broadcast((ev, config, optm))

    val sc = dataSets.getSparkContext()
    val partitionNum = dataSets.getPartitionNum()
    var wallClockTime = 0L
    val epochNum = maxEpoch.getOrElse(10)
    val state = T()
    for (i <- 1 to epochNum) {
      logger.info(s"[Epoch $i/$epochNum] Train start")
      val epochStart = System.nanoTime()
      logger.info("config" + config)
      logger.info(s"[Epoch $i/$epochNum] Shuffle data")
      dataSets.reset()
      val shuffleEnd = System.nanoTime()
      var accumulateCount = 0
      logger.info(s"[Epoch $i/$epochNum] Shuffle data complete. Takes" +
        s" ${(shuffleEnd - epochStart) / 1e9}s")
      config("epoch") = i
      while (!dataSets.epochFinished()) {
        val lossSum = sc.accumulator(0.0, "loss sum")
        val recordsNum = sc.accumulator(0, "record number")
        val stackCount = sc.accumulator(0, "stack count")
        val batchNum = sc.accumulator(0.0, "batch number")
        metrics.set("init gradient time", 0.0, sc, partitionNum)
        metrics.set("construct tensor time", 0.0, sc, partitionNum)
        metrics.set("computing time", 0.0, sc, partitionNum)
        metrics.set("worker update time", 0.0, sc, partitionNum)
        val start = System.nanoTime()
        val resultRDD = models.zipPartitions(
          dataSets.fetch(),
          pm.sync(models.map(_.weight)))(
          (modelIter, data, weights) => {
            weights.next() // Update local weights
            val (localEV, localConfig, localOptm) = broadcast.value
            val localCache = modelIter.next()
            val localModule = localCache.model
            val localCriterion = localCache.criterion
            val localWeight = localCache.weight
            val localGradient = localCache.gradient
            val localState = localCache.state
            while (data.hasNext) {
              var localLossSum = 0.0
              var stacks = 0
              var tmp = System.nanoTime()
              localModule.zeroGradParameters()
              localModule.training()
              metrics.add("init gradient time", System.nanoTime() - tmp)
              val batch = data.next()
              var recordsss = 0
              while (batch.hasNext) {
                tmp = System.nanoTime()
                val (input, target) = batch.next()
                metrics.add("construct tensor time", System.nanoTime() - tmp)
                tmp = System.nanoTime()
                val output = localModule.forward(input)
                localLossSum += (localEV.toType[Double](localCriterion.forward(output, target)))
                lossSum += localLossSum
                val errors = localCriterion.backward(output, target)
                localModule.backward(input, errors)
                metrics.add("computing time", System.nanoTime() - tmp)
                recordsNum += target.size(1)
                recordsss += target.size(1)
                stackCount += 1
                stacks += 1
              }

              tmp = System.nanoTime()
              localOptm.optimize(_ => (localEV.fromType(localLossSum / stacks),
                localGradient.div(localEV.fromType(stacks))), localWeight, localConfig, localState)
              metrics.add("worker update time", System.nanoTime() - tmp)
              batchNum += 1
            }
            Iterator.single(localWeight)
          })

        val pn = ev.fromType[Int](partitionNum)
        val reduceBefore = System.nanoTime()
        pm.sumAndUpdate(resultRDD, (weights, weightsSum, state) => {
          weights.copy(weightsSum.div(pn))
        })
        val reduceAfter = System.nanoTime()

        accumulateCount += recordsNum.value
        val end = System.nanoTime()
        logger.info(s"[Epoch $i/$epochNum $accumulateCount/${dataSets.total()}] Train" +
          s" ${recordsNum.value} in ${(end - start) / 1e9}seconds. " +
          s"Throughput is ${recordsNum.value / ((end - start) / 1e9)} records/second." +
          s" Loss is ${lossSum.value / stackCount.value}. " +
          s"Calculate time is ${(reduceBefore - start) / 1e9}seconds. " +
          s"Reduce time is ${(reduceAfter - reduceBefore) / 1e9}seconds.")
        logger.info("\n" + metrics.summary())
      }


      val epochEnd = System.nanoTime()
      wallClockTime = wallClockTime + epochEnd - epochStart
      logger.info(s"[Epoch $i/$epochNum] Epoch finished. " +
        s"Wall clock time is ${wallClockTime / 1e6}ms")
      saveModel(module, i)
      saveState(pm.getState(), i)
      test(module, i)
    }

    saveModel(module)
    saveState(pm.getState())
    module
  }

}
