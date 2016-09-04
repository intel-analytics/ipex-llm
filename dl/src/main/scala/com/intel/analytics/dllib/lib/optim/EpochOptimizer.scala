package com.intel.analytics.dllib.lib.optim

import com.intel.analytics.dllib.lib.nn.{Criterion, Module}
import com.intel.analytics.dllib.lib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.dllib.lib.tensor.{T, Table}
import scala.reflect.ClassTag

abstract class EpochOptimizer[T] (
    @transient module : Module[T], criterion: Criterion[T], optm: OptimMethod[T], communicator : Communicator[T],
    dataSets : DataSet[_, T] with HasEpoch, config : Table = T()) extends Optimizer(module, criterion, dataSets) {

  import EpochOptimizer._

  protected var regimes : Array[Regime] = Array[Regime]()

  protected var maxEpoch : Option[Int] = None

  def setMaxEpoch(maxEpoch : Int) : this.type = {
    if(maxEpoch > 0) {
      this.maxEpoch = Some(maxEpoch)
    }
    this
  }

  def setRegimes(regimes : Array[Regime]) : this.type = {
    this.regimes = regimes.clone()
    this
  }
}

class GradAggEpochOptimizer[@specialized(Float, Double) T : ClassTag] (
    @transient module : Module[T], criterion: Criterion[T], optm: OptimMethod[T], communicator : Communicator[T],
    dataSets : DataSet[_, T] with HasEpoch, config : Table = T())
    (implicit ev: TensorNumeric[T]) extends EpochOptimizer(module, criterion, optm, communicator, dataSets, config) {

  override def optimize(): Module[T] = {
    val (masterWeights, masterGrads) = module.getParameters()
    val broadcastEV = dataSets.getSparkContext().broadcast(ev) // don't send whole Optimizer in closure

    var wallClockTime = 0L
    val epochNum = maxEpoch.getOrElse(20)
    val state = T()
    for(i <- 1 to epochNum) {
      logInfo(s"[Epoch $i/$epochNum] Train start")
      val epochStart = System.nanoTime()

      // set optimize parameter from regime
      for(r <- regimes) {
        if(i >= r.startEpoch && i <= r.endEpoch) {
          config.add(r.config)
        }
      }
      logInfo("config" + config)

      logInfo(s"[Epoch $i/$epochNum] Shuffle data")
      dataSets.reset()
      val shuffleEnd = System.nanoTime()
      var accumulateCount = 0
      logInfo(s"[Epoch $i/$epochNum] Shuffle data complete. Takes ${(shuffleEnd - epochStart) / 1e9}s")
      while(!dataSets.epochFinished()) {
        val lossSum = dataSets.getSparkContext()accumulator(0.0, "loss sum")
        val recordsNum = dataSets.getSparkContext().accumulator(0, "record number")
        val stackCount = dataSets.getSparkContext().accumulator(0, "stack count")
        val initGradTime = dataSets.getSparkContext().accumulator(0.0, "init gradient time")
        val constructTensorTime = dataSets.getSparkContext().accumulator(0.0, "construct tensor time")
        val computingTime = dataSets.getSparkContext().accumulator(0.0, "computing time")

        val start = System.nanoTime()
        val resultRDD = dataSets.fetch().zipPartitions(
          models,
          communicator.broadcast(models.map(_.weight), masterWeights), true)(
          (data, modelIter, weights) => {
          weights.next()  // Update local weights

          val localEV = broadcastEV.value
          val localCache = modelIter.next()
          val localModule = localCache.model
          val localCriterion = localCache.criterion
          val localGradient = localCache.gradient

          var tmp = System.nanoTime()
          localModule.training()
          localModule.zeroGradParameters()
          initGradTime += (System.nanoTime() - tmp)
          require(data.hasNext)
          val batch = data.next()
          require(!data.hasNext)
          while(batch.hasNext) {
            tmp = System.nanoTime()
            val (input, target) = batch.next()
            constructTensorTime += System.nanoTime() - tmp
            tmp = System.nanoTime()
            val output = localModule.forward(input)
            lossSum += (localEV.toType[Double](localCriterion.forward(output, target)))
            val errors = localCriterion.backward(output, target)
            localModule.backward(input, errors)
            computingTime += (System.nanoTime() - tmp)
            recordsNum += target.size(1)
            stackCount += 1
          }
          Iterator.single(localGradient)
        })
        val reduceBefore = System.nanoTime()
        communicator.aggregate(resultRDD, masterGrads)
        val reduceAfter = System.nanoTime()
        val localUpdateBefore = System.nanoTime()
        masterGrads.div(ev.fromType[Int](stackCount.value))
        optm.optimize(_=>(ev.fromType(lossSum.value / stackCount.value), masterGrads), masterWeights, config, state)
        val localUpdateTime = System.nanoTime() - localUpdateBefore

        accumulateCount += recordsNum.value
        val end = System.nanoTime()
        logInfo(s"[Epoch $i/$epochNum $accumulateCount/${dataSets.total()}] Train ${recordsNum.value} in ${(end - start) / 1e9}seconds. " +
          s"Throughput is ${recordsNum.value / ((end - start) / 1e9)} records/second. Loss is ${lossSum.value / stackCount.value}. " +
          s"Calculate time is ${(reduceBefore - start) / 1e9}seconds. " +
          s"Construct tensor time is ${constructTensorTime.value / 1e9 / dataSets.getPartitionNum()}seconds. " +
          s"Init gradient time is ${initGradTime.value / 1e9 / dataSets.getPartitionNum()}seconds. " +
          s"Computing time is ${computingTime.value / 1e9 / dataSets.getPartitionNum()}seconds. " +
          s"Reduce time is ${(reduceAfter - reduceBefore) / 1e9}seconds. " +
          s"Local update time is ${localUpdateTime / 1e9}seconds. ")
        communicator match {
          case m : HasMetrics =>
            logInfo(s"[Epoch $i/$epochNum $accumulateCount/${dataSets.total()}] ${m.getMetrics().mkString(" ")}")
        }
      }


      val epochEnd = System.nanoTime()
      wallClockTime = wallClockTime + epochEnd - epochStart
      logInfo(s"[Epoch $i/$epochNum] Epoch finished. Wall clock time is ${wallClockTime / 1e6}ms")
      saveModel(module, i)
      test(module, i)
    }

    saveModel(module)
    module
  }
}

class WeightAvgEpochOptimizer[@specialized(Float, Double) T : ClassTag] (
    @transient module : Module[T], criterion: Criterion[T], optm: OptimMethod[T], communicator : Communicator[T],
    dataSets : DataSet[_, T] with HasEpoch, config : Table = T())(
  implicit ev: TensorNumeric[T]) extends EpochOptimizer(module, criterion, optm, communicator, dataSets, config) {

  override def optimize(): Module[T] = {
    val (masterWeights, masterGrads) = module.getParameters()
    val broadcast = dataSets.getSparkContext().broadcast((ev, config, optm)) // don't send whole Optimizer in closure

    var wallClockTime = 0L
    val epochNum = maxEpoch.getOrElse(10)
    val state = T()
    for(i <- 1 to epochNum) {
      logInfo(s"[Epoch $i/$epochNum] Train start")
      val epochStart = System.nanoTime()

      // set optimize parameter from regime
      for(r <- regimes) {
        if(i >= r.startEpoch && i <= r.endEpoch) {
          config.add(r.config)
        }
      }
      logInfo("config" + config)

      logInfo(s"[Epoch $i/$epochNum] Shuffle data")
      dataSets.reset()
      val shuffleEnd = System.nanoTime()
      var accumulateCount = 0
      logInfo(s"[Epoch $i/$epochNum] Shuffle data complete. Takes ${(shuffleEnd - epochStart) / 1e9}s")
      while(!dataSets.epochFinished()) {
        val lossSum = dataSets.getSparkContext().accumulator(0.0, "loss sum")
        val recordsNum = dataSets.getSparkContext().accumulator(0, "record number")
        val stackCount = dataSets.getSparkContext().accumulator(0, "stack count")
        val initGradTime = dataSets.getSparkContext().accumulator(0.0, "init gradient time")
        val constructTensorTime = dataSets.getSparkContext().accumulator(0.0, "construct tensor time")
        val computingTime = dataSets.getSparkContext().accumulator(0.0, "computing time")
        val workerUpdateTime = dataSets.getSparkContext().accumulator(0.0, "worker update time")
        val batchNum = dataSets.getSparkContext().accumulator(0.0, "batch number")

        val start = System.nanoTime()
        val resultRDD = models.zipPartitions(
          dataSets.fetch(),
          communicator.broadcast(models.map(_.weight), masterWeights))(
          (modelIter, data, weights) => {
            weights.next()  // Update local weights
            val (localEV, localConfig, localOptm) = broadcast.value
            val localCache = modelIter.next()
            val localModule = localCache.model
            val localCriterion = localCache.criterion
            val localWeight = localCache.weight
            val localGradient = localCache.gradient
            val localState = localCache.state
            while(data.hasNext) {
              var localLossSum = 0.0
              var stacks = 0
              var tmp = System.nanoTime()
              localModule.zeroGradParameters()
              initGradTime += (System.nanoTime() - tmp)
              val batch = data.next()
              var recordsss = 0
              while(batch.hasNext) {
                tmp = System.nanoTime()
                val (input, target) = batch.next()
                constructTensorTime += System.nanoTime() - tmp
                tmp = System.nanoTime()
                val output = localModule.forward(input)
                localLossSum += (localEV.toType[Double](localCriterion.forward(output, target)))
                lossSum += localLossSum
                val errors = localCriterion.backward(output, target)
                localModule.backward(input, errors)
                computingTime += (System.nanoTime() - tmp)
                recordsNum += target.size(1)
                recordsss += target.size(1)
                stackCount += 1
                stacks += 1
              }

              val before = System.nanoTime()
              localOptm.optimize(_ => (localEV.fromType(localLossSum / stacks),
                localGradient.div(localEV.fromType(stacks))), localWeight, localConfig, localState)
              workerUpdateTime += System.nanoTime() - before
              batchNum += 1
            }
            Iterator.single(localWeight)
          })

        val reduceBefore = System.nanoTime()
        communicator.aggregate(resultRDD, masterWeights)
        val reduceAfter = System.nanoTime()
        val localUpdateBefore = System.nanoTime()
        masterWeights.div(ev.fromType[Int](dataSets.getPartitionNum()))
        val localUpdateTime = System.nanoTime() - localUpdateBefore

        accumulateCount += recordsNum.value
        val end = System.nanoTime()
        logInfo(s"[Epoch $i/$epochNum $accumulateCount/${dataSets.total()}] Train ${recordsNum.value} in ${(end - start) / 1e9}seconds. " +
          s"Throughput is ${recordsNum.value / ((end - start) / 1e9)} records/second. Loss is ${lossSum.value / stackCount.value}. " +
          s"Calculate time is ${(reduceBefore - start) / 1e9}seconds. " +
          s"Init gradient time is ${initGradTime.value / 1e9 / dataSets.getPartitionNum()}seconds. " +
          s"Construct tensor time is ${constructTensorTime.value / 1e9 / dataSets.getPartitionNum()}seconds. " +
          s"Computing time is ${computingTime.value / 1e9 / dataSets.getPartitionNum()}seconds. " +
          s"Worker update time is ${workerUpdateTime.value / 1e9 / dataSets.getPartitionNum()}seconds. " +
          s"Reduce time is ${(reduceAfter - reduceBefore) / 1e9}seconds. " +
          s"Local update time is ${localUpdateTime / 1e9}seconds. ")
        communicator match {
          case m : HasMetrics =>
            logInfo(s"[Epoch $i/$epochNum $accumulateCount/${dataSets.total()}] ${m.getMetrics().mkString(" ")}")
        }
      }


      val epochEnd = System.nanoTime()
      wallClockTime = wallClockTime + epochEnd - epochStart
      logInfo(s"[Epoch $i/$epochNum] Epoch finished. Wall clock time is ${wallClockTime / 1e6}ms")
      saveModel(module, i)
      test(module, i)
    }

    saveModel(module)
    module
  }
}

object EpochOptimizer {
  case class Regime(startEpoch: Int, endEpoch: Int, config: Table)
}
