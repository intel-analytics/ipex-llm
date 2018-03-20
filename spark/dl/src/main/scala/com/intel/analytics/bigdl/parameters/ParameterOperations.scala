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
package com.intel.analytics.bigdl.parameters

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch}
import org.apache.spark.rdd.RDD
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.optim.DistriOptimizer.Cache
import com.intel.analytics.bigdl.optim.Metrics
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import org.apache.spark.broadcast.Broadcast

import scala.collection.mutable

/**
 * Process parameters trait, subclass must be independent of each other
 */
private[bigdl] trait ParameterProcessor
  extends Serializable {
  /**
   * Get meta data, only executed once in driver
   *
   * @param dataset a RDD of training data
   * @param parameters [[AllReduceParameter]]
   * @param state A table contained needed information
   */
  def init[T](dataset: DistributedDataSet[MiniBatch[T]],
    parameters: AllReduceParameter[T],
    state: Table)(implicit ev: TensorNumeric[T]) : Unit = {}

  /**
   * Collect global data according to operations list, usually executed in driver
   *
   * @param models cached models
   * @param parameters [[AllReduceParameter]]
   * @param metrics metrics
   * @param state A table contained needed information
   */
  def collectGlobalData[T](models: RDD[Cache[T]],
    parameters: AllReduceParameter[T],
    metrics: Metrics,
    state: Table)(implicit ev: TensorNumeric[T]) : Unit = {}

  /**
   * Advance operations to process parameters, usually executed in worker
   *
   * @param parameters [[AllReduceParameter]]
   * @param state A table contained needed information
   */
  def processParameters[T](parameters: AllReduceParameter[T],
    modelCache: Cache[T],
    state: Table)(implicit ev: TensorNumeric[T]): Unit = {}

  /**
   * Advance operations to process parameters, usually executed in local optimer
   *
   * @param model the model to be trained
   * @param state A table contained needed information
   */
  def processParameters[T](model: Module[T],
    state: Table)(implicit ev: TensorNumeric[T]): Unit = {}
}

/**
 * Process constant clipping
 */
private[bigdl] class ConstantClippingProcessor(min: Double, max: Double)
  extends ParameterProcessor {
  override def processParameters[T](parameters: AllReduceParameter[T],
    modelCache: Cache[T],
    state: Table)(implicit ev: TensorNumeric[T]): Unit = {
    parameters.gradientPartition.clamp(min, max)
  }

  override def processParameters[T](model: Module[T],
    state: Table)(implicit ev: TensorNumeric[T]): Unit = {
    val gradients = model.getParameters()._2
    gradients.clamp(min, max)
  }
}

/**
 * Process l2 norm clipping
 */
private[bigdl] class L2NormClippingProcessor(l2NormThreshold: Double)
  extends ParameterProcessor {
  override def collectGlobalData[T](models: RDD[Cache[T]],
    parameters: AllReduceParameter[T],
    metrics: Metrics,
    state: Table)(implicit ev: TensorNumeric[T]) : Unit = {
    val numFinishedModel = state.get[Int]("numFinishedModel").get
    val parallelism = state.get[Int]("parallelism").get
    val aggregatedG = state.get[Boolean]("aggregateG").get

    val sumSquare = models.mapPartitions(modelIter => {
      if (!aggregatedG) {
        val getG = System.nanoTime()
        parameters.aggregateGradientPartition(numFinishedModel)
        metrics.add("aggregrateGradientParition average executor",
          System.nanoTime() - getG)
      }

//      val gradLength = parameters.gradientPartition.nElement()
//      val taskSize = gradLength / threadNum
//      val extraTask = gradLength % threadNum
//      val parallelNum = if (taskSize == 0) extraTask else threadNum
//      val squares = new Array[Double](parallelNum)
//      Engine.default.invokeAndWait((0 until parallelNum).map(tid => () => {
//        val offset = tid * taskSize + math.min(tid, extraTask)
//        val length = taskSize + (if (tid < extraTask) 1 else 0)
//        squares(tid) = ev.toType[Double](
//          parameters.gradientPartition.narrow(1, offset + 1, length).sumSquare())
//      }))
//      var sum = 0.0
//      var i = 0
//      while (i < parallelNum) {
//        sum += squares(i)
//        i += 1
//      }
      val sum = Util.getSumsquareInParallel(parameters.gradientPartition, parallelism)
      Iterator.single(sum)
    }).reduce(_ + _)

    state("aggregateG") = true
    state("l2Norm") = math.sqrt(sumSquare)
  }

  override def processParameters[T](parameters: AllReduceParameter[T],
    modelCache: Cache[T],
    state: Table)(implicit ev: TensorNumeric[T]): Unit = {
    val l2Norm = state.get[Double]("l2Norm").get
    if (l2Norm > l2NormThreshold) {
      val scale = ev.fromType[Double](l2Norm / l2NormThreshold)
      parameters.gradientPartition.div(scale)
    }
  }

  override def processParameters[T](model: Module[T],
    state: Table)(implicit ev: TensorNumeric[T]): Unit = {
    val parallelism = state.get[Int]("parallelism").get
    val gradients = model.getParameters()._2
    val l2Norm = math.sqrt(Util.getSumsquareInParallel(gradients, parallelism))

    if (l2Norm > l2NormThreshold) {
      val scale = ev.fromType[Double](l2Norm / l2NormThreshold)
      gradients.div(scale)
    }
  }
}

/**
 * Process lars
 */
private[bigdl] class LarsProcessor()
  extends ParameterProcessor {

  private var lookupDicBroadcast: Broadcast[Array[mutable.HashMap[Int, (Int, Int)]]]
    = null

  override def init[T](dataset: DistributedDataSet[MiniBatch[T]],
    parameters: AllReduceParameter[T],
    state: Table)(implicit ev: TensorNumeric[T]): Unit = {
    val partitionNum = dataset.originRDD().partitions.length
    val weights = state.get[Array[Tensor[T]]]("weights").get
    val parameterPerLayerSizes = weights.map(_.nElement())
    val parameterPerNodeSizes = state.get[Array[(Int, Int, Int)]]("parameterPerNodeSizes").get

    // stores each layers start, end on each partition
    // Array index is partition id, hashmap key is layer Id, hash value is (start, length)
    val lookupDic = new Array[mutable.HashMap[Int, (Int, Int)]](parameterPerNodeSizes.length)
    var i = 0
    require(parameterPerNodeSizes.length == partitionNum,
      "each partition shoule return its parameter meta data")

    var layerIndex = -1
    var parameterPerLayerLeftLen = 0
    while (i < parameterPerNodeSizes.length) {
      var start = 1
      var parameterPerNodeLeftLen = parameterPerNodeSizes(i)._3
      if (parameterPerLayerLeftLen == 0) {
        layerIndex += 1
        parameterPerLayerLeftLen
          = parameterPerLayerSizes(layerIndex)
      }
      val map = new mutable.HashMap[Int, (Int, Int)]()
      while (parameterPerNodeLeftLen > 0) {
        if (parameterPerNodeLeftLen <= parameterPerLayerLeftLen) {
          parameterPerLayerLeftLen -= parameterPerNodeLeftLen
          map(layerIndex) = (start, parameterPerNodeLeftLen)
          parameterPerNodeLeftLen = 0
        } else {
          map(layerIndex) = (start, parameterPerLayerLeftLen)
          parameterPerNodeLeftLen -= parameterPerLayerLeftLen
          start += parameterPerLayerLeftLen
          layerIndex += 1
          parameterPerLayerLeftLen = parameterPerLayerSizes(layerIndex)
        }
      }
      lookupDic(i) = map
      i += 1
    }

    // Check if lookupDic is correct
    // array element is a tuple, first element is layer id, value is (wNorm2, gNorm2)
    val norm2PerLayer = dataset.originRDD().mapPartitions( _ => {
      parameters.squareSumForRegion(lookupDic).iterator
    }).reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2)).sortByKey(true).collect()

    require(parameterPerLayerSizes.length == norm2PerLayer.length,
      "lookupDic length is not correct!")
    for((parameter, i) <- weights.view.zipWithIndex) {
      require(ev.nearlyEqual(parameter.sumSquare(), ev.fromType(norm2PerLayer(i)._2._1), 1e-3),
        s"lookupDic value is not correct! ori is ${parameter.sumSquare()}" +
          s" now is ${norm2PerLayer(i)._2._1}")
    }
    state("lookupDic") = lookupDic
  }

  override def collectGlobalData[T](models: RDD[Cache[T]],
    parameters: AllReduceParameter[T],
    metrics: Metrics,
    state: Table)(implicit ev: TensorNumeric[T]) : Unit = {
    val aggregatedG = state.get[Boolean]("aggregateG").get
    val numFinishedModel = state.get[Int]("numFinishedModel").get
    val lookupDic: Array[mutable.HashMap[Int, (Int, Int)]] =
      state.get[Array[mutable.HashMap[Int, (Int, Int)]]]("lookupDic").get

    if (lookupDicBroadcast == null) {
      lookupDicBroadcast = models.context.broadcast(lookupDic)
    }
    // gwNorm2List array element is a tuple, first element is layer id,
    // value is (wNorm2, gNorm2)
    val gwNorm2List = models.mapPartitions(_ => {
      if (!aggregatedG) {
        val getG = System.nanoTime()
        parameters.aggregateGradientPartition(numFinishedModel)
        metrics.add("aggregrateGradientParition average executor",
          System.nanoTime() - getG)
      }
      parameters.squareSumForRegion(lookupDicBroadcast.value).iterator
    }).reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))
      .sortByKey(true).collect().map(x => (x._1, (math.sqrt(x._2._1), math.sqrt(x._2._2))))
    state("gwNorm2") = gwNorm2List
    state("aggregateG") = true
  }

  override def processParameters[T](parameters: AllReduceParameter[T],
    modelCache: Cache[T],
    state: Table)(implicit ev: TensorNumeric[T]): Unit = {

    modelCache.optimMethod.state("gwNorm2List") = state("gwNorm2")
    modelCache.optimMethod.state("lookupList") =
      state[Array[mutable.HashMap[Int, (Int, Int)]]]("lookupDic")(parameters.partitionId)
  }

  override def processParameters[T](model: Module[T],
    state: Table)(implicit ev: TensorNumeric[T]): Unit = {
    throw new NotImplementedError("Currently don't support LARS for local optimizer")
  }
}
