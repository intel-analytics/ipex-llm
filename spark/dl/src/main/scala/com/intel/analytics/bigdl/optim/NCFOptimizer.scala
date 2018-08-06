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

import java.lang.invoke.MethodHandles.Lookup

import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.example.recommendation.NeuralCFV2
import com.intel.analytics.bigdl.nn.{Graph, LookupTable, Utils}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.{SparseTensor, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import org.apache.log4j.Logger

import scala.collection.mutable
import scala.reflect.ClassTag

/**
 * Optimize a model on a single machine
 *
 * @param model model to be optimized
 * @param dataset data set
 * @param criterion criterion to be used
 */
class NCFOptimizer[T: ClassTag] (
  model: Module[T],
  dataset: LocalDataSet[MiniBatch[T]],
  criterion: Criterion[T]
)(implicit ev: TensorNumeric[T])
  extends Optimizer[T, MiniBatch[T]](
    model, dataset, criterion) {

  import NCFOptimizer._
  import Optimizer._

  private val coreNumber = Engine.coreNumber()

  private val subModelNumber = Engine.getEngineType match {
    case MklBlas => coreNumber
    case _ => throw new IllegalArgumentException
  }

  val ncfModel = model.asInstanceOf[NeuralCFV2[T]]

  // TODO: sharing failed
  private val workingEmbeddingModels = initModel(ncfModel.embeddingModel,
    subModelNumber, true)
  private val workingLinears = initModel(ncfModel.ncfLayers,
    subModelNumber, false)

  workingEmbeddingModels(0).parameters()._2.apply(0).setValue(1, 1, ev.fromType(0.01f))
  workingEmbeddingModels(0).parameters()._1.apply(0).setValue(1, 1, ev.fromType(1.01f))

  workingEmbeddingModels(0).getParameters()._2.setValue(1, ev.fromType(0.1f))
  workingEmbeddingModels(0).getParameters()._1.setValue(1, ev.fromType(1.1f))
  private val (embeddingWeight, embeddingGrad) = ncfModel.embeddingModel.getParameters()
  private val (linearsWeight, linearsGrad) = ncfModel.ncfLayers.getParameters()
  workingEmbeddingModels(0).getParameters()._2.setValue(1, ev.fromType(0.2f))
  workingEmbeddingModels(0).getParameters()._1.setValue(1, ev.fromType(1.2f))

  private val workingEmbeddingModelWAndG = workingEmbeddingModels.map(_.getParameters())
  private val workingLinearModelWAndG = workingLinears.map(_.getParameters())

  private val linearGradLength = linearsGrad.nElement()
  private val linearSyncGradTaskSize = linearGradLength / subModelNumber
  private val linearSyncGradExtraTask = linearGradLength % subModelNumber
  private val linearSyncGradParallelNum =
    if (linearSyncGradTaskSize == 0) linearSyncGradExtraTask else subModelNumber

  private val embeddingGradLength = embeddingGrad.nElement()
  private val embeddingSyncGradTaskSize = embeddingGradLength / subModelNumber
  private val embeddingSyncGradExtraTask = embeddingGradLength % subModelNumber
  private val embeddingSyncGradParallelNum =
    if (embeddingSyncGradTaskSize == 0) linearSyncGradExtraTask else subModelNumber

  private val workingCriterion =
    (1 to subModelNumber).map(_ => criterion.cloneCriterion()).toArray

  override def optimize(): Module[T] = {
    var wallClockTime = 0L
    var count = 0
    optimMethods.values.foreach { optimMethod =>
      optimMethod.clearHistory()
    }
    state("epoch") = state.get[Int]("epoch").getOrElse(1)
    state("neval") = state.get[Int]("neval").getOrElse(1)
    state("isLayerwiseScaled") = Utils.isLayerwiseScaled(model)
    val optimMethod: OptimMethod[T] = optimMethods(model.getName())
    val optimMethod2: Map[String, OptimMethod[T]] = Map(
    "mlpUserEmbedding" -> new EmbeddingAdam[T]().setNOutput(138493, 128),
    "mlpItemEmbedding" -> new EmbeddingAdam[T]().setNOutput(26744, 128),
    "mfUserEmbedding" -> new EmbeddingAdam[T]().setNOutput(138493, 64),
    "mfItemEmbedding" -> new EmbeddingAdam[T]().setNOutput(26744, 64)
    )
    dataset.shuffle()
    val numSamples = dataset.data(train = false).map(_.size()).reduce(_ + _)
    var iter = dataset.data(train = true)
    logger.info("model thread pool size is " + Engine.model.getPoolSize)
    val userMapping = new mutable.HashMap[Int, Int]()
    val itemMapping = new mutable.HashMap[Int, Int]()
    while (!endWhen(state)) {
      val start = System.nanoTime()
      println("start")

      val tasks = Engine.default.invoke((0 until embeddingSyncGradParallelNum).map(tid =>
        () => {
          val offset = tid * embeddingSyncGradTaskSize + math.min(tid, embeddingSyncGradExtraTask)
          val length = embeddingSyncGradTaskSize +
            (if (tid < embeddingSyncGradExtraTask) 1 else 0)
          embeddingGrad.narrow(1, offset + 1, length).zero()
        }))

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
      println("dataFetch")

      val input = batch.getInput().toTensor[Float]
      val inputUser = input.select(2, 1)
      val inputItem = input.select(2, 2)

      def getMapping(input: Tensor[Float],
                     userMapping: mutable.HashMap[Int, Int],
                     itemMapping: mutable.HashMap[Int, Int]): Unit = {
        val inputArray = input.storage().array()
        var userCount = 1
        var itemCount = 1
        var i = 0
        while(i < input.nElement()) {
          val userIndex = inputArray(i).toInt
          if (!userMapping.contains(userIndex)) {
            userMapping.put(userIndex, userCount)
            userCount += 1
          }
          val itemIndex = inputArray(i + 1).toInt
          if (!itemMapping.contains(itemIndex)) {
            itemMapping.put(itemIndex, itemCount)
            itemCount += 1
          }
          i += 2
        }
      }

      userMapping.clear()
      itemMapping.clear()
      getMapping(inputUser, userMapping, itemMapping)

      val getMappingTime = System.nanoTime()

      val lossSum = Engine.default.invokeAndWait(
        (0 until parallelism).map(i =>
          () => {
            val localEmbedding = workingEmbeddingModels(i)
            val localLinears = workingLinears(i)
//            localEmbedding.zeroGradParameters()
            localEmbedding.training()
            localLinears.training()
            localLinears.zeroGradParameters()
            val localCriterion = workingCriterion(i)
            val input = miniBatchBuffer(i).getInput()
            val target = miniBatchBuffer(i).getTarget()

            val embeddingOutput = localEmbedding.forward(input)
            val output = localLinears.forward(embeddingOutput)
            val _loss = ev.toType[Double](localCriterion.forward(output, target))
            val errors = localCriterion.backward(output, target)
            localEmbedding.updateGradInput(input,
              localLinears.backward(localEmbedding.output, errors))
            _loss
          })
      ).sum

      val loss = lossSum / parallelism

      val computingTime = System.nanoTime()
      println("computingTime")

      Engine.default.sync(tasks)
//      Engine.default.invokeAndWait(
//        (0 until embeddingSyncGradParallelNum).map(tid =>
//          () => {
//            val offset = tid * embeddingSyncGradTaskSize
//            + math.min(tid, embeddingSyncGradExtraTask)
//            val length = embeddingSyncGradTaskSize +
//              (if (tid < embeddingSyncGradExtraTask) 1 else 0)
//            embeddingGrad.narrow(1, offset + 1, length).zero()
//          })
//      )

      val zeroGradTime = System.nanoTime()
      println("zeroGrad")

      val embeddingGradients = (0 until parallelism).flatMap{ i =>
        val localEmbedding = workingEmbeddingModels(i).asInstanceOf[Graph[T]]
        val input1 = localEmbedding("userId").get.output.asInstanceOf[Tensor[T]]
        val input2 = localEmbedding("itemId").get.output.asInstanceOf[Tensor[T]]
        val mlpUserEmbedding = localEmbedding("mlpUserEmbedding").get
          .asInstanceOf[LookupTable[T]]
        val mlpItemEmbedding = localEmbedding("mlpItemEmbedding").get
          .asInstanceOf[LookupTable[T]]
        val mfUserEmbedding = localEmbedding("mfUserEmbedding").get
          .asInstanceOf[LookupTable[T]]
        val mfItemEmbedding = localEmbedding("mfItemEmbedding").get
          .asInstanceOf[LookupTable[T]]
        val localLinears = workingLinears(i)
        Iterator(
          (mlpUserEmbedding.gradWeight, userMapping,
            input1, localLinears.gradInput.toTable[Tensor[T]](1)),
          (mlpItemEmbedding.gradWeight, itemMapping,
            input2, localLinears.gradInput.toTable[Tensor[T]](2)),
          (mfUserEmbedding.gradWeight, userMapping,
            input1, localLinears.gradInput.toTable[Tensor[T]](3)),
          (mfItemEmbedding.gradWeight, itemMapping,
            input2, localLinears.gradInput.toTable[Tensor[T]](4)))
      }.groupBy(_._1).map(v => (v._1, v._2.map(v2 => (v2._3, v2._4)).toArray))

      def mergeEmbeddingGrad(inputGrad: Array[(Tensor[T], Tensor[T])],
                             gradMap: mutable.HashMap[Int, Int],
                             embeddingGradient: Tensor[T]): Unit = {
        var parallelism = 0
        while(parallelism < inputGrad.length) {
          val indices = inputGrad(parallelism)._1
          val grad = inputGrad(parallelism)._2
          var i = 1
          while(i <= indices.nElement()) {
            val currentIndex = indices.valueAt(i)
            val position = gradMap(currentIndex)
            embeddingGradient.select(1, position).add(grad.select(1, i))
            i += 1
          }
        }
        parallelism += 1
      }

      val mlpUser = workingEmbeddingModels.map{m =>
        m("mlpUserEmbedding").get
          .asInstanceOf[AbstractModule[Activity, Activity, T]]
      }

      val computingTime2 = System.nanoTime()
      println("computing2")


      // copy multi-model gradient to the buffer
      Engine.default.invokeAndWait(
        (0 until linearSyncGradParallelNum).map(tid =>
          () => {
            val offset = tid * linearSyncGradTaskSize + math.min(tid, linearSyncGradExtraTask)
            val length = linearSyncGradTaskSize + (if (tid < linearSyncGradExtraTask) 1 else 0)
            var i = 0
            while (i < parallelism) {
              if (i == 0) {
                linearsGrad.narrow(1, offset + 1, length)
                  .copy(workingLinearModelWAndG(i)._2.narrow(1, offset + 1, length))
              } else {
                linearsGrad.narrow(1, offset + 1, length)
                  .add(workingLinearModelWAndG(i)._2.narrow(1, offset + 1, length))
              }
              i += 1
            }
          })
      )

      val aggTime = System.nanoTime()
      println("agg")

      optimMethod.state.update("epoch", state.get("epoch"))
      optimMethod.state.update("neval", state.get("neval"))
      optimMethod.optimize(_ => (ev.fromType(loss), linearsGrad), linearsWeight)

      val updateWeightTime1 = System.nanoTime()

      optimMethod2.foreach{v =>
        val moduleName = v._1
        val optimMethod = v._2
        val embedding = model(moduleName).get
          .asInstanceOf[AbstractModule[Activity, Activity, T]]
      }

//      optimMethod2.state.update("epoch", state.get("epoch"))
//      optimMethod2.state.update("neval", state.get("neval"))
//      optimMethod2.optimize(_ => (ev.fromType(loss), embeddingGrad), embeddingWeight)


      val updateWeightTime2 = System.nanoTime()
      println("update weight")
      val end = System.nanoTime()
      wallClockTime += end - start
      count += batch.size()
      val head = header(state[Int]("epoch"), count, numSamples, state[Int]("neval"), wallClockTime)
      logger.info(s"$head " +
        s"loss is $loss, iteration time is ${(end - start) / 1e9}s " +
        s"train time ${(end - dataFetchTime) / 1e9}s. " +
        s"Throughput is ${batch.size().toDouble / (end - start) * 1e9} record / second. " +
        optimMethod.getHyperParameter()
        )
      logger.debug(s"data fetch time is ${(dataFetchTime - start) / 1e9}s \n" +
        s"get mapping time is ${(getMappingTime - dataFetchTime) / 1e9}s \n" +
        s"model computing time is ${(computingTime - getMappingTime) / 1e9}s \n" +
        s"zero grad time is ${(zeroGradTime - computingTime) / 1e9}s \n" +
        s"acc embedding time is ${(computingTime2 - zeroGradTime) / 1e9}s \n" +
        s"aggregate linear is ${(aggTime - computingTime2) / 1e9}s \n" +
        s"update linear time is ${(updateWeightTime1 - aggTime) / 1e9}s" +
        s"update embedding time is ${(updateWeightTime2 - updateWeightTime1) / 1e9}s")

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

    model
  }

  private def checkpoint(wallClockTime: Long): Unit = {
    if (checkpointTrigger.isEmpty || checkpointPath.isEmpty) {
      return
    }

    val trigger = checkpointTrigger.get
    if (trigger(state) && checkpointPath.isDefined) {
      logger.info(s"[Wall Clock ${wallClockTime / 1e9}s] Save model to path")
      saveModel(model, checkpointPath, isOverWrite, s".${state[Int]("neval")}")
      saveState(state, checkpointPath, isOverWrite, s".${state[Int]("neval")}")
    }
  }

  private def validate(header: String): Unit = {
//    if (validationTrigger.isEmpty || validationDataSet.isEmpty) {
//      return
//    }
//    val trigger = validationTrigger.get
//    if (!trigger(state)) {
//      return
//    }
//    val vMethods = validationMethods.get
//    val vMethodsArr = (1 to subModelNumber).map(i => vMethods.map(_.clone())).toArray
//    val dataIter = validationDataSet.get.toLocal().data(train = false)
//    logger.info(s"$header Validate model...")
//
//    workingModels.foreach(_.evaluate())
//
//    var count = 0
//    dataIter.map(batch => {
//      val stackSize = batch.size() / subModelNumber
//      val extraSize = batch.size() % subModelNumber
//      val parallelism = if (stackSize == 0) extraSize else subModelNumber
//      val start = System.nanoTime()
//      val result = Engine.default.invokeAndWait(
//        (0 until parallelism).map(b =>
//          () => {
//            val offset = b * stackSize + math.min(b, extraSize) + 1
//            val length = stackSize + (if (b < extraSize) 1 else 0)
//            val currentMiniBatch = batch.slice(offset, length)
//            val input = currentMiniBatch.getInput()
//            val target = currentMiniBatch.getTarget()
//            val output = workingModels(b).forward(input)
//            val validatMethods = vMethodsArr(b)
//            validatMethods.map(validation => {
//              validation(output, target)
//            })
//          }
//        )
//      ).reduce((left, right) => {
//        left.zip(right).map { case (l, r) =>
//          l + r
//        }
//      })
//      count += batch.size()
//      logger.info(s"$header Throughput is ${
//        batch.size() / ((System.nanoTime() - start) / 1e9)
//      } record / sec")
//      result
//    }).reduce((left, right) => {
//      left.zip(right).map { case (l, r) =>
//        l + r
//      }
//    }).zip(vMethods).foreach(r => {
//      logger.info(s"$header ${r._2} is ${r._1}")
//    })
  }
}

object NCFOptimizer {
  val logger = Logger.getLogger(this.getClass)

  def initModel[T: ClassTag](model: Module[T], copies: Int,
                             shareGradient: Boolean)(
      implicit ev: TensorNumeric[T]): Array[Module[T]] = {
    val (wb, grad) = Util.getAndClearWeightBiasGrad(model.parameters())

    val models = (1 to copies).map(i => {
      logger.info(s"Clone $i model...")
      val m = if (i == copies) {
        model
      } else {
        model.cloneModule()
      }
      Util.putWeightBias(wb, m)
      if (shareGradient) {
        Util.putGradWeightBias(grad, m)
      } else {
        Util.initGradWeightBias(wb, m)
      }
      m
    }).toArray
    models
  }
}

