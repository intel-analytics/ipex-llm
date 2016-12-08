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

import java.util.concurrent.{LinkedBlockingQueue, ThreadPoolExecutor, TimeUnit}

import com.intel.analytics.bigdl.dataset.DistributedDataSet
import com.intel.analytics.bigdl.nn.{Criterion, Module}
import com.intel.analytics.bigdl.ps.{AllReduceParameterManager, ParameterManager}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import org.apache.log4j.Logger
import org.apache.spark.Logging

import scala.collection.mutable
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.reflect.ClassTag

object DistriOptimizer {
  private var lossArray: Array[Double] = null
  private var recordsArray: Array[Int] = null

  val logger = Logger.getLogger(getClass)

  /**
   * Optimizer cache some metadata on each executor
   *
   * @param localModels cached models
   * @param modelWeights weights of the cached models
   * @param modelGradients gradients of the cached models
   * @param localCriterions cached criterion
   * @param localStates cached state
   * @param buffer tensor buffer
   * @tparam T
   */
  case class Cache[T](
    localModels: Array[Module[Activities, Activities, T]],
    modelWeights: Array[Tensor[T]],
    modelGradients: Array[Tensor[T]],
    localCriterions: Array[Criterion[Tensor[T], T]],
    localStates: Array[Table],
    buffer: Tensor[T]
  )
}

class DistriOptimizer[T: ClassTag](
  model: Module[Activities, Activities, T],
  criterion: Criterion[Tensor[T], T],
  optimMethod: OptimMethod[T],
  dataset: DistributedDataSet[(Tensor[T], Tensor[T])],
  endWhen: Trigger,
  nodeNumber: Int,
  coresPerNode: Int,
  state: Table = T())
  (implicit ev: TensorNumeric[T])
  extends Optimizer[T](model, endWhen) {

  val metrics = new Metrics

  def setParameterManager(pm: ParameterManager[T]): this.type = {
    this.pm = pm
    this
  }

  var pm: ParameterManager[T] =
    new AllReduceParameterManager[T](
      model.getParameters()._1,
      dataset.originRDD(),
      metrics,
      state)

  val sc = dataset.data().sparkContext

  import DistriOptimizer._

  private def initThreadModels() = {
    val broadcast = sc.broadcast((model, criterion, state))
    val _subModelNumber = Engine.getEngineType match {
      case MklBlas => coresPerNode
      case MklDnn => 1
    }

    require(dataset.originRDD().partitions.length == nodeNumber,
      s"Passed in rdd partition number ${dataset.originRDD().partitions.length}" +
        s" is not equal to configured node number ${nodeNumber}")

    val models = dataset.originRDD().mapPartitions(_ => {
      val (broadcastModel, broadcastCriterion, broadcastState) = broadcast.value
      require(Engine.checkSingleton(), "Detect multi-task run on one Executor/Container. " +
        "Currently not support this")
      val cached = (0 until _subModelNumber).map { _ =>
        val localModel = broadcastModel.cloneModule()
        val localCriterion = broadcastCriterion.cloneCriterion()
        val localState = broadcastState.clone()
        val (weights, grads) = localModel.getParameters()
        (localModel, weights, grads, localCriterion, localState)
      }.toArray
      Iterator(Cache(
        cached.map(_._1),  // models
        cached.map(_._2),  // weights
        cached.map(_._3),  // gradients
        cached.map(_._4),  // criterions
        cached.map(_._5),  // states
        cached.head._2.clone()  // a tensor buffer
      ))
    }).persist()
    models.setName("Thread Model RDD")
    logger.info("Cache thread models...")
    models.count()
    logger.info("Cache thread models... done")
    models
  }

  val models = initThreadModels()


  override def optimize(): Module[Activities, Activities, T] = {
    dataset.originRDD()
      .sparkContext
      .getConf
      .set("spark.task.cpus", coresPerNode.toString)

    // don't send whole Optimizer in closure
    // Todo: Move this method to object
    val broadcastEV = sc.broadcast(ev)
    val partitionNum = dataset.originRDD().partitions.length
    var wallClockTime = 0L
    val driverState = T("epoch" -> state.get[Int]("epoch").getOrElse(1),
      "neval" -> state.get[Int]("neval").getOrElse(1))
    val _subModelNumber = Engine.getEngineType match {
      case MklBlas => coresPerNode
      case MklBlas => 1
    }
    var accumulateCount = 0
    val shuffleBefore = System.nanoTime()
    logger.info(s"config $state")
    logger.info(s"Shuffle data")
    dataset.shuffle()
    val shuffleEnd = System.nanoTime()
    logger.info(s"Shuffle data complete. Takes ${(shuffleEnd - shuffleBefore) / 1e9}s")
    var epochStart = System.nanoTime()
    while (!endWhen(driverState)) {
      val _header = header(driverState[Int]("epoch"), accumulateCount, dataset.size(),
        driverState[Int]("neval"), wallClockTime)
      val lossSum = sc.accumulator(0.0, "loss sum")
      val recordsNum = sc.accumulator(0, "record number")
      val stackCount = sc.accumulator(0, "stack count")
      metrics.set("computing time for each node", mutable.ArrayBuffer[Double](), sc)
      metrics.set("init gradient time", 0.0, sc, partitionNum)
      metrics.set("construct tensor time", 0.0, sc, partitionNum)
      metrics.set("computing time average", 0.0, sc, partitionNum)
      metrics.set("prepare time", 0.0, sc, partitionNum)
      metrics.set("statics time", 0.0, sc, partitionNum)
      metrics.set("aggregate gradient time", 0.0, sc, partitionNum)

      val driverMetrics = metrics
      val start = System.nanoTime()
      val resultRDD = dataset.data().zipPartitions(
        models,
        pm.sync(models.mapPartitions(iter => Iterator.single(iter.next().buffer))), true)(
        (data, modelIter, weights) => {
          var time = System.nanoTime()

          val cached = modelIter.next()
          Engine.invoke(Seq(
            () => {
              weights.next() // Update local weights
              Engine.invoke(
                (0 until _subModelNumber).map(i =>
                  () => {
                    cached.modelWeights(i).copy(cached.buffer)
                  }
                )
              )
            }
          ))

          val localEV = broadcastEV.value
          val tensorBuffer = new Array[(Tensor[T], Tensor[T])](_subModelNumber)
          Engine.invoke(Seq(() => {
            val batch = data.next()
            var b = 0
            require(batch._1.size(1) == batch._2.size(1))
            require(batch._1.size(1) % _subModelNumber == 0)
            val stackSize = batch._1.size(1) / _subModelNumber
            while (b < _subModelNumber) {
              tensorBuffer(b) = (batch._1.narrow(1, b * stackSize + 1, stackSize),
                batch._2.narrow(1, b * stackSize + 1, stackSize))
              b += 1
            }
          }))
          Engine.wait()
          driverMetrics.add("prepare time", System.nanoTime() - time)

          if (lossArray == null || lossArray.length < _subModelNumber) {
            lossArray = new Array[Double](_subModelNumber)
          }

          if (recordsArray == null || recordsArray.length < _subModelNumber) {
            recordsArray = new Array[Int](_subModelNumber)
          }

          // ======================Start train models===================================
          time = System.nanoTime()
          Engine.invokeAndWait((0 until _subModelNumber).map(i =>
            () => {
              val localModel = cached.localModels(i)
              localModel.training()
              val localCriterion = cached.localCriterions(i)
              val (input, target) = tensorBuffer(i)
              val output = localModel.forward(input).asInstanceOf[Tensor[T]]
              lossArray(i) = localEV.toType[Double](localCriterion.forward(output, target))
              val errors = localCriterion.backward(output, target)
              localModel.backward(input, errors)
              recordsArray(i) = target.size(1)
            }))
          val computingTime = System.nanoTime() - time
          driverMetrics.add("computing time average", computingTime)
          driverMetrics.add("computing time for each node", computingTime)
          time = System.nanoTime()
          stackCount += tensorBuffer.size
          var i = 0
          while (i < _subModelNumber) {
            lossSum += lossArray(i)
            recordsNum += recordsArray(i)
            i += 1
          }
          driverMetrics.add("statics time", System.nanoTime() - time)

          time = System.nanoTime()
          val gradLength = cached.modelGradients(0).nElement()
          val taskSize = gradLength / _subModelNumber
          val extraTask = gradLength % _subModelNumber

          // copy multi-model gradient to the buffer
          val parallelNum = if (taskSize == 0) extraTask else _subModelNumber
          Engine.invokeAndWait((0 until parallelNum).map(tid => () => {
            val offset = tid * taskSize + math.min(tid, extraTask)
            val length = taskSize + (if (tid < extraTask) 1 else 0)
            var i = 0
            while (i < cached.modelGradients.length) {
              if (i == 0) {
                cached.buffer.narrow(1, offset + 1, length)
                  .copy(cached.modelGradients(i).narrow(1, offset + 1, length))
              } else {
                cached.buffer.narrow(1, offset + 1, length)
                  .add(cached.modelGradients(i).narrow(1, offset + 1, length))
              }
              i += 1
            }
          }))
          driverMetrics.add("aggregate gradient time", System.nanoTime() - time)

          Engine.invoke((0 until _subModelNumber).map(i => () => {
            cached.localModels(i).training()
            cached.localModels(i).zeroGradParameters()
          }))

          Iterator.single(cached.buffer)
        })
      val reduceBefore = System.nanoTime()
      val driverEV = ev
      val optM = optimMethod
      val driverParNum = partitionNum * _subModelNumber
      pm.sumAndUpdate(resultRDD, (weights, gradients, state) => {
        gradients.div(driverEV.fromType[Int](driverParNum))
        state("neval") = driverState[Int]("neval")
        state("epoch") = driverState[Int]("epoch")
        optM.optimize(_ => (driverEV.fromType(lossSum.value / stackCount.value), gradients),
          weights, state, state)
      })
      val reduceAfter = System.nanoTime()

      accumulateCount += recordsNum.value
      val end = System.nanoTime()
      logger.info(s"${_header} Train ${recordsNum.value} in ${(end - start) / 1e9}seconds. " +
        s"Throughput is ${recordsNum.value / ((end - start) / 1e9)} records/second. Loss is ${
          lossSum.value / stackCount.value
        }. " +
        s"Calculate time is ${(reduceBefore - start) / 1e9}seconds. ")
      logger.info("\n" + metrics.summary())
      driverState("neval") = driverState[Int]("neval") + 1
      if (accumulateCount >= dataset.size()) {
        val epochEnd = System.nanoTime()
        wallClockTime = wallClockTime + epochEnd - epochStart
        epochStart = System.nanoTime()
        logger.info(s"${_header} Epoch finished. Wall clock time is ${wallClockTime / 1e6}ms")

        driverState("epoch") = driverState[Int]("epoch") + 1
        dataset.shuffle()
        accumulateCount = 0
      }

      validate(wallClockTime, driverState)
      cache(wallClockTime, driverState)
    }
    validate(wallClockTime, driverState)
    cache(wallClockTime, driverState)

    getModel()
  }

  private def cache(wallClockTime: Long, state: Table): Unit = {
    cacheTrigger.foreach(trigger => {
      if (trigger(state) && cachePath.isDefined) {
        println(s"[Wall Clock ${wallClockTime / 1e9}s] Save model to ${cachePath.get}")
        saveModel(getModel(), s".${state[Int]("neval")}")
        saveState(pm.getState(), s".${state[Int]("neval")}")
      }
    })
  }

  private def validate(wallClockTime: Long, state: Table): Unit = {
    if(validationTrigger.isEmpty || validationDataSet.isEmpty) {
      return
    }
    val trigger = validationTrigger.get
    if(!trigger(state)) {
      return
    }
    val vMethods = validationMethods.get
    val validateRDD = validationDataSet.get.asInstanceOf[DistributedDataSet[(Tensor[T], Tensor[T])]]
      .data()
    logger.info(s"[Wall Clock ${wallClockTime / 1e9}s] Validate model...")
    val _subModelNumber = Engine.getEngineType match {
      case MklBlas => coresPerNode
      case MklDnn => 1
    }
    models.zipPartitions(validateRDD)((modelIter, dataIter) => {
      val workingModels = modelIter.next().localModels
      workingModels.foreach(_.evaluate())
      dataIter.map(batch => {
        require(batch._1.size(1) == batch._2.size(1))
        val stackSize = batch._1.size(1) / _subModelNumber
        val extraSize = batch._1.size(1) % _subModelNumber
        val parallelism = if(stackSize == 0) extraSize else _subModelNumber
        Engine.invokeAndWait(
          (0 until parallelism).map(b =>
            () => {
              val offset = b * stackSize + math.min(b, extraSize)
              val length = stackSize + (if(b < extraSize) 1 else 0)
              val input = batch._1.narrow(1, offset + 1, length)
              val target = batch._2.narrow(1, offset + 1, length)
              val output = workingModels(b).forward(input)
              vMethods.map(validation => {
                validation(output.asInstanceOf[Tensor[T]], target)
              })
            }
          )
        ).reduce((left, right) => {
          left.zip(right).map { case (l, r) =>
            l + r
          }
        })
      })
    }).reduce((left, right) => {
      left.zip(right).map { case (l, r) =>
        l + r
      }
    }).zip(vMethods).foreach(r => {
      logger.info(s"${r._2} is ${r._1}")
    })
  }

  private def getModel(): Module[Activities, Activities, T] = {
    val model = models.first().localModels.head
    val modelParameter = model.getParameters()._1
    modelParameter.copy(pm.getParameter())
    model
  }
}


