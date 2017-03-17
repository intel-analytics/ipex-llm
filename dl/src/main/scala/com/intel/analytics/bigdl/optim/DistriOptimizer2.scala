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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch}
import com.intel.analytics.bigdl.parameters.{CompressedTensor, ParameterManager2}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import org.apache.log4j.Logger
import org.apache.spark.{SparkEnv, TaskContext}
import org.apache.spark.rdd.{RDD, ZippedPartitionsWithLocalityRDD}

import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.concurrent.Future
import scala.reflect.ClassTag

object DistriOptimizer2 {
  import Optimizer._

  val logger = Logger.getLogger(getClass)

  /**
    * Optimizer cache some metadata on each executor
    *
    * @param localModels cached models
    * @param modelWeights weights of the cached models
    * @param modelGradients gradients of the cached models
    * @param localCriterions cached criterion
    * @param gradient tensor buffer
    * @tparam T
    */
  case class Cache[T](
    localModels: Array[Module[T]],
    modelWeights: Array[Tensor[T]],
    modelGradients: Array[Tensor[T]],
    localCriterions: Array[Criterion[T]],
    gradient: Tensor[T],
    var moduleTimeList: Array[Long] = null
  )

  private[optim] def optimize[T: ClassTag](
    dataset: DistributedDataSet[MiniBatch[T]],
    state: Table,
    endWhen: Trigger,
    metrics: Metrics,
    models: RDD[Cache[T]],
    optimMethod: OptimMethod[T],
    validationTrigger: Option[Trigger],
    validationDataSet: Option[DataSet[MiniBatch[T]]],
    validationMethods: Option[Array[ValidationMethod[T]]],
    cacheTrigger: Option[Trigger],
    cachePath: Option[String],
    isOverWrite: Boolean
  )(implicit ev: TensorNumeric[T]): Unit = {
    val sc = dataset.originRDD().sparkContext
    val partitionNum = dataset.originRDD().partitions.length
    
    var wallClockTime = 0L
    var lastEpochTime = 0L
    val driverState = T("epoch" -> state.get[Int]("epoch").getOrElse(1),
      "neval" -> state.get[Int]("neval").getOrElse(1))
    val _subModelNumber = Engine.getEngineType match {
      case MklBlas => Engine.coreNumber()
      case _ => throw new IllegalArgumentException()
    }
    var accumulateCount = 0
    val shuffleBefore = System.nanoTime()
    logger.info(s"config $state")
    logger.info(s"Shuffle data")
    dataset.shuffle()
    val shuffleEnd = System.nanoTime()
    logger.info(s"Shuffle data complete. Takes ${(shuffleEnd - shuffleBefore) / 1e9}s")

    var tasks: ArrayBuffer[Future[_]] = new ArrayBuffer()
    var threshold = Long.MaxValue
    var timeout = Long.MaxValue
    var iteration = 0
    val dropPercentage = state.get[Double]("dropPercentage").get
    val warmupIterationNum = state.get[Int]("warmupIterationNum").get
    val comupteThresholdbatchSize = state.get[Int]("comupteThresholdbatchSize").get
    val maxDropPercentage = state.get[Double]("maxDropPercentage").get
    val driverSubModelNum = partitionNum * _subModelNumber
    var dropModelNumBatch = 0
    var lossArray = new Array[Double](_subModelNumber)

    var epochStart = System.nanoTime()
    var dataRDD = dataset.data(train = true)
    
    val dummyRDD = models.coalesce(Engine.nodeNumber().get).mapPartitions { iter =>
      Iterator(1)
    }.cache()
    dummyRDD.count()
    
    while (!endWhen(driverState)) {
      val _header = header(driverState[Int]("epoch"), accumulateCount, dataset.size(),
        driverState[Int]("neval"), wallClockTime)
      val lossSum = sc.accumulator(0.0, "loss sum")
      val recordsNum = sc.accumulator(0, "record number")
      metrics.set("computing time average", 0.0, sc, partitionNum)
      metrics.set("aggregate gradient time", 0.0, sc, partitionNum)
      metrics.set("get weights average", 0.0, sc, partitionNum)
      metrics.set("send gradient partition", 0.0, sc, partitionNum)
      metrics.set("aggregate local gradient", 0.0, sc, Engine.nodeNumber().get)
      metrics.set("put gradient", 0.0, sc, Engine.nodeNumber().get)
      metrics.set("aggregrateGradientParition average executor", 0.0, sc, Engine.nodeNumber().get)
      metrics.set("compute weight average", 0.0, sc, Engine.nodeNumber().get)
      metrics.set("send weights average", 0.0, sc, Engine.nodeNumber().get)

      val driverMetrics = metrics
      val start = System.nanoTime()
      val finishedModelNum = dataRDD.zipPartitions(
        models, true)(
        (data, modelIter) => {
          val dd = System.nanoTime()
          val cached = modelIter.next()
          val executorId = SparkEnv.get.executorId
          val parameters = ParameterManager2.get(executorId)
          val syWStart = System.nanoTime()
          ParameterManager2.synchronized {
            if (!parameters.job1Start) {
              parameters.syncWeights(cached.modelWeights.head)
              parameters.job1Start = true
            }
          }
          val weightSyncTime = System.nanoTime() - syWStart
          driverMetrics.add("get weights average", weightSyncTime)

          val tensorBuffer = new Array[(Tensor[T], Tensor[T])](_subModelNumber)
          tasks += Engine.default.invoke(() => {
            val batch = data.next()
            var b = 0
            require(batch.data.size(1) == batch.labels.size(1),
              "data and label batch size not match")
            require(batch.data.size(1) >= _subModelNumber,
              "total batch size should be divided by spark.task.cpus")
            val stackSize = batch.data.size(1) / _subModelNumber
            while (b < _subModelNumber) {
              tensorBuffer(b) = (batch.data.narrow(1, b * stackSize + 1, stackSize),
                batch.labels.narrow(1, b * stackSize + 1, stackSize))
              b += 1
            }
          })
          Engine.default.sync(tasks)
          tasks.clear()

          // ======================Start train models===================================
          var time = System.nanoTime()
          if(dropPercentage > 0 && iteration > warmupIterationNum + comupteThresholdbatchSize - 1) {
            timeout = threshold - weightSyncTime
          }
          val pre = (iteration % comupteThresholdbatchSize) * _subModelNumber

          val trainingThreads = Engine.default.invokeAndWait2((0 until _subModelNumber).map(i =>
            () => {
              val trainStart = System.nanoTime()
              val localModel = cached.localModels(i)
              localModel.training()
              val localCriterion = cached.localCriterions(i)
              val (input, target) = tensorBuffer(i)
              val output = localModel.forward(input)
              lossArray(i) = ev.toType[Double](localCriterion.forward(output, target))
              val errors = localCriterion.backward(output, target)
              localModel.backward(input, errors)
              cached.moduleTimeList(i + pre) = System.nanoTime() - trainStart + weightSyncTime
              i
            }
          ), timeout)
          val computingTime = System.nanoTime() - time
          driverMetrics.add("computing time average", computingTime)

          time = System.nanoTime()
          val finishedThreads = trainingThreads.filter(!_.isCancelled).map(_.get())
          recordsNum += finishedThreads.size * tensorBuffer.head._2.size(1)
          var i = 0
          while (i < finishedThreads.size) {
            lossSum += lossArray(finishedThreads(i))
            i += 1
          }

          if (finishedThreads.size > 0) {
            val gradLength = cached.modelGradients(0).nElement()
            val taskSize = gradLength / _subModelNumber
            val extraTask = gradLength % _subModelNumber

            (0 until _subModelNumber).diff(finishedThreads).foreach(i =>
              cached.modelGradients(i).zero()
            )

            // copy multi-model gradient to the buffer
            val parallelNum = if (taskSize == 0) extraTask else _subModelNumber
            Engine.default.invokeAndWait((0 until parallelNum).map(tid => () => {
              val offset = tid * taskSize + math.min(tid, extraTask)
              val length = taskSize + (if (tid < extraTask) 1 else 0)
              var i = 0
              while (i < cached.modelGradients.length) {
                if (i == 0) {
                  cached.gradient.narrow(1, offset + 1, length)
                    .copy(cached.modelGradients(i).narrow(1, offset + 1, length))
                } else {
                  cached.gradient.narrow(1, offset + 1, length)
                    .add(cached.modelGradients(i).narrow(1, offset + 1, length))
                }
                i += 1
              }
            }))
          }
          tasks ++= Engine.default.invoke((0 until _subModelNumber).map(i => () => {
            cached.localModels(i).training()
            cached.localModels(i).zeroGradParameters()
          }))
          driverMetrics.add("aggregate gradient time", System.nanoTime() - time)
          
          time = System.nanoTime()
          parameters.sendGradientPartition(cached.gradient, TaskContext.getPartitionId())
          driverMetrics.add("send gradient partition", System.nanoTime() - time)
          
          Iterator(finishedThreads.size)
        }).reduce(_ + _)

      val job2start = System.nanoTime()
      dummyRDD.mapPartitions { iter =>
        val executorId = SparkEnv.get.executorId
        val parameters = ParameterManager2.get(executorId)
        var t = System.nanoTime()
        val gradient = parameters.aggregateLocalGradient()
        driverMetrics.add("aggregate local gradient", System.nanoTime() - t)

        t = System.nanoTime()
        parameters.putGradients(gradient)
        driverMetrics.add("put gradient", System.nanoTime() - t)
        parameters.job1Start = false
        Iterator.empty
      }.count()
      val job2end = System.nanoTime()
            
      dropModelNumBatch += (driverSubModelNum - finishedModelNum)
      if (dropPercentage == 0 || finishedModelNum >= driverSubModelNum * (1-maxDropPercentage)) {
        val value = lossSum.value / finishedModelNum

        val nodeNumber = Engine.nodeNumber().get

        val job3start = System.nanoTime()
        dummyRDD.mapPartitions { iter =>
          val executorId = SparkEnv.get.executorId
          val parameters = ParameterManager2.get(executorId)
          val params = new Array[CompressedTensor[T]](nodeNumber)
          val getG = System.nanoTime()
          parameters.aggregrateGradientParition(params)
          driverMetrics.add("aggregrateGradientParition average executor",
            System.nanoTime() - getG)

          var time = System.nanoTime()
          val gradients = parameters.getLocalParameter[T](parameters.getGradientExecutorId())
          gradients.div(ev.fromType(finishedModelNum))
          val weights = parameters.getLocalParameter[T](parameters.getWeightExecutorId())
          val state = parameters.getState()
          state("neval") = driverState[Int]("neval")
          state("epoch") = driverState[Int]("epoch")
          optimMethod.optimize(_ => (ev.fromType(value), gradients),
            weights, state, state)
          driverMetrics.add("compute weight average", System.nanoTime() - time)
          time = System.nanoTime()
          parameters.sendWeightExecutor()
          driverMetrics.add("send weights average", System.nanoTime() - time)
          Iterator.empty
        }.count()
        val job3end = System.nanoTime()

        accumulateCount += recordsNum.value
        val end = System.nanoTime()
        wallClockTime += end - start
        logger.debug(s"job1 driver: ${(job2start-start)/1e9} " +
          s"job2 driver: ${(job2end-job2start)/1e9} job3 driver: ${(job3end-job3start)/1e9}")
        logger.info(s"${_header} Train ${recordsNum.value} in ${(end - start) / 1e9}seconds. " +
          s"Throughput is ${recordsNum.value / ((end - start) / 1e9)} records/second. Loss is ${
            lossSum.value / finishedModelNum
          }. ")

        logger.debug("\n" + metrics.summary())
        logger.debug("Dropped modules: " + (driverSubModelNum - finishedModelNum))
        lossArray = new Array[Double](_subModelNumber)

        // compute threshold
        iteration += 1
        if (dropPercentage > 0 && iteration > warmupIterationNum &&
          iteration % comupteThresholdbatchSize == 0) {
          val moduleTimeList = models.mapPartitions { iter =>
            iter.next().moduleTimeList.iterator
          }.collect()

          val k = (dropPercentage * comupteThresholdbatchSize * driverSubModelNum).toInt
          if (k > dropModelNumBatch) {
            threshold = Util.kthLargest(moduleTimeList, 0, moduleTimeList.length-1,
              k - dropModelNumBatch)
          } else {
            threshold = (threshold * 1.01).toLong
          }
          logger.info("threshold: " + threshold)

          // clear moduleTimeList in each node
          models.mapPartitions { iter =>
            val timeList = iter.next.moduleTimeList
            var i = 0
            while (i < timeList.length) {
              timeList(i) = 0
              i += 1
            }
            Iterator.empty
          }.count()
          dropModelNumBatch = 0
        }

        driverState("neval") = driverState[Int]("neval") + 1
        if (accumulateCount >= dataset.size()) {
          val epochEnd = System.nanoTime()
          wallClockTime = lastEpochTime + epochEnd - epochStart
          lastEpochTime = wallClockTime
          epochStart = System.nanoTime()
          logger.info(s"${_header} Epoch finished. Wall clock time is ${wallClockTime / 1e6}ms")

          driverState("epoch") = driverState[Int]("epoch") + 1
          dataset.shuffle()
          dataRDD = dataset.data(train = true)
          accumulateCount = 0
        }
        validate(
          validationTrigger,
          validationDataSet,
          validationMethods,
          models,
          wallClockTime,
          driverState
        )

        checkpoint(
          cacheTrigger,
          cachePath,
          isOverWrite,
          wallClockTime,
          models,
          driverState
        )
      } else {
        logger.info(s"Warning!!! Ignore this iteration as more than maxDropPercentage " +
          s"module is dropped!! Finished modules number: ${finishedModelNum}")
      }
    }
  }


  private def checkpoint[T: ClassTag](
    cacheTrigger: Option[Trigger],
    cachePath: Option[String],
    isOverWrite: Boolean,
    wallClockTime: Long,
    models: RDD[Cache[T]],
    state: Table)
  : Unit = {
    if (cacheTrigger.isDefined) {
      val trigger = cacheTrigger.get
      if (trigger(state) && cachePath.isDefined) {
        println(s"[Wall Clock ${wallClockTime / 1e9}s] Save model to ${cachePath.get}")
        saveModel(getModel(models), cachePath, isOverWrite,
          s".${state[Int]("neval")}")
        val s = models.mapPartitions { iter =>
          val executorId = SparkEnv.get.executorId
          val parameter = ParameterManager2.get(executorId)
          Iterator(parameter.getState())
        }.first()
        saveState(s, cachePath, isOverWrite, s"" +
          s".${state[Int]("neval")}")
      }
    }
  }

  private def initThreadModels[T: ClassTag](
    model: Module[T],
    dataset: DistributedDataSet[MiniBatch[T]],
    criterion: Criterion[T],
    state: Table,
    nodeNumber: Int,
    checkSingleton: Boolean
  )(implicit ev: TensorNumeric[T]) = {
    val sc = dataset.originRDD().sparkContext
    val _subModelNumber = Engine.getEngineType match {
      case MklBlas => Engine.coreNumber()
      case _ => throw new IllegalArgumentException
    }
    val partitionNum = dataset.originRDD().partitions.length
    val comupteThresholdbatchSize = state.get[Int]("comupteThresholdbatchSize").get
    val parameterSize = model.getParameters()._1.nElement()

    val executorIdList = dataset.originRDD().mapPartitions { iter =>
      synchronized {
        Iterator(SparkEnv.get.executorId)
      }
    }.collect().distinct

    require(executorIdList.size == nodeNumber)
    val executorIdMap = new HashMap[String, Int]()
    var i = 0
    while (i < nodeNumber) {
      executorIdMap(executorIdList(i)) = i
      i += 1
    }
    val driver = SparkEnv.get.executorId
    if (!executorIdMap.contains(driver)) {
      executorIdMap(driver) = nodeNumber
    }

    ParameterManager2.createParameterManager(executorIdMap(driver), nodeNumber,
      partitionNum, parameterSize, true)
    
    val broadcast = sc.broadcast((model, criterion, state))
    val models = dataset.originRDD().mapPartitions(_ => {
      val (broadcastModel, broadcastCriterion, broadcastState)
      = broadcast.value
      if (!Engine.checkSingleton()) {
        if (checkSingleton) {
          require(Engine.checkSingleton(), "Detect multi-task run on one Executor/Container. " +
            "Please disable singleton check or repartition data")
        } else {
          logger.warn("Detect multi-task run on one Executor/Container.")
        }
      }
      val cached = (0 until _subModelNumber).map { _ =>
        val localModel = broadcastModel.cloneModule()
        val localCriterion = broadcastCriterion.cloneCriterion()
        val (weights, grads) = localModel.getParameters()
        (localModel, weights, grads, localCriterion)
      }.toArray
      logger.info("model thread pool size is " + Engine.model.getPoolSize)

      // init parameter manager
      val weights = cached.head._2
      
      val executorId = SparkEnv.get.executorId
      executorIdMap.foreach(x => println(s"key: ${x._1} value: ${x._2}"))
      ParameterManager2.synchronized {
        ParameterManager2.setExecutorMap(executorIdMap)
        var parameter = ParameterManager2.get(executorId)
        if (parameter == null) {
          parameter = ParameterManager2.createParameterManager(executorIdMap(executorId),
            nodeNumber, partitionNum, parameterSize, false)
        }
        if (!parameter.initFinished) {
          parameter.init(weights, broadcastState)
          parameter.initFinished = true
        }
        cached.map { c =>
          val blockId = parameter.getWeightId()
          c._2.storage().set(parameter.getLocalParameter[T](blockId).storage())
        }
      }

      Iterator(Cache(
        cached.map(_._1), // models
        cached.map(_._2), // weights
        cached.map(_._3), // gradients
        cached.map(_._4), // criterions
        cached.head._2.clone(), // a tensor buffer
        new Array[Long](_subModelNumber * comupteThresholdbatchSize)
      ))
    }).persist()
    models.setName("Thread Model RDD")
    logger.info("Cache thread models...")
    models.count()
    logger.info("Cache thread models... done")
    models
  }


  private def validate[T](
    validationTrigger: Option[Trigger],
    validationDataSet: Option[DataSet[MiniBatch[T]]],
    validationMethods: Option[Array[ValidationMethod[T]]],
    models: RDD[Cache[T]],
    wallClockTime: Long,
    state: Table
  ): Unit = {
    if (validationTrigger.isEmpty || validationDataSet.isEmpty) {
      return
    }
    val trigger = validationTrigger.get
    if (!trigger(state)) {
      return
    }
    val vMethods = validationMethods.get
    val validateRDD = validationDataSet.get.toDistributed().data(train = false)
    logger.info(s"[Wall Clock ${wallClockTime / 1e9}s] Validate model...")
    val _subModelNumber = Engine.getEngineType match {
      case MklBlas => Engine.coreNumber()
      case _ => throw new IllegalArgumentException
    }
    ZippedPartitionsWithLocalityRDD(models, validateRDD)((modelIter, dataIter) => {
      val cached = modelIter.next()
      val criterion = cached.localCriterions
      val workingModels = cached.localModels

      workingModels.foreach(_.evaluate())
      dataIter.map(batch => {
        require(batch.data.size(1) == batch.labels.size(1))
        val stackSize = batch.data.size(1) / _subModelNumber
        val extraSize = batch.data.size(1) % _subModelNumber
        val parallelism = if (stackSize == 0) extraSize else _subModelNumber
        Engine.default.invokeAndWait(
          (0 until parallelism).map(b =>
            () => {
              val offset = b * stackSize + math.min(b, extraSize)
              val length = stackSize + (if (b < extraSize) 1 else 0)
              val input = batch.data.narrow(1, offset + 1, length)
              val target = batch.labels.narrow(1, offset + 1, length)
              val output = workingModels(b).forward(input)
              vMethods.map(validation => {
                validation(output, target, criterion(b))
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

  private def getModel[T: ClassTag](models: RDD[Cache[T]])
  : Module[T] = {
    val trainedModel = models.map(_.localModels.head.clearState()).first()
    val weights = models.mapPartitions(iter => {
      val cached = iter.next()
      val executorId = SparkEnv.get.executorId
      val parameter = ParameterManager2.get(executorId)
      Iterator.single(Map(parameter.executorId ->
        parameter.getLocalParameter[T](parameter.getWeightExecutorId())))
    }).reduce(_ ++ _)
    
    val parameter = trainedModel.getParameters()._1
    val parameterLength = parameter.nElement()
    
    val taskSize = parameterLength / weights.size
    val extraSize = parameterLength % weights.size

    (0 until weights.size).map(pid => {
      val start = pid * taskSize + math.min(pid, extraSize)
      val length = taskSize + (if (pid < extraSize) 1 else 0)
      parameter.narrow(1, start + 1, length).copy(weights(pid))
    })

    trainedModel
  }
}

class DistriOptimizer2[T: ClassTag](
  model: Module[T],
  dataset: DistributedDataSet[MiniBatch[T]],
  criterion: Criterion[T]
  )(implicit ev: TensorNumeric[T])
  extends Optimizer[T, MiniBatch[T]](
    model, dataset, criterion) {

  val metrics = new Metrics

  private var models: RDD[DistriOptimizer2.Cache[T]] = null

  override def optimize(): Module[T] = {
    this.assertEngineInited()

    optimMethod.clearHistory(state)
    state("dropPercentage") = dropPercentage
    state("warmupIterationNum") = warmupIterationNum
    state("comupteThresholdbatchSize") = computeThresholdbatchSize
    state("maxDropPercentage") = maxDropPercentage
    
    val actualNodeNumber = dataset.originRDD().mapPartitions { iter =>
      Iterator(SparkEnv.get.executorId)
    }.collect().distinct.size
    Engine.setNodeNumber(Some(actualNodeNumber))
    dataset.originRDD().sparkContext.getConf.setExecutorEnv("DL_NODE_NUMBER",
      actualNodeNumber.toString)

    models = DistriOptimizer2.initThreadModels(model, dataset, criterion, state,
      actualNodeNumber, checkSingleton)

    DistriOptimizer2.optimize(
      dataset,
      state,
      endWhen,
      metrics,
      models,
      optimMethod,
      validationTrigger,
      validationDataSet,
      validationMethods,
      checkpointTrigger,
      checkpointPath,
      isOverWrite
    )

    DistriOptimizer2.getModel(models)
  }
}


