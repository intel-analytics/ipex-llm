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

import com.intel.analytics.bigdl.{Module, _}
import com.intel.analytics.bigdl.dataset.{DataSet, DistributedDataSet, MiniBatch, PaddingParam, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.nn.{Container, Module, Utils}
import com.intel.analytics.bigdl.parameters.{AllReduceParameter, ParameterProcessor}
import com.intel.analytics.bigdl.nn.{Container, Module, Utils}
import com.intel.analytics.bigdl.parameters.AllReduceParameter
import com.intel.analytics.bigdl.tensor.{FloatType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import java.io.{File, FilenameFilter}
import java.text.SimpleDateFormat
import java.util.Calendar

import com.intel.analytics.bigdl.models.utils.{CachedModels, ModelBroadcast}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.MklDnnContainer
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import org.apache.commons.lang.exception.ExceptionUtils
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import org.apache.log4j.Logger
import org.apache.spark.network.netty.SparkTransportConf
import org.apache.spark.{SparkContext, TaskContext}
import org.apache.spark.rdd.{RDD, ZippedPartitionsWithLocalityRDD}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Future
import scala.reflect.ClassTag

object DistriOptimizer {
  import Optimizer._

  val logger: Logger = Logger.getLogger(getClass)

  /**
   * Optimizer cache some metadata on each executor
   *
   * @param localModels cached models
   * @param modelWeights weights of the cached models
   * @param modelGradients gradients of the cached models
   * @param localCriterions cached criterion
   * @param localStates cached state
   * @param moduleTimeList module running time
   * @param localMethods cached validation methods
   * @param optimMethods cached optim methods
   * @tparam T Tensor element type
   */
  case class Cache[T](
    localModels: Array[Module[T]],
    modelWeights: Array[Tensor[T]],
    modelGradients: Array[Tensor[T]],
    localCriterions: Array[Criterion[T]],
    localStates: Array[Table],
    var moduleTimeList: Array[Long] = null,
    localMethods: Array[Option[Array[ValidationMethod[T]]]],
    optimMethods: Map[String, OptimMethod[T]]
  )

  /**
   * Train the model.
   *
   * @param dataset train dataset
   * @param coresPerNode cores per node
   * @param state state table
   * @param endWhen trigger to stop training
   * @param metrics metrics
   * @param models cached models
   * @param optimMethods optimization methods
   * @param parameters [[AllReduceParameter]]
   * @param validationTrigger validation trigger
   * @param validationDataSet validation dataset
   * @param validationMethods validation methods
   * @param cacheTrigger cache trigger
   * @param cachePath cache path
   * @param trainSummary train summary
   * @param validationSummary validation summary
   * @param isOverWrite if overwrite the checkpoint
   * @param parameterProcessers a list of ParameterProcessor used to process parameters
   */
  private[optim] def optimize[T: ClassTag](
    trainingModel: Module[T],
    dataset: DistributedDataSet[MiniBatch[T]],
    coresPerNode: Int,
    state: Table,
    endWhen: Trigger,
    metrics: Metrics,
    models: RDD[Cache[T]],
    optimMethods: Map[String, OptimMethod[T]],
    parameters: Map[String, AllReduceParameter[T]],
    validationTrigger: Option[Trigger],
    validationDataSet: Option[DataSet[MiniBatch[T]]],
    validationMethods: Option[Array[ValidationMethod[T]]],
    cacheTrigger: Option[Trigger],
    cachePath: Option[String],
    trainSummary: Option[TrainSummary],
    validationSummary: Option[ValidationSummary],
    isOverWrite: Boolean,
    parameterProcessers: Array[ParameterProcessor]
  )(implicit ev: TensorNumeric[T]): Unit = {
    val sc = dataset.originRDD().sparkContext
    val partitionNum = dataset.originRDD().partitions.length
    var wallClockTime = 0L
    var lastEpochTime = 0L

    // driverState is needed to prevent serializing the whole optimizer
    optimMethods.values.foreach{ optimMethod =>
      if (!optimMethod.state.contains("epoch")) optimMethod.state.update("epoch", 1)
      if (!optimMethod.state.contains("neval")) optimMethod.state.update("neval", 1)
      if (!optimMethod.state.contains("Loss")) {
        optimMethod.state.update("Loss", Float.PositiveInfinity)
      }
      if (!optimMethod.state.contains("score")) optimMethod.state.update("score", 0f)
      if (!optimMethod.state.contains("recordsProcessedThisEpoch")) {
        optimMethod.state.update("recordsProcessedThisEpoch", 0)
      }
    }

    val _subModelNumber = Engine.getEngineType() match {
      case MklBlas => coresPerNode
      case MklDnn => 1
    }
    val driverState = T(
      "epoch" -> optimMethods.values.head.state("epoch"),
      "neval" -> optimMethods.values.head.state("neval"),
      "Loss" -> optimMethods.values.head.state("Loss"),
      "score" -> optimMethods.values.head.state("score"),
      "parallelism" -> _subModelNumber
    )

    logger.info("Count dataset")
    val countBefore = System.nanoTime()
    val numSamples = dataset.data(train = false).map(_.size()).reduce(_ + _)
    val countAfter = System.nanoTime()
    logger.info(s"Count dataset complete. Time elapsed: ${(countAfter - countBefore) / 1e9}s")
    if (numSamples != dataset.size()) {
      logger.warn("If the dataset is built directly from RDD[Minibatch], the data in each " +
        "minibatch is fixed, and a single minibatch is randomly selected in each partition. If " +
        "the dataset is transformed from RDD[Sample], each minibatch will be constructed on the " +
        "fly from random samples, which is better for convergence.")
    }

    logger.info(s"config $state")
    var recordsProcessedThisEpoch = optimMethods.values.head.state[Int]("recordsProcessedThisEpoch")
    if (recordsProcessedThisEpoch == 0) {
      val shuffleBefore = System.nanoTime()
      logger.info("Shuffle data")
      dataset.shuffle()
      val shuffleEnd = System.nanoTime()
      logger.info(s"Shuffle data complete. Takes ${(shuffleEnd - shuffleBefore) / 1e9}s")
    }

    var tasks: ArrayBuffer[Future[_]] = new ArrayBuffer()
    var threshold = Long.MaxValue
    var timeout = Long.MaxValue
    var iteration = 0
    val dropPercentage = state.get[Double]("dropPercentage").get
    val warmupIterationNum = state.get[Int]("warmupIterationNum").get
    val computeThresholdbatchSize = state.get[Int]("computeThresholdbatchSize").get
    val maxDropPercentage = state.get[Double]("maxDropPercentage").get
    val driverSubModelNum = partitionNum * _subModelNumber
    var dropModelNumBatch = 0
    var lossArray = new Array[Double](_subModelNumber)

    var epochStart = System.nanoTime()
    var dataRDD = dataset.data(train = true)

    while (!endWhen(driverState)) {
      val lossSum = sc.accumulator(0.0, "loss sum")
      val recordsNum = sc.accumulator(0, "record number")
      metrics.set("computing time for each node", mutable.ArrayBuffer[Double](), sc)
      metrics.set("get weights for each node", mutable.ArrayBuffer[Double](), sc)
      metrics.set("computing time average", 0.0, sc, partitionNum)
      metrics.set("aggregate gradient time", 0.0, sc, partitionNum)
      metrics.set("get weights average", 0.0, sc, partitionNum)
      metrics.set("put gradient", 0.0, sc, Engine.nodeNumber())
      metrics.set("aggregrateGradientParition average executor", 0.0, sc, Engine.nodeNumber())
      metrics.set("compute weight average", 0.0, sc, Engine.nodeNumber())
      metrics.set("send weights average", 0.0, sc, Engine.nodeNumber())

      val driverMetrics = metrics
      val start = System.nanoTime()

      /*
        Run the forwards/backwards pass using multiple threads in each partition, and track the
        number of model updates that finished before the thread timeout mechanism.
       */
      val numFinishedModelUpdates: Int = dataRDD
        .zipPartitions(models, preservesPartitioning = true) { (data, modelIter) => {
          val cached = modelIter.next()
          val syWStart = System.nanoTime()
          /*
            Note: All models in `cached` share the same storage for weights, so we only need to
            copy the weights from parameter server into the first model's weights.
           */
          val weightsResults = parameters.values.map(p =>
            p.getWeights(cached.modelWeights.head.narrow(1, p.paramOffset, p.size))
          ).toArray
          val miniBatchBuffer = new Array[MiniBatch[T]](_subModelNumber)
          val batch = data.next()
          val stackSize = batch.size() / _subModelNumber
          tasks += Engine.default.invoke(() => {
            require((batch.size() >= _subModelNumber) &&
              (batch.size() % _subModelNumber == 0), "total batch size: " +
              s"${batch.size()} should be divided by total core number: ${_subModelNumber}")
            if (batch.size() < _subModelNumber * 2) {
              logger.warn("Warning: for better training speed, " +
                "total batch size is recommended to be at least two times of core number" +
                s"${_subModelNumber}, please tune your batch size accordingly")
            }
            var b = 0
            while (b < _subModelNumber) {
              miniBatchBuffer(b) = batch.slice(b * stackSize + 1, stackSize)
              b += 1
            }
          })
          Engine.default.sync(tasks)
          weightsResults.foreach(_.waitResult())
          val weightSyncTime = System.nanoTime() - syWStart
          driverMetrics.add("get weights average", weightSyncTime)
          driverMetrics.add("get weights for each node", weightSyncTime)
          tasks.clear()

          // ======================Start train models===================================
          var time = System.nanoTime()
          if (dropPercentage > 0.0 && iteration > warmupIterationNum +
            computeThresholdbatchSize - 1) {
            timeout = threshold - weightSyncTime
          }
          val pre = (iteration % computeThresholdbatchSize) * _subModelNumber
          val trainingThreads = Engine.default.invokeAndWait2((0 until _subModelNumber).map(i =>
            () => {
              val trainStart = System.nanoTime()
              val localModel = cached.localModels(i)
              localModel.training()
              val localCriterion = cached.localCriterions(i)
              val input = miniBatchBuffer(i).getInput()
              val target = miniBatchBuffer(i).getTarget()
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
          driverMetrics.add("computing time for each node", computingTime)

          val finishedThreads = trainingThreads.filter(!_.isCancelled).map(_.get())
          recordsNum += finishedThreads.size * stackSize
          var i = 0
          while (i < finishedThreads.size) {
            lossSum += lossArray(finishedThreads(i))
            i += 1
          }

          if (finishedThreads.nonEmpty) {
            val finishedGradients = finishedThreads.map(cached.modelGradients(_))
            parameters.values.foreach { p =>
              time = System.nanoTime()
              val pOffset = p.paramOffset
              val pLength = p.size
              val taskSize = pLength / _subModelNumber
              val extraTask = pLength % _subModelNumber

              // Aggregate multi-model's gradient to the first model's gradient
              val parallelNum = if (taskSize == 0) extraTask else _subModelNumber
              Engine.default.invokeAndWait((0 until parallelNum).map(tid => () => {
                val offset = pOffset + tid * taskSize + math.min(tid, extraTask)
                val length = taskSize + (if (tid < extraTask) 1 else 0)
                var i = 1
                while (i < finishedGradients.length) {
                  finishedGradients(0).narrow(1, offset, length)
                    .add(finishedGradients(i).narrow(1, offset, length))
                  i += 1
                }
              }))
              driverMetrics.add("aggregate gradient time", System.nanoTime() - time)
              val putG = System.nanoTime()
              // Put first finished model's gradient who aggregated
              // all other models' gradient to AllReduceParameter
              p.putGradients(finishedGradients(0).narrow(1, pOffset, pLength))
              driverMetrics.add("put gradient", System.nanoTime() - putG)
            }
          } else {
            val putG = System.nanoTime()
            // zero gradient in BlockManager when no thread finished.
            cached.modelGradients(0).zero()
            parameters.values.foreach{p =>
              p.putGradients(cached.modelGradients(0).narrow(1, p.paramOffset, p.size))
            }
            driverMetrics.add("put gradient", System.nanoTime() - putG)
          }

          tasks ++= Engine.default.invoke {
            (0 until _subModelNumber).map { i =>
              () => {
                cached.localModels(i).training()
                cached.localModels(i).zeroGradParameters()
              }
            }
          }
          Iterator.single(finishedThreads.size)
        }
      }.reduce(_ + _)

      dropModelNumBatch += (driverSubModelNum - numFinishedModelUpdates)
      if (dropPercentage == 0.0 ||
        numFinishedModelUpdates >= driverSubModelNum * (1.0 - maxDropPercentage)) {
        // enough records were processed for this batch, so update the model
        val value = lossSum.value / numFinishedModelUpdates

        driverState("numFinishedModel") = numFinishedModelUpdates
        // isGradientUpdated is flag to mark whether gradient is updated. May changed in the future.
        driverState("isGradientUpdated") = false
        // parameterProcesser like L2NormClippingProcessor may aggregate gradient,
        // and change the value of isGradientUpdated in driverState.
        parameters.foreach { p =>
          parameterProcessers.foreach(_.collectGlobalData(models, p._2, metrics, driverState))
        }
        val isGradientUpdated = driverState[Boolean]("isGradientUpdated")
        val stateBroadcast = sc.broadcast(driverState)

        models.mapPartitions { modelIter =>
          val modelCache = modelIter.next()
          // if parameterProcesser has aggregated gradient, we can skip this aggregation.
          if (!isGradientUpdated) {
            val getG = System.nanoTime()
            parameters.values.foreach(_.aggregateGradientPartition(numFinishedModelUpdates))
            driverMetrics.add("aggregrateGradientParition average executor",
              System.nanoTime() - getG)
          }
          parameters.foreach { p =>
            parameterProcessers.foreach(_.processParameters(p._2, modelCache, driverState))
          }
          modelCache.optimMethods.foreach{ case (name, optimMethod) =>
            var time = System.nanoTime()
            optimMethod.state.update("epoch", driverState[Int]("epoch"))
            optimMethod.state.update("neval", driverState[Int]("neval"))
            optimMethod.state.update("Loss", driverState[Float]("Loss"))
            if (validationMethods.isDefined) {
              optimMethod.state.update("score", driverState[Float]("score"))
            }

            val p = parameters(name)
            optimMethod.optimize(_ => (ev.fromType(value), p.gradientPartition),
              p.weightPartition)
            driverMetrics.add("compute weight average", System.nanoTime() - time)
            time = System.nanoTime()
            p.sendWeightPartition()
            driverMetrics.add("send weights average", System.nanoTime() - time)
          }
          Iterator.empty
        }.count()

        stateBroadcast.destroy()
        recordsProcessedThisEpoch += recordsNum.value
        val end = System.nanoTime()
        wallClockTime += end - start
        driverState("isGradientUpdated") = true
        driverState("Loss") = lossSum.value.toFloat / numFinishedModelUpdates
        optimMethods.foreach{ v =>
          v._2.updateHyperParameter()
        }
        // TODO: Support show learningrate for multiOptimMethod
        driverState(s"LearningRate") = optimMethods.head._2.getLearningRate().toFloat

        driverState("Throughput") = recordsNum.value.toFloat / ((end - start) / 1e9f)
        val _header = header(driverState[Int]("epoch"), recordsProcessedThisEpoch, numSamples,
          driverState[Int]("neval"), wallClockTime)
        logger.info(s"${_header} Trained ${recordsNum.value} records in ${(end - start) / 1e9} " +
          s"seconds. Throughput is ${driverState("Throughput")} records/second. Loss is ${
            driverState("Loss")}. ${getHyperParameterLog(optimMethods)}")
        logger.debug("\n" + metrics.summary())
        logger.debug("Dropped modules: " + (driverSubModelNum - numFinishedModelUpdates))
        lossArray = new Array[Double](_subModelNumber)

        // compute threshold
        iteration += 1
        if (dropPercentage > 0.0 && iteration > warmupIterationNum &&
          iteration % computeThresholdbatchSize == 0) {
          val moduleTimeList = models.mapPartitions { iter =>
            iter.next().moduleTimeList.iterator
          }.collect()

          val k = (dropPercentage * computeThresholdbatchSize * driverSubModelNum).toInt
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
        if (recordsProcessedThisEpoch >= numSamples) {
          // Epoch is finished
          val epochEnd = System.nanoTime()
          wallClockTime = lastEpochTime + epochEnd - epochStart
          lastEpochTime = wallClockTime
          epochStart = System.nanoTime()
          logger.info(s"${_header} Epoch finished. Wall clock time is ${wallClockTime / 1e6} ms")

          driverState("epoch") = driverState[Int]("epoch") + 1
          dataset.shuffle()
          dataRDD = dataset.data(train = true)
          recordsProcessedThisEpoch = 0
        }

        optimMethods.map { case (moduleName, optimMethod) =>
          optimMethod.state.update("recordsProcessedThisEpoch", recordsProcessedThisEpoch)
          optimMethod.state.update("epoch", driverState[Int]("epoch"))
          optimMethod.state.update("neval", driverState[Int]("neval"))
          optimMethod.state.update("Loss", driverState[Float]("Loss"))
          if (validationMethods.isDefined) {
            optimMethod.state.update("score", driverState[Float]("score"))
          }
        }

        validate(
          validationTrigger,
          validationDataSet,
          validationMethods,
          coresPerNode,
          models,
          driverState,
          validationSummary,
          _header
        )

        trainSummary.foreach { summary =>
          saveSummary(
            summary,
            models,
            driverState,
            parameters,
            trainingModel
          )
        }

        checkpoint(
          cacheTrigger,
          cachePath,
          isOverWrite,
          wallClockTime,
          models,
          driverState,
          parameters,
          optimMethods,
          trainingModel
        )

      } else {
        logger.info(s"Warning! Not enough training samples were successfully processed in this " +
          s"iteration due to some slow tasks. The gradients computed in this iteration will be " +
          s"discarded. Only $numFinishedModelUpdates/$driverSubModelNum threads successfully " +
          s"completed training.")
      }
    }
  }

  /**
   * Create checkpoint.
   *
   * @param cacheTrigger cache trigger
   * @param cachePath cache path
   * @param isOverWrite whether over write
   * @param wallClockTime wall clock time
   * @param models cached models
   * @param state state table
   * @param parameters all reduce parameters
   */
  private def checkpoint[T: ClassTag](
    cacheTrigger: Option[Trigger],
    cachePath: Option[String],
    isOverWrite: Boolean,
    wallClockTime: Long,
    models: RDD[Cache[T]],
    state: Table,
    parameters: Map[String, AllReduceParameter[T]],
    optimMethods: Map[String, OptimMethod[T]],
    trainingModel: Module[T]): Unit = {
    cacheTrigger.foreach { trigger =>
      cachePath.foreach { path =>
        if (trigger(state)) {
          saveModel(getModel(models, parameters, trainingModel), cachePath, isOverWrite,
            s".${state[Int]("neval")}")
          logger.info(s"[Wall Clock ${wallClockTime / 1e9}s] Save model to $path")
          optimMethods.foreach{case (name, optimMethod) =>
            optimMethod.state.update("epoch", state[Int]("epoch"))
            optimMethod.state.update("neval", state[Int]("neval"))
            saveOptimMethod(optimMethod, cachePath, isOverWrite, s"-$name.${state[Int]("neval")}")
            logger.info(s"[Wall Clock ${wallClockTime / 1e9}s] Save optimMethod " +
              s"${optimMethod} to $path")
          }
        }
      }
    }
  }

  /**
   * Save train summaries.
   *
   * @param trainSummary train logger
   * @param models cached models
   * @param driverState driver state
   * @param parameters [[AllReduceParameter]]
   */
  private def saveSummary[T: ClassTag](
    trainSummary: TrainSummary,
    models: RDD[Cache[T]],
    driverState: Table,
    parameters: Map[String, AllReduceParameter[T]],
    trainingModel: Module[T])(implicit ev: TensorNumeric[T]): Unit = {
    val currentIteration = driverState[Int]("neval") - 1
    val parametersTrigger = trainSummary.getSummaryTrigger("Parameters")
    if (parametersTrigger.isDefined && parametersTrigger.get(driverState)) {
      val model = getModel(models, parameters, trainingModel)
      val parametersTable = model.getParametersTable()
      // Parallelize to create Histogram.
      Engine.default.invokeAndWait(
        parametersTable.keySet.toSeq.map(moduleName => () => {
          val paramTable = parametersTable[Table](moduleName)
          paramTable.keySet.foreach { paramName =>
            trainSummary.addHistogram(
              s"$moduleName/$paramName", paramTable[Tensor[T]](paramName), currentIteration)}
        }))
    }
    val scalarTrigger = trainSummary.getScalarTriggers()
    // Not parallelizable, because driverState is changing each iteration.
    scalarTrigger.foreach { v =>
      if (v._2(driverState)) {
        // TODO: Support show learningrate for multiOptimMethod
        require(driverState.contains(v._1), s"DistriOptimizer.saveSummary: Summary ${v._1} " +
          s"is not supported now.")
        trainSummary.addScalar(
          v._1, driverState[Float](v._1), currentIteration
        )
      }
    }
  }

  /**
   * Init engine and cache models, weights, gradients, criterions, state tables
   * and validation methods on worker nodes.
   *
   * @param model train model
   * @param dataset train dataset
   * @param criterion loss function
   * @param state state table
   * @param nodeNumber node number
   * @param coresPerNode cores per node
   * @param checkSingleton if checkSingleton
   * @param parameters all reduce parameter instance
   * @param validationMethods validation methods
   * @param optimMethod optimization method
   * @param parameterProcessors a list of ParameterProcessor used to process parameters
   * @return cached models
   */
  private def initThreadModels[T: ClassTag](
    model: Module[T],
    dataset: DistributedDataSet[MiniBatch[T]],
    criterion: Criterion[T],
    state: Table,
    nodeNumber: Int,
    coresPerNode: Int,
    checkSingleton: Boolean,
    parameters: Map[String, AllReduceParameter[T]],
    validationMethods: Option[Array[ValidationMethod[T]]],
    optimMethod: Map[String, OptimMethod[T]],
    parameterProcessors: ArrayBuffer[ParameterProcessor]
  )(implicit ev: TensorNumeric[T]): (RDD[DistriOptimizer.Cache[T]], ModelBroadcast[T]) = {
    val sc = dataset.originRDD().sparkContext
    val broadcast = sc.broadcast((criterion, state, validationMethods, optimMethod))
    // ensure model's parameter is compacted for getting a better performance when broadcasting
    model.getParameters()
    // As cloneModel is using Serialization to implement deep copy, and will throw OOMError
    // when model's size is bigger than SerializationUtils' buffer size. So we can use
    // ModelBroadcast to clone model here.
    // Notes: All models returned by modelBroadcast.value() share the same weight&bias, while
    // gradWeight&gradBias is unshared.
    val modelBroadcast = ModelBroadcast[T]().broadcast(sc, model)
    val _subModelNumber = Engine.getEngineType match {
      case MklBlas => coresPerNode
      case MklDnn => 1
      case _ => throw new IllegalArgumentException
    }

    require(dataset.originRDD().partitions.length == nodeNumber,
      s"Passed in rdd partition number ${dataset.originRDD().partitions.length}" +
        s" is not equal to configured node number ${nodeNumber}")


    val computeThresholdbatchSize = state.get[Int]("computeThresholdbatchSize").get
    val nExecutor = Engine.nodeNumber()
    val executorCores = Engine.coreNumber()

    val models = dataset.originRDD().mapPartitions(_ => {
      val partitionId = TaskContext.getPartitionId
      val (broadcastCriterion, broadcastState, broadcastMethod,
      broadcastOptim) = broadcast.value
      if (!Engine.checkSingleton()) {
        if (checkSingleton) {
          require(Engine.checkSingleton(), "Partitions of the training data are not evenly" +
            "distributed across the executors in the Spark cluster; are there sufficient training" +
            "data to be distributed? Set property \"bigdl.check.singleton\" to false to skip " +
            "this check")
        } else {
          logger.warn("Partitions of the training data are not evenly" +
            "distributed across the executors in the Spark cluster; are there sufficient training" +
            "data to be distributed?")
        }
      }
      Engine.setNodeAndCore(nExecutor, executorCores)
      val cached = (0 until _subModelNumber).map { _ =>
        val localModel = modelBroadcast.value(true)
        localModel match {
          case container: MklDnnContainer => container.compile(TrainingPhase)
          case _ =>
        }
        // differentiate partition models from each other by partition ID
        setModelId(localModel, partitionId)
        val localCriterion = broadcastCriterion.cloneCriterion()
        val localState = broadcastState.clone()
        val localMethod =
          if (broadcastMethod.isDefined) Some(broadcastMethod.get.map(_.clone())) else None
        val (weights, grads) = localModel.getParameters()
        (localModel, weights, grads, localCriterion, localState, localMethod)
      }.toArray

      logger.info("model thread pool size is " + Engine.model.getPoolSize)
      val weights = cached.head._2
      parameters.foreach(v =>
        v._2.init(weights.narrow(1, v._2.paramOffset, v._2.size))
      )

      Iterator.single(Cache(
        cached.map(_._1), // models
        cached.map(_._2), // weights
        cached.map(_._3), // gradients
        cached.map(_._4), // criterions
        cached.map(_._5), // states
        new Array[Long](_subModelNumber * computeThresholdbatchSize),
        cached.map(_._6),
        broadcastOptim.map(v => (v._1, v._2.clone()))
      ))
    }).persist()
    models.setName("Thread Model RDD")
    logger.info("Cache thread models...")
    models.count()
    logger.info("Cache thread models... done")
    (models, modelBroadcast)
  }

  private def setModelId[T: ClassTag](model: Module[T], partitionId: Int): Unit = {
    model.setId(partitionId)
    if (model.isInstanceOf[Container[_, _, T]]) {
      model.asInstanceOf[Container[_, _, T]].modules.
        foreach(sub => setModelId(sub, partitionId))
    }
  }

  /**
   * Validate current model and save the result.
   *
   * @param validationTrigger validation trigger
   * @param validationDataSet validation dataset
   * @param validationMethods validation methods
   * @param coresPerNode cores per node
   * @param models cached models
   * @param state state table
   * @param validationSummary validation logger.
   * @param header log header string
   */
  private def validate[T](
    validationTrigger: Option[Trigger],
    validationDataSet: Option[DataSet[MiniBatch[T]]],
    validationMethods: Option[Array[ValidationMethod[T]]],
    coresPerNode: Int,
    models: RDD[Cache[T]],
    state: Table,
    validationSummary: Option[ValidationSummary],
    header: String
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
    logger.info(s"$header Validate model...")
    val _subModelNumber = Engine.getEngineType match {
      case MklBlas => coresPerNode
      case MklDnn => 1
      case _ => throw new IllegalArgumentException
    }
    val results = ZippedPartitionsWithLocalityRDD(models, validateRDD)((modelIter, dataIter) => {
      val cached = modelIter.next()
      val vMethodsArr = cached.localMethods
      val workingModels = cached.localModels

      workingModels.foreach(_.evaluate())
      dataIter.map(batch => {
        val stackSize = batch.size() / _subModelNumber
        val extraSize = batch.size() % _subModelNumber
        val parallelism = if (stackSize == 0) extraSize else _subModelNumber
        Engine.default.invokeAndWait(
          (0 until parallelism).map(b =>
            () => {
              val offset = b * stackSize + math.min(b, extraSize) + 1
              val length = stackSize + (if (b < extraSize) 1 else 0)
              val miniBatch = batch.slice(offset, length)
              val input = miniBatch.getInput()
              val target = miniBatch.getTarget()
              val output = workingModels(b).forward(input)
              val validatMethods = vMethodsArr(b).get
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
      })
    }).reduce((left, right) => {
      left.zip(right).map { case (l, r) =>
        l + r
      }
    }).zip(vMethods)
    results.foreach(r => {
      logger.info(s"$header ${r._2} is ${r._1}")
    })
    state("score") = results(0)._1.result._1
    if(validationSummary.isDefined) {
      results.foreach { r =>
        val result = r._1.result
        validationSummary.get.addScalar(r._2.toString(), result._1,
          state[Int]("neval") - 1
        )
      }
    }
  }

  /**
   * Fetch current model parameters to driver, and copy to trainingModel.
   *
   * @param models cached models
   * @param parameters [[AllReduceParameter]]
   * @param trainingModel the model is trained by optimizer
   * @return trained model
   */
  private def getModel[T: ClassTag](
    models: RDD[Cache[T]],
    parameters: Map[String, AllReduceParameter[T]],
    trainingModel: Module[T]): Module[T] = {
    val partitionNum = models.partitions.length
    val extraState = models.map(_.localModels.head.getExtraParameter()).first()
    trainingModel.setExtraParameter(extraState)

    // make sure gradient is as the same length as weight
    val parameterArray = trainingModel.parameters()
    (0 until parameterArray._2.length).foreach(i =>
      parameterArray._2(i).resizeAs(parameterArray._1(i))
    )

    val (parameter, gradientParameter) = trainingModel.getParameters()

    parameters.foreach { case (moduleName, p) =>
      val currentModule = trainingModel(moduleName)
      require(currentModule.isDefined, s"Couldn't find $moduleName in $trainingModel")
      val (weights, gradients) = models.mapPartitions(iter => {
        val cached = iter.next()
        val curPartitionId = TaskContext.getPartitionId()
        Iterator.single((Map(curPartitionId -> p.weightPartition),
          Map(curPartitionId -> p.gradientPartition)))
      }).reduce((a, b) => (a._1 ++ b._1, a._2 ++ b._2))

      val taskSize = p.size / partitionNum
      require(taskSize != 0, "parameter length should not less than partition number")
      val extraSize = p.size % partitionNum

      (0 until partitionNum).map(pid => {
        val start = p.paramOffset + pid * taskSize + math.min(pid, extraSize)
        val length = taskSize + (if (pid < extraSize) 1 else 0)
        parameter.narrow(1, start, length).copy(weights(pid))
        gradientParameter.narrow(1, start, length).copy(gradients(pid))
      })
    }

    trainingModel
  }
}

/**
 * The optimizer run on a distributed cluster.
 *
 * @param _model train model
 * @param _dataset train dataset
 * @param _criterion loss function
 */
class DistriOptimizer[T: ClassTag] (
  _model: Module[T],
  _dataset: DistributedDataSet[MiniBatch[T]],
  _criterion: Criterion[T]
 )(implicit ev: TensorNumeric[T])
  extends Optimizer[T, MiniBatch[T]](
    _model, _dataset, _criterion) {
  val metrics = new Metrics

  private var models: RDD[DistriOptimizer.Cache[T]] = null
  // this variable is used to check the models cloned when broadcast, if there're native resources,
  // it will be deleted at the end of Optimizer.
  private var modelBroadcast: ModelBroadcast[T] = null

  /**
   * Clean some internal states, so this or other optimizers can run optimize again
   *
   * This method will be called at the end of optimize. You need not call it if optimize succeed.
   * If the optimize fails, you may call it before next optimize.
   */
  def clearState() : Unit = {
    // Reset the singleton flag, so other optimizers can run
    models.mapPartitions(iter => {
      Engine.resetSingletonFlag()
      iter
    }).count()
  }

  private def endEpoch(): Unit = {
    optimMethods.foreach { case (moduleName, optimMethod) =>
      val records = optimMethod.state.get[Int]("recordsProcessedThisEpoch")
      if (records.isDefined && records.get != 0) {
        optimMethod.state("epoch") = optimMethod.state[Int]("epoch") + 1
        optimMethod.state("recordsProcessedThisEpoch") = 0
      }
    }
  }

  override def setTrainData(sampleRDD: RDD[Sample[T]],
    batchSize: Int,
    miniBatch: MiniBatch[T]): this.type = {
    this.dataset = (DataSet.rdd(sampleRDD) ->
      SampleToMiniBatch(miniBatch, batchSize, None))
      .asInstanceOf[DistributedDataSet[MiniBatch[T]]]
    // if current epoch is not finished, we will end the
    // current epoch and start a new epoch when optimize is called
    endEpoch()
    this
  }

  override def setTrainData(sampleRDD: RDD[Sample[T]],
    batchSize: Int,
    featurePaddingParam: PaddingParam[T] = null,
    labelPaddingParam: PaddingParam[T] = null) : this.type = {
    val _featurePaddingParam = if (featurePaddingParam != null) Some(featurePaddingParam) else None
    val _labelPaddingParam = if (labelPaddingParam != null) Some(labelPaddingParam) else None
    dataset = (DataSet.rdd(sampleRDD) ->
      SampleToMiniBatch(batchSize, _featurePaddingParam, _labelPaddingParam))
      .asInstanceOf[DistributedDataSet[MiniBatch[T]]]
    // if current epoch is not finished, we will end the
    // current epoch and start a new epoch when optimize is called
    endEpoch()
    this
  }


  override def prepareInput(): Unit = {
    import DistriOptimizer._
    if (!dataset.asInstanceOf[DistributedDataSet[MiniBatch[T]]].isCached) {
      logger.info("caching training rdd ...")
      dataset.asInstanceOf[DistributedDataSet[MiniBatch[T]]].cache()
      // FIXME dnn model must cache val dataset first, otherwise there will be a segment fault.
      if (validationDataSet.isDefined) {
        validationDataSet.get.toDistributed().cache()
      }
    }
  }

  override def optimize(): Module[T] = {

    val distDataset = dataset.asInstanceOf[DistributedDataSet[MiniBatch[T]]]

    optimMethods.values.foreach { optimMethod =>
      optimMethod.clearHistory()
    }

    // To be compatible with the old usage that user define hyperparameters in a table.
    if (optimMethods.size == 1) {
      optimMethods.head._2.loadFromTable(state)
    }

    state("dropPercentage") = dropPercentage
    state("warmupIterationNum") = warmupIterationNum
    state("computeThresholdbatchSize") = computeThresholdbatchSize
    state("maxDropPercentage") = maxDropPercentage
    state("isLayerwiseScaled") = Utils.isLayerwiseScaled(_model)

    val nodeNumber = Engine.nodeNumber()
    val coresPerNode = Engine.coreNumber()

    val partitionNum = distDataset.originRDD().partitions.length
    val modelParameters = model.getParameters()
    // subModuleName -> (storageOffset, length, AllReduceParameter)
    val parameters = if (optimMethods.size != 1) {
      val p = optimMethods.map{case (subModuleName, optimMethods) =>
        val subModule = model(subModuleName)
        require(subModule.isDefined, s"Optimizer couldn't find $subModuleName in $model")
        val subModuleWeights = subModule.get.getParameters()._1
        (subModuleName, subModuleWeights)
      }
      val sortedWeights = p.values.toArray.sortWith((a, b) => a.storageOffset() < b.storageOffset())
      val compactWeights = Module.isCompact(sortedWeights)
      require(modelParameters._1 == compactWeights,
        s"DistriOptimizer: All subModules should have an OptimMethod.")
      p.map{case (subModuleName, weights) =>
        (subModuleName, AllReduceParameter.newParameter[T](
          partitionNum, weights.nElement(), weights.storageOffset()))
      }
    } else if (optimMethods.contains(model.getName())) {
      Map(model.getName() -> AllReduceParameter.newParameter[T](
        partitionNum, modelParameters._1.nElement()))
    } else {
      throw new IllegalArgumentException(s"${model.getName()} doesn't " +
        s"have corresponding OptimMethod")
    }

    prepareInput()

    val modelsAndBroadcast = DistriOptimizer.initThreadModels(model, distDataset, criterion, state,
      nodeNumber, coresPerNode, checkSingleton, parameters, validationMethods,
      optimMethods, parameterProcessors)
    models = modelsAndBroadcast._1
    modelBroadcast = modelsAndBroadcast._2

    if (checkpointPath.isDefined) {
      val file = checkpointPath.get + "/" +
        new SimpleDateFormat("yyyyMMdd_HHmmss").format(Calendar.getInstance().getTime())
      new File(file).mkdir()
      checkpointPath = Some(file)
    }

    var retryNum = 0
    val maxRetry = System.getProperty("bigdl.failure.retryTimes", "5").toInt
    val retryTimeInterval = System.getProperty("bigdl.failure.retryTimeInterval", "120").toInt
    var lastFailureTimestamp = System.nanoTime()

    while (retryNum < maxRetry) {
      try {
        DistriOptimizer.optimize(
          model,
          distDataset,
          coresPerNode,
          state,
          endWhen,
          metrics,
          models,
          optimMethods,
          parameters,
          validationTrigger,
          validationDataSet,
          validationMethods,
          checkpointTrigger,
          checkpointPath,
          trainSummary,
          validationSummary,
          isOverWrite,
          parameterProcessors.toArray
        )
        retryNum = Int.MaxValue
      } catch {
        case e: IllegalArgumentException =>
          throw e
        case t: Throwable =>
          DistriOptimizer.logger.error("Error: " + ExceptionUtils.getStackTrace(t))
          if (checkpointPath.isDefined) {
            /* To avoid retry number is used up by first few exceptions, we count time here.
             * If exception exceeds maxRetry times in maxRetry*retryTimeInterval seconds,
             * we will give up retry Or we will reset retryNum
             */
            if (System.nanoTime() - lastFailureTimestamp < maxRetry * retryTimeInterval * 1e9) {
              retryNum += 1
              if (retryNum == maxRetry) {
                throw t
              }
            } else {
              retryNum = 1
            }
            DistriOptimizer.logger.info(s"Retrying $retryNum times")
            lastFailureTimestamp = System.nanoTime()

            val modelFile = getLatestFile(checkpointPath.get, "model")
            clearState()
            models.unpersist()
            val newModel = if (modelFile != null) {
              DistriOptimizer.logger.info("Model recover from last snapshot")
              Module.load[T](modelFile)
            } else {
              DistriOptimizer.logger.info("Model recover from origin model")
              model
            }
            optimMethods = optimMethods.map { case (moduleName, optimMethod) =>
              val methodFile = getLatestFile(checkpointPath.get, s"optimMethod-$moduleName")

              val newOptimMethod = if (methodFile != null) {
                DistriOptimizer.logger.info(s"$moduleName's OptimMethod recover from last snapshot")
                OptimMethod.load[T](methodFile)
              } else {
                DistriOptimizer.logger.info(s"$moduleName's OptimMethod recover from origin model")
                optimMethod
              }
              newOptimMethod.clearHistory()
              (moduleName, newOptimMethod)
            }
            val modelsAndBroadcast = DistriOptimizer.initThreadModels(newModel, distDataset,
              criterion, state, nodeNumber, coresPerNode, checkSingleton, parameters,
              validationMethods, optimMethods, parameterProcessors)
            models = modelsAndBroadcast._1
            modelBroadcast = modelsAndBroadcast._2
          } else {
            throw t
          }
      }
    }

    DistriOptimizer.getModel(models, parameters, model)

    // Reset some internal states, so this or other optimizers can run optimize again
    clearState()

    // unpersist the model because the next time optimize is called, new `models` will be
    // created
    shutdown()
    models.unpersist()

    model
  }

  private def getLatestFile(path: String, fileName: String): String = {
    val fl = new java.io.File(path)
    val files = fl.listFiles(new FilenameFilter {
      override def accept(dir: File, name: String): Boolean = {
        name.startsWith(fileName)
      }
    })

    var lastMod = Long.MinValue
    var choice: String = null
    files.map {file =>
      if (file.lastModified() > lastMod) {
        choice = file.getPath;
        lastMod = file.lastModified();
      }
    }
    return choice;
  }

  // this shutdown should not be called out of this scope.
  private[optim] override def shutdown(): Unit = {
    models.mapPartitions { iter =>
      iter.foreach { arrayModels =>
        arrayModels.localModels.foreach(_.release())
      }

      iter
    }.count()
    CachedModels.deleteKey(modelBroadcast.uuid)
  }
}
