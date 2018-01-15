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
import com.intel.analytics.bigdl.dataset.{DataSet, DistributedDataSet,
                        MiniBatch, SampleToMiniBatch, Sample, PaddingParam}
import com.intel.analytics.bigdl.nn.{Module, Utils}
import com.intel.analytics.bigdl.parameters.AllReduceParameter
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import java.io.{File, FilenameFilter}
import java.text.SimpleDateFormat
import java.util.Calendar

import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import org.apache.commons.lang.exception.ExceptionUtils
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import org.apache.log4j.Logger
import org.apache.spark.TaskContext
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
    optimMethod: OptimMethod[T]
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
   * @param optimMethod optimization method
   * @param parameters [[AllReduceParameter]]
   * @param validationTrigger validation trigger
   * @param validationDataSet validation dataset
   * @param validationMethods validation methods
   * @param cacheTrigger cache trigger
   * @param cachePath cache path
   * @param trainSummary train summary
   * @param validationSummary validation summary
   * @param isOverWrite  if overwrite the checkpoint
   * @param clippingParams  gradient clipping configurations
   */
  private[optim] def optimize[T: ClassTag](
    trainingModel: Module[T],
    dataset: DistributedDataSet[MiniBatch[T]],
    coresPerNode: Int,
    state: Table,
    endWhen: Trigger,
    metrics: Metrics,
    models: RDD[Cache[T]],
    optimMethod: OptimMethod[T],
    parameters: AllReduceParameter[T],
    validationTrigger: Option[Trigger],
    validationDataSet: Option[DataSet[MiniBatch[T]]],
    validationMethods: Option[Array[ValidationMethod[T]]],
    cacheTrigger: Option[Trigger],
    cachePath: Option[String],
    trainSummary: Option[TrainSummary],
    validationSummary: Option[ValidationSummary],
    isOverWrite: Boolean,
    clippingParams: GradientClippingParams
  )(implicit ev: TensorNumeric[T]): Unit = {
    val sc = dataset.originRDD().sparkContext
    val partitionNum = dataset.originRDD().partitions.length
    var wallClockTime = 0L
    var lastEpochTime = 0L
    val driverState = T(
      "epoch" -> optimMethod.state.get[Int]("epoch").getOrElse(1),
      "neval" -> optimMethod.state.get[Int]("neval").getOrElse(1),
      "Loss" -> optimMethod.state.get[Float]("Loss").getOrElse(Float.PositiveInfinity),
      "score" -> optimMethod.state.get[Float]("score").getOrElse(0f)
    )
    val _subModelNumber = Engine.getEngineType() match {
      case MklBlas => coresPerNode
    }

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

    val shuffleBefore = System.nanoTime()
    logger.info(s"config $state")
    logger.info("Shuffle data")
    dataset.shuffle()
    val shuffleEnd = System.nanoTime()
    logger.info(s"Shuffle data complete. Takes ${(shuffleEnd - shuffleBefore) / 1e9}s")

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

    // gradient clip settings
    val constantClippingEnable = clippingParams.enableConstantClipping
    val normClippingEnable = clippingParams.enableL2NormClipping
    val maxValueClip = clippingParams.maxValueClip
    val minValueClip = clippingParams.minValueClip
    val normValueClip = clippingParams.normValueClip

    var epochStart = System.nanoTime()
    var dataRDD = dataset.data(train = true)
    var recordsProcessedThisEpoch = 0
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
          val weightsResult = parameters.getWeights(cached.modelWeights.head)
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
          weightsResult.waitResult()
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
            time = System.nanoTime()
            val gradLength = finishedGradients(0).nElement()
            val taskSize = gradLength / _subModelNumber
            val extraTask = gradLength % _subModelNumber

            // Aggregate multi-model's gradient to the first model's gradient
            val parallelNum = if (taskSize == 0) extraTask else _subModelNumber
            Engine.default.invokeAndWait((0 until parallelNum).map(tid => () => {
              val offset = tid * taskSize + math.min(tid, extraTask)
              val length = taskSize + (if (tid < extraTask) 1 else 0)
              var i = 1
              while (i < finishedGradients.length) {
                finishedGradients(0).narrow(1, offset + 1, length)
                  .add(finishedGradients(i).narrow(1, offset + 1, length))
                i += 1
              }
            }))
            driverMetrics.add("aggregate gradient time", System.nanoTime() - time)
            val putG = System.nanoTime()
            // Put first finished model's gradient who aggregated
            // all other models' gradient to AllReduceParameter
            parameters.putGradients(finishedGradients(0))
            driverMetrics.add("put gradient", System.nanoTime() - putG)
          } else {
            val putG = System.nanoTime()
            // zero gradient in BlockManager when no thread finished.
            parameters.putGradients(cached.modelGradients(0).zero())
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

        var l2Norm = 0.0f
        var scale = ev.fromType(numFinishedModelUpdates)
        if (normClippingEnable) {
          val sumSquare = models.mapPartitions(modelIter => {
            val getG = System.nanoTime()
            parameters.aggregateGradientPartition()
            driverMetrics.add("aggregrateGradientParition average executor",
              System.nanoTime() - getG)

            val gradLength = parameters.gradientPartition.nElement()
            val taskSize = gradLength / _subModelNumber
            val extraTask = gradLength % _subModelNumber
            val parallelNum = if (taskSize == 0) extraTask else _subModelNumber
            val squares = new Array[Double](parallelNum)
            Engine.default.invokeAndWait((0 until parallelNum).map(tid => () => {
              val offset = tid * taskSize + math.min(tid, extraTask)
              val length = taskSize + (if (tid < extraTask) 1 else 0)
              squares(tid) = ev.toType[Double](
                parameters.gradientPartition.narrow(1, offset + 1, length).sumSquare())
            }))
            var sum = 0.0
            var i = 0
            while (i < parallelNum) {
              sum += squares(i)
              i += 1
            }
            Iterator.single(sum)
          }).reduce(_ + _)
          l2Norm = (math.sqrt(sumSquare) / numFinishedModelUpdates).toFloat
          if (l2Norm > normValueClip) {
            scale = ev.fromType[Double]((l2Norm * numFinishedModelUpdates) / normValueClip)
          }
        }

        models.mapPartitions { modelIter =>
          val modelCache = modelIter.next()
          if (!normClippingEnable) {
            val getG = System.nanoTime()
            parameters.aggregateGradientPartition()
            driverMetrics.add("aggregrateGradientParition average executor",
              System.nanoTime() - getG)
          }
          parameters.gradientPartition.div(scale)
          modelCache.optimMethod.state.update("epoch", driverState[Int]("epoch"))
          modelCache.optimMethod.state.update("neval", driverState[Int]("neval"))
          modelCache.optimMethod.state.update("Loss", driverState[Float]("Loss"))
          if (validationMethods.isDefined) {
            modelCache.optimMethod.state.update("score", driverState[Float]("score"))
          }
          var time = System.nanoTime()
          // gradient clipping
          if (constantClippingEnable) {
            parameters.gradientPartition.clamp(minValueClip, maxValueClip)
          }
          modelCache.optimMethod.optimize(_ => (ev.fromType(value), parameters.gradientPartition),
            parameters.weightPartition)
          driverMetrics.add("compute weight average", System.nanoTime() - time)
          time = System.nanoTime()
          parameters.sendWeightPartition()
          driverMetrics.add("send weights average", System.nanoTime() - time)
          Iterator.empty
        }.count()

        recordsProcessedThisEpoch += recordsNum.value
        val end = System.nanoTime()
        wallClockTime += end - start
        optimMethod.state.update("epoch", driverState[Int]("epoch"))
        optimMethod.state.update("neval", driverState[Int]("neval"))
        driverState("Loss") = lossSum.value.toFloat / numFinishedModelUpdates
        optimMethod.state.update("Loss", driverState[Float]("Loss"))
        if (validationMethods.isDefined) {
          optimMethod.state.update("score", driverState[Float]("score"))
        }
        optimMethod.updateHyperParameter()
        driverState("Throughput") = recordsNum.value.toFloat / ((end - start) / 1e9f)
        driverState("LearningRate") = -optimMethod.getLearningRate().toFloat
        val _header = header(driverState[Int]("epoch"), recordsProcessedThisEpoch, numSamples,
          driverState[Int]("neval"), wallClockTime)
        logger.info(s"${_header} Trained ${recordsNum.value} records in ${(end - start) / 1e9} " +
          s"seconds. Throughput is ${driverState("Throughput")} records/second. Loss is ${
            driverState("Loss")}. ${optimMethod.getHyperParameter()}")
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

        validate(
          validationTrigger,
          validationDataSet,
          validationMethods,
          coresPerNode,
          models,
          wallClockTime,
          driverState,
          validationSummary
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
          optimMethod,
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
    parameters: AllReduceParameter[T],
    optimMethod: OptimMethod[T],
    trainingModel: Module[T]): Unit = {
    cacheTrigger.foreach { trigger =>
      cachePath.foreach { path =>
        if (trigger(state)) {
          println(s"[Wall Clock ${wallClockTime / 1e9}s] Save model to $path")
          saveModel(getModel(models, parameters, trainingModel), cachePath, isOverWrite,
            s".${state[Int]("neval")}")
          optimMethod.state.update("epoch", state[Int]("epoch"))
          optimMethod.state.update("neval", state[Int]("neval"))
          saveOptimMethod(optimMethod, cachePath, isOverWrite, s".${state[Int]("neval")}")
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
    parameters: AllReduceParameter[T],
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
    parameters: AllReduceParameter[T],
    validationMethods: Option[Array[ValidationMethod[T]]],
    optimMethod: OptimMethod[T]
  )(implicit ev: TensorNumeric[T]) = {
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
      case _ => throw new IllegalArgumentException
    }

    require(dataset.originRDD().partitions.length == nodeNumber,
      s"Passed in rdd partition number ${dataset.originRDD().partitions.length}" +
        s" is not equal to configured node number ${nodeNumber}")

    val partitionNum = dataset.originRDD().partitions.length
    val computeThresholdbatchSize = state.get[Int]("computeThresholdbatchSize").get
    val nExecutor = Engine.nodeNumber()
    val executorCores = Engine.coreNumber()
    val models = dataset.originRDD().mapPartitions(_ => {
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
        val localCriterion = broadcastCriterion.cloneCriterion()
        val localState = broadcastState.clone()
        val localMethod =
          if (broadcastMethod.isDefined) Some(broadcastMethod.get.map(_.clone())) else None
        val (weights, grads) = localModel.getParameters()
        (localModel, weights, grads, localCriterion, localState, localMethod)
      }.toArray

      logger.info("model thread pool size is " + Engine.model.getPoolSize)
      val weights = cached.head._2
      parameters.init(weights)

      Iterator.single(Cache(
        cached.map(_._1), // models
        cached.map(_._2), // weights
        cached.map(_._3), // gradients
        cached.map(_._4), // criterions
        cached.map(_._5), // states
        new Array[Long](_subModelNumber * computeThresholdbatchSize),
        cached.map(_._6),
        broadcastOptim.clone()
      ))
    }).persist()
    models.setName("Thread Model RDD")
    logger.info("Cache thread models...")
    models.count()
    logger.info("Cache thread models... done")
    models
  }


  /**
   * Validate current model and save the result.
   *
   * @param validationTrigger validation trigger
   * @param validationDataSet validation dataset
   * @param validationMethods validation methods
   * @param coresPerNode cores per node
   * @param models cached models
   * @param wallClockTime wall clock time
   * @param state state table
   * @param validationSummary validation logger.
   */
  private def validate[T](
    validationTrigger: Option[Trigger],
    validationDataSet: Option[DataSet[MiniBatch[T]]],
    validationMethods: Option[Array[ValidationMethod[T]]],
    coresPerNode: Int,
    models: RDD[Cache[T]],
    wallClockTime: Long,
    state: Table,
    validationSummary: Option[ValidationSummary]
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
      case MklBlas => coresPerNode
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
      logger.info(s"${r._2} is ${r._1}")
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
    parameters: AllReduceParameter[T],
    trainingModel: Module[T]): Module[T] = {
    val partitionNum = models.partitions.length
    val extraState = models.map(_.localModels.head.getExtraParameter()).first()
    trainingModel.setExtraParameter(extraState)
    val (weights, gradients) = models.mapPartitions(iter => {
      val cached = iter.next()
      val curPartitionId = TaskContext.getPartitionId()
      Iterator.single((Map(curPartitionId -> parameters.weightPartition),
        Map(curPartitionId -> parameters.gradientPartition)))
    }).reduce((a, b) => (a._1 ++ b._1, a._2 ++ b._2))

    val parameterArray = trainingModel.parameters()
    (0 until parameterArray._2.length).foreach(i =>
      parameterArray._2(i).resizeAs(parameterArray._1(i))
    )
    val (parameter, gradientParameter) = trainingModel.getParameters()
    val parameterLength = parameter.nElement()
    val taskSize = parameterLength / partitionNum
    require(taskSize != 0, "parameter length should not less than partition number")
    val extraSize = parameterLength % partitionNum

    (0 until partitionNum).map(pid => {
      val start = pid * taskSize + math.min(pid, extraSize)
      val length = taskSize + (if (pid < extraSize) 1 else 0)
      parameter.narrow(1, start + 1, length).copy(weights(pid))
      gradientParameter.narrow(1, start + 1, length).copy(gradients(pid))
    })

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


  override def setTrainData(sampleRDD: RDD[Sample[T]],
    batchSize: Int,
    miniBatch: MiniBatch[T]): this.type = {
    this.dataset = (DataSet.rdd(sampleRDD) ->
      SampleToMiniBatch(miniBatch, batchSize, None))
      .asInstanceOf[DistributedDataSet[MiniBatch[T]]]
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
    this
  }


  override def prepareInput(): Unit = {
    import DistriOptimizer._
    if (!dataset.asInstanceOf[DistributedDataSet[MiniBatch[T]]].isCached) {
      logger.info("caching training rdd ...")
      dataset.asInstanceOf[DistributedDataSet[MiniBatch[T]]].cache()
    }
  }

  override def optimize(): Module[T] = {

    val distDataset = dataset.asInstanceOf[DistributedDataSet[MiniBatch[T]]]

    optimMethod.clearHistory()
    optimMethod.loadFromTable(state)
    state("dropPercentage") = dropPercentage
    state("warmupIterationNum") = warmupIterationNum
    state("computeThresholdbatchSize") = computeThresholdbatchSize
    state("maxDropPercentage") = maxDropPercentage
    state("isLayerwiseScaled") = Utils.isLayerwiseScaled(_model)

    val nodeNumber = Engine.nodeNumber()
    val coresPerNode = Engine.coreNumber()

    val partitionNum = distDataset.originRDD().partitions.length
    val size = model.getParameters()._1.nElement()
    val parameters = AllReduceParameter.newParameter(partitionNum, size)

    prepareInput()

    models = DistriOptimizer.initThreadModels(model, distDataset, criterion, state,
      nodeNumber, coresPerNode, checkSingleton, parameters, validationMethods, optimMethod)

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
          optimMethod,
          parameters,
          validationTrigger,
          validationDataSet,
          validationMethods,
          checkpointTrigger,
          checkpointPath,
          trainSummary,
          validationSummary,
          isOverWrite,
          gradientClippingParams
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
            val methodFile = getLatestFile(checkpointPath.get, "optimMethod")
            val modelFile = getLatestFile(checkpointPath.get, "model")
            clearState()
            models.unpersist()

            var newModel: Module[T] = null
            if (methodFile != null && modelFile != null) {
              newModel = Module.load[T](modelFile)
              optimMethod = OptimMethod.load[T](methodFile)
              DistriOptimizer.logger.info("Recover from last snapshot")
            } else {
              newModel = model
              DistriOptimizer.logger.info("Recover from origin model")
            }
            optimMethod.clearHistory()
            models = DistriOptimizer.initThreadModels(newModel, distDataset, criterion, state,
              nodeNumber, coresPerNode, checkSingleton, parameters, validationMethods, optimMethod)
          } else {
            throw t
          }
      }
    }

    DistriOptimizer.getModel(models, parameters, model)

    // Reset some internal states, so this or other optimizers can run optimize again
    clearState()

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
}


