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

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.mkldnn.MklDnnContainer
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
import com.intel.analytics.bigdl.nn.{Container, Utils}
import com.intel.analytics.bigdl.optim.DistriOptimizer._
import com.intel.analytics.bigdl.parameters.AllReduceParameter
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.{Module, _}
import java.io.{File, FilenameFilter}
import java.text.SimpleDateFormat
import java.util.Calendar
import org.apache.log4j.Logger
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Future
import scala.reflect.{ClassTag, classTag}

object ParallelOptimizer extends AbstractOptimizer {

  import Optimizer._

  val logger: Logger = Logger.getLogger(getClass)

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
   * @param validationTrigger validation trigger
   * @param validationDataSet validation dataset
   * @param validationMethods validation methods
   * @param cacheTrigger cache trigger
   * @param cachePath cache path
   * @param trainSummary train summary
   * @param validationSummary validation summary
   * @param isOverWrite if overwrite the checkpoint
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
    validationTrigger: Option[Trigger],
    validationDataSet: Option[DataSet[MiniBatch[T]]],
    validationMethods: Option[Array[ValidationMethod[T]]],
    cacheTrigger: Option[Trigger],
    cachePath: Option[String],
    trainSummary: Option[TrainSummary],
    validationSummary: Option[ValidationSummary],
    isOverWrite: Boolean
  )(implicit ev: TensorNumeric[T]): Unit = {
    val sc = dataset.originRDD().sparkContext
    val partitionNum = dataset.originRDD().partitions.length
    var wallClockTime = 0L
    var lastEpochTime = 0L

    // driverState is needed to prevent serializing the whole optimizer
    optimMethods.values.foreach { optimMethod =>
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

    require(_subModelNumber == 1, "currently only single model supported especially for mkldnn")

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
    val iterationPerTime = System.getProperty("bigdl.parallelOptimizer." +
      "iterationPerTime", "1").toInt
    val driverSubModelNum = partitionNum * _subModelNumber * iterationPerTime
    var dropModelNumBatch = 0
    var lossArray = new Array[Double](_subModelNumber)

    var epochStart = System.nanoTime()
    var dataRDD = dataset.data(train = true)

    while (!endWhen(driverState)) {
      var lossSum = 0.0
      var recordsNum = 0
      metrics.set("computing time for each node", mutable.ArrayBuffer[Double](), sc)
      metrics.set("computing time average", 0.0, sc, partitionNum)

      val driverMetrics = metrics
      val start = System.nanoTime()
      /*
        Run the forwards/backwards pass using multiple threads in each partition, and track the
        number of model updates that finished before the thread timeout mechanism.
       */
      val (numFinishedModelUpdates, localLossSum, localRecordsNum) = dataRDD
        .zipPartitions(models, preservesPartitioning = true) { (data, modelIter) => {
          var count = 0
          var finishedThreadSize = 0
          val cached = modelIter.next()
          // val miniBatchBuffer = new Array[MiniBatch[T]](_subModelNumber)
          var miniBatch: MiniBatch[T] = null
          while (count < iterationPerTime) {
            val syWStart = System.nanoTime()
            miniBatch = data.next()
            // ======================Start train models===================================
            var time = System.nanoTime()
            if (dropPercentage > 0.0 && iteration > warmupIterationNum +
              computeThresholdbatchSize - 1) {
              timeout = threshold
            }
            val pre = (iteration % computeThresholdbatchSize) * _subModelNumber
            val trainingThreads = Engine.default.invokeAndWait2(Seq(() => {
              val trainStart = System.nanoTime()
              val localModel = cached.localModels(0)
              localModel.training()
              val localCriterion = cached.localCriterions(0)
              val input = miniBatch.getInput()
              val target = miniBatch.getTarget()
              val output = localModel.forward(input)
              lossArray(0) = ev.toType[Double](localCriterion.forward(output, target))
              val errors = localCriterion.backward(output, target)
              localModel.backward(input, errors)
              cached.moduleTimeList(0 + pre) = System.nanoTime() - trainStart
              0
            }), timeout)
            val computingTime = System.nanoTime() - time
            driverMetrics.add("computing time average", computingTime)
            driverMetrics.add("computing time for each node", computingTime)

            val finishedThreads = trainingThreads.filter(!_.isCancelled).map(_.get())
            val currFinishedSize = finishedThreads.size
            finishedThreadSize += currFinishedSize
            recordsNum += currFinishedSize * miniBatch.size
            var i = 0
            while (i < currFinishedSize) {
              lossSum += lossArray(finishedThreads(i))
              i += 1
            }
            count += 1
          }
          val end = System.nanoTime()
          wallClockTime += end - start
          Iterator.single(finishedThreadSize, lossSum, recordsNum)
        }
        }.reduce((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3))

      dropModelNumBatch += (driverSubModelNum - numFinishedModelUpdates)

      if (dropPercentage == 0.0 ||
        numFinishedModelUpdates >= driverSubModelNum * (1.0 - maxDropPercentage)) {
        driverState("numFinishedModel") = numFinishedModelUpdates
        recordsProcessedThisEpoch += localRecordsNum
        val end = System.nanoTime()
        wallClockTime += end - start
        driverState("Loss") = localLossSum / numFinishedModelUpdates
        optimMethods.foreach { v =>
          v._2.updateHyperParameter()
        }

        driverState(s"LearningRate") = optimMethods.head._2.getLearningRate().toFloat

        driverState("Throughput") = localRecordsNum.toFloat / ((end - start) / 1e9f)
        val _header = header(driverState[Int]("epoch"), recordsProcessedThisEpoch, numSamples,
          driverState[Int]("neval"), wallClockTime)
        logger.info(s"${_header} Trained ${localRecordsNum} records in ${(end - start) / 1e9} " +
          s"seconds. Throughput is ${driverState("Throughput")} records/second. Loss is ${
            driverState("Loss")
          }.")
        logger.debug("\n" + metrics.summary())
        logger.debug("Dropped modules: " + (driverSubModelNum - numFinishedModelUpdates))
        lossArray = new Array[Double](_subModelNumber)

        iteration += 1
        if (dropPercentage > 0.0 && iteration > warmupIterationNum &&
          iteration % computeThresholdbatchSize == 0) {
          val moduleTimeList = models.mapPartitions { iter =>
            iter.next().moduleTimeList.iterator
          }.collect()

          val k = (dropPercentage * computeThresholdbatchSize * driverSubModelNum).toInt
          if (k > dropModelNumBatch) {
            threshold = Util.kthLargest(moduleTimeList, 0, moduleTimeList.length - 1,
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
        driverState("neval") = driverState[Int]("neval") + iterationPerTime
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

        // update parameters for last iteration
        if (endWhen(driverState)) {
          logger.info(s"training finished, updating all layers parameters")
          models.mapPartitions(modelIter => {
            val localModels = modelIter.next.localModels
            val updateTaskes = localModels.map(localModel => () => {
              updateLayerParameters(localModel)
            })
            Engine.default.invokeAndWait2(updateTaskes)
            Iterator.empty
          }).collect
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
            null,
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
          null,
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

  private def updateLayerParameters[T: ClassTag](module: Module[T]): Unit = {
    module.updateParameter
    if (module.isInstanceOf[Container[_, _, T]]) {
      module.asInstanceOf[Container[_, _, T]].modules.foreach(sub => {
        updateLayerParameters(sub)
      })
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
   * @param validationMethods validation methods
   * @param optimMethod optimization method
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
    validationMethods: Option[Array[ValidationMethod[T]]],
    optimMethod: Map[String, OptimMethod[T]],
    priorities: mutable.Map[String, Int]
  )(implicit ev: TensorNumeric[T]) = {
    val sc = dataset.originRDD().sparkContext
    val broadcast = sc.broadcast((criterion, state, validationMethods, optimMethod))
    val modelBroadcast = ModelBroadcast[T]().broadcast(sc, model)
    model.getParameters()
    val _subModelNumber = Engine.getEngineType match {
      case MklBlas => coresPerNode
      case MklDnn => 1
    }

    require(dataset.originRDD().partitions.length == nodeNumber,
      s"Passed in rdd partition number ${dataset.originRDD().partitions.length}" +
        s" is not equal to configured node number ${nodeNumber}")

    val computeThresholdbatchSize = state.get[Int]("computeThresholdbatchSize").get
    val nExecutor = Engine.nodeNumber()

    val parameterBlocks = System.getProperty("bigdl.parallelOptimizer." +
      "parameterBlocks", "10").toInt

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
      Engine.setNodeAndCore(nExecutor, coresPerNode)
      // initialize synchronizer with partition ID and parition number
      val synchronizer = new BlockManagerParameterSynchronizer[T](partitionId, nExecutor)
      val cached = (0 until _subModelNumber).map { _ =>
        val localModel = modelBroadcast.value(true, false)
        localModel match {
          case container: MklDnnContainer => container.compile(TrainingPhase)
          case _ =>
        }
        // differentiate partition models from each other by partition ID
        setModelId(localModel, partitionId)
        // set parameter synchronizer
        setDistriPartitionsynchronizer(localModel, synchronizer, new mutable.HashMap[Int, Int](),
          parameterBlocks)
        val localCriterion = broadcastCriterion.cloneCriterion()
        val localState = broadcastState.clone()
        val localMethod = if (broadcastMethod.isDefined) {
          Some(broadcastMethod.get.map(_.clone()))
        } else None
        (localModel, Tensor[T](0), Tensor[T](0), localCriterion, localState, localMethod)
      }.toArray

      logger.info("model thread pool size is " + Engine.model.getPoolSize)
      val weights = cached.head._2
      Iterator.single(CacheV1(
        cached.map(_._1), // models
        cached.map(_._2), // weights
        cached.map(_._3), // gradients
        cached.map(_._4), // criterions
        cached.map(_._5), // states
        new Array[Long](_subModelNumber * computeThresholdbatchSize),
        cached.map(_._6),
        broadcastOptim.map(v => (v._1, v._2.clone())),
        synchronizer
      ))
    }).persist()
    models.setName("Thread Model RDD")
    logger.info("Cache thread models...")
    models.count()
    logger.info("Cache thread models... done")
    models
  }

  private def getExecutionOrder[T: ClassTag](module: Module[T]): ArrayBuffer[Module[T]] = {
    val res = new ArrayBuffer[Module[T]]
    if (module.isInstanceOf[Container[_, _, T]]) {
      val subModules = module.asInstanceOf[Container[_, _, T]].modules
      subModules.foreach(sub => {
        res ++= getExecutionOrder(sub)
      })
    } else {
      if (module.parameters() != null) {
        res += module
      }
    }
    res
  }

  private def setDistriPartitionsynchronizer[T: ClassTag](model: Module[T],
    parameterSynchronizer: DistriParameterSynchronizer[T],
    barrierLayers: mutable.Map[Int, Int], slices: Int): Unit = {
    val globalWeights = model.getParameters()._1
    val globalGrads = model.getParameters()._2
    val totalSize = globalGrads.nElement
    val executorOrders = getExecutionOrder(model)
    var i = executorOrders.length - 1
    val size = totalSize / slices - 1
    val extraSize = totalSize - size * (slices - 1)
    var lastOffSet = totalSize
    while (i >= 0) {
      val currModule = executorOrders(i)
      if (currModule.parameters() != null) {
        val grads = currModule.getParameters()._1
        val offSet = grads.storageOffset - 1
        val index = if (offSet == 0) 0 else (offSet - 1) / size + 1
        val currParSize = lastOffSet - offSet
        if (index < slices) {
          if (!barrierLayers.contains(index)) {
            barrierLayers.put(index, offSet)
            val weightsPar = globalWeights.narrow(1, offSet + 1, currParSize)
            val gradsPar = globalGrads.narrow(1, offSet + 1, currParSize)
            parameterSynchronizer.init(currModule.getName, currParSize,
              executorOrders.length - i, weightsPar, gradsPar)
            currModule.setParameterSynchronizer(parameterSynchronizer)
            lastOffSet = offSet
          }
        }
      }
      i -= 1
    }
  }

  private def setModelId[T: ClassTag](model: Module[T], partitionId: Int): Unit = {
    model.setId(partitionId)

    if (model.isInstanceOf[Container[_, _, T]]) {
      model.asInstanceOf[Container[_, _, T]].modules.
        foreach(sub => setModelId(sub, partitionId))
    }
  }

  /**
   * Fetch current model parameters to driver, and copy to trainingModel.
   *
   * @param models cached models
   * @param trainingModel the model is trained by optimizer
   * @return trained model
   */
  override protected def getModel[T: ClassTag](
    models: RDD[Cache[T]],
    parameters: AllReduceParameter[T],
    trainingModel: Module[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    val partitionNum = models.partitions.length
    val extraState = models.map(_.localModels.head.getExtraParameter()).first()
    trainingModel.setExtraParameter(extraState)
    // make sure gradient is as the same length as weight
    val parameterArray = trainingModel.parameters()
    (0 until parameterArray._2.length).foreach(i =>
      parameterArray._2(i).resizeAs(parameterArray._1(i))
    )

    val parameter = trainingModel.getParameters()._1
    val _classTag = classTag[T]
    val size = parameter.storage().array().length
    val taskSize = size / partitionNum
    val extraSize = size % partitionNum
    val weights = models.mapPartitions(iter => {
      val localCache = iter.next
      val localModels = localCache.localModels
      val localWeights = localModels.head.getParameters()._1
      val synchronizer = localCache.parameterSynchronizer
        .asInstanceOf[BlockManagerParameterSynchronizer[T]]
      val partitionId = synchronizer.partitionID
      val start = partitionId * taskSize + math.min(partitionId, extraSize)
      val length = taskSize + (if (partitionId < extraSize) 1 else 0)
      val partitionWeight = Tensor[T](length)
      partitionWeight.copy(localWeights.narrow(1, start + 1, length))
      Iterator.single(Map(partitionId -> partitionWeight))
    }).reduce((a, b) => (a ++ b))

    (0 until partitionNum).map(pid => {
      val start = parameter.storageOffset + pid * taskSize + math.min(pid, extraSize)
      val length = taskSize + (if (pid < extraSize) 1 else 0)
      parameter.narrow(1, start, length).copy(weights(pid))
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
class ParallelOptimizer[T: ClassTag](
  _model: Module[T],
  _dataset: DistributedDataSet[MiniBatch[T]],
  _criterion: Criterion[T]
)(implicit ev: TensorNumeric[T])
  extends Optimizer[T, MiniBatch[T]](
    _model, _dataset, _criterion) {
  val metrics = new Metrics

  private var models: RDD[DistriOptimizer.CacheV1[T]] = null

  private var _priorities: mutable.Map[String, Int] = null

  def setPriorities(priorities: mutable.Map[String, Int]): Unit = {
    this._priorities = priorities
  }

  /**
   * Clean some internal states, so this or other optimizers can run optimize again
   *
   * This method will be called at the end of optimize. You need not call it if optimize succeed.
   * If the optimize fails, you may call it before next optimize.
   */
  def clearState(): Unit = {
    ParallelOptimizer.clearState(models.asInstanceOf[RDD[Cache[T]]])
  }

  private def endEpoch(): Unit = {
    ParallelOptimizer.endEpoch(optimMethods)
  }

  override def setTrainData(sampleRDD: RDD[Sample[T]],
    batchSize: Int,
    miniBatch: MiniBatch[T]): this.type = {
    this.dataset = ParallelOptimizer.setTrainData(sampleRDD, batchSize, miniBatch)
    // if current epoch is not finished, we will end the
    // current epoch and start a new epoch when optimize is called
    endEpoch()
    this
  }

  override def setTrainData(sampleRDD: RDD[Sample[T]],
    batchSize: Int,
    featurePaddingParam: PaddingParam[T] = null,
    labelPaddingParam: PaddingParam[T] = null): this.type = {
    val _featurePaddingParam = if (featurePaddingParam != null) Some(featurePaddingParam) else None
    val _labelPaddingParam = if (labelPaddingParam != null) Some(labelPaddingParam) else None
    this.dataset = ParallelOptimizer.setTrainData(sampleRDD, batchSize,
      featurePaddingParam, labelPaddingParam)
    // if current epoch is not finished, we will end the
    // current epoch and start a new epoch when optimize is called
    endEpoch()
    this
  }


  override def prepareInput(): Unit = {
    if (!dataset.toDistributed().isCached) {
      ParallelOptimizer.logger.info("caching training rdd ...")
      ParallelOptimizer.prepareInput(this.dataset, this.validationDataSet)
    }
  }

  private def expandOptimMethods(optimMethodMap: mutable.Map[String, OptimMethod[T]]): Unit = {
    if (this.model.isInstanceOf[Container[_, _, T]]) {
      expandOptimMethodsForSubModules(this.model.
        asInstanceOf[Container[_, _, T]].modules,
        optimMethodMap(this.model.getName), optimMethodMap)
    } else {
      require(optimMethodMap.contains(this._model.getName),
        "single layer model should have optim method set")
    }

    if (optimMethodMap.contains(this.model.getName)) {
      this.model.setOptimMethod(optimMethodMap.get(this.model.getName).get)
    }
  }

  private def expandOptimMethodsForSubModules(subModules: ArrayBuffer[Module[T]],
    parentMethod: OptimMethod[T],
    optimMethodMap: mutable.Map[String, OptimMethod[T]]): Unit = {
    subModules.foreach(sub => {
      if (optimMethodMap.get(sub.getName) == None) {
        require(parentMethod != null, s"${sub.getName}'s parent optim method should not be null")
        val subOptimMethod = parentMethod.clone
        sub.setOptimMethod(subOptimMethod)
        optimMethodMap(sub.getName) = subOptimMethod
      }
      if (sub.isInstanceOf[Container[_, _, T]]) {
        val currMethod = optimMethodMap(sub.getName)
        expandOptimMethodsForSubModules(sub.asInstanceOf[Container[_, _, T]].modules,
          currMethod, optimMethodMap)
      }
    })
  }

  private def defaultPrioritize(): mutable.HashMap[String, Int] = {
    val priorities = new mutable.HashMap[String, Int]
    val orders = ParallelOptimizer.getExecutionOrder(this._model)
    val len = orders.size
    orders.zipWithIndex.foreach(order => {
      priorities.put(order._1.getName, len - order._2)
    })
    priorities
  }

  override def optimize(): Module[T] = {

    val distDataset = dataset.toDistributed()

    optimMethods.values.foreach { optimMethod =>
      optimMethod.clearHistory()
    }

    // To be compatible with the old usage that user define hyperparameters in a table.
    if (optimMethods.size == 1) {
      optimMethods.head._2.loadFromTable(state)
    }
    val parallelOptimMethods = scala.collection.mutable.Map(optimMethods.toSeq: _*)
    // expand optim method so that each layer has its own optim method
    expandOptimMethods(parallelOptimMethods)

    if (_priorities == null) {
      _priorities = defaultPrioritize
    }

    optimMethods = collection.immutable.Map(parallelOptimMethods.toSeq: _*)

    state("dropPercentage") = dropPercentage
    state("warmupIterationNum") = warmupIterationNum
    state("computeThresholdbatchSize") = computeThresholdbatchSize
    state("maxDropPercentage") = maxDropPercentage
    state("isLayerwiseScaled") = Utils.isLayerwiseScaled(_model)

    val nodeNumber = Engine.nodeNumber()
    val coresPerNode = Engine.coreNumber()

    val partitionNum = distDataset.originRDD().partitions.length

    prepareInput()

    models = ParallelOptimizer.initThreadModels(model, distDataset, criterion, state,
      nodeNumber, coresPerNode, checkSingleton, validationMethods,
      optimMethods, _priorities)

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

    ParallelOptimizer.optimize(
      model,
      distDataset,
      coresPerNode,
      state,
      endWhen,
      metrics,
      models.asInstanceOf[RDD[Cache[T]]],
      optimMethods,
      validationTrigger,
      validationDataSet,
      validationMethods,
      checkpointTrigger,
      checkpointPath,
      trainSummary,
      validationSummary,
      isOverWrite
    )

    ParallelOptimizer.getModel(models.asInstanceOf[RDD[Cache[T]]], null, model)

    // Reset some internal states, so this or other optimizers can run optimize again
    clearState()

    // release distributed synchronizer resources

    models.foreach(modelIter => {
      modelIter.parameterSynchronizer.clear
    })

    // unpersist the model because the next time optimize is called, new `models` will be
    // created
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
    files.map { file =>
      if (file.lastModified() > lastMod) {
        choice = file.getPath;
        lastMod = file.lastModified();
      }
    }
    return choice;
  }
}
