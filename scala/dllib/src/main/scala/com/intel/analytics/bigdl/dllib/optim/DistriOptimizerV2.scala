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

import java.io.File
import java.text.SimpleDateFormat
import java.util.Calendar

import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch, PaddingParam, Sample}
import com.intel.analytics.bigdl.models.utils.{CachedModels, ModelBroadcast}
import com.intel.analytics.bigdl.nn.{Container, Module}
import com.intel.analytics.bigdl.parameters.{AllReduceParameter, ParameterProcessor}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.utils.intermediate.ConversionUtils
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.{Module, _}
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, TaskContext}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Future
import scala.reflect.ClassTag

object DistriOptimizerV2 extends AbstractOptimizer {

  case class Cache[T](
    localModels: Array[Module[T]],
    modelWeights: Array[Tensor[T]],
    modelGradients: Array[Tensor[T]],
    localCriterions: Array[Criterion[T]],
    localStates: Array[Table],
    var moduleTimeList: Array[Long] = null,
    localMethods: Array[Option[Array[ValidationMethod[T]]]],
    var optimMethods: Map[String, OptimMethod[T]],
    parameterSynchronizer: DistriParameterSynchronizer[T] = null,
    parameter: AllReduceParameter[T] = null,
    parameterSplits: Map[String, (Int, Int)] = null,
    parameterProcessers: Array[ParameterProcessor] = null) extends DistriOptimizer.Cache[T]

  import Optimizer._

  private[DistriOptimizerV2] var _logger: Option[OptimizerLogger] = None

  def logger: OptimizerLogger = {
    if (_logger.isEmpty) {
      _logger = Some(new DistriLogger)
    }

    _logger.get
  }

  private[optim] def optimize[T: ClassTag](
    cacheOfMaster: MasterCache[T],
    cacheOfSlave: RDD[Cache[T]],
    dataset: DistributedDataSet[MiniBatch[T]],
    endWhen: Trigger,
    validationTrigger: Option[Trigger],
    validationDataSet: Option[DataSet[MiniBatch[T]]],
    validationMethods: Option[Array[ValidationMethod[T]]],
    cacheTrigger: Option[Trigger],
    cachePath: Option[String],
    trainSummary: Option[TrainSummary],
    validationSummary: Option[ValidationSummary],
    isOverWrite: Boolean,
    context: TrainingContext[T]
  )(implicit ev: TensorNumeric[T]): Unit = {
    val headOptimMethod = cacheOfMaster.optimMethods.values.head
    context.loadState(headOptimMethod.state)
    logger.info(s"config ${context.state}")

    if (headOptimMethod.state[Int](StateEntry.RECORDS_PROCESSED) == 0) {
      val shuffleBefore = System.nanoTime()
      logger.info("Shuffle data")
      dataset.shuffle()
      val shuffleEnd = System.nanoTime()
      logger.info(s"Shuffle data complete. Takes ${(shuffleEnd - shuffleBefore) / 1e9}s")
    }

    val sc = dataset.originRDD().sparkContext
    var trainingDataSet = dataset.data(train = true)
    val trainingTrace = TrainingTrace(headOptimMethod.state)

    while (!endWhen(context.state)) {
      iteration(sc, trainingDataSet, cacheOfSlave, cacheOfMaster, context, trainingTrace)

      if (context.hasCompleteAllSamples(trainingTrace.recordsOfEpoch, cacheOfMaster.model)) {
        dataset.shuffle()
        trainingDataSet = dataset.data(train = true)
      }

      val _header = header(
        trainingTrace.epochs,
        trainingTrace.recordsOfEpoch,
        context.numSamples,
        trainingTrace.iterations,
        trainingTrace.trainingTakes)

      validate(
        validationTrigger,
        validationDataSet,
        validationMethods,
        context.subModelNumber,
        cacheOfSlave.asInstanceOf[RDD[DistriOptimizer.Cache[T]]],
        context.state,
        validationSummary,
        _header,
        cacheOfMaster.parameter
      )

      checkpoint(
        cacheTrigger,
        cachePath,
        isOverWrite,
        trainingTrace.trainingTakes,
        cacheOfSlave.asInstanceOf[RDD[DistriOptimizer.Cache[T]]],
        context.state,
        cacheOfMaster.parameter,
        cacheOfMaster.optimMethods,
        cacheOfMaster.model
      )

      trainSummary.foreach { summary =>
        saveSummary(
          summary,
          cacheOfSlave.asInstanceOf[RDD[DistriOptimizer.Cache[T]]],
          context.state,
          cacheOfMaster.parameter,
          cacheOfMaster.model
        )
      }
    }
  }

  private def initMetrics(sc: SparkContext, metrics: Metrics, partitionNum: Int): Unit = {
    metrics.set(COMPUTING_TIME_EACH_NODE.value, mutable.ArrayBuffer[Double](), sc)
    metrics.set(GET_WEIGHTS_EACH_NODE.value, mutable.ArrayBuffer[Double](), sc)
    metrics.set(COMPUTING_TIME_AVERAGE.value, 0.0, sc, partitionNum)
    metrics.set(AGGREGATE_GRADIENT_TIME.value, 0.0, sc, partitionNum)
    metrics.set(GET_WEIGHTS_AVERAGE.value, 0.0, sc, partitionNum)
    metrics.set(PUT_GRADIENT.value, 0.0, sc, Engine.nodeNumber())
    metrics.set(AGGREGATE_PARTITION_GRADIENT.value, 0.0, sc, Engine.nodeNumber())
    metrics.set(COMPUTE_WEIGHT_AVERAGE.value, 0.0, sc, Engine.nodeNumber())
    metrics.set(SEND_WEIGHTS_AVERAGE.value, 0.0, sc, Engine.nodeNumber())
  }

  private def iteration[T: ClassTag](
    sc: SparkContext,
    dataRDD: RDD[MiniBatch[T]],
    models: RDD[Cache[T]],
    cacheOfMaster: MasterCache[T],
    context: TrainingContext[T], trainingTrace: TrainingTrace
    )(implicit ev: TensorNumeric[T]): Unit = {
    val lossSum = sc.doubleAccumulator("loss sum")
    val recordsNum = sc.doubleAccumulator("record number")
    val metrics = cacheOfMaster.metrics
    val partitionNum = cacheOfMaster.partitionNum
    initMetrics(sc, metrics, partitionNum)

    /*
      Run the forwards/backwards pass using multiple threads in each partition, and track the
      number of model updates that finished before the thread timeout mechanism.
     */
    trainingTrace.traceIteration({
      val successModels = dataRDD.zipPartitions(models, preservesPartitioning = true) {
      (data, iter) =>
        val cached = iter.next()
        /*
          Note: All models in `cached` share the same storage for weights, so we only need to
          copy the weights from parameter server into the first model's weights.
         */
        val offset = cached.parameter.paramOffset
        val size = cached.parameter.size
        val weights = cached.modelWeights.head.narrow(1, offset, size)

        val miniBatchBuffer = TrainingTrace.time (
          {
            val weightsResults = cached.parameter.getWeights(weights)
            val batch = context.fetchBatch(data)
            weightsResults.waitResult()
            batch
          },
          metrics
        )(Array(GET_WEIGHTS_AVERAGE, GET_WEIGHTS_EACH_NODE))

        val results = train(cached, miniBatchBuffer, context, metrics)

        lossSum.add(results.loss)
        recordsNum.add(results.records)

        Iterator.single(results.successed)
      }.reduce(_ + _)

      parameterSync(lossSum.value, successModels, cacheOfMaster, models, context)
    })

    driverStatesUpdate(cacheOfMaster, (recordsNum.value).toInt,
      context, trainingTrace, metrics)
  }

  /**
   * Init engine and cache models, weights, gradients, criterions, state tables
   * and validation methods on worker nodes.
   *
   * @param model train model
   * @param dataset train dataset
   * @param criterion loss function
   * @param state state table
   * @param allReduceParameter all reduce parameter instance
   * @param validationMethods validation methods
   * @param optimMethod optimization method
   * @return cached models
   */
  private def initCacheOfSlave[T: ClassTag](
    cacheOfMaster: MasterCache[T],
    dataset: DistributedDataSet[MiniBatch[T]],
    context: TrainingContext[T])(
    implicit ev: TensorNumeric[T]): (RDD[Cache[T]], ModelBroadcast[T]) = {
    case class TrainingConfig[T](criterion: Criterion[T],
      validationMethods: Option[Array[ValidationMethod[T]]],
      optimMethods: Map[String, OptimMethod[T]],
      parameterSplits: Map[String, (Int, Int)],
      parameterProcessers: Array[ParameterProcessor]
    )

    val config = TrainingConfig(
      cacheOfMaster.criterion,
      cacheOfMaster.validationMethods,
      cacheOfMaster.optimMethods,
      cacheOfMaster.parameterSplits,
      cacheOfMaster.parameterProcessers)

    val sc = dataset.originRDD().sparkContext
    val broadcast = sc.broadcast(config)

    val model = ConversionUtils.convert(cacheOfMaster.model)
    // ensure model's parameter is compacted for getting a better performance when broadcasting
    model.getParameters()
    // As cloneModel is using Serialization to implement deep copy, and will throw OOMError
    // when model's size is bigger than SerializationUtils' buffer size. So we can use
    // ModelBroadcast to clone model here.
    // Notes: All models returned by modelBroadcast.value() share the same weight&bias, while
    // gradWeight&gradBias is unshared.
    val modelBroadcast = ModelBroadcast[T]().broadcast(sc, model)

    val nExecutor = Engine.nodeNumber()
    val executorCores = Engine.coreNumber()
    val allReduceParameter = cacheOfMaster.parameter

    val subModelNumber = context.subModelNumber
    val state = context.state

    val cache = dataset.originRDD().mapPartitions(_ => {
      val partitionId = TaskContext.getPartitionId
      val config = broadcast.value

      val replicas = (0 until subModelNumber).map { _ =>
        val localModel = modelBroadcast.value(true)
        val localCriterion = config.criterion.cloneCriterion()
        val localState = state.clone()
        val localMethod = if (config.validationMethods.isDefined) {
            Some(config.validationMethods.get.map(_.clone()))
        } else {
          None
        }
        val (weights, grads) = localModel.getParameters()

        // at last, we bind the model to partition id
        setModelId(localModel, partitionId)

        Replica(localModel, weights, grads, localCriterion, localState, localMethod)
      }.toArray

      logger.info("model thread pool size is " + Engine.model.getPoolSize)

      // now we should init the all reduce parameters by the weights
      // note: only get the head, because they are same in the array
      val offset = allReduceParameter.paramOffset
      val size = allReduceParameter.size
      allReduceParameter.init(replicas.head.weights.narrow(1, offset, size))

      Iterator.single(Cache(
        replicas.map(_.model),
        replicas.map(_.weights),
        replicas.map(_.gradients),
        replicas.map(_.criterion),
        replicas.map(_.state),
        new Array[Long](subModelNumber),
        replicas.map(_.validationMethods),
        config.optimMethods.map(v => (v._1, v._2.clone())),
        null,
        allReduceParameter,
        config.parameterSplits,
        config.parameterProcessers
      ))
    }).persist()

    cache.setName("Thread Model RDD")
    logger.info("Cache thread models...")
    cache.count()
    logger.info("Cache thread models... done")
    (cache, modelBroadcast)
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
   * @param parameters [[AllReduceParameter]]
   * @param trainingModel the model is trained by optimizer
   * @return trained model
   */
  override protected def getModel[T: ClassTag](
    models: RDD[DistriOptimizer.Cache[T]],
    parameters: AllReduceParameter[T],
    trainingModel: Module[T])(implicit
    ev: TensorNumeric[T])
  : Module[T] = {
    val partitionNum = models.partitions.length
    val extraState = models.map(_.localModels.head.getExtraParameter()).first()
    trainingModel.setExtraParameter(extraState)

    // make sure gradient is as the same length as weight
    val parameterArray = trainingModel.parameters()
    (0 until parameterArray._2.length).foreach(i =>
      parameterArray._2(i).resizeAs(parameterArray._1(i))
    )

    val (parameter, gradientParameter) = trainingModel.getParameters()


    val (weights, gradients) = models.mapPartitions(iter => {
      val cached = iter.next()
      val curPartitionId = TaskContext.getPartitionId()
      Iterator.single((Map(curPartitionId -> parameters.weightPartition),
        Map(curPartitionId -> parameters.gradientPartition)))
    }).reduce((a, b) => (a._1 ++ b._1, a._2 ++ b._2))

    val taskSize = parameters.size / partitionNum
    require(taskSize != 0, "parameter length should not less than partition number")
    val extraSize = parameters.size % partitionNum

    (0 until partitionNum).map(pid => {
      val start = parameters.paramOffset + pid * taskSize + math.min(pid, extraSize)
      val length = taskSize + (if (pid < extraSize) 1 else 0)
      parameter.narrow(1, start, length).copy(weights(pid))
      gradientParameter.narrow(1, start, length).copy(gradients(pid))
    })

    trainingModel
  }


  private case class TrainingResults(successed: Int, loss: Double, records: Int)
  private def train[T: ClassTag](
    cached: Cache[T],
    data: Array[MiniBatch[T]],
    context: TrainingContext[T],
    metrics: Metrics)(implicit ev: TensorNumeric[T]): TrainingResults = {
    val stackSize = data.head.size()
    var tasks: ArrayBuffer[Future[_]] = new ArrayBuffer()

    // ======================Start train models===================================
    val modelsResult = TrainingTrace.time (
      context.train(data, cached.localModels, cached.localCriterions),
      metrics
    )(Array(COMPUTING_TIME_EACH_NODE, COMPUTING_TIME_AVERAGE))

    var lossSum = 0.0
    var i = 0
    while (i < modelsResult.size) {
      lossSum += modelsResult(i).loss
      cached.moduleTimeList(i) = modelsResult(i).elapsed
      i += 1
    }

    val gradients = TrainingTrace.time (
      {
        if (modelsResult.nonEmpty) {
          val successedGradients = modelsResult.map {
            result => cached.modelGradients(result.index)
          }.toArray
          context.aggregate(successedGradients)
        } else {
          cached.modelGradients(0).zero()
          cached.modelGradients(0)
        }
      }, metrics)(Array(AGGREGATE_GRADIENT_TIME))

    TrainingTrace.time (
      cached.parameter.putGradients(gradients),
      metrics
    )(Array(PUT_GRADIENT))

    tasks ++= Engine.default.invoke {
      (0 until context.subModelNumber).map { i =>
        () => {
          cached.localModels(i).training()
          cached.localModels(i).zeroGradParameters()
        }
      }
    }

    TrainingResults(modelsResult.size, lossSum, modelsResult.size * stackSize)
  }

  private def updateStates[T](optimMethods: Map[String, OptimMethod[T]], state: Table,
    updateScore: Boolean): Unit = {
    import StateEntry._
    optimMethods.map { case (moduleName, optimMethod) =>
      optimMethod.state.update(EPOCH, state[Int](EPOCH))
      optimMethod.state.update(NEVAL, state[Int](NEVAL))
      optimMethod.state.update(LOSS, state[Float](LOSS))
      if (updateScore) {
        optimMethod.state.update(SCORE, state[Float](SCORE))
      }

      if (optimMethod.state.keySet.contains(RECORDS_PROCESSED)) {
        optimMethod.state.update(RECORDS_PROCESSED, state[Int](RECORDS_PROCESSED))
      }
    }
  }

  private def driverStatesUpdate[T: ClassTag](
    cacheOfMaster: MasterCache[T],
    recordsNum: Int,
    context: TrainingContext[T],
    trainingTrace: TrainingTrace,
    metrics: Metrics)(
    implicit ev: TensorNumeric[T]): Unit = {
    val optimMethods = cacheOfMaster.optimMethods
    val updateScore = cacheOfMaster.validationMethods.isDefined

    optimMethods.foreach { v =>
      v._2.updateHyperParameter()
    }

    val trainingTakes = trainingTrace.trainingTakes
    val iterationTakes = trainingTrace.iterationTakes
    val throughput = recordsNum.toFloat / (iterationTakes / 1e9f)
    val records = trainingTrace.updateRecords(recordsNum).recordsOfEpoch
    val _header = header(trainingTrace.epochs, records, context.numSamples,
      trainingTrace.iterations, trainingTakes)
    val loss = context.state[Float](StateEntry.LOSS)
    logger.info(s"${_header} Trained $recordsNum records in ${(iterationTakes) /1e9f} seconds. " +
      s"Throughput is $throughput records/second. " +
      s"Loss is $loss. " +
      s"${getHyperParameterLog(optimMethods)}")
    logger.debug("\n" + metrics.summary())

    context.state(StateEntry.THROUGHPUT) = recordsNum.toFloat / (iterationTakes / 1e9f)
    context.state(StateEntry.NEVAL) = trainingTrace.iterations + 1
    // for next iteration training
    context.state(StateEntry.LEARNING_RATE) = optimMethods.head._2.getLearningRate().toFloat

    if (context.hasCompleteAllSamples(trainingTrace.recordsOfEpoch, cacheOfMaster.model)) {
      // Epoch is finished
      trainingTrace.startNewEpoch()
      logger.info(s"${_header} Epoch finished. Wall clock time is ${trainingTakes / 1e6} ms")
    }
    context.state(StateEntry.EPOCH) = trainingTrace.epochs
    context.state(StateEntry.RECORDS_PROCESSED) = trainingTrace.recordsOfEpoch

    updateStates(optimMethods, context.state, updateScore)
  }

  private def parameterSync[T: ClassTag](
    lossSum: Double,
    successedModels: Int,
    cacheOfMaster: MasterCache[T],
    cacheOfSlave: RDD[Cache[T]],
    context: TrainingContext[T])(implicit ev: TensorNumeric[T]): Unit = {
    val metrics = cacheOfMaster.metrics
    val parameter = cacheOfMaster.parameter
    val updateScore = cacheOfMaster.validationMethods.isDefined

    context.state(StateEntry.NUM_FINISHED_MODELS) = successedModels
    context.state(StateEntry.IS_GRADIENT_UPDATED) = false
    cacheOfMaster.parameterProcessers.foreach { processer =>
      processer.collectGlobalData(cacheOfSlave.asInstanceOf[RDD[DistriOptimizer.Cache[T]]],
        parameter, metrics, context.state)
    }

    val isGradientUpdated = context.state[Boolean](StateEntry.IS_GRADIENT_UPDATED)
    cacheOfSlave.mapPartitions { iter =>
      val cache = iter.next()
      val localOptimMethods = cache.optimMethods
      val parameterProcessers = cache.parameterProcessers
      val parameterSplits = cache.parameterSplits
      val (paramLocalStart, paramLocalLen) = cache.parameter.localPartitionRange

      // if parameterProcesser has aggregated gradient, we can skip this aggregation.
      if (!isGradientUpdated) {
        TrainingTrace.time (
          cache.parameter.aggregateGradientPartition(successedModels),
          metrics
        )(Array(AGGREGATE_PARTITION_GRADIENT))
      }

      parameterProcessers.foreach(_.processParameters(parameter, cache, context.state))

      updateStates(localOptimMethods, context.state, updateScore)

      val optimSegments = localOptimMethods.map {
        case (name, method) =>
          val p = parameterSplits(name)
          val startIdx = Math.max(paramLocalStart, p._1)
          val endIdx = Math.min(paramLocalLen + paramLocalStart, p._1 + p._2)
          (name, ParamSegments(startIdx - paramLocalStart + 1, endIdx - startIdx, method))
      }

      val weights = cache.parameter.weightPartition
      val gradients = cache.parameter.gradientPartition
      val loss = lossSum / successedModels

      TrainingTrace.time (
        context.update(optimSegments, weights, gradients, loss),
        metrics
      )(Array(COMPUTE_WEIGHT_AVERAGE))

      TrainingTrace.time (
        cache.parameter.sendWeightPartition(), metrics
      )(Array(SEND_WEIGHTS_AVERAGE))

      Iterator.empty
    }.count()

    context.state(StateEntry.IS_GRADIENT_UPDATED) = true
    context.state(StateEntry.LOSS) = lossSum.toFloat / successedModels
  }
}

/**
 * The optimizer run on a distributed cluster.
 *
 * @param _model train model
 * @param _dataset train dataset
 * @param _criterion loss function
 */
class DistriOptimizerV2[T: ClassTag](
  _model: Module[T],
  _dataset: DistributedDataSet[MiniBatch[T]],
  _criterion: Criterion[T]
)(implicit ev: TensorNumeric[T])
  extends Optimizer[T, MiniBatch[T]](
    _model, _dataset, _criterion) {
  private var _context: Option[TrainingContext[T]] = None
  private var _canBeReused: Boolean = false

  def setContext(context: TrainingContext[T]): Unit = _context = Some(context)

  def resetContext(): this.type = { _context = None
    this
  }

  def context: TrainingContext[T] = {
    if (_context.isEmpty) {
      val subModelNumber = Engine.getEngineType() match {
        case MklBlas => Engine.coreNumber()
        case MklDnn => 1
      }

      DistriOptimizer.logger.info("Count dataset")
      val countBefore = System.nanoTime()
      val numSamples = dataset.toDistributed().data(train = false).map(_.size()).reduce(_ + _)
      val countAfter = System.nanoTime()
      DistriOptimizer.logger.info(
        s"Count dataset complete. Time elapsed: ${(countAfter - countBefore) / 1e9}s")

      if (numSamples != dataset.size()) {
        DistriOptimizer.logger.warn("""
            If the dataset is built directly from RDD[Minibatch], the data in each
            minibatch is fixed, and a single minibatch is randomly selected in each partition. If
            the dataset is transformed from RDD[Sample], each minibatch will be constructed on the
            fly from random samples, which is better for convergence.""")
      }

      val state = T()

      _context = Some(new TrainingContext(subModelNumber, numSamples, state))
    }

    _context.get
  }

  private var _allReduceParameter: AllReduceParameter[T] = _
  private var _parameterSplits: Map[String, (Int, Int)] = _
  private var cacheOfSlave: RDD[DistriOptimizerV2.Cache[T]] = null
  // this variable is used to check the models cloned when broadcast, if there're native resources,
  // it will be deleted at the end of Optimizer.
  private var modelBroadcast: ModelBroadcast[T] = null

  /**
   * Clean some internal states, so this or other optimizers can run optimize again
   *
   * This method will be called at the end of optimize. You need not call it if optimize succeed.
   * If the optimize fails, you may call it before next optimize.
   */
  def clearState(): Unit = {
    DistriOptimizerV2.clearState(cacheOfSlave.asInstanceOf[RDD[DistriOptimizer.Cache[T]]])
  }


  // By default, optimMethod internal state for each worker will not be reserved and reuse.
  private var reserveOptimMethod = false
  private[bigdl] var previousOptim: RDD[Map[String, OptimMethod[T]]] = null
  /**
   * If you want to reserve optimMethod for each worker, and reuse those methods in
   * next training task, you can call it.
   */

  /**
   * If you want to reserve optimMethod for each worker and reuse those methods in
   * next training task, please set reserve = true
   * Otherwise, if just using optimMethod you set in optimizer, please set reserve = false
   *
   * @param reserve whether to reserve optim method for each worker
   * @return
   */
  override def reserveOptim(reserve: Boolean): this.type = {
    reserveOptimMethod = reserve
    this
  }

  // replace optim methods with previous
  private def resetOptimMethods[T: ClassTag](
    models: RDD[DistriOptimizerV2.Cache[T]],
    previousOptimMethods: RDD[Map[String,
      OptimMethod[T]]]):
  RDD[DistriOptimizerV2.Cache[T]] = {
    models.zipPartitions(previousOptimMethods) { (m1, m2) => {
      val cache = m1.next()
      cache.optimMethods = m2.next()
      Iterator(cache)
    }
    }
  }

  private def endEpoch(): Unit = {
    DistriOptimizer.endEpoch(optimMethods)
  }

  override def setTrainData(sampleRDD: RDD[Sample[T]],
    batchSize: Int,
    miniBatch: MiniBatch[T]): this.type = {
    this.dataset = DistriOptimizer.setTrainData(sampleRDD, batchSize, miniBatch)
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
    this.dataset = DistriOptimizer.setTrainData(sampleRDD, batchSize,
      featurePaddingParam, labelPaddingParam)
    // if current epoch is not finished, we will end the
    // current epoch and start a new epoch when optimize is called
    endEpoch()
    this
  }

  override def prepareInput(): Unit = {
    if (!dataset.toDistributed().isCached) {
      DistriOptimizer.logger.info("caching training rdd ...")
      DistriOptimizer.prepareInput(this.dataset, this.validationDataSet)
    }
  }

  override def optimize(): Module[T] = {
    require(validArgs(), "please check the args you set, there's some wrong")

    val modelParameters = model.getParameters()
    val size = modelParameters._1.nElement()
    val partitionNum = dataset.toDistributed().originRDD().partitions.length
    val reuse = _canBeReused &&
      _allReduceParameter != null && _parameterSplits != null

    if (!reuse) {
      _allReduceParameter = AllReduceParameter.newParameter[T](partitionNum, size)
      _parameterSplits = initOptimMethods(optimMethods, modelParameters._1)
    }

    prepareInput()

    val cacheOfMaster = new MasterCache(model,
      _allReduceParameter,
      optimMethods,
      _parameterSplits,
      parameterProcessors.toArray,
      new Metrics,
      criterion,
      validationMethods,
      dataset.toDistributed().originRDD().partitions.length)

    if (!reuse) {
      import DistriOptimizerV2.initCacheOfSlave
      val modelsAndBroadcast = initCacheOfSlave(cacheOfMaster, dataset.toDistributed(), context)
      cacheOfSlave = if (reserveOptimMethod && previousOptim != null) {
        // replace optimMethods with previous ones
        resetOptimMethods(modelsAndBroadcast._1, previousOptim)
      } else {
        modelsAndBroadcast._1
      }
      modelBroadcast = modelsAndBroadcast._2
    }

    if (checkpointPath.isDefined) {
      val file = checkpointPath.get + "/" +
        new SimpleDateFormat("yyyyMMdd_HHmmss").format(Calendar.getInstance().getTime())
      new File(file).mkdir()
      checkpointPath = Some(file)
    }

    DistriOptimizerV2.optimize(
      cacheOfMaster,
      cacheOfSlave,
      dataset.toDistributed(),
      endWhen,
      validationTrigger,
      validationDataSet,
      validationMethods,
      checkpointTrigger,
      checkpointPath,
      trainSummary,
      validationSummary,
      isOverWrite,
      context
    )

    DistriOptimizerV2.getModel(cacheOfSlave.asInstanceOf[RDD[DistriOptimizer.Cache[T]]],
      cacheOfMaster.parameter, cacheOfMaster.model)
    // reserve optimMethod internal state for each worker if need
    if (reserveOptimMethod) {
      previousOptim = cacheOfSlave.map(m => m.optimMethods).cache()
      previousOptim.count()
    } else {
      if (previousOptim != null) previousOptim.unpersist()
    }

    // Reset some internal states, so this or other optimizers can run optimize again
    clearState()

    // unpersist the model because the next time optimize is called, new `models` will be
    // created
    shutdown()

    if (!reuse) {
      cacheOfSlave.unpersist()
    }

    cacheOfMaster.model
  }

  def setReuse(): this.type = {
    _canBeReused = true
    this
  }

  private type ModuleOptimMethods[R] = Map[String, OptimMethod[R]]
  private def initOptimMethods(optimMethods: ModuleOptimMethods[T],
    parameters: Tensor[T]): Map[String, (Int, Int)] = {
    import StateEntry._
    optimMethods.values.foreach { optimMethod =>
      optimMethod.clearHistory()
    }

    // To be compatible with the old usage that user define hyperparameters in a table.
    if (optimMethods.size == 1) {
      optimMethods.head._2.loadFromTable(state)
    }

    optimMethods.values.foreach { optimMethod =>
      if (!optimMethod.state.contains(EPOCH)) optimMethod.state.update(EPOCH, 1)
      if (!optimMethod.state.contains(NEVAL)) optimMethod.state.update(NEVAL, 0)
      if (!optimMethod.state.contains(LOSS)) {
        optimMethod.state.update(LOSS, Float.PositiveInfinity)
      }
      if (!optimMethod.state.contains(SCORE)) optimMethod.state.update(SCORE, 0f)
      if (!optimMethod.state.contains(RECORDS_PROCESSED)) {
        optimMethod.state.update(RECORDS_PROCESSED, 0)
      }
    }

    // subModuleName -> (storageOffset, length, AllReduceParameter)
    val parameterSplits = if (optimMethods.size != 1) {
      val p = optimMethods.map { case (subModuleName, optimMethod) =>
        val subModule = model(subModuleName)
        require(subModule.isDefined, s"Optimizer couldn't find $subModuleName in $model")
        val subModuleWeights = subModule.get.getParameters()._1
        (subModuleName, subModuleWeights)
      }

      // check the weights of submodule with whole model's weight, they should be the same
      val sortedWeights = p.values.toArray.sortWith((a, b) => a.storageOffset() < b.storageOffset())
      val compactWeights = Module.isCompact(sortedWeights)
      require(parameters == compactWeights,
        s"DistriOptimizer: All subModules should have an OptimMethod.")

      p.map { case (subModuleName, weights) =>
        (subModuleName, (weights.storageOffset(), weights.nElement()))
      }
    } else if (optimMethods.contains(model.getName())) {
      Map(model.getName() -> (1, parameters.nElement()))
    } else {
      throw new IllegalArgumentException(s"${model.getName()} doesn't " +
        s"have corresponding OptimMethod")
    }

    // LarsSSD will check the optimMethods and append LarsProcessor
    // if there's no LarsSSD in optimMethods map, it will do nothing.
    // should be refactored later if possible.
    LarsSGD.containsLarsSGD(optimMethods).foreach(weightDecay =>
      parameterProcessors.append(new LarsProcessor(parameterSplits, weightDecay)))

    parameterSplits
  }

  // this shutdown should not be called out of this scope.
  private[optim] override def shutdown(): Unit = {
    cacheOfSlave.mapPartitions { iter =>
      iter.foreach { arrayModels =>
        arrayModels.localModels.foreach(_.release())
      }

      iter
    }.count()
    CachedModels.deleteKey(modelBroadcast.uuid)
  }

  def setLogger(logger: OptimizerLogger): Unit = {
    DistriOptimizerV2._logger = Some(logger)
  }

  private def validArgs(): Boolean = {
    val checkSingleton = this.checkSingleton
    val nodeNumber = Engine.nodeNumber()
    val executorCores = Engine.coreNumber()
    val partitionNumber = dataset.toDistributed().originRDD().partitions.length
    require(partitionNumber == nodeNumber,
      s"Passed in rdd partition number $partitionNumber " +
        s" is not equal to configured node number $nodeNumber")

    dataset.toDistributed().originRDD().foreachPartition { _ =>
      Engine.setNodeAndCore(nodeNumber, executorCores)
      if (!Engine.checkSingleton()) {
        if (checkSingleton) {
          require(Engine.checkSingleton(), "Partitions of the training data are not evenly" +
            "distributed across the executors in the Spark cluster; are there sufficient " +
            "training" +
            "data to be distributed? Set property \"bigdl.check.singleton\" to false to skip " +
            "this check")
        } else {
          DistriOptimizer.logger.warn("Partitions of the training data are not evenly" +
            "distributed across the executors in the Spark cluster; are there sufficient " +
            "training" +
            "data to be distributed?")
        }
      }
    }

    true
  }
}

case class LossWithElapsedTime(index: Int, loss: Double, elapsed: Long)
case class ParamSegments[T](start: Int, length: Int, method: OptimMethod[T])

class TrainingContext[T: ClassTag](
  val subModelNumber: Int,
  val numSamples: Int,
  val state: Table) extends Serializable {

  def hasCompleteAllSamples(recordsProcessed: Int, model: Module[T]): Boolean = {
    recordsProcessed >= numSamples
  }

  def fetchBatch[T: ClassTag](data: Iterator[MiniBatch[T]]): Array[MiniBatch[T]] = {
    val miniBatchBuffer = new Array[MiniBatch[T]](subModelNumber)
    val batch = data.next()
    val stackSize = batch.size() / subModelNumber
    // TODO performance call Engine.invoke
    require((batch.size() >= subModelNumber) &&
      (batch.size() % subModelNumber == 0), "total batch size: " +
      s"${batch.size()} should be divided by total core number: $subModelNumber")

    if (batch.size() < subModelNumber * 2) {
      Logger.getLogger(this.getClass).warn(
        s"Warning: for better training speed, total batch size is recommended to be " +
          s"at least two times of core number $subModelNumber. " +
          s"please tune your batch size accordingly")
    }

    var b = 0
    while (b < subModelNumber) {
      miniBatchBuffer(b) = batch.slice(b * stackSize + 1, stackSize)
      b += 1
    }

    miniBatchBuffer
  }

  def train[T: ClassTag](data: Array[MiniBatch[T]], models: Array[Module[T]],
    criterion: Array[Criterion[T]])(implicit ev: TensorNumeric[T]): Seq[LossWithElapsedTime] = {
    val trainingThreads = Engine.default.invokeAndWait2(models.indices.map(i =>
      () => {
        val start = System.nanoTime()

        val localModel = models(i)
        val localCriterion = criterion(i)
        val input = data(i).getInput()
        val target = data(i).getTarget()
        var loss = 0.0

        localModel.training()

        val output = localModel.forward(input)
        loss = ev.toType[Double](localCriterion.forward(output, target))
        val errors = localCriterion.backward(output, target)
        localModel.backward(input, errors)

        val end = System.nanoTime()

        LossWithElapsedTime(i, loss, end - start)
      }
    ), Long.MaxValue)
    trainingThreads.filter(!_.isCancelled).map(_.get())
  }

  def update[T: ClassTag](
    optimSegments: Map[String, ParamSegments[T]],
    weight: Tensor[T], gradient: Tensor[T],
    averageLoss: Double
  )(implicit ev: TensorNumeric[T]): Unit = {
    optimSegments.foreach { case (name, ParamSegments(start, length, method)) =>
      if (length > 0) {
        method.optimize(
          _ => (ev.fromType(averageLoss), gradient.narrow(1, start, length)),
          weight.narrow(1, start, length))
      }
    }
  }

  def aggregate[T: ClassTag](gradients: Array[Tensor[T]]): Tensor[T] = {
    // NOTE: optimizer requires the gradients will be seprated with each other.
    //       so you should not merge them.
    val start = gradients.head.storageOffset()
    val length = gradients.head.nElement()

    val taskSize = length / this.subModelNumber
    val extraTask = length % this.subModelNumber

    // Aggregate multi-model's gradient to the first model's gradient
    val parallelNum = if (taskSize == 0) extraTask else this.subModelNumber
    if (parallelNum != 1) {
      Engine.default.invokeAndWait((0 until parallelNum).map(tid => () => {
        val offset = start + tid * taskSize + math.min(tid, extraTask)
        val length = taskSize + (if (tid < extraTask) 1 else 0)
        var i = 1
        while (i < gradients.length) {
          val target = gradients(0).narrow(1, offset, length)
          val source = gradients(i).narrow(1, offset, length)
          target.add(source)
          i += 1
        }
      }))
    }

    gradients(0)
  }

  final def loadState(state: Table): this.type = {
    this.state.update(StateEntry.EPOCH, state(StateEntry.EPOCH))
    this.state.update(StateEntry.NEVAL, state(StateEntry.NEVAL))
    this.state.update(StateEntry.LOSS, state(StateEntry.LOSS))
    this.state.update(StateEntry.SCORE, state(StateEntry.SCORE))
    this.state.update(StateEntry.PARALLELISM, subModelNumber)
    this.state.update(StateEntry.RECORDS_PROCESSED, state(StateEntry.RECORDS_PROCESSED))
    this
  }
}

object StateEntry {
  val NEVAL: String = "neval"
  val EPOCH: String = "epoch"
  val RECORDS_PROCESSED: String = "recordsProcessedThisEpoch"
  val LOSS: String = "Loss"
  val SCORE: String = "score"
  val PARALLELISM: String = "parallelism"
  val LEARNING_RATE: String = "LearningRate"
  val THROUGHPUT: String = "Throughput"
  val NUM_FINISHED_MODELS = "numFinishedModel"

  // for parameter processers, it's not a good design of that, but for compatible, keep it now.
  // parameterProcesser like L2NormClippingProcessor may aggregate gradient,
  // and change the value of isGradientUpdated in driverState.
  val IS_GRADIENT_UPDATED = "isGradientUpdated"
}

trait OptimizerLogger {
  def info(message: String): Unit

  def debug(message: String): Unit

  def trace(message: String): Unit

  def warn(message: String): Unit

  def error(message: String): Unit
}


private class MetricEntry(val value: String)
private case object AGGREGATE_GRADIENT_TIME extends MetricEntry("aggregate gradient time")
private case object COMPUTING_TIME_EACH_NODE extends MetricEntry("computing time for each node")
private case object COMPUTING_TIME_AVERAGE extends MetricEntry("computing time average")
private case object COMPUTE_WEIGHT_AVERAGE extends MetricEntry("compute weight average")
private case object GET_WEIGHTS_EACH_NODE extends MetricEntry("get weights for each node")
private case object GET_WEIGHTS_AVERAGE extends MetricEntry("get weights average")
private case object PUT_GRADIENT extends MetricEntry("put gradient")
// scalastyle:off
private case object AGGREGATE_PARTITION_GRADIENT extends MetricEntry("aggregrateGradientParition average executor")
// scalastyle:on
private case object SEND_WEIGHTS_AVERAGE extends MetricEntry("send weights average")

private case class Replica[T](model: Module[T], weights: Tensor[T], gradients: Tensor[T],
                      criterion: Criterion[T], state: Table,
                      validationMethods: Option[Array[ValidationMethod[T]]])

private class TrainingTrace(
  private var _records: Int = 0,
  private var _iterations: Int = 0,
  private var _epochs: Int = 1) {

  private var _epochStart: Long = 0
  private var _iterationTakes: Long = 0
  private var _trainingStart: Long = System.nanoTime()

  def startNewEpoch(): Unit = {
    _epochStart = System.nanoTime()
    // we can't reset iterations to 0 for compatible
//    _iterations = 0
    _records = 0
    _epochs += 1
  }

  def trainingTakes: Long = System.nanoTime() - _trainingStart

  def epochTakes: Long = System.nanoTime() - _epochStart

  def iterationTakes: Long = _iterationTakes

  def traceIteration[R](block: => R): R = {
    val (ret, elapsed) = TrainingTrace.time(block)
    _iterationTakes = elapsed
    _iterations += 1
    ret
  }

  def recordsOfEpoch: Int = _records

  def updateRecords(num: Int): this.type = {
    _records += num
    this
  }

  def iterations: Int = _iterations

  def epochs: Int = _epochs
}

private object TrainingTrace {
  def apply(records: Int, iterations: Int, epochs: Int): TrainingTrace = {
    new TrainingTrace(records, iterations, epochs)
  }

  def apply(state: Table): TrainingTrace = {
    val records = state[Int](StateEntry.RECORDS_PROCESSED)
    val iterations = state[Int](StateEntry.NEVAL) - 1 // for compatible
    val epochs = state[Int](StateEntry.EPOCH)

    new TrainingTrace(records, iterations, epochs)
  }

  def time[R](block: => R): (R, Long) = {
    val start = System.nanoTime()
    val ret = block
    val end = System.nanoTime()
    val elapsed = end - start

    (ret, elapsed)
  }

  def time[R](block: => R, metrics: Metrics)(entries: Array[MetricEntry]): R = {
    val (ret, elapsed) = time(block)

    var i = 0
    while (i < entries.length) {
      metrics.add(entries(0).value, elapsed)
      i += 1
    }

    ret
  }
}

private class DistriLogger extends OptimizerLogger {
  override def info(message: String): Unit = {
    Logger.getLogger(getClass).info(message)
  }

  override def debug(message: String): Unit = {
    Logger.getLogger(getClass).debug(message)
  }

  override def trace(message: String): Unit = {
    Logger.getLogger(getClass).trace(message)
  }

  override def warn(message: String): Unit = {
    Logger.getLogger(getClass).warn(message)
  }

  override def error(message: String): Unit = {
    Logger.getLogger(getClass).error(message)
  }
}

private class MasterCache[T](
  val model: Module[T],
  val parameter: AllReduceParameter[T],
  val optimMethods: Map[String, OptimMethod[T]],
  val parameterSplits: Map[String, (Int, Int)],
  val parameterProcessers: Array[ParameterProcessor],
  val metrics: Metrics,
  val criterion: Criterion[T],
  val validationMethods: Option[Array[ValidationMethod[T]]],
  val partitionNum: Int)

