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
 package com.intel.analytics.bigdl.dllib.estimator

 import com.intel.analytics.bigdl.dllib.feature.dataset.MiniBatch
 import com.intel.analytics.bigdl.dllib.nn.Module
 import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
 import com.intel.analytics.bigdl.{Criterion, Module}
 import com.intel.analytics.bigdl.dllib.optim.{OptimMethod, ValidationMethod, ValidationResult}
 import com.intel.analytics.bigdl.dllib.tensor.{Storage, Tensor}
 import com.intel.analytics.bigdl.dllib.utils.{RandomGenerator, ThreadPool}
 import org.slf4j.LoggerFactory

 import scala.reflect.ClassTag


 /**
 * This is a pojo style local Estimator, will fit, train, evaluate on pojo data set.
 *
 * @param model         the model defined as AbstractModule
 * @param criterion     the criterion for loss
 * @param optmizeMethod the optimize method
 * @param validations   the validation metrics
 * @param threadNum     the number of threads
 */
 case class LocalEstimator(model: AbstractModule[Activity, Activity, Float],
                          criterion: Criterion[Float],
                          optmizeMethod: OptimMethod[Float],
                          validations: Array[ValidationMethod[Float]],
                          threadNum: Int) extends EstimateSupportive {

  val logger = LoggerFactory.getLogger(getClass)

  require(threadNum >= 1, "the number of threads should >= 1")
  @volatile private var defaultThreadPool: ThreadPool = new ThreadPool(threadNum)

  model.training()
  val parameters = retrieveParameters(model)
  val weight = parameters._1
  val grad = parameters._2
  val weightBias = model.getWeightsBias()
  val metaModel = makeMetaModel(model)

  val workingModels = (1 to threadNum).map(i => {
    logger.info(s"clone the ${i}th model...")
    val clonedModel = metaModel.cloneModule()
    val clonedModelWithWeightsBias = makeUpModel(clonedModel, weightBias)
    initGradWeightBias(weightBias, clonedModelWithWeightsBias)
    clonedModelWithWeightsBias
  })
  logger.info(s"working models: $workingModels")

  val gradLength = grad.nElement()
  val syncGradTaskSize = gradLength / threadNum
  val syncGradExtraTask = gradLength % threadNum
  val syncGradParallelNum = if (syncGradTaskSize == 0) syncGradExtraTask else threadNum
  logger.info(s"gradLength: $gradLength, " +
    s"threadNum: $threadNum, " +
    s"syncGradTaskSize: $syncGradTaskSize, " +
    s"syncGradExtraTask: $syncGradExtraTask, " +
    s"syncGradParallelNum: $syncGradParallelNum")

  val workingModelsWAndG = workingModels.map(retrieveParameters(_))
  val workingCriterions = (1 to threadNum).map(_ => criterion.cloneCriterion())

  /**
   * fit with pojo train data
   *
   * @param trainData     the array of train data
   * @param testData      the array of test data
   * @param transformer   the transformer to transfer data array to mini-batch
   * @param batchSize     the batch size for training
   * @param epoch         the epoch for training
   * @tparam T            the type of the data
   */
  def fit[T: ClassTag](trainData: Array[T],
                       testData: Array[T],
                       transformer: Array[T] => MiniBatch[Float],
                       batchSize: Int,
                       epoch: Int): Unit = {
    val testMiniBatches = groupDataToMiniBatches(testData, transformer, batchSize, false)
    List.range(0, epoch).map(i => {
      if (i == 0) {
        val trainMiniBatches = groupDataToMiniBatches(trainData, transformer, batchSize, false)
        train(i + 1, trainMiniBatches, testMiniBatches)
      } else {
        val trainMiniBatches = groupDataToMiniBatches(trainData, transformer, batchSize, true)
        train(i + 1, trainMiniBatches, testMiniBatches)
      }
    })
  }

  /**
   * train with mini-batches
   *
   * @param epoch            the epoch number
   * @param trainMiniBatches the train data in mini-batches
   * @param testMiniBatches  the test data in mini-batches
   */
  def train(epoch: Int,
            trainMiniBatches: Seq[MiniBatch[Float]],
            testMiniBatches: Seq[MiniBatch[Float]]): Unit = {
    var iteration = 1
    val evaluationDataSize = testMiniBatches.size * testMiniBatches(0).size()
    trainMiniBatches.map(miniBatch => {
      throughputingWithLoss(s"train for epoch: $epoch, iteration: $iteration",
        miniBatch.size()) {
        optimize(miniBatch)
      }
      iteration = iteration + 1
    })
    throughputing(s"eval for epoch: $epoch", evaluationDataSize) {
      validate(testMiniBatches)
    }
  }

  /**
   * fit with data in mini-batches
   *
   * @param trainMiniBatches      the train data in mini-batches
   * @param miniBatchNumPerEpoch  the number of the mini-batches per epoch
   * @param testMiniBatches       the test data in mini-batches
   */
  def fit(trainMiniBatches: Seq[MiniBatch[Float]],
          miniBatchNumPerEpoch: Int,
          testMiniBatches: Seq[MiniBatch[Float]]): Unit = {
    var iteration = 1
    var epoch = 1
    val evaluationDataSize = testMiniBatches.size * testMiniBatches(0).size()
    trainMiniBatches.map(miniBatch => {
      val loss = throughputingWithLoss(s"train for epoch: $epoch, iteration: $iteration",
        miniBatch.size()) {
        optimize(miniBatch)
      }
      if (iteration % miniBatchNumPerEpoch == 0) {
        throughputing(s"eval for epoch: $epoch, iteration: $iteration", evaluationDataSize) {
          validate(testMiniBatches)
        }
        epoch = epoch + 1
      }
      iteration = iteration + 1
    })
  }

  /**
   * optimize the model with mini-batch
   *
   * @param miniBatch the train data in mini-batch
   * @return  the loss after this iteration training
   */
  def optimize(miniBatch: MiniBatch[Float]): Float = {
    var b = 0
    val stackSize = miniBatch.size() / threadNum
    val extraSize = miniBatch.size() % threadNum
    val parallelism = if (stackSize == 0) extraSize else threadNum

    val miniBatchBuffer = new Array[MiniBatch[Float]](parallelism)
    while (b < parallelism) {
      val offset = b * stackSize + math.min(b, extraSize) + 1
      val length = stackSize + (if (b < extraSize) 1 else 0)
      miniBatchBuffer(b) = miniBatch.slice(offset, length)
      b += 1
    }

    val lossSum = defaultThreadPool.invokeAndWait((0 until parallelism).map(i =>
      () => {
        val localModel = workingModels(i)
        localModel.zeroGradParameters()
        localModel.training()
        val localCriterion = workingCriterions(i)
        val input = miniBatchBuffer(i).getInput()
        val target = miniBatchBuffer(i).getTarget()
        val output = localModel.forward(input)
        val _loss = localCriterion.forward(output, target)
        val errors = localCriterion.backward(output, target)
        localModel.backward(input, errors)
        _loss
      })).sum

    defaultThreadPool.invokeAndWait(
      (0 until syncGradParallelNum).map(tid =>
        () => {
          val offset = tid * syncGradTaskSize + math.min(tid, syncGradExtraTask)
          val length = syncGradTaskSize + (if (tid < syncGradExtraTask) 1 else 0)
          var i = 0
          while (i < parallelism) {
            if (i == 0) {
              grad.narrow(1, offset + 1, length)
                .copy(workingModelsWAndG(i)._2.narrow(1, offset + 1, length))
            } else {
              grad.narrow(1, offset + 1, length)
                .add(workingModelsWAndG(i)._2.narrow(1, offset + 1, length))
            }
            i += 1
          }
        })
    )
    val loss = lossSum / parallelism
    grad.div(parallelism)

    optmizeMethod.optimize(_ => (loss, grad), weight)
    loss
  }

  /**
   * validate the model with the mini-batch
   *
   * @param miniBatches the validation data in mini-batch
   */
  def validate(miniBatches: Seq[MiniBatch[Float]]): Unit = {
    val results: Seq[Seq[ValidationResult]] =
      miniBatches.map(miniBatch => {
        val stackSize = miniBatch.size() / threadNum
        val extraSize = miniBatch.size() % threadNum
        val parallelism = if (stackSize == 0) extraSize else threadNum

        var b = 0
        val miniBatchBuffer = new Array[MiniBatch[Float]](parallelism)
        while (b < parallelism) {
          val offset = b * stackSize + math.min(b, extraSize) + 1
          val length = stackSize + (if (b < extraSize) 1 else 0)
          miniBatchBuffer(b) = miniBatch.slice(offset, length)
          b += 1
        }

        val validationsArray = (0 until parallelism).map(i => validations.map(_.clone())).toArray
        defaultThreadPool.invokeAndWait((0 until parallelism).map(i =>
          () => {
            val localModel = workingModels(i)
            val input = miniBatchBuffer(i).getInput()
            val target = miniBatchBuffer(i).getTarget()
            val output = localModel.forward(input)
            validationsArray(i).map(validation => {
              validation(output, target)
            })
          })).flatten
      })

    val aggregateds = List.range(0, validations.length).map(i => {
      val validation = validations(i)
      val result = results.map(r => r(i)).reduce((l, r) => {
        l + r
      })
      (validation, result)
    })

    aggregateds.foreach(aggregated => logger.info(s"${aggregated._1}:${aggregated._2}"))
  }

  private def retrieveParameters(model: Module[Float]) = {
    val (weightParameters, gradParameters) = model.parameters()
    require(weightParameters != null && weightParameters.length > 0,
      s"model ${model.getName()} doesn't have any trainable parameters.")
    require(weightParameters.size == gradParameters.size,
      "weights and gradient number are not match")
    weightParameters.zip(gradParameters).foreach { case (w, g) => g.resizeAs(w) }
    (Module.flatten[Float](weightParameters), Module.flatten[Float](gradParameters))
  }

  private def initGradWeightBias(broadcastWeightBias: Array[Tensor[Float]],
                                        localModel: Module[Float]): Unit = {
    val (localWeightBias, localGradWeightBias) = localModel.parameters()
    val size = localGradWeightBias.map(_.nElement()).sum
    val storage = Storage[Float](size)
    var i = 0
    while (i < localWeightBias.length) {
      if (localWeightBias(i) != null) {
        val wb = broadcastWeightBias(i)
        localGradWeightBias(i).set(storage, wb.storageOffset(), wb.size(), wb.stride())
      }
      i += 1
    }
  }

  private def groupDataToMiniBatches[T: ClassTag](data: Array[T],
                                                  transformer: Array[T] => MiniBatch[Float],
                                                  batchSize: Int,
                                                  ifShuffle: Boolean = false)
  : List[MiniBatch[Float]] = {
    val shuffledData: Array[T] = timing("shuffle data") {
      if (ifShuffle) {
        val randomIndex = RandomGenerator.RNG.uniform(0, data.length).toInt
        val (left, right) = data.splitAt(randomIndex)
        right ++ left
      } else {
        data
      }
    }
    val miniBatches = shuffledData.sliding(batchSize, batchSize).map(transformer(_)).toList
    if (miniBatches.last.size() < batchSize) {
      miniBatches.init
    } else {
      miniBatches
    }
  }
 }
