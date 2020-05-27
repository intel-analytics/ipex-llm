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

import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DataSet, SampleToMiniBatch, _}

import scala.collection.mutable
import com.intel.analytics.bigdl.parameters.{ConstantClippingProcessor,
  L2NormClippingProcessor, ParameterProcessor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.reflect.{ClassTag, classTag}

/**
 * [[Optimizer]] is an abstract class which is used to train a model automatically
 * with some certain optimization algorithms.
 *
 * @param model the model to be trained
 * @param dataset the data set used to train a model
 * @param criterion the criterion used to evaluate the loss of the model given an input
 * @tparam T numeric type, which can be [[Float]] or [[Double]]
 * @tparam D the type of elements in DataSet, such as [[MiniBatch]]
 */
// TODO: remove D to be MiniBatch[T]
abstract class Optimizer[T: ClassTag, D](
  protected var model: Module[T],
  protected var dataset: DataSet[D],
  protected var criterion: Criterion[T])(implicit ev : TensorNumeric[T])
{
  import Optimizer.{logger, checkSubModules}
  protected var state: Table = T()
  protected var optimMethods: Map[String, OptimMethod[T]] = Map(model.getName -> new SGD())
  protected var endWhen: Trigger = Trigger.maxIteration(100)

  protected var checkpointTrigger: Option[Trigger] = None
  protected var checkpointPath: Option[String] = None
  protected var isOverWrite: Boolean = false

  protected var validationTrigger: Option[Trigger] = None
  protected var validationMethods: Option[Array[ValidationMethod[T]]] = None
  protected var validationDataSet: Option[DataSet[MiniBatch[T]]] = None

  // To save the summaries.
  protected var trainSummary: Option[TrainSummary] = None
  protected var validationSummary: Option[ValidationSummary] = None

  // To achieve better performance, please set dropPercentage as 0.04
  protected var dropPercentage: Double = 0.0
  protected var maxDropPercentage: Double = 0.0
  protected var computeThresholdbatchSize: Int = 100
  protected var warmupIterationNum: Int = 200

  /**
   * a list of ParameterProcessor, orders matter
   */
  protected var parameterProcessors = ArrayBuffer[ParameterProcessor]()

  model.checkDuplicate()

  /**
   * Trigger the optimization process
   * @return the model to be trained
   */
  def optimize(): Module[T]

  /**
   * make optimizer not check the singleton model on a node
   * @return
   */
  @deprecated("Use bigdl.check.singleton instead", "0.1.0")
  def disableCheckSingleton(): this.type = {
    this.checkSingleton = false
    println("disableCheckSingleton is deprecated. Please use bigdl.check.singleton instead")
    this
  }

  // TODO: Remove below code to DistriOptimizer after disableCheckSingleton is not supported
  protected var checkSingleton = System.getProperty("bigdl.check.singleton",
    false.toString).toBoolean

  /**
   * Set a validate evaluation
   *
   * @param trigger how often to evaluation validation set
   * @param dataset validate data set in type of [[DataSet]] of [[MiniBatch]]
   * @param vMethods a set of validation method [[ValidationMethod]]
   * @return this optimizer
   */
  def setValidation(trigger: Trigger, dataset: DataSet[MiniBatch[T]],
    vMethods : Array[ValidationMethod[T]])
  : this.type = {
    this.validationTrigger = Some(trigger)
    this.validationDataSet = Some(dataset)
    this.validationMethods = Some(vMethods)
    this
  }

  /**
   * Set a validate evaluation
   *
   * @param trigger how often to evaluation validation set
   * @param sampleRDD validate data set in type of [[RDD]] of [[Sample]]
   * @param vMethods a set of validation method [[ValidationMethod]]
   * @param batchSize batch size
   * @param featurePaddingParam feature padding strategy, see
   *                            [[com.intel.analytics.bigdl.dataset.PaddingParam]] for details.
   * @param labelPaddingParam   label padding strategy, see
   *                            [[com.intel.analytics.bigdl.dataset.PaddingParam]] for details.
   *
   * @return this optimizer
   */
  def setValidation(trigger: Trigger, sampleRDD: RDD[Sample[T]],
    vMethods : Array[ValidationMethod[T]], batchSize: Int,
    featurePaddingParam: PaddingParam[T],
    labelPaddingParam: PaddingParam[T]
  ): this.type = {
    this.validationTrigger = Some(trigger)
    val dataSet =
      (DataSet.rdd(sampleRDD) ->
        SampleToMiniBatch(batchSize, Some(featurePaddingParam), Some(labelPaddingParam)))
        .toDistributed()
    this.validationDataSet = Some(dataSet)
    this.validationMethods = Some(vMethods)
    this
  }

  /**
   * Set a validate evaluation
   *
   * @param trigger how often to evaluation validation set
   * @param sampleRDD validate data set in type of [[RDD]] of [[Sample]]
   * @param vMethods a set of validation method [[ValidationMethod]]
   * @param batchSize batch size
   * @return this optimizer
   */
  def setValidation(trigger: Trigger, sampleRDD: RDD[Sample[T]],
      vMethods : Array[ValidationMethod[T]], batchSize: Int)
  : this.type = {
    this.validationTrigger = Some(trigger)
    val dataSet =
      (DataSet.rdd(sampleRDD) -> SampleToMiniBatch(batchSize))
        .toDistributed()
    this.validationDataSet = Some(dataSet)
    this.validationMethods = Some(vMethods)
    this
  }

  /**
   * Set validate evaluation
   * @param trigger how often to evaluation validation set
   * @param sampleRDD validate data set in type of [[RDD]] of [[Sample]]
   * @param vMethods a set of validation method [[ValidationMethod]]
   * @param batchSize batch size
   * @param miniBatch construct MiniBatch with a specified miniBatch type
   * @return
   */
  def setValidation(trigger: Trigger, sampleRDD: RDD[Sample[T]],
                    vMethods : Array[ValidationMethod[T]], batchSize: Int, miniBatch: MiniBatch[T])
  : this.type = {
    this.validationTrigger = Some(trigger)
    val dataSet =
      (DataSet.rdd(sampleRDD) -> SampleToMiniBatch(miniBatch, batchSize, None))
        .toDistributed()
    this.validationDataSet = Some(dataSet)
    this.validationMethods = Some(vMethods)
    this
  }

  /**
   * Set a check point saved at `path` triggered by `trigger`
   *
   * @param path the directory to save
   * @param trigger how often to save the check point
   * @return the optimizer
   */
  def setCheckpoint(path: String, trigger: Trigger): this.type = {
    if (!path.startsWith(File.hdfsPrefix)) {
      require(Files.isDirectory(Paths.get(path)), s"Optimizer.setCheckpoint: $path is not a folder")
    }
    this.checkpointPath = Some(path)
    this.checkpointTrigger = Some(trigger)
    this
  }

  /**
   * Get the directory of saving checkpoint
   */
  def getCheckpointPath(): Option[String] = {
    this.checkpointPath
  }

  /**
   * Enable train summary.
   */
  def setTrainSummary(trainSummary: TrainSummary): this.type = {
    this.trainSummary = Some(trainSummary)
    this
  }

  /**
   * Enable validation summary.
   */
  def setValidationSummary(validationSummary: ValidationSummary): this.type = {
    this.validationSummary = Some(validationSummary)
    this
  }

  /**
   * Enable overwrite saving checkpoint
   */
  def overWriteCheckpoint() : this.type = {
    isOverWrite = true
    this
  }

  private def resetEpoch(): Unit = {
    optimMethods.foreach{ case (moduleName, optimMethod) =>
      optimMethod.state.update("epoch", 1)
      optimMethod.state.update("neval", 1)
      optimMethod.state.update("Loss", Float.PositiveInfinity)
      optimMethod.state.update("score", 0f)
      optimMethod.state.update("recordsProcessedThisEpoch", 0)
    }
  }


  /**
   * Set a model to the optimizer.
   * Notice: if current optimMethod in this optimizer is not a global optimMethod,
   * this setModel will throw an exception. You should use setModelAndOptimMethods instead.
   *
   * @param newModel new model
   */
  def setModel(newModel: Module[T]): this.type = {
    // check if the old optimMethods is a global one.
    if (optimMethods.size == 1 && optimMethods.contains(model.getName())) {
      if (newModel.getName() != model.getName()) {
        optimMethods = Map(newModel.getName() -> optimMethods(model.getName()))
      }
      logger.info(s"Optimizer.setModel: Detect current optimMethod is a global optimMethod." +
        s" Automatically associate the current optimMethod with the new model.")
    } else {
      throw new IllegalArgumentException("Optimizer.setModel: Detect current optimMethod" +
        " is not a global optimMethod. Please use setModelAndOptimMethods")
    }

    model = newModel

    model.checkDuplicate()

    // if a new Model is set, then reset "epoch", "neval" .etc.
    resetEpoch()
    this
  }

  /**
   * Set new model and new optimMethods to the optimizer.
   *
   * @param newModel new model
   * @param newOptimMethods new optimMethods
   */
  def setModelAndOptimMethods(
        newModel: Module[T],
        newOptimMethods: Map[String, OptimMethod[T]]): this.type = {
    // check if the old optimMethods is a global one.
    model = newModel
    optimMethods = newOptimMethods

    model.checkDuplicate()

    // if a new Model is set, then reset "epoch", "neval" .etc.
    resetEpoch()
    this
  }


  /**
   * Set new train dataset.
   * User can supply a customized implementation of trait MiniBatch to define
   * how data is organized and retrieved in a mini batch.
   *
   * @param sampleRDD training Samples
   * @param batchSize mini batch size
   * @param miniBatchImpl An User-Defined MiniBatch implementation.
   * @return the Optimizer
   */
  def setTrainData(sampleRDD: RDD[Sample[T]],
                 batchSize: Int,
                 miniBatchImpl: MiniBatch[T]): this.type = {
    throw new UnsupportedOperationException(
      s"setTrainData(sampleRDD, batchSize,miniBatch) " +
        s"is only supported in distributed optimizer")
    this
  }

  /**
   * Set new train dataset.
   *
   * @param sampleRDD           training Samples
   * @param batchSize           mini batch size
   * @param featurePaddingParam feature padding strategy, see
   *                            [[com.intel.analytics.bigdl.dataset.PaddingParam]] for details.
   * @param labelPaddingParam   label padding strategy, see
   *                            [[com.intel.analytics.bigdl.dataset.PaddingParam]] for details.
   * @return the optimizer
   */
  def setTrainData(sampleRDD: RDD[Sample[T]],
                 batchSize: Int,
                 featurePaddingParam: PaddingParam[T] = null,
                 labelPaddingParam: PaddingParam[T] = null): this.type = {
    throw new UnsupportedOperationException(
      s"setTrainData(sampleRDD,batchSize,featurePaddingParam=null,labelPaddingParam=null) " +
        s"is only supported in distributed optimizer")
    this
  }


  /**
   * Set a new criterion to the optimizer
   *
   * @param newCriterion new criterion
   */
  def setCriterion(newCriterion: Criterion[T]): this.type = {
    this.criterion = newCriterion
    this
  }


  /**
   * Set a state(learning rate, epochs...) to the optimizer
   *
   * @param state the state to be saved
   */
  def setState(state: Table): this.type = {
    this.state = state
    this
  }

  /**
   * Set an optimization method
   *
   * @param method optimization method
   */
  def setOptimMethod(method : OptimMethod[T]): this.type = {
    checkSubModules(model, Array(model.getName()))
    this.optimMethods = Map(model.getName -> method)
    this
  }

  /**
   * Set optimization methods for each submodule.
   *
   * @param method A mapping of submodule -> OptimMethod
   */
  def setOptimMethods(method: Map[String, OptimMethod[T]]): this.type = {
    checkSubModules(model, method.keys.toSeq)
    this.optimMethods = method
    this
  }

  /**
   * When to stop, passed in a [[Trigger]]
   *
   * @param endWhen when to end
   * @return the optimizer
   */
  def setEndWhen(endWhen: Trigger): this.type = {
    this.endWhen = endWhen
    this
  }

  /**
   * Set dropping a certain percentage (`dropPercentage`) of models during distributed
   * training to accelerate, because some cached model may take too long.
   *
   * @param dropPercentage drop percentage
   * @param maxDropPercentage max drop percentage
   * @param batchsize batch size
   * @param warmupIteration how may iteration to warm up
   * @return this optimizer
   */
  def setDropModuleProperty(dropPercentage: Double, maxDropPercentage: Double,
    batchsize: Int = 100, warmupIteration: Int = 200): this.type = {
    this.dropPercentage = dropPercentage
    this.maxDropPercentage = maxDropPercentage
    require(dropPercentage >= 0 && dropPercentage <= maxDropPercentage)
    this.computeThresholdbatchSize = batchsize
    this.warmupIterationNum = warmupIteration
    this
  }

  def prepareInput(): Unit = {}

  /**
   * Disable gradient clipping
   * @return
   */
  def disableGradientClipping()
  : this.type = {
    parameterProcessors = parameterProcessors.filterNot(processor =>
      (processor.isInstanceOf[ConstantClippingProcessor] ||
        processor.isInstanceOf[L2NormClippingProcessor]))
    this
  }

  /**
   * Set constant gradient clipping
   * @param min the minimum value to clip by
   * @param max the maximum value to clip by
   * @return
   */
  def setConstantGradientClipping(min: Double, max: Double)
  : this.type = {
    require(min <= max, "min value can not be larger than max")
    val index = Optimizer.findIndex[ConstantClippingProcessor](parameterProcessors)
    if (index == -1) {
      parameterProcessors.append(new ConstantClippingProcessor(min, max))
    } else {
      parameterProcessors(index) = new ConstantClippingProcessor(min, max)
    }
    this
  }


  /**
   * Clip gradient to a maximum L2-norm
   * @param l2NormThreshold gradient L2-Norm threshold
   * @return
   */
  def setGradientClippingByl2Norm(l2NormThreshold: Double)
  : this.type = {
    require(optimMethods.size == 1, "Only support 1 optimMethod.")
    require(l2NormThreshold > 0, "l2NormThreshold should larger than zero")
    val index = Optimizer.findIndex[L2NormClippingProcessor](parameterProcessors)
    if (index == -1) {
      parameterProcessors.append(new L2NormClippingProcessor(l2NormThreshold))
    } else {
      parameterProcessors(index) = new L2NormClippingProcessor(l2NormThreshold)
    }
    this
  }

  /**
   * shutdown the optimizer, which will release the native resources if exists.
   */
  private[optim] def shutdown(): Unit = {}

  def reserveOptim(reserve: Boolean): this.type = {
    throw new UnsupportedOperationException(
      "Only support DistriOptimizer to reserve optim methods for each worker")
  }
}

object Optimizer {
  private val logger: Logger = Logger.getLogger(getClass)

  private[bigdl] def header(epoch: Int, count: Int, total: Long, iter: Int, wallClockTime: Long)
  : String = {
    s"[Epoch $epoch $count/$total][Iteration $iter][Wall Clock ${wallClockTime / 1e9}s]"
  }

  /**
   * Check if the sub modules are in the model, if each sub modules' parameter
   * is contiguous, if sub modules' parameter is duplicated.
   * @param model
   * @param subModuleNames
   * @param ev
   * @tparam T
   */
  private[bigdl] def checkSubModules[T: ClassTag](
        model: Module[T],
        subModuleNames: Seq[String])(implicit ev: TensorNumeric[T]): Unit = {
    val modelParameters = model.getParameters()
    val p = subModuleNames.map{subModuleName =>
      val subModule = model(subModuleName)
      require(subModule.isDefined, s"Optimizer: couldn't find $subModuleName in $model")
      val subModuleWeights = subModule.get.getParameters()._1
      require(subModuleWeights.nElement() > 0, s"Optimizer: $subModuleName doesn't have" +
        s" any trainable parameters, please check your model and optimMethods.")
      // If the storage subModule's parameter is the same with the storage of the submodule,
      // then subModule's parameter is contiguous.
      require(modelParameters._1.storage() == subModuleWeights.storage(), s"Optimizer:" +
        s" $subModuleName's parameter is not contiguous.")
      (subModuleName, subModuleWeights)
    }.toArray

    // make sure if parameters in submodules aren't duplicated.
    if (p.length != 1) {
      val sortedWeights = p.sortWith((a, b) => a._2.storageOffset() < b._2.storageOffset())
      var i = 0
      while (i < sortedWeights.length - 1) {
        val current = sortedWeights(i)
        val next = sortedWeights(i + 1)
        require(current._2.storageOffset() + current._2.nElement() <= next._2.storageOffset(),
          s"Optimizer: ${current._1} and ${next._1}'s parameters are duplicated." +
            s" Please check your model and optimMethods.")
        i += 1
      }
    }
  }

  /**
   * Combine the hyper parameters in optimMethods.
   */
  private[bigdl] def getHyperParameterLog(optimMethods: Map[String, OptimMethod[_]]): String = {
    optimMethods.map{ case (moduleName, optimMethod) =>
        val log = optimMethod.getHyperParameter()
        if (log.isEmpty) {
          log
        } else {
          s"${moduleName}'s hyper parameters: ${log} "
        }
      }.reduce(_ + _)
  }

  /**
   * Save a model to a directory as a checkpoint
   *
   * @param model the model to be saved
   * @param checkpointPath the directory to save at
   * @param overWrite if save name model exists in the directory,
   *                  is overwrite or not.
   * @param postfix the postfix of a model name
   * @tparam T model data type [[Double]] or [[Float]]
   */
  private[bigdl] def saveModel[T](model: Module[T], checkpointPath : Option[String],
    overWrite : Boolean, postfix: String = ""): Unit = {
    if (checkpointPath.isDefined) {
      model.save(s"${checkpointPath.get}/model$postfix", overWrite)
    }
  }

  /**
   * Save a state to a directory as a checkpoint
   *
   * @param state the state (learning rate, epochs...) to be saved
   * @param checkpointPath the directory to save at
   * @param overWrite if save name model exists in the directory,
   *                  is overwrite or not.
   * @param postfix the postfix of a state name
   */
  private[bigdl] def saveState(state: Table, checkpointPath : Option[String], overWrite : Boolean,
    postfix: String = ""): Unit = {
    if (checkpointPath.isDefined) {
      state.save(s"${checkpointPath.get}/state$postfix", overWrite)
    }
  }

  /**
   * Save OptimMethod to a directory as a checkpoint
   * @param optimMethod the method to be saved
   * @param checkpointPath the directory to save at
   * @param overWrite if save name method exists in the directory,
   *                  is overwrite or not.
   * @param postfix the postfix of a method name
   * @tparam T
   */
  private[bigdl] def saveOptimMethod[T: ClassTag]
  (optimMethod: OptimMethod[T], checkpointPath : Option[String], overWrite : Boolean,
   postfix: String = ""): Unit = {
    if (checkpointPath.isDefined) {
      optimMethod.save(s"${checkpointPath.get}/optimMethod$postfix", overWrite)
    }
  }


  /**
   * Apply an Optimizer.
   *
   * @param model               model will be optimized
   * @param sampleRDD           training Samples
   * @param criterion           loss function
   * @param batchSize           mini batch size
   * @param featurePaddingParam feature padding strategy, see
   *                            [[com.intel.analytics.bigdl.dataset.PaddingParam]] for details.
   * @param labelPaddingParam   label padding strategy, see
   *                            [[com.intel.analytics.bigdl.dataset.PaddingParam]] for details.
   * @return An optimizer
   */
  def apply[T: ClassTag](
      model: Module[T],
      sampleRDD: RDD[Sample[T]],
      criterion: Criterion[T],
      batchSize: Int,
      featurePaddingParam: PaddingParam[T] = null,
      labelPaddingParam: PaddingParam[T] = null
         )(implicit ev: TensorNumeric[T]): Optimizer[T, MiniBatch[T]] = {

    val _featurePaddingParam = if (featurePaddingParam != null) Some(featurePaddingParam) else None
    val _labelPaddingParam = if (labelPaddingParam != null) Some(labelPaddingParam) else None

    Engine.getOptimizerVersion() match {
      case OptimizerV1 =>
        new DistriOptimizer[T](
          _model = model,
          _dataset = (DataSet.rdd(sampleRDD) ->
            SampleToMiniBatch(batchSize, _featurePaddingParam, _labelPaddingParam))
            .toDistributed(),
          _criterion = criterion
        ).asInstanceOf[Optimizer[T, MiniBatch[T]]]
      case OptimizerV2 =>
        new DistriOptimizerV2[T](
          _model = model,
          _dataset = (DataSet.rdd(sampleRDD) ->
            SampleToMiniBatch(batchSize, _featurePaddingParam, _labelPaddingParam))
            .toDistributed(),
          _criterion = criterion
        ).asInstanceOf[Optimizer[T, MiniBatch[T]]]
    }
  }


  /**
   * Apply an optimizer.
   * User can supply a customized implementation of trait MiniBatch to define
   * how data is organize and retrieved in a mini batch.
   *
   * @param model model will be optimized
   * @param sampleRDD training Samples
   * @param criterion loss function
   * @param batchSize mini batch size
   * @param miniBatchImpl An User-Defined MiniBatch implementation
   * @return an new Optimizer
   */
  def apply[T: ClassTag](
          model: Module[T],
          sampleRDD: RDD[Sample[T]],
          criterion: Criterion[T],
          batchSize: Int,
          miniBatchImpl: MiniBatch[T]
        )(implicit ev: TensorNumeric[T]): Optimizer[T, MiniBatch[T]] = {
    Engine.getOptimizerVersion() match {
      case OptimizerV1 =>
        new DistriOptimizer[T](
          _model = model,
          _dataset = (DataSet.rdd(sampleRDD) ->
            SampleToMiniBatch(miniBatchImpl, batchSize, None))
            .toDistributed(),
          _criterion = criterion
        ).asInstanceOf[Optimizer[T, MiniBatch[T]]]
      case OptimizerV2 =>
        new DistriOptimizerV2[T](
          _model = model,
          _dataset = (DataSet.rdd(sampleRDD) ->
            SampleToMiniBatch(miniBatchImpl, batchSize, None))
            .toDistributed(),
          _criterion = criterion
        ).asInstanceOf[Optimizer[T, MiniBatch[T]]]
    }
  }

  /**
   * Apply an optimizer.
   *
   * @param model model will be optimizied
   * @param dataset the input dataset - determines the type of optimizer
   * @param criterion loss function
   * @return an new Optimizer
   */
  def apply[T: ClassTag, D](
    model: Module[T],
    dataset: DataSet[D],
    criterion: Criterion[T]
  )(implicit ev: TensorNumeric[T]): Optimizer[T, D] = {
    dataset match {
      case d: DistributedDataSet[_] =>
        Engine.getOptimizerVersion() match {
          case OptimizerV1 =>
            new DistriOptimizer[T](
              _model = model,
              _dataset = d.toDistributed().asInstanceOf[DistributedDataSet[MiniBatch[T]]],
              _criterion = criterion
            ).asInstanceOf[Optimizer[T, D]]
          case OptimizerV2 =>
            new DistriOptimizerV2[T](
              _model = model,
              _dataset = d.toDistributed().asInstanceOf[DistributedDataSet[MiniBatch[T]]],
              _criterion = criterion
            ).asInstanceOf[Optimizer[T, D]]
        }
      case d: LocalDataSet[_] =>
        new LocalOptimizer[T](
          model = model,
          dataset = d.toLocal().asInstanceOf[LocalDataSet[MiniBatch[T]]],
          criterion = criterion
        ).asInstanceOf[Optimizer[T, D]]
      case _ =>
        throw new UnsupportedOperationException
    }
  }

  /**
   * find the index of type T
   * @param parameterProcessors
   * @return index
   */
  private[Optimizer] def findIndex[T <: ParameterProcessor: ClassTag](
        parameterProcessors: ArrayBuffer[ParameterProcessor]): Int = {
    var i = 0
    while(i < parameterProcessors.size) {
      if (classTag[T].runtimeClass.isInstance(parameterProcessors(i))) {
        return i
      }
      i += 1
    }
    return -1
  }
}
