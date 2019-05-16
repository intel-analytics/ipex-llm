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

import com.intel.analytics.bigdl.dataset.{DistributedDataSet, _}
import com.intel.analytics.bigdl.{DataSet, Module}
import com.intel.analytics.bigdl.optim.DistriOptimizer.{Cache, logger}
import com.intel.analytics.bigdl.optim.Optimizer.{saveModel, saveOptimMethod}
import com.intel.analytics.bigdl.parameters.AllReduceParameter
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.intermediate.IRGraph
import com.intel.analytics.bigdl.utils.{Engine, MklBlas, MklDnn, Table}
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import org.apache.spark.rdd.{RDD, ZippedPartitionsWithLocalityRDD}

import scala.reflect.ClassTag

abstract class AbstractOptimizer {

  protected def getModel[T: ClassTag](
    models: RDD[Cache[T]],
    parameters: AllReduceParameter[T],
    trainingModel: Module[T])(implicit ev: TensorNumeric[T]): Module[T]

  /**
   * Save train summaries.
   * @param trainSummary train logger
   * @param models cached models
   * @param driverState driver state
   * @param parameters [[AllReduceParameter]]
   */
  protected def saveSummary[T: ClassTag](
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
   * Validate current model and save the result.
   * @param validationTrigger validation trigger
   * @param validationDataSet validation dataset
   * @param validationMethods validation methods
   * @param coresPerNode cores per node
   * @param models cached models
   * @param state state table
   * @param validationSummary validation logger.
   * @param header log header string
   */
  protected def validate[T](validationTrigger: Option[Trigger],
    validationDataSet: Option[DataSet[MiniBatch[T]]],
    validationMethods: Option[Array[ValidationMethod[T]]],
    coresPerNode: Int,
    models: RDD[Cache[T]],
    state: Table,
    validationSummary: Option[ValidationSummary],
    header: String,
    parameters: AllReduceParameter[T] = null): Unit = {
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
    val start = System.nanoTime()
    val results = ZippedPartitionsWithLocalityRDD(models, validateRDD)((modelIter, dataIter) => {
      val cached = modelIter.next()
      val vMethodsArr = cached.localMethods
      val workingModels = cached.localModels

      // update with latest weight for validation
      if (parameters != null) {
        parameters.getWeights(cached.modelWeights.head.narrow(1,
          parameters.paramOffset, parameters.size))
          .waitResult()
      }

      if (Engine.getEngineType() == MklDnn) {
        if (dataIter.hasNext) workingModels.foreach(_.evaluate())
      } else {
        workingModels.foreach(_.evaluate())
      }
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
              if (Engine.getEngineType() == MklDnn && !workingModels(b).isInstanceOf[IRGraph[T]]) {
                Engine.dnnComputing.invokeAndWait2(Array(0).map(_ => () => {
                  workingModels(b).forward(input)
                }))
              } else {
                workingModels(b).forward(input)
              }

              val output = workingModels(b).output
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

    val validateTime = (System.nanoTime() - start) / 1e9f
    val count = results(0)._1.result()._2.toFloat
    // print validation throughput
    logger.info(s"$header validate model throughput is ${count / validateTime} records/second")

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
   ** Create checkpoint.
   * @param cacheTrigger cache trigger
   * @param cachePath cache path
   * @param isOverWrite whether over write
   * @param wallClockTime wall clock time
   * @param models cached models
   * @param state state table
   * @param parameters all reduce parameters
   * @param optimMethods all optim methods
   * @param trainingModel training model
   */
  protected def checkpoint[T: ClassTag](
    cacheTrigger: Option[Trigger],
    cachePath: Option[String],
    isOverWrite: Boolean,
    wallClockTime: Long,
    models: RDD[Cache[T]],
    state: Table,
    parameters: AllReduceParameter[T],
    optimMethods: Map[String, OptimMethod[T]],
    trainingModel: Module[T])(implicit ev: TensorNumeric[T]): Unit = {
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
   * Clean some internal states, so this or other optimizers can run optimize again
   * This method will be called at the end of optimize. You need not call it if optimize succeed.
   * If the optimize fails, you may call it before next optimize.
   */
  private[bigdl] def clearState[T: ClassTag](models: RDD[DistriOptimizer.Cache[T]]) : Unit = {
    // Reset the singleton flag, so other optimizers can run
    models.mapPartitions(iter => {
      Engine.resetSingletonFlag()
      iter
    }).count()
  }

  private[bigdl] def endEpoch[T: ClassTag](optimMethods: Map[String, OptimMethod[T]]): Unit = {
    optimMethods.foreach { case (moduleName, optimMethod) =>
      val records = optimMethod.state.get[Int]("recordsProcessedThisEpoch")
      if (records.isDefined && records.get != 0) {
        optimMethod.state("epoch") = optimMethod.state[Int]("epoch") + 1
        optimMethod.state("recordsProcessedThisEpoch") = 0
      }
    }
  }

  private[bigdl] def setTrainData[T: ClassTag](
    sampleRDD: RDD[Sample[T]],
    batchSize: Int,
    miniBatch: MiniBatch[T])(implicit ev: TensorNumeric[T])
    : DistributedDataSet[MiniBatch[T]] = {
    (DataSet.rdd(sampleRDD) ->
      SampleToMiniBatch(miniBatch, batchSize, None))
      .asInstanceOf[DistributedDataSet[MiniBatch[T]]]
  }

  private[bigdl] def setTrainData[T: ClassTag](sampleRDD: RDD[Sample[T]],
    batchSize: Int,
    featurePaddingParam: PaddingParam[T] = null,
    labelPaddingParam: PaddingParam[T] = null)(implicit ev: TensorNumeric[T])
    : DistributedDataSet[MiniBatch[T]] = {
    val _featurePaddingParam = if (featurePaddingParam != null) Some(featurePaddingParam) else None
    val _labelPaddingParam = if (labelPaddingParam != null) Some(labelPaddingParam) else None
    (DataSet.rdd(sampleRDD) ->
      SampleToMiniBatch(batchSize, _featurePaddingParam, _labelPaddingParam))
      .asInstanceOf[DistributedDataSet[MiniBatch[T]]]
  }


  private[bigdl] def prepareInput[T: ClassTag](dataset: DataSet[MiniBatch[T]],
    validationDataSet: Option[DataSet[MiniBatch[T]]]): Unit = {
    dataset.asInstanceOf[DistributedDataSet[MiniBatch[T]]].cache()
    if (validationDataSet.isDefined) {
      validationDataSet.get.toDistributed().cache()
    }
  }
}
