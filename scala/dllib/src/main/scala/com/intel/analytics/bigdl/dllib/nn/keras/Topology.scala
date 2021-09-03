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

package com.intel.analytics.bigdl.nn.keras

import com.intel.analytics.bigdl.{Criterion, DataSet}
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{Container, StaticGraph, Sequential => TSequential}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.serialization.Bigdl.BigDLModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{LoggerFilter, Shape}
import com.intel.analytics.bigdl.utils.serializer._
import org.apache.spark.rdd.RDD

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

abstract class KerasModel[T: ClassTag](implicit ev: TensorNumeric[T])
  extends KerasLayer[Activity, Activity, T] {

  def getSubModules(): List[AbstractModule[Activity, Activity, T]] = {
    require(this.labor.isInstanceOf[Container[Activity, Activity, T]],
      "labor should be a container, but we got: $this")
    this.labor.asInstanceOf[Container[Activity, Activity, T]].modules.toList
  }

  private var optimMethod: OptimMethod[T] = null
  private var criterion: Criterion[T] = null
  private var vMethods: Array[ValidationMethod[T]] = null

  /**
   * Configure the learning process. Must be called before fit or evaluate.
   * @param optimizer Optimization method to be used.
   * @param loss Criterion to be used.
   * @param metrics Array of validation methods to be used.
   */
  // TODO: support checkpoint, summary, etc.
  def compile(optimizer: OptimMethod[T],
              loss: Criterion[T],
              metrics: Array[ValidationMethod[T]] = null): Unit = {
    LoggerFilter.redirectSparkInfoLogs()
    this.optimMethod = optimizer
    this.criterion = loss
    this.vMethods = metrics
  }

  /**
   * Alternatively, one can pass in string representations when calling compile.
   * For example: optimizer = "sgd", loss = "mse", metrics = Array("accuracy")
   */
  def compile(optimizer: String,
              loss: String,
              metrics: Array[String])
    (implicit ev: TensorNumeric[T]): Unit = {
    this.compile(KerasUtils.toBigDLOptimMethod[T](optimizer),
      KerasUtils.toBigDLCriterion[T](loss),
      KerasUtils.toBigDLMetrics[T](metrics))
  }

  private def toDataSet(x: RDD[Sample[T]], batchSize: Int)
  : DataSet[MiniBatch[T]] = {
    if (x != null) DataSet.rdd(x) -> SampleToMiniBatch[T](batchSize)
    else null
  }

  /**
   * Train a model for a fixed number of epochs on a dataset.
   * @param x Training dataset. If x is an instance of LocalDataSet, train in local mode.
   * @param nbEpoch Number of iterations to train.
   * @param validationData Dataset for validation, or null if validation is not configured.
   */
  def fit[D: ClassTag](x: DataSet[D], nbEpoch: Int,
                       validationData: DataSet[MiniBatch[T]])
    (implicit ev: TensorNumeric[T]): Unit = {
    require(this.optimMethod != null && this.criterion != null,
      "compile must be called before fit")
    val optimizer = Optimizer(
      model = this,
      dataset = x,
      criterion = this.criterion)
    if (validationData != null) {
      require(this.vMethods != null, "Validation metrics haven't been set yet")
      optimizer.setValidation(trigger = Trigger.everyEpoch,
        dataset = validationData,
        vMethods = this.vMethods)
    }
    optimizer.setOptimMethod(this.optimMethod)
      .setEndWhen(Trigger.maxEpoch(nbEpoch))
    optimizer.optimize()
  }

  /**
   * Train a model for a fixed number of epochs on a dataset.
   * @param x Training dataset, RDD of Sample.
   * @param batchSize Number of samples per gradient update.
   * @param nbEpoch Number of iterations to train.
   * @param validationData RDD of Sample, or null if validation is not configured.
   */
  def fit(x: RDD[Sample[T]], batchSize: Int = 32, nbEpoch: Int = 10,
          validationData: RDD[Sample[T]] = null)
    (implicit ev: TensorNumeric[T]): Unit = {
    this.fit(toDataSet(x, batchSize), nbEpoch, toDataSet(validationData, batchSize))
  }

  /**
   * Evaluate a model on a given dataset.
   * @param x Evaluation dataset, RDD of Sample.
   * @param batchSize Number of samples per batch.
   */
  def evaluate(x: RDD[Sample[T]],
               batchSize: Int)
      (implicit ev: TensorNumeric[T]): Array[(ValidationResult, ValidationMethod[T])] = {
    require(this.vMethods != null, "Evaluation metrics haven't been set yet")
    this.evaluate(x, this.vMethods, Some(batchSize))
  }

  /**
   * Evaluate a model in local mode.
   * @param x Evaluation dataset, LocalDataSet.
   */
  def evaluate(x: LocalDataSet[MiniBatch[T]])
    (implicit ev: TensorNumeric[T]): Array[(ValidationResult, ValidationMethod[T])] = {
    require(this.vMethods != null, "Evaluation metrics haven't been set yet")
    this.evaluate(x, this.vMethods)
  }

  /**
   * Use a model to do prediction.
   * @param x Prediction data, RDD of Sample.
   * @param batchSize Number of samples per batch.
   */
  def predict(x: RDD[Sample[T]],
              batchSize: Int)(implicit ev: TensorNumeric[T]): RDD[Activity] = {
    this.predict(x, batchSize, false)
  }

  /**
   * Use a model to do prediction in LOCAL mode.
   * @param x Prediction data, LocalDataSet.
   */
  def predict(x: LocalDataSet[MiniBatch[T]])(implicit ev: TensorNumeric[T]): Array[Activity] = {
    val localPredictor = LocalPredictor(this)
    localPredictor.predict(x)
  }

}


@deprecated("`Model` is deprecated." +
  "com.intel.analytics.bigdl.nn.keras is deprecated in BigDL 0.11, " +
  "and will be removed in future releases", "0.10.0")
class Model[T: ClassTag](private val _inputs : Seq[ModuleNode[T]],
      private val _outputs : Seq[ModuleNode[T]])(implicit ev: TensorNumeric[T])
  extends KerasModel[T] {
  this.labor = doBuild(null)

  excludeInvalidLayers(this.labor.asInstanceOf[StaticGraph[T]].
    getForwardExecutions().map {_.element})

  this.inputShapeValue = Shape(_inputs.map{n => n.element.getInputShape()}.toList)

  this.outputShapeValue = Shape(_outputs.map{_.element.getOutputShape()}.toList)

  override def isKerasStyle(): Boolean = true

  override def computeOutputShape(inputShape: Shape): Shape = {
    getOutputShape()
  }

  override def doBuild(inputShape: Shape): StaticGraph[T] =
    new StaticGraph[T](_inputs, _outputs, None, false)

  override def build(calcInputShape: Shape): Shape = {
    checkWithCurrentInputShape(calcInputShape)
    getOutputShape()
  }
}

object Model extends KerasLayerSerializable{
  /**
   * Build multiple inputs, multiple outputs graph container.
   * @param input input node
   * @param output output node
   * @return a graph container
   */
  def apply[T: ClassTag](
      input : Array[ModuleNode[T]],
      output : Array[ModuleNode[T]])(implicit ev: TensorNumeric[T]) : Model[T] = {
    new Model[T](input, output)
  }

  /**
   * Build a single input, multiple outputs graph container
   * @param input input node
   * @param output output nodes
   * @return a graph container
   */
  def apply[T: ClassTag](input : ModuleNode[T], output : Array[ModuleNode[T]])
                        (implicit ev: TensorNumeric[T]) : Model[T] = {
    new Model[T](Seq(input), output)
  }

  /**
   * Build a multiple inputs, single output graph container
   * @param input input nodes
   * @param output output node
   * @return a graph container
   */
  def apply[T: ClassTag](input : Array[ModuleNode[T]], output : ModuleNode[T])
                        (implicit ev: TensorNumeric[T]) : Model[T] = {
    new Model[T](input, Seq(output))
  }
  /**
   * Build a single input, single output graph container
   * @param input input nodes
   * @param output output nodes
   * @return a graph container
   */
  def apply[T: ClassTag](input : ModuleNode[T], output : ModuleNode[T])
                        (implicit ev: TensorNumeric[T]) : Model[T] = {
    new Model[T](Seq(input), Seq(output))
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
      builder: BigDLModule.Builder)
    (implicit ev: TensorNumeric[T]): Unit = {
    val labor = context.moduleData.module.
      asInstanceOf[KerasLayer[Activity, Activity, T]].labor
    val subModule = ModuleSerializer.serialize(SerializeContext(ModuleData(labor,
      new ArrayBuffer[String](), new ArrayBuffer[String]()), context.storages,
      context.storageType, _copyWeightAndBias))
    builder.addSubModules(subModule.bigDLModule)
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
    val subProtoModules = context.bigdlModule.getSubModulesList.asScala
    val subModules = subProtoModules.map(module => {
      val subModuleData = ModuleSerializer.load(DeserializeContext(module,
        context.storages, context.storageType, _copyWeightAndBias))
      subModuleData.module
    })
    val tGraph = subModules(0).asInstanceOf[StaticGraph[T]]
    Model(tGraph.inputs.toArray, tGraph.outputs.toArray)
  }

}

@deprecated("`Sequential` is deprecated." +
  "com.intel.analytics.bigdl.nn.keras is deprecated in BigDL 0.11, " +
  "and will be removed in future releases", "0.10.0")
class Sequential[T: ClassTag]()
(implicit ev: TensorNumeric[T]) extends KerasModel[T] {

  private[bigdl] var frozen: Boolean = false

  this.labor = doBuild(null)

  private def triggerBuilding(module: AbstractModule[_ <: Activity, _ <: Activity, T]): Unit = {
    if (!this.isBuilt()) {
      if (module.getInputShape() == null) {
        throw new RuntimeException("The first layer should explicitly declare inputshape")
      } else {
        val outputShape = module.build(module.getInputShape())
        // The inputShape of Sequential should only be init here.
        this.inputShapeValue = module.getInputShape()
        this.outputShapeValue = outputShape
      }
    } else {
      val outputShape = module.build(this.getOutputShape())
      this.outputShapeValue = outputShape
    }
  }

  /**
   * Add a sub-module to the contained `modules`
   *
   * @param module module to be add
   * @return this container
   */
  def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]): this.type = {
    if (frozen) {
      throw new RuntimeException(
        "This Sequential has been frozen, as it has been added into other container")
    }
    if (module.isInstanceOf[Sequential[T]]) {
      module.asInstanceOf[Sequential[T]].frozen = true
    }
    validateInput[T](Seq(module))

    triggerBuilding(module)

    labor.asInstanceOf[TSequential[T]].modules +=
      module.asInstanceOf[AbstractModule[Activity, Activity, T]]
    checkDuplicate()
    this
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    if (labor.asInstanceOf[TSequential[T]].modules.isEmpty) {
      inputShape
    } else {
      labor.asInstanceOf[TSequential[T]].modules.last.getOutputShape()
    }
  }

  override def doBuild(inputShape: Shape): TSequential[T] = TSequential[T]()

  override def build(calcInputShape: Shape): Shape = {
    checkWithCurrentInputShape(calcInputShape)
    getOutputShape()
  }
}

object Sequential extends KerasLayerSerializable{
  def apply[@specialized(Float, Double) T: ClassTag]()
     (implicit ev: TensorNumeric[T]) : Sequential[T] = {
    new Sequential[T]()
  }
}
