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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Tensor, TensorDataType}
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.utils.serializer._
import serialization.Bigdl.{AttrValue, BigDLModule}

import scala.reflect.ClassTag

object LaborAdapterSerializer extends ModuleSerializable {

  override def doLoadModule[T: ClassTag](context : DeserializeContext)
           (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val laborAdapter = super.doLoadModule(context).asInstanceOf[LaborAdapter[Activity, Activity, T]]
    val attrMap = context.bigdlModule.getAttrMap
    laborAdapter.labor = DataConverter.getAttributeValue(context, attrMap.get("labor")).
      asInstanceOf[AbstractModule[Activity, Activity, T]]
    laborAdapter
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              moduleBuilder : BigDLModule.Builder)
                                             (implicit ev: TensorNumeric[T]) : Unit = {

    super.doSerializeModule(context, moduleBuilder)
    val laborAdapterModule =
      context.moduleData.module.asInstanceOf[LaborAdapter[Activity, Activity, T]]
    val laborBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, laborBuilder, laborAdapterModule.labor,
      ModuleSerializer.abstractModuleType)
    moduleBuilder.putAttr("labor", laborBuilder.build)

    val serializerFlagBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, serializerFlagBuilder, true,
      scala.reflect.runtime.universe.typeOf[Boolean])
    moduleBuilder.putAttr("is_labor_module", serializerFlagBuilder.build)
  }
}

abstract class LaborAdapter[A <: Activity: ClassTag, B <: Activity: ClassTag, T: ClassTag]
(implicit ev: TensorNumeric[T]) extends AbstractModule[A, B, T]{

  var labor: AbstractModule[A, B, T] = null

  override def getBatchInputShape(): Activity = labor.getBatchInputShape()

  override def getBatchOutputShape(): Activity = labor.getBatchOutputShape()

  override def build(inputShape: Activity): Unit = {
    labor = doBuild(inputShape)

    output = labor.output

    gradInput = labor.gradInput

    labor.build(inputShape)
  }

  def doBuild(inputShape: Activity): AbstractModule[A, B, T]


  override def forward(input: A): B = labor.forward(input)

  override def backward(input: A, gradOutput: B): A = labor.backward(input, gradOutput)

  /**
   * Get the scale of gradientWeight
   */
  override def getScaleW(): Double = labor.getScaleW()

  /**
   * Get the scale of gradientBias
   */
  override def getScaleB(): Double = labor.getScaleB()

  /**
   * Set the scale of gradientWeight
   *
   * @param w the value of the scale of gradientWeight
   * @return this
   */
  override def setScaleW(w: Double): this.type = {
    labor.setScaleW(w)
    this
  }

  /**
   * Set the scale of gradientBias
   *
   * @param b the value of the scale of gradientBias
   * @return this
   */
  override def setScaleB(b: Double): this.type = {
    labor.setScaleB(b)
    this
  }

  /**
   * Clear cached activities to save storage space or network bandwidth. Note that we use
   * Tensor.set to keep some information like tensor share
   *
   * The subclass should override this method if it allocate some extra resource, and call the
   * super.clearState in the override method
   *
   * @return
   */
  override def clearState() : this.type = {
    labor.clearState()
    this
  }

  override def toString(): String = {
    val prefix = getPrintName()
    val details = if (labor != null) {"<" + labor.toString + ">"} else ""
    prefix + details
  }


  override def getTimes(): Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)] = {
    labor.getTimes()
  }

  override def resetTimes(): Unit = labor.resetTimes()

  /**
   * freeze the module,
   * i.e. their parameters(weight/bias, if exists) are not changed in training process
   * if names is not empty,
   * set an array of layers that match the given ```names``` to be "freezed",
   *
   * @param names an array of layer names
   * @return current graph model
   */
  override def freeze(names: String*): this.type = {
    labor.freeze(names : _*)
    this
  }

  /**
   * "unfreeze" module, i.e. make the module parameters(weight/bias, if exists)
   * to be trained(updated) in training process
   * if names is not empty, unfreeze layers that match given names
   *
   * @param names array of module names to unFreeze
   */
  override def unFreeze(names: String*): this.type = {
    labor.unFreeze(names : _*)
    this
  }


  /**
   * Computes the output using the current parameter set of the class and input. This function
   * returns the result which is stored in the output field.
   *
   * @param input
   * @return
   */
  override def updateOutput(input: A): B = labor.updateOutput(input)

  /**
   * Computing the gradient of the module with respect to its own input. This is returned in
   * gradInput. Also, the gradInput state variable is updated accordingly.
   *
   * @param input
   * @param gradOutput
   * @return
   */
  override def updateGradInput(input: A, gradOutput: B): A = {
    labor.updateGradInput(input, gradOutput)
  }

  /**
   * Computing the gradient of the module with respect to its own parameters. Many modules do not
   * perform this step as they do not have any parameters. The state variable name for the
   * parameters is module dependent. The module is expected to accumulate the gradients with
   * respect to the parameters in some variable.
   *
   * @param input
   * @param gradOutput
   */
  override def accGradParameters(input: A, gradOutput: B): Unit = {
    labor.accGradParameters(input, gradOutput)
  }

  /**
   * If the module has parameters, this will zero the accumulation of the gradients with respect
   * to these parameters. Otherwise, it does nothing.
   */
  override def zeroGradParameters(): Unit = labor.zeroGradParameters()

  override def updateParameters(learningRate: T): Unit = labor.updateParameters(learningRate)

  /**
   * This method compact all parameters and gradients of the model into two tensors. So it's easier
   * to use optim method
   *
   * @return
   */
  override def getParameters(): (Tensor[T], Tensor[T]) = labor.getParameters()

  /**
   * This function returns two arrays. One for the weights and the other the gradients
   * Custom modules should override this function if they have parameters
   *
   * @return (Array of weights, Array of grad)
   */
  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = labor.parameters()

  /**
   * Get extra parameter in this module.
   * Extra parameter means the trainable parameters beside weight and bias. Such as runningMean
   * and runningVar in BatchNormalization.
   *
   * The subclass should override this method if it has some parameters besides weight and bias.
   *
   * @return an array of tensor
   */
  override def getExtraParameter(): Array[Tensor[T]] = labor.getExtraParameter()

  /**
   * Set extra parameter to this module.
   * Extra parameter means the trainable parameters beside weight and bias. Such as runningMean
   * and runningVar in BatchNormalization.
   *
   * @return this
   */
  override def setExtraParameter(extraParam: Array[Tensor[T]]): this.type = {
    labor.setExtraParameter(extraParam)
    this
  }

  /**
   * This function returns a table contains ModuleName, the parameter names and parameter value
   * in this module.
   * The result table is a structure of Table(ModuleName -> Table(ParameterName -> ParameterValue)),
   * and the type is Table[String, Table[String, Tensor[T]]].
   *
   * For example, get the weight of a module named conv1:
   *   table[Table]("conv1")[Tensor[T]]("weight").
   *
   * Custom modules should override this function if they have parameters.
   *
   * @return Table
   */
  override def getParametersTable(): Table = labor.getParametersTable()

  override def training(): this.type = {
    labor.training()
    this
  }

  override def evaluate(): this.type = {
    labor.evaluate()
    this
  }

  override def isTraining(): Boolean = labor.isTraining()

  override def reset(): Unit = labor.reset()

  /**
   * get execution engine type
   */
  override def checkEngineType(): this.type = {
    labor.checkEngineType()
    this
  }

  /**
   * Set weight and bias for the module
   * @param newWeights array of weights and bias
   * @return
   */
  override def setWeightsBias(newWeights: Array[Tensor[T]]): this.type = {
    labor.setWeightsBias(newWeights)
    this
  }

  /**
   * Get weight and bias for the module
   * @return array of weights and bias
   *
   */
  override def getWeightsBias(): Array[Tensor[T]] = labor.getWeightsBias()

  override def quantize(): Module[T] = labor.quantize()

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) =
    labor.getClassTagNumerics()

  override def getNumericType(): TensorDataType = labor.getNumericType()
  
}
