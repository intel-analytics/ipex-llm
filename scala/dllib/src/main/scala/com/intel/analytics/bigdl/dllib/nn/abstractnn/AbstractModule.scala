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

package com.intel.analytics.bigdl.nn.abstractnn

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.tensor.{Tensor, TensorDataType}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.utils.TorchObject.TYPE_MODULE
import org.apache.commons.lang3.SerializationUtils
import org.apache.spark.rdd.RDD
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.Graph.ModuleNode

import scala.reflect.ClassTag

/**
 * [[TensorModule]] is an abstract sub-class of [[AbstractModule]], whose
 * input and output type both are [[Tensor]].
 *
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
abstract class TensorModule[T: ClassTag]
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Tensor[T], Tensor[T], T]

/**
 * Module is the basic component of a neural network. It forward activities and backward gradients.
 * Modules can connect to others to construct a complex neural network.
 *
 * @tparam A Input data type
 * @tparam B Output data type
 * @tparam T Numeric type. Only support float/double now
 */
abstract class AbstractModule[A <: Activity: ClassTag, B <: Activity: ClassTag,
@specialized(Float, Double) T: ClassTag](
  implicit ev: TensorNumeric[T]) extends Serializable {

  private val namePostfix = Integer.toHexString(java.util.UUID.randomUUID().hashCode())
  /**
   * The cached output. So we don't compute it again when need it
   */
  var output: B = Activity[B, T]()

  /**
   * The cached gradient of activities. So we don't compute it again when need it
   */
  var gradInput: A = Activity[A, T]()

  /**
   * Copy the useful running status from src to this.
   *
   * The subclass should override this method if it has some parameters besides weight and bias.
   * Such as runningMean and runningVar of BatchNormalization.
   *
   * @param src source Module
   * @return this
   */
  def copyStatus(src: Module[T]) : this.type = {
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
  def clearState() : this.type = {
    if (output.isInstanceOf[Tensor[T]]) {
      output.asInstanceOf[Tensor[T]].set()
    }

    if (gradInput.isInstanceOf[Tensor[T]]) {
      gradInput.asInstanceOf[Tensor[T]].set()
    }

    this
  }

  private[nn] def allocateAs(dest: Activity): Activity = dest match {
    case tensor: Tensor[T] => Tensor[T]()
    case table: Table => T()
    case _ => throw new IllegalArgumentException("Activity only support tensor and table now")
  }

  /**
   * The name of the module
   */
  private var name : String = null

  /**
   * Set the module name
   *
   * @param name
   * @return
   */
  def setName(name : String) : this.type = {
    this.name = name
    this
  }

  /**
   * Get the module name, default name is className@namePostfix
   *
   * @return
   */
  def getName() : String = {
    if (this.name == null) {
      s"${this.getClass.getName}@${namePostfix}"
    } else {
      this.name
    }
  }

  protected var forwardTime = 0L

  protected var backwardTime = 0L

  def getTimes(): Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)] = {
    Array((this, forwardTime, backwardTime))
  }

  def resetTimes(): Unit = {
    forwardTime = 0
    backwardTime = 0
  }

  /**
   * Takes an input object, and computes the corresponding output of the module. After a forward,
   * the output state variable should have been updated to the new value.
   *
   * @param input input data
   * @return output data
   */
  final def forward(input: A): B = {
    val before = System.nanoTime()
    updateOutput(input)
    forwardTime += System.nanoTime() - before

    output
  }

  /**
   * Performs a back-propagation step through the module, with respect to the given input. In
   * general this method makes the assumption forward(input) has been called before, with the same
   * input. This is necessary for optimization reasons. If you do not respect this rule, backward()
   * will compute incorrect gradients.
   *
   * @param input input data
   * @param gradOutput gradient of next layer
   * @return gradient corresponding to input data
   */
  def backward(input: A, gradOutput: B): A = {
    val before = System.nanoTime()
    updateGradInput(input, gradOutput)
    accGradParameters(input, gradOutput)
    backwardTime += System.nanoTime() - before

    gradInput
  }

  /**
   * Computes the output using the current parameter set of the class and input. This function
   * returns the result which is stored in the output field.
   *
   * @param input
   * @return
   */
  def updateOutput(input: A): B

  /**
   * Computing the gradient of the module with respect to its own input. This is returned in
   * gradInput. Also, the gradInput state variable is updated accordingly.
   *
   * @param input
   * @param gradOutput
   * @return
   */
  def updateGradInput(input: A, gradOutput: B): A

  /**
   * Computing the gradient of the module with respect to its own parameters. Many modules do not
   * perform this step as they do not have any parameters. The state variable name for the
   * parameters is module dependent. The module is expected to accumulate the gradients with
   * respect to the parameters in some variable.
   *
   * @param input
   * @param gradOutput
   * @param scale
   */
  def accGradParameters(input: A, gradOutput: B, scale: Double = 1.0): Unit = {}

  /**
   * If the module has parameters, this will zero the accumulation of the gradients with respect
   * to these parameters. Otherwise, it does nothing.
   */
  def zeroGradParameters(): Unit = {}

  def updateParameters(learningRate: T): Unit = {}

  /**
   * This method compact all parameters and gradients of the model into two tensors. So it's easier
   * to use optim method
   *
   * @return
   */
  def getParameters(): (Tensor[T], Tensor[T]) = {
    val (weightParameters, gradParameters) = this.parameters()
    (Module.flatten[T](weightParameters), Module.flatten[T](gradParameters))
  }

  /**
   * This function returns two arrays. One for the weights and the other the gradients
   * Custom modules should override this function if they have parameters
   *
   * @return (Array of weights, Array of grad)
   */
  def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = null

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
  def getParametersTable(): Table = null

  /**
   * Module status. It is useful for modules like dropout/batch normalization
   */
  protected var train: Boolean = true

  def training(): this.type = {
    train = true
    this
  }

  def evaluate(): this.type = {
    train = false
    this
  }

  final def isTraining(): Boolean = {
    this.train
  }

  def reset(): Unit = {}

  protected var line = "\n"

  def setLine(line: String): this.type = {
    this.line = line
    this
  }

  private val engineType: EngineType = Engine.getEngineType()

  /**
   * get execution engine type
   */
  def checkEngineType(): this.type = {
    if (engineType != Engine.getEngineType()) {
      throw new Error("Module's EngineType doesn't march global EngineType")
    }
    this
  }

  def cloneModule(): AbstractModule[A, B, T] = {
    SerializationUtils.clone(this)
  }

  def canEqual(other: Any): Boolean = other.isInstanceOf[AbstractModule[A, B, T]]

  override def equals(other: Any): Boolean = other match {
    case that: AbstractModule[A, B, T] =>
      (that canEqual this) &&
        (that.getClass equals this.getClass) &&
        output == that.output &&
        gradInput == that.gradInput
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Object): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(output, gradInput, this.getClass)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }

  def save(path : String, overWrite: Boolean = false) : this.type = {
    this.clearState()
    File.save(this, path, overWrite)
    this
  }

  def saveTorch(path : String, overWrite: Boolean = false) : this.type = {
    this.clearState()
    File.saveTorch(this, path, TYPE_MODULE, overWrite)
    this
  }

  /**
   * @return Float or Double
   */
  def getNumericType(): TensorDataType = {
    ev.getType()
  }

  /**
   * module predict, return the probability distribution
   * @param dataset dataset for prediction
   */
  def predict(dataset: RDD[Sample[T]]): RDD[Activity] = {
    Predictor(this).predict(dataset)
  }

  /**
   * module predict, return the predict label
   * @param dataset dataset for prediction
   */
  def predictClass(dataset: RDD[Sample[T]]): RDD[Int] = {
    Predictor(this).predictClass(dataset)
  }

  /**
   * Set weight and bias for the module
   * @param newWeights array of weights and bias
   * @return
   */
  def setWeightsBias(newWeights: Array[Tensor[T]]): this.type = {
    require(parameters() != null, "this layer does not have weight/bias")
    require(parameters()._1.length == newWeights.length,
      "the number of input weight/bias is not consistant with number of weight/bias of this layer")
    val weights = parameters()._1
    for(i <- newWeights.indices) {
      weights(i).copy(newWeights(i))
    }
    this
  }

  /**
   * Get weight and bias for the module
   * @return array of weights and bias
   *
   */
  def getWeightsBias(): Array[Tensor[T]] = {
    if (parameters() != null) {
      parameters()._1
    } else {
      null
    }
  }

  /**
   * Some other modules point to current module
   * @param nodes upstream module nodes
   * @return node containing current module
   */
  def apply(nodes : ModuleNode[T]*): ModuleNode[T] = {
    require(this.isInstanceOf[AbstractModule[_, Tensor[T], T]],
      "AbstractModule: Only module with tensor output can be added into a graph node")
    val curNode = new ModuleNode[T](this.asInstanceOf[AbstractModule[Activity, Tensor[T], T]])
    nodes.foreach(node => {
      node -> curNode
    })
    curNode
  }

  /**
   * Find a module with given name. If there is no module with given name, it will return None. If
   * there are multiple modules with the given name, an exception will be thrown.
   * @param name
   * @return
   */
  def apply(name : String): Option[AbstractModule[Activity, Activity, T]] = {
    if (this.getName() == name) {
      Some(this)
    } else {
      None
    }
  }

  /**
   * use ValidationMethod to evaluate module
   * @param dataset dataset for test
   * @param vMethods validation methods
   * @param batchSize total batchsize of all partitions,
   *                  optional param and default 4 * partitionNum of dataset
   * @return
   */
  def evaluate(dataset: RDD[Sample[T]],
   vMethods: Array[ValidationMethod[T]],
   batchSize: Option[Int] = None): Array[(ValidationResult, ValidationMethod[T])] = {
    Evaluator(this).test(dataset, vMethods, batchSize)
  }
}

