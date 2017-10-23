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

import java.nio.ByteOrder

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.tensor.{Tensor, TensorDataType}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.nn.{Module, _}
import com.intel.analytics.bigdl.utils.TorchObject.TYPE_MODULE
import org.apache.commons.lang3.SerializationUtils
import org.apache.spark.rdd.RDD
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch, Sample}
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.quantized.Quantization
import com.intel.analytics.bigdl.utils.caffe.CaffePersister
import com.intel.analytics.bigdl.utils.serializer.ModulePersister
import com.intel.analytics.bigdl.utils.tf.{TensorflowDataFormat, TensorflowSaver}

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
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now
 */
abstract class AbstractModule[A <: Activity: ClassTag, B <: Activity: ClassTag, T: ClassTag](
  implicit ev: TensorNumeric[T]) extends Serializable {

  private var namePostfix = Integer.toHexString(java.util.UUID.randomUUID().hashCode())

  def getNamePostfix : String = namePostfix

  def setNamePostfix(namePostfix : String) : Unit = this.namePostfix = namePostfix

  /**
   * The cached output. So we don't compute it again when need it
   */
  var output: B = Activity.allocate[B, T]()

  /**
   * The cached gradient of activities. So we don't compute it again when need it
   */
  var gradInput: A = Activity.allocate[A, T]()

  /**
   * The scale of gradient weight and gradient bias
   * before gradParameters being accumulated.
   */
  protected var scaleW: Double = 1.0
  protected var scaleB: Double = 1.0

  /**
   * Get the scale of gradientWeight
   */
  def getScaleW(): Double = {
    scaleW
  }

  /**
   * Get the scale of gradientBias
   */
  def getScaleB(): Double = {
    scaleB
  }

  /**
   * Set the scale of gradientWeight
   *
   * @param w the value of the scale of gradientWeight
   * @return this
   */
  def setScaleW(w: Double): this.type = {
    scaleW = w
    this
  }

  /**
   * Set the scale of gradientBias
   *
   * @param b the value of the scale of gradientBias
   * @return this
   */
  def setScaleB(b: Double): this.type = {
    scaleB = b
    this
  }

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
    if (output.isInstanceOf[Tensor[_]]) {
      output.asInstanceOf[Tensor[_]].set()
    }

    if (gradInput.isInstanceOf[Tensor[_]]) {
      gradInput.asInstanceOf[Tensor[_]].set()
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

  def hasName: Boolean = name != null

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
      s"${this.getClass.getSimpleName}${namePostfix}"
    } else {
      this.name
    }
  }

  protected def getPrintName(): String = {
    val postfix = if (name == null) {
      namePostfix
    } else {
      name
    }
    s"${this.getClass.getSimpleName}[${postfix}]"

  }

  override def toString(): String = getPrintName

  protected var forwardTime = 0L

  protected var backwardTime = 0L

  def getTimes(): Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)] = {
    Array((this, forwardTime, backwardTime))
  }

  def resetTimes(): Unit = {
    forwardTime = 0
    backwardTime = 0
  }

  private var scaleWCache: Double = scaleW
  private var scaleBCache: Double = scaleB

  /**
   * freeze the module,
   * i.e. their parameters(weight/bias, if exists) are not changed in training process
   * if names is not empty,
   * set an array of layers that match the given ```names``` to be "freezed",
   *
   * @param names an array of layer names
   * @return current graph model
   */
  def freeze(names: String*): this.type = {
    if (names.isEmpty) {
      // in case when freeze is called many times
      if (scaleW != 0) {
        scaleWCache = scaleW
        scaleW = 0
      }
      if (scaleB != 0) {
        scaleBCache = scaleB
        scaleB = 0
      }
    } else {
      names.foreach(name => {
        this (name) match {
          case Some(x) => x.freeze()
          case _ => throw new Exception(s"cannot match module named $name")
        }
      })
    }
    this
  }

  /**
   * "unfreeze" module, i.e. make the module parameters(weight/bias, if exists)
   * to be trained(updated) in training process
   * if names is not empty, unfreeze layers that match given names
   *
   * @param names array of module names to unFreeze
   */
  def unFreeze(names: String*): this.type = {
    if (names.isEmpty) {
      scaleW = scaleWCache
      scaleB = scaleBCache
    } else {
      names.foreach(name => {
        this (name) match {
          case Some(x) => x.unFreeze()
          case _ => throw new Exception(s"cannot match module named $name")
        }
      })
    }
    this
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
    try {
      updateOutput(input)
    } catch {
      case l: LayerException =>
        l.layerMsg = this.toString() + "/" + l.layerMsg
        throw l
      case e: Throwable =>
        throw new LayerException(this.toString(), e)
    }
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
   */
  def accGradParameters(input: A, gradOutput: B): Unit = {}

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
        gradInput == that.gradInput &&
        name == that.name
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Object): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(output, gradInput, this.getClass, this.name)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }

  /**
   * Save this module to path.
   * @param path path to save module, local file system, HDFS and Amazon S3 is supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx"
   *             Amazon S3 path should be like "s3a://bucket/xxx"
   * @param overWrite if overwrite
   * @return self
   */
  @deprecated("please use recommended saveModule(path, overWrite)")
  def save(path : String, overWrite: Boolean = false) : this.type = {
    this.clearState()
    File.save(this, path, overWrite)
    this
  }

  /**
   * Save this module to path with protobuf format
   * @param path path to save module, local file system, HDFS and Amazon S3 is supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx"
   *             Amazon S3 path should be like "s3a://bucket/xxx"
   * @param overWrite if overwrite
   * @return self
   */
  def saveModule(path : String, overWrite: Boolean = false) : this.type = {
    this.clearState()
    ModulePersister.saveToFile(path, this, overWrite)
    this
  }

  /**
   * Save this module definition to path.
   * @param path path to save module, local file system, HDFS and Amazon S3 is supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx"
   *             Amazon S3 path should be like "s3a://bucket/xxx"
   * @param overWrite if overwrite
   * @return self
   */
  def saveDefinition(path : String, overWrite: Boolean = false) : this.type = {
    this.clearState()
    ModulePersister.saveModelDefinitionToFile(path, this, overWrite)
    this
  }

  def saveTorch(path : String, overWrite: Boolean = false) : this.type = {
    this.clearState()
    File.saveTorch(this, path, TYPE_MODULE, overWrite)
    this
  }

  def saveCaffe(prototxtPath: String, modelPath: String,
    useV2 : Boolean = true, overwrite : Boolean = false) : this.type = {
    this.clearState()
    CaffePersister.persist[T](prototxtPath, modelPath, this, useV2, overwrite)
    this
  }

  def saveTF(
              inputs : Seq[(String, Seq[Int])],
              path: String,
              byteOrder: ByteOrder = ByteOrder.LITTLE_ENDIAN,
              dataFormat: TensorflowDataFormat = TensorflowDataFormat.NHWC): this.type = {
    require(this.isInstanceOf[Graph[T]], "only Graph container can be saved as Tensorflow model")
    this.clearState()
    TensorflowSaver.saveGraph(this.asInstanceOf[Graph[T]], inputs, path, byteOrder, dataFormat)
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
   * @param batchSize total batchSize for all partitions.
   *                  if -1, default is 4 * partitionNumber of datatset
   * @param shareBuffer whether to share same memory for each batch predict results
   */
  def predict(dataset: RDD[Sample[T]],
              batchSize: Int = -1,
              shareBuffer: Boolean = false): RDD[Activity] = {
    Predictor(this).predict(dataset, batchSize, shareBuffer)
  }

  /**
   * module predict, return the predict label
   * @param dataset dataset for prediction
   * @param batchSize total batchSize for all partitions.
   *                  if -1, default is 4 * partitionNumber of dataset
   */
  def predictClass(dataset: RDD[Sample[T]], batchSize: Int = -1): RDD[Int] = {
    Predictor(this).predictClass(dataset, batchSize)
  }

  /**
   * Set weight and bias for the module
   * @param newWeights array of weights and bias
   * @return
   */
  def setWeightsBias(newWeights: Array[Tensor[T]]): this.type = {
    require(parameters() != null, "this layer does not have weight/bias")
    require(parameters()._1.length == newWeights.length,
      "the number of input weight/bias is not consistant with " +
        "number of weight/bias of this layer, " +
        s"number of input ${parameters()._1.length}," +
        s" number of output ${newWeights.length}")
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
   * save weights and bias to file
   * @param path file to save
   * @param overWrite whether to overwrite or not
   */
  def saveWeights(path: String, overWrite: Boolean): Unit = {
    val parameterTable = getParametersTable()
    val weightsBiasTable = T()
    parameterTable.foreach {
      case (name: String, params: Table) =>
        val wb = T()
        if (params.contains("weight")) {
          wb("weight") = params("weight")
        }
        if (params.contains("bias")) {
          wb("bias") = params("bias")
        }
        weightsBiasTable(name) = wb
      case _ => throw new UnsupportedOperationException("invalid parameter table")
    }
    weightsBiasTable.save(path, overWrite)
  }

  /**
   * load pretrained weights and bias to current module
   * @param weightPath file to store weights and bias
   * @param matchAll whether to match all layers' weights and bias,
   *                 if not, only load existing pretrained weights and bias
   * @return current module
   */
  def loadWeights(weightPath: String, matchAll: Boolean = true): this.type = {
    val srcParameter = File.load[Table](weightPath)
    val targetParameter = getParametersTable()
    copyWeights(targetParameter, srcParameter, matchAll)
    this
  }

  /**
   * copy weights from another model, mapping by layer name
   * @param srcModel model to copy from
   * @param matchAll whether to match all layers' weights and bias,
   * @return current module
   */
  def loadModelWeights(srcModel: Module[Float], matchAll: Boolean = true): this.type = {
    val srcParameter = srcModel.getParametersTable()
    val targetParameter = getParametersTable()
    copyWeights(targetParameter, srcParameter, matchAll)
    this
  }

  private def copyWeights(target: Table, src: Table, matchAll: Boolean): Unit = {
    target.foreach {
      case (name: String, targetParams: Table) =>
        if (src.contains(name)) {
          val srcParams = src[Table](name)
          if (srcParams.contains("weight")) {
            val w = srcParams[Tensor[T]]("weight")
            targetParams[Tensor[T]]("weight").resizeAs(w).copy(w)
          }
          if (srcParams.contains("bias")) {
            val b = srcParams[Tensor[T]]("bias")
            targetParams[Tensor[T]]("bias").resizeAs(b).copy(b)
          }
        } else {
          if (matchAll) new Exception(s"module $name cannot find corresponding weight bias")
        }
    }
  }

  /**
   * Build graph: some other modules point to current module
   * @param nodes upstream module nodes
   * @return node containing current module
   */
  def inputs(nodes : ModuleNode[T]*): ModuleNode[T] = {
    val curNode = new ModuleNode[T](this)
    nodes.foreach(node => {
      node.add(curNode, Edge())
    })
    curNode
  }

  /**
   * Build graph: some other modules point to current module
   * @param first distinguish from another inputs when input parameter list is empty
   * @param nodesWithIndex upstream module nodes and the output tensor index. The start index is 1.
   * @return node containing current module
   */
  def inputs(first: (ModuleNode[T], Int), nodesWithIndex : (ModuleNode[T], Int)*): ModuleNode[T] = {
    val curNode = new ModuleNode[T](this)
    first._1.add(curNode, Edge(first._2))
    nodesWithIndex.foreach(nodeWithIndex => {
      nodeWithIndex._1.add(curNode, Edge(nodeWithIndex._2))
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


  def evaluate(dataSet: LocalDataSet[MiniBatch[T]],
               vMethods: Array[ValidationMethod[T]]
              ): Array[(ValidationResult, ValidationMethod[T])] = {
    Validator(this, dataSet).test(vMethods)
  }

  def quantize(): Module[T] = {
    Quantization.quantize(this)
  }


  /**
   * Generate end nodes of current module with start nodes
   * @param startNodes: current start nodes
   * @return current end nodes
   */
  private[bigdl] def getEndNodes(startNodes: Array[ModuleNode[T]]): Array[ModuleNode[T]] = {
    val endNodes = Array(this.inputs(startNodes: _*))
    endNodes
  }

  /**
   * Generate graph module with start nodes
   * @param startNodes
   * @return
   */
  def toGraph(startNodes: ModuleNode[T]*): Graph[T] = {
    val starts = if (startNodes.isEmpty) Array(Input[T]()) else startNodes.toArray
    val endNodes = this.getEndNodes(starts)
    Graph(starts, endNodes)
  }
}

