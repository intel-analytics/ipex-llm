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
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.quantized.Quantization
import com.intel.analytics.bigdl.nn.{Module, _}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{QuantizedTensor, Tensor, TensorDataType}
import com.intel.analytics.bigdl.transform.vision.image.{DistributedImageFrame, ImageFeature, ImageFrame, LocalImageFrame}
import com.intel.analytics.bigdl.utils.TorchObject.TYPE_MODULE
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.utils.caffe.CaffePersister
import com.intel.analytics.bigdl.utils.intermediate.ConversionUtils
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.tf.{TensorflowDataFormat, TensorflowSaver}
import org.apache.commons.lang3.SerializationUtils
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.reflect.ClassTag

/**
 * [[TensorModule]] is an abstract sub-class of [[AbstractModule]], whose
 * input and output type both are [[Tensor]].
 *
 * @tparam T The numeric type in this module parameters
 */
abstract class TensorModule[T: ClassTag]
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Tensor[T], Tensor[T], T]

/**
 * Module is the basic component of a neural network. It forward activities and backward gradients.
 * Modules can connect to others to construct a complex neural network.
 *
 * @tparam A Input data type
 * @tparam B Output data type
 * @tparam T The numeric type in this module parameters.
 */
abstract class AbstractModule[A <: Activity: ClassTag, B <: Activity: ClassTag, T: ClassTag](
  implicit ev: TensorNumeric[T]) extends Serializable with InferShape{

  // ================================= Public APIs =============================================


  /**
   * The cached output. So we don't compute it again when need it
   */
  var output: B = Activity.allocate[B, T]()

  /**
   * The cached gradient of activities. So we don't compute it again when need it
   */
  var gradInput: A = Activity.allocate[A, T]()

  /**
   * Get the scale of gradientWeight
   */
  final def getScaleW(): Double = {
    scaleW
  }

  /**
   * Get the scale of gradientBias
   */
  final def getScaleB(): Double = {
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

  /**
   * Whether user set a name to the module before
   * @return
   */
  final def hasName: Boolean = name != null

  /**
   * Set the module name
   *
   * @param name
   * @return
   */
  final def setName(name : String) : this.type = {
    this.name = name
    this
  }

  /**
   * Get the module name, default name is className@namePostfix
   *
   * @return
   */
  final def getName() : String = {
    if (this.name == null) {
      s"${this.getClass.getSimpleName}${namePostfix}"
    } else {
      this.name
    }
  }

  override def toString(): String = getPrintName

  /**
   * Get the forward/backward cost time for the module or its submodules
   * @return
   */
  def getTimes(): Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)] = {
    Array((this, forwardTime, backwardTime))
  }

  /**
   * Get the forward/backward cost time for the module or its submodules
   * and group by module type.
   * @return (module type name, forward time, backward time)
   */
  final def getTimesGroupByModuleType():
      Array[(String, Long, Long)] = {
    this.getTimes().map(v => (v._1.getClass().getName(), v._2, v._3)).groupBy(_._1)
      .map(v => (v._1, v._2.reduce((a, b) => (v._1, a._2 + b._2, a._3 + b._3))))
      .map(v => (v._1, v._2._2, v._2._3))
      .toArray
      .sortWith((a, b) => (a._2 + a._3) > (b._2 + b._3))
  }

  /**
   * Reset the forward/backward record time for the module or its submodules
   * @return
   */
  def resetTimes(): Unit = {
    forwardTime = 0
    backwardTime = 0
  }

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
      updateParameter
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
    asyncGradient
    gradInput
  }

  private[bigdl] def asyncGradient(): Unit = {
    if (this.getParameterSynchronizer() != null) {
      if (this.parameters() != null) {
        this.getParameterSynchronizer.put(this.getName)
      }
    }
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
  def zeroGradParameters(): Unit = {
    if (parameters() != null) {
      parameters()._1.zip(parameters()._2)foreach{ case (weight, grad) =>
        grad.resizeAs(weight).zero()
      }
    }
  }

  /**
   * This function returns two arrays. One for the weights and the other the gradients
   * Custom modules should override this function if they have parameters
   *
   * @return (Array of weights, Array of grad)
   */
  def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = null

  /**
   * Get extra parameter in this module.
   * Extra parameter means the trainable parameters beside weight and bias. Such as runningMean
   * and runningVar in BatchNormalization.
   *
   * The subclass should override this method if it has some parameters besides weight and bias.
   *
   * @return an array of tensor
   */
  def getExtraParameter(): Array[Tensor[T]] = null

  /**
   * Set extra parameter to this module.
   * Extra parameter means the trainable parameters beside weight and bias. Such as runningMean
   * and runningVar in BatchNormalization.
   *
   * @return this
   */
  final def setExtraParameter(extraParam: Array[Tensor[T]]): this.type = {
    val currentExtraParam = this.getExtraParameter()
    if (extraParam != null && currentExtraParam != null) {
      require(extraParam.length == currentExtraParam.length,
        "state's length doesn't match, excepted:" +
          s"${currentExtraParam.length}, but got  ${extraParam.length}")
      var i = 0
      while (i < extraParam.length) {
        currentExtraParam(i).copy(extraParam(i))
        i += 1
      }
      this
    } else if (extraParam == null && currentExtraParam == null) {
      this
    } else {
      throw new IllegalArgumentException(s"module's extraParameter is $currentExtraParam" +
        s", while setting param is ${extraParam}")
    }
  }

  /**
   * This function returns a table contains ModuleName, the parameter names and parameter value
   * in this module.
   *
   * The result table is a structure of Table(ModuleName -> Table(ParameterName -> ParameterValue)),
   * and the type is Table[String, Table[String, Tensor[T]]].
   *
   * For example, get the weight of a module named conv1:
   *   table[Table]("conv1")[Tensor[T]]("weight").
   *
   * The names of the parameters follow such convention:
   *
   * 1. If there's one parameter, the parameter is named as "weight", the gradient is named as
   * "gradWeight"
   *
   * 2. If there're two parameters, the first parameter is named as "weight", the first gradient is
   * named as "gradWeight"; the second parameter is named as "bias", the seconcd gradient is
   * named as "gradBias"
   *
   * 3. If there're more parameters, the weight is named as "weight" with a seq number as suffix,
   * the gradient is named as "gradient" with a seq number as suffix
   *
   * Custom modules should override this function the default impl if the convention doesn't meet
   * the requirement.
   *
   * @return Table
   */
  def getParametersTable(): Table = {
    val params = parameters()
    if (params == null) return null
    val (weights, gradients) = params
    require(gradients.length == weights.length, "weight number is not equal to grad number")

    if (weights.length == 1) {
      T(getName() -> T("weight" -> weights(0), "gradWeight" -> gradients(0)))
    } else if (weights.length == 2) {
      T(getName() -> T("weight" -> weights(0), "bias" -> weights(1),
        "gradWeight" -> gradients(0), "gradBias" -> gradients(1)))
    } else {
      val result = T()
      weights.zip(gradients).zipWithIndex.map { case ((w, g), i) =>
        result(s"weight$i") = w
        result(s"gradient$i") = g
      }
      T(getName() -> result)
    }
  }

  /**
   * Set the module to training mode
   * @return
   */
  def training(): this.type = {
    train = true
    this
  }

  /**
   * Set the module to evaluate mode
   * @return
   */
  def evaluate(): this.type = {
    train = false
    this
  }

  /**
   * Check if the model is in training mode
   * @return
   */
  final def isTraining(): Boolean = {
    this.train
  }

  /**
   * Reset module parameters, which is re-initialize the parameter with given initMethod
   */
  def reset(): Unit = {}

  /**
   * Set the line separator when print the module
   * @param line
   * @return
   */
  final def setLine(line: String): this.type = {
    this.line = line
    this
  }

  /**
   * Clone the model
   * @return
   */
  final def cloneModule(): this.type = {
    SerializationUtils.clone(this)
  }

  /**
   * Clone the module, deep or shallow copy
   * @param deepCopy
   * @return
   */
  final def clone(deepCopy : Boolean): AbstractModule[A, B, T] = {
    val moduleData = ModuleData[T](this.
      asInstanceOf[AbstractModule[Activity, Activity, T]], Seq[String](), Seq[String]())
    val storages = new mutable.HashMap[Int, Any]()
    val context = SerializeContext(moduleData, storages, ProtoStorageType, false)
    val serializedModule = ModuleSerializer.serialize[T](context).bigDLModule
    ModulePersister.setTensorStorage(serializedModule, storages)

    storages.clear()

    val deserializeContext = DeserializeContext(serializedModule.build,
      storages, ProtoStorageType, false)
    ModuleLoader.initTensorStorage[T](deserializeContext)
    val copy = ModuleSerializer.load[T](deserializeContext).module
      .asInstanceOf[AbstractModule[A, B, T]]
    setWeightAndBias(copy, deepCopy)
    copy
  }

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
  @deprecated("please use recommended saveModule(path, overWrite)", "0.3.0")
  final def save(path : String, overWrite: Boolean = false) : this.type = {
    this.clearState()
    File.save(this, path, overWrite)
    this
  }

  /**
   * Save this module to path with protobuf format
   * @param path path to save module, local file system, HDFS and Amazon S3 is supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx"
   *             Amazon S3 path should be like "s3a://bucket/xxx"
   * @param weightPath where to store weight
   * @param overWrite if overwrite
   * @return self
   */
  final def saveModule(path : String, weightPath : String = null,
    overWrite: Boolean = false) : this.type = {
    this.clearState()
    ModulePersister.saveToFile(path, weightPath, this, overWrite)
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
  final def saveDefinition(path : String, overWrite: Boolean = false) : this.type = {
    this.clearState()
    ModulePersister.saveModelDefinitionToFile(path, this, overWrite)
    this
  }

  /**
   * Save this module to path in torch7 readable format
   * @param path
   * @param overWrite
   * @return
   */
  final def saveTorch(path : String, overWrite: Boolean = false) : this.type = {
    this.clearState()
    File.saveTorch(this, path, TYPE_MODULE, overWrite)
    this
  }

  /**
   * Save this module to path in caffe readable format
   * @param prototxtPath
   * @param modelPath
   * @param useV2
   * @param overwrite
   * @return
   */
  final def saveCaffe(prototxtPath: String, modelPath: String,
    useV2 : Boolean = true, overwrite : Boolean = false) : this.type = {
    this.clearState()
    CaffePersister.persist[T](prototxtPath, modelPath, this, useV2, overwrite)
    this
  }

  /**
   * Save this module to path in tensorflow readable format
   * @param inputs
   * @param path
   * @param byteOrder
   * @param dataFormat
   * @return
   */
  final def saveTF(
    inputs : Seq[(String, Seq[Int])],
    path: String,
    byteOrder: ByteOrder = ByteOrder.LITTLE_ENDIAN,
    dataFormat: TensorflowDataFormat = TensorflowDataFormat.NHWC): this.type = {
    require(this.isInstanceOf[Graph[T]], "only Graph container can be saved as Tensorflow model")
    this.clearState()
    val inTrainMode = train
    if (inTrainMode) {
      this.evaluate()
    }
    TensorflowSaver.saveGraph(this.asInstanceOf[Graph[T]], inputs, path, byteOrder, dataFormat)
    if (inTrainMode) {
      this.training()
    }
    this
  }

  /**
   * Get numeric type of module parameters
   * @return
   */
  final def getNumericType(): TensorDataType = {
    ev.getType()
  }

  /**
   * module predict, return the probability distribution
   * @param dataset dataset for prediction
   * @param batchSize total batchSize for all partitions.
   *                  if -1, default is 4 * partitionNumber of datatset
   * @param shareBuffer whether to share same memory for each batch predict results
   */
  final def predict(dataset: RDD[Sample[T]],
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

  final def predictClass(dataset: RDD[Sample[T]], batchSize: Int = -1): RDD[Int] = {
    Predictor(this).predictClass(dataset, batchSize)
  }

  /**
   * model predict images, return imageFrame with predicted tensor,
   * if you want to call predictImage multiple times,
   * it is recommended to use Predictor for DistributedImageFrame
   * or LocalPredictor for LocalImageFrame
   * @param imageFrame imageFrame that contains images
   * @param outputLayer if outputLayer is not null, the output of layer that matches
   *                      outputLayer will be used as predicted output
   * @param shareBuffer whether to share same memory for each batch predict results
   * @param batchPerPartition batch size per partition, default is 4
   * @param predictKey key to store predicted result
   * @param featurePaddingParam featurePaddingParam if the inputs have variant size
   * @return
   */
  final def predictImage(imageFrame: ImageFrame,
    outputLayer: String = null,
    shareBuffer: Boolean = false,
    batchPerPartition: Int = 4,
    predictKey: String = ImageFeature.predict,
    featurePaddingParam: Option[PaddingParam[T]] = None): ImageFrame = {
    imageFrame match {
      case distributedImageFrame: DistributedImageFrame =>
        Predictor(this, featurePaddingParam, batchPerPartition)
          .predictImage(distributedImageFrame, outputLayer, shareBuffer, predictKey)
      case localImageFrame: LocalImageFrame =>
        val predictor = LocalPredictor(this, featurePaddingParam, batchPerPartition)
        val imageFrame = predictor.predictImage(localImageFrame, outputLayer, shareBuffer,
          predictKey)
        predictor.shutdown()
        imageFrame
    }
  }

  /**
   * Set weight and bias for the module
   * @param newWeights array of weights and bias
   * @return
   */
  final def setWeightsBias(newWeights: Array[Tensor[T]]): this.type = {
    require(parameters() != null, "this layer does not have weight/bias")
    require(parameters()._1.length == newWeights.length,
      "the number of input weight/bias is not consistant with " +
        "number of weight/bias of this layer, " +
        s"number of input ${parameters()._1.length}," +
        s" number of output ${newWeights.length}")
    val weights = parameters()._1
    for(i <- newWeights.indices) {
      // TODO: enable this checking as we don't respect shape right now.
      //      require(weights(i).size().deep == newWeights(i).size().deep,
      //        s"Mismatch shape, ${weights(i).size().mkString(",")}" +
      //          s" vs ${newWeights(i).size().mkString(",")} ")
      weights(i).copy(newWeights(i))
    }
    this
  }

  /**
   * Get weight and bias for the module
   * @return array of weights and bias
   *
   */
  final def getWeightsBias(): Array[Tensor[T]] = {
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
  final def saveWeights(path: String, overWrite: Boolean): Unit = {
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
  final def loadWeights(weightPath: String, matchAll: Boolean = true): this.type = {
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
  final def loadModelWeights(srcModel: Module[Float], matchAll: Boolean = true): this.type = {
    val srcParameter = srcModel.getParametersTable()
    val targetParameter = getParametersTable()
    copyWeights(targetParameter, srcParameter, matchAll)
    this
  }

  protected def processInputs(nodes: Seq[ModuleNode[T]]): ModuleNode[T] = {
    val curNode = new ModuleNode[T](this)
    nodes.foreach(node => {
      node.add(curNode, Edge())
    })
    curNode
  }

  protected def processInputs(first: (ModuleNode[T], Int),
      nodesWithIndex : (ModuleNode[T], Int)*): ModuleNode[T] = {
    val curNode = new ModuleNode[T](this)
    first._1.add(curNode, Edge(first._2))
    nodesWithIndex.foreach(nodeWithIndex => {
      nodeWithIndex._1.add(curNode, Edge(nodeWithIndex._2))
    })
    curNode
  }

  /**
   * Build graph: some other modules point to current module
   * @param nodes upstream module nodes
   * @return node containing current module
   */
  def inputs(nodes : ModuleNode[T]*): ModuleNode[T] = {
    validateInput(nodes.map(_.element))
    processInputs(nodes)
  }

  /**
   * Build graph: some other modules point to current module
   * @param nodes upstream module nodes in an array
   * @return node containing current module
   */
  def inputs(nodes : Array[ModuleNode[T]]): ModuleNode[T] = {
    validateInput(nodes.map(_.element))
    processInputs(nodes)
  }

  // debug
  def toGraphInputs(nodes : ModuleNode[T]*): ModuleNode[T] = {
    processInputs(nodes)
  }

  def toGraphInputs(nodes : Array[ModuleNode[T]]): ModuleNode[T] = {
    processInputs(nodes)
  }
  // debug

  /**
   * Build graph: some other modules point to current module
   * @param first distinguish from another inputs when input parameter list is empty
   * @param nodesWithIndex upstream module nodes and the output tensor index. The start index is 1.
   * @return node containing current module
   */
  def inputs(first: (ModuleNode[T], Int), nodesWithIndex : (ModuleNode[T], Int)*): ModuleNode[T] = {
    validateInput(List(first._1.element))
    validateInput(nodesWithIndex.map(_._1.element))
    processInputs(first, nodesWithIndex: _*)
  }

  /**
   * Generate graph module with start nodes
   * @param startNodes
   * @return
   */
  def toGraph(startNodes: ModuleNode[T]*): Graph[T] = {
    val starts = if (startNodes.isEmpty) Array(Input[T]()) else startNodes.toArray
    val endNodes = this.getEndNodes(starts)
    val graph = Graph(starts, endNodes)
    if (graph.isInstanceOf[StaticGraph[T]]) {
      graph.asInstanceOf[StaticGraph[T]].toSingleGraph()
    } else {
      graph
    }
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
   * use ValidationMethod to evaluate module on the given rdd dataset
   * @param dataset dataset for test
   * @param vMethods validation methods
   * @param batchSize total batchsize of all partitions,
   *                  optional param and default 4 * partitionNum of dataset
   * @return
   */
  final def evaluate(
    dataset: RDD[Sample[T]],
    vMethods: Array[_ <:ValidationMethod[T]],
    batchSize: Option[Int] = None
  ): Array[(ValidationResult, ValidationMethod[T])] = {
    Evaluator(this).test(dataset, vMethods.map(v => v), batchSize)
  }


  /**
   * use ValidationMethod to evaluate module on the given rdd dataset
   * @param dataset
   * @param vMethods
   * @return
   */
  final def evaluate(
    dataset: RDD[MiniBatch[T]],
    vMethods: Array[_ <:ValidationMethod[T]]
  ): Array[(ValidationResult, ValidationMethod[T])] = {
    Evaluator(this).testMiniBatch(dataset, vMethods.map(v => v))
  }

  /**
   * use ValidationMethod to evaluate module on the given ImageFrame
   *  @param imageFrame ImageFrame for valudation
   *  @param vMethods validation methods
   *  @param batchSize total batch size of all partitions
   *  @return
   */

  final def evaluateImage(imageFrame: ImageFrame,
    vMethods: Array[_ <:ValidationMethod[T]],
    batchSize: Option[Int] = None
    ): Array[(ValidationResult, ValidationMethod[T])] = {
    require(imageFrame.isDistributed(), "ImageFrame must be distributed")
    val rdd = imageFrame.toDistributed().rdd.map(imageFeature => {
      if (imageFeature.isValid) {
        require(imageFeature.contains(ImageFeature.sample), "ImageFeature must have sample")
        imageFeature[Sample[T]](ImageFeature.sample)
      } else {
        null
      }
    }).filter(_ != null)
    evaluate(rdd, vMethods, batchSize)
  }

  /**
   * use ValidationMethod to evaluate module on the given local dataset
   * @param dataSet
   * @param vMethods
   * @return
   */
  final def evaluate(
    dataSet: LocalDataSet[MiniBatch[T]],
    vMethods: Array[_ <:ValidationMethod[T]]
  ): Array[(ValidationResult, ValidationMethod[T])] = {
    Validator(this, dataSet).test(vMethods.map(v => v))
  }

  /**
   * Quantize this module, which reduces the precision of the parameter. Get a higher speed with a
   * little accuracy cost.
   * @return
   */
  final def quantize(): Module[T] = {
    ConversionUtils.convert[T](this, true)
  }

  // ================================= Internal APIs ===========================================

  private var namePostfix = Integer.toHexString(java.util.UUID.randomUUID().hashCode())

  final private[bigdl] def getNamePostfix : String = namePostfix

  final private[bigdl] def setNamePostfix(namePostfix : String) : Unit =
    this.namePostfix = namePostfix

  /**
   * The scale of gradient weight and gradient bias
   * before gradParameters being accumulated.
   */
  protected var scaleW: Double = 1.0
  protected var scaleB: Double = 1.0

  private[nn] final def allocateAs(dest: Activity): Activity = dest match {
    case tensor: Tensor[T] => Tensor[T]()
    case table: Table => T()
    case _ => throw new IllegalArgumentException("Activity only support tensor and table now")
  }

  /**
   * The name of the module
   */
  private var name : String = null

  private var id: Int = 0

  private[bigdl] def setId(id: Int): Unit = {
    this.id = id
  }

  private[bigdl] def getId(): Int = this.id

  protected final def getPrintName(): String = {
    val postfix = if (name == null) {
      namePostfix
    } else {
      name
    }
    s"${this.getClass.getSimpleName}[${postfix}]"

  }

  protected var forwardTime = 0L

  protected var backwardTime = 0L

  private var scaleWCache: Double = scaleW
  private var scaleBCache: Double = scaleB

  /**
   * This function returns two tensors. One for the flattened trainable parameters flatParameters
   * and another for the gradients of the energy wrt to the trainable parameters flatGradParameters.
   *
   * Custom modules should not override this function. They should instead override parameters(...)
   * which is, in turn, called by the present function.
   *
   * This function will go over all the weights and gradWeights and make them view into a single
   * tensor (one for weights and one for gradWeights).
   *
   * @return
   */
  final private[bigdl] def getParameters(): (Tensor[T], Tensor[T]) = {
    val (weightParameters, gradParameters) = this.parameters()

    // maybe null if not weights in this module.
    require(weightParameters != null && weightParameters.length > 0,
      s"model ${this.getName()} doesn't have any trainable parameters.")

    // If some gradParameters are not allocated storage, allocate it
    require(weightParameters.size == gradParameters.size,
      "weights and gradient number are not match")
    weightParameters.zip(gradParameters).foreach{ case(w, g) => g.resizeAs(w)}
    (Module.flatten[T](weightParameters), Module.flatten[T](gradParameters))
  }

  /**
   * Module status. It is useful for modules like dropout/batch normalization
   */
  protected var train: Boolean = true


  protected var line = "\n"


  private val engineType: EngineType = Engine.getEngineType()

  /**
   * get execution engine type
   */
  private[bigdl] def checkEngineType(): this.type = {
    if (engineType != Engine.getEngineType()) {
      throw new Error("Module's EngineType doesn't march global EngineType")
    }
    this
  }

  final private def setWeightAndBias(copy : AbstractModule[A, B, T], deepCopy : Boolean): Unit = {
    val parameterTable = this.getParametersTable
    val copiedModuleParamTable = copy.getParametersTable
    if (parameterTable != null) {
      require(copiedModuleParamTable != null, "cloned module should have params")
      parameterTable.foreach {
        case (name: String, params: Table) =>
          require(copiedModuleParamTable.get(name) != None, s"cloned module should have for $name")
          setLayerWeightAndBias(params,
            copiedModuleParamTable.get(name).get.asInstanceOf[Table], deepCopy)
      }
    }
  }

  final private def setLayerWeightAndBias(params : Table,
                                    copyParams : Table, deepCopy : Boolean): Unit = {
    params.foreach(param => {
      copyParam(params, copyParams, deepCopy, param._1.toString)
    })
  }

  final private def copyParam(params : Table, copyParams : Table,
                        deepCopy : Boolean, paraName : String) : Unit = {
    if (params.contains(paraName)) {
      // this is for quantization tensors where the weight might be an array
      if (params.get(paraName).get
        .isInstanceOf[Array[Tensor[T]]]) {
        val copies = copyParams.get(paraName).get
          .asInstanceOf[Array[Tensor[T]]]
        val origins = params.get(paraName).get
          .asInstanceOf[Array[Tensor[T]]]
        var i = 0
        while (i < copies.length) {
          copyTensor(origins(i), copies(i), deepCopy)
          i += 1
        }
      } else {
        // For normal layers, their params are just tensors
        copyTensor(params.get(paraName).get.asInstanceOf[Tensor[T]],
          copyParams.get(paraName).get.asInstanceOf[Tensor[T]], deepCopy)
      }
    }
  }

  final private def copyTensor(t1 : Tensor[T], t2 : Tensor[T], deepCopy : Boolean) = {
    if (t2.isInstanceOf[QuantizedTensor[_]]) {
      t2.asInstanceOf[QuantizedTensor[_]].release()
    }
    if (deepCopy) {
      t2.copy(t1)
    } else {
      t2.set(t1)
    }
  }

  final private def copyWeights(target: Table, src: Table, matchAll: Boolean): Unit = {
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

  private[bigdl] def canEqual(other: Any): Boolean = other.isInstanceOf[AbstractModule[A, B, T]]


  /**
   * Generate end nodes of current module with start nodes
   * @param startNodes: current start nodes
   * @return current end nodes
   */
  private[bigdl] def getEndNodes(startNodes: Array[ModuleNode[T]]): Array[ModuleNode[T]] = {
    // debug
    // val endNodes = Array(this.inputs(startNodes: _*))
    val endNodes = Array(this.toGraphInputs(startNodes: _*))
    // debug
    endNodes
  }

  /**
   * Return classTag numerics for module serialization. If your module contains multiple classtag
   * in the constructor, you should override this method
   * @return
   */
  private[bigdl] def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array(scala.reflect.classTag[T]), Array(ev))
  }

  /**
   * Check if some module is duplicated in the model. For a layer it cannot be duplicated.
   * Container should override this method
   */
  private[bigdl] def checkDuplicate(
    record: mutable.HashSet[Int] = mutable.HashSet()
  ): Unit = {
    val errMsg = "Some module is duplicate in the current model: "
    val curId = System.identityHashCode(this)
    require(this.skipDuplicateCheck() || !record.contains(curId), errMsg + this.getName())
    record.add(curId)
  }

  /**
   * Sometimes, some layer need skip the duplicate check process, e.g. Keras-like input layer
   * @return
   */
  private[nn] def skipDuplicateCheck(): Boolean = false

  /**
   * if the model contains native resources such as aligned memory, we should release it by manual.
   * JVM GC can't release them reliably.
   */
  def release(): Unit = {}


  /**
   * parameter synchronizer for gradient synchronization
   */
  private var _parameterSynchronizer: DistriParameterSynchronizer[T] = null

  /**
   * set parameter synchronizer
   * @param parameterSynchronizer parameter synchronizer
   */
  private[bigdl] def setParameterSynchronizer(parameterSynchronizer:
    DistriParameterSynchronizer[T]): Unit = {
    _parameterSynchronizer = parameterSynchronizer
  }


  /**
   * get parameter synchronizer
   * @return parameter synchronizer
   */
  private[bigdl] def getParameterSynchronizer():
    DistriParameterSynchronizer[T] = _parameterSynchronizer


  private var _optimMethod: OptimMethod[T] = null

  /**
   * set optim method
   */

  private[bigdl] def setOptimMethod(optimMethod: OptimMethod[T]): Unit = {
    _optimMethod = optimMethod
  }

  /**
   * get optim method for layer
   */

  private[bigdl] def getOptimMethod(): OptimMethod[T] = _optimMethod

  private[bigdl] def updateParameter(): Unit = {
    if (this.getParameterSynchronizer() != null && this.isTraining) {
      if (this.parameters() != null) {
        val before = System.nanoTime()
        val (weights, grads) = this.getParameterSynchronizer.get(this.getName)
        val syncEndTime = System.nanoTime()
        if (grads != null) {
          val optimMethod = this.getOptimMethod
          require(optimMethod != null, s"optim method for ${this.getName} cannot be null")
          optimMethod.optimize(_ => (ev.fromType(0.0f), grads),
            weights)
          this.zeroGradParameters
        }
      }
    }
  }
}

