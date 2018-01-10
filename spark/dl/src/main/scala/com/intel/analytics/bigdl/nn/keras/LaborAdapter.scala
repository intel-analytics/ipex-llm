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

import java.nio.ByteOrder

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch, PaddingParam, Sample}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.{ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Tensor, TensorDataType}
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFrame}
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.utils.tf.TensorflowDataFormat
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

abstract class LaborAdapter[A <: Activity: ClassTag, B <: Activity: ClassTag, T: ClassTag]
(implicit ev: TensorNumeric[T]) extends AbstractModule[A, B, T]{

  protected  var labor: AbstractModule[A, B, T] = null

  override def getBatchInputShape(): Activity = labor.getBatchInputShape()

  override def getBatchOutputShape(): Activity = labor.getBatchOutputShape()

  override def build(inputShape: Activity): Unit = {
    labor = doBuild(inputShape)

    // TODO: redirect other variables as well
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
    labor.setScaleW(w).asInstanceOf[this.type]
  }

  /**
   * Set the scale of gradientBias
   *
   * @param b the value of the scale of gradientBias
   * @return this
   */
  override def setScaleB(b: Double): this.type = {
    labor.setScaleB(b).asInstanceOf[this.type]
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
  override def freeze(names: String*): this.type = labor.freeze(names : _*).asInstanceOf[this.type]

  /**
   * "unfreeze" module, i.e. make the module parameters(weight/bias, if exists)
   * to be trained(updated) in training process
   * if names is not empty, unfreeze layers that match given names
   *
   * @param names array of module names to unFreeze
   */
  override def unFreeze(names: String*): this.type = {
    labor.unFreeze(names : _*).asInstanceOf[this.type]
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
  override def setExtraParameter(extraParam: Array[Tensor[T]]): this.type =
  labor.setExtraParameter(extraParam).asInstanceOf[this.type]

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

  override def training(): this.type = labor.training().asInstanceOf[this.type]

  override def evaluate(): this.type = labor.evaluate().asInstanceOf[this.type]

  override def isTraining(): Boolean = labor.isTraining()

  override def reset(): Unit = labor.reset()

//  override def setLine(line: String): this.type = labor.setLine(line).asInstanceOf[this.type]

  /**
   * get execution engine type
   */
  override def checkEngineType(): this.type = labor.checkEngineType().asInstanceOf[this.type]

  // TODO: clone itself??
  override def cloneModule(): this.type = labor.cloneModule().asInstanceOf[this.type]

  /**
   * Save this module to path.
   * @param path path to save module, local file system, HDFS and Amazon S3 is supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx"
   *             Amazon S3 path should be like "s3a://bucket/xxx"
   * @param overWrite if overwrite
   * @return self
   */
  @deprecated("please use recommended saveModule(path, overWrite)")
  override def save(path : String, overWrite: Boolean = false) : this.type = {
    labor.save(path, overWrite).asInstanceOf[this.type]
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
  override def saveModule(path : String, weightPath : String = null,
                 overWrite: Boolean = false) : this.type = {
    labor.saveModule(path, weightPath, overWrite).asInstanceOf[this.type]
  }

  /**
   * Save this module definition to path.
   * @param path path to save module, local file system, HDFS and Amazon S3 is supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx"
   *             Amazon S3 path should be like "s3a://bucket/xxx"
   * @param overWrite if overwrite
   * @return self
   */
  override def saveDefinition(path : String, overWrite: Boolean = false)
  : this.type = {
    labor.saveDefinition(path, overWrite).asInstanceOf[this.type]
  }

  override def saveTorch(path : String, overWrite: Boolean = false) : this.type = {
    labor.saveTorch(path, overWrite).asInstanceOf[this.type]
  }

  override def saveCaffe(prototxtPath: String, modelPath: String,
                useV2 : Boolean = true, overwrite : Boolean = false) : this.type = {
    labor.saveCaffe(prototxtPath, modelPath, useV2, overwrite).asInstanceOf[this.type]
  }

  override def saveTF(
              inputs : Seq[(String, Seq[Int])],
              path: String,
              byteOrder: ByteOrder = ByteOrder.LITTLE_ENDIAN,
              dataFormat: TensorflowDataFormat = TensorflowDataFormat.NHWC)
  : this.type = {
    labor.saveTF(inputs, path, byteOrder, dataFormat).asInstanceOf[this.type]
  }

  /**
   * module predict, return the probability distribution
   * @param dataset dataset for prediction
   * @param batchSize total batchSize for all partitions.
   *                  if -1, default is 4 * partitionNumber of datatset
   * @param shareBuffer whether to share same memory for each batch predict results
   */
  override def predict(dataset: RDD[Sample[T]],
              batchSize: Int = -1,
              shareBuffer: Boolean = false): RDD[Activity] = {
    labor.predict(dataset, batchSize, shareBuffer)
  }
  /**
   * module predict, return the predict label
   * @param dataset dataset for prediction
   * @param batchSize total batchSize for all partitions.
   *                  if -1, default is 4 * partitionNumber of dataset
   */
  override def predictClass(dataset: RDD[Sample[T]], batchSize: Int = -1): RDD[Int] = {
    labor.predictClass(dataset, batchSize)
  }

  /**
   * model predict images, return imageFrame with predicted tensor
   * @param imageFrame imageFrame that contains images
   * @param outputLayer if outputLayer is not null, the output of layer that matches
   *                      outputLayer will be used as predicted output
   * @param shareBuffer whether to share same memory for each batch predict results
   * @param batchPerPartition batch size per partition, default is 4
   * @param predictKey key to store predicted result
   * @return
   */
  override   def predictImage(imageFrame: ImageFrame,
                              outputLayer: String = null,
                              shareBuffer: Boolean = false,
                              batchPerPartition: Int = 4,
                              predictKey: String = ImageFeature.predict,
                              featurePaddingParam: Option[PaddingParam[T]] = None): ImageFrame = {
    labor.predictImage(imageFrame, outputLayer, shareBuffer, batchPerPartition, predictKey)
  }

  /**
   * Set weight and bias for the module
   * @param newWeights array of weights and bias
   * @return
   */
  override def setWeightsBias(newWeights: Array[Tensor[T]]): this.type =
   labor.setWeightsBias(newWeights).asInstanceOf[this.type]

  /**
   * Get weight and bias for the module
   * @return array of weights and bias
   *
   */
  override def getWeightsBias(): Array[Tensor[T]] = labor.getWeightsBias()

  /**
   * save weights and bias to file
   * @param path file to save
   * @param overWrite whether to overwrite or not
   */
  override def saveWeights(path: String, overWrite: Boolean): Unit = {
    labor.saveWeights(path, overWrite)
  }
  /**
   * load pretrained weights and bias to current module
   * @param weightPath file to store weights and bias
   * @param matchAll whether to match all layers' weights and bias,
   *                 if not, only load existing pretrained weights and bias
   * @return current module
   */
  override def loadWeights(weightPath: String, matchAll: Boolean = true)
  : this.type = {
    labor.loadWeights(weightPath, matchAll).asInstanceOf[this.type]
  }

  /**
   * copy weights from another model, mapping by layer name
   * @param srcModel model to copy from
   * @param matchAll whether to match all layers' weights and bias,
   * @return current module
   */
  override def loadModelWeights(srcModel: Module[Float], matchAll: Boolean = true)
  : this.type =
    labor.loadModelWeights(srcModel = srcModel, matchAll = matchAll).asInstanceOf[this.type]


  /**
   * Find a module with given name. If there is no module with given name, it will return None. If
   * there are multiple modules with the given name, an exception will be thrown.
   * @param name
   * @return
   */
  override def apply(name : String): Option[AbstractModule[Activity, Activity, T]] = {
    labor.apply(name)
  }

  /**
   * use ValidationMethod to evaluate module
   * @param dataset dataset for test
   * @param vMethods validation methods
   * @param batchSize total batchsize of all partitions,
   *                  optional param and default 4 * partitionNum of dataset
   * @return
   */
  override def evaluate(dataset: RDD[Sample[T]],
               vMethods: Array[ValidationMethod[T]],
               batchSize: Option[Int] = None): Array[(ValidationResult, ValidationMethod[T])] =
  labor.evaluate(dataset, vMethods, batchSize)

  override def evaluate(dataSet: LocalDataSet[MiniBatch[T]],
               vMethods: Array[ValidationMethod[T]]
              ): Array[(ValidationResult, ValidationMethod[T])] = labor.evaluate(dataSet, vMethods)

  override def quantize(): Module[T] = labor.quantize()

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) =
    labor.getClassTagNumerics()

  override def getNumericType(): TensorDataType = labor.getNumericType()
  
}
