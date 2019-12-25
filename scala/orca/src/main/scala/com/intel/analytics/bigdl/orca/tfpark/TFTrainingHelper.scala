/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.tfpark

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.Utils
import com.intel.analytics.zoo.core.TFNetNative
import org.slf4j.LoggerFactory
import org.tensorflow.DataType
import org.tensorflow.framework.GraphDef

import scala.reflect.io.Path

// variables and gradVariables need to be sorted by name if you want to use multiple
// optimization methods for a TensorFlow model according to variable names.
private[zoo] class TFTrainingHelper private (graphRunner: GraphRunner,
                                    checkpointPath: String,
                                    inputs: Array[String],
                                    inputTypes: Array[Int],
                                    outputs: Array[String],
                                    variables: Array[String],
                                    variableTypes: Array[Int],
                                    variableAssignPlaceholders: Array[String],
                                    assignVariableOp: String,
                                    extraVariables: Array[String],
                                    extraVariableTypes: Array[Int],
                                    extraVariableAssignPlaceholders: Array[String],
                                    assignExtraVariableOP: String,
                                    gradVariables: Array[String],
                                    updateOp: String,
                                    defaultTensorValue: Array[Array[Float]])
  extends AbstractModule[Activity, Activity, Float] {

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (weights, gradWeights)
  }

  override def getExtraParameter(): Array[Tensor[Float]] = {
    extraParameters
  }

  private val extraParameters: Array[Tensor[Float]] = initVariables(extraVariables)

  private val weights = initVariables(variables)

  private val weightsMap = {
    val map = collection.mutable.Map[String, Tensor[Float]]()
    var i = 0
    while (i < variables.length) {
      map(variables(i)) = weights(i)
      i += 1
    }
    map
  }

  private def initVariables(variableNames: Array[String]): Array[Tensor[Float]] = {
    val ws = new Array[Tensor[Float]](variableNames.length)
    var i = 0
    while (i < ws.length) {
      ws(i) = Tensor[Float]()
      i += 1
    }
    ws
  }

  private val gradWeights = variables.map(_ => Tensor[Float]())

  private val graphOutputs = {
    val graphOuts = Vector.newBuilder[Tensor[Float]]

    var i = 0
    while (i < outputs.length) {
      graphOuts += Tensor[Float]()
      i += 1
    }

    i = 0
    while (i < gradVariables.length) {
      graphOuts += Tensor[Float]()
      i += 1
    }

    graphOuts.result()
  }

  private val gradWeightsBuffer =
    graphOutputs.slice(outputs.length, graphOutputs.length)

  output = {
    if (outputs.length == 1) {
      graphOutputs(0)
    } else {
      val out = T()
      var i = 0
      while (i < outputs.length) {
        out.insert(graphOutputs(i))
        i += 1
      }
      out
    }
  }

  override def evaluate(): TFTrainingHelper.this.type = {
    super.evaluate()
    setVariableIntoTF(weights, variableAssignPlaceholders,
      variableTypes.map(TFUtils.tfenum2datatype), assignVariableOp)
    this
  }


  private def getVariableFromTF(weights: Array[Tensor[Float]],
                                     variableNames: Array[String]) = {
    graphRunner.run(
      input = Vector.empty,
      inputTypes = Vector.empty,
      output = weights.toVector,
      inputNames = Vector.empty,
      outputNames = variableNames.toVector,
      targets = Vector.empty)
  }

  private def setVariableIntoTF(weights: Array[Tensor[Float]],
                                         inputNames: Array[String],
                                         variableTypes: Array[DataType],
                                         assignOp: String) = {
    graphRunner.run(
      input = weights.toVector,
      inputTypes = variableTypes.toVector,
      output = Vector.empty,
      inputNames = inputNames.toVector,
      outputNames = Vector.empty,
      targets = Vector(assignOp)
    )
  }

  def saveCheckpoint(): Unit = {
    setVariableIntoTF(weights, variableAssignPlaceholders,
      variableTypes.map(TFUtils.tfenum2datatype), assignVariableOp)
    setVariableIntoTF(extraParameters, extraVariableAssignPlaceholders,
      extraVariableTypes.map(TFUtils.tfenum2datatype), assignExtraVariableOP)
    graphRunner.saveToFile(checkpointPath)
  }

  @transient
  private var extraParameterRestored: Boolean = false

  def restoreFromCheckpoint(): Unit = {
    graphRunner.restoreFromFile(checkpointPath)
    if (weights.length > 0) {
      getVariableFromTF(weights, variableNames = variables)
    }

    if (extraParameters.length > 0) {
      getVariableFromTF(extraParameters, variableNames = extraVariables)
    }

    extraParameterRestored = true

  }

  override def apply(name: String): Option[AbstractModule[Activity, Activity, Float]] = {
    val targetVariables = if (name == getName()) variables else variables.filter(_.startsWith(name))
    if (targetVariables == null) {
      None
    }
    else {
      val targetWeights = targetVariables.map(weightsMap)
      Some(new TFSubGraph(targetWeights))
    }
  }

  override def updateOutput(input: Activity): Activity = {
    Utils.timeIt("updateOutput") {
      if (this.isTraining()) {
        Utils.timeIt("setTrainingVariableIntoTF") {
          setVariableIntoTF(weights, variableAssignPlaceholders,
            variableTypes.map(TFUtils.tfenum2datatype), assignVariableOp)
        }
      }

      if (!extraParameterRestored) {
        setVariableIntoTF(extraParameters, extraVariableAssignPlaceholders,
          extraVariableTypes.map(TFUtils.tfenum2datatype), assignExtraVariableOP)
        extraParameterRestored = true
      }

      val feeds = Utils.activity2VectorBuilder(input)

      if (this.isTraining()) {
        var i = 0
        while (i < defaultTensorValue.length) {
          feeds += Tensor.scalar[Float](defaultTensorValue(i)(0))
          i += 1
        }
      } else {
        var i = 0
        while (i < defaultTensorValue.length) {
          feeds += Tensor.scalar[Float](defaultTensorValue(i)(1))
          i += 1
        }
      }

      val types = inputTypes.toVector.map(TFUtils.tfenum2datatype)

      val (outputNames, outputTensors) = if (isTraining()) {
        (outputs.toVector ++ gradVariables.toVector, graphOutputs)
      } else {
        (outputs.toVector, graphOutputs.slice(0, outputs.length))
      }

      graphRunner.run(
        input = feeds.result(),
        inputTypes = types,
        output = outputTensors,
        inputNames = inputs.toVector,
        outputNames = outputNames,
        targets = Vector(updateOp))


      if (extraParameters.length > 0) {
        Utils.timeIt("getExtraVariableFromTF") {
          getVariableFromTF(extraParameters, variableNames = extraVariables)
        }
      }

      if (isTraining()) {
        gradWeights.zipWithIndex.foreach { case (grad, idx) =>
          grad.resizeAs(weights(idx)).add(gradWeightsBuffer(idx))
        }
      }
    }

    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput
  }
}

object TFTrainingHelper {

  assert(TFNetNative.isLoaded)

  val logger = LoggerFactory.getLogger(getClass)

  def apply(graphDef: GraphDef,
            checkpointPath: String,
            trainMeta: TrainMeta,
            config: Array[Byte]): TFTrainingHelper = {

    val graphRunner = new GraphRunner(
      graphDef.toByteArray,
      trainMeta.restoreOp,
      trainMeta.restorePathPlaceholder,
      trainMeta.saveOp,
      trainMeta.savePathPlaceholder,
      config)

    val helper = new TFTrainingHelper(graphRunner,
      checkpointPath,
      trainMeta.inputs,
      trainMeta.inputTypes,
      trainMeta.metricTensors ++ Array(trainMeta.batchSizeTensor, trainMeta.lossTensor),
      trainMeta.variables,
      trainMeta.variableTypes,
      trainMeta.variableAssignPlaceholders,
      trainMeta.assignVariableOp,
      trainMeta.extraVariables,
      trainMeta.extraVariableTypes,
      trainMeta.extraVariableAssignPlaceholders,
      trainMeta.assignExtraVariableOp,
      trainMeta.gradVariables,
      trainMeta.updateOp,
      trainMeta.defaultTensorValue
    )
    helper.restoreFromCheckpoint()
    helper
  }

  def apply(modelPath: String, sessionConfig: Array[Byte] = null): TFTrainingHelper = {

    val folderPath = Path(modelPath)
    val trainingMetaPath = folderPath / Path("training_meta.json")
    val graphDefPath = folderPath / Path("model.meta")
    val checkpointPath = folderPath / Path("model")

    val trainingMeta = TFUtils.getTrainMeta(trainingMetaPath)
    val graphDef = TFUtils.parseGraph(graphDefPath.toString())
    val config = Option(sessionConfig).getOrElse(TFUtils.defaultSessionConfig.toByteArray())

    val helper = TFTrainingHelper(graphDef, checkpointPath.toString(), trainingMeta, config)
    helper
  }
}
