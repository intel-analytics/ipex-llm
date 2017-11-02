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

package com.intel.analytics.bigdl.utils.keras

import com.google.protobuf.GeneratedMessage
import com.intel.analytics.bigdl.dataset.DataSet._
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat, Initializable}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.log4j.Logger
import play.api.libs.json.{JsArray, JsNull, JsPath, JsValue}

import scala.collection.mutable
import scala.reflect.ClassTag
import reflect.runtime.universe.MethodMirror
import scala.reflect.runtime.universe.newTermName
import scala.reflect.runtime.universe.runtimeMirror

object LayerConverter {
  def getPadding(borderMode: String): (Int, Int) = {
    val (bpadW, bpadH) = borderMode match {
      case "same" => (-1, -1)
      case "valid" => (0, 0)
    }
    (bpadW, bpadH)
  }
}

// TODO: We should be able to find unittest on python side.
abstract class LayerConverter[T: ClassTag](kerasJson: KModel)(implicit ev: TensorNumeric[T]) {
  protected val logger = Logger.getLogger(getClass)

//  private var kerasToBigDLCreator = new mutable.HashMap[String, (Layer) => AbstractModule[Activity, Activity, T]]()

  private var kerasToBigDLCreator = new mutable.HashMap[String, MethodMirror]()

  private val nodeIdToKerasLayer = new mutable.HashMap[String, Layer]()
  private val nodeIdToNodeInstance = new mutable.HashMap[String, ModuleNode[T]]()

  init()
  // refer: https://stackoverflow.com/questions/11062166/dynamic-method-invocation-with-new-scala-reflection-api
  private def reflectFunctionByName(name: String) = {
    val r1 = runtimeMirror(getClass.getClassLoader).reflect(this)
    val func = r1.symbol.typeSignature.member(newTermName(name))
    r1.reflectMethod(func.asMethod)
  }

  def init(): Unit = {
    val methods = this.getClass.getMethods()
    val method_names = methods.filter(_.getName().startsWith("create")).map(_.getName)
      val tmp = method_names.map(name => (name, this.reflectFunctionByName(name)))
      .foreach{case (name, func) =>
        kerasToBigDLCreator(name.substring(6)) = func // remove "create" from name
      }

    // Create name to keras layer mapping
    kerasJson.config.layers.foreach {layer =>
      if (nodeIdToKerasLayer.contains(layer.name)) {
        throw new RuntimeException(s"Duplicate node id: ${layer.name}")
      }
      nodeIdToKerasLayer(layer.name) = layer
    }
  }

  private def convertInOrOutForModel(boundNodes: Seq[JsArray]): Array[ModuleNode[T]] = {
      boundNodes.map { node =>
        val nodeName = node.value(0).toString().replaceAll("^\"|\"$", "")
        // TODO: parse nodeID and tensorID
        this.nodeIdToNodeInstance(nodeName)
      }.toArray
  }

  def createGraph(kerasJson: KModel): Graph[T] = {
    // ensure each node instances is created
    kerasJson.config.layers.foreach { layer =>
      if (!this.nodeIdToNodeInstance.contains(layer.name)) {
        doCreateNode(layer)
      }
    }

    val input = convertInOrOutForModel(kerasJson.config.inputLayers)
    val output = convertInOrOutForModel(kerasJson.config.outputLayers)
    Graph[T](input = input, output = output)
  }

  def doCreateNode(layer: Layer): ModuleNode[T] = {
     if (layer.className == "InputLayer") {
       val input = Input[T]() // input cannot set name
       this.nodeIdToNodeInstance(layer.name) = input
       return input
     }
    val inNodes = layer.inboundNodes.map { node =>
      val nodeName = node(0)(0).get.toString().replaceAll("^\"|\"$", "")
      // TODO why always o here?, "Dense" remove ""
      // todo: parse nodeindex or tensorindex
      if (!this.nodeIdToNodeInstance.contains(nodeName)) {
        val node = doCreateNode(this.nodeIdToKerasLayer(nodeName))
        logger.info(s"Creating: $nodeName")
      }
      this.nodeIdToNodeInstance(nodeName)
    }
    val bigDLLayer = this.kerasToBigDLCreator(
      layer.className)(layer).asInstanceOf[AbstractModule[Activity, Activity, T]]
    val newNode = bigDLLayer.inputs(inNodes : _*)
    this.nodeIdToNodeInstance(layer.name) = newNode
    newNode
    null
  }

  def createInput(layer: Layer): AbstractModule[Activity, Activity, T]

  def createConvolution2D(layer: Layer): AbstractModule[Activity, Activity, T]

  def createEmbedding(layer: Layer): AbstractModule[Activity, Activity, T]

  def createFlatten(layer: Layer): AbstractModule[Activity, Activity, T]

  def createMerge(layer: Layer): AbstractModule[Activity, Activity, T]

  def createDense(layer: Layer): AbstractModule[Activity, Activity, T]

  def createDropout(layer: Layer): AbstractModule[Activity, Activity, T]

  def createActivation(layer: Layer): AbstractModule[Activity, Activity, T]

}


object ParameterLayerHelper {
  def combo[T: ClassTag](blayer: AbstractModule[Activity, Activity, T],
                 layerConfig: ParametersLayerConfig)(implicit ev: TensorNumeric[T])
  : AbstractModule[Activity, Activity, T] = {
    blayer.setName(layerConfig.name) // assuming layer.name is always equals to layerConfig.name
    val initMethod = InitMethodHelper.toBigDL(layerConfig.initMethod)
    if (blayer.isInstanceOf[Initializable]) {
      throw new RuntimeException(s"Not an initializable for: ${blayer}")
    }
    blayer.asInstanceOf[Initializable].setInitMethod(initMethod, Zeros)
    // Keras always set this to be Zero.

    ActivationHelper.fuse[T](blayer,
      layerConfig.activation,
      s"${layerConfig.name}_${layerConfig.activation}")
  }
}

object InitMethodHelper {
  def toBigDL[T](initName: String): InitializationMethod = {
    initName match {
      case "glorot_uniform" => RandomUniform
      // TODO: this is a correct mapping? case object cannot use isinstance of
      case "one" => Ones
      case i: String => throw new RuntimeException(s"not supported yet $i")
    }
  }
}

object RegularizerHelper {
  def toBigDL[T](reg: JsValue): Regularizer[T] = {
    reg match {
      case JsNull => null  // case object cannot use isinstance of
      case _ => throw new RuntimeException("not supported yet")
    }
  }
}

object OrderingHelper {
  def toBigDL[T](ordering: String): DataFormat = {
    ordering match {
      case "tf" => DataFormat.NHWC
      case "th" => DataFormat.NCHW
    }
  }
}



object ActivationHelper {
  def toBigDL[T: ClassTag](activationName: String,
                           layerName: String)
                          (implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
    val layer = activationName match {
      case "tanh" => Tanh[T]()
      case "sigmoid" => Sigmoid[T]()
      case "relu" => ReLU[T]()
      case "softmax" => SoftMax[T]()
      case _ => throw new IllegalArgumentException(
        s"unsupported type: ${activationName}")
    }
    layer.setName(layerName).asInstanceOf[AbstractModule[Activity, Activity, T]]
  }

  def fuse[T: ClassTag](srcLayer: AbstractModule[Activity, Activity, T],
                        activationName: String,
                        name: String)
                       (implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
    // "linear" meaning do nothing
    if (activationName != "linear") {
      val seq = Sequential[T]()
      seq.add(srcLayer)
      seq.add(toBigDL(activationName, name))
      seq.setName(srcLayer.getName())
    } else {
      srcLayer
    }
  }
}
