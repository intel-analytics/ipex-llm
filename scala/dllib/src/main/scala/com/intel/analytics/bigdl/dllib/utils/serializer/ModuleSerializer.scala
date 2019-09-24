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
package com.intel.analytics.bigdl.utils.serializer

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.maskrcnn.MaskRCNN
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.nn.keras.{KerasLayer, KerasLayerSerializer, Model, Sequential => KSequential}
import com.intel.analytics.bigdl.nn.ops.{RandomUniform => RandomUniformOps}
import com.intel.analytics.bigdl.nn.tf.{DecodeRawSerializer, ParseExample, ParseSingleExample, StridedSlice}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.ReflectionUtils
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter

import scala.collection.mutable
import scala.language.existentials
import scala.reflect.ClassTag
import scala.reflect.runtime.universe

object ModuleSerializer extends ModuleSerializable{

  private val runtimeMirror = universe.runtimeMirror(getClass.getClassLoader)

  private val serializerMaps = new mutable.HashMap[String, ModuleSerializable]

  // group serializer for one serializer to handle multiple layers of the same super type

  // super type class to serializer

  private val groupSerializerMaps = new mutable.HashMap[String, ModuleSerializable]()

  private[serializer] val _lock = new Object

  // generic type definition for type matching

  var tensorNumericType : universe.Type = null
  var tensorType : universe.Type = null
  var regularizerType : universe.Type = null
  var abstractModuleType : universe.Type = null
  var tensorModuleType : universe.Type = null
  var moduleType : universe.Type = null
  var boundedModuleType : universe.Type = null
  var tType : universe.Type = null

  init

  /**
   * Serialization entry for all modules based on corresponding class instance of module
   * @param serializerContext : serialization context
   * @return protobuf format module instance
   */
  def serialize[T: ClassTag](serializerContext : SerializeContext[T])
                            (implicit ev: TensorNumeric[T])
    : SerializeResult = {
    val module = serializerContext.moduleData.module
    // For those layers which have their own serialization/deserialization methods
    val clsName = module.getClass.getName
    val (serializer, serContext) = if (serializerMaps.contains(clsName)) {
      (serializerMaps(clsName), serializerContext)
    } else {
      // if no layer specific implementation, check if serializer of the same type exists
      val (groupSerializer, group) = findGroupSerializer(serializerContext.moduleData.module)
      if (groupSerializer != null) {
        val context = SerializeContext[T](serializerContext.moduleData,
        serializerContext.storages, serializerContext.storageType,
          serializerContext.copyWeightAndBias, group)
        (groupSerializer, context)
      } else {
      val m = module.asInstanceOf[AbstractModule[_, _, _]]
      m match {
        case kerasLayer: KerasLayer[_, _, _] =>
          (KerasLayerSerializer, serializerContext)
        case container: Container[_, _, _] =>
          (ContainerSerializer, serializerContext)
        case cell: Cell[_] =>
          (CellSerializer, serializerContext)
        case _ => (ModuleSerializer, serializerContext)
        }
      }
    }
    serializer.setCopyWeightAndBias(serContext.copyWeightAndBias).
      serializeModule(serContext)
  }

  private def findGroupSerializer[T: ClassTag](module : Module[T])
    (implicit ev: TensorNumeric[T]): (ModuleSerializable, String) = {
    var cls : Class[_] = module.getClass.getSuperclass
    var clsName = cls.getName
    while (clsName != "java.lang.Object") {
      if (groupSerializerMaps.contains(clsName)) {
        return (groupSerializerMaps(clsName), clsName)
      }
      cls = cls.getSuperclass
      clsName = cls.getName
    }
    (null, null)
  }

  /**
   *  Deserialization entry for all modules based on corresponding module type
   *  @param context : context for deserialization
   *  @return BigDL module
   */
  def load[T: ClassTag](context: DeserializeContext)
                       (implicit ev: TensorNumeric[T]) : ModuleData[T] = {
    try {
      val model = context.bigdlModule
      val deSerializer = if (serializerMaps.contains(model.getModuleType)) {
        serializerMaps(model.getModuleType)
      } else {
        val attrMap = model.getAttrMap
        if (attrMap.containsKey(SerConst.GROUP_TYPE)) {
          val groupTypeAttr = attrMap.get(SerConst.GROUP_TYPE)
          val groupType = DataConverter.getAttributeValue(context, groupTypeAttr).
            asInstanceOf[String]
          require(groupSerializerMaps.contains(groupType), s" Group serializer does" +
            s" not exist for $groupType")
          groupSerializerMaps(groupType)
        } else {
          val subModuleCount = model.getSubModulesCount
          if (subModuleCount > 0) {
            ContainerSerializer
          } else {
            if (attrMap.containsKey("is_cell_module")) {
              CellSerializer
            } else if (attrMap.containsKey("is_keras_module")) {
              KerasLayerSerializer
            } else {
              ModuleSerializer
            }
          }
        }
      }
      deSerializer.setCopyWeightAndBias(context.copyWeightAndBias).
        loadModule(context)
    } catch {
      case e: Exception =>
        throw new RuntimeException(
          s"Loading module ${context.bigdlModule.getModuleType} exception :", e)
    }
  }

  /**
   * register module for single module, used for standard BigDL module and user defined module
   * @param moduleType,must be unique
   * @param serializer serialzable implementation for this module
   */
  def registerModule(moduleType : String, serializer : ModuleSerializable) : Unit = {
    require(!serializerMaps.contains(moduleType), s"$moduleType already registered!")
    require(!groupSerializerMaps.contains(moduleType), s"$moduleType already " +
      s"registered with group serializer!")
    serializerMaps(moduleType) = serializer
  }

  /**
   * register module for modules of the same type, used for
   * standard BigDL module and user defined module
   * @param superModuleType,must be unique
   * @param groupSerializer serialzable implementation for this module
   */
  def registerGroupModules(superModuleType : String, groupSerializer :
    ModuleSerializable) : Unit = {
    require(!serializerMaps.contains(superModuleType), s"$moduleType already " +
      s"registered with single serializer!")
    require(!groupSerializerMaps.contains(superModuleType), s"$moduleType already " +
      s"registered with group serializer!")
    groupSerializerMaps(superModuleType) = groupSerializer
  }



  private def init() : Unit = {
    initializeDeclaredTypes
    registerModules
  }

  private def initializeDeclaredTypes() : Unit = {

    var wrapperCls = Class.forName("com.intel.analytics.bigdl.utils.serializer.GenericTypeWrapper")
    val fullParams = ReflectionUtils.getPrimCtorMirror(wrapperCls).symbol.paramss
    fullParams.foreach(map => {
      map.foreach(param => {
        val name = param.name.decodedName.toString
        val ptype = param.typeSignature
        if (name == "tensor") {
          tensorType = ptype
        } else if (name == "regularizer") {
          regularizerType = ptype
        } else if (name == "abstractModule") {
          abstractModuleType = ptype
        } else if (name == "tensorModule") {
          tensorModuleType = ptype
        } else if (name == "module") {
          moduleType = ptype
        } else if (name == "boundedModule") {
          boundedModuleType = ptype
        } else if (name == "ev") {
          tensorNumericType = ptype
        } else if (name == "ttpe") {
          tType = ptype
        }
      })
    })
  }
  // Add those layers that need to overwrite serialization method

  private def registerModules : Unit = {

    registerModule("com.intel.analytics.bigdl.nn.BatchNormalization", BatchNormalization)
    registerModule("com.intel.analytics.bigdl.nn.SpatialBatchNormalization", BatchNormalization)
    registerModule("com.intel.analytics.bigdl.nn.BinaryTreeLSTM", BinaryTreeLSTM)
    registerModule("com.intel.analytics.bigdl.nn.BiRecurrent", BiRecurrent)
    registerModule("com.intel.analytics.bigdl.nn.CAddTable", CAddTable)
    registerModule("com.intel.analytics.bigdl.nn.StaticGraph", Graph)
    registerModule("com.intel.analytics.bigdl.nn.DynamicGraph", Graph)
    registerModule("com.intel.analytics.bigdl.nn.keras.Model", Model)
    registerModule("com.intel.analytics.bigdl.nn.keras.Sequential", KSequential)
    registerModule("com.intel.analytics.bigdl.nn.keras.KerasLayerWrapper", KerasLayerSerializer)
    registerModule("com.intel.analytics.bigdl.nn.MapTable", MapTable)
    registerModule("com.intel.analytics.bigdl.nn.Maxout", Maxout)
    registerModule("com.intel.analytics.bigdl.nn.MaskedSelect", MaskedSelect)
    registerModule("com.intel.analytics.bigdl.nn.Recurrent", Recurrent)
    registerModule("com.intel.analytics.bigdl.nn.RecurrentDecoder", RecurrentDecoder)
    registerModule("com.intel.analytics.bigdl.nn.Reshape", Reshape)
    registerModule("com.intel.analytics.bigdl.nn.Scale", Scale)
    registerModule("com.intel.analytics.bigdl.nn.SpatialContrastiveNormalization",
      SpatialContrastiveNormalization)
    registerModule("com.intel.analytics.bigdl.nn.SpatialDivisiveNormalization",
      SpatialDivisiveNormalization)
    registerModule("com.intel.analytics.bigdl.nn.SpatialFullConvolution",
      SpatialFullConvolution)
    registerModule("com.intel.analytics.bigdl.nn.SpatialMaxPooling",
      SpatialMaxPooling)
    registerModule("com.intel.analytics.bigdl.nn.SpatialSubtractiveNormalization",
      SpatialSubtractiveNormalization)
    registerModule("com.intel.analytics.bigdl.nn.Transpose", Transpose)
    registerModule("com.intel.analytics.bigdl.nn.TimeDistributed", TimeDistributed)
    registerModule("com.intel.analytics.bigdl.nn.VolumetricMaxPooling", VolumetricMaxPooling)
    registerModule("com.intel.analytics.bigdl.nn.Echo", Echo)
    registerModule("com.intel.analytics.bigdl.nn.quantized.SpatialConvolution",
      quantized.SpatialConvolution)
    registerModule("com.intel.analytics.bigdl.nn.quantized.SpatialDilatedConvolution",
      quantized.SpatialDilatedConvolution)
    registerModule("com.intel.analytics.bigdl.nn.quantized.Linear",
      quantized.Linear)
    registerModule("com.intel.analytics.bigdl.nn.tf.ParseExample", ParseExample)
    registerModule("com.intel.analytics.bigdl.nn.tf.ParseSingleExample", ParseSingleExample)
    registerModule("com.intel.analytics.bigdl.nn.SReLU", SReLU)
    registerModule("com.intel.analytics.bigdl.nn.tf.DecodeRaw", DecodeRawSerializer)
    registerModule("com.intel.analytics.bigdl.nn.ops.RandomUniform", RandomUniformOps)
    registerModule("com.intel.analytics.bigdl.nn.MultiRNNCell", MultiRNNCell)
    registerModule("com.intel.analytics.bigdl.nn.SpatialSeparableConvolution",
      SpatialSeparableConvolution)
    registerModule("com.intel.analytics.bigdl.nn.Transformer",
      Transformer)
    registerModule("com.intel.analytics.bigdl.models.maskrcnn.MaskRCNN",
      MaskRCNN)
  }
}

private case class GenericTypeWrapper[T: ClassTag](tensor : Tensor[T],
  regularizer : Regularizer[T],
  abstractModule: AbstractModule[Activity, Activity, T],
  tensorModule : TensorModule[T],
  module: Module[T],
  boundedModule: AbstractModule[_ <: Activity, _ <: Activity, T],
  ttpe : T
  )(implicit ev: TensorNumeric[T])

