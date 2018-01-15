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

import java.lang.reflect.Field

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn._

import scala.collection.JavaConverters._
import scala.reflect.runtime.universe
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.nn.ops.{DecodeRawSerializer, ParseExample, RandomUniform => RandomUniformOps}
import com.intel.analytics.bigdl.nn.tf.StrideSlice
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.{Tensor, TensorNumericMath}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import serialization.Bigdl.{AttrValue, BigDLModule}

import scala.collection.mutable
import scala.reflect.ClassTag

object ModuleSerializer extends ModuleSerializable{

  private val runtimeMirror = universe.runtimeMirror(getClass.getClassLoader)

  private val serializerMaps = new mutable.HashMap[String, ModuleSerializable]()

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
    if (serializerMaps.contains(clsName)) {
      serializerMaps(clsName).serializeModule(serializerContext)
    } else {
      val m = module.asInstanceOf[AbstractModule[_, _, _]]
      m match {
        case container : Container[_, _, _] =>
          ContainerSerializer.serializeModule(serializerContext)
        case cell : Cell[_] =>
          CellSerializer.serializeModule(serializerContext)
        case _ => ModuleSerializer.serializeModule(serializerContext)
      }
    }
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
      if (serializerMaps.contains(model.getModuleType)) {
        serializerMaps(model.getModuleType).loadModule(context)
      } else {
        val attrMap = model.getAttrMap
        val subModuleCount = model.getSubModulesCount
        if (subModuleCount > 0) {
          ContainerSerializer.loadModule(context)
        } else {
          if (attrMap.containsKey("is_cell_module")) {
            CellSerializer.loadModule(context)
          } else {
            ModuleSerializer.loadModule(context)
          }
        }
      }
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
    serializerMaps(moduleType) = serializer
  }

  private[serializer] def getCostructorMirror[T : ClassTag](cls : Class[_]):
    universe.MethodMirror = {
    lock.synchronized {
      val clsSymbol = runtimeMirror.classSymbol(cls)
      val cm = runtimeMirror.reflectClass(clsSymbol)
      // to make it compatible with both 2.11 and 2.10
      val ctorCs = clsSymbol.toType.declaration(universe.nme.CONSTRUCTOR)
      val primary: Option[universe.MethodSymbol] = ctorCs.asTerm.alternatives.collectFirst {
        case cstor: universe.MethodSymbol if cstor.isPrimaryConstructor => cstor
      }
      cm.reflectConstructor(primary.get)
    }
  }

  private def init() : Unit = {
    initializeDeclaredTypes
    registerModules
  }

  private def initializeDeclaredTypes() : Unit = {

    var wrapperCls = Class.forName("com.intel.analytics.bigdl.utils.serializer.GenericTypeWrapper")
    val fullParams = getCostructorMirror(wrapperCls).symbol.paramss
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
    registerModule("com.intel.analytics.bigdl.nn.StaticGraph", Graph)
    registerModule("com.intel.analytics.bigdl.nn.DynamicGraph", Graph)
    registerModule("com.intel.analytics.bigdl.nn.MapTable", MapTable)
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
    registerModule("com.intel.analytics.bigdl.nn.VolumetricMaxPooling", VolumetricMaxPooling)
    registerModule("com.intel.analytics.bigdl.nn.Echo", Echo)
    registerModule("com.intel.analytics.bigdl.nn.quantized.SpatialConvolution",
      quantized.SpatialConvolution)
    registerModule("com.intel.analytics.bigdl.nn.quantized.SpatialDilatedConvolution",
      quantized.SpatialDilatedConvolution)
    registerModule("com.intel.analytics.bigdl.nn.quantized.Linear",
      quantized.Linear)
    registerModule("com.intel.analytics.bigdl.nn.ops.ParseExample", ParseExample)
    registerModule("com.intel.analytics.bigdl.nn.SReLU", SReLU)
    registerModule("com.intel.analytics.bigdl.nn.ops.DecodeRaw", DecodeRawSerializer)
    registerModule("com.intel.analytics.bigdl.nn.ops.RandomUniform", RandomUniformOps)
    registerModule("com.intel.analytics.bigdl.nn.tf.StrideSlice", StrideSlice)
    registerModule("com.intel.analytics.bigdl.nn.MultiRNNCell", MultiRNNCell)
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

