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

import com.intel.analytics.bigdl.nn._

import scala.collection.JavaConverters._
import scala.reflect.runtime.universe
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.{Tensor, TensorNumericMath}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import serialization.Bigdl.{AttrValue, BigDLModule}

import scala.collection.mutable
import scala.reflect.ClassTag

object ModuleSerializer extends ModuleSerializable{

  val runtimeMirror = universe.runtimeMirror(getClass.getClassLoader)

  private val moduleMaps = new mutable.HashMap[String, Class[_]]()
  private val classMaps = new mutable.HashMap[Class[_], String]()
  private val deserializerMaps = new mutable.HashMap[String, ModuleSerializable]()
  private val serializerMaps = new mutable.HashMap[Class[_], ModuleSerializable]()

  // generic type definition for type matching

  var tensorNumericType : universe.Type = null
  var tensorType : universe.Type = null
  var regularizerType : universe.Type = null
  var abstractModuleType : universe.Type = null
  var tensorModuleType : universe.Type = null
  var tType : universe.Type = null

  init

  override def loadModule[T: ClassTag](model : BigDLModule)
    (implicit ev: TensorNumeric[T]) : ModuleData[T] = {

    val evidence = scala.reflect.classTag[T]
    val modelAttributes = model.getAttrMap
    val moduleType = model.getModuleType
    val cls = ModuleSerializer.getModuleClsByType(moduleType)
    val constructorMirror = getCostructorMirror(cls)
    val constructorFullParams = constructorMirror.symbol.paramss
    val args = new Array[Object](constructorFullParams(0).size + constructorFullParams(1).size)
    var i = 0;
    constructorFullParams.foreach(map => {
      map.foreach(param => {
        val name = param.name.decodedName.toString
        val ptype = param.typeSignature
        if (ptype.toString == "scala.reflect.ClassTag[T]") {
          args(i) = evidence
        } else if (ptype.toString ==
          tensorNumericType.toString) {
          args(i) = ev
        } else {
          require(modelAttributes.containsKey(name), s"$name value cannot be found")
          val attribute = modelAttributes.get(name)
          val value = DataConverter.getAttributeValue(attribute)
          args(i) = value
        }
        i+= 1
      })
    })
    val module = constructorMirror.apply(args : _*).
      asInstanceOf[AbstractModule[Activity, Activity, T]]
    createBigDLModule(model, module)
  }

  override def serializeModule[T: ClassTag](module : ModuleData[T])
                                           (implicit ev: TensorNumeric[T]) : BigDLModule = {
    val bigDLModelBuilder = BigDLModule.newBuilder
    val cls = module.module.getClass
    val moduleType = getModuleTypeByCls(cls)
    bigDLModelBuilder.setModuleType(moduleType)
    val fullParams = getCostructorMirror(cls).symbol.paramss
    val clsTag = scala.reflect.classTag[T]
    val constructorParams = fullParams(0)
    constructorParams.foreach(param => {
      val paramName = param.name.decodedName.toString
      var ptype = param.typeSignature
      val attrBuilder = AttrValue.newBuilder
      // For some modules, fields are declared inside but passed to Super directly
      var field : Field = null
      try {
        field = cls.getDeclaredField(paramName)
      } catch {
        case e : NoSuchFieldException =>
          field = cls.getSuperclass.getDeclaredField(paramName)
      }
      field.setAccessible(true)
      val fieldValue = field.get(module.module)
      DataConverter.setAttributeValue(attrBuilder, fieldValue, ptype)
      bigDLModelBuilder.putAttr(paramName, attrBuilder.build)
    })
    createSerializeBigDLModule(bigDLModelBuilder, module)
  }


  def serialize[T: ClassTag](bigDLModule : ModuleData[T])
                            (implicit ev: TensorNumeric[T])
    : BigDLModule = {
    val module = bigDLModule.module
    val cls = module.getClass
    serializerMaps(cls).serializeModule(bigDLModule)
  }

  def load[T: ClassTag](model: BigDLModule)
                       (implicit ev: TensorNumeric[T]) : ModuleData[T] = {
    deserializerMaps(model.getModuleType).loadModule(model)
  }



  def registerModule(moduleType : String, moduleCls : Class[_],
    serializer : ModuleSerializable) : Unit = {
    moduleMaps(moduleType) = moduleCls
    classMaps(moduleCls) = moduleType
    serializerMaps(moduleCls) = serializer
    deserializerMaps(moduleType) = serializer
  }

  def getModuleClsByType(moduleType : String) : Class[_] = {
    require(moduleMaps.contains(moduleType), s"$moduleType is not supported")
    moduleMaps(moduleType)
  }

  def getModuleTypeByCls(cls : Class[_]) : String = {
    require(classMaps.contains(cls), s"$cls is not supported")
    classMaps(cls)
  }

  def getCostructorMirror[T : ClassTag](cls : Class[_]) : universe.MethodMirror = {

    val clsSymbol = runtimeMirror.classSymbol(cls)
    val cm = runtimeMirror.reflectClass(clsSymbol)
    // to make it compatible with both 2.11 and 2.10
    val ctorCs = clsSymbol.toType.declaration(universe.nme.CONSTRUCTOR)
    val primary : Option[universe.MethodSymbol] = ctorCs.asTerm.alternatives.collectFirst{
      case cstor : universe.MethodSymbol if cstor.isPrimaryConstructor => cstor
    }
    cm.reflectConstructor(primary.get)
  }

  private def init() : Unit = {
    initializeDeclaredTypes
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
        } else if (name == "ev") {
          tensorNumericType = ptype
        } else if (name == "ttpe") {
          tType = ptype
        }
      })
    })
  }
}

private case class GenericTypeWrapper[T: ClassTag](tensor : Tensor[T],
                                              regularizer : Regularizer[T],
                                              abstractModule: AbstractModule[Activity, Activity, T],
                                              tensorModule : TensorModule[T],
                                              ttpe : T
                                             )(implicit ev: TensorNumeric[T])

