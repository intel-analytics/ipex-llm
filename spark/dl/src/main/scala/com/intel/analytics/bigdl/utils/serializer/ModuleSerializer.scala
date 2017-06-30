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

import com.intel.analytics.bigdl.nn.{Abs, Add, AddConstant}

import scala.collection.JavaConverters._
import scala.reflect.runtime.universe
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.{DoubleType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import serialization.Model.AttrValue.DataType
import serialization.Model.{AttrValue, BigDLModel, BigDLTensor}

import scala.collection.mutable
import scala.reflect.ClassTag

object ModuleSerializer extends ModuleSerializable{

  private val moduleMaps = new mutable.HashMap[String, Class[_]]()
  private val classMaps = new mutable.HashMap[Class[_], String]()
  private val deserializerMaps = new mutable.HashMap[String, ModuleSerializable]()
  private val serializerMaps = new mutable.HashMap[Class[_], ModuleSerializable]()

  init

  override def loadModule[T: ClassTag](model : BigDLModel)
    (implicit ev: TensorNumeric[T]) : BigDLModule[T] = {
    val dataType = ev.getType
    val evidence = scala.reflect.classTag[T]
    val modelAttributes = model.getAttrMap
    val moduleType = model.getModuleType
    val cls = ModuleSerializer.getModuleClsByType(moduleType)
    val constructors = cls.getConstructors()
    require(constructors.length == 1, "only support one constructor")
    val constructor = constructors(0)
    val constructorFullParams = getCostructorFullParams(cls)
    val args = new Array[Object](constructorFullParams(0).size + constructorFullParams(1).size)
    var i = 0;
    constructorFullParams.foreach(map => {
      map.foreach(param => {
        val name = param.name.decodedName.toString
        val ptype = param.typeSignature
        if (ptype.toString == "scala.reflect.ClassTag[T]") {
          args(i) = evidence
        } else if (ptype.toString ==
          "com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric[T]") {
          args(i) = ev
        } else {
          require(modelAttributes.containsKey(name), s"$name value cannot be found")
          val attribute = modelAttributes.get(name)
          args(i) = DataConverter.getAttributeValue(attribute)
        }
        i+= 1
      })
    })

    val module = constructor.newInstance(args : _*).
      asInstanceOf[AbstractModule[Activity, Activity, T]]
    createBigDLModule(model, module)
  }

  override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                           (implicit ev: TensorNumeric[T]) : BigDLModel = {
    val bigDLModelBuilder = BigDLModel.newBuilder
    val cls = module.module.getClass
    val moduleType = getModuleTypeByCls(module.module.getClass)
    bigDLModelBuilder.setModuleType(moduleType)
    val constructors = cls.getConstructors()
    require(constructors.length == 1, "only support one constructor")
    val constructor = constructors(0)
    val fullParams = getCostructorFullParams(cls)
    val constructorParams = fullParams(0)
    constructorParams.foreach(param => {
      val paramName = param.name.decodedName.toString
      val ptype = param.typeSignature
      val attrBuilder = AttrValue.newBuilder
      val field = cls.getDeclaredField(paramName)
      field.setAccessible(true)
      val fieldValue = field.get(module.module)
      DataConverter.setAttributeValue(attrBuilder, fieldValue, ptype)
      bigDLModelBuilder.putAttr(paramName, attrBuilder.build)
    })
    copyFromBigDL(module, bigDLModelBuilder)
    createSerializeBigDLModule(bigDLModelBuilder, module)
  }

  def serialize[T: ClassTag](bigDLModule : BigDLModule[T])
                            (implicit ev: TensorNumeric[T])
    : BigDLModel = {
    serializerMaps(bigDLModule.module.getClass).serializeModule(bigDLModule)
  }

  def load[T: ClassTag](model: BigDLModel)
                       (implicit ev: TensorNumeric[T]) : BigDLModule[T] = {
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

  def getCostructorFullParams[T : ClassTag](cls : Class[_]) : List[List[universe.Symbol]] = {
    val m = universe.runtimeMirror(getClass.getClassLoader)
    val clsSymbol = m.classSymbol(cls)
    val cm = m.reflectClass(clsSymbol)
    // to make it compatible with both 2.11 and 2.10
    val ctorC = clsSymbol.toType.declaration(universe.nme.CONSTRUCTOR).asMethod
    val ctorm = cm.reflectConstructor(ctorC)
    /*
    val params0 = ctorm.symbol.paramss(0).foldLeft(Map(): Map[String, universe.Type])((p, a) => {
      p + (a.name.decodedName.toString -> a.typeSignature)
    })
    val params1 = ctorm.symbol.paramss(1).foldLeft(Map(): Map[String, universe.Type])((p, a) => {
      p + (a.name.decodedName.toString -> a.typeSignature)
    })
    */
   // List(params0, params1)
    ctorm.symbol.paramss
  }

  def init() : Unit = {
    registerModule("ADD", Class.forName("com.intel.analytics.bigdl.nn.Add"), Add)
    registerModule("ADDCONSTANT", Class.forName("com.intel.analytics.bigdl.nn.AddConstant"),
      AddConstant)
    registerModule("LINEAR", Class.forName("com.intel.analytics.bigdl.nn.Linear"),
      AddConstant)
  }
}
