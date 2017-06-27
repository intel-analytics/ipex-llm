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


trait ModuleSerializable extends Loadable with Savable{

  override def loadModule[T: ClassTag](model : BigDLModel)
                                       (implicit ev: TensorNumeric[T]) : BigDLModule[T] = {
    ModuleSerializer.loadModule(model)
  }

  override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                           (implicit ev: TensorNumeric[T]) : BigDLModel = {
    ModuleSerializer.serializeModule(module)
  }

  protected def createBigDLModule[T: ClassTag](model : BigDLModel,
                                               module : AbstractModule[Activity, Activity, T])
                                              (implicit ev: TensorNumeric[T])
  : BigDLModule[T] = {
    val preModules = model.getPreModulesList.asScala
    val nextModules = model.getNextModulesList.asScala
    val bigDLModule = BigDLModule(module, preModules, nextModules)
    module.setName(model.getName)
    copy2BigDL(model, bigDLModule)
    bigDLModule
  }

  protected def createSerializeBigDLModule[T: ClassTag](
    modelBuilder : BigDLModel.Builder, module : BigDLModule[T])(implicit ev: TensorNumeric[T])
  : BigDLModel = {
    module.pre.foreach(pre => modelBuilder.addPreModules(pre))
    module.next.foreach(next => modelBuilder.addNextModules(next))
    modelBuilder.setName(module.module.getName)
    copyFromBigDL(module, modelBuilder)
    modelBuilder.build
  }

  /**
    *  copy serialized data (weight and bias if exist) to BigDL module
    *  @param model serialized module
    *  @param module  bigDL Module with relationships
    */
  protected def copy2BigDL[T: ClassTag](model : BigDLModel, module : BigDLModule[T])
                                       (implicit ev: TensorNumeric[T]): Unit = {
    val paramTable : Table = module.module.getParametersTable
    if (paramTable != null && paramTable.contains(model.getName)) {
      val modulePramTable : Table = paramTable(module.module.getName)
      val weight : Tensor[T] = if (modulePramTable.contains("weight")) {
        modulePramTable("weight") }
      else null
      val bias : Tensor[T] = if (modulePramTable.contains("bias")) {
        modulePramTable("bias") }
      else null
      if (weight != null) copy2BigDLTensor(weight, model.getWeight)
      if (bias != null) copy2BigDLTensor(bias, model.getBias)
    }
  }

  private def copy2BigDLTensor[T: ClassTag](tensor : Tensor[T], serializedTensor : BigDLTensor)
                                           (implicit ev: TensorNumeric[T]) : Unit = {
    val serializedData = serializedTensor.getDataList
    require(tensor.nElement() == serializedData.size(), "data size is not equal")
    var i = 0
    val tensorData = tensor.storage().array()
    var offset = tensor.storageOffset() - 1
    while (i < serializedData.size()) {
      tensorData(offset) = ev.fromType[Double](serializedData.get(i))
      offset += 1
      i += 1
    }
  }

  /**
    * copy BigDL module data (weight and bias if exist) to BigDL Model to be persisted
    * @param modelBuilder serialized module builder
    * @param module  bigDL Module with relationships
    */
  protected def copyFromBigDL[T: ClassTag](module : BigDLModule[T],
    modelBuilder : BigDLModel.Builder)(implicit ev : TensorNumeric[T]) : Unit = {
    val paramTable : Table = module.module.getParametersTable
    if (paramTable != null && paramTable.contains(module.module.getName)) {
      val modulePramTable : Table = paramTable(module.module.getName)
      val weight : Tensor[T] = if (modulePramTable.contains("weight")) {
        modulePramTable("weight") }
      else null
      val bias : Tensor[T] = if (modulePramTable.contains("bias")) {
        modulePramTable("bias") }
      else null
      if (weight != null) {
        val weightTensorBuilder = BigDLTensor.newBuilder
        copyFromBigDLTensor(weight, weightTensorBuilder)
        modelBuilder.setWeight(weightTensorBuilder.build)
      }
      if (bias != null) {
        val biasTensorBuilder = BigDLTensor.newBuilder
        copyFromBigDLTensor(bias, biasTensorBuilder)
        modelBuilder.setBias(biasTensorBuilder.build)
      }
    }
  }

  private def copyFromBigDLTensor[T: ClassTag](tensor : Tensor[T],
    serializedTensor : BigDLTensor.Builder)(implicit ev: TensorNumeric[T]) : Unit = {
    var i = 0
    val tensorData = tensor.storage().array()
    var offset = tensor.storageOffset() - 1
    while (i < tensorData.length) {
      serializedTensor.addData(ev.toType[Double](tensorData(i)))
      i += 1
    }
    tensor.size().foreach(_ => serializedTensor.addSize(_))
  }


}

trait Loadable {

  def loadModule[T: ClassTag](model : BigDLModel)
                             (implicit ev: TensorNumeric[T]) : BigDLModule[T]
}
trait Savable {

  def serializeModule[T: ClassTag](module : BigDLModule[T])
                                  (implicit ev: TensorNumeric[T]) : BigDLModel
}

object ModuleSerializer extends ModuleSerializable{

  private val moduleMaps = new mutable.HashMap[String, Class[_]]()
  private val classMaps = new mutable.HashMap[Class[_], String]()
  private val deserializerMaps = new mutable.HashMap[String, ModuleSerializable]()
  private val serializerMaps = new mutable.HashMap[Class[_], ModuleSerializable]()

  init

  override def loadModule[T: ClassTag](model : BigDLModel)
    (implicit ev: TensorNumeric[T]) : BigDLModule[T] = {
    val dataType = ev.getType
    val evidence = if (dataType == DoubleType) classManifest[Double] else classManifest[Float]
    val modelAttributes = model.getAttrMap
    val moduleType = model.getModuleType
    val cls = ModuleSerializer.getModuleClsByType(moduleType)
    val constructors = cls.getConstructors()
    require(constructors.length == 1, "only support one constructor")
    val constructor = constructors(0)
    val constructorFullParams = getCostructorFullParams(cls)
   // val paramsMap = new mutable.HashMap[String, universe.Type]()
   // constructorFullParams(1).foreach(p => paramsMap.put(p._1, p._2))
   // constructorFullParams(0).foreach(p => paramsMap.put(p._1, p._2))
    val args = new Array[Object](constructorFullParams(0).size + constructorFullParams(1).size)
    var i = 0;
    constructorFullParams.foreach(map => {
      map.foreach(param => {
        val name = param._1
        val ptype = param._2
        if (ptype.toString == "scala.reflect.ClassTag[T]") {
          args(i) = evidence
        } else if (ptype.toString ==
          "com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric[T]") {
          args(i) = ev
        } else {
          require(modelAttributes.containsKey(name), s"$name value cannot be found")
          val attribute = modelAttributes.get(name)
          args(i) = getAttributeValue(attribute)
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
      val paramName = param._1
      val attrBuilder = AttrValue.newBuilder
      val field = cls.getDeclaredField(paramName)
      field.setAccessible(true)
      val fieldValue = field.get(module.module)
      setAttributeValue(attrBuilder, fieldValue)
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


  private def getAttributeValue(attribute: AttrValue) : Object = {
    attribute.getDataType match {
      case DataType.INT32 => Integer.valueOf(attribute.getInt32Value)
      case DataType.INT64 => Long.box(attribute.getInt64Value)
      case DataType.DOUBLE => Double.box(attribute.getFloatValue.toDouble)
      case DataType.FLOAT => Float.box(attribute.getFloatValue)
      case DataType.BOOL => Boolean.box(attribute.getBoolValue)
      case _ => throw new IllegalArgumentException
        (s"${attribute.getDataType} can not be recognized")
    }
  }

  private def setAttributeValue(
    attributeBuilder : AttrValue.Builder, value: Object) : Unit = {
    if (value.isInstanceOf[Int]) {
      attributeBuilder.setDataType(DataType.INT32)
      attributeBuilder.setInt32Value(value.asInstanceOf[Int])
    } else if (value.isInstanceOf[Long]) {
      attributeBuilder.setDataType(DataType.INT64)
      attributeBuilder.setInt64Value(value.asInstanceOf[Long])
    } else if ( value.isInstanceOf[Float]) {
      attributeBuilder.setDataType(DataType.FLOAT)
      attributeBuilder.setFloatValue(value.asInstanceOf[Float])
    } else if (value.isInstanceOf[Double]) {
      attributeBuilder.setDataType(DataType.DOUBLE)
      attributeBuilder.setFloatValue(value.asInstanceOf[Double].toFloat)
    } else if (value.isInstanceOf[Boolean]) {
      attributeBuilder.setDataType(DataType.BOOL)
      attributeBuilder.setBoolValue(value.asInstanceOf[Boolean])
    }
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

  def getConstructorParams(clazz : Class[_]): Map[String, universe.Type] = {
    val tpe = universe.runtimeMirror(clazz.getClassLoader).classSymbol(clazz).toType
    var list : List[universe.Symbol] = List()
    list = list ++ tpe.member(universe.termNames.CONSTRUCTOR).asMethod.paramLists(0)
    list = list ++ tpe.member(universe.termNames.CONSTRUCTOR).asMethod.paramLists(1)
    list.foldLeft(Map(): Map[String, universe.Type])((p, a) => {
      p + (a.name.decodedName.toString -> a.typeSignature)
    })
  }

  def getParamList(clazz : Class[_]): Seq[String] = {
    val tpe = universe.runtimeMirror(clazz.getClassLoader).classSymbol(clazz).toType
    tpe.
      member(universe.termNames.CONSTRUCTOR).
      asMethod.paramLists(0).map(_.name.decodedName.toString)
  }

  def getCostructorFullParams[T : ClassTag](cls : Class[_]) : List[Map[String, universe.Type]] = {
    val m = universe.runtimeMirror(getClass.getClassLoader)
    val clsSymbol = m.classSymbol(cls)
    val cm = m.reflectClass(clsSymbol)
    // to make it compatible with both 2.11 and 2.10
    val ctorC = clsSymbol.toType.declaration(universe.nme.CONSTRUCTOR).asMethod
    var list : List[universe.Symbol] = List()
    val ctorm = cm.reflectConstructor(ctorC)
    val params0 = ctorm.symbol.paramss(0).foldLeft(Map(): Map[String, universe.Type])((p, a) => {
      p + (a.name.decodedName.toString -> a.typeSignature)
    })
    val params1 = ctorm.symbol.paramss(1).foldLeft(Map(): Map[String, universe.Type])((p, a) => {
      p + (a.name.decodedName.toString -> a.typeSignature)
    })
    List(params0, params1)
  }

  def init() : Unit = {
    registerModule("ADD", Class.forName("com.intel.analytics.bigdl.nn.Add"), Add)
    registerModule("ADDCONSTANT", Class.forName("com.intel.analytics.bigdl.nn.AddConstant"),
      AddConstant)
  }
}

case class BigDLModule[T: ClassTag](module : AbstractModule[Activity, Activity, T],
                                    pre : Seq[String], next : Seq[String])