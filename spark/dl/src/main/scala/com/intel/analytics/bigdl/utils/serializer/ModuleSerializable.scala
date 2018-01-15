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

import com.intel.analytics.bigdl.nn.Container

import scala.collection.JavaConverters._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.utils.serializer.DataConverter.TensorConverter
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializer._
import serialization.Bigdl.DataType
import serialization.Bigdl.{AttrValue, BigDLModule, BigDLTensor}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.reflect.runtime.universe

/**
 * [[ModuleSerializable]] trait inherits [[Loadable]] and [[Savable]]
 * traits for module serialization
 * it provides default implementation from [[ModuleSerializer]] using reflection
 */
trait ModuleSerializable extends Loadable with Savable{


  private val bigDLVersion = com.intel.analytics.bigdl.BIGDL_VERSION

  protected val lock = new Object

  // Separate this two methods for reuse in sub-classes
  protected def checkVersion[T: ClassTag](module : BigDLModule)
                                         (implicit ev: TensorNumeric[T]) : Unit = {
    val version = module.getVersion
    require(version <= bigDLVersion, s"bigDL version mismatch," +
      s"module version $version," +
      s"bigdl version $bigDLVersion, you cannot use low version bigdl" +
      s" to load a higher version module")
  }

  protected def setVersion[T: ClassTag](modelBuilder : BigDLModule.Builder)
                                       (implicit ev: TensorNumeric[T]) : Unit = {
    modelBuilder.setVersion(bigDLVersion)
  }

  /**
   * Default deserialization to provide the template
   * @return BigDL module instance with linkages with other modules
   */
  override def loadModule[T: ClassTag](context : DeserializeContext)
                                      (implicit ev: TensorNumeric[T]) : ModuleData[T] = {

    val model = context.bigdlModule

    // step 1 : check version
    checkVersion(model)

    // step2 : module specific logic to load module, either default, cell, container or graph
    val moduleId = context.bigdlModule.getId

    val storages = context.storages

    val module = if (storages.contains(moduleId)) {
      storages.get(moduleId).get.asInstanceOf[AbstractModule[Activity, Activity, T]]
    } else {
      val loadedModule = doLoadModule(context)
      storages(moduleId) = loadedModule
      loadedModule
    }
    // step3 : copy params (weight & bias) and linkage
    createBigDLModule(context, module)
  }

  /**
   * Default deserialization using reflection
   * @param context deserialize context
   * @return BigDL module
   */
  protected def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {

    val (tags, numerics) = getTypes(context)
    val tagIter = tags.iterator
    val numericIter = numerics.iterator
    val evidence = scala.reflect.classTag[T]
    val model = context.bigdlModule
    val modelAttributes = model.getAttrMap
    val moduleType = model.getModuleType
    val cls = Class.forName(moduleType)
    val constructorMirror = getCostructorMirror(cls)
    val constructorFullParams = constructorMirror.symbol.paramss
    val args = new Array[Object](constructorFullParams.map(_.size).sum)
    var i = 0
    lock.synchronized {
      constructorFullParams.foreach(map => {
        map.foreach(param => {
          val name = param.name.decodedName.toString
          val ptype = param.typeSignature
          if (ptype <:< universe.typeOf[ClassTag[_]]||
            ptype.typeSymbol == universe.typeOf[ClassTag[_]].typeSymbol) {
            args(i) = tagIter.next
          } else if (ptype <:< universe.typeOf[TensorNumeric[_]]
            || ptype.typeSymbol == universe.typeOf[TensorNumeric[_]].typeSymbol) {
            args(i) = numericIter.next
          } else {
            require(modelAttributes.containsKey(name), s"$name value cannot be found")
            val attribute = modelAttributes.get(name)
            val value = DataConverter.getAttributeValue(context, attribute)
            args(i) = value
          }
          i+= 1
        })
      })
    }
   constructorMirror.apply(args : _*).
      asInstanceOf[AbstractModule[Activity, Activity, T]]
  }

  private def getTypes(context: DeserializeContext):
  (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    val attrMap = context.bigdlModule.getAttrMap
    val tags = attrMap.get(SerConst.MODULE_TAGES).getArrayValue.getStrList.asScala
      .map(ClassTagMapper.apply(_)).toArray
    val numeris = attrMap.get(SerConst.MODULE_NUMERICS).getArrayValue.getStrList.
      asScala.map(TensorNumericMapper.apply(_)).toArray
    (tags, numeris)
  }

  /**
   *  Default serialization skeleton using reflection
   *  @param context Serialization context
   *  @return serialized protobuf module instace
   */
  override def serializeModule[T: ClassTag](context: SerializeContext[T])
                                  (implicit ev: TensorNumeric[T]): SerializeResult = {

    val bigDLModelBuilder = BigDLModule.newBuilder

    // step 1 : set module version
    setVersion(bigDLModelBuilder)

    val moduleData = context.moduleData
    val cls = moduleData.module.getClass

    // step 2: set module type
    bigDLModelBuilder.setModuleType(cls.getName)

    // step 3 : set data types (ClassTag and TensorNumric)
    setDataTypes(context, bigDLModelBuilder)

    // step 4 : apply module specific logic to create module
    doSerializeModule(context, bigDLModelBuilder)

    // step 5 : copy params (weight & bias) a and linkage
    createSerializeBigDLModule(bigDLModelBuilder, context)
  }

  protected def setDataTypes[T: ClassTag](context: SerializeContext[T],
    bigDLModelBuilder: BigDLModule.Builder)(implicit ev: TensorNumeric[T]): Unit = {
    val (tags, numerics) = context.moduleData.module.getClassTagNumerics
    val tagsSer = tags.map(ClassTagMapper.apply(_))
    val tagAttrValue = AttrValue.newBuilder
    DataConverter.setAttributeValue[T](context, tagAttrValue,
      tagsSer, universe.typeOf[Array[String]])
    bigDLModelBuilder.putAttr(SerConst.MODULE_TAGES, tagAttrValue.build)
    val numericAttrValue = AttrValue.newBuilder
    val numericSer = numerics.map(TensorNumericMapper.apply(_))
    DataConverter.setAttributeValue[T](context,
      numericAttrValue, numericSer, universe.typeOf[Array[String]])
    bigDLModelBuilder.putAttr(SerConst.MODULE_NUMERICS, numericAttrValue.build)
  }

  protected def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                               bigDLModelBuilder : BigDLModule.Builder)
                                              (implicit ev: TensorNumeric[T]) : Unit = {
    val module = context.moduleData.module
    val cls = module.getClass
    val fullParams = getCostructorMirror(cls).symbol.paramss
    val constructorParams = fullParams(0)
    lock.synchronized {
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
        val fieldValue = field.get(module)
        DataConverter.setAttributeValue(context, attrBuilder, fieldValue, ptype)

        bigDLModelBuilder.putAttr(paramName, attrBuilder.build)
      })
    }
  }

  protected def createBigDLModule[T: ClassTag](context: DeserializeContext,
                                               module : AbstractModule[Activity, Activity, T])
                                              (implicit ev: TensorNumeric[T])
  : ModuleData[T] = {
    val model = context.bigdlModule
    val preModules = model.getPreModulesList.asScala
    val nextModules = model.getNextModulesList.asScala
    val bigDLModule = ModuleData(module, preModules, nextModules)
    if (model.getName != "") {
      module.setName(model.getName)
    }
    module.setNamePostfix(model.getNamePostfix)
    if (model.getTrain) {
      module.training()
    } else {
      module.evaluate()
    }
    copy2BigDL(context, bigDLModule)
    bigDLModule
  }

  protected def createSerializeBigDLModule[T: ClassTag](
    modelBuilder : BigDLModule.Builder, context: SerializeContext[T])(implicit ev: TensorNumeric[T])
  : SerializeResult = {
    val module = context.moduleData
    module.pre.foreach(pre => modelBuilder.addPreModules(pre))
    module.next.foreach(next => modelBuilder.addNextModules(next))
    if (module.module.hasName) {
      modelBuilder.setName(module.module.getName)
    }
    modelBuilder.setNamePostfix(module.module.getNamePostfix)
    modelBuilder.setTrain(module.module.isTraining())
    modelBuilder.setId(System.identityHashCode(module.module))
    copyFromBigDL(context, modelBuilder)
    SerializeResult(modelBuilder, context.storages)
  }

  /**
   * copy serialized data (weight and bias if exist) to BigDL module
   * @param context deserialized context
   * @param module  bigDL Module with relationships
   */
  protected def copy2BigDL[T: ClassTag](context: DeserializeContext, module : ModuleData[T])
                                       (implicit ev: TensorNumeric[T]): Unit = {
    val paramTable : Table = module.module.getParametersTable
    if (paramTable != null && paramTable.contains(module.module.getName)) {
      val modulePramTable : Table = paramTable(module.module.getName)
      if (modulePramTable.contains("weight")) {
        val attrValue = AttrValue.newBuilder
        attrValue.setTensorValue(context.bigdlModule.getWeight)
        val weight = TensorConverter.getAttributeValue(context, attrValue.build)
        modulePramTable("weight").asInstanceOf[Tensor[T]].
          copy(weight.asInstanceOf[Tensor[T]])
      }
      if (modulePramTable.contains("bias")) {
        val attrValue = AttrValue.newBuilder
        attrValue.setTensorValue(context.bigdlModule.getBias)
        val bias = TensorConverter.getAttributeValue(context, attrValue.build)
        modulePramTable("bias").asInstanceOf[Tensor[T]].
          copy(bias.asInstanceOf[Tensor[T]])
      }
    }
  }

  /**
   * copy BigDL module data (weight and bias if exist) to BigDL Model to be persisted
   * @param modelBuilder serialized module builder
   * @param context  serialization context
   */
  protected def copyFromBigDL[T: ClassTag](context : SerializeContext[T],
    modelBuilder : BigDLModule.Builder)(implicit ev : TensorNumeric[T]) : Unit = {
    val module = context.moduleData
    val paramTable : Table = module.module.getParametersTable
    if (paramTable != null && paramTable.contains(module.module.getName)) {
      val modulePramTable: Table = paramTable(module.module.getName)
      val weight: Tensor[T] = if (modulePramTable.contains("weight")) {
        modulePramTable("weight")
      }
      else null
      val bias: Tensor[T] = if (modulePramTable.contains("bias")) {
        modulePramTable("bias")
      }
      else null
      val storageType = context.storageType
      if (weight != null) {
        val weightAttr = AttrValue.newBuilder
        TensorConverter.setAttributeValue(context, weightAttr, weight)
        modelBuilder.setWeight(weightAttr.getTensorValue)
      }
      if (bias != null) {
        val biasAttr = AttrValue.newBuilder
        TensorConverter.setAttributeValue(context, biasAttr, bias)
        modelBuilder.setBias(biasAttr.getTensorValue)
      }
    }
  }

}

trait ContainerSerializable extends ModuleSerializable {

  override def doLoadModule[T: ClassTag](context : DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val module = super.doLoadModule(context)
    val container = module.asInstanceOf[Container[Activity, Activity, T]]
    val subModules = context.bigdlModule.getSubModulesList.asScala
    subModules.foreach(module => {
      val subModuleData = ModuleSerializer.load(DeserializeContext(module,
      context.storages, context.storageType))
      container.modules.append(subModuleData.module)
    })
    module
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              containerBuilder : BigDLModule.Builder)
                                           (implicit ev: TensorNumeric[T]) : Unit = {

    super.doSerializeModule(context, containerBuilder)
    val subModulesData = context.moduleData.module.
      asInstanceOf[Container[Activity, Activity, T]].modules
    subModulesData.foreach(module => {
      val subModule = ModuleSerializer.serialize(SerializeContext(ModuleData(module,
        new ArrayBuffer[String](), new ArrayBuffer[String]()), context.storages,
      context.storageType))
      containerBuilder.addSubModules(subModule.bigDLModule)
    })
  }
}

object ContainerSerializer extends ContainerSerializable

trait Loadable {

  def loadModule[T: ClassTag](context: DeserializeContext)
                             (implicit ev: TensorNumeric[T]) : ModuleData[T]
}

trait Savable {

  def serializeModule[T: ClassTag](context: SerializeContext[T])
                                  (implicit ev: TensorNumeric[T]) : SerializeResult
}
