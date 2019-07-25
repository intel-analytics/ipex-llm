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

import com.intel.analytics.bigdl.nn.{Container, MklInt8Convertible}

import scala.collection.JavaConverters._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.serialization.Bigdl.AttrValue.ArrayValue
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{ReflectionUtils, Table, Shape => BigDLShape}
import com.intel.analytics.bigdl.utils.serializer.converters.{DataConverter, ShapeConverter, TensorConverter}
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializer._
import com.intel.analytics.bigdl.serialization.Bigdl._

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

  protected var _copyWeightAndBias = true

  protected def getLock: Object = ModuleSerializer._lock

  // Separate this two methods for reuse in sub-classes
  protected def checkVersion[T: ClassTag](module : BigDLModule)
                                         (implicit ev: TensorNumeric[T]) : Unit = {
    val moduleVersion = module.getVersion
    val modelVersionSplits = moduleVersion.split(".")
    val bigdlVersionSplits = bigDLVersion.split(".")
    require(modelVersionSplits.length == bigdlVersionSplits.length,
      s"model version ${moduleVersion} has different format as BigDL version ${bigDLVersion}")
    (0 until modelVersionSplits.length).foreach(idx => {
      require(modelVersionSplits(idx).toInt <= bigdlVersionSplits(idx).toInt,
        s"bigDL version mismatch," +
          s"module version $moduleVersion," +
          s"bigdl version $bigDLVersion, you cannot use low version bigdl" +
          s" to load a higher version module")
    })
  }

  protected def setVersion[T: ClassTag](modelBuilder : BigDLModule.Builder)
                                       (implicit ev: TensorNumeric[T]) : Unit = {
    modelBuilder.setVersion(bigDLVersion)
  }

  protected def copyWeightAndBias() = _copyWeightAndBias

  def setCopyWeightAndBias(copyWeightAndBias : Boolean): this.type = {
    _copyWeightAndBias = copyWeightAndBias
    this
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
      getLock.synchronized {
        val loadedModule = doLoadModule(context)
        storages(moduleId) = loadedModule
        loadedModule
      }
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
    (implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {

    val (tags, numerics) = getTypes(context)
    val model = context.bigdlModule
    val moduleType = model.getModuleType
    val cls = Class.forName(moduleType)

    val modelAttributes = model.getAttrMap.asScala.map{
      case (key, value) =>
        (key, DataConverter.getAttributeValue(context, value)
    )}.toMap

    val module = ReflectionUtils.reflection(cls, modelAttributes,
      tags, numerics)

    module.asInstanceOf[AbstractModule[Activity, Activity, T]]
  }


  protected def getTypes(context: DeserializeContext):
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

    // step 3 : set group information

    if (context.groupType != null) {
      val groupTypeAttrValue = AttrValue.newBuilder
      DataConverter.setAttributeValue[T](context, groupTypeAttrValue,
        context.groupType, universe.typeOf[String])
      bigDLModelBuilder.putAttr(SerConst.GROUP_TYPE, groupTypeAttrValue.build)
    }

    getLock.synchronized {
      // step 4 : set data types (ClassTag and TensorNumric)
      setDataTypes(context, bigDLModelBuilder)
      // step 5 : apply module specific logic to create module
      doSerializeModule(context, bigDLModelBuilder)
    }

    // step 6 : copy params (weight & bias) a and linkage
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
    val fullParams = ReflectionUtils.getPrimCtorMirror(cls).symbol.paramss
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
      val fieldValue = field.get(module)
      DataConverter.setAttributeValue(context, attrBuilder, fieldValue, ptype)

      bigDLModelBuilder.putAttr(paramName, attrBuilder.build)
      })
  }

  /**
   * Re-create BigDL module by deserializing protobuf context.
   * @param context Deserialization context
   * @param module The BigDL module to be re-created
   * @param ev
   * @tparam T
   * @return Tuple3 contains information of current module and modules adjacent to it
   */
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
    module.inputShapeValue = ShapeConverter.shapeToBigDL(context, model, "input")
    module.outputShapeValue = ShapeConverter.shapeToBigDL(context, model, "output")

    // container does not need to be copied paramters again
    if (_copyWeightAndBias && context.bigdlModule.getSubModulesCount == 0) {
      copy2BigDL(context, bigDLModule)
    }

    // Load MKL-DNN INT8 attributes (scales&mask of input&output) into
    // BigDL Module from protobuf definition if the MKL-DNN INT8 flag is ON
    if (model.getIsMklInt8Enabled) {
      loadMklInt8Attr(context, module.asInstanceOf[MklInt8Convertible])

    }

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
    val inputShape = module.module.inputShapeValue
    if (inputShape != null) {
      modelBuilder.setInputShape(ShapeConverter.shapeToProto(context, inputShape))
    }
    val outputShape = module.module.outputShapeValue
    if (outputShape != null) {
      modelBuilder.setOutputShape(ShapeConverter.shapeToProto(context, outputShape))
    }
    // container does not need to be copied paramters again
    if (_copyWeightAndBias && !module.isInstanceOf[Container[_, _, _]]) {
      copyFromBigDL(context, modelBuilder)
    }

    // Save MKL-DNN attributes (scales and masks) into model of protobuf definition if
    // the module is with trait of MklInt8COnvertible, and set the MKL-DNN INT8 flag to true
    if (module.module.isInstanceOf[MklInt8Convertible]) {
      saveMklInt8Attr(context.moduleData.module.asInstanceOf[MklInt8Convertible], modelBuilder)
      modelBuilder.setIsMklInt8Enabled(true)
    } else {
      modelBuilder.setIsMklInt8Enabled(false)
    }

    SerializeResult(modelBuilder, context.storages)
  }

  /**
   * copy serialized data (weight and bias if exist) to BigDL module
   * @param context deserialized context
   * @param module  bigDL Module with relationships
   */
  protected def copy2BigDL[T: ClassTag](context: DeserializeContext, module : ModuleData[T])
                                       (implicit ev: TensorNumeric[T]): Unit = {

    if (context.bigdlModule.getHasParameters) {
      copyParameters2BigDL(context, module)
    } else {
      // for legacy format models
      copyWeightAndBias(context, module)
    }
  }

  /**
   * Deserialize MKL-DNN INT8 attributes from protobuf context
   * and load them into BigDL Module object
   * @param context deserialized context
   * @param module  bigDL Module with relationships
   */
   private def loadMklInt8Attr[T: ClassTag](context: DeserializeContext,
                                           module: MklInt8Convertible)
                                          (implicit ev: TensorNumeric[T]): Unit = {
     val protobufModel = context.bigdlModule
     // Extract ArrayValue for each AttrValue, and then get FltList as input scales
     val inputScales = protobufModel.getInputScalesList.iterator().asScala
      .map(attrValueToFloatArray)
     // Extract ArrayValue for each AttrValue, and then get FltList as output scales
     val outputScales = protobufModel.getOutputScalesList.iterator().asScala
       .map(attrValueToFloatArray)
     // Extract ArrayValue for each AttrValue, and then get FltList as weight scales
     val weightScales = protobufModel.getWeightScalesList.iterator().asScala
      .map(attrValueToFloatArray)

     module.setInputDimMask(protobufModel.getInputDimMasks)
     module.setInputScales(inputScales.toArray)
     module.setOutputDimMask(protobufModel.getOutputDimMasks)
     module.setOutputScales(outputScales.toArray)
     module.setWeightDimMask(protobufModel.getWeightDimMasks)
     module.setWeightScales(weightScales.toArray)
  }

  /**
   * Convert Attr Value object to Array of Float
   * @param attr
   * @return Array[Float]
   */
  protected def attrValueToFloatArray(attr: AttrValue): Array[Float] = {
    attr.getArrayValue.getFltList.asScala.toArray.map(_.asInstanceOf[Float])
  }

  private def copyParameters2BigDL[T: ClassTag]
    (context: DeserializeContext, module : ModuleData[T])
    (implicit ev: TensorNumeric[T]): Unit = {

    val serializedParameters = context.bigdlModule.getParametersList.asScala.toArray

    val arrayValue = ArrayValue.newBuilder
    arrayValue.setDatatype(DataType.TENSOR)
    serializedParameters.foreach(param => arrayValue.addTensor(param))
    arrayValue.setSize(serializedParameters.length)
    val attrValue = AttrValue.newBuilder
    attrValue.setArrayValue(arrayValue.build)
    attrValue.setDataType(DataType.ARRAY_VALUE)
    val convertedParameters = DataConverter.getAttributeValue(context, attrValue.build).
      asInstanceOf[Array[Tensor[T]]]

    val parameters = module.module.parameters()._1

    var i = 0
    while (i < parameters.length) {
      parameters(i).copy(convertedParameters(i))
      i += 1
    }
  }




  // to keep compatible with models saved by release <= 0.5.0
  private def copyWeightAndBias[T: ClassTag](context: DeserializeContext, module : ModuleData[T])
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
   * Serialize and save MKL DNN INT8 attributes into BigDL Model of protobuf definition
   * @param module
   * @param modelBuilder serialized module builder
   */
  protected def saveMklInt8Attr[T: ClassTag](module : MklInt8Convertible,
                                             modelBuilder : BigDLModule.Builder)
                                            (implicit ev : TensorNumeric[T]) : Unit = {
    // Save scale and mask of input into BigDL model builder
    val inputScales : Array[Array[Float]] = module.getInputScales()
    val inputMasks : Int = module.getInputDimMask()
    val inputScalesAttrList = inputScales.map(floatArrayToAttrValue)
    modelBuilder.addAllInputScales(inputScalesAttrList.toIterable.asJava)
    modelBuilder.setInputDimMasks(inputMasks)

    // Save scale and mask of output into BigDL model builder
    val outputScales : Array[Array[Float]] = module.getOutputScales()
    val outputMasks : Int = module.getOutputDimMask()
    val outputScalesAttrList = outputScales.map(floatArrayToAttrValue)
    modelBuilder.addAllOutputScales(outputScalesAttrList.toIterable.asJava)
    modelBuilder.setOutputDimMasks(outputMasks)

    // Save scale and mask of weight into BigDL model builder
    val weightScales: Array[Array[Float]] = module.getWeightScales()
    val weightMasks: Int = module.getWeightDimMask()
    val weightScalesAttrList = weightScales.map(floatArrayToAttrValue)
    modelBuilder.addAllWeightScales(weightScalesAttrList.toIterable.asJava)
    modelBuilder.setWeightDimMasks(weightMasks)
  }


  /**
   * Convert an array of float into an attr value object
   * @param arry
   * @return AttrValue
   */
  private def floatArrayToAttrValue(arry : Array[Float]) : AttrValue = {
    val tempAttrValBuilder = AttrValue.newBuilder()
    tempAttrValBuilder.setDataType(DataType.ARRAY_VALUE)

    val tempArryValBuilder = ArrayValue.newBuilder()
    tempArryValBuilder.setSize(arry.length)
    tempArryValBuilder.setDatatype(DataType.FLOAT)

    arry.foreach(tempArryValBuilder.addFlt)
    tempAttrValBuilder.setArrayValue(tempArryValBuilder).build()
  }

  /**
   * copy BigDL module data (weight and bias if exist) to BigDL Model to be persisted
   * @param modelBuilder serialized module builder
   * @param context  serialization context
   */
  protected def copyFromBigDL[T: ClassTag](context : SerializeContext[T],
    modelBuilder : BigDLModule.Builder)(implicit ev : TensorNumeric[T]) : Unit = {
    val parameters = context.moduleData.module.parameters
    if (parameters != null && parameters._1 != null) {
      modelBuilder.setHasParameters(true)
      parameters._1.foreach(parameter => {
        val tensorAttr = AttrValue.newBuilder
        TensorConverter.setAttributeValue(context, tensorAttr, parameter)
        modelBuilder.addParameters(tensorAttr.getTensorValue)
      })
    }
  }

}

trait ContainerSerializable extends ModuleSerializable {

  protected def loadSubModules[T: ClassTag](context : DeserializeContext,
                                            module : AbstractModule[Activity, Activity, T])
    (implicit ev: TensorNumeric[T]) : Unit = {
    val container = module.asInstanceOf[Container[Activity, Activity, T]]
    val subModules = context.bigdlModule.getSubModulesList.asScala
    subModules.foreach(module => {
      val subModuleData = ModuleSerializer.load(DeserializeContext(module,
        context.storages, context.storageType, _copyWeightAndBias))
      container.modules.append(subModuleData.module)
    })
  }

  override def doLoadModule[T: ClassTag](context : DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val module = super.doLoadModule(context)
    loadSubModules(context, module)
    module
  }

  protected def serializeSubModules[T: ClassTag](context: SerializeContext[T],
                                                 containerBuilder : BigDLModule.Builder)
                                                (implicit ev: TensorNumeric[T]) : Unit = {
    val subModulesData = context.moduleData.module.
      asInstanceOf[Container[Activity, Activity, T]].modules
    subModulesData.foreach(module => {
      val subModule = ModuleSerializer.serialize(SerializeContext(ModuleData(module,
        new ArrayBuffer[String](), new ArrayBuffer[String]()), context.storages,
        context.storageType, _copyWeightAndBias))
      containerBuilder.addSubModules(subModule.bigDLModule)
    })
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              containerBuilder : BigDLModule.Builder)
                                           (implicit ev: TensorNumeric[T]) : Unit = {

    super.doSerializeModule(context, containerBuilder)
    serializeSubModules(context, containerBuilder)
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
