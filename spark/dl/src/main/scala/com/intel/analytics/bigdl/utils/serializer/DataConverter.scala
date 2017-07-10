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

import scala.collection.JavaConverters._
import scala.reflect.runtime.universe
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.{L1L2Regularizer, L1Regularizer, L2Regularizer, Regularizer}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import serialization.Model._
import serialization.Model.AttrValue.{ArrayValue, DataType, ListValue}

import scala.collection.mutable
import scala.reflect.ClassTag

/**
 * Trait which defines get attribute value from saved protobuf data and convert BigDL object to
 * protobuf format attribute data
 */
trait DataConverter {

/**
 * Get attribute value from protobuf attribute data
 * @tparam T data type
 * @param attribute  protobuf generated Attribute instance
 * @return BigDL compatible param value
 */
  def getAttributeValue[T : ClassTag](attribute: AttrValue)(
    implicit ev: TensorNumeric[T]) : AnyRef

/**
 * Set attribute value to protobuf format
 * @tparam T data type
 * @param attributeBuilder  the attribute value writable instance
 * @param value the value to be written to protobuf file
 * @param valueType the type of the value to help set the data type
 */
  def setAttributeValue[T : ClassTag](attributeBuilder : AttrValue.Builder, value: Any,
                                      valueType : universe.Type = null)
    (implicit ev: TensorNumeric[T]) : Unit
}

/**
 * General implementation of [[DataConverter]], it provides the conversion entry for all types
 */
object DataConverter extends DataConverter{

  val typePlaceHolder = universe.typeOf[DataConverter]

  private def getTypeTag[T : universe.TypeTag](a : T) : universe.TypeTag[T] = universe.typeTag[T]

  override def getAttributeValue[T : ClassTag](attribute: AttrValue)
    (implicit ev: TensorNumeric[T]) : AnyRef = {
    attribute.getDataType match {
      case DataType.INT32 => Integer.valueOf(attribute.getInt32Value)
      case DataType.INT64 => Long.box(attribute.getInt64Value)
      case DataType.DOUBLE => Double.box(attribute.getDoubleValue)
      case DataType.FLOAT => Float.box(attribute.getFloatValue)
      case DataType.STRING => attribute.getStringValue
      case DataType.BOOL => Boolean.box(attribute.getBoolValue)
      case DataType.REGULARIZER => RegularizerConverter.getAttributeValue(attribute)
      case DataType.TENSOR => TensorConverter.getAttributeValue(attribute)
      case DataType.VARIABLE_FORMAT => VariableFormatConverter.getAttributeValue(attribute)
      case DataType.INITMETHOD => InitMethodConverter.getAttributeValue(attribute)
      case DataType.MODULE => ModuleConverter.getAttributeValue(attribute)
      case DataType.NAME_ATTR_LIST => NameListConverter.getAttributeValue(attribute)
      case DataType.LIST_VALUE => ListConverter.getAttributeValue(attribute)
      case DataType.ARRAY_VALUE => ArrayConverter.getAttributeValue(attribute)
      case _ => throw new IllegalArgumentException
        (s"${attribute.getDataType} can not be recognized")
    }
  }

  override def setAttributeValue[T : ClassTag](
    attributeBuilder : AttrValue.Builder, value: Any, valueType : universe.Type = typePlaceHolder)
    (implicit ev: TensorNumeric[T]): Unit = {
    if (valueType == universe.typeOf[Int]) {
      attributeBuilder.setDataType(DataType.INT32)
      attributeBuilder.setInt32Value(value.asInstanceOf[Int])
    } else if (valueType == universe.typeOf[Long]) {
      attributeBuilder.setDataType(DataType.INT64)
      attributeBuilder.setInt64Value(value.asInstanceOf[Long])
    } else if (valueType == universe.typeOf[Float]) {
      attributeBuilder.setDataType(DataType.FLOAT)
      attributeBuilder.setFloatValue(value.asInstanceOf[Float])
    } else if (valueType == universe.typeOf[Double]) {
      attributeBuilder.setDataType(DataType.DOUBLE)
      attributeBuilder.setDoubleValue(value.asInstanceOf[Double])
    } else if (valueType == universe.typeOf[String]) {
      attributeBuilder.setDataType(DataType.STRING)
      attributeBuilder.setStringValue(value.asInstanceOf[String])
    } else if (valueType == universe.typeOf[Boolean]) {
      attributeBuilder.setDataType(DataType.BOOL)
      attributeBuilder.setBoolValue(value.asInstanceOf[Boolean])
    } else if (valueType == universe.typeOf[VariableFormat]) {
      VariableFormatConverter.setAttributeValue(attributeBuilder, value)
    } else if (valueType == universe.typeOf[InitializationMethod]) {
      InitMethodConverter.setAttributeValue(attributeBuilder, value)
    } else if (valueType.toString == ModuleSerializer.regularizerType.toString) {
      RegularizerConverter.setAttributeValue(attributeBuilder, value)
    } else if (valueType.toString == ModuleSerializer.tensorType.toString) {
      TensorConverter.setAttributeValue(attributeBuilder, value)
    } else if (valueType.toString == ModuleSerializer.tType.toString) {
      if (ev == com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericDouble) {
        attributeBuilder.setDataType(DataType.DOUBLE)
        attributeBuilder.setDoubleValue(value.asInstanceOf[Double])
      } else {
        attributeBuilder.setDataType(DataType.FLOAT)
        attributeBuilder.setFloatValue(value.asInstanceOf[Float])
      }
    } else if (valueType.toString == ModuleSerializer.abstractModuleType.toString
      || valueType.toString == ModuleSerializer.tensorModuleType.toString
      || valueType.toString.startsWith("com.intel.analytics.bigdl.Module")
      || valueType.toString.startsWith("com.intel.analytics.bigdl.nn.abstractnn.AbstractModule")) {
      ModuleConverter.setAttributeValue(attributeBuilder, value)
    } else if (value.isInstanceOf[Map[String, _ <: Any]]) {
      NameListConverter.setAttributeValue(attributeBuilder, value)
    } else if (value.isInstanceOf[List[_ <: Any]]) {
      ListConverter.setAttributeValue(attributeBuilder, value)
    } else if (value.isInstanceOf[Array[_ <: Any]]) {
      ArrayConverter.setAttributeValue(attributeBuilder, value)
    }
  }

/**
 * DataConverter for [[com.intel.analytics.bigdl.optim.Regularizer]]
 */
  object RegularizerConverter extends DataConverter {

    override def getAttributeValue[T : ClassTag](attribute: AttrValue)
      (implicit ev: TensorNumeric[T]): AnyRef = {
      val regularizer = attribute.getRegularizerValue
      val regularizerType = regularizer.getRegularizerType
      if (regularizer.getRegularDataCount == 0) {
        return null
      }
      val l1 = regularizer.getRegularDataList.get(0)
      val l2 = regularizer.getRegularDataList.get(1)
      regularizerType match {
        case serialization.Model.RegularizerType.L1Regularizer => L1Regularizer[T](l1)
        case serialization.Model.RegularizerType.L2Regularizer => L2Regularizer[T](l2)
        case serialization.Model.RegularizerType.L1L2Regularizer => L1L2Regularizer[T](l1, l2)
      }
    }

    override def setAttributeValue[T : ClassTag]
    (attributeBuilder: AttrValue.Builder, value: Any,
     valueType : universe.Type = null)
    (implicit ev: TensorNumeric[T]): Unit = {
      attributeBuilder.setDataType(DataType.REGULARIZER)
      if (value != null) {
        var regularizerBuilder = serialization.Model.Regularizer.newBuilder
        val regularizer = value.asInstanceOf[L1L2Regularizer[T]]
        val l1 = regularizer.l1
        val l2 = regularizer.l2
        regularizerBuilder.addRegularData(l1)
        regularizerBuilder.addRegularData(l2)
        val regularizerType = regularizer match {
          case l1: L1Regularizer[_] => serialization.Model.RegularizerType.L1Regularizer
          case l2: L2Regularizer[_] => serialization.Model.RegularizerType.L2Regularizer
          case l1l2: L1L2Regularizer[_] => serialization.Model.RegularizerType.L1L2Regularizer
        }
        regularizerBuilder.setRegularizerType(regularizerType)
        attributeBuilder.setRegularizerValue(regularizerBuilder.build)
      }
    }

  }

/**
 * DataConverter for [[com.intel.analytics.bigdl.tensor.Tensor]]
 */
  object TensorConverter extends DataConverter {

    override def getAttributeValue[T: ClassTag](attribute: AttrValue)
      (implicit ev: TensorNumeric[T]): AnyRef = {
      val serializedTensor = attribute.getTensorValue
      val data = serializedTensor.getDataList.asScala
      if (data.size == 0) {
        return null
      }
      val sizes = serializedTensor.getSizeList.asScala
      val strorageArray = new Array[T](data.size)
      var i = 0;
      while (i < data.size) {
        strorageArray(i) = ev.fromType[Double](data(i))
        i += 1
      }
      val sizeArray = new Array[Int](sizes.size)
      i = 0;
      while (i < sizes.size) {
        sizeArray(i) = sizes(i)
        i += 1
      }
      Tensor[T](strorageArray, sizeArray)
    }

    override def setAttributeValue[T: ClassTag]
      (attributeBuilder: AttrValue.Builder, value: Any,
       valueType : universe.Type = null)
      (implicit ev: TensorNumeric[T]): Unit = {
      attributeBuilder.setDataType(DataType.TENSOR)
      if (value != null) {
        val tensor = value.asInstanceOf[Tensor[T]]
        val tensorBuilder = BigDLTensor.newBuilder
        tensor.storage().array().foreach(data => tensorBuilder.addData(ev.toType[Double](data)))
        tensor.size().foreach(size => tensorBuilder.addSize(size))
        attributeBuilder.setTensorValue(tensorBuilder.build)
      }
    }

  }

/**
 * DataConverter for [[com.intel.analytics.bigdl.nn.VariableFormat]]
 */
  object  VariableFormatConverter extends DataConverter {

    override def getAttributeValue[T: ClassTag](attribute: AttrValue)
      (implicit ev: TensorNumeric[T]): AnyRef = {
      val format = attribute.getVariableFormatValue
      format match {
        case ValFormat.DEFAULT => VariableFormat.Default
        case ValFormat.ONE_D => VariableFormat.ONE_D
        case ValFormat.IN_OUT => VariableFormat.IN_OUT
        case ValFormat.OUT_IN => VariableFormat.OUT_IN
        case ValFormat.IN_OUT_KW_KH => VariableFormat.IN_OUT_KW_KH
        case ValFormat.OUT_IN_KW_KH => VariableFormat.OUT_IN_KW_KH
        case ValFormat.GP_OUT_IN_KW_KH => VariableFormat.GP_OUT_IN_KW_KH
        case ValFormat.GP_IN_OUT_KW_KH => VariableFormat.GP_IN_OUT_KW_KH
        case ValFormat.OUT_IN_KT_KH_KW => VariableFormat.OUT_IN_KT_KH_KW
        case ValFormat.EMPTY_FORMAT => null
      }
    }

    override def setAttributeValue[T: ClassTag](attributeBuilder: AttrValue.Builder,
      value: Any, valueType: universe.Type = null)(implicit ev: TensorNumeric[T]): Unit = {
      attributeBuilder.setDataType(DataType.VARIABLE_FORMAT)
      if (value != null) {
        val format = value.asInstanceOf[VariableFormat]
        val formatValue = format match {
          case VariableFormat.Default => ValFormat.DEFAULT
          case VariableFormat.ONE_D => ValFormat.ONE_D
          case VariableFormat.IN_OUT => ValFormat.IN_OUT
          case VariableFormat.OUT_IN => ValFormat.OUT_IN
          case VariableFormat.IN_OUT_KW_KH => ValFormat.IN_OUT_KW_KH
          case VariableFormat.OUT_IN_KW_KH => ValFormat.OUT_IN_KW_KH
          case VariableFormat.GP_OUT_IN_KW_KH => ValFormat.GP_OUT_IN_KW_KH
          case VariableFormat.GP_IN_OUT_KW_KH => ValFormat.GP_IN_OUT_KW_KH
          case VariableFormat.OUT_IN_KT_KH_KW => ValFormat.OUT_IN_KT_KH_KW
        }
        attributeBuilder.setVariableFormatValue(formatValue)
      } else {
        attributeBuilder.setVariableFormatValue(ValFormat.EMPTY_FORMAT)
      }
   }
}
/**
 * DataConverter for [[com.intel.analytics.bigdl.nn.InitializationMethod]]
 */
  object InitMethodConverter extends DataConverter {

    override def getAttributeValue[T: ClassTag](attribute: AttrValue)
      (implicit ev: TensorNumeric[T]): AnyRef = {
      val initMemethod = attribute.getInitMethodValue
      val initType = initMemethod.getMethodType
      val methodData = initMemethod.getDataList
      initType match {
        case InitMethodType.RANDOM_UNIFORM => RandomUniform
        case InitMethodType.RANDOM_UNIFORM_PARAM =>
          RandomUniform(methodData.get(0), methodData.get(1))
        case InitMethodType.RANDOM_NORMAL =>
          RandomNormal(methodData.get(0), methodData.get(1))
        case InitMethodType.ZEROS => Zeros
        case InitMethodType.ONES => Ones
        case InitMethodType.CONST => ConstInitMethod(methodData.get(0))
        case InitMethodType.XAVIER => Xavier
        case InitMethodType.BILINEARFILLER => BilinearFiller
        case InitMethodType.EMPTY_INITIALIZATION => null
      }
   }

    override def setAttributeValue[T: ClassTag](attributeBuilder: AttrValue.Builder,
      value: Any, valueType: universe.Type = null)(implicit ev: TensorNumeric[T]): Unit = {
      attributeBuilder.setDataType(DataType.INITMETHOD)
      val initMethodBuilder = InitMethod.newBuilder
      if (value != null) {
        val initMethod = value.asInstanceOf[InitializationMethod]
        initMethod match {
          case RandomUniform =>
            initMethodBuilder.setMethodType(InitMethodType.RANDOM_UNIFORM)
          case ru: RandomUniform =>
            initMethodBuilder.setMethodType(InitMethodType.RANDOM_UNIFORM_PARAM)
            initMethodBuilder.addData(ru.lower)
            initMethodBuilder.addData(ru.upper)
          case rm: RandomNormal =>
            initMethodBuilder.setMethodType(InitMethodType.RANDOM_NORMAL)
            initMethodBuilder.addData(rm.mean)
            initMethodBuilder.addData(rm.stdv)
          case Zeros =>
            initMethodBuilder.setMethodType(InitMethodType.ZEROS)
          case Ones =>
            initMethodBuilder.setMethodType(InitMethodType.ONES)
          case const: ConstInitMethod =>
            initMethodBuilder.setMethodType(InitMethodType.CONST)
            initMethodBuilder.addData(const.value)
          case Xavier =>
            initMethodBuilder.setMethodType(InitMethodType.XAVIER)
          case BilinearFiller =>
            initMethodBuilder.setMethodType(InitMethodType.BILINEARFILLER)
        }
        attributeBuilder.setInitMethodValue(initMethodBuilder.build)
      } else {
        initMethodBuilder.setMethodType(InitMethodType.EMPTY_INITIALIZATION)
        attributeBuilder.setInitMethodValue(initMethodBuilder.build)
      }
    }
  }

/**
 * DataConverter for [[com.intel.analytics.bigdl.nn.abstractnn.AbstractModule]]
 */
  object ModuleConverter extends DataConverter {

    override def getAttributeValue[T: ClassTag](attribute: AttrValue)
     (implicit ev: TensorNumeric[T]): AnyRef = {
     val serializedModule = attribute.getBigDLModuleValue
      if (serializedModule.getModuleType != null && serializedModule.getModuleType != "") {
        ModuleSerializer.load(serializedModule).module
      } else {
        null
      }
   }

    override def setAttributeValue[T: ClassTag](attributeBuilder: AttrValue.Builder,
      value: Any, valueType: universe.Type = null)(implicit ev: TensorNumeric[T]): Unit = {
      attributeBuilder.setDataType(DataType.MODULE)
      if (value != null) {
        val module = value.asInstanceOf[AbstractModule[Activity, Activity, T]]
        val serializableModule = ModuleSerializer.
          serialize(ModuleData(module, Seq[String](), Seq[String]()))
        attributeBuilder.setBigDLModuleValue(serializableModule)
      }
    }
  }

/**
 * DataConverter for name list
 */
  object NameListConverter extends DataConverter {

    override def getAttributeValue[T: ClassTag]
      (attribute: AttrValue)(implicit ev: TensorNumeric[T]): AnyRef = {
      val nameListMap = new mutable.HashMap[String, mutable.Map[String, Any]]()
      val listMap = new mutable.HashMap[String, Any]()
      val nameAttrListValue = attribute.getNameAttrListValue
      val listName = nameAttrListValue.getName
      nameAttrListValue.getAttrMap.asScala.foreach(attributePair => {
        val name = attributePair._1
        val attrValue = attributePair._2
        val convetedObj = DataConverter.getAttributeValue(attrValue)
        listMap(name) = convetedObj
      })
      nameListMap(listName) = listMap
      nameListMap
    }

    override def setAttributeValue[T: ClassTag](attributeBuilder: AttrValue.Builder,
      value: Any, valueType: universe.Type = null)(implicit ev: TensorNumeric[T]): Unit = {
      attributeBuilder.setDataType(DataType.NAME_ATTR_LIST)
      val listMap = value.asInstanceOf[Map[String, mutable.Map[String, Any]]]
      val (name, nameListMap) = listMap.head
      val nameAttrList = NameAttrList.newBuilder
      nameAttrList.setName(name)
      nameListMap.foreach(attributePair => {
        val name = attributePair._1
        val obj = attributePair._2
        val nextedAttr = AttrValue.newBuilder
        DataConverter.setAttributeValue(nextedAttr, obj)
        nameAttrList.putAttr(name, nextedAttr.build)
      })
      attributeBuilder.setNameAttrListValue(nameAttrList.build)
    }

  }


/**
 * DataConvert for list container, it's different from List[AttrValue]
 * it's a list of specific type value
 * For each specific type, wrapper it as corresponding attribute and call related converter
 */
  object ListConverter extends DataConverter {

    override def getAttributeValue[T: ClassTag]
   (attribute: AttrValue)(implicit ev: TensorNumeric[T]): AnyRef = {
      val list = List[Any]()
      val valueList = attribute.getListValue
      val listType = valueList.getDatatype
      listType match {
        case DataType.INT32 =>
          list ++ valueList.getI32List.asScala
        case DataType.INT64 =>
          list ++ valueList.getI64List.asScala
       case DataType.DOUBLE =>
          list ++ valueList.getDblList.asScala
        case DataType.FLOAT =>
          list ++ valueList.getFltList.asScala
        case DataType.STRING =>
          list ++ valueList.getStrList.asScala
        case DataType.BOOL =>
          list ++ valueList.getBooleanList.asScala
        case DataType.REGULARIZER =>
          val regList = valueList.getRegularizerList.asScala
          regList.foreach(reg => {
            val attrValue = AttrValue.newBuilder
            attrValue.setDataType(DataType.REGULARIZER)
           attrValue.setRegularizerValue(reg)
            list :+ RegularizerConverter.getAttributeValue(attrValue.build)
         })
        case DataType.TENSOR =>
          val tensorList = valueList.getTensorList.asScala
          tensorList.foreach(tensor => {
            val attrValue = AttrValue.newBuilder
           attrValue.setDataType(DataType.TENSOR)
            attrValue.setTensorValue(tensor)
            list :+ TensorConverter.getAttributeValue(attrValue.build)
          })
        case DataType.VARIABLE_FORMAT =>
          val formatList = valueList.getVariableFormatList.asScala
          formatList.foreach(format => {
            val attrValue = AttrValue.newBuilder
            attrValue.setDataType(DataType.VARIABLE_FORMAT)
            attrValue.setVariableFormatValue(format)
            list :+ VariableFormatConverter.getAttributeValue(attrValue.build)
          })
        case DataType.INITMETHOD =>
          val methodList = valueList.getInitMethodList.asScala
         methodList.foreach(method => {
            val attrValue = AttrValue.newBuilder
            attrValue.setDataType(DataType.INITMETHOD)
            attrValue.setInitMethodValue(method)
            list :+ InitMethodConverter.getAttributeValue(attrValue.build)
          })
        case DataType.MODULE =>
          val moduleList = valueList.getBigDLModuleList.asScala
          moduleList.foreach(module => {
            val attrValue = AttrValue.newBuilder
            attrValue.setDataType(DataType.MODULE)
            attrValue.setBigDLModuleValue(module)
            list :+ ModuleConverter.getAttributeValue(attrValue.build)
          })
        case DataType.NAME_ATTR_LIST =>
          val nameAttriLists = valueList.getNameAttrListList.asScala
          nameAttriLists.foreach(nameList => {
            val attrValue = AttrValue.newBuilder
            attrValue.setDataType(DataType.NAME_ATTR_LIST)
            attrValue.setNameAttrListValue(nameList)
            list :+ NameListConverter.getAttributeValue(attrValue.build)
          })
      }
     list
    }

    override def setAttributeValue[T: ClassTag](attributeBuilder: AttrValue.Builder,
      value: Any, valueType: universe.Type = null)(implicit ev: TensorNumeric[T]): Unit = {
      attributeBuilder.setDataType(DataType.LIST_VALUE)
      val listBuilder = ListValue.newBuilder
      if (value.isInstanceOf[List[Int]]) {
        listBuilder.setDatatype(DataType.INT32)
        value.asInstanceOf[List[Int]].foreach(_ => listBuilder.addI32(_))
      } else if (value.isInstanceOf[List[Long]]) {
       listBuilder.setDatatype(DataType.INT64)
       value.asInstanceOf[List[Long]].foreach(_ => listBuilder.addI64(_))
      } else if (value.isInstanceOf[List[Float]]) {
        listBuilder.setDatatype(DataType.FLOAT)
        value.asInstanceOf[List[Float]].foreach(_ => listBuilder.addFlt(_))
      } else if (value.isInstanceOf[List[Double]]) {
        listBuilder.setDatatype(DataType.DOUBLE)
        value.asInstanceOf[List[Double]].foreach(_ => listBuilder.addDbl(_))
      } else if (value.isInstanceOf[List[Boolean]]) {
        listBuilder.setDatatype(DataType.BOOL)
        value.asInstanceOf[List[Boolean]].foreach(_ => listBuilder.addBoolean(_))
      } else if (value.isInstanceOf[List[String]]) {
        listBuilder.setDatatype(DataType.STRING)
        value.asInstanceOf[List[String]].foreach(_ => listBuilder.addStr(_))
      } else if (value.isInstanceOf[List[Regularizer[T]]]) {
        listBuilder.setDatatype(DataType.REGULARIZER)
        value.asInstanceOf[List[Regularizer[T]]].foreach(reg => {
          val attrValueBuilder = AttrValue.newBuilder
          RegularizerConverter.setAttributeValue(attrValueBuilder, reg)
          listBuilder.addRegularizer(attrValueBuilder.getRegularizerValue)
        })
      } else if (value.isInstanceOf[List[Tensor[T]]]) {
        listBuilder.setDatatype(DataType.TENSOR)
        value.asInstanceOf[List[Tensor[T]]].foreach(tensor => {
          val attrValueBuilder = AttrValue.newBuilder
          TensorConverter.setAttributeValue(attrValueBuilder, tensor)
          listBuilder.addTensor(attrValueBuilder.getTensorValue)
        })
      } else if (value.isInstanceOf[List[VariableFormat]]) {
        listBuilder.setDatatype(DataType.VARIABLE_FORMAT)
        value.asInstanceOf[List[VariableFormat]].foreach(format => {
          val attrValueBuilder = AttrValue.newBuilder
          VariableFormatConverter.setAttributeValue(attrValueBuilder, format)
          listBuilder.addVariableFormat(attrValueBuilder.getVariableFormatValue)
        })
      } else if (value.isInstanceOf[List[InitializationMethod]]) {
        listBuilder.setDatatype(DataType.INITMETHOD)
        value.asInstanceOf[List[InitializationMethod]].foreach(method => {
          val attrValueBuilder = AttrValue.newBuilder
          InitMethodConverter.setAttributeValue(attrValueBuilder, method)
          listBuilder.addInitMethod(attrValueBuilder.getInitMethodValue)
        })
      } else if (value.isInstanceOf[List[_ <: AbstractModule[Activity, Activity, T]]]) {
        listBuilder.setDatatype(DataType.INITMETHOD)
        value.asInstanceOf[List[_ <: AbstractModule[Activity, Activity, T]]].foreach(module => {
          val attrValueBuilder = AttrValue.newBuilder
          ModuleConverter.setAttributeValue(attrValueBuilder, module)
          listBuilder.addBigDLModule(attrValueBuilder.getBigDLModuleValue)
        })
      } else if (value.isInstanceOf[List[Map[String, Any]]]) {
        listBuilder.setDatatype(DataType.NAME_ATTR_LIST)
        value.asInstanceOf[List[Map[String, Any]]].foreach(map => {
          val attrValueBuilder = AttrValue.newBuilder
          NameListConverter.setAttributeValue(attrValueBuilder, map)
          listBuilder.addNameAttrList(attrValueBuilder.getNameAttrListValue)
        })
      }
    }
  }
  /**
   * DataConvert for array container, it's different from Array[AttrValue]
   * it's an array  of specific type value
   * For each specific type, wrapper it as corresponding attribute and call related converter
   */
  object ArrayConverter extends DataConverter {

    override def getAttributeValue[T: ClassTag]
    (attribute: AttrValue)(implicit ev: TensorNumeric[T]): AnyRef = {
      val valueArray = attribute.getArrayValue
      val size = valueArray.getSize
      val listType = valueArray.getDatatype
      val arr = listType match {
        case DataType.INT32 =>
          val int32s = new Array[Int](size)
          var i = 0
          valueArray.getI32List.asScala.toArray.foreach(v => {
            int32s(i) = v.toInt
            i += 1
          })
          int32s
        case DataType.INT64 =>
          val int64s = new Array[Long](size)
          var i = 0
          valueArray.getI64List.asScala.toArray.foreach(v => {
            int64s(i) = v.toLong
            i += 1
          })
          int64s
        case DataType.DOUBLE =>
          val dbls = Array[Double](size)
          dbls ++ valueArray.getDblList.asScala
          dbls
        case DataType.FLOAT =>
          val flts = Array[Float](size)
          flts ++ valueArray.getFltList.asScala
          flts
        case DataType.STRING =>
          val strs = new Array[String](size)
          strs ++ valueArray.getStrList.asScala
          strs
        case DataType.BOOL =>
          val bools = new Array[Boolean](size)
          bools ++ valueArray.getBooleanList.asScala
          bools
        case DataType.REGULARIZER =>
          val regularizers = new Array[Regularizer[T]](size)
          val regList = valueArray.getRegularizerList.asScala
          regList.foreach(reg => {
            val attrValue = AttrValue.newBuilder
            attrValue.setDataType(DataType.REGULARIZER)
            attrValue.setRegularizerValue(reg)
            regularizers :+ RegularizerConverter.getAttributeValue(attrValue.build)
          })
          regularizers
        case DataType.TENSOR =>
          val tensors = new Array[Tensor[T]](size)
          val tensorList = valueArray.getTensorList.asScala
          tensorList.foreach(tensor => {
            val attrValue = AttrValue.newBuilder
            attrValue.setDataType(DataType.TENSOR)
            attrValue.setTensorValue(tensor)
            tensors :+ TensorConverter.getAttributeValue(attrValue.build)
          })
          tensors
        case DataType.VARIABLE_FORMAT =>
          val formats = new Array[ValFormat](size)
          val formatList = valueArray.getVariableFormatList.asScala
          formatList.foreach(format => {
            val attrValue = AttrValue.newBuilder
            attrValue.setDataType(DataType.VARIABLE_FORMAT)
            attrValue.setVariableFormatValue(format)
            formats :+ VariableFormatConverter.getAttributeValue(attrValue.build)
          })
          formats
        case DataType.INITMETHOD =>
          val methods = new Array[InitializationMethod](size)
          val methodList = valueArray.getInitMethodList.asScala
          methodList.foreach(method => {
            val attrValue = AttrValue.newBuilder
            attrValue.setDataType(DataType.INITMETHOD)
            attrValue.setInitMethodValue(method)
            methods :+ InitMethodConverter.getAttributeValue(attrValue.build)
          })
          methods
        case DataType.MODULE =>
          val modules = new Array[AbstractModule[Activity, Activity, T]](size)
          val moduleList = valueArray.getBigDLModuleList.asScala
          moduleList.foreach(module => {
            val attrValue = AttrValue.newBuilder
            attrValue.setDataType(DataType.MODULE)
            attrValue.setBigDLModuleValue(module)
            modules :+ ModuleConverter.getAttributeValue(attrValue.build)
          })
          modules
        case DataType.NAME_ATTR_LIST =>
          val nameArray = new Array[Map[String, Map[String, Any]]](size)
          val nameAttriLists = valueArray.getNameAttrListList.asScala
          val i = 0
          nameAttriLists.foreach(nameList => {
            val attrValue = AttrValue.newBuilder
            attrValue.setDataType(DataType.NAME_ATTR_LIST)
            attrValue.setNameAttrListValue(nameList)
            nameArray(i) = NameListConverter.getAttributeValue(attrValue.build)
              .asInstanceOf[Map[String, Map[String, Any]]]
          })
          nameArray
      }
      arr
    }

    override def setAttributeValue[T: ClassTag](attributeBuilder: AttrValue.Builder,
      value: Any, valueType: universe.Type = null)(implicit ev: TensorNumeric[T]): Unit = {
      attributeBuilder.setDataType(DataType.ARRAY_VALUE)
      val arrayBuilder = ArrayValue.newBuilder
      if (value.isInstanceOf[Array[Int]]) {
        val int32s = value.asInstanceOf[Array[Int]]
        arrayBuilder.setDatatype(DataType.INT32)
        int32s.foreach(i32 => arrayBuilder.addI32(i32))
        arrayBuilder.setSize(int32s.size)
      } else if (value.isInstanceOf[Array[Long]]) {
        val int64s = value.asInstanceOf[Array[Long]]
        arrayBuilder.setDatatype(DataType.INT64)
        int64s.foreach(i64 => arrayBuilder.addI64(i64))
        arrayBuilder.setSize(int64s.size)
      } else if (value.isInstanceOf[Array[Float]]) {
        arrayBuilder.setDatatype(DataType.FLOAT)
        value.asInstanceOf[Array[Float]].foreach(_ => arrayBuilder.addFlt(_))
      } else if (value.isInstanceOf[Array[Double]]) {
        arrayBuilder.setDatatype(DataType.DOUBLE)
        value.asInstanceOf[Array[Double]].foreach(_ => arrayBuilder.addDbl(_))
      } else if (value.isInstanceOf[Array[Boolean]]) {
        arrayBuilder.setDatatype(DataType.BOOL)
        value.asInstanceOf[Array[Boolean]].foreach(_ => arrayBuilder.addBoolean(_))
      } else if (value.isInstanceOf[Array[String]]) {
        arrayBuilder.setDatatype(DataType.STRING)
        value.asInstanceOf[Array[String]].foreach(_ => arrayBuilder.addStr(_))
      } else if (value.isInstanceOf[Array[Regularizer[T]]]) {
        arrayBuilder.setDatatype(DataType.REGULARIZER)
        value.asInstanceOf[Array[Regularizer[T]]].foreach(reg => {
          val attrValueBuilder = AttrValue.newBuilder
          RegularizerConverter.setAttributeValue(attrValueBuilder, reg)
          arrayBuilder.addRegularizer(attrValueBuilder.getRegularizerValue)
        })
      } else if (value.isInstanceOf[Array[Tensor[T]]]) {
        arrayBuilder.setDatatype(DataType.TENSOR)
        val tensors = value.asInstanceOf[Array[Tensor[T]]]
        tensors.foreach(tensor => {
          val attrValueBuilder = AttrValue.newBuilder
          TensorConverter.setAttributeValue(attrValueBuilder, tensor)
          arrayBuilder.addTensor(attrValueBuilder.getTensorValue)
        })
        arrayBuilder.setSize(tensors.size)
      } else if (value.isInstanceOf[Array[VariableFormat]]) {
        arrayBuilder.setDatatype(DataType.VARIABLE_FORMAT)
        value.asInstanceOf[Array[VariableFormat]].foreach(format => {
          val attrValueBuilder = AttrValue.newBuilder
          VariableFormatConverter.setAttributeValue(attrValueBuilder, format)
          arrayBuilder.addVariableFormat(attrValueBuilder.getVariableFormatValue)
        })
      } else if (value.isInstanceOf[Array[InitializationMethod]]) {
        arrayBuilder.setDatatype(DataType.INITMETHOD)
        value.asInstanceOf[Array[InitializationMethod]].foreach(method => {
          val attrValueBuilder = AttrValue.newBuilder
          InitMethodConverter.setAttributeValue(attrValueBuilder, method)
          arrayBuilder.addInitMethod(attrValueBuilder.getInitMethodValue)
        })
      } else if (value.isInstanceOf[Array[_ <: AbstractModule[Activity, Activity, T]]]) {
        arrayBuilder.setDatatype(DataType.INITMETHOD)
        value.asInstanceOf[Array[_ <: AbstractModule[Activity, Activity, T]]].foreach(module => {
          val attrValueBuilder = AttrValue.newBuilder
          ModuleConverter.setAttributeValue(attrValueBuilder, module)
          arrayBuilder.addBigDLModule(attrValueBuilder.getBigDLModuleValue)
        })
      } else if (value.isInstanceOf[Array[Map[String, Any]]]) {
        arrayBuilder.setDatatype(DataType.NAME_ATTR_LIST)
        value.asInstanceOf[Array[Map[String, Any]]].foreach(map => {
          val attrValueBuilder = AttrValue.newBuilder
          NameListConverter.setAttributeValue(attrValueBuilder, map)
          arrayBuilder.addNameAttrList(attrValueBuilder.getNameAttrListValue)
        })
      }
      attributeBuilder.setArrayValue(arrayBuilder.build)
    }

  }
}