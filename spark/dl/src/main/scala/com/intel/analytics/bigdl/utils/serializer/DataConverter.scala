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
import serialization.Model.AttrValue.{DataType, ListValue}

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
  def setAttributeValue[T : ClassTag](attributeBuilder : AttrValue.Builder, value: AnyRef,
                                      valueType : universe.Type = null)
    (implicit ev: TensorNumeric[T]) : Unit
}

/**
 * General implementation of [[DataConverter]], it provides the conversion entry for all types
 */
object DataConverter extends DataConverter{

  def getAttributeValue[T : ClassTag](attribute: AttrValue)
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
      case _ => throw new IllegalArgumentException
        (s"${attribute.getDataType} can not be recognized")
    }
  }

  def setAttributeValue[T : ClassTag](
    attributeBuilder : AttrValue.Builder, value: AnyRef, valueType : universe.Type)
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
    } else if (valueType == universe.typeOf[Map[String, AnyRef]]) {
      NameListConverter.setAttributeValue(attributeBuilder, value)
    } else if (valueType == universe.typeOf[List[AnyRef]]) {
      ListConverter.setAttributeValue(attributeBuilder, value)
    } else if (valueType.toString == "com.intel.analytics.bigdl.optim.Regularizer[T]") {
      RegularizerConverter.setAttributeValue(attributeBuilder, value)
    } else if (valueType.toString == "com.intel.analytics.bigdl.tensor.Tensor[T]") {
      TensorConverter.setAttributeValue(attributeBuilder, value)
    } else if (valueType.toString.
      startsWith("com.intel.analytics.bigdl.nn.abstractnn.AbstractModule")) {
      ModuleConverter.setAttributeValue(attributeBuilder, value)
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
    (attributeBuilder: AttrValue.Builder, value: AnyRef,
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
      (attributeBuilder: AttrValue.Builder, value: AnyRef,
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
      }
    }

    override def setAttributeValue[T: ClassTag](attributeBuilder: AttrValue.Builder,
      value: AnyRef, valueType: universe.Type = null)(implicit ev: TensorNumeric[T]): Unit = {
      attributeBuilder.setDataType(DataType.VARIABLE_FORMAT)
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
      }
   }

    override def setAttributeValue[T: ClassTag](attributeBuilder: AttrValue.Builder,
      value: AnyRef, valueType: universe.Type = null)(implicit ev: TensorNumeric[T]): Unit = {
      attributeBuilder.setDataType(DataType.INITMETHOD)
      val initMethodBuilder = InitMethod.newBuilder
      val initMethod = value.asInstanceOf[InitializationMethod]
      initMethod match {
        case RandomUniform =>
          initMethodBuilder.setMethodType(InitMethodType.RANDOM_UNIFORM)
        case ru : RandomUniform =>
          initMethodBuilder.setMethodType(InitMethodType.RANDOM_UNIFORM_PARAM)
          initMethodBuilder.addData(ru.lower)
          initMethodBuilder.addData(ru.upper)
        case rm : RandomNormal =>
          initMethodBuilder.setMethodType(InitMethodType.RANDOM_NORMAL)
          initMethodBuilder.addData(rm.mean)
          initMethodBuilder.addData(rm.stdv)
        case Zeros =>
          initMethodBuilder.setMethodType(InitMethodType.ZEROS)
        case Ones =>
          initMethodBuilder.setMethodType(InitMethodType.ONES)
        case const : ConstInitMethod =>
          initMethodBuilder.setMethodType(InitMethodType.CONST)
          initMethodBuilder.addData(const.value)
        case Xavier =>
          initMethodBuilder.setMethodType(InitMethodType.XAVIER)
        case BilinearFiller =>
          initMethodBuilder.setMethodType(InitMethodType.BILINEARFILLER)
      }
      attributeBuilder.setInitMethodValue(initMethodBuilder.build)
    }
  }

/**
 * DataConverter for [[com.intel.analytics.bigdl.nn.abstractnn.AbstractModule]]
 */
  object ModuleConverter extends DataConverter {

    override def getAttributeValue[T: ClassTag](attribute: AttrValue)
     (implicit ev: TensorNumeric[T]): AnyRef = {
     val serializedModule = attribute.getBigDLModleValue
     ModuleSerializer.loadModule(serializedModule).module
   }

    override def setAttributeValue[T: ClassTag](attributeBuilder: AttrValue.Builder,
      value: AnyRef, valueType: universe.Type = null)(implicit ev: TensorNumeric[T]): Unit = {
      attributeBuilder.setDataType(DataType.MODULE)
      val module = value.asInstanceOf[AbstractModule[Activity, Activity, T]]
     val serializableModule = ModuleSerializer.
        serializeModule(BigDLModule(module, Seq[String](), Seq[String]()))
      attributeBuilder.setBigDLModleValue(serializableModule)
    }
  }

/**
 * DataConverter for name list
 */
  object NameListConverter extends DataConverter {

    override def getAttributeValue[T: ClassTag]
      (attribute: AttrValue)(implicit ev: TensorNumeric[T]): AnyRef = {
      val nameListMap = new mutable.HashMap[String, mutable.Map[String, AnyRef]]()
      val listMap = new mutable.HashMap[String, AnyRef]()
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
      value: AnyRef, valueType: universe.Type = null)(implicit ev: TensorNumeric[T]): Unit = {
      attributeBuilder.setDataType(DataType.NAME_ATTR_LIST)
      val listMap = value.asInstanceOf[Map[String, mutable.Map[String, AnyRef]]]
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
      val list = List[AnyRef]()
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
          val moduleList = valueList.getBigDLModelList.asScala
          moduleList.foreach(module => {
            val attrValue = AttrValue.newBuilder
            attrValue.setDataType(DataType.MODULE)
            attrValue.setBigDLModleValue(module)
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
      value: AnyRef, valueType: universe.Type = null)(implicit ev: TensorNumeric[T]): Unit = {
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
          listBuilder.addBigDLModel(attrValueBuilder.getBigDLModleValue)
        })
      } else if (value.isInstanceOf[List[Map[String, AnyRef]]]) {
        listBuilder.setDatatype(DataType.NAME_ATTR_LIST)
        value.asInstanceOf[List[Map[String, AnyRef]]].foreach(map => {
          val attrValueBuilder = AttrValue.newBuilder
          NameListConverter.setAttributeValue(attrValueBuilder, map)
          listBuilder.addNameAttrList(attrValueBuilder.getNameAttrListValue)
        })
      }
    }
  }
}