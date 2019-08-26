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
package com.intel.analytics.bigdl.utils.serializer.converters

import com.google.protobuf.ByteString

import scala.collection.JavaConverters._
import scala.reflect.runtime.universe
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.{MultiShape, SingleShape, Shape => BigDLShape}
import com.intel.analytics.bigdl.serialization.Bigdl._
import com.intel.analytics.bigdl.serialization.Bigdl.AttrValue.ArrayValue

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
 * @param context deserialization context
 * @param attribute  protobuf generated Attribute instance
 * @return BigDL compatible param value
 */
  def getAttributeValue[T : ClassTag](context: DeserializeContext,
                                      attribute: AttrValue)(
    implicit ev: TensorNumeric[T]) : AnyRef

/**
 * Set attribute value to protobuf format
 * @tparam T data type
 * @param context serialization context
 * @param attributeBuilder  the attribute value writable instance
 * @param value the value to be written to protobuf file
 * @param valueType the type of the value to help set the data type
 */
  def setAttributeValue[T : ClassTag](context: SerializeContext[T],
                                      attributeBuilder : AttrValue.Builder, value: Any,
                                      valueType: universe.Type = null)
    (implicit ev: TensorNumeric[T]) : Unit

  protected def getLock: Object = ModuleSerializer._lock
}

/**
 * General implementation of [[DataConverter]], it provides the conversion entry for all types
 */
object DataConverter extends DataConverter{

  private val typePlaceHolder = universe.typeOf[DataConverter]

  // Customized data converter map, key is the string representation of user defined class type

  private val customizedConverter = new mutable.HashMap[String, DataConverter]

  def registerConverter(tpe : String, converter : DataConverter) : Unit = {
    require(!customizedConverter.contains(tpe), s"converter for $tpe already exists!")
    customizedConverter(tpe) = converter
  }

  private def getRuntimeType[T : ClassTag](value : Any) (implicit ev: TensorNumeric[T])
    : universe.Type = {
    getLock.synchronized {
      if (value.isInstanceOf[Tensor[_]]) {
        ModuleSerializer.tensorType
      } else if (value.isInstanceOf[AbstractModule[_, _, _]]) {
        ModuleSerializer.abstractModuleType
      } else if (value.isInstanceOf[Regularizer[_]]) {
        ModuleSerializer.regularizerType
      } else if (value.isInstanceOf[InitializationMethod]) {
        universe.typeOf[InitializationMethod]
      } else if (value.isInstanceOf[VariableFormat]) {
        universe.typeOf[VariableFormat]
      } else if (value.isInstanceOf[DataFormat]) {
        universe.typeOf[DataFormat]
      } else if (value.isInstanceOf[BigDLShape]) {
        universe.typeOf[BigDLShape]
      } else {
        val cls = value.getClass
        val runtimeMirror = universe.runtimeMirror(getClass.getClassLoader)
        val clsSymbol = runtimeMirror.classSymbol(cls)
        clsSymbol.toType
      }
    }
  }

  override def getAttributeValue[T : ClassTag](context: DeserializeContext, attribute: AttrValue)
    (implicit ev: TensorNumeric[T]) : AnyRef = {
    attribute.getDataType match {
      case DataType.INT32 => Integer.valueOf(attribute.getInt32Value)
      case DataType.INT64 => Long.box(attribute.getInt64Value)
      case DataType.DOUBLE => Double.box(attribute.getDoubleValue)
      case DataType.FLOAT => Float.box(attribute.getFloatValue)
      case DataType.STRING => attribute.getStringValue
      case DataType.BOOL => Boolean.box(attribute.getBoolValue)
      case DataType.REGULARIZER => RegularizerConverter.getAttributeValue(context, attribute)
      case DataType.TENSOR => TensorConverter.getAttributeValue(context, attribute)
      case DataType.VARIABLE_FORMAT =>
        VariableFormatConverter.getAttributeValue(context, attribute)
      case DataType.INITMETHOD => InitMethodConverter.getAttributeValue(context, attribute)
      case DataType.MODULE => ModuleConverter.getAttributeValue(context, attribute)
      case DataType.NAME_ATTR_LIST => NameListConverter.getAttributeValue(context, attribute)
      case DataType.ARRAY_VALUE => ArrayConverter.getAttributeValue(context, attribute)
      case DataType.DATA_FORMAT => DataFormatConverter.getAttributeValue(context, attribute)
      case DataType.CUSTOM => CustomConverterDelegator.getAttributeValue(context, attribute)
      case DataType.SHAPE => ShapeConverter.getAttributeValue(context, attribute)
      case _ => throw new IllegalArgumentException
        (s"${attribute.getDataType} can not be recognized")
    }
  }

  override def setAttributeValue[T : ClassTag](
    context: SerializeContext[T], attributeBuilder: AttrValue.Builder,
    value: Any, valueType : universe.Type = typePlaceHolder)
    (implicit ev: TensorNumeric[T]): Unit = {
    getLock.synchronized {
      // to make it compatible with Java types
      if (valueType =:= universe.typeOf[Int] ||
        valueType =:= universe.typeOf[java.lang.Integer]) {
        attributeBuilder.setDataType(DataType.INT32)
        attributeBuilder.setInt32Value(value.asInstanceOf[Int])
      } else if (valueType =:= universe.typeOf[Long] ||
        valueType =:= universe.typeOf[java.lang.Long]) {
        attributeBuilder.setDataType(DataType.INT64)
        attributeBuilder.setInt64Value(value.asInstanceOf[Long])
      } else if (valueType =:= universe.typeOf[Float] ||
        valueType =:= universe.typeOf[java.lang.Float]) {
        attributeBuilder.setDataType(DataType.FLOAT)
        attributeBuilder.setFloatValue(value.asInstanceOf[Float])
      } else if (valueType =:= universe.typeOf[Double] ||
        valueType =:= universe.typeOf[java.lang.Double]) {
        attributeBuilder.setDataType(DataType.DOUBLE)
        attributeBuilder.setDoubleValue(value.asInstanceOf[Double])
      } else if (valueType =:= universe.typeOf[String] ||
        valueType =:= universe.typeOf[java.lang.String]) {
        attributeBuilder.setDataType(DataType.STRING)
        attributeBuilder.setStringValue(value.asInstanceOf[String])
      } else if (valueType =:= universe.typeOf[Boolean] ||
        valueType =:= universe.typeOf[java.lang.Boolean]) {
        attributeBuilder.setDataType(DataType.BOOL)
        attributeBuilder.setBoolValue(value.asInstanceOf[Boolean])
      } else if (valueType =:= universe.typeOf[VariableFormat]) {
        VariableFormatConverter.setAttributeValue(context, attributeBuilder, value)
      } else if (valueType =:= universe.typeOf[InitializationMethod]) {
        InitMethodConverter.setAttributeValue(context, attributeBuilder, value)
      } else if (valueType.toString == ModuleSerializer.regularizerType.toString
        || valueType <:< universe.typeOf[Regularizer[_]]) {
        RegularizerConverter.setAttributeValue(context, attributeBuilder, value)
      } else if (valueType <:< universe.typeOf[Tensor[_]]) {
        TensorConverter.setAttributeValue(context, attributeBuilder, value)
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
        || valueType.toString == ModuleSerializer.moduleType.toString
        || valueType.toString == ModuleSerializer.boundedModuleType.toString
        || valueType <:< universe.typeOf[AbstractModule[_, _, _]]
      ) {
        ModuleConverter.setAttributeValue(context, attributeBuilder, value)
      } else if (value.isInstanceOf[mutable.Map[_, _]]) {
        NameListConverter.setAttributeValue(context, attributeBuilder, value)
      } else if (valueType <:< universe.typeOf[Array[_]] ||
        valueType.typeSymbol == universe.typeOf[Array[_]].typeSymbol) {
        ArrayConverter.setAttributeValue(context, attributeBuilder, value, valueType)
      } else if (valueType =:= universe.typeOf[DataFormat]) {
        DataFormatConverter.setAttributeValue(context, attributeBuilder, value)
      } else if (valueType =:= universe.typeOf[BigDLShape]) {
        ShapeConverter.setAttributeValue(context, attributeBuilder, value)
      } else {
        CustomConverterDelegator.setAttributeValue(context, attributeBuilder, value, valueType)
      }
    }
  }

/**
 * DataConverter for name list
 */
  object NameListConverter extends DataConverter {

    override def getAttributeValue[T: ClassTag]
      (context: DeserializeContext, attribute: AttrValue)(implicit ev: TensorNumeric[T]): AnyRef = {
      val nameListMap = new mutable.HashMap[String, mutable.Map[String, Any]]()
      val listMap = new mutable.HashMap[String, Any]()
      val nameAttrListValue = attribute.getNameAttrListValue
      val listName = nameAttrListValue.getName
      nameAttrListValue.getAttrMap.asScala.foreach(attributePair => {
        val name = attributePair._1
        val attrValue = attributePair._2
        val convetedObj = DataConverter.getAttributeValue(context, attrValue)
        listMap(name) = convetedObj
      })
      nameListMap(listName) = listMap
      nameListMap
    }

    override def setAttributeValue[T: ClassTag](context: SerializeContext[T],
      attributeBuilder: AttrValue.Builder,
      value: Any, valueType: universe.Type = null)(implicit ev: TensorNumeric[T]): Unit = {
      attributeBuilder.setDataType(DataType.NAME_ATTR_LIST)
      val listMap = value.asInstanceOf[mutable.Map[String, mutable.Map[String, Any]]]
      val (name, nameListMap) = listMap.head
      val nameAttrList = NameAttrList.newBuilder
      nameAttrList.setName(name)
      nameListMap.foreach(attributePair => {
        val name = attributePair._1
        val obj = attributePair._2
        val nextedAttr = AttrValue.newBuilder
        DataConverter.setAttributeValue(context, nextedAttr, obj, getRuntimeType(obj))
        nameAttrList.putAttr(name, nextedAttr.build)
      })
      attributeBuilder.setNameAttrListValue(nameAttrList.build)
    }

  }

  /**
   * DataConvert for array container, it's different from Array[AttrValue]
   * it's an array  of specific type value
   * For each specific type, wrapper it as corresponding attribute and call related converter
   */
  object ArrayConverter extends DataConverter {

    override def getAttributeValue[T: ClassTag]
    (context: DeserializeContext, attribute: AttrValue)(implicit ev: TensorNumeric[T]): AnyRef = {
      val valueArray = attribute.getArrayValue
      val size = valueArray.getSize
      if (size == -1) {
        return null
      }
      val listType = valueArray.getDatatype
      val arr = listType match {
        case DataType.INT32 =>
          if (size == 0) {
            return new Array[Int](0)
          }
          valueArray.getI32List.asScala.toArray.map(_.intValue)
        case DataType.INT64 =>
          if (size == 0) {
            return new Array[Long](0)
          }
          valueArray.getI64List.asScala.toArray.map(_.longValue())
        case DataType.DOUBLE =>
          if (size == 0) {
            return new Array[Double](0)
          }
          valueArray.getDblList.asScala.toArray.map(_.doubleValue())
        case DataType.FLOAT =>
          if (size == 0) {
            return new Array[Float](0)
          }
          valueArray.getFltList.asScala.toArray.map(_.floatValue())
        case DataType.STRING =>
          if (size == 0) {
            return new Array[String](0)
          }
          valueArray.getStrList.asScala.toArray
        case DataType.BOOL =>
          if (size == 0) {
            return new Array[Boolean](0)
          }
          valueArray.getBooleanList.asScala.toArray.map(_.booleanValue())
        case DataType.REGULARIZER =>
          val regularizers = new Array[Regularizer[T]](size)
          val regList = valueArray.getRegularizerList.asScala
          var i = 0
          regList.foreach(reg => {
            val attrValue = AttrValue.newBuilder
            attrValue.setDataType(DataType.REGULARIZER)
            attrValue.setRegularizerValue(reg)
            regularizers(i) = RegularizerConverter.
              getAttributeValue(context, attrValue.build).asInstanceOf[Regularizer[T]]
            i += 1
          })
          regularizers
        case DataType.TENSOR =>
          val tensors = new Array[Tensor[T]](size)
          val tensorList = valueArray.getTensorList.asScala
          var i = 0
          tensorList.foreach(tensor => {
            val attrValue = AttrValue.newBuilder
            attrValue.setDataType(DataType.TENSOR)
            attrValue.setTensorValue(tensor)
            tensors(i) = TensorConverter.
              getAttributeValue(context, attrValue.build).asInstanceOf[Tensor[T]]
            i += 1
          })
          tensors
        case DataType.VARIABLE_FORMAT =>
          val formats = new Array[VariableFormat](size)
          val formatList = valueArray.getVariableFormatList.asScala
          var i = 0
          formatList.foreach(format => {
            val attrValue = AttrValue.newBuilder
            attrValue.setDataType(DataType.VARIABLE_FORMAT)
            attrValue.setVariableFormatValue(format)
            formats(i) = VariableFormatConverter.
              getAttributeValue(context, attrValue.build).asInstanceOf[VariableFormat]
          })
          formats
        case DataType.INITMETHOD =>
          val methods = new Array[InitializationMethod](size)
          val methodList = valueArray.getInitMethodList.asScala
          var i = 0
          methodList.foreach(method => {
            val attrValue = AttrValue.newBuilder
            attrValue.setDataType(DataType.INITMETHOD)
            attrValue.setInitMethodValue(method)
            methods(i) = InitMethodConverter.getAttributeValue(context, attrValue.build)
            .asInstanceOf[InitializationMethod]
            i += 1
          })
          methods
        case DataType.MODULE =>
          val modules = new Array[AbstractModule[Activity, Activity, T]](size)
          val moduleList = valueArray.getBigDLModuleList.asScala
          var i = 0
          moduleList.foreach(module => {
            val attrValue = AttrValue.newBuilder
            attrValue.setDataType(DataType.MODULE)
            attrValue.setBigDLModuleValue(module)
            modules(i) = ModuleConverter.getAttributeValue(context, attrValue.build)
              .asInstanceOf[AbstractModule[Activity, Activity, T]]
            i += 1
          })
          modules
        case DataType.NAME_ATTR_LIST =>
          val nameArray = new Array[Map[String, Map[String, Any]]](size)
          val nameAttriLists = valueArray.getNameAttrListList.asScala
          var i = 0
          nameAttriLists.foreach(nameList => {
            val attrValue = AttrValue.newBuilder
            attrValue.setDataType(DataType.NAME_ATTR_LIST)
            attrValue.setNameAttrListValue(nameList)
            nameArray(i) = NameListConverter.getAttributeValue(context, attrValue.build)
              .asInstanceOf[Map[String, Map[String, Any]]]
            i += 1
          })
          nameArray
        case DataType.DATA_FORMAT =>
          val dataFormats = new Array[DataFormat](size)
          val dataFormatList = valueArray.getDataFormatList.asScala
          var i = 0
          dataFormatList.foreach(format => {
            val attrValue = AttrValue.newBuilder
            attrValue.setDataType(DataType.DATA_FORMAT)
            attrValue.setDataFormatValue(format)
            dataFormats(i) = DataFormatConverter.
              getAttributeValue(context, attrValue.build).asInstanceOf[DataFormat]
            i += 1
          })
          dataFormats
        case DataType.CUSTOM =>
          val customValues = new Array[Any](size)
          val customValueList = valueArray.getCustomList.asScala
          var i = 0
          customValueList.foreach(custom => {
            val attrValue = AttrValue.newBuilder
            attrValue.setDataType(DataType.CUSTOM)
            attrValue.setCustomValue(custom)
            customValues(i) = CustomConverterDelegator.
              getAttributeValue(context, attrValue.build)
            i += 1
          })
          customValues
        case DataType.SHAPE =>
          valueArray.getShapeList.asScala.map(shape => {
            val attrValue = AttrValue.newBuilder
            attrValue.setDataType(DataType.SHAPE)
            attrValue.setShape(shape)
            ShapeConverter.getAttributeValue(context, attrValue.build).asInstanceOf[BigDLShape]
          }).toArray

        case _ => throw new UnsupportedOperationException("Unsupported data type: " + listType)
      }
      arr
    }

    override def setAttributeValue[T: ClassTag](context: SerializeContext[T],
                                                attributeBuilder: AttrValue.Builder,
      value: Any, valueType: universe.Type = null)(implicit ev: TensorNumeric[T]): Unit = {
      attributeBuilder.setDataType(DataType.ARRAY_VALUE)
      getLock.synchronized {
        val arrayBuilder = ArrayValue.newBuilder
        arrayBuilder.setSize(-1)
        if (valueType =:= universe.typeOf[Array[Int]]) {
          arrayBuilder.setDatatype(DataType.INT32)
          if (value != null) {
            val int32s = value.asInstanceOf[Array[Int]]
            int32s.foreach(i32 => arrayBuilder.addI32(i32))
            arrayBuilder.setSize(int32s.size)
          }
        } else if (valueType =:= universe.typeOf[Array[Long]]) {
          arrayBuilder.setDatatype(DataType.INT64)
          if (value != null) {
            val int64s = value.asInstanceOf[Array[Long]]
            int64s.foreach(i64 => arrayBuilder.addI64(i64))
            arrayBuilder.setSize(int64s.size)
          }
        } else if (valueType =:= universe.typeOf[Array[Float]]) {
          arrayBuilder.setDatatype(DataType.FLOAT)
          if (value != null) {
            val flts = value.asInstanceOf[Array[Float]]
            flts.foreach(flt => arrayBuilder.addFlt(flt))
            arrayBuilder.setSize(flts.size)
          }
        } else if (valueType =:= universe.typeOf[Array[Double]]) {
          arrayBuilder.setDatatype(DataType.DOUBLE)
          if (value != null) {
            val dbs = value.asInstanceOf[Array[Double]]
            dbs.foreach(dbl => arrayBuilder.addDbl(dbl))
            arrayBuilder.setSize(dbs.size)
          }
        } else if (valueType =:= universe.typeOf[Array[Boolean]]) {
          arrayBuilder.setDatatype(DataType.BOOL)
          if (value != null) {
            val bls = value.asInstanceOf[Array[Boolean]]
            bls.foreach(bl => arrayBuilder.addBoolean(bl))
            arrayBuilder.setSize(bls.size)
          }
        } else if (valueType =:= universe.typeOf[Array[String]]) {
          arrayBuilder.setDatatype(DataType.STRING)
          if (value != null) {
            val strs = value.asInstanceOf[Array[String]]
            strs.foreach(str => arrayBuilder.addStr(str))
            arrayBuilder.setSize(strs.size)
          }
        } else if (valueType <:< universe.typeOf[Array[_ <: Regularizer[_ <: Any]]]) {
          arrayBuilder.setDatatype(DataType.REGULARIZER)
          if (value != null) {
            val regularizers = value.asInstanceOf[Array[Regularizer[T]]]
            regularizers.foreach(reg => {
              val attrValueBuilder = AttrValue.newBuilder
              RegularizerConverter.setAttributeValue(context, attrValueBuilder, reg)
              arrayBuilder.addRegularizer(attrValueBuilder.getRegularizerValue)
            })
            arrayBuilder.setSize(regularizers.size)
          }
        } else if (valueType <:< universe.
          typeOf[Array[_ <: Tensor[_ <: Any]]]) {
          arrayBuilder.setDatatype(DataType.TENSOR)
          if (value != null) {
            val tensors = value.asInstanceOf[Array[Tensor[T]]]
            tensors.foreach(tensor => {
              val attrValueBuilder = AttrValue.newBuilder
              TensorConverter.setAttributeValue(context, attrValueBuilder, tensor)
              arrayBuilder.addTensor(attrValueBuilder.getTensorValue)
            })
            arrayBuilder.setSize(tensors.size)
          }
        } else if (valueType =:= universe.typeOf[Array[VariableFormat]]) {
          arrayBuilder.setDatatype(DataType.VARIABLE_FORMAT)
          if (value != null) {
            val formats = value.asInstanceOf[Array[VariableFormat]]
            formats.foreach(format => {
              val attrValueBuilder = AttrValue.newBuilder
              VariableFormatConverter.setAttributeValue(context, attrValueBuilder, format)
              arrayBuilder.addVariableFormat(attrValueBuilder.getVariableFormatValue)
            })
            arrayBuilder.setSize(formats.size)
          }
        } else if (valueType =:= universe.typeOf[Array[InitializationMethod]]) {
          arrayBuilder.setDatatype(DataType.INITMETHOD)
          if (value != null) {
            val methods = value.asInstanceOf[Array[InitializationMethod]]
            methods.foreach(method => {
              val attrValueBuilder = AttrValue.newBuilder
              InitMethodConverter.setAttributeValue(context, attrValueBuilder, method)
              arrayBuilder.addInitMethod(attrValueBuilder.getInitMethodValue)
            })
            arrayBuilder.setSize(methods.size)
          }
        } else if (valueType <:< universe.
          typeOf[Array[_ <: AbstractModule[_ <: Activity, _ <: Activity, _ <: Any]]]) {
          arrayBuilder.setDatatype(DataType.MODULE)
          if (value != null) {
            val modules = value.asInstanceOf[Array[_ <: AbstractModule[Activity, Activity, T]]]
            modules.foreach(module => {
              val attrValueBuilder = AttrValue.newBuilder
              ModuleConverter.setAttributeValue(context, attrValueBuilder, module)
              arrayBuilder.addBigDLModule(attrValueBuilder.getBigDLModuleValue)
            })
            arrayBuilder.setSize(modules.size)
          }
        } else if (value.isInstanceOf[Array[Map[_, _]]]) {
          arrayBuilder.setDatatype(DataType.NAME_ATTR_LIST)
          value.asInstanceOf[Array[Map[String, Any]]].foreach(map => {
            val attrValueBuilder = AttrValue.newBuilder
            NameListConverter.setAttributeValue(context, attrValueBuilder, map)
            arrayBuilder.addNameAttrList(attrValueBuilder.getNameAttrListValue)
          })
        } else if (valueType =:= universe.typeOf[Array[DataFormat]]) {
          arrayBuilder.setDatatype(DataType.DATA_FORMAT)
          if (value != null) {
            val formats = value.asInstanceOf[Array[DataFormat]]
            formats.foreach(format => {
              val attrValueBuilder = AttrValue.newBuilder
              DataFormatConverter.setAttributeValue(context, attrValueBuilder, format)
              arrayBuilder.addDataFormat(attrValueBuilder.getDataFormatValue)
            })
            arrayBuilder.setSize(formats.size)
          }
        } else if (valueType =:= universe.typeOf[Array[BigDLShape]]) {
          arrayBuilder.setDatatype(DataType.SHAPE)
          if (value != null) {
            val shapes = value.asInstanceOf[Array[BigDLShape]]
            shapes.foreach(shape => {
              val attrValueBuilder = AttrValue.newBuilder
              ShapeConverter.setAttributeValue(context, attrValueBuilder, shape)
              arrayBuilder.addShape(attrValueBuilder.getShape)
            })
            arrayBuilder.setSize(shapes.size)
          }
        } else {
          arrayBuilder.setDatatype(DataType.CUSTOM)
          if (value != null) {
            val customValues = value.asInstanceOf[Array[Any]]
            customValues.foreach(custom => {
              val attrValueBuilder = AttrValue.newBuilder
              CustomConverterDelegator.setAttributeValue(context, attrValueBuilder, custom)
              arrayBuilder.addCustom(attrValueBuilder.getCustomValue)
            })
            arrayBuilder.setSize(customValues.size)
          }
        }
        attributeBuilder.setArrayValue(arrayBuilder.build)
      }
    }
  }
  /**
   * DataConvert for custom value
   */
  object CustomConverterDelegator extends DataConverter {
    override def getAttributeValue[T: ClassTag](context: DeserializeContext, attribute: AttrValue)
                                               (implicit ev: TensorNumeric[T]): AnyRef = {
      val subType = attribute.getSubType
      require(customizedConverter.contains(subType), s"unrecognized type $subType")
      val customConverter = customizedConverter.get(subType).get
      customConverter.getAttributeValue(context, attribute)
    }

    override def setAttributeValue[T: ClassTag](context: SerializeContext[T],
                                                attributeBuilder: AttrValue.Builder,
      value: Any, valueType: universe.Type)(implicit ev: TensorNumeric[T]): Unit = {
      require(customizedConverter.contains(valueType.toString), s"unrecognized type $valueType")
      val customConverter = customizedConverter.get(valueType.toString).get
      attributeBuilder.setDataType(DataType.CUSTOM)
      attributeBuilder.setSubType(valueType.toString)
      customConverter.setAttributeValue(context, attributeBuilder, value, valueType)
    }
  }
}
