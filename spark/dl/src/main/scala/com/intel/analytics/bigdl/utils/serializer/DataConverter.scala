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

import com.google.protobuf.ByteString

import scala.collection.JavaConverters._
import scala.reflect.runtime.universe
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat.{NCHW, NHWC}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.nn.quantized._
import com.intel.analytics.bigdl.optim.{L1L2Regularizer, L1Regularizer, L2Regularizer, Regularizer}
import com.intel.analytics.bigdl.tensor.{DenseType, QuantizedTensor, QuantizedType, Tensor, Storage}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.{NumericBoolean, NumericChar, NumericDouble, NumericFloat, NumericInt, NumericLong, NumericShort, NumericString}
import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString
import serialization.Bigdl._
import serialization.Bigdl.AttrValue.ArrayValue

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
    if (value.isInstanceOf[Tensor[_]]) {
      ModuleSerializer.tensorType
    } else if (value.isInstanceOf[AbstractModule[_, _, T]]) {
      ModuleSerializer.abstractModuleType
    } else if (value.isInstanceOf[Regularizer[_]]) {
      ModuleSerializer.regularizerType
    } else if (value.isInstanceOf[InitializationMethod]) {
      universe.typeOf[InitializationMethod]
    } else if (value.isInstanceOf[VariableFormat]) {
      universe.typeOf[VariableFormat]
    } else if (value.isInstanceOf[DataFormat]) {
      universe.typeOf[DataFormat]
    } else {
      val cls = value.getClass
      val runtimeMirror = universe.runtimeMirror(getClass.getClassLoader)
      val clsSymbol = runtimeMirror.classSymbol(cls)
      clsSymbol.toType
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
      case _ => throw new IllegalArgumentException
        (s"${attribute.getDataType} can not be recognized")
    }
  }

  override def setAttributeValue[T : ClassTag](
    context: SerializeContext[T], attributeBuilder: AttrValue.Builder,
    value: Any, valueType : universe.Type = typePlaceHolder)
    (implicit ev: TensorNumeric[T]): Unit = {
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
      attributeBuilder.setDataType(DataType.BOOL )
      attributeBuilder.setBoolValue(value.asInstanceOf[Boolean])
    } else if (valueType =:= universe.typeOf[VariableFormat]) {
      VariableFormatConverter.setAttributeValue(context, attributeBuilder, value)
    } else if (valueType =:= universe.typeOf[InitializationMethod]) {
      InitMethodConverter.setAttributeValue(context, attributeBuilder, value)
    } else if (valueType.toString == ModuleSerializer.regularizerType.toString) {
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
    } else if (value.isInstanceOf[mutable.Map[String, _ <: Any]]) {
      NameListConverter.setAttributeValue(context, attributeBuilder, value)
    } else if (valueType <:< universe.typeOf[Array[_]] ||
      valueType.typeSymbol == universe.typeOf[Array[_ ]].typeSymbol) {
      ArrayConverter.setAttributeValue(context, attributeBuilder, value, valueType)
    } else if (valueType =:= universe.typeOf[DataFormat]) {
      DataFormatConverter.setAttributeValue(context, attributeBuilder, value)
    } else {
      CustomConverterDelegator.setAttributeValue(context, attributeBuilder, value, valueType)
    }
  }

/**
 * DataConverter for [[com.intel.analytics.bigdl.optim.Regularizer]]
 */
  object RegularizerConverter extends DataConverter {

    override def getAttributeValue[T : ClassTag](context: DeserializeContext,
                                                 attribute: AttrValue)
      (implicit ev: TensorNumeric[T]): AnyRef = {
      val regularizer = attribute.getRegularizerValue
      val regularizerType = regularizer.getRegularizerType
      if (regularizer.getRegularDataCount == 0) {
        return null
      }
      regularizerType match {
        case serialization.Bigdl.RegularizerType.L1Regularizer =>
          val l1 = regularizer.getRegularDataList.get(0)
          L1Regularizer[T](l1)
        case serialization.Bigdl.RegularizerType.L2Regularizer =>
          val l2 = regularizer.getRegularDataList.get(1)
          L2Regularizer[T](l2)
        case serialization.Bigdl.RegularizerType.L1L2Regularizer =>
          val l1 = regularizer.getRegularDataList.get(0)
          val l2 = regularizer.getRegularDataList.get(1)
          L1L2Regularizer[T](l1, l2)
      }
    }

    override def setAttributeValue[T : ClassTag]
    (context: SerializeContext[T], attributeBuilder: AttrValue.Builder, value: Any,
     valueType : universe.Type = null)
    (implicit ev: TensorNumeric[T]): Unit = {
      attributeBuilder.setDataType(DataType.REGULARIZER)
      if (value != null) {
        var regularizerBuilder = serialization.Bigdl.Regularizer.newBuilder
        val regularizer = value.asInstanceOf[L1L2Regularizer[T]]
        val l1 = regularizer.l1
        val l2 = regularizer.l2
        regularizerBuilder.addRegularData(l1)
        regularizerBuilder.addRegularData(l2)
        val regularizerType = regularizer match {
          case l1: L1Regularizer[_] => serialization.Bigdl.RegularizerType.L1Regularizer
          case l2: L2Regularizer[_] => serialization.Bigdl.RegularizerType.L2Regularizer
          case l1l2: L1L2Regularizer[_] => serialization.Bigdl.RegularizerType.L1L2Regularizer
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


    private def isEmptyTensor(tensor : Tensor[_]): Boolean = {
      val emptyTensor = tensor.getTensorType match {
        case DenseType =>
          tensor.storage == null
        case QuantizedType =>
          tensor.asInstanceOf[QuantizedTensor[_]].getStorage == null
      }
      emptyTensor
    }

    override def getAttributeValue[T: ClassTag](context: DeserializeContext,
                                                attribute: AttrValue)
      (implicit ev: TensorNumeric[T]): AnyRef = {
      val serializedTensor = attribute.getTensorValue
      if (!serializedTensor.hasStorage) {
        return null
      }
      val storages = context.storages
      val tensorId = serializedTensor.getId
      if (storages.contains(tensorId)) {
        return storages.get(tensorId).get.asInstanceOf[AnyRef]
      }
      val dataType = serializedTensor.getDatatype
      val tensorType = serializedTensor.getTensorType
      val sizes = serializedTensor.getSizeList.asScala.toArray.map(_.intValue())
      val strides = serializedTensor.getStrideList.asScala.toArray.map(_.intValue())
      val offSet = serializedTensor.getOffset
      val isScalr = serializedTensor.getIsScalar
      val serializedStorage = serializedTensor.getStorage
      val storageId = serializedStorage.getId
      val created = if (storages.contains(storageId)) {
        storages.get(storageId).get
      } else {
        null
      }

      def quant(): Tensor[T] = {
        var bytes: Array[Byte] = null
        if (context.storageType == ProtoStorageType) {
          bytes = serializedStorage.getBytesDataList.asScala.toArray.head.toByteArray
        } else {
          created
        }
        val serializedParams = serializedStorage.getIntDataList.asScala.toArray.map(_.intValue())
        val paramsNum = serializedParams.head
        val paramsArray = serializedParams.slice(1, paramsNum + 1)
        val descTypeEnum = serializedParams(1 + paramsNum)

        val start = paramsNum + 2 // params number indicator + params number + desc type

        val length = if (sizes.length == 1) {
          1 // if the size is 1, means it's a vector
        } else {
          sizes(0)
        }
        val max = new Array[T](length)
        val min = new Array[T](length)
        val sum = new Array[T](length)

        dataType match {
          case DataType.FLOAT =>
            val data = serializedStorage.getFloatDataList.asScala.toArray.map(_.floatValue())
            var i = 0
            while (i < length) {
              max(i) = ev.fromType[Float](data(i))
              min(i) = ev.fromType[Float](data(i + length))
              sum(i) = ev.fromType[Float](data(i + 2 * length))
              i += 1
            }
        }

        var params: DescParams = null

        descTypeEnum match {
          case 0 =>
            params = ConvDataParams(paramsArray)
          case 1 =>
            params = ConvWeightParams(paramsArray)
          case 2 =>
            params = LinearDataParams(paramsArray)
          case 3 =>
            params = LinearWeightParams(paramsArray)
        }

        QuantizedTensor[T](bytes, max, min, sum, sizes, params)
      }

      val tensor = dataType match {
        case DataType.FLOAT =>
          tensorType match {
            case TensorType.DENSE =>
              val storage : Storage[Float] = if (created == null ) {
                if (storageId == -1) {
                  null
                } else {
                  val data = serializedStorage.getFloatDataList.asScala.toArray.map(_.floatValue())
                  val newStorage = Storage[Float](data)
                  storages(storageId) = newStorage
                  newStorage
                }
              } else created.asInstanceOf[Storage[Float]]
              Tensor[Float](storage, offSet, sizes, strides)
            case TensorType.QUANT => quant()
          }
        case DataType.DOUBLE =>
          val storage : Storage[Double] = if (created == null ) {
            if (storageId == -1) {
              null
            } else {
              val data = serializedStorage.getDoubleDataList.asScala.toArray.map(_.doubleValue())
              val newStorage = Storage[Double](data)
              storages(storageId) = newStorage
              newStorage
            }
          } else created.asInstanceOf[Storage[Double]]
          Tensor[Double](storage, offSet, sizes, strides)
        case DataType.BOOL =>
          val storage : Storage[Boolean] = if (created == null ) {
            if (storageId == -1) {
              null
            } else {
              val data = serializedStorage.getBoolDataList.asScala.toArray.map(_.booleanValue())
              val newStorage = Storage[Boolean](data)
              storages(storageId) = newStorage
              newStorage
            }
          } else created.asInstanceOf[Storage[Boolean]]
          Tensor[Boolean](storage, offSet, sizes, strides)
        case DataType.CHAR =>
          val storage: Storage[Char] = if (created == null ) {
            if (storageId == -1) {
              null
            } else {
              val data = serializedStorage.getIntDataList.asScala.toArray.map(_.toChar.charValue())
              val newStorage = Storage[Char](data)
              storages(storageId) = newStorage
              newStorage
            }
          } else created.asInstanceOf[Storage[Char]]
          Tensor[Char](storage, offSet, sizes, strides)
        case DataType.STRING =>
          val storage: Storage[String] = if (created == null ) {
            if (storageId == -1) {
              null
            } else {
              val data = serializedStorage.getStringDataList.asScala.toArray
              val newStorage = Storage[String](data)
              storages(storageId) = newStorage
              newStorage
            }
          } else created.asInstanceOf[Storage[String]]
          Tensor[String](storage, offSet, sizes, strides)
        case DataType.INT32 =>
          val storage: Storage[Int] = if (created == null ) {
            if (storageId == -1) {
              null
            } else {
              val data = serializedStorage.getIntDataList.asScala.toArray.map(_.intValue())
              val newStorage = Storage[Int](data)
              storages(storageId) = newStorage
              newStorage
            }
          } else created.asInstanceOf[Storage[Int]]
          Tensor[Int](storage, offSet, sizes, strides)
        case DataType.SHORT =>
          val storage: Storage[Short] = if (created == null ) {
            if (storageId == -1) {
              null
            } else {
              val data = serializedStorage.getIntDataList.asScala.toArray.map(_.shortValue())
              val newStorage = Storage[Short](data)
              storages(storageId) = newStorage
              newStorage
            }
          } else created.asInstanceOf[Storage[Short]]
          Tensor[Short](storage, offSet, sizes, strides)
        case DataType.INT64 =>
          val storage: Storage[Long] = if (created == null ) {
            if (storageId == -1) {
              null
            } else {
              val data = serializedStorage.getLongDataList.asScala.toArray.map(_.longValue())
              val newStorage = Storage[Long](data)
              storages(storageId) = newStorage
              newStorage
            }
          } else created.asInstanceOf[Storage[Long]]
          Tensor[Long](storage, offSet, sizes, strides)
        case DataType.BYTES =>
          val storage: Storage[ByteString] = if (created == null ) {
            if (storageId == -1) {
              null
            } else {
              val data = serializedStorage.getBytesDataList.asScala.toArray
              val newStorage = Storage[ByteString](data)
              storages(storageId) = newStorage
              newStorage
            }
          } else created.asInstanceOf[Storage[ByteString]]
          Tensor[ByteString](storage, offSet, sizes, strides)
        case _ => throw new IllegalArgumentException(s"$dataType not supported in tensor now !")
      }
      storages(tensorId) = tensor
      tensor
    }

    private def setStorage[T: ClassTag](context: SerializeContext[T],
      tensorBuilder: BigDLTensor.Builder, tensor: Tensor[_]): Unit = {
      val storageType = context.storageType
      if (storageType == ProtoStorageType) {
        ProtoTensorStorageManager.setStorage(context, tensorBuilder, tensor)
      } else if (storageType == BigDLStorage) {
        BigDLTensorStorageManager.setStorage(context, tensorBuilder, tensor)
      } else {
        throw new IllegalArgumentException(s"$storageType not supported")
      }
    }

    override def setAttributeValue[T: ClassTag]
      (context: SerializeContext[T], attributeBuilder: AttrValue.Builder, value: Any,
       valueType : universe.Type = null)
      (implicit ev: TensorNumeric[T]): Unit = {
      attributeBuilder.setDataType(DataType.TENSOR)
      if (value != null) {
        val tensor = value.asInstanceOf[Tensor[_]]
        val tensorId = System.identityHashCode(tensor)
        val storages = context.storages
        // Check if tensor has been shared
        if (storages.contains(tensorId)) {
          attributeBuilder.setTensorValue(resetTensor(storages.get(tensorId).get
            .asInstanceOf[BigDLTensor]))
        } else {
          val totalElement = tensor.nElement()
          val dimension = tensor.dim()
          val tensorBuilder = BigDLTensor.newBuilder
          tensorBuilder.setId(tensorId)
          tensorBuilder.setDimension(dimension)
          tensorBuilder.setNElements(totalElement)
          tensor.getTensorType match {
            case DenseType =>
              tensorBuilder.setOffset(tensor.storageOffset())
              tensorBuilder.setIsScalar(tensor.isScalar)
              tensorBuilder.setTensorType(TensorType.DENSE)
            case QuantizedType =>
              tensorBuilder.setTensorType(TensorType.QUANT)
          }

          val tensorEmpty = isEmptyTensor(tensor)

          if (!tensorEmpty) {
            tensor.size().foreach(size => tensorBuilder.addSize(size))
            tensor.stride().foreach(stride => tensorBuilder.addStride(stride))
          }
          setStorage(context, tensorBuilder, tensor)
          val tensorBuild = tensorBuilder.build
          attributeBuilder.setTensorValue(resetTensor(tensorBuild))
          storages(tensorId) = tensorBuild
        }
      }
    }


    private def resetStorage(originStorage : TensorStorage) : TensorStorage = {
      val storageBuilder = TensorStorage.newBuilder
      storageBuilder.setDatatype(originStorage.getDatatype)
      storageBuilder.setId(originStorage.getId)
      storageBuilder.build
    }

    private def resetTensor(originTensor: BigDLTensor) : BigDLTensor = {
      val tensorBuilder = BigDLTensor.newBuilder(originTensor)
      tensorBuilder.clearStorage
      tensorBuilder.setDatatype(originTensor.getDatatype)
      tensorBuilder.setId(originTensor.getId)
      if (originTensor.hasStorage) {
        tensorBuilder.setStorage(resetStorage(originTensor.getStorage))
      }
      tensorBuilder.build
    }
  }

/**
 * DataConverter for [[com.intel.analytics.bigdl.nn.VariableFormat]]
 */
  object VariableFormatConverter extends DataConverter {

    override def getAttributeValue[T: ClassTag](context: DeserializeContext, attribute: AttrValue)
      (implicit ev: TensorNumeric[T]): AnyRef = {
      val format = attribute.getVariableFormatValue
      format match {
        case VarFormat.DEFAULT => VariableFormat.Default
        case VarFormat.ONE_D => VariableFormat.ONE_D
        case VarFormat.IN_OUT => VariableFormat.IN_OUT
        case VarFormat.OUT_IN => VariableFormat.OUT_IN
        case VarFormat.IN_OUT_KW_KH => VariableFormat.IN_OUT_KW_KH
        case VarFormat.OUT_IN_KW_KH => VariableFormat.OUT_IN_KW_KH
        case VarFormat.GP_OUT_IN_KW_KH => VariableFormat.GP_OUT_IN_KW_KH
        case VarFormat.GP_IN_OUT_KW_KH => VariableFormat.GP_IN_OUT_KW_KH
        case VarFormat.OUT_IN_KT_KH_KW => VariableFormat.OUT_IN_KT_KH_KW
        case VarFormat.EMPTY_FORMAT => null
      }
    }

    override def setAttributeValue[T: ClassTag](
    context: SerializeContext[T], attributeBuilder: AttrValue.Builder,
      value: Any, valueType: universe.Type = null)(implicit ev: TensorNumeric[T]): Unit = {
      attributeBuilder.setDataType(DataType.VARIABLE_FORMAT)
      if (value != null) {
        val format = value.asInstanceOf[VariableFormat]
        val formatValue = format match {
          case VariableFormat.Default => VarFormat.DEFAULT
          case VariableFormat.ONE_D => VarFormat.ONE_D
          case VariableFormat.IN_OUT => VarFormat.IN_OUT
          case VariableFormat.OUT_IN => VarFormat.OUT_IN
          case VariableFormat.IN_OUT_KW_KH => VarFormat.IN_OUT_KW_KH
          case VariableFormat.OUT_IN_KW_KH => VarFormat.OUT_IN_KW_KH
          case VariableFormat.GP_OUT_IN_KW_KH => VarFormat.GP_OUT_IN_KW_KH
          case VariableFormat.GP_IN_OUT_KW_KH => VarFormat.GP_IN_OUT_KW_KH
          case VariableFormat.OUT_IN_KT_KH_KW => VarFormat.OUT_IN_KT_KH_KW
        }
        attributeBuilder.setVariableFormatValue(formatValue)
      } else {
        attributeBuilder.setVariableFormatValue(VarFormat.EMPTY_FORMAT)
      }
   }
}
/**
 * DataConverter for [[com.intel.analytics.bigdl.nn.InitializationMethod]]
 */
  object InitMethodConverter extends DataConverter {

    override def getAttributeValue[T: ClassTag](context: DeserializeContext, attribute: AttrValue)
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

    override def setAttributeValue[T: ClassTag](
    context: SerializeContext[T], attributeBuilder: AttrValue.Builder,
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
   * DataConverter for [[com.intel.analytics.bigdl.nn.abstractnn.DataFormat]]
   */
  object DataFormatConverter extends DataConverter {
    override def getAttributeValue[T: ClassTag](context: DeserializeContext, attribute: AttrValue)
      (implicit ev: TensorNumeric[T]): AnyRef = {
      val dataFormat = attribute.getDataFormatValue
      dataFormat match {
        case InputDataFormat.NCHW => NCHW
        case InputDataFormat.NHWC => NHWC
      }

    }

    override def setAttributeValue[T: ClassTag]
    (context: SerializeContext[T],
     attributeBuilder: AttrValue.Builder, value: Any, valueType: universe.Type)
    (implicit ev: TensorNumeric[T]): Unit = {
      attributeBuilder.setDataType(DataType.DATA_FORMAT)
      if (value != null) {
        val dataFormat = value.asInstanceOf[DataFormat]
        val inputFormat = dataFormat match {
          case NCHW => InputDataFormat.NCHW
          case NHWC => InputDataFormat.NHWC
        }
        attributeBuilder.setDataFormatValue(inputFormat)
      }
    }
  }

/**
 * DataConverter for [[com.intel.analytics.bigdl.nn.abstractnn.AbstractModule]]
 */
  object ModuleConverter extends DataConverter {

    override def getAttributeValue[T: ClassTag](context: DeserializeContext, attribute: AttrValue)
     (implicit ev: TensorNumeric[T]): AnyRef = {
     val serializedModule = attribute.getBigDLModuleValue
      if (serializedModule.getModuleType != null && serializedModule.getModuleType != "") {
        ModuleSerializer.load(DeserializeContext(serializedModule,
            context.storages, context.storageType)).module
      } else {
        null
      }
   }

    override def setAttributeValue[T: ClassTag](context: SerializeContext[T],
      attributeBuilder: AttrValue.Builder,
      value: Any, valueType: universe.Type = null)(implicit ev: TensorNumeric[T]): Unit = {
      attributeBuilder.setDataType(DataType.MODULE)
      if (value != null) {
        val module = value.asInstanceOf[AbstractModule[Activity, Activity, T]]
        val serializableModule = ModuleSerializer.
            serialize(SerializeContext(ModuleData(module, Seq[String](), Seq[String]()),
              context.storages, context.storageType)).bigDLModule
          attributeBuilder.setBigDLModuleValue(serializableModule)
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
      }
      arr
    }

    override def setAttributeValue[T: ClassTag](context: SerializeContext[T],
                                                attributeBuilder: AttrValue.Builder,
      value: Any, valueType: universe.Type = null)(implicit ev: TensorNumeric[T]): Unit = {
      attributeBuilder.setDataType(DataType.ARRAY_VALUE)
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
        typeOf[Array[_ <: AbstractModule[_ <: Activity, _ <:  Activity, _ <: Any]]]) {
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
      } else if (value.isInstanceOf[Array[Map[String, Any]]]) {
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
